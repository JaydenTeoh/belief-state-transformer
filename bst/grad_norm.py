from functools import partial, cache

import torch
import torch.distributed as dist
from torch.autograd import grad
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Module, ModuleList, Parameter
from typing import Optional, Union, Dict, List, Tuple


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers
def l1norm(t, dim = -1):
    return F.normalize(t, p = 1, dim = dim)

# distributed helpers
@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t


class GradNormLossWeighter(nn.Module):
    def __init__(
        self,
        num_losses,
        learning_rate=1e-4,
        restoring_force_alpha=0.0,
        grad_norm_parameters: Optional[Parameter] = None,
        initial_losses_decay = 1.,
    ):
        super().__init__()
        assert exists(num_losses)
        self.num_losses = num_losses
        self.learning_rate = learning_rate

        # restoring force
        assert restoring_force_alpha >= 0.
        self.restoring_force_alpha = restoring_force_alpha
        self.has_restoring_force = self.restoring_force_alpha > 0

        self._grad_norm_parameters = [grad_norm_parameters] # hack

        # if initial loss decay set to less than 1, will EMA smooth the initial loss
        assert 0 <= initial_losses_decay <= 1.
        self.initial_losses_decay = initial_losses_decay
        self.register_buffer('initial_losses', torch.zeros(num_losses))

        loss_weights = torch.ones((num_losses,))
        self.register_buffer('loss_weights', loss_weights)
        self.register_buffer('loss_weights_sum', self.loss_weights.sum()) # for renormalizing loss weights at end
        self.register_buffer('loss_weights_grad', torch.zeros_like(loss_weights), persistent = False) # for gradient accumulation
        self.register_buffer('initted', torch.tensor(False))

    @property
    def grad_norm_parameters(self):
        return self._grad_norm_parameters[0]

    def backward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, 
                losses: Tensor,
                activations: Optional[Tensor] = None,
                scale = 1.0,
                **backward_kwargs
        ):
        assert losses.ndim == 1, 'losses must be 1 dimensional'
        assert len(losses) == self.num_losses, "Number of losses must match num_losses."
        assert all([loss.numel() == 1 for loss in losses]), "All losses must be scalar."

        # auto move gradnorm module to the device of the losses
        if self.initted.device != losses.device:
            self.to(losses.device)

        total_weighted_loss = (losses * self.loss_weights.detach()).sum() * scale
        total_weighted_loss.backward(retain_graph=True, **backward_kwargs)

        if self.has_restoring_force:
            if not self.initted.item():
                initial_losses = maybe_distributed_mean(losses)
                self.initial_losses.copy_(initial_losses)
                self.initted.copy_(True)

            elif self.initial_losses_decay < 1.:
                meaned_losses = maybe_distributed_mean(losses)
                self.initial_losses.lerp_(meaned_losses, 1. - self.initial_losses_decay)

        # determine which tensor to get grad norm from
        grad_norm_tensor = default(activations, self.grad_norm_parameters)
        assert exists(grad_norm_tensor), 'you need to either set `grad_norm_parameters` on init or `activations` on backwards'
        grad_norm_tensor.requires_grad_()

        # get grad norm with respect to each loss
        grad_norms = []
        loss_weights = self.loss_weights.clone()
        loss_weights = Parameter(loss_weights)
        all_losses = (loss_weights * losses).sum()
        gradients = grad(all_losses, grad_norm_tensor, create_graph=True, retain_graph=True)[0]
        grad_norms = gradients.view(self.num_losses, -1).norm(p=2, dim=1)

        grad_norms = torch.stack(grad_norms)

        # main algorithm for loss balancing
        grad_norm_average = maybe_distributed_mean(grad_norms.mean())

        if self.has_restoring_force:
            loss_ratio = losses.detach() / self.initial_losses

            relative_training_rate = l1norm(loss_ratio) * self.num_losses

            gradient_target = (grad_norm_average * (relative_training_rate ** self.restoring_force_alpha)).detach()
        else:
            gradient_target = grad_norm_average.repeat(self.num_losses).detach()

        grad_norm_loss = F.l1_loss(grad_norms, gradient_target) * scale
        grad_norm_loss.backward(**backward_kwargs)

        # accumulate gradients
        self.loss_weights_grad.add_(loss_weights.grad)

        # manually take a single gradient step
        updated_loss_weights = loss_weights - self.loss_weights_grad * self.learning_rate
        renormalized_loss_weights = l1norm(updated_loss_weights) * self.loss_weights_sum
        self.loss_weights.copy_(renormalized_loss_weights)
        self.loss_weights_grad.zero_()

        return total_weighted_loss