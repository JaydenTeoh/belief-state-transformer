import torch
from torch.nn import functional as F
import wandb
from contextlib import nullcontext

from bst.helpers import get_bst_model
from bst.grad_norm import GradNormLossWeighter
from utils.training_utils import get_lr

class BSTTrainer:
    def __init__(
            self, 
            args,
            device,
            detach_grad=True, # more efficient training
        ):
        self.args = args
        self.model = get_bst_model(args)
        self.detach_grad = detach_grad
        self.clip_gradients = args.clip_gradients
        self.clip_grad_norm = args.clip_grad_norm
        self.wandb_logging = args.use_wandb

        if args.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)

        self.model.to(device)
        self.model.train()

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.lr = args.lr
        self.decay_lr = args.decay_lr
        self.warmup_iters = args.warmup_iters
        self.lr_decay_iters = args.lr_decay_iters
        self.min_lr = args.min_lr
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, \
                                           weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
        if args.use_grad_norm:
            self.weighter = GradNormLossWeighter(
                num_losses = 2,
                learning_rate = args.gradnorm_lr,
                restoring_force_alpha = args.gradnorm_alpha, 
                grad_norm_parameters = self.model.text_head.shared_mlp[-2].weight, # last layer of the shared mlp
                initial_losses_decay = args.gradnorm_init_loss_decay,
                update_every=args.gradnorm_update_every
            )
        else:
            self.weighter = None

        self.num_iters = 0

    def step(self, x, y):
        """
        Efficient training for the BST model.
        """
        self.num_iters += 1
        self.model.train() # IMPORTANT to invalidate empty backward latent cache
        lr = get_lr(self.num_iters, self.lr, self.warmup_iters, self.lr_decay_iters, self.min_lr) if self.decay_lr else self.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Compute the combined loss
        logits, losses, accs, states = self.model(x, y, self.detach_grad)
        if self.detach_grad:
            _f, _b, forward_states, backward_states = states

        # set encoders here in case using detach_grad=False
        if not self.detach_grad:
            self.model.set_encoder(["forward_encoder", "backward_encoder"])

        if self.weighter is not None:
            # have to scale losses independently
            loss = torch.stack((self.scaler.scale(losses["next_loss"]), self.scaler.scale(losses["prev_loss"])), dim=0)
            loss = self.weighter.backward(loss)
        else:
            loss = self.scaler.scale(losses["orig_loss"])
            loss.backward()

        if self.detach_grad:
            self.model.set_encoder("forward_encoder")
            forward_states.backward(_f.grad)

            self.model.set_encoder("backward_encoder")
            backward_states.backward(_b.grad)

        # need to set encoders here for peft encoder gradients to show
        self.model.set_encoder(["forward_encoder", "backward_encoder"]) 
        
        self.scaler.unscale_(self.optimizer)
        # gradient clipping
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        if self.wandb_logging:
            forward_grad_norm = torch.norm(torch.cat([p.grad.flatten() for n, p in self.model.named_parameters() if "forward_encoder" in n and p.grad is not None]))
            backward_grad_norm = torch.norm(torch.cat([p.grad.flatten() for n, p in self.model.named_parameters() if "backward_encoder" in n and p.grad is not None]))
            wandb.log({"train/forward_grad_norm": forward_grad_norm.item(), "train/backward_grad_norm": backward_grad_norm.item()})
            wandb.log({
                "train/forward_loss": losses["next_loss"].item(), 
                "train/backward_loss": losses["prev_loss"].item(), 
                "train/orig_loss": losses["orig_loss"].item(),
            })

            wandb.log({"train/loss": loss.item(), "train/acc": accs["acc"],
                        "train/forward_acc": accs["forward_acc"], 
                        "train/backward_acc": accs["backward_acc"],
                        "learning_rate": self.lr, "step": self.num_iters})
            if self.weighter is not None:
                wandb.log({
                    "train/loss_weights_forward": self.weighter.loss_weights[0].item(),
                    "train/loss_weights_backward": self.weighter.loss_weights[1].item()
                })

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        return logits, loss, accs

    def save(self, path, ep):
        torch.save(self.model.state_dict(), path + "_epoch_" + str(ep))

