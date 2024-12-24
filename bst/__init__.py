from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F

class TextHead(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=2):
        """
        Args:
            input_dim (int): Input dimension size.
            hidden_size (int): Size of the hidden layers.
            vocab_size (int): Vocabulary size.
            num_layers (int): Number of layers in the shared MLP.
        """
        super().__init__()

        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        self.shared_mlp = nn.Sequential(*layers)

        # output layer for next and previous token predictions
        self.output_layer = nn.Linear(hidden_size, vocab_size * 2, bias=False)

    def forward(self, f, b):
        combined = torch.cat([f, b], dim=-1)
        shared_output = self.shared_mlp(combined)
        logits = self.output_layer(shared_output)
        return logits


class BeliefStateTransformer(nn.Module, ABC):
    """
    A base class for neural network models that enforces the implementation of specific methods.
    """

    def __init__(self, args):
        super().__init__()

         # for B(∅) in the backward encoder
        assert args.empty_suffix_id is not None, "Please provide backward encoder empty suffix."
        self.empty_suffix_id = args.empty_suffix_id

        # block size for cropping sequences
        self.block_size = args.block_size

        # output network for next and previous token predictions
        self.vocab_size = args.vocab_size
        
        # base BST or pre-trained model
        # self.model = model
        # self.model.gradient_checkpointing_enable() # enable gradient checkpointing to save memory during training

        # backward suffix cache attributes
        self._backward_suffix_cache = None
        self.cache_valid = False

        # KV cache
        self.use_cache = args.use_cache

    def create_text_head(self, args):
        """
        Create the text head for the model. Call after finalizing n_embd.
        """
        self.text_head = TextHead(
                            input_dim=args.n_embd * 2,
                            hidden_size=args.text_head_hidden, 
                            vocab_size=args.vocab_size,
                        )

    @abstractmethod
    def get_latent_states(self, x, direction):
        """
        Get the forward or backward latent states of the model.
        """
        pass
    
    @abstractmethod
    def set_encoder(self, encoder):
        """
        Set the encoder to use during the forward pass.
        """
        pass

    def get_forward_and_backward_latent(self, x, detach_grad=False):
        """
        Get the forward and backward latent states of the model.
        """
        forward_states = self.get_latent_states(x, direction="forward")
        if detach_grad:
            _f = forward_states.detach()
            _f.requires_grad = True
        backward_states = self.get_latent_states(torch.flip(x, dims=[1]), direction="backward")
        if detach_grad:
            _b = backward_states.detach()
            _b.requires_grad = True
            return _f, _b, forward_states, backward_states
        
        return forward_states, backward_states, None, None
    
    # caching the backward latent states
    def invalidate_cache(self):
        """Mark the backward suffix latent cache as invalid."""
        self.cache_valid = False

    def get_backward_suffix_latent(self):
        """
        Property to access the backward suffix latent state cache.
        If the cache is invalid, it will be recomputed.
        """
        if not self.cache_valid:
            empty_suffix = torch.tensor([self.empty_suffix_id]).unsqueeze(0)
            backward_state = self.get_latent_states(empty_suffix, direction="backward")  # fixed backward state
            self._backward_suffix_cache = backward_state
            self.cache_valid = True
        
        return self._backward_suffix_cache

    def train(self, mode: bool = True):
        """
        Overrides the default train() method to invalidate the cache when switching to training mode.
        """
        super().train(mode)
        if mode:
            self.invalidate_cache()


    def forward(self, x, targets=None, detach_grad=False):
        """
        Belief state objective of the model. Must be implemented by subclasses.
        """
        if targets is not None:
            _f, _b, orig_forward_states, orig_backward_states = self.get_forward_and_backward_latent(x, detach_grad=detach_grad)

            bs, T = x.shape
            forward_states = _f
            backward_states = _b.flip(1)
            # generate all valid combinations of forward and backward indices
            ft = torch.arange(T, dtype=torch.int32)  # forward indices
            bt = torch.arange(T, dtype=torch.int32)  # backward indices
            combinations = torch.cartesian_prod(ft, bt) 
            combinations = combinations[(combinations[:, 1] - combinations[:, 0] >= 2)]  # filter valid pairs
            fb_pairs = combinations[combinations[:, 1] < T]  # ensure backward indices are within range

            # extract valid indices
            f_idxs, b_idxs = fb_pairs[:, 0], fb_pairs[:, 1]
            # TODO: make this more general, currently y corresponds to next token labels of x, which means it is
            # shifted by 1
            nt_idxs = f_idxs  # indices for next token labels
            pt_idxs = (b_idxs - 2)  # indices for prev token labels
            next_labels = targets[:, nt_idxs].reshape(-1)
            prev_labels = targets[:, pt_idxs].reshape(-1)

            # gather forward and backward features
            f = forward_states[:, f_idxs]
            b = backward_states[:, b_idxs]

            # compute logits from the text head
            logits = self.text_head(f, b)  # combine forward and backward states
            fb_numpairs = fb_pairs.shape[0]  # no of valid forward-backward pairs

            # reshape logits and labels for loss computation
            logits = logits.view(bs, fb_numpairs, 2, -1)  # split into next and previous logits

            # flatten logits and labels
            next_logits = logits[:, :, 0, :].reshape(-1, logits.size(-1)) 
            prev_logits = logits[:, :, 1, :].reshape(-1, logits.size(-1)) 

            # compute the loss independently for next and previous token predictions
            # this also sums the negative log likelihood for 
            # all next and previous token predictions together, aligning with the paper
            # ignore_index=-1 is used to skip the gradient contributions from unnecessary tokens in target
            next_loss = F.cross_entropy(next_logits, next_labels, ignore_index=-1)
            prev_loss = F.cross_entropy(prev_logits, prev_labels, ignore_index=-1)

            losses = {
                "next_loss": next_loss,
                "prev_loss": prev_loss, 
                "orig_loss": next_loss + prev_loss
            }

            # calculate accuracy
            next_mask = next_labels != -1
            prev_mask = prev_labels != -1

            next_pred = torch.argmax(next_logits[next_mask], dim=-1)
            prev_pred = torch.argmax(prev_logits[prev_mask], dim=-1)

            next_correct = next_pred.eq(next_labels[next_mask]).to(torch.float)
            prev_correct = prev_pred.eq(prev_labels[prev_mask]).to(torch.float)

            next_acc = next_correct.mean().item()
            prev_acc = prev_correct.mean().item()

            total_valid_tokens = next_mask.sum() + prev_mask.sum()
            overall_acc = (
                next_correct.sum() + prev_correct.sum()
            ).item() / total_valid_tokens.item()

            accs = {
                "acc": overall_acc, 
                "forward_acc": next_acc, 
                "backward_acc": prev_acc
            }

            if detach_grad:
                # return latents for separate backprop
                return logits, losses, accs, (_f, _b, orig_forward_states, orig_backward_states)
        else:
            # Step 1: Precompute backward latent state for empty suffix B(∅)
            bsz, _ = x.shape
            backward_state = self.get_backward_suffix_latent().repeat(bsz, 1, 1).to(x.device)  # generate/get cached backward state
            forward_state = self.get_latent_states(x, direction="forward")[:, -1:, :]

            # Step 4: Compute logits for the next token prediction
            logits = self.text_head(forward_state, backward_state)
            losses, accs = None, None
        
        return logits, losses, accs, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=1):
        """
        Generate text autoregressively using the forward model.

        Args:
            idx (torch.Tensor): Input tensor containing the tokenized prefix (batch_size, sequence_length).
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling.

        Returns:
            torch.Tensor: Tokenized output sequence.
        """
        out = idx.clone()

        for i in range(max_new_tokens):
            if self.use_cache and self.forward_encoder.cache.use_caching:
                # If we're caching, only propagate the last token
                idx = idx_next
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # compute logits for the next token prediction
            logits, _, _, _ = self(idx_cond)
            next_logits = logits[:, 0, :self.text_head.output_layer.out_features // 2] # extract next token logits

            # apply temperature scaling
            next_logits = next_logits / temperature

            # apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')

            # convert logits to probabilities and sample
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((out, idx_next), dim=1)
            out = idx.clone()

        return out
