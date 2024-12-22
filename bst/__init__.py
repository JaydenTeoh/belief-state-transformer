from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from utils.training_utils import accuracy
from bst.grad_norm import GradNormLossWeighter
from torch.nn import functional as F
import torch.nn as nn
import torch
import wandb

class TiedTextHead(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, tied_weights=None):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # output is twice vocab size
        # first half is for next token prediction: x_{t+1}
        # second half is for previous token prediction: x_{t+k-1}
        self.output_layer = nn.Linear(hidden_size, vocab_size * 2, bias=False)

    def forward(self, f, b):
        combined = torch.cat([f, b], dim=-1)
        shared_output = self.shared_mlp(combined)
        logits = self.output_layer(shared_output)

        return logits


class BeliefStateTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        assert args.eos_token_id is not None, "Please provide the EOS token ID for backward encoder null suffix."
        self.eos_token_id = args.eos_token_id

        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype=args.ptdtype,
                                    bnb_4bit_use_double_quant=True
                                )
        else:
            # Assume default or no quantization
            quantization_config = None

        if args.use_flash:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None
        
        # TODO: support more pretrained models
        # import gpt2, has to be a causal LM else LoraConfig will throw an error
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            attn_implementation=attn_implementation,
            torch_dtype=args.ptdtype,
            quantization_config=quantization_config,
            device_map="auto"
        )
        del self.model.lm_head

        # lora adapter config
        lora_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            target_modules=["c_attn"],  # apply lora to attention layers
            layers_to_transform=[8,9,10,11],
            layers_pattern="h",
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=True
        )

        # create separate forward and backward lora adapters
        self.model.add_adapter(lora_config, adapter_name="forward_encoder")
        self.model.add_adapter(lora_config, adapter_name="backward_encoder")

        # add tied text head for next and previous token predictions
        self.vocab_size = args.vocab_size
        self.text_head = TiedTextHead(
                            input_dim=self.model.config.hidden_size * 2,
                            hidden_size=512, # TODO; allow this to be configurable
                            vocab_size=args.vocab_size,
                            # tied_weights=self.model.transformer.wte.weight  # use input embeddings' weights
                        )

        # GradNorm
        self.use_grad_norm = args.use_grad_norm
        if self.use_grad_norm:
            backbone_parameter = self.text_head.shared_mlp[-1].weight
            self.gradnorm_weighter = GradNormLossWeighter(
                num_losses=2,  # next and prev token prediction tasks
                learning_rate=args.gradnorm_lr,  # a small learning rate for GradNorm updates
                restoring_force_alpha=args.gradnorm_alpha,  # hyperparameter to control task balancing
                grad_norm_parameters=backbone_parameter, # update the loss weights wrt activations of last shared layer
                initial_losses_decay=args.gradnorm_init_loss_decay  # decay factor for determining initial losses
            )

        # gradient clipping
        self.clip_gradients = args.clip_gradients
        self.clip_grad_norm = args.clip_grad_norm

        # enable gradient checkpointing to save memory during training
        self.model.gradient_checkpointing_enable()
        
        # miscellanous
        self.wandb_logging = args.use_wandb
        
    def get_num_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params

    def forward(self, f, b):
        # forward encoding
        self.model.set_adapter("forward_encoder")
        forward_states = self.model(f).last_hidden_state  # get forward states

        # backward encoding
        self.model.set_adapter("backward_encoder")
        backward_input = torch.flip(b, dims=[1])  # reverse the input sequence
        backward_states = self.model(backward_input).last_hidden_state
        backward_states = torch.flip(backward_states, dims=[1])  # flip the backward states back

        # Text head for next and previous token predictions
        next_logits, prev_logits = self.text_head(forward_states, backward_states)
        # acc, token_acc = accuracy(logits, targets)
        # accs = {"acc": acc, "token_acc": token_acc}
        return next_logits, prev_logits
    
    def belief_state_objective(self, all_f, all_b, x, targets):
        """
        Compute the belief state objective as described in the BST paper.
        """
        bs, T = x.shape
        forward_states = all_f
        backward_states = all_b.flip(1)
        # generate all valid combinations of forward and backward indices
        ft = torch.arange(T, dtype=torch.int32)  # forward indices
        bt = torch.arange(T, dtype=torch.int32)  # backward indices
        combinations = torch.cartesian_prod(ft, bt) 
        combinations = combinations[(combinations[:, 1] - combinations[:, 0] >= 2)]  # filter valid pairs
        fb_pairs = combinations[combinations[:, 1] < T]  # ensure backward indices are within range

        # extract valid indices
        f_idxs, b_idxs = fb_pairs[:, 0], fb_pairs[:, 1]
        # nt_idxs = (combinations[:, 0] + 1)  # indices for next token labels
        # TODO: make this more general, currently y corresponds to next token labels of x, which means it is
        # shifted by 1
        nt_idxs = f_idxs  # indices for next token labels
        pt_idxs = (b_idxs - 2)  # indices for prev token labels

        # gather forward and backward features
        f = forward_states[:, f_idxs]
        b = backward_states[:, b_idxs]

        # prepare labels
        single_labels_f = targets[:, nt_idxs].unsqueeze(2)  # labels for next-token prediction
        single_labels_b = targets[:, pt_idxs].unsqueeze(2)   # labels for prev-token prediction
        single_labels = torch.cat((single_labels_f, single_labels_b), dim=2)

        # compute logits from the text head
        logits = self.text_head(f, b)  # combine forward and backward states
        fb_numpairs = fb_pairs.shape[0]  # no of valid forward-backward pairs

        # reshape logits and labels for loss computation
        logits = logits.view(bs, fb_numpairs, 2, -1)  # split into next and previous logits

        # flatten logits and labels
        next_logits = logits[:, :, 0, :].reshape(-1, logits.size(-1)) 
        prev_logits = logits[:, :, 1, :].reshape(-1, logits.size(-1)) 

        single_labels = single_labels.view(bs, fb_numpairs, 2)
        next_labels = single_labels[:, :, 0].reshape(-1) 
        prev_labels = single_labels[:, :, 1].reshape(-1)

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
    
        return logits, losses, accs
    
    def update(self, x, y, optimizer, scaler):
        """
        Efficient training for the BST model.
        """
        # Compute forward states
        self.model.set_adapter("forward_encoder")
        forward_states = self.model(x).last_hidden_state
        _f = forward_states.detach()
        _f.requires_grad = True

        # Compute backward states
        self.model.set_adapter("backward_encoder")
        backward_input = torch.flip(x, dims=[1])
        backward_states = self.model(backward_input).last_hidden_state
        _b = backward_states.detach()
        _b.requires_grad = True

        # Compute the combined loss
        logits, losses, accs = self.belief_state_objective(_f, _b, x, y)
        if self.use_grad_norm:
            # have to scale losses independently
            loss = torch.stack((scaler.scale(losses["next_loss"]), scaler.scale(losses["prev_loss"])), dim=0)
            loss = self.gradnorm_weighter.backward(loss)
        else:
            loss = scaler.scale(losses["orig_loss"])
            loss.backward()

        self.model.set_adapter("forward_encoder")
        forward_states.backward(_f.grad)

        self.model.set_adapter("backward_encoder")
        backward_states.backward(_b.grad)

        self.model.set_adapter(["forward_encoder", "backward_encoder"])
        
        # gradient clipping
        if self.clip_gradients:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

        if self.wandb_logging:
            forward_grad_norm = torch.norm(torch.cat([p.grad.flatten() for n, p in self.named_parameters() if "forward_encoder" in n and p.grad is not None]))
            backward_grad_norm = torch.norm(torch.cat([p.grad.flatten() for n, p in self.named_parameters() if "backward_encoder" in n and p.grad is not None]))
            wandb.log({"train/forward_grad_norm": forward_grad_norm.item(), "train/backward_grad_norm": backward_grad_norm.item()})
            wandb.log({
                "train/forward_loss": losses["next_loss"].item(), 
                "train/backward_loss": losses["prev_loss"].item(), 
                "train/orig_loss": losses["orig_loss"].item(),
                "train/loss_weights_forward": self.gradnorm_weighter.loss_weights[0].item(),
                "train/loss_weights_backward": self.gradnorm_weighter.loss_weights[1].item()
            })

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return logits, loss, accs
    
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
        bsz, prefix_len = idx.shape
        device = idx.device

        # Step 1: Precompute backward latent state for empty suffix B(∅)
        empty_suffix = torch.full((bsz, 1), self.eos_token_id, dtype=idx.dtype, device=device)
        self.model.set_adapter("backward_encoder")
        backward_state = self.model(empty_suffix).last_hidden_state[:, 0:1, :]  # Fixed backward state

        # Step 2: Initialize the generated sequence with the prefix
        out = idx.clone()

        for i in range(max_new_tokens):
            # Step 3: Compute forward latent state for current prefix F(x1:t)
            self.model.set_adapter("forward_encoder")
            forward_states = self.model(out).last_hidden_state

            # Step 4: Compute logits for the next token prediction
            logits = self.text_head(forward_states[:, -1:, :], backward_state)

            next_logits = logits[:, 0, :self.text_head.output_layer.out_features // 2]

            # Step 5: Apply temperature scaling
            next_logits = next_logits / temperature

            # Step 6: Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')

            # Step 7: Convert logits to probabilities and sample
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Step 8: Append sampled index to the running sequence
            out = torch.cat((out, idx_next), dim=1)

        return out

    
    # def _create_optimizer(self, lr, weight_decay):
    #     return torch.optim.AdamW(
    #         [
    #             {"params": self.model.get_adapter("forward_encoder").parameters(), "lr": lr},
    #             {"params": self.model.get_adapter("backward_encoder").parameters(), "lr": lr},
    #             {"params": self.text_head.parameters(), "lr": lr}
    #         ],
    #         lr=lr,
    #         weight_decay=weight_decay
    #     )

