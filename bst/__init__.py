from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from utils.training_utils import accuracy
from torch.nn import functional as F
import torch.nn as nn
import torch

class TiedTextHead(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, tied_weights=None):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        # output is twice vocab size
        # first half is for next token prediction: x_{t+1}
        # second half is for previous token prediction: x_{t+k-1}
        self.output_layer = nn.Linear(hidden_size, vocab_size * 2)

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
                                    bnb_4bit_compute_dtype=args.ptdtype
                                )
        else:
            # Assume default or no quantization
            quantization_config = None

        if args.use_flash:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None
        
        # TODO: support more pretrained models
        # import gpt2 with no specific head on top
        self.model = AutoModel.from_pretrained(
            args.model, 
            attn_implementation=attn_implementation,
            torch_dtype=args.ptdtype,
            quantization_config=quantization_config
        )

        # lora adapter config
        lora_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            # target_modules=["q_proj", "v_proj"],  # apply lora to attention layers
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=True
        )

        # create separate forward and backward lora adapters
        self.model.add_adapter(lora_config, adapter_name="forward_encoder")
        self.model.add_adapter(lora_config, adapter_name="backward_encoder")

        # enable gradient checkpointing to save memory during training
        self.model.gradient_checkpointing_enable()

        # add tied text head for next and previous token predictions
        self.text_head = TiedTextHead(
                            input_dim=self.model.config.hidden_size * 2,
                            hidden_size=512, # TODO; allow this to be configurable
                            vocab_size=args.vocab_size,
                            # tied_weights=self.model.transformer.wte.weight  # use input embeddings' weights
                        )

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
    
    def belief_state_objective(self, all_f, all_b, x):
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
        nt_idxs = (combinations[:, 0] + 1)  # indices for next token labels

        # gather forward and backward features
        f = forward_states[:, f_idxs]
        b = backward_states[:, b_idxs]

        # prepare labels
        single_labels_f = x[:, nt_idxs].unsqueeze(2)  # labels for next-token prediction
        single_labels_b = x[:, b_idxs].unsqueeze(2)   # labels for prev-token prediction
        single_labels = torch.cat((single_labels_f, single_labels_b), dim=2)

        # compute logits from the text head
        logits = self.text_head(f, b)  # combine forward and backward states
        fb_numpairs = fb_pairs.shape[0]  # no of valid forward-backward pairs

        # reshape logits and labels for loss computation
        logits = logits.reshape((bs, fb_numpairs, 2, -1))  # split into next and previous logits
        logits = logits.reshape((bs * fb_numpairs * 2, -1))  # flatten for CEL
        single_labels = single_labels.reshape((bs * fb_numpairs * 2))  # flatten labels

        # compute the loss independently for next and previous token predictions
        # this also sums the negative log likelihood for 
        # all next and previous token predictions together, aligning with the paper
        loss = nn.CrossEntropyLoss()(logits, single_labels)
        return loss
    
    def update(self, x, optimizer, scaler, targets=None):
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
        loss = self.belief_state_objective(_f, _b, x)
        scaler.scale(loss).backward()

        self.model.set_adapter("forward_encoder")
        forward_states.backward(_f.grad)

        self.model.set_adapter("backward_encoder")
        backward_states.backward(_b.grad)

        self.model.set_adapter(["forward_encoder", "backward_encoder"])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return loss
    
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

