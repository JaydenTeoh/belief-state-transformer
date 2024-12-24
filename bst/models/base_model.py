import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.cache import Cache
from models.lib import Attention, MLP, LayerNorm
from utils.training_utils import accuracy

from bst import BeliefStateTransformer


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config, layer_idx, rotary=False)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.n_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.n_embd)
        if config.use_cache:
            # Instantiated but not occupying memory yet
            self.cache = Cache(config)
        else:
            self.cache = None
    
    def forward(self, x):
        for block in self.layers:
            x = block(x, self.cache)
        x = self.final_layernorm(x)
        return x

class BaseBST(BeliefStateTransformer):
    def __init__(self, args):
        super().__init__(args)
        self.create_text_head(args)
        self.config = args

        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.n_embd)

        self.pos_encoding = nn.Embedding(args.block_size, args.n_embd)
        # self.embed_tokens.weight = self.lm_head.weight  # TODO: check if correct to remove weight tying

        self.forward_encoder = Encoder(args)
        self.backward_encoder = Encoder(args)

        self.use_cache = args.use_cache

        # Initialize weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('mlp.projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layer))

        # report number of parameters
        all_params, non_emb_params = self.get_num_params()
        print("Number of parameters: %.2fM" % (all_params/1e6,),
              " Number of non-embedding parameters: %.2fM" % (non_emb_params/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        all_params = sum(p.numel() for p in self.parameters())
        non_emb_params = all_params

        if non_embedding:
            # Count the parameters of the embedding and head if not tied
            # if self.embed_tokens != self.lm_head:
            #     non_emb_params -= self.embed_tokens.weight.numel()
            #     non_emb_params -= self.lm_head.weight.numel()
            # else:
            non_emb_params -= self.embed_tokens.weight.numel()
            # Subtract positional embeddings if used
            non_emb_params -= self.pos_encoding.weight.numel()
            non_emb_params = non_emb_params

        return all_params, non_emb_params
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.pos_encoding.weight = nn.Parameter(self.pos_encoding.weight[:block_size])
        for block in self.layers:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_latent_states(self, x, direction):
        """
        Get the forward or backward latent states of the model.
        """
        _, seq_len = x.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only " \
                                                  f"{self.config.block_size}"
        if direction == "forward":
            return self.forward_encoder(x)
        
        # else backward
        return self.backward_encoder(x)


    def set_cache(self, device=None, mode=True):
        """
        Activates caching. After set_cache() memory is allocated and cache is ready to be populated
        """
        self.forward_encoder.cache.use_caching = mode
        self.backward_encoder.cache.use_caching = mode
        for cache in [self.forward_encoder.cache, self.backward_encoder.cache]:
            if mode and cache is None:
                # Allocate memory for caching
                cache.build(device)

    def empty_cache(self):
        """
        Free memory by removing cache.
        """
        self.set_cache(mode=False)
        self.forward_encoder.cache.delete()
        self.backward_encoder.cache.delete()

    def reset_cache(self):
        """
        Set cache back to zero entries
        """
        self.forward_encoder.cache.empty()
        self.backward_encoder.cache.empty()

    def set_encoder(self, name):
        """
        Set the encoder to forward or backward.
        """
        pass