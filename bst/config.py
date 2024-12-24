from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from peft import LoraConfig

@dataclass
class BaseBSTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True if torch.cuda.is_available() else False
    dtype = torch.bfloat16
    cache: bool = True
    max_bsz: int = 16
    text_head_layers: int = 3
    text_head_hidden: int = 512
    eos_token_id: int = 50256

@dataclass
class PretrainBSTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_flash: bool = True if torch.cuda.is_available() else False
    dtype = torch.bfloat16
    cache: bool = True
    max_bsz: int = 16
    text_head_layers: int = 3
    text_head_hidden: int = 512
    eos_token_id: int = 50256
    load_in_4bit: bool = False
    lora_config: LoraConfig = None