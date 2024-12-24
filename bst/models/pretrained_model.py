from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from utils.training_utils import accuracy
from bst.grad_norm import GradNormLossWeighter
from torch.nn import functional as F
import torch.nn as nn
import torch
import wandb

from bst import BeliefStateTransformer


class PretrainBST(BeliefStateTransformer):
    def __init__(self, args):
        super().__init__(args)
        self.config = args
        self.is_pretrained = True

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
        args.n_embd = self.model.config.hidden_size
        self.create_text_head(args)
        # lora adapter config
        lora_config = args.lora_config

        # create separate forward and backward lora adapters
        self.model.add_adapter(lora_config, adapter_name="forward_encoder")
        self.model.add_adapter(lora_config, adapter_name="backward_encoder")

        # assume AutoModel already has caching during generation
        self.use_cache = False

        total_params, trainable_params = self.get_num_params()
        print(f"Total params: {total_params}, Trainable params: {trainable_params}")

    def get_latent_states(self, x, direction):
        # forward pass through the base transformer model, ignore the lm_head
        if direction == "forward":
            self.model.set_adapter("forward_encoder")
        else:
            self.model.set_adapter("backward_encoder")
        return self.model.transformer(x).last_hidden_state
        
    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def set_encoder(self, encoder):
        self.model.set_adapter(encoder)

