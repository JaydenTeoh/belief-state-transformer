import torch
from transformers import Conv1D
from peft import LoraConfig

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 
            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    # check https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220
    # for target modules mapping
    return list(set(layer_names))

def get_bst_model(args):
    if args.pretrained:
        from bst.models.pretrained_model import PretrainBST
        lora_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            target_modules=args.lora_target_modules,  # apply lora to attention layers
            layers_to_transform=args.lora_layers,
            layers_pattern=args.lora_layer_pattern,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=True
        )
        args.lora_config = lora_config
        model = PretrainBST(args)
    else:
        from bst.models.base_model import BaseBST
        model = BaseBST(args)

    return model