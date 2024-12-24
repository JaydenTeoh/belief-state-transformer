import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb

from tokenizing import get_tokenizer
from utils.training_utils import get_run_name, AverageMeter
from data import get_dataset
from evaluate import evaluate
from bst.trainer import BSTTrainer


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=False, help="Whether to train from scratch or use a pretrained model",
    )
parser.add_argument(
        "--text_head_layers", type=int, default=3, help="Number of layers in the text head",
    )
parser.add_argument(
        "--text_head_hidden", type=int, default=512, help="Hidden dim of the text head",
    )
parser.add_argument(
        "--add_eos", action=argparse.BooleanOptionalAction, default=False, help="Add eos token to end of tokenized sequence (act as empty suffix for backward encoder)",
    )

# base model
parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers",
    )
parser.add_argument(
        "--n_embd", type=int, default=768, help="Embedding size",
    )
parser.add_argument(
        "--n_head", type=int, default=6, help="Number of heads",
    )
parser.add_argument(
        "--use_cache", action=argparse.BooleanOptionalAction, default=True, help="USe KV cache",
    )

# pretrained model
parser.add_argument(
        "--model", default='gpt2', type=str, help="Type of model"
    )
parser.add_argument(
        "--load_in_4bit", action=argparse.BooleanOptionalAction, default=False, help="Load in 4-bit",
    )
parser.add_argument(
        "--lora_r", type=int, default=32, help="Lora r",
    )
parser.add_argument(
        "--lora_alpha", type=int, default=64, help="Lora alpha",
    )
parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Lora dropout",
    )
parser.add_argument(
        "--lora_layers", type=list, default=[5, 7, 9, 11], help="Lora layers",
    )
parser.add_argument(
        "--lora_layer_pattern", type=str, default="h", help="Lora layers pattern",
    )
parser.add_argument(
        "--lora_target_modules", type=list, default=["c_attn"], help="Lora target modules",
    )


# training
parser.add_argument(
        "--use_grad_norm", action=argparse.BooleanOptionalAction, default=False, help="Use GradNorm to balance losses",
    )
parser.add_argument(
        "--gradnorm_lr", type=float, default=1e-4, help="GradNorm learning rate",
    )
parser.add_argument(
        "--gradnorm_update_every", type=float, default=100.0, help="Update weightages for loss interval",
    )
parser.add_argument(
        "--gradnorm_alpha", type=float, default=1.5, help="GradNorm alpha",
    )
parser.add_argument(
        "--gradnorm_init_loss_decay", type=float, default=0.95, help="GradNorm alpha",
    )
parser.add_argument(
        "--clip_gradients", action=argparse.BooleanOptionalAction, default=False, help="Use gradient clipping",
    )
parser.add_argument(
        "--clip_grad_norm", type=float, default=10.0, help="Clip gradient max norm",
    )

parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=200000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=10000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--mate_in", default=2, type=int, help="For chess, number of moves to checkmate"
    )
parser.add_argument(
        "--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state",
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=100000, help="Interval (in steps) at which to save model",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
    )
parser.add_argument(
        "--eval_every", type=int, default=10000, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb username",
    )

args = parser.parse_args()
# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model stuff
top_k = 1

# Evaluation stuff
eval_iters = 1000
eval_interval = 5
log_interval = 10

# Optimiser
dtype = 'bfloat16' if args.load_in_4bit else 'float32'
args.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
args.beta1 = 0.9
args.beta2 = 0.999
args.decay_lr = True
args.compile = True if device == 'cuda' else False
args.use_flash = True if device == 'cuda' and args.load_in_4bit else False
args.warmup_iters = 100
args.min_lr = 1e-5

run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and de-tokenizer
args.add_eos = True # IMPORTANT: add eos token to the end of the sequence for BST training
tokenizer = get_tokenizer(args)
if args.add_eos: # IMPORTANT: eos token for null BST suffix during generation
    args.empty_suffix_id = tokenizer.eos_token_id
train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

max_iters = len(train_data) * args.epochs
args.lr_decay_iters = max_iters

args.block_size = train_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
trainer = BSTTrainer(args, device)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=args.dtype)

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__,)
    wandb.run.name = run_name

results = {}
num_iters = 0

for ep in range(args.epochs):
    if ep % args.save_every == 0 and ep > 0:
        trainer.save(path, ep)

    train_bar = tqdm(train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()

    for x, y in train_bar:
        # determine and set the learning rate for this iteration
        with ctx:
            logits, loss, accs = trainer.step(x, y)
        
        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(accs["acc"], x.shape[0] * train_data.num_target_tokens)
        # Backpropagation with mixed precision
        # scaler.scale(loss).backward()

        # # Unscale and update optimizer
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad(set_to_none=True)
        # total_acc.update(accs['acc'], x.shape[0]) # this accuracy include backward encoder embedding
        num_iters += 1
        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}'.format(ep, args.epochs, total_loss.get(),
             total_acc.get(percentage=True))
        )

        # evaluate the loss on train/val sets and write checkpoints
        if num_iters % args.eval_every == 0:
            # Generate sequences and check accuracies
            if args.eval_train:
                results = evaluate(trainer.model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='train', remove_eos=args.add_eos)
                # results = evaluate_forced(model, train_loader, results=results, mode='train')

            results = evaluate(trainer.model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test', remove_eos=args.add_eos)
            # results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')

            if wandb_log:
                wandb.log(results)
