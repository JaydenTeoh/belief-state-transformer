## Usage
To finetune a GPT2 model as a Belief State Transformer using 4 bit quantization and LoRA adapters on a star graph with degree 2 and path length 5 with 50 possible node values, run the command
> python3 bst_finetune.py --model gpt2 --load_in_4bit --n_layer 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256 --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.0001