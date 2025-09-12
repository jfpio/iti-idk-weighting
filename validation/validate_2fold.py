# Using pyvene to validate_2fold

import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

import sys
sys.path.append('../')

# Specific pyvene imports
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
from interveners import wrapper, Collector, ITI_Intervener
from dataset_utils.path_utils import get_default_dataset_path
import pyvene as pv

HF_NAMES = {
    # Base models
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Llama-3.1-8B',
    'llama3_8B_instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',

    # HF edited models (ITI baked-in)
    'honest_llama_7B': 'jujipotle/honest_llama_7B', # Heads=48, alpha=15
    'honest_llama2_chat_7B': 'jujipotle/honest_llama2_chat_7B', # Heads=48, alpha=15
    'honest_llama2_chat_13B': 'jujipotle/honest_llama2_chat_13B', # Heads=48, alpha=15
    'honest_llama2_chat_70B': 'jujipotle/honest_llama2_chat_70B', # Heads=48, alpha=15
    'honest_llama3_8B_instruct': 'jujipotle/honest_llama3_8B_instruct', # Heads=48, alpha=15
    'honest_llama3_70B_instruct': 'jujipotle/honest_llama3_70B_instruct', # Heads=48, alpha=15
    # Locally edited models (ITI baked-in)
    'local_llama_7B': 'results_dump/edited_models_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_7B': 'results_dump/edited_models_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_13B': 'results_dump/edited_models_dump/llama2_chat_13B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_70B': 'results_dump/edited_models_dump/llama2_chat_70B_seed_42_top_48_heads_alpha_15',
    'local_llama3_8B_instruct': 'results_dump/edited_models_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15',
    'local_llama3_70B_instruct': 'results_dump/edited_models_dump/llama3_70B_instruct_seed_42_top_48_heads_alpha_15'
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix to model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--sequential_loading', action='store_true', default=True, help='unload main model before loading judge models to save memory')
    parser.add_argument('--no-sequential_loading', dest='sequential_loading', action='store_false', help='disable sequential loading (requires more GPU memory)')
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(get_default_dataset_path())
    df = df.iloc[:10]
    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)

    # create model
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    # load activations 
    head_wise_activations = np.load(f"../features/{args.model_name}_{args.dataset_name}_head_wise.npy")
    labels = np.load(f"../features/{args.model_name}_{args.dataset_name}_labels.npy")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    tuning_activations = np.load(f"../features/{args.model_name}_{activations_dataset}_head_wise.npy")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    tuning_labels = np.load(f"../features/{args.model_name}_{activations_dataset}_labels.npy")

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"splits/fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"splits/fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_idxs].to_csv(f"splits/fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        else:
            com_directions = None
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)

        print("Heads intervened: ", sorted(top_heads))

        interveners = []
        pv_config = []
        top_heads_by_layer = {}
        for layer, head, in top_heads:
            if layer not in top_heads_by_layer:
                top_heads_by_layer[layer] = []
            top_heads_by_layer[layer].append(head)
        for layer, heads in top_heads_by_layer.items():
            direction = torch.zeros(head_dim * num_heads).to("cpu")
            for head in heads:
                dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer, head, num_heads)], dtype=torch.float32).to("cpu")
                dir = dir / torch.norm(dir)
                activations = torch.tensor(tuning_activations[:,layer,head,:], dtype=torch.float32).to("cpu") # batch x 128
                proj_vals = activations @ dir.T
                proj_val_std = torch.std(proj_vals)
                direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
            intervener = ITI_Intervener(direction, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
            interveners.append(intervener)
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        intervened_model = pv.IntervenableModel(pv_config, model)
        
        filename = f'{args.model_prefix}{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'

        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'
                                
        curr_fold_results = alt_tqa_evaluate(
            models={args.model_name: intervened_model},
            metric_names=['judge', 'info', 'mc'],
            input_path=f'splits/fold_{i}_test_seed_{args.seed}.csv',
            output_path=f'results_dump/answer_dump/{filename}.csv',
            summary_path=f'results_dump/summary_dump/{filename}.csv',
            device="cuda", 
            interventions=None, 
            intervention_fn=None, 
            instruction_prompt=args.instruction_prompt,
            sequential_loading=args.sequential_loading
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)
    
    # Handle case where CE Loss and KL divergence might not be available (sequential loading)
    ce_loss = final[4] if len(final) > 4 else np.nan
    kl_wrt_orig = final[5] if len(final) > 5 else np.nan
    
    print(f'alpha: {args.alpha}, heads: {args.num_heads}, True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {ce_loss}, KL wrt Original: {kl_wrt_orig}')

if __name__ == "__main__":
    main()
