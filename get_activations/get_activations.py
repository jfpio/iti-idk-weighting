# Pyvene method of getting activations
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')

import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
from dataset_utils.load_dataset import load_csv_as_mc2_dataset, load_csv_as_gen_dataset
from dataset_utils.path_utils import get_default_dataset_path
from dataset_utils.binary_sample_loader import load_condition_samples

HF_NAMES = {
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
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct'
}

def tokenized_binary_samples(binary_samples, tokenizer):
    """
    Tokenize binary samples for activation extraction
    
    Args:
        binary_samples: List of binary sample dicts from CSV
        tokenizer: HuggingFace tokenizer
        
    Returns:
        (prompts, labels) ready for activation extraction
    """
    all_prompts = []
    all_labels = []
    
    for i, sample in enumerate(binary_samples):
        # Format as "Q: question A: answer" (matching format_truthfulqa)
        prompt = f"Q: {sample['question']} A: {sample['answer']}"
        
        if i == 0:
            print("Sample prompt:", prompt)
        
        # Tokenize
        prompt = tokenizer(prompt, return_tensors='pt').input_ids
        all_prompts.append(prompt)
        all_labels.append(sample['label'])
    
    return all_prompts, all_labels

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--condition', type=str, default=None, choices=['c0', 'c1', 'c2', 'c3'], 
                        help='IDK steering condition: c0=original, c1=true-only, c2=rephrased-idk, c3=oversampled-idk')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, dtype=torch.float16, device_map="auto")
    device = "cuda"

    # Check if using condition-specific binary samples
    if args.condition:
        print(f"Loading condition-specific binary samples: {args.condition}")
        binary_samples = load_condition_samples(args.condition, base_dir="../datasets/binary_samples")
        print(f"Loaded {len(binary_samples)} binary samples for condition {args.condition}")
        dataset = binary_samples
        formatter = tokenized_binary_samples
    elif args.dataset_name == "tqa_mc2": 
        dataset = load_csv_as_mc2_dataset(get_default_dataset_path())
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_csv_as_gen_dataset(get_default_dataset_path())
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_csv_as_gen_dataset(get_default_dataset_path())
        formatter = tokenized_tqa_gen_end_q
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'../features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations.copy())

    # Create output filename suffix and directory
    if args.condition:
        suffix = f"{args.condition}"
        dataset_name = "binary_samples"
        # Create condition-specific subdirectory
        output_dir = f'../features/{args.condition}'
        os.makedirs(output_dir, exist_ok=True)
    else:
        suffix = args.dataset_name
        dataset_name = args.dataset_name
        output_dir = '../features'

    print(f"Saving to directory: {output_dir}")
    
    print("Saving labels")
    np.save(f'{output_dir}/{args.model_name}_{dataset_name}_{suffix}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'{output_dir}/{args.model_name}_{dataset_name}_{suffix}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'{output_dir}/{args.model_name}_{dataset_name}_{suffix}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
