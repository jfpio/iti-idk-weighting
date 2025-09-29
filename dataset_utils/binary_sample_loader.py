#!/usr/bin/env python3
"""
Binary Sample Loader for get_activations.py

Loads condition-specific binary sample CSV files and formats them 
for direct use in the activation extraction pipeline.
"""

import pandas as pd
from typing import List, Tuple, Dict
import torch


def load_binary_samples(csv_path: str) -> List[Dict]:
    """
    Load binary samples from CSV file
    
    Args:
        csv_path: Path to binary sample CSV file
        
    Returns:
        List of binary samples with question, answer, label, category
    """
    df = pd.read_csv(csv_path)
    
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'question': row['question'],
            'answer': row['answer'], 
            'label': int(row['label']),
            'category': row['category'],
            'sample_type': row['sample_type']
        })
    
    return samples


def format_binary_samples_for_activations(samples: List[Dict], tokenizer, device='cuda') -> Tuple[List[str], List[int]]:
    """
    Format binary samples for activation extraction pipeline
    
    Args:
        samples: List of binary sample dicts
        tokenizer: HuggingFace tokenizer
        device: Device for tokenization
        
    Returns:
        (prompts, labels) ready for get_activations.py
    """
    prompts = []
    labels = []
    
    for sample in samples:
        # Format as "Q: question A: answer"
        prompt = f"Q: {sample['question']} A: {sample['answer']}"
        
        # Tokenize and move to device
        tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        
        prompts.append(tokenized)
        labels.append(sample['label'])
    
    return prompts, labels


def get_condition_path(condition: str, base_dir: str = "datasets/binary_samples") -> str:
    """
    Get path to binary sample file for given condition
    
    Args:
        condition: Condition name (c0, c1, c2, c3)
        base_dir: Base directory containing binary sample files
        
    Returns:
        Path to condition-specific binary sample CSV
    """
    import os
    
    condition_files = {
        'c0': 'binary_samples_c0_original.csv',
        'c1': 'binary_samples_c1_true_only.csv', 
        'c2': 'binary_samples_c2_rephrased_idk.csv',
        'c3': 'binary_samples_c3_oversampled_idk.csv'
    }
    
    if condition not in condition_files:
        raise ValueError(f"Invalid condition '{condition}'. Must be one of: {list(condition_files.keys())}")
    
    return os.path.join(base_dir, condition_files[condition])


def load_condition_samples(condition: str, base_dir: str = "datasets/binary_samples") -> List[Dict]:
    """
    Load binary samples for a specific condition
    
    Args:
        condition: Condition name (c0, c1, c2, c3)  
        base_dir: Base directory containing binary sample files
        
    Returns:
        List of binary samples for the condition
    """
    csv_path = get_condition_path(condition, base_dir)
    return load_binary_samples(csv_path)