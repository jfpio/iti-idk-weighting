#!/usr/bin/env python3
"""
Binary Sample Generator for IDK Steering Research

Extracts binary (Question, Answer, True/False) samples from TruthfulQA 
and generates condition-specific CSV files for the 4 experimental conditions:
- C0: TRUE ∪ IDK ∪ FALSE (all samples)
- C1: TRUE ∪ FALSE (filter out IDK samples) 
- C2: TRUE ∪ rephrased-IDK ∪ FALSE (rephrase IDK samples)
- C3: TRUE ∪ 2×IDK ∪ FALSE (oversample IDK samples)

This directly feeds into get_activations.py for precise experimental control.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from .dataset_generator import TruthfulQAGenerator, ResponseType
from .idk_rephraser import IDKRephraser


class BinarySampleGenerator:
    """Generator for binary sample CSV files for each research condition"""
    
    def __init__(self, original_csv_path: str, seed: int = 42):
        """
        Initialize generator with original TruthfulQA dataset
        
        Args:
            original_csv_path: Path to original TruthfulQA.csv
            seed: Random seed for reproducibility
        """
        self.original_path = original_csv_path
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.rephraser = IDKRephraser(seed=seed)
        self.tqa_generator = TruthfulQAGenerator(original_csv_path, seed=seed)
        
        print(f"Initialized binary sample generator with {len(self.tqa_generator.df)} questions")
    
    def extract_binary_samples_from_tqa(self) -> List[Dict]:
        """
        Extract all binary samples from TruthfulQA dataset
        
        Returns:
            List of binary samples: {question, answer, label, category, sample_type}
        """
        binary_samples = []
        
        for _, row in self.tqa_generator.df.iterrows():
            question = row['Question']
            category = row['Category']
            
            # Extract and classify answers
            answers_by_type = self.tqa_generator._extract_answers_by_type(row)
            
            # Add TRUE samples (label=1)
            for answer in answers_by_type[ResponseType.TRUE]:
                binary_samples.append({
                    'question': question,
                    'answer': answer,
                    'label': 1,
                    'category': category,
                    'sample_type': 'TRUE'
                })
            
            # Add IDK samples (label=1, but marked as IDK)
            for answer in answers_by_type[ResponseType.IDK]:
                binary_samples.append({
                    'question': question,
                    'answer': answer,
                    'label': 1,
                    'category': category,
                    'sample_type': 'IDK'
                })
            
            # Add FALSE samples (label=0)
            for answer in answers_by_type[ResponseType.FALSE]:
                binary_samples.append({
                    'question': question,
                    'answer': answer,
                    'label': 0,
                    'category': category,
                    'sample_type': 'FALSE'
                })
        
        return binary_samples
    
    def generate_c0_samples(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate C0: All binary samples (TRUE ∪ IDK ∪ FALSE)
        This is the baseline condition matching original ITI approach
        """
        binary_samples = self.extract_binary_samples_from_tqa()
        df_c0 = pd.DataFrame(binary_samples)
        
        if output_path:
            df_c0.to_csv(output_path, index=False)
            print(f"C0: Saved {len(df_c0)} binary samples to {output_path}")
        
        return df_c0
    
    def generate_c1_samples(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate C1: TRUE ∪ FALSE (filter out IDK binary samples)
        Tests steering with only informative positive samples
        """
        binary_samples = self.extract_binary_samples_from_tqa()
        
        # Filter out IDK samples
        c1_samples = [s for s in binary_samples if s['sample_type'] != 'IDK']
        df_c1 = pd.DataFrame(c1_samples)
        
        if output_path:
            df_c1.to_csv(output_path, index=False)
            print(f"C1: Saved {len(df_c1)} binary samples (filtered {len(binary_samples) - len(df_c1)} IDK samples) to {output_path}")
        
        return df_c1
    
    def generate_c2_samples(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate C2: TRUE ∪ rephrased-IDK ∪ FALSE
        Tests steering with stylistically diverse IDK samples
        """
        binary_samples = self.extract_binary_samples_from_tqa()
        
        # Rephrase IDK samples
        c2_samples = []
        rephrasing_log = []
        
        for sample in binary_samples:
            if sample['sample_type'] == 'IDK':
                # Rephrase IDK answer
                original_answer = sample['answer']
                rephrased_answer = self.rephraser.rephrase_idk_answer(original_answer, sample['question'])
                
                # Create rephrased sample
                rephrased_sample = sample.copy()
                rephrased_sample['answer'] = rephrased_answer
                c2_samples.append(rephrased_sample)
                
                # Log rephrasing
                rephrasing_log.append({
                    'question': sample['question'],
                    'original_idk': original_answer,
                    'rephrased_idk': rephrased_answer
                })
            else:
                # Keep TRUE and FALSE samples unchanged
                c2_samples.append(sample)
        
        df_c2 = pd.DataFrame(c2_samples)
        
        if output_path:
            df_c2.to_csv(output_path, index=False)
            print(f"C2: Saved {len(df_c2)} binary samples (rephrased {len(rephrasing_log)} IDK samples) to {output_path}")
            
            # Save rephrasing log
            log_path = output_path.replace('.csv', '_rephrasing_log.csv')
            pd.DataFrame(rephrasing_log).to_csv(log_path, index=False)
            print(f"C2: Saved rephrasing log to {log_path}")
        
        return df_c2
    
    def generate_c3_samples(self, output_path: str = None, oversample_factor: float = 2.0) -> pd.DataFrame:
        """
        Generate C3: TRUE ∪ 2×IDK ∪ FALSE (oversample IDK binary samples)
        Tests effect of increasing IDK representation in positive set
        """
        binary_samples = self.extract_binary_samples_from_tqa()
        
        # Separate samples by type
        true_samples = [s for s in binary_samples if s['sample_type'] == 'TRUE']
        idk_samples = [s for s in binary_samples if s['sample_type'] == 'IDK']
        false_samples = [s for s in binary_samples if s['sample_type'] == 'FALSE']
        
        # Oversample IDK samples
        original_idk_count = len(idk_samples)
        target_idk_count = int(original_idk_count * oversample_factor)
        extra_idk_needed = target_idk_count - original_idk_count
        
        # Generate additional IDK samples with rephrasing
        extra_idk_samples = []
        for _ in range(extra_idk_needed):
            # Pick random IDK sample to duplicate and rephrase
            base_sample = self.rng.choice(idk_samples).copy()
            rephrased_answer = self.rephraser.rephrase_idk_answer(base_sample['answer'], base_sample['question'])
            base_sample['answer'] = rephrased_answer
            extra_idk_samples.append(base_sample)
        
        # Combine all samples
        c3_samples = true_samples + idk_samples + extra_idk_samples + false_samples
        df_c3 = pd.DataFrame(c3_samples)
        
        if output_path:
            df_c3.to_csv(output_path, index=False)
            print(f"C3: Saved {len(df_c3)} binary samples (added {extra_idk_needed} extra IDK samples) to {output_path}")
        
        return df_c3
    
    def generate_all_binary_datasets(self, output_dir: str = "datasets/binary_samples") -> Dict[str, pd.DataFrame]:
        """
        Generate all 4 binary sample datasets (C0-C3)
        
        Args:
            output_dir: Directory to save binary sample CSV files
            
        Returns:
            Dict mapping condition names to DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("Generating Binary Sample Datasets for IDK Steering Research")
        print("=" * 60)
        
        datasets = {}
        
        # C0: All samples (baseline)
        c0_path = os.path.join(output_dir, "binary_samples_c0_original.csv")
        datasets['C0'] = self.generate_c0_samples(c0_path)
        
        # C1: TRUE + FALSE only (no IDK)
        c1_path = os.path.join(output_dir, "binary_samples_c1_true_only.csv")
        datasets['C1'] = self.generate_c1_samples(c1_path)
        
        # C2: TRUE + rephrased-IDK + FALSE
        c2_path = os.path.join(output_dir, "binary_samples_c2_rephrased_idk.csv")
        datasets['C2'] = self.generate_c2_samples(c2_path)
        
        # C3: TRUE + 2×IDK + FALSE 
        c3_path = os.path.join(output_dir, "binary_samples_c3_oversampled_idk.csv")
        datasets['C3'] = self.generate_c3_samples(c3_path)
        
        print("=" * 60)
        print("✅ All binary sample datasets generated successfully!")
        
        # Print summary statistics
        self.print_summary_statistics(datasets)
        
        return datasets
    
    def print_summary_statistics(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Print summary statistics for all conditions"""
        print("\n📊 Binary Sample Statistics:")
        print("-" * 80)
        print(f"{'Condition':<12} {'Total':<8} {'TRUE':<8} {'IDK':<8} {'FALSE':<8} {'IDK%':<8}")
        print("-" * 80)
        
        for condition, df in datasets.items():
            total = len(df)
            true_count = len(df[df['sample_type'] == 'TRUE'])
            idk_count = len(df[df['sample_type'] == 'IDK']) 
            false_count = len(df[df['sample_type'] == 'FALSE'])
            idk_percentage = (idk_count / (true_count + idk_count)) * 100 if (true_count + idk_count) > 0 else 0
            
            print(f"{condition:<12} {total:<8} {true_count:<8} {idk_count:<8} {false_count:<8} {idk_percentage:<8.1f}")
        
        print("-" * 80)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get detailed statistics about binary sample extraction"""
        binary_samples = self.extract_binary_samples_from_tqa()
        
        stats = {
            'total_samples': len(binary_samples),
            'true_samples': len([s for s in binary_samples if s['sample_type'] == 'TRUE']),
            'idk_samples': len([s for s in binary_samples if s['sample_type'] == 'IDK']),
            'false_samples': len([s for s in binary_samples if s['sample_type'] == 'FALSE']),
            'unique_questions': len(set(s['question'] for s in binary_samples))
        }
        
        stats['positive_samples'] = stats['true_samples'] + stats['idk_samples']
        stats['idk_percentage'] = (stats['idk_samples'] / stats['positive_samples']) * 100 if stats['positive_samples'] > 0 else 0
        
        return stats


def demo_binary_generator():
    """Demonstrate binary sample generation"""
    print("=== Binary Sample Generator Demo ===\n")
    
    # Initialize generator
    generator = BinarySampleGenerator('datasets/TruthfulQA_original.csv')
    
    # Show statistics
    stats = generator.get_statistics()
    print("📊 Binary Sample Statistics:")
    print(f"   Total binary samples: {stats['total_samples']}")
    print(f"   TRUE samples: {stats['true_samples']}")
    print(f"   IDK samples: {stats['idk_samples']} ({stats['idk_percentage']:.1f}%)")
    print(f"   FALSE samples: {stats['false_samples']}")
    print(f"   Unique questions: {stats['unique_questions']}")
    print()
    
    # Show sample binary samples
    binary_samples = generator.extract_binary_samples_from_tqa()
    print("🔍 Sample Binary Samples:")
    for i, sample in enumerate(binary_samples[:5]):
        print(f"{i+1}. Q: {sample['question'][:50]}...")
        print(f"   A: {sample['answer'][:50]}...")
        print(f"   Label: {sample['label']} ({sample['sample_type']})")
        print()
    
    print("✅ Binary Sample Generator Demo Complete!")


if __name__ == "__main__":
    demo_binary_generator()