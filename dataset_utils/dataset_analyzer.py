#!/usr/bin/env python3
"""
Dataset Analyzer for IDK Steering Research

Analyzes and compares the 4 experimental conditions (C0-C3) to assess:
- Dataset sizes and TRUE/IDK distributions  
- Response length and complexity differences
- Quality of IDK rephrasing
- Sample comparisons for manual inspection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from collections import Counter
import re
from .const import DATASET_TYPES, DATASETS_PATHS
from .dataset_generator import TruthfulQAGenerator, ResponseType


class DatasetAnalyzer:
    """Analyzer for comparing TruthfulQA dataset variants"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.datasets = {}
        self.stats = {}
    
    def load_datasets(self, dataset_paths: Dict[str, str]) -> None:
        """
        Load multiple datasets for comparison
        
        Args:
            dataset_paths: Dict mapping condition names to file paths
        """
        print("Loading datasets for analysis...")
        
        for condition, path in dataset_paths.items():
            if os.path.exists(path):
                self.datasets[condition] = pd.read_csv(path)
                print(f"   {condition}: {len(self.datasets[condition])} questions from {path}")
            else:
                print(f"   {condition}: File not found - {path}")
        
        print()
    
    def _count_answers(self, answer_string: str) -> int:
        """Count number of semicolon-separated answers"""
        if pd.isna(answer_string) or not answer_string.strip():
            return 0
        return len([ans.strip() for ans in answer_string.split(';') if ans.strip()])
    
    def _get_answer_lengths(self, answer_string: str) -> List[int]:
        """Get lengths of individual answers"""
        if pd.isna(answer_string) or not answer_string.strip():
            return []
        answers = [ans.strip() for ans in answer_string.split(';') if ans.strip()]
        return [len(ans) for ans in answers]
    
    def _classify_idk_answers(self, answer_string: str) -> Tuple[int, int]:
        """
        Classify answers as IDK vs non-IDK
        
        Returns:
            (idk_count, non_idk_count)
        """
        if pd.isna(answer_string) or not answer_string.strip():
            return 0, 0
        
        answers = [ans.strip() for ans in answer_string.split(';') if ans.strip()]
        
        # IDK detection patterns (simplified)
        idk_patterns = [
            r"i have no comment", r"no comment",
            r"i don't know", r"don't know",
            r"i'm not sure", r"not sure", 
            r"uncertain", r"unclear", r"ambiguous",
            r"cannot say", r"can't say",
            r"no recorded", r"no known", r"unknown",
            r"mystery", r"insufficient"
        ]
        
        idk_count = 0
        for answer in answers:
            answer_lower = answer.lower()
            for pattern in idk_patterns:
                if re.search(pattern, answer_lower):
                    idk_count += 1
                    break
        
        return idk_count, len(answers) - idk_count
    
    def analyze_basic_statistics(self) -> Dict[str, Dict]:
        """Analyze basic statistics for each dataset"""
        print("📊 Analyzing Basic Statistics...")
        
        for condition, df in self.datasets.items():
            stats = {
                'total_questions': len(df),
                'correct_answer_counts': [],
                'incorrect_answer_counts': [],
                'correct_answer_lengths': [],
                'incorrect_answer_lengths': [],
                'idk_counts': [],
                'true_counts': []
            }
            
            for _, row in df.iterrows():
                # Count answers
                correct_count = self._count_answers(row['Correct Answers'])
                incorrect_count = self._count_answers(row['Incorrect Answers'])
                stats['correct_answer_counts'].append(correct_count)
                stats['incorrect_answer_counts'].append(incorrect_count)
                
                # Answer lengths
                correct_lengths = self._get_answer_lengths(row['Correct Answers'])
                incorrect_lengths = self._get_answer_lengths(row['Incorrect Answers'])
                stats['correct_answer_lengths'].extend(correct_lengths)
                stats['incorrect_answer_lengths'].extend(incorrect_lengths)
                
                # IDK classification
                idk_count, true_count = self._classify_idk_answers(row['Correct Answers'])
                stats['idk_counts'].append(idk_count)
                stats['true_counts'].append(true_count)
            
            # Calculate summary statistics
            stats['avg_correct_per_question'] = np.mean(stats['correct_answer_counts'])
            stats['avg_incorrect_per_question'] = np.mean(stats['incorrect_answer_counts'])
            stats['avg_correct_answer_length'] = np.mean(stats['correct_answer_lengths']) if stats['correct_answer_lengths'] else 0
            stats['avg_incorrect_answer_length'] = np.mean(stats['incorrect_answer_lengths']) if stats['incorrect_answer_lengths'] else 0
            stats['total_idk_answers'] = sum(stats['idk_counts'])
            stats['total_true_answers'] = sum(stats['true_counts'])
            stats['idk_percentage'] = (stats['total_idk_answers'] / (stats['total_idk_answers'] + stats['total_true_answers'])) * 100 if (stats['total_idk_answers'] + stats['total_true_answers']) > 0 else 0
            
            self.stats[condition] = stats
        
        return self.stats
    
    def print_summary_table(self) -> None:
        """Print summary table of all conditions"""
        print("\n📋 Dataset Summary Table:")
        print("=" * 80)
        
        # Header
        print(f"{'Condition':<12} {'Questions':<10} {'Avg Correct':<12} {'Avg Length':<12} {'IDK %':<8} {'TRUE %':<8}")
        print("-" * 80)
        
        # Data rows
        for condition in ['C0', 'C1', 'C2', 'C3']:
            if condition in self.stats:
                stats = self.stats[condition]
                print(f"{condition:<12} {stats['total_questions']:<10} "
                      f"{stats['avg_correct_per_question']:<12.1f} "
                      f"{stats['avg_correct_answer_length']:<12.1f} "
                      f"{stats['idk_percentage']:<8.1f} "
                      f"{100-stats['idk_percentage']:<8.1f}")
        
        print("=" * 80)
    
    def compare_rephrasing_quality(self, original_condition: str = 'C0', 
                                   rephrased_condition: str = 'C2') -> None:
        """
        Compare rephrasing quality between original and rephrased conditions
        
        Args:
            original_condition: Condition with original IDK responses
            rephrased_condition: Condition with rephrased IDK responses
        """
        if original_condition not in self.datasets or rephrased_condition not in self.datasets:
            print(f"Cannot compare - missing datasets: {original_condition}, {rephrased_condition}")
            return
        
        print(f"\n🔍 Rephrasing Quality Analysis ({original_condition} vs {rephrased_condition}):")
        print("-" * 60)
        
        orig_df = self.datasets[original_condition]
        repr_df = self.datasets[rephrased_condition]
        
        # Find questions with IDK responses for comparison
        comparison_samples = []
        
        for i in range(min(len(orig_df), len(repr_df))):
            orig_row = orig_df.iloc[i]
            repr_row = repr_df.iloc[i]
            
            # Check if this question has IDK responses
            orig_idk, _ = self._classify_idk_answers(orig_row['Correct Answers'])
            repr_idk, _ = self._classify_idk_answers(repr_row['Correct Answers'])
            
            if orig_idk > 0 and repr_idk > 0:
                comparison_samples.append({
                    'question': orig_row['Question'],
                    'original_answers': orig_row['Correct Answers'],
                    'rephrased_answers': repr_row['Correct Answers']
                })
        
        # Show sample comparisons
        print(f"Found {len(comparison_samples)} questions with IDK responses for comparison")
        print("\nSample Comparisons (first 5):")
        
        for i, sample in enumerate(comparison_samples[:5]):
            print(f"\n{i+1}. Q: {sample['question'][:80]}{'...' if len(sample['question']) > 80 else ''}")
            
            # Extract IDK responses
            orig_answers = [ans.strip() for ans in sample['original_answers'].split(';')]
            repr_answers = [ans.strip() for ans in sample['rephrased_answers'].split(';')]
            
            # Find IDK answers in each
            orig_idks = [ans for ans in orig_answers if self._classify_idk_answers(ans)[0] > 0]
            repr_idks = [ans for ans in repr_answers if self._classify_idk_answers(ans)[0] > 0]
            
            print(f"   Original IDK: {orig_idks}")
            print(f"   Rephrased IDK: {repr_idks}")
    
    def analyze_length_distributions(self) -> None:
        """Analyze and plot length distributions"""
        print("\n📏 Response Length Analysis:")
        
        # Collect length data for all conditions
        length_data = {}
        for condition, stats in self.stats.items():
            length_data[condition] = stats['correct_answer_lengths']
        
        # Print summary statistics
        print("\nLength Statistics (characters per answer):")
        print(f"{'Condition':<12} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<6} {'Max':<6}")
        print("-" * 50)
        
        for condition, lengths in length_data.items():
            if lengths:
                mean_len = np.mean(lengths)
                median_len = np.median(lengths)
                std_len = np.std(lengths)
                min_len = np.min(lengths)
                max_len = np.max(lengths)
                
                print(f"{condition:<12} {mean_len:<8.1f} {median_len:<8.1f} "
                      f"{std_len:<8.1f} {min_len:<6} {max_len:<6}")
    
    def export_analysis_report(self, output_path: str) -> None:
        """Export comprehensive analysis report to CSV"""
        report_data = []
        
        for condition, stats in self.stats.items():
            report_data.append({
                'condition': condition,
                'total_questions': stats['total_questions'],
                'avg_correct_per_question': stats['avg_correct_per_question'],
                'avg_incorrect_per_question': stats['avg_incorrect_per_question'],
                'avg_correct_answer_length': stats['avg_correct_answer_length'],
                'avg_incorrect_answer_length': stats['avg_incorrect_answer_length'],
                'total_idk_answers': stats['total_idk_answers'],
                'total_true_answers': stats['total_true_answers'],
                'idk_percentage': stats['idk_percentage']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path, index=False)
        print(f"\n📄 Analysis report exported to: {output_path}")
    
    def run_full_analysis(self, dataset_dir: str = "datasets", 
                          output_dir: str = "datasets") -> None:
        """
        Run complete analysis pipeline
        
        Args:
            dataset_dir: Directory containing dataset CSV files
            output_dir: Directory to save analysis outputs
        """
        print("🔬 Running Full Dataset Analysis")
        print("=" * 50)
        
        # Define dataset paths
        dataset_paths = {
            'C0': os.path.join(dataset_dir, 'TruthfulQA_original.csv'),
            'C1': os.path.join(dataset_dir, 'TruthfulQA_true.csv'),
            'C2': os.path.join(dataset_dir, 'TruthfulQA_true_and_idk_rephrased.csv'),
            'C3': os.path.join(dataset_dir, 'TruthfulQA_true_and_2x_idk.csv')
        }
        
        # Load datasets
        self.load_datasets(dataset_paths)
        
        # Run analysis
        self.analyze_basic_statistics()
        self.print_summary_table()
        self.analyze_length_distributions()
        self.compare_rephrasing_quality('C0', 'C2')
        
        # Export report
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'dataset_analysis_report.csv')
        self.export_analysis_report(report_path)
        
        print("\n✅ Full analysis complete!")


def demo_analyzer():
    """Demonstrate the dataset analyzer functionality"""
    print("=== Dataset Analyzer Demo ===\n")
    
    analyzer = DatasetAnalyzer()
    
    # Load sample dataset (just original for demo)
    if os.path.exists('datasets/TruthfulQA_original.csv'):
        analyzer.load_datasets({'C0': 'datasets/TruthfulQA_original.csv'})
        analyzer.analyze_basic_statistics()
        analyzer.print_summary_table()
    else:
        print("No datasets found for demo")
    
    print("\n✅ Dataset Analyzer Demo Complete!")


if __name__ == "__main__":
    demo_analyzer()