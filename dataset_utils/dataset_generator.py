#!/usr/bin/env python3
"""
Dataset Generator for IDK Steering Research

Generates the 4 experimental conditions (C0-C3) for comparing different 
positive sets in Mass Mean Shift steering directions:
- C0: TRUE ∪ IDK (original ITI approach)
- C1: TRUE-only 
- C2: TRUE ∪ rephrased-IDK (diverse IDK phrasings)
- C3: TRUE ∪ 2×IDK (oversampled IDK)
"""

import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Tuple, Set
from enum import Enum
from .const import DATASET_TYPES, DATASETS_PATHS
from .idk_rephraser import IDKRephraser


class ResponseType(Enum):
    """Types of responses in TruthfulQA"""
    TRUE = "true"
    IDK = "idk" 
    FALSE = "false"


class TruthfulQAGenerator:
    """Generator for TruthfulQA dataset variants for IDK steering research"""
    
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
        
        # Load original data
        self.df = pd.read_csv(original_csv_path)
        print(f"Loaded {len(self.df)} questions from {original_csv_path}")
        
        # IDK detection patterns
        self.idk_patterns = [
            r"i have no comment",
            r"i don't know", r"i do not know",
            r"i'm not sure", r"i am not sure", 
            r"not sure",
            r"uncertain", r"unclear", 
            r"ambiguous",
            r"no recorded", r"no known",
            r"cannot say", r"can't say",
            r"don't have.*information", r"do not have.*information",
            r"no information",
            r"question is ambiguous",
            r"there is no.*known", r"there are no.*known",
            r"mystery", r"unknown",
            r"insufficient.*information", r"lack.*information"
        ]
        
        # Compile regex patterns
        self.idk_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.idk_patterns]
    
    def _classify_answer(self, answer: str) -> ResponseType:
        """
        Classify an answer as TRUE, IDK, or FALSE based on content
        
        Args:
            answer: Answer text to classify
            
        Returns:
            ResponseType classification
        """
        answer_lower = answer.lower().strip()
        
        # Check for IDK patterns
        for pattern in self.idk_regex:
            if pattern.search(answer_lower):
                return ResponseType.IDK
        
        # Default to TRUE for correct answers (we'll separate FALSE answers differently)
        return ResponseType.TRUE
    
    def _extract_answers_by_type(self, row: pd.Series) -> Dict[ResponseType, List[str]]:
        """
        Extract and classify answers from a TruthfulQA row
        
        Args:
            row: DataFrame row containing question and answers
            
        Returns:
            Dict mapping ResponseType to lists of answers
        """
        # Parse correct and incorrect answers
        correct_answers = [ans.strip() for ans in row['Correct Answers'].split(';')]
        incorrect_answers = [ans.strip() for ans in row['Incorrect Answers'].split(';')]
        
        # Classify correct answers into TRUE vs IDK
        true_answers = []
        idk_answers = []
        
        for answer in correct_answers:
            if self._classify_answer(answer) == ResponseType.IDK:
                idk_answers.append(answer)
            else:
                true_answers.append(answer)
        
        return {
            ResponseType.TRUE: true_answers,
            ResponseType.IDK: idk_answers,
            ResponseType.FALSE: incorrect_answers
        }
    
    def _create_base_row(self, original_row: pd.Series) -> Dict:
        """Create base row structure maintaining TruthfulQA format"""
        return {
            'Type': original_row['Type'],
            'Category': original_row['Category'], 
            'Question': original_row['Question'],
            'Best Answer': original_row['Best Answer'],
            'Best Incorrect Answer': original_row['Best Incorrect Answer'],
            'Source': original_row['Source']
        }
    
    def generate_c0_original(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate C0 dataset: TRUE ∪ IDK (original ITI approach)
        This is just a copy of the original dataset
        
        Args:
            output_path: Optional path to save CSV
            
        Returns:
            DataFrame with C0 dataset
        """
        df_c0 = self.df.copy()
        
        if output_path:
            df_c0.to_csv(output_path, index=False)
            print(f"Saved C0 dataset to {output_path}")
        
        return df_c0
    
    def generate_c1_true_only(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate C1 dataset: TRUE-only (remove IDK from correct answers)
        
        Args:
            output_path: Optional path to save CSV
            
        Returns:
            DataFrame with C1 dataset
        """
        rows = []
        
        for _, row in self.df.iterrows():
            answers_by_type = self._extract_answers_by_type(row)
            true_answers = answers_by_type[ResponseType.TRUE]
            false_answers = answers_by_type[ResponseType.FALSE]
            
            # Skip questions with no TRUE answers
            if not true_answers:
                continue
            
            # Create new row with only TRUE answers
            new_row = self._create_base_row(row)
            new_row['Correct Answers'] = '; '.join(true_answers)
            new_row['Incorrect Answers'] = '; '.join(false_answers)
            
            # Update Best Answer if it was IDK
            if self._classify_answer(row['Best Answer']) == ResponseType.IDK:
                new_row['Best Answer'] = true_answers[0]  # Use first TRUE answer
            
            rows.append(new_row)
        
        df_c1 = pd.DataFrame(rows)
        
        if output_path:
            df_c1.to_csv(output_path, index=False)
            print(f"Saved C1 dataset to {output_path}")
        
        return df_c1
    
    def generate_c2_rephrased_idk(self, output_path: str = None) -> pd.DataFrame:
        """
        Generate C2 dataset: TRUE ∪ rephrased-IDK (replace IDK with paraphrases)
        
        Args:
            output_path: Optional path to save CSV
            
        Returns:
            DataFrame with C2 dataset
        """
        rows = []
        rephrasing_log = []
        
        for _, row in self.df.iterrows():
            answers_by_type = self._extract_answers_by_type(row)
            true_answers = answers_by_type[ResponseType.TRUE]
            idk_answers = answers_by_type[ResponseType.IDK]
            false_answers = answers_by_type[ResponseType.FALSE]
            
            # Rephrase IDK answers
            rephrased_idk = []
            for idk_answer in idk_answers:
                rephrased = self.rephraser.rephrase_idk_answer(idk_answer, row['Question'])
                rephrased_idk.append(rephrased)
                rephrasing_log.append({
                    'question': row['Question'],
                    'original_idk': idk_answer,
                    'rephrased_idk': rephrased
                })
            
            # Combine TRUE + rephrased IDK
            all_correct = true_answers + rephrased_idk
            
            new_row = self._create_base_row(row)
            new_row['Correct Answers'] = '; '.join(all_correct) if all_correct else ''
            new_row['Incorrect Answers'] = '; '.join(false_answers)
            
            # Update Best Answer if it was IDK
            if self._classify_answer(row['Best Answer']) == ResponseType.IDK and rephrased_idk:
                new_row['Best Answer'] = rephrased_idk[0]
            
            rows.append(new_row)
        
        df_c2 = pd.DataFrame(rows)
        
        # Save rephrasing log
        if output_path:
            df_c2.to_csv(output_path, index=False)
            print(f"Saved C2 dataset to {output_path}")
            
            # Save rephrasing log
            log_path = output_path.replace('.csv', '_rephrasing_log.csv')
            pd.DataFrame(rephrasing_log).to_csv(log_path, index=False)
            print(f"Saved rephrasing log to {log_path}")
        
        return df_c2
    
    def generate_c3_oversampled_idk(self, output_path: str = None, oversample_factor: float = 2.0) -> pd.DataFrame:
        """
        Generate C3 dataset: TRUE ∪ 2×IDK (oversample IDK responses)
        
        Args:
            output_path: Optional path to save CSV
            oversample_factor: How much to oversample IDK (2.0 = double)
            
        Returns:
            DataFrame with C3 dataset
        """
        rows = []
        
        for _, row in self.df.iterrows():
            answers_by_type = self._extract_answers_by_type(row)
            true_answers = answers_by_type[ResponseType.TRUE]
            idk_answers = answers_by_type[ResponseType.IDK]
            false_answers = answers_by_type[ResponseType.FALSE]
            
            # Oversample IDK answers
            if idk_answers:
                # Calculate how many extra IDKs to add
                current_idk_count = len(idk_answers)
                target_idk_count = int(current_idk_count * oversample_factor)
                extra_needed = target_idk_count - current_idk_count
                
                # Generate additional IDK variants
                extra_idks = []
                for _ in range(extra_needed):
                    base_idk = self.rng.choice(idk_answers)
                    rephrased = self.rephraser.rephrase_idk_answer(base_idk, row['Question'])
                    extra_idks.append(rephrased)
                
                # Combine original + extra IDK
                all_idk = idk_answers + extra_idks
            else:
                all_idk = idk_answers
            
            # Combine TRUE + oversampled IDK
            all_correct = true_answers + all_idk
            
            new_row = self._create_base_row(row)
            new_row['Correct Answers'] = '; '.join(all_correct) if all_correct else ''
            new_row['Incorrect Answers'] = '; '.join(false_answers)
            
            rows.append(new_row)
        
        df_c3 = pd.DataFrame(rows)
        
        if output_path:
            df_c3.to_csv(output_path, index=False)
            print(f"Saved C3 dataset to {output_path}")
        
        return df_c3
    
    def generate_all_datasets(self, output_dir: str = "datasets") -> Dict[str, pd.DataFrame]:
        """
        Generate all 4 dataset conditions (C0-C3)
        
        Args:
            output_dir: Directory to save datasets
            
        Returns:
            Dict mapping condition names to DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating all dataset conditions...")
        
        datasets = {}
        
        # C0: Original (TRUE ∪ IDK)
        c0_path = os.path.join(output_dir, "TruthfulQA_original.csv")
        datasets['C0'] = self.generate_c0_original(c0_path)
        
        # C1: TRUE-only
        c1_path = os.path.join(output_dir, "TruthfulQA_true.csv") 
        datasets['C1'] = self.generate_c1_true_only(c1_path)
        
        # C2: TRUE ∪ rephrased-IDK
        c2_path = os.path.join(output_dir, "TruthfulQA_true_and_idk_rephrased.csv")
        datasets['C2'] = self.generate_c2_rephrased_idk(c2_path)
        
        # C3: TRUE ∪ 2×IDK
        c3_path = os.path.join(output_dir, "TruthfulQA_true_and_2x_idk.csv")
        datasets['C3'] = self.generate_c3_oversampled_idk(c3_path)
        
        print("\n✅ All dataset conditions generated successfully!")
        return datasets
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the original dataset"""
        stats = {
            'total_questions': len(self.df),
            'true_count': 0,
            'idk_count': 0, 
            'false_count': 0,
            'answers_per_question': {
                'correct': [],
                'incorrect': []
            }
        }
        
        for _, row in self.df.iterrows():
            answers_by_type = self._extract_answers_by_type(row)
            
            true_count = len(answers_by_type[ResponseType.TRUE])
            idk_count = len(answers_by_type[ResponseType.IDK])
            false_count = len(answers_by_type[ResponseType.FALSE])
            
            stats['true_count'] += true_count
            stats['idk_count'] += idk_count
            stats['false_count'] += false_count
            
            stats['answers_per_question']['correct'].append(true_count + idk_count)
            stats['answers_per_question']['incorrect'].append(false_count)
        
        # Calculate averages
        stats['avg_correct_per_question'] = np.mean(stats['answers_per_question']['correct'])
        stats['avg_incorrect_per_question'] = np.mean(stats['answers_per_question']['incorrect'])
        
        return stats


def demo_generator():
    """Demonstrate the dataset generator functionality"""
    print("=== Dataset Generator Demo ===\n")
    
    # Initialize generator
    generator = TruthfulQAGenerator('datasets/TruthfulQA_original.csv')
    
    # Show statistics
    stats = generator.get_statistics()
    print("📊 Original Dataset Statistics:")
    print(f"   Total questions: {stats['total_questions']}")
    print(f"   TRUE answers: {stats['true_count']}")
    print(f"   IDK answers: {stats['idk_count']} ({stats['idk_count']/(stats['true_count']+stats['idk_count'])*100:.1f}%)")
    print(f"   FALSE answers: {stats['false_count']}")
    print(f"   Avg correct per question: {stats['avg_correct_per_question']:.1f}")
    print(f"   Avg incorrect per question: {stats['avg_incorrect_per_question']:.1f}")
    print()
    
    # Show example IDK detection
    print("🎯 IDK Detection Examples:")
    for i in range(min(3, len(generator.df))):
        row = generator.df.iloc[i]
        answers_by_type = generator._extract_answers_by_type(row)
        if answers_by_type[ResponseType.IDK]:
            print(f"   Q: {row['Question']}")
            print(f"   TRUE: {answers_by_type[ResponseType.TRUE]}")
            print(f"   IDK: {answers_by_type[ResponseType.IDK]}")
            print()
    
    print("✅ Dataset Generator Demo Complete!")


if __name__ == "__main__":
    demo_generator()