import pandas as pd
from dataset_utils.idk_rephraser import IDKRephraser

def load_csv_as_mc2_dataset(csv_path):
    """Convert TruthfulQA CSV to HuggingFace dataset format for MC2"""

    df = pd.read_csv(csv_path)
    dataset = []
    
    # Initialize IDK rephraser for classification
    rephraser = IDKRephraser()
    
    for i, row in df.iterrows():
        # Parse correct and incorrect answers
        correct_answers = [ans.strip() for ans in row['Correct Answers'].split(';')]
        incorrect_answers = [ans.strip() for ans in row['Incorrect Answers'].split(';')]
        
        # Classify each correct answer as TRUE or IDK using flexible substring matching
        is_idk_flags = [rephraser.is_idk_response(ans) for ans in correct_answers]
        
        # Create choices list and labels (1 for correct, 0 for incorrect)
        choices = correct_answers + incorrect_answers
        labels = [1] * len(correct_answers) + [0] * len(incorrect_answers)
        
        dataset.append({
            'question': row['Question'],
            'mc2_targets': {
                'choices': choices,
                'labels': labels,
                'is_idk': is_idk_flags  # NEW: IDK classification metadata
            },
            'category': row['Category']
        })
    
    return dataset

def load_csv_as_gen_dataset(csv_path):
    """Convert TruthfulQA CSV to HuggingFace dataset format for generation"""
    df = pd.read_csv(csv_path)
    dataset = []
    
    for i, row in df.iterrows():
        # Parse correct and incorrect answers
        correct_answers = [ans.strip() for ans in row['Correct Answers'].split(';')]
        incorrect_answers = [ans.strip() for ans in row['Incorrect Answers'].split(';')]
        
        dataset.append({
            'question': row['Question'],
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers,
            'category': row['Category']
        })
    
    return dataset