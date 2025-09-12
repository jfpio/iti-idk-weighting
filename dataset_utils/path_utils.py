"""
Dataset path utilities for centralized dataset loading configuration.
"""

import os

def get_default_dataset_path() -> str:
    relative_path = "datasets/TruthfulQA_original.csv"
    
    # Get the project root directory by finding the location of this module
    # dataset_utils/path_utils.py is in dataset_utils/, so project root is parent dir
    module_dir = os.path.dirname(os.path.abspath(__file__))  # .../dataset_utils
    project_root = os.path.dirname(module_dir)  # .../honest_llama (project root)
    
    # Handle both relative and absolute paths
    if os.path.isabs(relative_path):
        return relative_path
    else:
        return os.path.join(project_root, relative_path)
