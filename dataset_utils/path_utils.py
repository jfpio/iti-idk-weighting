"""
Dataset path utilities for centralized dataset loading configuration.
"""

import os
from typing import Optional
from .const import DATASET_TYPES, DATASETS_PATHS


def get_dataset_path(dataset_type: DATASET_TYPES) -> str:
    """
    Get the file path for a specific dataset type.
    
    Args:
        dataset_type: The type of dataset to load
        
    Returns:
        Absolute path to the dataset file
        
    Raises:
        ValueError: If dataset_type is not configured or path is empty
    """
    if dataset_type not in DATASETS_PATHS:
        raise ValueError(f"Dataset type {dataset_type} not found in DATASETS_PATHS")
    
    relative_path = DATASETS_PATHS[dataset_type]
    if not relative_path or relative_path.strip() == '':
        raise ValueError(f"Path for dataset type {dataset_type} is not configured (empty)")
    
    # Get the project root directory (where this function is called from)
    # Assume we're always called from the project root /workspace/honest_llama
    project_root = os.getcwd()
    
    # Handle both relative and absolute paths
    if os.path.isabs(relative_path):
        return relative_path
    else:
        return os.path.join(project_root, relative_path)


def get_default_dataset_path() -> str:
    """
    Get the default dataset path (ORIGINAL type).
    
    Returns:
        Absolute path to the original TruthfulQA dataset
    """
    return get_dataset_path(DATASET_TYPES.ORIGINAL)


def get_dataset_path_by_name(dataset_name: str) -> str:
    """
    Get dataset path by string name (convenience function).
    
    Args:
        dataset_name: String name of dataset type (e.g., 'original', 'true_only')
        
    Returns:
        Absolute path to the dataset file
        
    Raises:
        ValueError: If dataset_name is not a valid dataset type
    """
    try:
        dataset_type = DATASET_TYPES(dataset_name)
        return get_dataset_path(dataset_type)
    except ValueError as e:
        valid_names = [dt.value for dt in DATASET_TYPES]
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Valid names: {valid_names}") from e


def list_available_datasets() -> dict:
    """
    List all available dataset types and their configured paths.
    
    Returns:
        Dictionary mapping dataset type names to their paths (or 'NOT_CONFIGURED')
    """
    available = {}
    for dataset_type in DATASET_TYPES:
        try:
            path = get_dataset_path(dataset_type)
            available[dataset_type.value] = path
        except ValueError:
            available[dataset_type.value] = 'NOT_CONFIGURED'
    
    return available


def validate_dataset_exists(dataset_type: DATASET_TYPES) -> bool:
    """
    Check if a dataset file actually exists on disk.
    
    Args:
        dataset_type: The dataset type to check
        
    Returns:
        True if the dataset file exists, False otherwise
    """
    try:
        path = get_dataset_path(dataset_type)
        return os.path.isfile(path)
    except ValueError:
        return False