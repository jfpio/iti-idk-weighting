from enum import Enum

class DATASET_TYPES(Enum):
    ORIGINAL = 'original'
    TRUE_AND_IDK_REPHRASED = 'true_and_idk_rephrased'
    TRUE = 'true'
    IDK = 'idk'
    
    
DATASETS_PATHS = {
    DATASET_TYPES.ORIGINAL: 'datasets/TruthfulQA_original.csv',
    DATASET_TYPES.TRUE_AND_IDK_REPHRASED: 'datasets/TruthfulQA_true_and_idk_rephrased.csv',
    DATASET_TYPES.TRUE: 'datasets/TruthfulQA_true.csv',
    DATASET_TYPES.IDK: 'datasets/TruthfulQA_idk.csv',
}