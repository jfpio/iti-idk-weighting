from enum import Enum

class DATASET_TYPES(Enum):
    ORIGINAL = 'original'
    TRUE_ONLY = 'true'
    TRUE_AND_IDK_REPHRASED = 'true_and_idk_rephrased'
    TRUE_2xIDK_REPHRASED = 'true_and_2xidk_rephrased'
    
    
    
DATASETS_PATHS = {
    DATASET_TYPES.ORIGINAL: 'datasets/TruthfulQA_original.csv',
    DATASET_TYPES.TRUE_AND_IDK_REPHRASED: '',
    DATASET_TYPES.TRUE_ONLY: '',
    DATASET_TYPES.TRUE_2xIDK_REPHRASED: '',
}