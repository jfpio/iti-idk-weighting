# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Inference-Time Intervention (ITI)** for large language models, specifically designed to enhance truthfulness in LLaMA, Alpaca, and Vicuna models. The codebase uses the [pyvene](https://github.com/stanfordnlp/pyvene) library for mechanistic interventions on transformer models.

## Environment Setup

The project uses conda for environment management:

```bash
conda env create -f environment.yaml
conda activate iti
python -m ipykernel install --user --name iti --display-name "iti"
```

Required directories for results:
```bash
mkdir -p validation/results_dump/answer_dump
mkdir -p validation/results_dump/summary_dump
mkdir -p validation/results_dump/edited_models_dump
mkdir validation/splits
mkdir validation/sweeping/logs
mkdir get_activations/logs
mkdir features
```

Clone TruthfulQA dependency:
```bash
git clone https://github.com/sylinrl/TruthfulQA.git
```

## Core Workflow Commands

### 1. Get Model Activations
```bash
cd get_activations
python get_activations.py --model_name llama3_8B_instruct --dataset_name tqa_mc2
```

Or use batch scripts:
```bash
bash get_activations.sh  # Single model
bash sweep_activations.sh  # Multiple models via SLURM
```

### 2. Validate ITI Performance
```bash
cd validation
CUDA_VISIBLE_DEVICES=0 python validate_2fold.py \
    --model_name llama_7B \
    --num_heads 48 \
    --alpha 15 \
    --device 0 \
    --num_fold 2 \
    --use_center_of_mass \
    --instruction_prompt default \
    --judge_name <your-GPT-judge-name> \
    --info_name <your-GPT-info-name>
```

### 3. Create ITI-Modified Models
```bash
cd validation
python edit_weight.py --model_name llama2_chat_7B
python push_hf.py  # Upload to HuggingFace
```

## Architecture Overview

### Core Components

- **`utils.py`**: Central utilities for model loading, data processing, activation extraction, and TruthfulQA evaluation
- **`interveners.py`**: Intervention classes including `ITI_Intervener` and `Collector` for pyvene integration
- **`get_activations/get_activations.py`**: Extracts and stores layer-wise and head-wise activations
- **`validation/validate_2fold.py`**: Main evaluation script for ITI performance on TruthfulQA
- **`validation/edit_weight.py`**: Creates models with baked-in ITI interventions

### Model Support

Models are defined in `utils.py` `ENGINE_MAP`:
- LLaMA: 7B, 2-Chat (7B/13B/70B), 3 (8B/70B), 3-Instruct (8B/70B)
- Alpaca 7B, Vicuna 7B

### Intervention Methods

1. **Runtime ITI**: Apply interventions during inference using pyvene
2. **Baked-in ITI**: Modify model weights to include intervention effects permanently
3. **Legacy ITI**: Original baukit-based implementation (in `legacy/` folder)

## Dataset Formats

- **TruthfulQA MC2**: Multiple choice format (`tqa_mc2`)
- **TruthfulQA Generation**: Question-answer pairs (`tqa_gen`, `tqa_gen_end_q`)
- **Transfer datasets**: Modified NQ-Open and Trivia-QA for evaluation

## Important Notes

- **Multi-GPU**: For large models (70B), omit `CUDA_VISIBLE_DEVICES=0` and use multiple GPUs
- **Local models**: Use `--model_prefix "local_"` when loading locally downloaded models
- **OpenAI API**: Required for TruthfulQA evaluation metrics (GPT-judge, GPT-info)
- **Pyvene transition**: Recent updates use pyvene instead of baukit for better model compatibility

## Evaluation Metrics

- **MC1/MC2**: Multiple choice accuracy on TruthfulQA
- **True Score**: Truthfulness based on GPT-judge
- **Info Score**: Informativeness based on GPT-info
- **CE Loss**: Cross-entropy loss on OpenWebText
- **KL Divergence**: KL divergence from original model