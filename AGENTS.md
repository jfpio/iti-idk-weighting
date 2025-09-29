# Repository Guidelines

## Project Structure & Module Organization
Core orchestration lives in `validation/` (evaluation, weight editing, HF pushes), while activation extraction sits under `get_activations/`. Data loaders and helpers are in `dataset_utils/`, with metrics helpers in `utils.py` and custom intervention logic in `interveners.py`. Datasets (TruthfulQA variants, eval subsets) reside in `datasets/` and `TruthfulQA/`; keep large generated artifacts in `results_dump/` or `validation/sweeping/` to isolate experiments.

## Build, Test, and Development Commands
Set up the GPU-ready conda env with `conda env create -f environment.yaml` and `conda activate iti`. Extract activations via `python get_activations/get_activations.py --model_name llama_7B --dataset_name tqa_mc2`. Run the main evaluation loop with `python validation/validate_2fold.py --model_name llama_7B --sequential_loading --device 0`. To bake interventions into checkpoints, use `python validation/edit_weight.py --model_name llama2_chat_7B`, and publish to Hugging Face via `python validation/push_hf.py --model_name llama2_chat_7B` once results are verified.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and descriptive snake_case identifiers (`get_default_dataset_path`, `flattened_idx_to_layer_head`). Keep CLI entry points self-contained with `if __name__ == "__main__":` guards and argparse-based configuration, mirroring existing scripts. Prefer explicit type hints when touching `utils.py` or shared helpers, and place reusable paths in `dataset_utils/path_utils.py` rather than hard-coding strings.

## Testing & Evaluation Practices
Before pushing changes, rerun `validation/validate_2fold.py` with the target model/dataset pair and confirm fold CSVs in `validation/splits/` match expectations (do not commit generated splits). For quick sanity checks, run a single fold with `--num_fold 1 --val_ratio 0.2` and inspect the printed head list. When modifying dataset loaders, regenerate activations on a small subset (`--max_samples` flag) to ensure shapes align, then spot-check metrics via `validation/evaluate_with_llama_judges.py`.

## Commit & Pull Request Guidelines
Commits should be concise, imperative summaries similar to `Update validate_2fold.py` and grouped by logical change. Reference datasets or models touched (`Edit ITI directions for llama3_8B`) to clarify scope. Pull requests need a short problem statement, the commands used for validation, links to any tracked issues, and (when editing outputs) representative metric tables or logs. Include GPU specs when performance or memory usage is relevant.

## Compute & Environment Notes
Scripts assume CUDA 11.6+ with bfloat16-capable GPUs; enable sequential loading flags when memory is tight. Store credentials (e.g., Hugging Face tokens) via environment variables, and avoid committing `.env` or cached model weights. Large artifacts should be git-ignored or uploaded to external storage when sharing results.
