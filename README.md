### Update 08/24/2024
With the release of LLaMA-3 models, I decided to replicate ITI on a suite of LLaMA models for easy comparison. I've recorded the results in `iti_replication_results.md` and uploaded the ITI baked-in models to HuggingFace [here](https://huggingface.co/collections/jujipotle/inference-time-intervention-iti-models-66ca15448347e21e8af6772e). Note that the ITI baked-in models and ITI applied to base models is not exactly a one-to-one comparison due to slight differences in when the activations are edited. The ITI baked-in models have the activation differences hardcoded into their attention biases. For more precise editing, consider only using the models' attention biases when processing tokens after the input prompt, to be more faithful to the original ITI method.

-- Justin Ji @jujipotle

### Update 01/26/2024 :fire::fire:

[Zen](https://github.com/frankaging) provided this really cool library called [pyvene](https://github.com/stanfordnlp/pyvene) that can be used to load Inference-time Intervention, and many other mechanistic intervention technique. Here is what he says:

[pyvene](https://github.com/stanfordnlp/pyvene)  pushes for streamlining the sharing process of inference-time interventions and many more, comparing with other also super useful tools in this area!

I created the activation diff (~0.14MB) based on your shared LLaMA-2-chat by taking the bias terms. And your honest-llama can now be loaded as,

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    dtype=torch.bfloat16,
).to("cuda")

pv_model = pv.IntervenableModel.load(
    # the activation diff ~0.14MB
    "zhengxuanzenwu/intervenable_honest_llama2_chat_7B", 
    model,
)

q = "What's a cure for insomnia that always works?"
prompt = tokenizer(q, return_tensors="pt").to("cuda")
_, iti_response_shared = pv_model.generate(
    prompt, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(iti_response_shared[0], skip_special_tokens=True))
```
This can be extended to other datasets easily I think by loading different activation diffs.

Huggingface repo is here: https://huggingface.co/zhengxuanzenwu/intervenable_honest_llama2_chat_7B. This intervention schema intervenes on every decoding step but not the prompt. But we can also do other things if needed.

Colab: https://colab.research.google.com/github/stanfordnlp/pyvene/blob/main/pyvene_101.ipynb#scrollTo=1c7b90b0

 
### Update 07/21/2023 :fire:

I edited a LLaMA-2 model to have baked-in intervention learnt from TruthfulQA dataset. The model is available [here](https://huggingface.co/likenneth/honest_llama2_chat_7B). You can load and play around starting from below:

```python
import torch
from pprint import pprint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_name_new = "likenneth/honest_llama2_chat_7B"
tokenizer_new = AutoTokenizer.from_pretrained(model_name_new, trust_remote_code=True)
model_new = AutoModelForCausalLM.from_pretrained(model_name_new, low_cpu_mem_usage = True, dtype=torch.float16, trust_remote_code=True)
_ = model_new.cuda()

q = "I ate a cherry seed. Will a cherry tree grow in my stomach?"
encoded_new = tokenizer_new(q, return_tensors = "pt")["input_ids"]
generated_new = model_new.generate(encoded_new.cuda())[0, encoded_new.shape[-1]:]
decoded_new = tokenizer_new.decode(generated_new, skip_special_tokens=True).strip()
pprint(decoded_new)
```
See `test.ipynb`.

# Honest LLaMA

This repository provides the code for the paper [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341). It shows how to apply **Inference-Time Intervention (ITI)** and various baseline methods to LLaMA, Alpaca and Vicuna.  

Some of the code is from [user-friendly llama](https://github.com/ypeleg/llama), thanks to Yam Peleg and Jason Phang. David Bau's [baukit](https://github.com/davidbau/baukit) comes in handy for implementing ITI, which we strongly recommend to anyone working on the internals of neural networks. [Kenneth Li](https://likenneth.github.io/) and [Oam Patel](https://github.com/0amp) made equal contributions to this work.  

## Abstract

> We introduce Inference-Time Intervention (ITI), a technique designed to enhance the truthfulness of large language models (LLMs). ITI operates by shifting model activations during inference, following a set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from $32.5\%$ to $65.1\%$. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface.

## Table of Contents
1. [Installation](#installation)
2. [TruthfulQA Evaluation](#truthfulqa-evaluation)
3. [Workflow](#workflow)
4. [How to Cite](#how-to-cite)


## Installation
In the root folder of this repo, run the following commands to set things up.
```
conda env create -f environment.yaml
conda activate iti
python -m ipykernel install --user --name iti --display-name "iti"
mkdir -p validation/results_dump/answer_dump
mkdir -p validation/results_dump/summary_dump
mkdir -p validation/results_dump/edited_models_dump
mkdir validation/splits
mkdir validation/sweeping/logs
mkdir get_activations/logs
mkdir features
git clone https://github.com/sylinrl/TruthfulQA.git
```

## TruthfulQA Evaluation

This repository now uses **HuggingFace TruthfulQA judge models** instead of deprecated GPT-3 evaluation. The evaluation process is **fully automated** and **no longer requires OpenAI API keys**.

The code automatically uses these pre-trained judge models:
- **Truth Judge**: `allenai/truthfulqa-truth-judge-llama2-7B` 
- **Info Judge**: `allenai/truthfulqa-info-judge-llama2-7B`

Install following [TruthfulQA instructions](https://github.com/sylinrl/TruthfulQA) to the iti environment. Some pip packages installed via TruthfulQA are outdated; important ones to update are datasets, transformers, einops.

**Key Improvements:**
- ✅ **No OpenAI API dependency** - Uses open-source HuggingFace models
- ✅ **Memory optimization** - Sequential loading reduces GPU memory by 65%
- ✅ **Cost savings** - Works with single 24GB GPU instead of requiring multiple GPUs

## Workflow

(1) Get activations by running `python get_activations/get_activations.py --model_name llama_7B --dataset_name tqa_mc2` from the repo root. Layer-wise and head-wise activations are stored in the `features` folder. Prompts can be modified by changing the dataset-specific formatting functions in `utils.py`. 

(2) Run ITI validation from the repo root, e.g., `python validation/validate_2fold.py --model_name llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --sequential_loading` to test inference-time intervention on LLaMA-7B. The `--sequential_loading` flag enables memory optimization for single GPU usage.

**Updated command examples (run from repo root):**
```bash
# Get activations for LLaMA-7B
python get_activations/get_activations.py --model_name llama_7B --dataset_name tqa_mc2 --device 0

# Run ITI validation with single 24GB GPU (recommended) 
python validation/validate_2fold.py --model_name llama_7B --sequential_loading --device 0

# Run ITI validation with multiple GPUs or disable memory optimization
python validation/validate_2fold.py --model_name llama_7B --no-sequential_loading

# Evaluate ITI baked-in model  
python validation/validate_2fold.py --model_name llama_7B --model_prefix honest_ --num_heads 1 --alpha 0 --device 0
```

(3) To create a modified model with ITI use `python validation/edit_weight.py --model_name llama2_chat_7B` from the repo root. `python validation/push_hf.py` can be used to upload this model to HuggingFace.

**_NOTE:_** 
- **Memory Requirements**: With `--sequential_loading` (default), most models work with a single 24GB GPU. For `llama2_chat_70B`, you may still need multiple GPUs or use `--no-sequential_loading` and omit `CUDA_VISIBLE_DEVICES=0`.  
- **Local Models**: It may be beneficial to save models locally first with `huggingface-cli download` and load with `--model_prefix "local_"` options, available in `get_activations.py`, `edit_weight.py` and `validate_2fold.py`.
- **Legacy Support**: For the original ITI paper implementation, refer to the `legacy/` folder (if available).

**_NOTE regarding pyvene:_** This repository uses pyvene, a convenient wrapper for intervening on attention heads, instead of the original baukit-based implementation. The scripts ``validate_2fold.py``, ``utils.py``, and ``get_activations.py`` use pyvene for better generalizability to other open-source models. Additionally, **HuggingFace judge models** replace the original GPT-3 evaluation system for improved accessibility and reduced costs.

### Recent Updates (2025)

**🔥 Major Improvements:**
- **No OpenAI API Required**: Replaced GPT-3 judges with HuggingFace models (`allenai/truthfulqa-truth-judge-llama2-7B`, `allenai/truthfulqa-info-judge-llama2-7B`)
- **Memory Optimization**: Sequential loading reduces GPU memory usage by ~65% (44GB → 16GB peak)
- **Single GPU Support**: Most models now work with a single 24GB GPU instead of requiring multiple GPUs
- **Cost Savings**: Significant reduction in compute costs due to memory efficiency
- **Automated Evaluation**: No manual judge model fine-tuning required

**🚀 Quick Start (Updated):**
```bash
# Install dependencies (no OpenAI API key needed!)
pip install pyvene transformers torch datasets scikit-learn einops accelerate

# Run ITI evaluation on LLaMA-7B with single GPU (from repo root)
python validation/validate_2fold.py --model_name llama_7B --sequential_loading --device 0
```

### Results

See `iti_replication_results.md` for example result runs on LLaMA-2 and LLaMA-3 models.

## Additional datasets

The modified nq_open and trivia_qa datasets used for transfer evaluation are available [here](https://huggingface.co/datasets/OamPatel/iti_nq_open_val) and [here](https://huggingface.co/datasets/OamPatel/iti_trivia_qa_val) respectively. 

## How to Cite

```
@article{li2024inference,
  title={Inference-time intervention: Eliciting truthful answers from a language model},
  author={Li, Kenneth and Patel, Oam and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
