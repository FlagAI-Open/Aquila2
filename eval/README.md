## Overview
For the evaluation of these datasets of the base model, we utilized [Eleuther Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)  for the assessment.
* MMLU (5-shot)
* arc-e (0-shot)
* HellaSwag (10-shot)
* TruthfulQA (0-shot)
* BoolQ (0-shot)
* winograde (0-shot)
* OpenBookQA (0-shot)
* PIQA (0-shot)


## Quickstart
### 1.  install 
install `lm-eval`from the github repository main branch, run:

```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
### 2. run eval

run MMLU (5-shot)
If you want to use a model that is already downloaded locally, just replace ```BAAI/Aquila2-7B``` with the path to the weights and it will work.
```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks hendrycksTest* \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```

run Aarc-e  (0-shot)

```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks arc_easy \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```

run HellaSwag (10-shot) 
```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks hellaswag \
    --no_cache \
    --num_fewshot 10 \
    --device cuda:0
```

run TruthfulQA (0-shot)
```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks truthfulqa_mc \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```

run BoolQ (0-shot)

```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks boolq \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```


run winograde (0-shot)

```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks winogrande \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```

run OpenBookQA (0-shot)

```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks openbookqa \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```

run  PIQA (0-shot)

```
python main.py \
    --model hf-causal-experimental \
    --model_args trust_remote_code=True,use_accelerate=True,pretrained=BAAI/Aquila2-7B \
    --tasks piqa \
    --no_cache \
    --num_fewshot 0 \
    --device cuda:0
```
