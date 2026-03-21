# smol-vllm

[![PyPI version](https://img.shields.io/pypi/v/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small paged-attention inference engine with paged KV cache, continuous batching, and preemption. 

## Workflow

**1. Start with FakeModel (default)** — zero deps, instant. Use raw token IDs.

**2. Switch to CausalLM** — add `use_real_model=True` and install `smol-vllm[real]`. Use the tokenizer to encode text.

Same API for both, only the engine constructor changes:

| | Test first | Real inference |
|--|------------|----------------|
| Engine | `LLMEngine()` | `LLMEngine(use_real_model=True)` |
| Input | token IDs, e.g. `[1, 2, 3, 4, 5]` | `tokenizer.encode("Hello!")` |

## Why this project exists

smol-vLLM is a **learning tool**, not a production engine. It focuses on:

- **PagedAttention** — block-based KV cache and ref counting
- **Continuous batching** — throughput gains as short jobs finish
- **Preemption & swapping** — handling memory pressure
- **Prefill vs decode** — compute-bound → memory-bound transition

Start with FakeModel, then switch to CausalLM to compare timing and memory. It is fully Python based and requires no CUDA understanding, allowing anyone to learn basic inferencing without the additional complexity. 

## Install

```bash
pip install smol-vllm
```

For CausalLM (real models):

```bash
pip install smol-vllm[real]
```

## Usage

**Step 1: FakeModel** (no extra install)

```python
from smol_vllm import LLMEngine

engine = LLMEngine()
for token in engine.generate([1, 2, 3, 4, 5], max_tokens=20):
    print(token, end=" ")
```

**Step 2: CausalLM** — set `use_real_model=True` and use the tokenizer:

```python
engine = LLMEngine(use_real_model=True)
tokenizer = engine.model.tokenizer
tokens = tokenizer.encode("Hello!", add_special_tokens=False)
for token in engine.generate(tokens, max_tokens=20):
    print(tokenizer.decode([token]), end="")
```

Other models: `LLMEngine(use_real_model=True, model_name="Qwen/Qwen2-0.5B-Instruct")`

## Metrics

With `enable_metrics=True` (default), each step prints latency, throughput (tok/s), KV util, and optional GPU/CPU stats. 

At the end, `engine.metrics.print_summary()` and logs go to `logs/smol_vllm_*.csv`.

## Demo

```bash
pip install smol-vllm
smol-vllm-demo
```

## License

MIT
