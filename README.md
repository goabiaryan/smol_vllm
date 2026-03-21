# smol-vllm

[![PyPI version](https://img.shields.io/pypi/v/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Smol_vllm is a small paged-attention inference engine with paged KV cache, continuous batching, preemption.

## Why this project exists

smol-vLLM is **not a production engine** - it's a learning tool to deeply understand:

- **PagedAttention** — how block-based KV cache + ref counting avoids wasted memory
- **Continuous batching** — why it gives huge throughput gains (short jobs fill slots immediately)
- **Preemption & swapping** — handling memory pressure when blocks run low
- **Prefill vs decode** — the compute-bound → memory-bound transition (KV cache reads)

Start with the fake model (zero deps, instant), then flip to CausalLM to feel the difference.

## Metrics

With `enable_metrics=True` (default), each step prints vLLM/SGLang-style metrics to the console:

- **Latency**: prefill_ms, decode_ms
- **Throughput**: tok/s (tokens per second)
- **Counters**: prompt_tokens_total, generation_tokens_total
- **Server state**: running, waiting, swapped, kv_util%
- **Optional**: GPU memory (if CUDA), CPU % (if psutil installed: `pip install smol-vllm[metrics]`)

At the end of a run, `engine.metrics.print_summary()` shows TTFT, e2e latency, and averages.

## CausalLM (real models)

The `CausalLM` backend supports any **HuggingFace causal LM** via `model_name`: TinyLlama, Llama, Phi, Qwen, Gemma, Mistral, etc. Uses `AutoModelForCausalLM` under the hood.

## Install

```bash
pip install smol-vllm
```

For real CausalLM backend:

```bash
pip install smol-vllm[real]
```

Or: `pip install torch transformers accelerate`

## Usage

Default (fake model, token IDs):

```python
from smol_vllm import LLMEngine

engine = LLMEngine(num_gpu_blocks=64, block_size=16, max_batch_size=8)
for token in engine.generate([1, 2, 3, 4, 5], max_tokens=20):
    print(token, end=" ")
```

Real model (install with `pip install smol-vllm[real]`). Defaults to TinyLlama; pass `model_name` for any HuggingFace causal LM:

```python
engine = LLMEngine(use_real_model=True)
tokenizer = engine.model.tokenizer
tokens = tokenizer.encode("Hello!", add_special_tokens=False)
for token in engine.generate(tokens, max_tokens=20):
    print(tokenizer.decode([token]), end="")
```

## Demo

```bash
pip install smol-vllm
smol-vllm-demo
```

## License

MIT
