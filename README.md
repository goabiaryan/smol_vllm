# smol-vllm

[![PyPI version](https://img.shields.io/pypi/v/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small paged-attention inference engine with paged KV cache, continuous batching, preemption. 

Pure Python, no external dependencies.

## Install

```bash
pip install smol-vllm
```

Or from source:

```bash
git clone .... 
pip install .
```

## Usage

```python
from smol_vllm import LLMEngine

engine = LLMEngine(num_gpu_blocks=64, block_size=16, max_batch_size=8)

# Single request (streaming)
for token in engine.generate([1, 2, 3, 4, 5], max_tokens=20):
    print(token, end=" ")

# Batched: add requests and step
engine.add_request([10, 20, 30], max_tokens=10)
engine.add_request([40, 50, 60], max_tokens=10)
while True:
    outputs = engine.step()
    for out in outputs:
        print(out.output_tokens)
    if all(o.finished for o in outputs):
        break
```

## Demo

```bash
pip install smol-vllm
smol-vllm-demo
```

## License

MIT
