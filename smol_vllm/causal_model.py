import time
from typing import List

from .sequence import SequenceGroup


class CausalLM:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str | None = None,
        token: str | None = None,
    ):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[smol-vllm] Loading {model_name} on {device} ...")
        t0 = time.perf_counter()
        import os
        load_token = token or os.environ.get("HF_TOKEN")
        load_kw = {"token": load_token} if load_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            **load_kw,
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.kv_caches = {}
        print(f"[smol-vllm] Loaded in {time.perf_counter() - t0:.1f}s")

    def prefill(self, groups: List[SequenceGroup]) -> List[int]:
        if not groups:
            return []

        import torch

        prompts = [g.sequences[0].prompt_tokens for g in groups]
        max_len = max(len(p) for p in prompts)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        padded = []
        for p in prompts:
            padded.append([pad_id] * (max_len - len(p)) + p)

        input_ids = torch.tensor(padded, dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        attention_mask[input_ids == pad_id] = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        elapsed = time.perf_counter() - t0

        logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1).cpu().tolist()

        past = outputs.past_key_values
        for i, group in enumerate(groups):
            layer_caches = []
            for k, v in past:
                layer_caches.append((k[i : i + 1].clone(), v[i : i + 1].clone()))
            self.kv_caches[group.group_id] = tuple(layer_caches)

        avg_len = sum(len(p) for p in prompts) / len(prompts)
        print(
            f"[edu] Prefill batch={len(groups)} compute-bound "
            f"prompt_tokens≈{avg_len:.0f} {elapsed*1000:.0f}ms → {next_tokens}"
        )
        return next_tokens

    def decode(self, groups: List[SequenceGroup], block_tables: List[List[int]]) -> List[int]:
        _ = block_tables
        if not groups:
            return []

        import torch

        next_tokens = []
        t0 = time.perf_counter()
        for group in groups:
            last_token = group.sequences[0].output_tokens[-1]
            input_ids = torch.tensor([[last_token]], dtype=torch.long).to(self.device)
            past = self.kv_caches.get(group.group_id)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1).item()
            next_tokens.append(next_tok)

            layer_caches = []
            for k, v in outputs.past_key_values:
                layer_caches.append((k.clone(), v.clone()))
            self.kv_caches[group.group_id] = tuple(layer_caches)

        elapsed = time.perf_counter() - t0
        print(
            f"[edu] Decode batch={len(groups)} memory-bound "
            f"KV cache reads {elapsed*1000:.0f}ms → {next_tokens}"
        )
        return next_tokens

    def clear_cache(self, group_id: int):
        if group_id in self.kv_caches:
            del self.kv_caches[group_id]
