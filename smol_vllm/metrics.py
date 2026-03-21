import time
class Metrics:
    def __init__(self):
        self.step_count = 0
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.prefill_latencies: list[float] = []
        self.decode_latencies: list[float] = []
        self.e2e_latencies: list[float] = []
        self.ttft_latencies: list[float] = []
        self._request_start: dict[int, float] = {}
        self._request_first_token: dict[int, float] = {}
        self._last_token_time: dict[int, float] = {}

    def record_request_start(self, group_id: int):
        self._request_start[group_id] = time.perf_counter()

    def record_first_token(self, group_id: int):
        if group_id in self._request_start:
            self.ttft_latencies.append(time.perf_counter() - self._request_start[group_id])
            self._request_first_token[group_id] = time.perf_counter()

    def record_request_finish(self, group_id: int):
        if group_id in self._request_start:
            self.e2e_latencies.append(time.perf_counter() - self._request_start[group_id])
            del self._request_start[group_id]
        if group_id in self._request_first_token:
            del self._request_first_token[group_id]
        if group_id in self._last_token_time:
            del self._last_token_time[group_id]

    def record_inter_token(self, group_id: int) -> float | None:
        now = time.perf_counter()
        delta = None
        if group_id in self._last_token_time:
            delta = now - self._last_token_time[group_id]
        self._last_token_time[group_id] = now
        return delta

    def _gpu_stats(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                return f" gpu_mem={mem_alloc:.0f}/{mem_reserved:.0f}MB"
        except Exception:
            pass
        return ""

    def _cpu_stats(self) -> str:
        try:
            import psutil
            return f" cpu={psutil.cpu_percent():.0f}%"
        except Exception:
            pass
        return ""

    def print_step(
        self,
        step: int,
        prefill_ms: float,
        decode_ms: float,
        prefill_tokens: int,
        gen_tokens: int,
        running: int,
        waiting: int,
        swapped: int,
        block_util: float,
    ):
        self.step_count = step
        self.prompt_tokens_total += prefill_tokens
        self.generation_tokens_total += gen_tokens
        elapsed = (prefill_ms + decode_ms) / 1000
        tok_s = gen_tokens / elapsed if elapsed > 0 else 0

        line = (
            f"  [metrics] step={step} "
            f"prefill={prefill_ms:.0f}ms decode={decode_ms:.0f}ms "
            f"gen_tokens={gen_tokens} tok/s={tok_s:.0f} "
            f"running={running} waiting={waiting} swapped={swapped} "
            f"kv_util={block_util*100:.0f}%"
        )
        line += self._gpu_stats()
        line += self._cpu_stats()
        print(line)

    def print_summary(self):
        total_tokens = self.prompt_tokens_total + self.generation_tokens_total
        avg_ttft = sum(self.ttft_latencies) / len(self.ttft_latencies) * 1000 if self.ttft_latencies else 0
        avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies) * 1000 if self.e2e_latencies else 0

        print("\n  [metrics] summary:")
        print(f"    prompt_tokens_total={self.prompt_tokens_total} generation_tokens_total={self.generation_tokens_total}")
        print(f"    time_to_first_token_avg_ms={avg_ttft:.0f} e2e_request_latency_avg_ms={avg_e2e:.0f}")
        if self.prefill_latencies:
            print(f"    prefill_latency_avg_ms={sum(self.prefill_latencies)/len(self.prefill_latencies)*1000:.0f}")
        if self.decode_latencies:
            print(f"    decode_latency_avg_ms={sum(self.decode_latencies)/len(self.decode_latencies)*1000:.0f}")
        print()
