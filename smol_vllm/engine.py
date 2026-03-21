import time
from typing import Dict, List

from .block_manager import BlockSpaceManager
from .metrics import Metrics
from .model import FakeModel
from .scheduler import Scheduler
from .sequence import RequestOutput, Sequence, SequenceGroup, SequenceStatus


class LLMEngine:
    def __init__(
        self,
        num_gpu_blocks: int = 64,
        block_size: int = 16,
        max_batch_size: int = 8,
        use_real_model: bool = False,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        enable_metrics: bool = True,
    ):
        self.block_manager = BlockSpaceManager(num_gpu_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager, max_batch_size)
        if use_real_model:
            from .causal_model import CausalLM
            self.model = CausalLM(model_name=model_name)
        else:
            self.model = FakeModel()
        self.request_counter = 0
        self.groups: Dict[int, SequenceGroup] = {}
        self.enable_metrics = enable_metrics
        self.metrics = Metrics()
        self._step_count = 0

    def add_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 50,
        temperature: float = 1.0,
        stop_token_ids: List[int] | None = None,
    ):
        if stop_token_ids is None:
            stop_token_ids = [0]

        group_id = self.request_counter
        self.request_counter += 1
        seq = Sequence(group_id, prompt_tokens.copy())
        group = SequenceGroup(
            group_id=group_id,
            sequences=[seq],
            sampling_params={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop_token_ids": stop_token_ids,
            },
        )
        self.groups[group_id] = group
        self.scheduler.waiting.append(group)
        if self.enable_metrics:
            self.metrics.record_request_start(group_id, prompt_len=len(prompt_tokens))

    def step(self) -> List[RequestOutput]:
        step_start = time.perf_counter()
        sched_out = self.scheduler.schedule()

        prefill_groups = [
            g for g in sched_out.scheduled_groups
            if not g.sequences[0].output_tokens
        ]
        decode_groups = [
            g for g in sched_out.scheduled_groups
            if g.sequences[0].output_tokens
        ]

        block_tables = [
            self.block_manager.get_block_table(g.group_id)
            for g in sched_out.scheduled_groups
        ]

        prefill_ms = 0.0
        decode_ms = 0.0

        next_tokens: List[int] = []
        if prefill_groups:
            t0 = time.perf_counter()
            next_tokens = self.model.prefill(prefill_groups)
            prefill_ms = (time.perf_counter() - t0) * 1000
            if self.enable_metrics:
                self.metrics.prefill_latencies.append(prefill_ms / 1000)
                for g in prefill_groups:
                    self.metrics.record_first_token(g.group_id)
        if decode_groups:
            t0 = time.perf_counter()
            decode_block_tables = block_tables[len(prefill_groups) :]
            next_tokens_decode = self.model.decode(decode_groups, decode_block_tables)
            next_tokens += next_tokens_decode
            decode_ms = (time.perf_counter() - t0) * 1000
            if self.enable_metrics:
                self.metrics.decode_latencies.append(decode_ms / 1000)
                for g in decode_groups:
                    self.metrics.record_inter_token(g.group_id)

        prefill_tokens = sum(len(g.sequences[0].prompt_tokens) for g in prefill_groups)
        gen_tokens = len(sched_out.scheduled_groups)

        outputs = []
        for i, group in enumerate(sched_out.scheduled_groups):
            token = next_tokens[i]
            seq = group.sequences[0]
            seq.output_tokens.append(token)

            self.block_manager.append_token(group.group_id)

            finished = (
                token in group.sampling_params["stop_token_ids"]
                or len(seq.output_tokens) >= group.sampling_params["max_tokens"]
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                if self.enable_metrics:
                    self.metrics.record_request_finish(group.group_id)
                self.block_manager.free(group.group_id)
                if hasattr(self.model, "clear_cache"):
                    self.model.clear_cache(group.group_id)
                if group in self.scheduler.running:
                    self.scheduler.running.remove(group)

            outputs.append(
                RequestOutput(
                    group_id=group.group_id,
                    seq_id=seq.seq_id,
                    output_tokens=seq.output_tokens.copy(),
                    finished=finished,
                )
            )

        if self.enable_metrics and (prefill_ms > 0 or decode_ms > 0):
            self._step_count += 1
            self.metrics.print_step(
                step=self._step_count,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                prefill_tokens=prefill_tokens,
                gen_tokens=gen_tokens,
                running=len(self.scheduler.running),
                waiting=len(self.scheduler.waiting),
                swapped=len(self.scheduler.swapped),
                block_util=self.block_manager.utilization(),
            )

        return outputs

    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 50,
        temperature: float = 1.0,
        stop_token_ids: List[int] | None = None,
        **kwargs,
    ):
        if stop_token_ids is None:
            stop_token_ids = [0]
        self.add_request(
            prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token_ids=stop_token_ids,
            **kwargs,
        )
        group_id = self.request_counter - 1
        while True:
            outputs = self.step()
            for out in outputs:
                if out.group_id == group_id:
                    if out.output_tokens:
                        yield out.output_tokens[-1]
                    if out.finished:
                        return
            group = self.groups.get(group_id)
            if group and group.sequences[0].is_finished:
                return
