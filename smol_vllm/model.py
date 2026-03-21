import random
import time
from typing import List

from .sequence import SequenceGroup


class FakeModel:
    def prefill(self, groups: List[SequenceGroup]) -> List[int]:
        total_prompt = sum(g.sequences[0].num_tokens for g in groups)
        time.sleep(0.01 * total_prompt / 100)
        return [self._fake_next_token(g) for g in groups]

    def decode(
        self, groups: List[SequenceGroup], block_tables: List[List[int]]
    ) -> List[int]:
        _ = block_tables
        time.sleep(0.005 * len(groups))
        return [self._fake_next_token(g) for g in groups]

    def _fake_next_token(self, group: SequenceGroup) -> int:
        seq = group.sequences[0]
        stop_ids = group.sampling_params.get("stop_token_ids", [0])
        if random.random() < 0.10:
            return stop_ids[0]
        return (seq.num_tokens * 7 + 13) % 1000 + 1
