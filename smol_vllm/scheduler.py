import math
from collections import deque
from typing import Deque, List

from .block_manager import BlockSpaceManager
from .sequence import SchedulerOutputs, SequenceGroup, SequenceStatus


class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager, max_batch_size: int):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

    def schedule(self) -> SchedulerOutputs:
        blocks_to_swap_in: List = []
        blocks_to_swap_out: List = []

        while self.waiting and len(self.running) < self.max_batch_size:
            group = self.waiting[0]
            seq = group.sequences[0]
            num_tokens = seq.num_tokens
            num_blocks = math.ceil(num_tokens / self.block_manager.block_size)
            free_after = self.block_manager.num_free_blocks() - num_blocks
            running_after = len(self.running) + 1

            if not self.block_manager.can_allocate(num_tokens):
                break
            if free_after < running_after:
                break

            self.block_manager.allocate(group.group_id, num_tokens)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(group)

        while self.running and (
            len(self.running) > self.max_batch_size
            or self.block_manager.utilization() > 0.95
            or self.block_manager.num_free_blocks() < len(self.running)
        ):
            group = self.running.pop()
            seq = group.sequences[0]
            seq.status = SequenceStatus.SWAPPED
            self.block_manager.free(group.group_id)
            self.swapped.append(group)
            blocks_to_swap_out.append(group.group_id)
            print(f"  [preempt] group {group.group_id} -> swapped (blocks freed)")

        i = 0
        while i < len(self.swapped) and len(self.running) < self.max_batch_size:
            group = self.swapped[i]
            seq = group.sequences[0]
            num_tokens = seq.num_tokens
            num_blocks_needed = math.ceil(
                num_tokens / self.block_manager.block_size
            )
            free_after_swap = (
                self.block_manager.num_free_blocks()
                - num_blocks_needed
            )
            running_after_swap = len(self.running) + 1

            if (
                self.block_manager.can_allocate(num_tokens)
                and free_after_swap >= running_after_swap
            ):
                self.block_manager.allocate(group.group_id, num_tokens)
                seq.status = SequenceStatus.RUNNING
                self.running.append(group)
                self.swapped.pop(i)
                blocks_to_swap_in.append(group.group_id)
                print(f"  [swap-in] group {group.group_id} <- swapped")
            else:
                i += 1

        return SchedulerOutputs(
            scheduled_groups=self.running.copy(),
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
        )
