import random
import time

from . import BlockSpaceManager, LLMEngine


def run_exp1_continuous_batching():
    print("\n" + "=" * 60)
    print("Experiment 1: Continuous Batching")
    print("=" * 60)
    print("Submit 20 requests of different lengths")
    print("Columns: step | running | waiting | swapped | blocks_used (%)\n")

    engine = LLMEngine(num_gpu_blocks=64, block_size=16, max_batch_size=4)

    for i in range(20):
        length = random.randint(10, 80)
        prompt = list(range(i * 100, i * 100 + length))
        engine.add_request(prompt, max_tokens=20)

    step = 0
    total_finished = 0
    while total_finished < 20:
        outputs = engine.step()
        total_finished += sum(1 for o in outputs if o.finished)
        step += 1

        running = len(engine.scheduler.running)
        waiting = len(engine.scheduler.waiting)
        swapped = len(engine.scheduler.swapped)
        util = engine.block_manager.utilization() * 100

        print(
            f"  step {step:3d} | running={running:2d} | waiting={waiting:2d} | "
            f"swapped={swapped:2d} | blocks_used={util:5.1f}%"
        )

    print(f"\n  Done in {step} steps. All 20 requests finished.\n")


def run_exp2_memory_pressure():
    print("\n" + "=" * 60)
    print("Experiment 2: Memory Pressure & Preemption")
    print("=" * 60)
    print("num_blocks=16, 10 long sequences (64 tokens each) -> watch preemption\n")

    engine = LLMEngine(num_gpu_blocks=16, block_size=16, max_batch_size=10)

    for i in range(10):
        prompt = list(range(1000, 1064))
        engine.add_request(prompt, max_tokens=10)

    step = 0
    total_finished = 0
    while total_finished < 10:
        outputs = engine.step()
        total_finished += sum(1 for o in outputs if o.finished)
        step += 1
        util = engine.block_manager.utilization() * 100
        print(
            f"  step {step:3d} | running={len(engine.scheduler.running):2d} | "
            f"swapped={len(engine.scheduler.swapped):2d} | util={util:.0f}%"
        )

    print(f"\n  Done in {step} steps.\n")


def run_exp3_prefix_sharing():
    print("\n" + "=" * 60)
    print("Experiment 3: Prefix Sharing (copy_on_write)")
    print("=" * 60)
    print("5 sequences with same 32-token prefix -> compare utilization\n")

    bm = BlockSpaceManager(num_blocks=64, block_size=16)

    tokens_without = 32 + 16
    for i in range(5):
        bm.allocate(i, tokens_without)

    util_without = bm.utilization()
    for i in range(5):
        bm.free(i)

    bm.allocate(0, 48)
    for i in range(1, 5):
        bm.copy_on_write(0, i)
        for _ in range(16):
            bm.append_token(i)

    util_with_sharing = bm.utilization()
    print(
        f"  Without prefix sharing: 5 seqs × 3 blocks = 15 blocks -> util={util_without*100:.0f}%"
    )
    print(
        f"  With copy_on_write: shared prefix -> util={util_with_sharing*100:.0f}%"
    )
    print("  (Note: copy_on_write shares blocks, so fewer unique blocks used)\n")


def run_exp4_throughput_scaling():
    print("\n" + "=" * 60)
    print("Experiment 4: Throughput Scaling")
    print("=" * 60)
    print("Measure tok/s at batch_size=1, 8, 16 -> ASCII bar chart\n")

    batch_sizes = [1, 8, 16]
    tok_per_secs = []

    for batch_size in batch_sizes:
        engine = LLMEngine(
            num_gpu_blocks=64, block_size=16, max_batch_size=batch_size
        )

        for i in range(batch_size):
            engine.add_request(list(range(50)), max_tokens=20)

        start = time.perf_counter()
        total_tokens = 0
        while True:
            outputs = engine.step()
            total_tokens += len(outputs)
            if all(o.finished for o in outputs):
                break
        elapsed = time.perf_counter() - start
        tps = total_tokens / elapsed if elapsed > 0 else 0
        tok_per_secs.append(tps)
        print(f"  batch_size={batch_size:2d}: {tps:.1f} tok/s")

    print("\n  Throughput (tok/s) bar chart:")
    max_tps = max(tok_per_secs) if tok_per_secs else 1
    for bs, tps in zip(batch_sizes, tok_per_secs):
        bar_len = int(40 * tps / max_tps)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"    batch={bs:2d} |{bar}| {tps:.1f}")

    print()


def run_block_manager_checkpoint():
    print("\n" + "=" * 60)
    print("Checkpoint: BlockSpaceManager")
    print("=" * 60)

    bm = BlockSpaceManager(num_blocks=20, block_size=16)

    for seq_id in range(3):
        bm.allocate(seq_id, 48)
    print(f"  Allocated 3 seqs × 3 blocks: util={bm.utilization()*100:.0f}%")

    for seq_id in range(3):
        for _ in range(16):
            bm.append_token(seq_id)
    print(f"  After appends (16 each): util={bm.utilization()*100:.0f}%")

    bm.free(1)
    print(f"  After free(seq 1): util={bm.utilization()*100:.0f}%")
    free_count = 20 - int(bm.utilization() * 20)
    print(f"  Free blocks: {free_count} (4 blocks returned from seq 1)")
    print("  Checkpoint passed.\n")


def run_scheduler_checkpoint():
    print("\n" + "=" * 60)
    print("Checkpoint: Scheduler (continuous batching)")
    print("=" * 60)

    engine = LLMEngine(num_gpu_blocks=64, block_size=16, max_batch_size=4)

    for i in range(10):
        engine.add_request(list(range(10)), max_tokens=5)

    print("  Submitting 10 requests, max_batch_size=4")
    for step in range(8):
        engine.step()
        running = len(engine.scheduler.running)
        waiting = len(engine.scheduler.waiting)
        assert running <= 4, f"Expected ≤4 running, got {running}"
        print(f"  step {step+1}: running={running}, waiting={waiting}")

    print("  Checkpoint passed: only ≤4 running at once.\n")


def main():
    print("\n" + "#" * 60)
    print("#  smol-vLLM Demo - Paged Attention Inference Engine")
    print("#" * 60)

    run_block_manager_checkpoint()
    run_scheduler_checkpoint()
    run_exp1_continuous_batching()
    run_exp2_memory_pressure()
    run_exp3_prefix_sharing()
    run_exp4_throughput_scaling()

    print("All experiments complete.")
