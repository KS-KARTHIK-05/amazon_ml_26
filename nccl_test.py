import os
import socket
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[Rank {rank}] Host: {socket.gethostname()}", flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)

    print(
        f"[Rank {rank}] GPU: {torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )

    # Each GPU starts with a different value
    tensor = torch.tensor(
        [float(rank + 1)],
        device=device
    )

    print(
        f"[Rank {rank}] Before all_reduce: {tensor.item()}",
        flush=True,
    )

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()

    print(
        f"[Rank {rank}] After all_reduce: {tensor.item()}",
        flush=True,
    )

    expected = world_size * (world_size + 1) / 2

    if tensor.item() == expected:
        print(
            f"[Rank {rank}] SUCCESS - NCCL communication works!",
            flush=True,
        )
    else:
        print(
            f"[Rank {rank}] FAILED - expected {expected}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
