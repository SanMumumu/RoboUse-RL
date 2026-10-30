"""Deterministic global plans; Accelerate shards each slot once across ranks."""
import math
import random


def window_plan(lengths, epochs, accumulation, *, seed=0, world=8, bucket_size=128):
    order = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        indices = list(range(len(lengths)))
        rng.shuffle(indices)
        buckets = [
            indices[i : i + bucket_size] for i in range(0, len(indices), bucket_size)
        ]
        for bucket in buckets:
            bucket.sort(key=lambda i: lengths[i])
        rng.shuffle(buckets)
        flat = [i for b in buckets for i in b]
        multiple = world * accumulation
        flat += [None] * ((-len(flat)) % multiple)
        order.extend(flat)
    return order


def update_plan(size, epochs, world=8):
    if size < 1:
        raise ValueError("Empty update")
    one = list(range(size)) + [None] * ((-size) % world)
    return one * epochs, math.ceil(size / world)
