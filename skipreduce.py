import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


class SkipReduce(AllReduce):
    def __init__(self, num_ranks, chunk_factor, s):
        # AllReduce handles the necessary self.buffers initialization
        super().__init__(num_ranks, chunk_factor, inplace=True)
        self.s = s


def create_skip_reduce(size, s):
    # fully_connected avoids "No link" AssertionErrors
    topology = fully_connected(size)
    collective = SkipReduce(size, size, s)

    with MSCCLProgram("skip_reduce_simple", topology, collective, instances=1):
        # Calculate reduction steps (Total ranks - 1 - skipped steps)
        reduce_steps = (size - 1) - s

        for i in range(size):
            # Start with the chunk at its home rank 'i'
            c = chunk(i, Buffer.input, i)

            # --- REDUCE-SCATTER PHASE ---
            # In this version, reduce() requires a ChunkRef (target)
            for step in range(reduce_steps):
                next_rank = (i + step + 1) % size
                target = chunk(next_rank, Buffer.input, i)
                c = c.reduce(target)

            # --- ALL-GATHER PHASE ---
            # In this version, copy() requires an integer (next_rank)
            for _ in range(size - 1):
                next_rank = (c.rank + 1) % size
                # The copy() call automatically uses the Ref's current buffer/index
                c = c.copy(next_rank)

        XML()


if __name__ == "__main__":
    # Example: 4 GPUs, skipping 1 reduction step
    # This will generate the XML trace to stdout
    create_skip_reduce(4, 1)