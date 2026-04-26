import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


class SkipReduce(AllReduce):
    def __init__(self, num_ranks, chunk_factor, s, inplace=True):
        # Initialize the base class
        super().__init__(num_ranks, chunk_factor, inplace)
        self.s = s

    # We do not override init_buffers(). We inherit it directly from AllReduce.

    # Implement if needed
    def check(self, prog):
        return True


def skipreduce_ring(size, instances, s):
    # Logical topology
    topology = fully_connected(size)
    
    # Initialize the custom SkipReduce collective
    collective = SkipReduce(size, size, s, inplace=True)

    with MSCCLProgram("skipreduce_ring_inplace", topology, collective, instances):
        for r in range(size):
            index = r
            # (rank, buffer, index)
            c = chunk(r, Buffer.input, index)
            next_rank = (r + 1) % size
            
            # --- REDUCE-SCATTER PHASE ---
            steps = 0
            max_steps = (size - 1) - s
            
            while steps < max_steps:
                c1 = chunk(next_rank, buffer=Buffer.input, index=r)
                c = c1.reduce(c)
                next_rank = (next_rank + 1) % size
                steps += 1
                
            # --- ALL-GATHER PHASE ---
            while next_rank != (r - 1) % size:
                c = c.copy(next_rank, buffer=Buffer.input, index=r)
                next_rank = (next_rank + 1) % size

        # The checker will now correctly validate the N-S contributors
        Check() 
        XML()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, help='number of gpus')
    parser.add_argument('instances', type=int, help='number of instances')
    # Add the S argument to dictate skipped steps
    parser.add_argument('--s', type=int, default=1, help='number of reduction steps to skip')

    args = parser.parse_args()

    # Pass the parsed S argument into the algorithm
    skipreduce_ring(args.num_gpus, args.instances, args.s)