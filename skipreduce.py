import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

class SkipReduce(AllReduce):
    def __init__(self, num_ranks, chunk_factor, s):
        super().__init__(num_ranks, chunk_factor, inplace=True)
        self.s = s

def create_skip_reduce(size, instances, s):
    # fully_connected ensures we don't hit topology link errors
    topology = fully_connected(size)
    collective = SkipReduce(size, size, s)

    # The program name should ideally match your workload's expected name
    with MSCCLProgram(f"skip_reduce_{size}_npus", topology, collective, instances):
        reduce_steps = (size - 1) - s

        for i in range(size):
            # Each chunk starts at rank i
            c = chunk(i, Buffer.input, i)
            
            # IMPORTANT: Isolation by Thread Block (TB)
            # We use the chunk index 'i' to give each chunk its own 'lane'.
            tb_id = i 

            # --- REDUCE-SCATTER PHASE ---
            for step in range(reduce_steps):
                next_rank = (i + step + 1) % size
                target = chunk(next_rank, Buffer.input, i)
                # Pass explicit TBs to ensure the analytical backend can track dependencies
                c = c.reduce(target, sendtb=tb_id, recvtb=tb_id)

            # --- ALL-GATHER PHASE ---
            for _ in range(size - 1):
                next_rank = (c.rank + 1) % size
                # Use a separate TB offset (tb_id + size) for the gather phase 
                # to prevent the simulator from creating a dependency loop 
                # between the two phases.
                c = c.copy(next_rank, sendtb=tb_id + size, recvtb=tb_id + size)

        XML()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_npus', type=int, help='Must match your workload config')
    parser.add_argument('instances', type=int, help='Usually 1 for custom collectives')
    parser.add_argument('--s', type=int, default=1, help='Steps to skip')
    args = parser.parse_args()
    
    if args.num_npus > 0:
        create_skip_reduce(args.num_npus, args.instances, args.s)