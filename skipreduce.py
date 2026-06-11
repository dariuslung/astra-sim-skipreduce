import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

class SkipReduce(AllReduce):
    def __init__(self, num_ranks, chunk_factor, s):
        # FIX: Set inplace=False to force the compiler to preserve Phase 2
        super().__init__(num_ranks, chunk_factor, inplace=False)
        self.s = s

def create_skip_reduce(size, instances, s):
    topology = fully_connected(size)
    collective = SkipReduce(size, size, s)

    with MSCCLProgram(f"skip_reduce_{size}_npus", topology, collective, instances):
        reduce_steps = (size - 1) - s

        for i in range(size):
            # Phase 1 starts in Buffer.input
            c = chunk(i, Buffer.input, i)
            
            # Keep the entire chunk lifecycle in a single lane
            tb_id = i 

            # --- REDUCE-SCATTER PHASE ---
            for step in range(reduce_steps):
                next_rank = (i + step + 1) % size
                target = chunk(next_rank, Buffer.input, i)
                c = c.reduce(target, sendtb=tb_id, recvtb=tb_id)

            # --- ALL-GATHER PHASE ---
            # 1. Bridge from input buffer to output buffer on the current rank
            c = c.copy(c.rank, buffer=Buffer.output, index=i, sendtb=tb_id, recvtb=tb_id)

            # 2. Execute Gather across the network into the output buffers
            for _ in range(size - 1):
                next_rank = (c.rank + 1) % size
                # Reusing the SAME tb_id perfectly chains Phase 2 sequentially after Phase 1!
                c = c.copy(next_rank, buffer=Buffer.output, index=i, sendtb=tb_id, recvtb=tb_id)

        XML()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_npus', type=int, help='Must match your workload config')
    parser.add_argument('instances', type=int, help='Usually 1 for custom collectives')
    parser.add_argument('--s', type=int, default=1, help='Steps to skip')
    args = parser.parse_args()
    
    if args.num_npus > 0:
        create_skip_reduce(args.num_npus, args.instances, args.s)