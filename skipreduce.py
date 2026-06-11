import argparse
from msccl.language import *
from msccl.topologies import Topology
from msccl.language.collectives import AllReduce


class SkipReduce(AllReduce):
    def __init__(self, num_ranks, chunk_factor, s):
        super().__init__(num_ranks, chunk_factor, inplace=False)
        self.s = s

# --- THE TOPOLOGY LOCK ---
# This explicitly defines a Ring via an Adjacency List.
# Because there are no cross-links, the MSCCL optimizer is physically
# forbidden from rewriting your sequential chains into 1-step broadcasts.
def custom_ring(size):
    links = [{} for _ in range(size)]
    for i in range(size):
        # 1. Add a self-link to bypass the MSCCL local-copy bug
        links[i][i] = 1.0

        # 2. Add the forward and backward ring links
        links[i][(i + 1) % size] = 1.0
        links[(i + 1) % size][i] = 1.0

    return Topology(f"custom_ring_{size}", links)


def create_skip_reduce(size, instances, s):
    topology = custom_ring(size)
    collective = SkipReduce(size, size, s)

    with MSCCLProgram(f"skip_reduce_{size}_npus", topology, collective, instances):
        reduce_steps = (size - 1) - s

        for i in range(size):
            c = chunk(i, Buffer.input, i)
            tb_id = i

            # --- REDUCE-SCATTER PHASE ---
            for step in range(reduce_steps):
                next_rank = (i + step + 1) % size
                target = chunk(next_rank, Buffer.input, i)

                c.reduce(target, sendtb=tb_id, recvtb=tb_id)

                # Advance the pointer to chain the dependency
                c = target

            # --- ALL-GATHER PHASE ---
            # Bridge to the output buffer natively
            c = c.copy(c.rank, buffer=Buffer.output,
                       index=i, sendtb=tb_id, recvtb=tb_id)

            for _ in range(size - 1):
                next_rank = (c.rank + 1) % size

                # copy() natively returns the destination chunk.
                # Reassigning it to 'c' chains the dependency forward.
                c = c.copy(next_rank, buffer=Buffer.output,
                           index=i, sendtb=tb_id, recvtb=tb_id)

        XML()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_npus', type=int,
                        help='Must match your workload config')
    parser.add_argument('instances', type=int,
                        help='Usually 1 for custom collectives')
    parser.add_argument('--s', type=int, default=1, help='Steps to skip')
    args = parser.parse_args()

    if args.num_npus > 0:
        create_skip_reduce(args.num_npus, args.instances, args.s)
