import argparse
import os
import sys

# Append chakra root to path to resolve imports
sys.path.append("/app/chakra")

from schema.protobuf.et_def_pb2 import GlobalMetadata, Node, AttributeProto
from src.third_party.utils.protolib import decodeMessage as decode_message
from src.third_party.utils.protolib import encodeMessage as encode_message


def patch_et_files(file_prefix, num_gpus, comm_size):
    for i in range(num_gpus):
        filename = f"{file_prefix}.{i}.et"
        temp_filename = f"{file_prefix}.{i}.et.tmp"
        
        with open(filename, "rb") as fin, open(temp_filename, "wb") as fout:
            # 1. Copy GlobalMetadata
            meta = GlobalMetadata()
            decode_message(fin, meta)
            encode_message(fout, meta)
            
            # 2. Read and Patch Node 0
            node = Node()
            decode_message(fin, node)
            
            if node.id == 0:
                # Inject the missing comm_size attribute
                attr = AttributeProto(name="comm_size", uint64_val=comm_size)
                node.attr.append(attr)
            
            encode_message(fout, node)
            
            # 3. Copy the remaining execution trace nodes
            while True:
                next_node = Node()
                try:
                    if not decode_message(fin, next_node):
                        break
                    encode_message(fout, next_node)
                except Exception:
                    break
        
        # Overwrite the original file with the patched one
        os.replace(temp_filename, filename)
        print(f"Successfully patched node 0 in {filename} with comm_size={comm_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Chakra ET files with comm_size")
    parser.add_argument('--prefix', type=str, required=True, help="Prefix of the ET files")
    parser.add_argument('--gpus', type=int, default=8, help="Number of GPUs")
    parser.add_argument('--size', type=int, default=1048576, help="Collective size in bytes")
    
    args = parser.parse_args()
    patch_et_files(args.prefix, args.gpus, args.size)