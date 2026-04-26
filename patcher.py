import argparse
import os
import sys

# Append chakra root to path to resolve imports
sys.path.append("/app/chakra")

from schema.protobuf.et_def_pb2 import GlobalMetadata, Node, AttributeProto
from src.third_party.utils.protolib import decodeMessage as decode_message
from src.third_party.utils.protolib import encodeMessage as encode_message

def patch_et_files(file_prefix, num_gpus):
    # 1MB / 8 NPUs = 131072 bytes per chunk
    bytes_per_chunk = 131072 

    for i in range(num_gpus):
        filename = f"{file_prefix}.{i}.et"
        temp_filename = f"{file_prefix}.{i}.et.tmp"
        
        if not os.path.exists(filename):
            continue
            
        nodes = []
        with open(filename, "rb") as fin:
            # 1. Read Metadata
            meta = GlobalMetadata()
            decode_message(fin, meta)
            
            # 2. Read all nodes into a list to allow dependency linking
            while True:
                node = Node()
                try:
                    if not decode_message(fin, node):
                        break
                    nodes.append(node)
                except Exception:
                    break
        
        # 3. Process and Link Nodes
        with open(temp_filename, "wb") as fout:
            encode_message(fout, meta)
            
            # Group nodes by Thread Block (tb_id) to link them sequentially
            tb_last_node = {}

            for node in nodes:
                # --- CLEANUP AND BASIC PATCHING ---
                forbidden = ["comm_size", "message_size", "payload_size", "comm_type"]
                new_attrs = [a for a in node.attr if a.name not in forbidden]
                node.ClearField("attr")
                node.attr.extend(new_attrs)

                # Find metadata needed for logic
                tb_id = -1
                is_comm = node.type in [3, 4, 6]
                
                for attr in node.attr:
                    if attr.name == "tb_id":
                        tb_id = attr.int32_val

                # --- DEPENDENCY INJECTION ---
                # If this isn't the first node in this TB, make it depend on the previous one
                if tb_id != -1:
                    if tb_id in tb_last_node:
                        prev_node_id = tb_last_node[tb_id]
                        if prev_node_id not in node.data_deps:
                            node.data_deps.append(prev_node_id)
                    tb_last_node[tb_id] = node.id

                # --- ATTRIBUTE PATCHING ---
                node.attr.append(AttributeProto(name="comm_size", int32_val=num_gpus))
                
                if is_comm:
                    node.attr.append(AttributeProto(name="message_size", uint64_val=bytes_per_chunk))
                    node.attr.append(AttributeProto(name="payload_size", uint64_val=bytes_per_chunk))
                    
                    # Detect Send/Recv for proper type handling
                    is_send = any(a.name == "comm_dst" for a in node.attr)
                    if is_send:
                        node.type = 3
                    else:
                        node.type = 4
                    node.attr.append(AttributeProto(name="comm_type", int32_val=node.type))

                encode_message(fout, node)

        os.replace(temp_filename, filename)
        print(f"Sequentially linked and patched nodes in {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    patch_et_files(args.prefix, args.gpus)