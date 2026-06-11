#!/usr/bin/env python3

import logging

from xml.etree import ElementTree
from typing import Any, List
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    NodeType,
    Node,
    AttributeProto as ChakraAttr,
    COMP_NODE,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    GlobalMetadata
)


class MSCCLStep:
    def add_parent(
        self,
        parent: "MSCCLStep",
    ) -> None:
        parent_id = parent.comp_node.id if type(
            parent) is MSCCLReceiveReduceComputeStep else parent.node.id
        if parent_id not in self.node.data_deps:
            self.node.data_deps.append(parent_id)

    def encode_message(
        self,
        g
    ) -> None:
        encode_message(g, self.node)


class MSCCLCompStep(MSCCLStep):
    def __init__(
        self,
        tb_xml_node: ElementTree.Element,
        step,
        node_id: int,
        comp_data_size_bytes: int
    ) -> None:
        tb_id = tb_xml_node.attrib['id']
        step_id = int(step.attrib['s'])

        node = Node()
        node.id = node_id
        node.name = f"COMP_NODE_tb{tb_id}_step{step_id}"
        node.type = COMP_NODE
        # We do not fill in the compute duration because the data size is
        # resolved within the simulator, not here.
        self.node = node


class MSCCLSendStep(MSCCLStep):
    def __init__(self, tb_xml_node: ElementTree.Element, step, node_id: int, total_chunk_cnt: int, msg_chunk_cnt: int, msg_chunk_idx: int, num_npus: int) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.dst = int(tb_xml_node.attrib['send'])
        self.tag = int(tb_xml_node.attrib['chan'])
        step_id = int(step.attrib['s'])

        bytes_per_chunk = 1000 * 1000 // num_npus
        payload_size = bytes_per_chunk * msg_chunk_cnt

        node = Node()
        node.id = node_id
        node.name = f'COMM_SEND_NODE_tb{tb_id}_step{step_id}'
        node.type = COMM_SEND_NODE

        # FIX: Changed int64_val to int32_val
        node.attr.append(ChakraAttr(
            name="comm_type", int32_val=COMM_SEND_NODE))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=self.dst))
        node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.tag))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=payload_size))
        node.attr.append(ChakraAttr(
            name="message_size", uint64_val=payload_size))
        node.attr.append(ChakraAttr(
            name="tensor_size", uint64_val=payload_size))
        node.attr.append(ChakraAttr(
            name="msg_chunk_cnt", int32_val=msg_chunk_cnt))
        node.attr.append(ChakraAttr(
            name="msg_chunk_idx", int32_val=msg_chunk_idx))
        node.attr.append(ChakraAttr(
            name="total_chunk_cnt", int32_val=total_chunk_cnt))
        node.attr.append(ChakraAttr(name="local_time_step", int32_val=step_id))
        node.attr.append(ChakraAttr(name="chunk_offset",
                         int32_val=int(step.attrib['srcoff'])))
        node.attr.append(ChakraAttr(
            name="hasdep", int32_val=int(step.attrib['hasdep'])))
        node.attr.append(ChakraAttr(
            name="deps", int32_val=int(step.attrib['deps'])))
        node.attr.append(ChakraAttr(
            name="depid", int32_val=int(step.attrib['depid'])))
        node.attr.append(ChakraAttr(name="tb_id", int32_val=int(tb_id)))

        self.node = node


class MSCCLReceiveStep(MSCCLStep):
    def __init__(self, tb_xml_node: ElementTree.Element, step, node_id: int, total_chunk_cnt: int, msg_chunk_cnt: int, msg_chunk_idx: int, num_npus: int) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.src = int(tb_xml_node.attrib['recv'])
        self.tag = int(tb_xml_node.attrib['chan'])
        step_id = int(step.attrib['s'])

        bytes_per_chunk = 1000 * 1000 // num_npus
        payload_size = bytes_per_chunk * msg_chunk_cnt

        node = Node()
        node.id = node_id
        node.name = f'COMM_RECV_NODE_tb{tb_id}_step{step_id}'
        node.type = COMM_RECV_NODE

        # FIX: Changed int64_val to int32_val
        node.attr.append(ChakraAttr(
            name="comm_type", int32_val=COMM_RECV_NODE))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=self.src))
        node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.tag))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=payload_size))
        node.attr.append(ChakraAttr(
            name="message_size", uint64_val=payload_size))
        node.attr.append(ChakraAttr(
            name="tensor_size", uint64_val=payload_size))
        node.attr.append(ChakraAttr(
            name="msg_chunk_cnt", int32_val=msg_chunk_cnt))
        node.attr.append(ChakraAttr(
            name="msg_chunk_idx", int32_val=msg_chunk_idx))
        node.attr.append(ChakraAttr(
            name="total_chunk_cnt", int32_val=total_chunk_cnt))
        node.attr.append(ChakraAttr(name="local_time_step", int32_val=step_id))
        node.attr.append(ChakraAttr(name="chunk_offset",
                         int32_val=int(step.attrib['dstoff'])))
        node.attr.append(ChakraAttr(
            name="hasdep", int32_val=int(step.attrib['hasdep'])))
        node.attr.append(ChakraAttr(
            name="deps", int32_val=int(step.attrib['deps'])))
        node.attr.append(ChakraAttr(
            name="depid", int32_val=int(step.attrib['depid'])))
        node.attr.append(ChakraAttr(name="tb_id", int32_val=int(tb_id)))

        self.node = node


class MSCCLReceiveReduceComputeStep(MSCCLStep):
    def __init__(self, tb_xml_node: ElementTree.Element, step, recv_node_id: int, comp_node_id: int, total_chunk_cnt: int, msg_chunk_cnt: int, msg_chunk_idx: int, num_npus: int) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.src = int(tb_xml_node.attrib['recv'])
        self.tag = int(tb_xml_node.attrib['chan'])
        step_id = int(step.attrib['s'])

        bytes_per_chunk = 1000 * 1000 // num_npus
        payload_size = bytes_per_chunk * msg_chunk_cnt

        recv_node = Node()
        recv_node.id = recv_node_id
        recv_node.name = f'COMM_RECV_NODE_tb{tb_id}_step{step_id}'
        recv_node.type = COMM_RECV_NODE

        # FIX: Changed int64_val to int32_val
        recv_node.attr.append(ChakraAttr(
            name="comm_type", int32_val=COMM_RECV_NODE))
        recv_node.attr.append(ChakraAttr(name="comm_src", int32_val=self.src))
        recv_node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.tag))
        recv_node.attr.append(ChakraAttr(
            name="comm_size", int64_val=payload_size))
        recv_node.attr.append(ChakraAttr(
            name="message_size", uint64_val=payload_size))
        recv_node.attr.append(ChakraAttr(
            name="tensor_size", uint64_val=payload_size))
        recv_node.attr.append(ChakraAttr(
            name="msg_chunk_cnt", int32_val=msg_chunk_cnt))
        recv_node.attr.append(ChakraAttr(
            name="msg_chunk_idx", int32_val=msg_chunk_idx))
        recv_node.attr.append(ChakraAttr(
            name="total_chunk_cnt", int32_val=total_chunk_cnt))
        recv_node.attr.append(ChakraAttr(
            name="local_time_step", int32_val=step_id))
        recv_node.attr.append(ChakraAttr(
            name="chunk_offset", int32_val=int(step.attrib['dstoff'])))
        recv_node.attr.append(ChakraAttr(name="is_rrc", bool_val=True))
        recv_node.attr.append(ChakraAttr(
            name="hasdep", int32_val=int(step.attrib['hasdep'])))
        recv_node.attr.append(ChakraAttr(
            name="deps", int32_val=int(step.attrib['deps'])))
        recv_node.attr.append(ChakraAttr(
            name="depid", int32_val=int(step.attrib['depid'])))
        recv_node.attr.append(ChakraAttr(name="tb_id", int32_val=int(tb_id)))
        self.recv_node = recv_node

        comp_node = Node()
        comp_node.id = comp_node_id
        comp_node.name = f"COMP_NODE_tb{tb_id}_step{step_id}"
        comp_node.type = COMP_NODE
        comp_node.data_deps.append(recv_node.id)
        self.comp_node = comp_node

    def encode_message(
        self,
        g
    ) -> None:
        encode_message(g, self.recv_node)
        encode_message(g, self.comp_node)

    def add_parent(
        self,
        parent: "MSCCLStep",
    ) -> None:
        parent_id = parent.comp_node.id if type(
            parent) is MSCCLReceiveReduceComputeStep else parent.node.id
        if parent_id not in self.recv_node.data_deps:
            self.recv_node.data_deps.append(parent_id)


class MSCCLCopyStep(MSCCLStep):
    def __init__(self, tb_xml_node: ElementTree.Element, step, node_id: int) -> None:
        tb_id = tb_xml_node.attrib['id']
        step_id = int(step.attrib['s'])

        node = Node()
        node.id = node_id
        node.name = f"COPY_NODE_tb{tb_id}_step{step_id}"
        node.type = COMP_NODE
        self.node = node


class MSCCLReceiveCopySendStep(MSCCLStep):
    def __init__(self, tb_xml_node: ElementTree.Element, step, recv_node_id: int, send_node_id: int, total_chunk_cnt: int, msg_chunk_cnt: int, msg_chunk_idx: int, num_npus: int) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.src = int(tb_xml_node.attrib['recv'])
        self.dst = int(tb_xml_node.attrib['send'])
        self.tag = int(tb_xml_node.attrib['chan'])
        step_id = int(step.attrib['s'])

        bytes_per_chunk = 1000 * 1000 // num_npus
        payload_size = bytes_per_chunk * msg_chunk_cnt

        # 1. Create the Receive Node
        recv_node = Node()
        recv_node.id = recv_node_id
        recv_node.name = f'COMM_RECV_NODE_tb{tb_id}_step{step_id}'
        recv_node.type = COMM_RECV_NODE
        recv_node.attr.append(ChakraAttr(
            name="comm_type", int32_val=COMM_RECV_NODE))
        recv_node.attr.append(ChakraAttr(name="comm_src", int32_val=self.src))
        recv_node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.tag))
        recv_node.attr.append(ChakraAttr(
            name="comm_size", int64_val=payload_size))
        recv_node.attr.append(ChakraAttr(
            name="message_size", uint64_val=payload_size))
        recv_node.attr.append(ChakraAttr(
            name="tensor_size", uint64_val=payload_size))
        self.recv_node = recv_node

        # 2. Create the Send Node (Depends on the Receive Node finishing!)
        send_node = Node()
        send_node.id = send_node_id
        send_node.name = f'COMM_SEND_NODE_tb{tb_id}_step{step_id}'
        send_node.type = COMM_SEND_NODE
        send_node.data_deps.append(recv_node.id)
        send_node.attr.append(ChakraAttr(
            name="comm_type", int32_val=COMM_SEND_NODE))
        send_node.attr.append(ChakraAttr(name="comm_dst", int32_val=self.dst))
        send_node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.tag))
        send_node.attr.append(ChakraAttr(
            name="comm_size", int64_val=payload_size))
        send_node.attr.append(ChakraAttr(
            name="message_size", uint64_val=payload_size))
        send_node.attr.append(ChakraAttr(
            name="tensor_size", uint64_val=payload_size))
        self.send_node = send_node

    def encode_message(self, g) -> None:
        encode_message(g, self.recv_node)
        encode_message(g, self.send_node)

    def add_parent(self, parent: "MSCCLStep") -> None:
        parent_id = parent.comp_node.id if type(
            parent) is MSCCLReceiveReduceComputeStep else parent.node.id
        if parent_id not in self.recv_node.data_deps:
            self.recv_node.data_deps.append(parent_id)


class MSCCLNopStep(MSCCLStep):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        self.name = "NOP_Node"


class MSCCL2ChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_npus: int,
        logger: logging.Logger
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.logger = logger
        self.next_node_id = 0

    # Creates the global metadata info that is added to the start of all ET files.

    def create_global_metadata(self):
        input_text = ""
        with open(self.input_filename, "r") as input_file:
            input_text = input_file.read()
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=input_text),
            ChakraAttr(name="collective", string_val=self.collective)
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata

    # Creates an ET node, and assigns a node id to it.
    # Increment the node id, to be assigned to the next ET node.
    def get_et_node_id(self) -> int:
        id = self.next_node_id
        self.next_node_id += 1
        return id

    # This function is called to reset the node id when starting to add nodes for a new ET trace file.
    # There will be one ET trace file for each NPU.
    def reset_node_id(
        self
    ): self.next_node_id = 0

    # Add 'parent_node' as the parent to 'child_node'.
    # Note that parent_node and child_node has to be within the same ET trace file.
    def add_parent(
        self,
        child_node: Any,
        parent_node: Any
    ) -> None:
        child_node.data_deps.append(parent_node.id)

    def convert(self) -> None:
        node_map = {}
        step_map = {}
        tree = ElementTree.parse(self.input_filename)
        root = tree.getroot()

        collective = root.attrib['coll']
        if collective not in ["allreduce", "allgather", "alltoall", "reduce_scatter", "reduce", "broadcast"]:
            print(f"Error: Unsupported collective type {collective}")
            exit()
        self.collective = collective

        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            total_chunk_cnt = int(gpu.attrib['i_chunks'])
            if total_chunk_cnt == 0:
                total_chunk_cnt = int(gpu.attrib['o_chunks'])
            node_map[gpu_id] = {}
            step_map[gpu_id] = {}
            self.reset_node_id()
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                node_map[gpu_id][tb_id] = {}
                step_map[gpu_id][tb_id] = {}

                # THE FIX: Enumerate generates a unique key for every step,
                # guaranteeing the 'Receive' node is never overwritten by the 'Send'!
                for step_idx, step in enumerate(tb.findall('step')):
                    msg_chunk_cnt = int(step.attrib['cnt'])
                    src_off = int(step.attrib['srcoff'])
                    msg_chunk_idx = src_off
                    step_map[gpu_id][tb_id][step_idx] = step
                    et_node_id = self.get_et_node_id()

                    if step.attrib['type'] == "s":
                        node = MSCCLSendStep(
                            tb, step, et_node_id, total_chunk_cnt, msg_chunk_cnt, msg_chunk_idx, self.num_npus)
                    elif step.attrib['type'] == "r":
                        node = MSCCLReceiveStep(
                            tb, step, et_node_id, total_chunk_cnt, msg_chunk_cnt, msg_chunk_idx, self.num_npus)
                    elif step.attrib['type'] == "rrc":
                        comp_et_node_id = self.get_et_node_id()
                        node = MSCCLReceiveReduceComputeStep(
                            tb, step, et_node_id, comp_et_node_id, total_chunk_cnt, msg_chunk_cnt, msg_chunk_idx, self.num_npus)
                    elif step.attrib['type'] == "cpy":
                        node = MSCCLCopyStep(tb, step, et_node_id)
                    elif step.attrib['type'] == "rcs":
                        send_et_node_id = self.get_et_node_id()
                        node = MSCCLReceiveCopySendStep(
                            tb, step, et_node_id, send_et_node_id, total_chunk_cnt, msg_chunk_cnt, msg_chunk_idx, self.num_npus)
                    else:
                        node = MSCCLNopStep()

                    node_map[gpu_id][tb_id][step_idx] = node

        bytes_per_chunk = 1000 * 1000 // self.num_npus

        # --- NATIVE DAG DEPENDENCY RESOLUTION ---
        for gpu_id in node_map:
            for tb_id in node_map[gpu_id]:

                # 1. Strict Sequential intra-TB Dependencies (Program Order)
                # This guarantees that a Send will always wait for its preceding Receive!
                last_real_idx = -1
                sorted_idxs = sorted(node_map[gpu_id][tb_id].keys())
                for step_idx in sorted_idxs:
                    et_node = node_map[gpu_id][tb_id][step_idx]
                    if type(et_node) is not MSCCLNopStep:
                        if last_real_idx != -1:
                            et_node.add_parent(
                                node_map[gpu_id][tb_id][last_real_idx])
                        last_real_idx = step_idx

                # 2. Cross-TB Dependencies (Native MSCCL depid/deps)
                for step_idx, et_node in node_map[gpu_id][tb_id].items():
                    if type(et_node) is MSCCLNopStep:
                        continue

                    step = step_map[gpu_id][tb_id][step_idx]
                    dep_tb_id = int(step.attrib['depid'])
                    dep_s_attr = int(step.attrib['deps'])

                    if dep_tb_id != -1 and dep_tb_id in node_map[gpu_id]:
                        parent_node = None
                        # Find the target node in the other Threadblock
                        for d_idx in sorted(node_map[gpu_id][dep_tb_id].keys()):
                            d_node = node_map[gpu_id][dep_tb_id][d_idx]
                            d_step = step_map[gpu_id][dep_tb_id][d_idx]
                            if int(d_step.attrib['s']) == dep_s_attr:
                                if type(d_node) is not MSCCLNopStep:
                                    parent_node = d_node

                        if parent_node is not None:
                            et_node.add_parent(parent_node)

        # --- OUTPUT GENERATION ---
        for gpu_id in node_map:
            output_filename = "%s.%s.et" % (self.output_filename, gpu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.create_global_metadata()
                encode_message(g, global_metadata)
                for tb_id in node_map[gpu_id]:
                    for step_idx in sorted(node_map[gpu_id][tb_id].keys()):
                        et_node = node_map[gpu_id][tb_id][step_idx]
                        if type(et_node) is MSCCLNopStep:
                            continue

                        # Attribute Scrubbing
                        nodes_to_patch = []
                        if type(et_node) is MSCCLReceiveReduceComputeStep:
                            nodes_to_patch.extend(
                                [et_node.recv_node, et_node.comp_node])
                        elif type(et_node) is MSCCLReceiveCopySendStep:
                            nodes_to_patch.extend(
                                [et_node.recv_node, et_node.send_node])
                        elif type(et_node) is MSCCLCopyStep:
                            nodes_to_patch.append(et_node.node)
                        else:
                            nodes_to_patch.append(et_node.node)

                        for target_node in nodes_to_patch:
                            forbidden = ["comm_size", "message_size",
                                         "payload_size", "comm_type", "tensor_size"]
                            new_attrs = [
                                a for a in target_node.attr if a.name not in forbidden]
                            target_node.ClearField("attr")
                            target_node.attr.extend(new_attrs)

                            msg_chunk_cnt = 1
                            for a in new_attrs:
                                if a.name == "msg_chunk_cnt":
                                    msg_chunk_cnt = a.int32_val
                            payload_size = bytes_per_chunk * msg_chunk_cnt

                            target_node.attr.append(ChakraAttr(
                                name="comm_size", int64_val=payload_size))
                            is_comm = target_node.type in [3, 4, 6]
                            if is_comm:
                                target_node.attr.append(ChakraAttr(
                                    name="message_size", uint64_val=payload_size))
                                target_node.attr.append(ChakraAttr(
                                    name="tensor_size", uint64_val=payload_size))
                                is_send = any(
                                    a.name == "comm_dst" for a in target_node.attr)
                                target_node.type = 3 if is_send else 4
                                target_node.attr.append(ChakraAttr(
                                    name="comm_type", int32_val=target_node.type))

                        et_node.encode_message(g)
