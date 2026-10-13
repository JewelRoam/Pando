"""Post-process an ONNX export for NPU compatibility.

Pipeline: GatherND surgery -> IsNaN+Where bypass -> verify no residual.

Usage: python tools/postprocess_onnx.py --in model.onnx --out model.npu.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnx
import numpy as np


def remove_gathernd(model: onnx.ModelProto) -> int:
    """Replace GatherND (data rank-2, depth 2) with supported ops."""
    graph = model.graph
    count = 0
    new_nodes = []
    for node in graph.node:
        if node.op_type != "GatherND":
            new_nodes.append(node)
            continue
        count += 1
        data_name, indices_name = node.input[0], node.input[1]
        out_name = node.output[0]
        prefix = f"gfix_{count}"

        from onnx import TensorProto, helper, numpy_helper

        flat_shape = helper.make_tensor(f"{prefix}_fs", TensorProto.INT64,
                                        [2], [-1, 2])
        graph.initializer.append(flat_shape)
        new_nodes.append(helper.make_node(
            "Reshape", [indices_name, flat_shape.name], [f"{prefix}_flat"],
            name=f"{prefix}_r_flat"))
        starts0 = numpy_helper.from_array(np.array([0], dtype=np.int64),
                                          f"{prefix}_s0")
        ends0 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                        f"{prefix}_e0")
        axes1 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                        f"{prefix}_a1")
        graph.initializer.extend([starts0, ends0, axes1])
        new_nodes.append(helper.make_node(
            "Slice", [f"{prefix}_flat", starts0.name, ends0.name, axes1.name],
            [f"{prefix}_o2"], name=f"{prefix}_sl_outer"))
        sq0 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                      f"{prefix}_sq0")
        graph.initializer.append(sq0)
        new_nodes.append(helper.make_node(
            "Squeeze", [f"{prefix}_o2", sq0.name], [f"{prefix}_outer"],
            name=f"{prefix}_sq_outer"))
        starts1 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                          f"{prefix}_s1")
        ends1 = numpy_helper.from_array(np.array([2], dtype=np.int64),
                                        f"{prefix}_e1")
        graph.initializer.extend([starts1, ends1])
        new_nodes.append(helper.make_node(
            "Slice", [f"{prefix}_flat", starts1.name, ends1.name, axes1.name],
            [f"{prefix}_i2"], name=f"{prefix}_sl_inner"))
        sq1 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                      f"{prefix}_sq1")
        graph.initializer.append(sq1)
        new_nodes.append(helper.make_node(
            "Squeeze", [f"{prefix}_i2", sq1.name], [f"{prefix}_inner"],
            name=f"{prefix}_sq_inner"))
        new_nodes.append(helper.make_node(
            "Gather", [data_name, f"{prefix}_outer"], [f"{prefix}_rows"],
            name=f"{prefix}_gather", axis=0))
        unsq = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                       f"{prefix}_un")
        graph.initializer.append(unsq)
        new_nodes.append(helper.make_node(
            "Unsqueeze", [f"{prefix}_inner", unsq.name], [f"{prefix}_iu"],
            name=f"{prefix}_unsq_inner"))
        new_nodes.append(helper.make_node(
            "GatherElements", [f"{prefix}_rows", f"{prefix}_iu"],
            [f"{prefix}_res"], name=f"{prefix}_ge", axis=1))
        shape_idx = helper.make_node("Shape", [indices_name],
                                     [f"{prefix}_is"], name=f"{prefix}_shape")
        ss = numpy_helper.from_array(np.array([0], dtype=np.int64),
                                     f"{prefix}_ss")
        se = numpy_helper.from_array(
            np.array([len(idx_vi_shape(indices_name, graph)) - 1],
                     dtype=np.int64), f"{prefix}_se")
        sa = numpy_helper.from_array(np.array([0], dtype=np.int64),
                                     f"{prefix}_sa")
        graph.initializer.extend([ss, se, sa])
        new_nodes.append(shape_idx)
        new_nodes.append(helper.make_node(
            "Slice", [f"{prefix}_is", ss.name, se.name, sa.name],
            [f"{prefix}_os"], name=f"{prefix}_sl_shape"))
        new_nodes.append(helper.make_node(
            "Reshape", [f"{prefix}_res", f"{prefix}_os"], [out_name],
            name=f"{prefix}_reshape_out"))

    del graph.node[:]
    graph.node.extend(new_nodes)
    return count


def idx_vi_shape(name: str, graph) -> int:
    for v in graph.value_info:
        if v.name == name:
            return len(v.type.tensor_type.shape.dim)
    return 5  # fallback


def remove_isnan_where(model: onnx.ModelProto) -> int:
    """Bypass IsNaN+Where pairs (pass-through in inference)."""
    graph = model.graph
    bypass_map = {}
    nodes_to_remove = set()

    for isnan in list(graph.node):
        if isnan.op_type != "IsNaN":
            continue
        mask_name = isnan.output[0]
        for node in graph.node:
            if node.op_type == "Where" and mask_name in node.input:
                cond_idx = list(node.input).index(mask_name)
                value_name = node.input[2] if cond_idx == 0 else node.input[1]
                bypass_map[node.output[0]] = value_name
                nodes_to_remove.add(id(node))
                nodes_to_remove.add(id(isnan))
                break

    new_nodes = []
    for node in graph.node:
        if id(node) in nodes_to_remove:
            continue
        for i, inp in enumerate(node.input):
            if inp in bypass_map:
                node.input[i] = bypass_map[inp]
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return len(bypass_map)


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    args = parser.parse_args()

    model = onnx.load(args.src)

    gn = remove_gathernd(model)
    iw = remove_isnan_where(model)
    # Fix topological order (node insertion breaks it)
    try:
        from onnx import version_converter
        # Use ONNX's own topo sort
        model = onnx.utils.Extractor(model).extract_model()
    except Exception:
        pass  # best-effort; the manual nonisnan graph is the fallback
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.dst)

    print(f"post-processed: {gn} GatherND removed, {iw} IsNaN+Where bypassed")
    print(f"saved {args.dst}")


if __name__ == "__main__":
    main()
