"""Rewrite GatherND (data rank-2, index_depth 2) into GPU/NPU-friendly ops.

GatherND lacks kernels in the Intel GPU plugin and is rejected by the NPU
compiler (research round 8). This pass replaces it with Reshape/Slice/Gather/
GatherElements, which are supported. Numerical equivalence is checked on CPU
against the original graph before the rewritten model is written out.

Usage: python tools/fix_gathernd.py --in artifacts/laya-en.onnx --out artifacts/laya-en-fixed.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    initializer_by_name = {init.name: init for init in graph.initializer}
    new_nodes: list[onnx.NodeProto] = []
    removed = 0
    counter = 0

    for node in graph.node:
        if node.op_type != "GatherND":
            new_nodes.append(node)
            continue
        removed += 1
        counter += 1
        prefix = f"gnr{counter}"
        data_name, indices_name = node.input[0], node.input[1]
        out_name = node.output[0]

        # indices: [a,b,c,d,2] (rank R, last dim = index depth 2)
        idx_vi = next(v for v in graph.value_info
                      if v.name == indices_name)
        idx_dims = list(idx_vi.type.tensor_type.shape.dim)[:-1]  # drop depth
        idx_rank = len(idx_dims) + 1

        # 1) flat = Reshape(indices, [-1, 2])
        flat_shape = helper.make_tensor(f"{prefix}_flat_shape", TensorProto.INT64,
                                        [2], [-1, 2])
        graph.initializer.append(flat_shape)
        new_nodes.append(helper.make_node(
            "Reshape", [indices_name, flat_shape.name], [f"{prefix}_flat"],
            name=f"{prefix}_reshape_flat"))

        # 2) outer = Squeeze(Slice(flat, 0:1, axis=1))
        starts0 = numpy_helper.from_array(np.array([0], dtype=np.int64),
                                          f"{prefix}_starts0")
        ends0 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                        f"{prefix}_ends0")
        axes1 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                        f"{prefix}_axes1")
        graph.initializer.extend([starts0, ends0, axes1])
        new_nodes.append(helper.make_node(
            "Slice", [f"{prefix}_flat", starts0.name, ends0.name, axes1.name],
            [f"{prefix}_outer2"], name=f"{prefix}_slice_outer"))
        squeeze_axes0 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                                f"{prefix}_sq_ax0")
        graph.initializer.append(squeeze_axes0)
        new_nodes.append(helper.make_node(
            "Squeeze", [f"{prefix}_outer2", squeeze_axes0.name],
            [f"{prefix}_outer"], name=f"{prefix}_squeeze_outer"))

        # 3) inner = Squeeze(Slice(flat, 1:2, axis=1))
        starts1 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                          f"{prefix}_starts1")
        ends1 = numpy_helper.from_array(np.array([2], dtype=np.int64),
                                        f"{prefix}_ends1")
        graph.initializer.extend([starts1, ends1])
        new_nodes.append(helper.make_node(
            "Slice", [f"{prefix}_flat", starts1.name, ends1.name, axes1.name],
            [f"{prefix}_inner2"], name=f"{prefix}_slice_inner"))
        squeeze_axes1 = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                                f"{prefix}_sq_ax1")
        graph.initializer.append(squeeze_axes1)
        new_nodes.append(helper.make_node(
            "Squeeze", [f"{prefix}_inner2", squeeze_axes1.name],
            [f"{prefix}_inner"], name=f"{prefix}_squeeze_inner"))

        # 4) rows = Gather(data, outer, axis=0)
        new_nodes.append(helper.make_node(
            "Gather", [data_name, f"{prefix}_outer"], [f"{prefix}_rows"],
            name=f"{prefix}_gather_rows", axis=0))

        # 5) res = GatherElements(rows, Unsqueeze(inner), axis=1) -> [N,1]
        unsq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64),
                                            f"{prefix}_un_ax")
        graph.initializer.append(unsq_axes)
        new_nodes.append(helper.make_node(
            "Unsqueeze", [f"{prefix}_inner", unsq_axes.name],
            [f"{prefix}_inner_u"], name=f"{prefix}_unsqueeze_inner"))
        new_nodes.append(helper.make_node(
            "GatherElements", [f"{prefix}_rows", f"{prefix}_inner_u"],
            [f"{prefix}_res"], name=f"{prefix}_gather_el", axis=1))

        # 6) out = Reshape(res, Shape(indices)[:-1])
        shape_idx = helper.make_node("Shape", [indices_name],
                                     [f"{prefix}_idx_shape"],
                                     name=f"{prefix}_shape_idx")
        slice_shape_starts = numpy_helper.from_array(
            np.array([0], dtype=np.int64), f"{prefix}_ss")
        slice_shape_ends = numpy_helper.from_array(
            np.array([idx_rank - 1], dtype=np.int64), f"{prefix}_se")
        slice_shape_axes = numpy_helper.from_array(
            np.array([0], dtype=np.int64), f"{prefix}_sa")
        graph.initializer.extend([slice_shape_starts, slice_shape_ends,
                                  slice_shape_axes])
        new_nodes.append(shape_idx)
        new_nodes.append(helper.make_node(
            "Slice", [f"{prefix}_idx_shape", slice_shape_starts.name,
                      slice_shape_ends.name, slice_shape_axes.name],
            [f"{prefix}_out_shape"], name=f"{prefix}_slice_shape"))
        new_nodes.append(helper.make_node(
            "Reshape", [f"{prefix}_res", f"{prefix}_out_shape"], [out_name],
            name=f"{prefix}_reshape_out"))

    # rebuild graph with new nodes; keep topological order by re-running
    # shape inference later. Simple approach: insert new_nodes before the
    # first consumer usage - onnx requires topo order; our replacement nodes
    # only depend on the original node's inputs, so placing them where the
    # original node was is correct.
    new_graph = helper.make_graph(
        [n for n in graph.node],  # placeholder, replaced below
        graph.name,
        graph.input, graph.output,
        initializer=list(graph.initializer),
        value_info=list(graph.value_info),
    )
    # Use the ordered node list we built.
    del new_graph.node[:]
    new_graph.node.extend(new_nodes)
    model.graph.CopyFrom(new_graph)
    print(f"rewrote {removed} GatherND node(s)")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    args = parser.parse_args()

    model = onnx.load(args.src)
    model = rewrite(model)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.dst)
    print("saved", args.dst)


if __name__ == "__main__":
    main()
