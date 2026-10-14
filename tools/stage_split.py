"""Split one ONNX graph into two pipeline stages at a node boundary.

Stage A keeps ``nodes[:cut]`` and exposes the tensors that stage B consumes
as its outputs (the interface). Stage B keeps ``nodes[cut:]``, takes those
interface tensors as inputs, and keeps the original graph outputs.
Initializers stay external and are shared, so splitting does not duplicate
the weights.

This is the graph-surgery half of the Round 93 pipeline experiment; it does
not run anything. See tools/stage_pipeline_check.py for the measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto, helper


def _value_info(name: str, source: dict) -> onnx.ValueInfoProto:
    proto = onnx.ValueInfoProto()
    known = source.get(name)
    if known is not None:
        proto.CopyFrom(known)
    proto.name = name
    if not proto.type.HasField("tensor_type") or \
            proto.type.tensor_type.elem_type == TensorProto.UNDEFINED:
        proto.type.tensor_type.elem_type = TensorProto.FLOAT
    return proto


def _used_initializers(nodes, initializer_names: set) -> set:
    used = set()

    def scan(node_list) -> None:
        for node in node_list:
            for name in node.input:
                if name in initializer_names:
                    used.add(name)
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    scan(attribute.g.node)
                elif attribute.type == AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        scan(subgraph.node)

    scan(nodes)
    return used


def split_graph(src: Path, out_dir: Path, cut: int,
                names: tuple = ("stage-a.onnx", "stage-b.onnx")) -> dict:
    model = onnx.load(str(src), load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    if not 0 < cut < len(nodes):
        raise ValueError(f"cut {cut} outside (0, {len(nodes)})")

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        info = {value.name: value for value in inferred.graph.value_info}
    except Exception:  # shape inference is best-effort
        info = {}
    info.update({value.name: value for value in graph.input})
    info.update({value.name: value for value in graph.output})

    head, tail = nodes[:cut], nodes[cut:]
    produced = {name for node in head for name in node.output if name}
    tail_consumed = {name for node in tail for name in node.input if name}
    interface = sorted(produced & tail_consumed)
    if not interface:
        raise ValueError(f"cut {cut} has an empty interface")

    initializer_names = {initializer.name for initializer in graph.initializer}
    by_name = {initializer.name: initializer
               for initializer in graph.initializer}
    head_inputs = {name for node in head for name in node.input if name}
    tail_inputs = {name for node in tail for name in node.input if name}

    def build(node_list, inputs, outputs, name):
        used = _used_initializers(node_list, initializer_names)
        return helper.make_graph(
            node_list, name, inputs, outputs,
            [by_name[init] for init in sorted(used)])

    a_inputs = [value for value in graph.input if value.name in head_inputs]
    b_inputs = ([_value_info(name, info) for name in interface]
                + [value for value in graph.input
                   if value.name in tail_inputs])
    a_outputs = [_value_info(name, info) for name in interface]

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    specs = ((head, a_inputs, a_outputs, names[0]),
             (tail, b_inputs, list(graph.output), names[1]))
    for node_list, inputs, outputs, name in specs:
        stage = helper.make_model(
            build(node_list, inputs, outputs, name),
            opset_imports=list(model.opset_import),
            producer_name="pandosi-stage-split")
        stage.ir_version = model.ir_version
        target = out_dir / name
        # save_as_external_data=False keeps the original external references,
        # so both stages read the same weight blob as the source graph.
        onnx.save_model(stage, str(target), save_as_external_data=False)
        paths.append(target)

    return {
        "source": str(src),
        "cut": cut,
        "nodes_a": len(head),
        "nodes_b": len(tail),
        "interface": interface,
        "stage_a": str(paths[0]),
        "stage_b": str(paths[1]),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src",
                        default=str(root / "artifacts" / "laya-en.onnx"))
    parser.add_argument("--out-dir", default=None,
                        help="defaults to the source directory (the shared "
                             "external weight blob lives there)")
    parser.add_argument("--cut", type=int, default=900)
    args = parser.parse_args()
    src = Path(args.src)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    print(json.dumps(split_graph(src, out_dir, args.cut), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
