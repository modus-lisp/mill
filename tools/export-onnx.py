#!/usr/bin/env python3
"""Convert an ONNX model into the two files mill reads: a graph and a blob.

    export-onnx.py model.onnx outdir/

Writes outdir/<stem>.graph (an s-expression, read by Lisp's READ) and
outdir/<stem>.bin (every tensor's bytes, concatenated, little-endian).

This exists so that the Lisp side never parses protobuf.  Protobuf parsing is a
solved problem that would be a large, dull, and entirely untestable-in-isolation
part of a TTS engine, and it only ever runs once per model — at build time, on a
workstation, not on the Pi.  Keeping it here also keeps the runtime's input
format something a person can read: the graph file is text, and every tensor in
the blob is named, so a wrong answer can be traced to a wrong weight by eye.

The graph file is deliberately close to the ONNX structure rather than an
optimized form.  mill executes the graph as the exporter emitted it, node for
node, which is what makes every node comparable against onnxruntime.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# ONNX TensorProto dtypes we know how to store.  Everything the runtime sees is
# one of these; anything else should stop the export rather than be guessed at.
DTYPES = {
    1: ("f32", np.float32),
    6: ("i32", np.int32),
    7: ("i64", np.int64),
    9: ("bool", np.bool_),
    11: ("f64", np.float64),
}


def lisp_atom(x):
    """A Python value as an s-expression token."""
    if isinstance(x, bool):
        return "t" if x else "nil"
    if isinstance(x, bytes):
        x = x.decode("utf-8", "replace")
    if isinstance(x, str):
        return '"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(x, float):
        # Lisp reads 1.0d0 as a double; plain 1.0 is single, and the attribute
        # values here (epsilons, alphas) are compared against double math.
        if x != x:
            return "nan"
        if x == float("inf"):
            return "inf"
        if x == float("-inf"):
            return "-inf"
        # A float whose repr already carries an exponent must use Lisp's DOUBLE marker
        # instead of gaining a second one: "9.99e-06" + "d0" is not a number to the Lisp
        # reader, it is a SYMBOL, and it fails much later as a type error inside whichever
        # op read the attribute.  Epsilons are exactly the values that repr this way.
        r = repr(float(x))
        if "e" in r or "E" in r:
            return r.replace("E", "d").replace("e", "d")
        return r + "d0"
    return str(x)


def lisp_list(xs):
    return "(" + " ".join(lisp_atom(x) for x in xs) + ")"


class Blob:
    """The concatenated tensor data, and where each tensor landed in it."""

    def __init__(self):
        self.parts = []
        self.offset = 0

    def add(self, arr):
        # Contiguous little-endian, which is what the Lisp reader assumes; the
        # Pi is little-endian too, so this never needs byte-swapping in practice
        # — but the dtype tag in the graph file says what it is, so a big-endian
        # reader could be written without re-exporting.
        arr = np.ascontiguousarray(arr)
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        raw = arr.tobytes()
        at = self.offset
        self.parts.append(raw)
        self.offset += len(raw)
        return at, len(raw)


def tensor_entry(blob, name, arr):
    dt = {
        np.dtype(np.float32): "f32",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(np.bool_): "bool",
        np.dtype(np.float64): "f64",
    }.get(arr.dtype)
    if dt is None:
        raise SystemExit(f"unsupported dtype {arr.dtype} for tensor {name}")
    at, n = blob.add(arr)
    return f'({lisp_atom(name)} {lisp_atom(dt)} {lisp_list(list(arr.shape))} {at} {n})'


def attr_value(a, blob, node_name):
    """One ONNX attribute as (name kind value...)."""
    t = a.type
    if t == onnx.AttributeProto.INT:
        return f"({lisp_atom(a.name)} :int {a.i})"
    if t == onnx.AttributeProto.FLOAT:
        return f"({lisp_atom(a.name)} :float {lisp_atom(float(a.f))})"
    if t == onnx.AttributeProto.STRING:
        return f"({lisp_atom(a.name)} :string {lisp_atom(a.s)})"
    if t == onnx.AttributeProto.INTS:
        return f"({lisp_atom(a.name)} :ints {lisp_list(list(a.ints))})"
    if t == onnx.AttributeProto.FLOATS:
        return f"({lisp_atom(a.name)} :floats {lisp_list([float(x) for x in a.floats])})"
    if t == onnx.AttributeProto.TENSOR:
        arr = numpy_helper.to_array(a.t)
        nm = f"{node_name}::{a.name}"
        return f"({lisp_atom(a.name)} :tensor {tensor_entry(blob, nm, arr)})"
    if t == onnx.AttributeProto.GRAPH:
        return f"({lisp_atom(a.name)} :graph {subgraph_entry(a.g, blob, node_name)})"
    raise SystemExit(f"unsupported attribute type {t} on {node_name}.{a.name}")


def node_entry(n, i, blob, prefix=""):
    """One node as (op name inputs outputs attrs), attributes included."""
    name = prefix + (n.name or f"{n.op_type}_{i}")
    attrs = " ".join(attr_value(a, blob, name) for a in n.attribute)
    return (f"({lisp_atom(n.op_type)} {lisp_atom(name)} "
            f"{lisp_list(list(n.input))} {lisp_list(list(n.output))} ({attrs}))")


def subgraph_entry(g, blob, owner):
    """A nested graph — the branches of an If, and eventually a Loop body.

    A subgraph reads its enclosing scope by name, so it needs no inputs of its
    own and none are emitted.  Loop and Scan do pass formal inputs, and rather
    than emit something a reader would have to guess at, this stops."""
    if len(g.input) > 0:
        raise SystemExit(f"subgraph {owner}.{g.name} declares inputs "
                         f"{[i.name for i in g.input]}; only the outer-scope form "
                         f"(If's branches) is supported")
    # NOT prefixed, unlike the node names below.  An initializer's name is what
    # the nodes inside this subgraph read it by, and a subgraph is given a scope
    # of its own, so there is nothing for a prefix to disambiguate against — it
    # would only make the binding unreachable.
    inits = [tensor_entry(blob, i.name, numpy_helper.to_array(i))
             for i in g.initializer]
    nodes = [node_entry(n, i, blob, prefix=f"{owner}/") for i, n in enumerate(g.node)]
    return ("(:initializers (" + " ".join(inits) + ")"
            " :nodes (" + " ".join(nodes) + ")"
            " :outputs " + lisp_list([o.name for o in g.output]) + ")")


def shape_of(vi):
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    return dims


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.name.replace(".onnx", "")

    model = onnx.load(str(src))
    g = model.graph
    blob = Blob()

    inits = [tensor_entry(blob, i.name, numpy_helper.to_array(i)) for i in g.initializer]

    nodes = [node_entry(n, i, blob) for i, n in enumerate(g.node)]

    # A value's declared dtype, where the exporter knows it.  The runtime infers
    # dtypes as it goes, but the graph inputs have to come from somewhere.
    def io_entry(vi):
        et = vi.type.tensor_type.elem_type
        dt = DTYPES.get(et, ("?", None))[0]
        return f"({lisp_atom(vi.name)} {lisp_atom(dt)} {lisp_list(shape_of(vi))})"

    graph_path = outdir / f"{stem}.graph"
    with open(graph_path, "w") as f:
        f.write(";;; Generated by tools/export-onnx.py — do not edit.\n")
        f.write(f";;; source: {src.name}\n")
        f.write("(:version 1\n")
        f.write(f" :producer {lisp_atom(model.producer_name)}\n")
        f.write(f" :opset {max((o.version for o in model.opset_import), default=0)}\n")
        # metadata_props is what the trainer wrote down about how to DRIVE the
        # model, as against how to evaluate it.  A streaming model's chunk
        # advance and a transducer's context size live nowhere else: the graph
        # says what one call computes and cannot say how many frames to step
        # between calls.  Carrying it costs a line, and the alternative is a
        # table of magic numbers on the Lisp side — a second copy to get wrong.
        f.write(" :metadata (" +
                " ".join(f"({lisp_atom(p.key)} {lisp_atom(p.value)})"
                         for p in model.metadata_props) + ")\n")
        f.write(" :inputs (" + " ".join(io_entry(v) for v in g.input) + ")\n")
        f.write(" :outputs (" + " ".join(io_entry(v) for v in g.output) + ")\n")
        f.write(" :initializers (\n  " + "\n  ".join(inits) + ")\n")
        f.write(" :nodes (\n  " + "\n  ".join(nodes) + "))\n")

    bin_path = outdir / f"{stem}.bin"
    with open(bin_path, "wb") as f:
        for p in blob.parts:
            f.write(p)

    # A config beside the model rides along when there is one — a Piper voice
    # keeps its phoneme id map and sample rate there, and whatever it holds
    # belongs with the weights it matches.
    cfg = Path(str(src) + ".json")
    if cfg.exists():
        (outdir / f"{stem}.config.json").write_text(cfg.read_text())

    print(f"{graph_path.name}: {len(g.node)} nodes, {len(inits)} initializers")
    print(f"{bin_path.name}: {blob.offset/1e6:.1f} MB")


if __name__ == "__main__":
    main()
