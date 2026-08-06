# mill

An ONNX graph interpreter in pure Common Lisp. No FFI, no C, no foreign
libraries — the arithmetic runs in Lisp.

mill loads a graph exported by `tools/export-onnx.py` and runs it node for node,
in the order it was exported. There is no scheduler and no fusion pass, and that
is the point: the graph mill runs is the graph onnxruntime ran, so every
intermediate value has a golden counterpart and a wrong answer names the exact
node that first diverged.

    (asdf:load-system :mill)
    (let* ((m (mill:load-model "model.graph"))
           (vals (mill:run-model m (list (cons "input" some-tensor)))))
      (mill:model-output-tensor m vals))

MIT licensed.

## State

**57 ops, and the thing they run is checked against onnxruntime node by node.**

mill was extracted from [chord](https://github.com/modus-lisp/chord), which runs
a Piper VITS voice — 2755 nodes, 50 op types, 15.65M parameters. All 2761 of that
graph's comparable intermediate values match onnxruntime within
`1e-4 + 1e-4*|ref|`, and its finished waveform differs by at most 5.3e-5. That
gate is what every optimization below had to survive, and it reports the *same*
worst error after each one as before it — that invariance is the proof summation
order survived.

Speed, on chord's voice: it started at 0.051x realtime and now runs at 2.70x on
one thread and 6.57x on sixteen, on a busy 116-core EPYC. What that took, roughly
in order of what it was worth:

*Typed kernels.* The ops dispatch on dtype once and then call a kernel written
against a concrete element type. `DIM` narrowed to 30 bits is what keeps a
composed offset a fixnum — that alone took MatMul from 22% of the run to 0.8%.

*SIMD.* `src/simd.lisp` is **seven operations and a lane count wide** —
`f32v-ref`, `f32v-broadcast`, the four arithmetic operations, `f32v-done` — and
every kernel is written in those, so a port is those seven plus a constant. The
bar for admitting an operation to that list is that it must be a single NEON
instruction (`vld1q`/`vdupq`/`vaddq`/`vsubq`/`vmulq`/`vdivq`): nothing goes in
that a port would have to emulate. Measured against the same build with the
vector unit off, one thread went 0.78x → 2.70x, sixteen 3.91x → 6.57x, and
sixty-four 4.87x → 6.91x.

> ⚠️ **The AVX–SSE transition penalty is the whole story on x86.** SBCL compiles
> ordinary single-float code to *legacy, non-VEX* SSE. A VEX-256 instruction
> leaves upper YMM state live, and then every legacy SSE instruction takes a
> merge assist — ~115–125 ns, flat in the length, and not confined to function
> boundaries: a scalar `aref` of a float array inside a loop pays it per
> iteration. `VZEROUPPER`, which is what `f32v-done` is, costs a few cycles and
> removes all of it. It must sit *inside* `do-vectorized`. Hoisting it out of a
> hot caller looks like a saving and made ConvTranspose 10–20x slower than
> scalar.

There is deliberately **no fused multiply-add**, even though the CPU has it. FMA
is both faster and more accurate, and more accurate is the problem: it changes
the order the terms are added in, which breaks `inspect/conv-gate.lisp`'s exact
compare and moves the answer away from onnxruntime, which has no FMA either.
Lanes go across the output, never across the kernel taps.

*The fallback arm is the scalar reference.* Setting `+f32-lanes+` to 1 turns
every operation into its scalar self — it is the same source with the width
turned down, not a second implementation that could drift. It produces chord's
205568 samples byte for byte identically to the AVX2 build, which is the
invariant a NEON port will rest on.

*Convolution*, which is over half of chord's run: drop the `i * stride` multiply
and unroll by 4, then fuse all KW kernel taps into one pass so the output row is
read and written once per input channel instead of KW times (0.75 → 3.15 G MAC/s
on one core). ConvTranspose is the odd one, because a strided store is not one
store: it de-interleaves the output row by residue class first, since a given
kernel position only ever reaches outputs sharing one residue mod the stride, so
writing those contiguously turns each pass into the same stride-1 loop the Convs
use. Same terms, same order, permuted addresses — 6x on the op.

*The strided copy.* Slice, Transpose, Expand and Split are one loop between
them, and it was a fifth of a single-threaded run — which turned out to be mostly
a misreading of the work. `collapse-gather-axes` rewrites the walk with the
fewest axes that describe it: an axis of extent 1 is a step of no size, and two
adjacent axes where the outer's stride is exactly the inner's span visit one
contiguous ascending block. A slice that only cuts an outer axis then collapses
to a single stride-1 axis, which is a block move. 0.158 s to 0.018 s.

*The tensor pool: GC was free, the allocator was not.* A run consed ~240 MB with
`*gc-run-time*` at ~0 — collecting a generation that survives nothing costs
nothing. The cost was at the other end: multi-MB intermediates are large objects,
so each is fresh pages to map and zero. `run-model` returns a dead value's data
vector to the pool instead of the collector, which took consing to 53 MB. The
subtle part is that a value can be dead under one name and alive under another,
because reshape and the identity-shaped ops give one data vector a second name —
so the count is per data vector, by `EQ`, not per value.

*Threads.* `src/parallel.lisp` is a fixed worker pool splitting convolution rows.
Float traps are per-thread, so worker loops mask them themselves.

Two of those are worth more the more threads you have, and for the same reason:
the strided copy and the pool both removed *serial* work, so they are worth +32%
and +9% at sixteen threads against +7% and +4% at one.

## Gates

`inspect/conv-gate.lisp` covers the convolution kernel: 11342 cases of Conv and
ConvTranspose against the definition, over strides, dilations, padding and kernel
widths no single model uses, compared **exactly** rather than within a tolerance.
Every optimization in that kernel claims to leave the summation order alone, so a
tolerance would hide the only kind of mistake worth catching.

`inspect/gather-gate.lisp` does the same for the strided copy — 20000 random
walks against the definition, also exactly, since a copy does no arithmetic and
one differing element is a bug. Both of these need only what is in this
repository.

The per-node gate is not here, because it needs a model, and mill does not have
one. It lives with whoever does: see chord's `inspect/node-gate.lisp` for the
shape of it, and `tools/dump-fixtures.py` below for the half that is generic.

    sbcl --dynamic-space-size 4096 --load inspect/conv-gate.lisp    ; 11342 cases
    sbcl --dynamic-space-size 4096 --load inspect/gather-gate.lisp  ; 20000 walks

## Layout

    src/tensor.lisp     flat typed vector + shape; broadcasting via zero strides
    src/simd.lisp       one vector width, named once (AVX2 today, NEON next;
                        lanes = 1 is the fallback, and the scalar reference)
    src/parallel.lisp   a fixed worker pool, for the kernels that split
    src/model.lisp      reads the exported graph and weight blob
    src/graph.lisp      the interpreter: run the nodes in order, drop dead values
    src/ops-shape.lisp  Reshape, Transpose, Slice, Concat, Expand, Split, Pad
    src/ops-math.lisp   elementwise and matmul
    src/ops-reduce.lisp the reductions and the softmaxes
    src/ops-index.lisp  Gather, GatherND, Scatter, TopK, the index ops
    src/ops-nn.lisp     Conv, ConvTranspose, the normalizations, the activations
    src/npy.lisp        read numpy's .npy (system :mill/npy, for gates only)

    tools/export-onnx.py    ONNX -> .graph (s-expressions) + .bin (weights)
    tools/dump-fixtures.py  onnxruntime -> a golden value for every node

Tensors are a flat typed vector plus a shape, with **no strides field**: reshape
is free, transpose is an explicit copy, and every kernel is a linear-index loop.
Broadcasting is stride-0 on the stretched axes.

Protobuf parsing lives in Python because it runs once per model, at build time,
on a workstation — never on the small machine this is aimed at. What ships is a
text graph a person can read and a blob of little-endian floats.

`dump-fixtures.py` takes the model and an `.npz` of feeds, promotes every node
output to a graph output, and dumps one `.npy` per value plus a manifest. What to
feed is the caller's business, because it is the one thing the script cannot
know — and the feed has to be one the model answers deterministically, which for
anything that samples noise means neutralizing that on the way in. The script
asserts determinism rather than assuming it: two runs must agree bit for bit, or
the fixtures are not fixtures.

## Why an interpreter

The alternative is to reimplement a model from its paper. That is smaller and
faster, and it has exactly one test: does the output look right. An arithmetic
mistake in the middle of a 2755-node graph does not look wrong, it looks slightly
off, and there is no way to bisect it.

Running the graph as exported means every node has a reference answer. When
something is wrong, the gate names the node.
