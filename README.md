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

**61 ops, and the thing they run is checked against onnxruntime node by node.**

mill was extracted from [chord](https://github.com/modus-lisp/chord), which runs
a Piper VITS voice — 2755 nodes, 50 op types, 15.65M parameters. It is now also
what [stave](https://github.com/modus-lisp/stave) runs a streaming Zipformer
transducer on: three graphs, 9441 nodes between them, and every op they use is
implemented. All three now match onnxruntime node for node — the encoder on all
9418 of its values, the decoder on 20, the joiner on 3.

All 2761 of the voice graph's comparable intermediate values match
onnxruntime within `1e-4 + 1e-4*|ref|`, and its finished waveform differs by at most 5.3e-5. That
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

## Control flow

An ONNX graph is usually straight-line, but not always: where a tracer met a
branch it could not fold away, it emitted an `If` whose two arms are whole graphs
sitting in the node's attributes. Those arms declare no inputs. They read
whatever they did not compute themselves straight out of the scope containing the
`If`, by name — so the runner carries a chain of scopes, a branch runs with a
fresh one pushed, and everything it computed but did not return goes away with
it.

Two things that were invisible on a straight-line graph become bugs the moment
that exists, and both were real here rather than hypothetical:

*A value's last reader may be inside a branch*, several nodes after the last
plain node that mentioned it. Count only the plain readers and the runner retires
the value — hands its buffer to the pool — before the branch that needs it runs.
That is exactly what happened the first time the zipformer decoder ran.

*A `Constant` hands back one of the model's own attributes*, and an attribute
outlives the run while the value bound to it is an ordinary intermediate the pool
may recycle. Return the attribute itself and the first run quietly rewrites the
model for the second. `Constant` returns a copy.

## Gates

`inspect/subgraph-gate.lisp` covers both of those, plus `Tile`. It builds graphs
by hand rather than loading one, because a model only exercises the branch its
own shapes select — the zipformer decoder's `else` arm is unreachable — and
because the two hazards need a graph arranged to provoke them. Each check was
confirmed to fail with its fix removed; a gate that cannot fail is decoration.

`inspect/conv-gate.lisp` covers the convolution kernel: 13703 cases of Conv and
ConvTranspose against the definition, over strides, dilations, padding and kernel
widths no single model uses, compared **exactly** rather than within a tolerance.
Every optimization in that kernel claims to leave the summation order alone, so a
tolerance would hide the only kind of mistake worth catching. Two-dimensional
Conv is swept the same way and by the same reference, which is what makes that
exactness available at all: a 2-D convolution is a sum of 1-D ones, so it reuses
`conv1d-line` unchanged and accumulates in one order across both ranks.

`inspect/gemm-gate.lisp` covers the transpose-and-bias fold — 1260 cases over
both transposes, alpha and beta, and every shape C is allowed to broadcast from.
**Every case is non-square, and that is the whole design of the sweep.** A
transpose written with the wrong extent is still a transpose when the matrix is
square: it indexes the same elements in the same order, and every square test
passes. mill shipped exactly that bug. It survived the zipformer's decoder, whose
projection is 512x512, and died on its joiner's 500x512 — twenty logits out of
place, which is a different word. With the bug back in, 770 of the 1260 cases
fail.

`inspect/gather-gate.lisp` does the same for the strided copy — 20000 random
walks against the definition, also exactly, since a copy does no arithmetic and
one differing element is a bug. All of these need only what is in this
repository.

The per-node gate is not here, because it needs a model, and mill does not have
one. It lives with whoever does: see chord's `inspect/node-gate.lisp` for the
shape of it, and `tools/dump-fixtures.py` below for the half that is generic.

    sbcl --dynamic-space-size 4096 --load inspect/conv-gate.lisp     ; 13703 cases
    sbcl --dynamic-space-size 4096 --load inspect/gather-gate.lisp   ; 20000 walks
    sbcl --load inspect/gemm-gate.lisp                               ; 1260 non-square
    sbcl --load inspect/subgraph-gate.lisp                           ; scopes, lifetimes

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
    src/ops-nn.lisp     Conv, ConvTranspose, Gemm, the normalizations, activations
    src/ops-control.lisp If — the ops that run a graph instead of arithmetic
    src/npy.lisp        read numpy's .npy (system :mill/npy, for gates only)

    tools/export-onnx.py    ONNX -> .graph (s-expressions) + .bin (weights)
    tools/dump-fixtures.py  onnxruntime -> a golden value for every node

Tensors are a flat typed vector plus a shape, with **no strides field**: reshape
is free, transpose is an explicit copy, and every kernel is a linear-index loop.
Broadcasting is stride-0 on the stretched axes.

Protobuf parsing lives in Python because it runs once per model, at build time,
on a workstation — never on the small machine this is aimed at. What ships is a
text graph a person can read and a blob of little-endian floats.

The export carries the model's `metadata_props` through as well, reachable as
`(mill:model-meta model "decode_chunk_len")`. That is not decoration. A graph
describes one call and cannot say how the calls follow one another, so a
streaming model's chunk advance and a transducer's context size live there and
nowhere else in the file; the alternative is a table of magic numbers on the
Lisp side, which is a second copy of the truth to get wrong.

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
