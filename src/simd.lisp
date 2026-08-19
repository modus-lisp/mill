;;;; simd.lisp — one vector width, named once, so a kernel is written once.
;;;;
;;;; The convolution is 70% of a synthesis and its inner loop was running at
;;;; about half of what scalar floating point can do on one core, which is where
;;;; scalar code stops.  The rest of the machine is a vector unit.
;;;;
;;;; This file is deliberately narrow.  Everything below is spelled in terms of
;;;; F32V-REF, F32V-BROADCAST, F32V-BROADCAST-REF, the four arithmetic
;;;; operations, F32V-DONE and +F32-LANES+, and a port is those eight plus a
;;;; constant — which is the whole point, because the machine this engine is
;;;; aimed at is a Raspberry Pi running modus rather than SBCL, and its vector
;;;; unit is NEON.  A kernel written against these compiles unchanged there; only
;;;; this file gets a third arm.  Every one of them is a single NEON instruction
;;;; (vld1q/vdupq/vld1q_dup/vaddq/vsubq/vmulq/vdivq), which is the test a
;;;; candidate for this list has to pass: nothing goes in here that a port would
;;;; have to emulate.
;;;;
;;;; F32V-DONE is the one that is not arithmetic, and it is not decoration.  SBCL
;;;; compiles ordinary single-float code to legacy, non-VEX SSE.  On x86 a VEX
;;;; 256-bit instruction leaves the upper half of the vector registers live, and
;;;; every legacy SSE instruction executed while that is true takes a merge
;;;; assist — in the scalar remainder below, and in the caller, and in the
;;;; caller's caller.  Measured here it is ~120 ns per call that touches a vector
;;;; at all, flat in the length: a 38656-element row does not notice and a
;;;; 62-element row is four times slower than the scalar loop it replaced, which
;;;; is exactly the shape of the first, wrong, version of this.  VZEROUPPER
;;;; before returning to scalar code costs a few cycles and removes all of it.
;;;; NEON has no split register file and no such state, so the port's F32V-DONE
;;;; is empty — but the call has to be in the kernel for x86 to be correct.
;;;;
;;;; F32V-BROADCAST-REF exists for the same reason and is the more expensive
;;;; lesson.  Reading one float out of an array is legacy SSE — SBCL compiles
;;;; (AREF A I) on a single-float array to MOVSS — so a loop that reads a scalar
;;;; and broadcasts it pays the transition on every pass, and if the scalar reads
;;;; are interleaved with the vector arithmetic it pays it several times per pass.
;;;; Measured on this machine that is ~115 ns each, and it is not a rounding
;;;; error: the same matmul kernel written with (F32V-BROADCAST (AREF DA I))
;;;; inside the loop ran at 2185 us against 44 us for one that kept the
;;;; multiplier in a register — fifty times.  F32V-BROADCAST-REF loads and
;;;; splats in one VEX instruction and never leaves the vector unit, and it is
;;;; what makes an accumulator-in-registers kernel possible at all.  It costs a
;;;; port nothing: on NEON it is vld1q_dup_f32, one instruction, and the whole
;;;; distinction it is drawing does not exist there.
;;;;
;;;; The rule the two of them add up to: inside a vector loop there is no such
;;;; thing as a cheap scalar float.  Not a load, not a comparison — ZEROP on a
;;;; single-float is a legacy COMISS and costs the same transition.  Either keep
;;;; the scalars out of the loop entirely, or group them so the loop pays one
;;;; transition instead of four.
;;;;
;;;; The fallback arm sets +F32-LANES+ to 1 and every operation to its scalar
;;;; self, so a build with no vector unit at all compiles the same source and
;;;; gets the same answers.  That is what keeps the SIMD path honest: the scalar
;;;; loop is not a separate implementation that could drift, it is this one with
;;;; the width turned down.
;;;;
;;;; NOT fused multiply-add, on purpose.  This CPU has FMA and sb-simd exposes
;;;; it, and an FMA would be both faster and more accurate — it keeps the product
;;;; at full width instead of rounding it before the add.  More accurate is the
;;;; problem: every optimization in the convolution claims to leave the
;;;; arithmetic alone, inspect/conv-gate.lisp compares exactly on the strength of
;;;; that claim, and inspect/node-gate.lisp measures the distance to onnxruntime,
;;;; which is not using FMA either.  A separate multiply and add is bit-identical
;;;; to the scalar loop; that is worth more here than the last few percent.

(in-package #:mill)

(eval-when (:compile-toplevel :load-toplevel :execute)
  ;; sb-simd ships with SBCL but is not loaded by default, and it defines the
  ;; AVX2 operations whether or not the CPU has them — so ask, rather than
  ;; assume.  A failure here is not an error: it means the scalar arm.
  #+x86-64
  (when (ignore-errors
          (require :sb-simd)
          (let ((available (find-symbol "INSTRUCTION-SET-AVAILABLE-P" "SB-SIMD-INTERNALS"))
                (find (find-symbol "FIND-INSTRUCTION-SET" "SB-SIMD-INTERNALS")))
            (and available find (funcall available (funcall find :avx2)))))
    (pushnew :mill-avx2 *features*)))

;;; F32V-PACK is the eighth name, and it was not optional: a kernel that carries a
;;; vector in a variable across loop iterations — an accumulator — needs the
;;; compiler to know the variable holds a vector, or SBCL boxes it on the heap
;;; every time round.  Undeclared, MATMUL-CORE's register accumulators ran fifty
;;; times slower than the memory-bound loop they replaced, which is what heap
;;; allocation in an inner loop looks like.  A port names its own vector type
;;; here (float32x4_t on NEON) and the kernels are unchanged.  It is
;;; F32V-PACK rather than F32V because ops-math.lisp already calls a float32
;;; ARRAY an F32V, and one of those two names had to be the longer one.

#+mill-avx2
(progn
  (defconstant +f32-lanes+ 8)
  (deftype f32v-pack () 'sb-simd-avx2:f32.8)

  (defmacro f32v-ref (a i)
    "Lanes of single floats starting at index I of A.  SETF-able.  The access is
unaligned: a convolution reads overlapping windows, so demanding alignment would
mean a copy per tap."
    `(sb-simd-avx2:f32.8-aref ,a ,i))

  (defmacro f32v-broadcast (x) `(sb-simd-avx2:f32.8 ,x))

  (defmacro f32v-broadcast-ref (a i)
    "Element I of A in every lane, without leaving the vector unit.  SB-SIMD's
F32-AREF is the VEX-encoded scalar load; ordinary AREF is not, and the
difference is a transition penalty per iteration."
    `(sb-simd-avx2:f32.8-broadcast (sb-simd-avx2:f32-aref ,a ,i)))

  (defmacro f32v+ (a b) `(sb-simd-avx2:f32.8+ ,a ,b))
  (defmacro f32v- (a b) `(sb-simd-avx2:f32.8- ,a ,b))
  (defmacro f32v* (a b) `(sb-simd-avx2:f32.8* ,a ,b))
  (defmacro f32v/ (a b) `(sb-simd-avx2:f32.8/ ,a ,b))
  (defmacro f32v-done () `(sb-simd-avx2:vzeroupper)))

#-mill-avx2
(progn
  (defconstant +f32-lanes+ 1)
  (deftype f32v-pack () 'single-float)

  (defmacro f32v-ref (a i) `(aref (the (simple-array single-float (*)) ,a) ,i))
  (defmacro f32v-broadcast (x) `(the single-float ,x))
  (defmacro f32v-broadcast-ref (a i)
    `(aref (the (simple-array single-float (*)) ,a) ,i))
  (defmacro f32v+ (a b) `(+ ,a ,b))
  (defmacro f32v- (a b) `(- ,a ,b))
  (defmacro f32v* (a b) `(* ,a ,b))
  (defmacro f32v/ (a b) `(/ ,a ,b))
  (defmacro f32v-done () nil))

(defmacro do-vectorized ((iv is n) vector-body scalar-body)
  "Walk [0, N): VECTOR-BODY with IV bound to each index of a full lane, then
SCALAR-BODY with IS bound to each index of the remainder.

The two bodies have to be written twice because the vector one works on lanes
and the last few elements are not a lane.  They must compute the same thing in
the same order — every caller here builds them from the same expression.

The F32V-DONE sits between them because the remainder is scalar code, and so is
whatever called this; see the note at the top of this file for what it costs to
leave it out.  It is unconditional rather than inside the loop: a caller may have
broadcast a vector before getting here even when N is short enough that the
vector loop never runs, and that alone is enough to dirty the state.

Hoisting it out of a hot caller is a trap worth naming, because it looks like a
saving: reading one element of a float array is itself legacy SSE, so a loop
that reads a scalar, broadcasts it and vectorizes pays the transition every pass
unless the fence is inside.  Measured on ConvTranspose, hoisting it made that op
ten times slower than the scalar code it replaced."
  (let ((end (gensym "END")))
    `(let ((,end (- ,n (mod ,n +f32-lanes+))))
       (declare (type dim ,end))
       (do ((,iv 0 (+ ,iv +f32-lanes+))) ((>= ,iv ,end))
         (declare (type dim ,iv))
         ,vector-body)
       (f32v-done)
       (loop for ,is of-type dim from ,end below ,n
             do ,scalar-body))))
