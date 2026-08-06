;;;; ops-nn.lisp — the ops that do the actual work: matmul and convolution.
;;;;
;;;; Everything else in this graph is bookkeeping around these.  All of this
;;;; model's convolutions are 1-D — VITS is a waveform model, and its "images"
;;;; are (batch, channels, time) — so that is what is written here, with the
;;;; grouped and dilated cases the decoder needs.
;;;;
;;;; The loops are ordered so the innermost one walks time, which is contiguous
;;;; in memory and is the axis with 17152 elements in the last decoder layer.
;;;; That ordering is also what a vectorized version wants, so this is the shape
;;;; the SIMD rewrite starts from rather than something it has to undo.

(in-package #:mill)

;;; ---- matmul ----------------------------------------------------------------
;;;
;;; Split into a typed core and a shape-checking op for the same reason the conv
;;; kernels below are, and it was worth more here than anywhere else: MatMul is
;;; 26 nodes out of 2755 and was 22% of the run.  Untyped, every index expression
;;; in the inner loop went through GENERIC-+ and every AREF through
;;; HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS, so the arithmetic was a small part of
;;; what the arithmetic cost.
;;;
;;; A row here is one (batch item, output row) pair, flattened, which is what the
;;; worker pool splits on.  The batch offsets are gathered into three vectors up
;;; front rather than walked with DO-BROADCAST inside the kernel: a broadcast
;;; walk is an odometer with a carry, which is inherently sequential, and there
;;; are only a handful of batch items (attention heads) to gather.  Splitting on
;;; the flattened pair rather than on rows-within-a-matrix matters because this
;;; graph has both shapes — a few big matrices AND many small ones.
;;;
;;; Rows write disjoint slices of DR and each row's sum is accumulated in the
;;; same order as before, so the split changes no float result; the node gate is
;;; what says so.

(defun matmul-core (da db dr ias ibs ios m k n row-lo row-hi)
  "Accumulate rows [ROW-LO, ROW-HI) of a batch of matrix products into DR, which
the caller has left zeroed.  IAS/IBS/IOS are the per-batch-item element offsets
into DA/DB/DR; row R belongs to batch item (FLOOR R M)."
  (declare (type (simple-array single-float (*)) da db dr)
           (type (simple-array fixnum (*)) ias ibs ios)
           (type dim m k n row-lo row-hi)
           (optimize (speed 3) (safety 0)))
  (loop for row of-type dim from row-lo below row-hi do
    (multiple-value-bind (item r) (floor row m)
      (declare (type dim item r))
      (let ((arow (the dim (+ (aref ias item) (* r k))))
            (orow (the dim (+ (aref ios item) (* r n))))
            (bbase (the dim (aref ibs item))))
        ;; k on the outside, n contiguous on the inside: one row of B is
        ;; streamed per accumulation step instead of strided over.  The
        ;; zero test is on the k loop, not the n loop, so it costs one branch
        ;; per N multiply-adds — and this graph's attention matrices are full
        ;; of exact zeros from masking.
        (dotimes (kk k)
          (declare (type dim kk))
          (let ((x (aref da (the dim (+ arow kk))))
                (brow (the dim (+ bbase (* kk n)))))
            (unless (zerop x)
              (dotimes (col n)
                (declare (type dim col))
                (incf (aref dr (the dim (+ orow col)))
                      (* x (aref db (the dim (+ brow col))))))))))))
  (values))

(defop "MatMul" (node ins)
  (destructuring-bind (a b) ins
    (let* ((ashape (tensor-shape a)) (bshape (tensor-shape b))
           (ra (length ashape)) (rb (length bshape)))
      (when (or (< ra 2) (< rb 2))
        (error "MatMul with a rank-~d operand is not implemented" (min ra rb)))
      (let* ((m (aref ashape (- ra 2))) (k (aref ashape (1- ra)))
             (k2 (aref bshape (- rb 2))) (n (aref bshape (1- rb))))
        (unless (= k k2)
          (error "MatMul inner dimensions disagree: ~d vs ~d" k k2))
        (let* ((batch (broadcast-shapes (subseq ashape 0 (- ra 2))
                                        (subseq bshape 0 (- rb 2))))
               ;; a step along a batch axis skips a whole matrix
               (sa (map 'vector (lambda (s) (* s m k))
                        (broadcast-strides (subseq ashape 0 (- ra 2)) batch)))
               (sb (map 'vector (lambda (s) (* s k n))
                        (broadcast-strides (subseq bshape 0 (- rb 2)) batch)))
               (so (map 'vector (lambda (s) (* s m n)) (row-major-strides batch)))
               (out-shape (concatenate 'simple-vector batch (vector m n)))
               (res (make-tensor (tensor-dtype a) out-shape))
               (da (the (simple-array single-float (*)) (tensor-data a)))
               (db (the (simple-array single-float (*)) (tensor-data b)))
               (dr (the (simple-array single-float (*)) (tensor-data res))))
          (declare (type dim m k n))
          (let* ((nb (max 1 (shape-size batch)))
                 (ias (make-array nb :element-type 'fixnum :initial-element 0))
                 (ibs (make-array nb :element-type 'fixnum :initial-element 0))
                 (ios (make-array nb :element-type 'fixnum :initial-element 0)))
            (do-broadcast ((ia sa) (ib sb) (io so) batch)
              (setf (aref ias i) ia (aref ibs i) ib (aref ios i) io))
            ;; a row costs K * N multiply-adds; hand a thread work worth having
            (parallel-range (* nb m)
                            (lambda (lo hi)
                              (matmul-core da db dr ias ibs ios m k n lo hi))
                            :min-chunk (max 1 (ceiling +parallel-grain+
                                                       (max 1 (* k n))))))
          res)))))

;;; A fully-connected layer arrives as Gemm rather than MatMul: same product,
;;; with the transpose and the bias folded into the node.  Written as MatMul plus
;;; that fold rather than as a second matrix kernel — one kernel is one thing to
;;; tune and one thing for the conv gate's sibling to check.

(defun transpose-2d (a)
  ;; Element (i, j) of the result is element (j, i) of A, which sits at j*N + i —
  ;; so the walk steps by 1 along the result's first axis and by N along its
  ;; second.  N is the width of A, and it is not M: the two agree only when A is
  ;; square, which is why the transducer's decoder (a 512x512 projection) was
  ;; happy with the wrong one and its joiner (500x512) was not.
  (let ((m (aref (tensor-shape a) 0))
        (n (aref (tensor-shape a) 1)))
    (tensor-gather-strided a 0 (make-array 2 :element-type 'fixnum
                                             :initial-contents (list 1 n))
                           (vector n m))))

(defop "Gemm" (node ins)
  ;; alpha * A' * B' + beta * C, where ' is an optional transpose and C
  ;; broadcasts over the result.
  (destructuring-bind (a b &optional c) ins
    (let* ((alpha (node-attr node "alpha" 1))
           (beta (node-attr node "beta" 1))
           (a (if (eql 1 (node-attr node "transA" 0)) (transpose-2d a) a))
           (b (if (eql 1 (node-attr node "transB" 0)) (transpose-2d b) b))
           (res (op-matmul node (list a b)))
           ;; float32 throughout, because MatMul above is: a Gemm of any other
           ;; type cannot reach this line
           (dr (the (simple-array single-float (*)) (tensor-data res)))
           (al (float alpha 1f0)))
      (unless (= al 1f0)
        (dotimes (i (length dr)) (setf (aref dr i) (* al (aref dr i)))))
      (when c
        (let ((dc (the (simple-array single-float (*)) (tensor-data c)))
              (be (float beta 1f0))
              (sc (broadcast-strides (tensor-shape c) (tensor-shape res))))
          (do-broadcast ((ic sc) (tensor-shape res))
            (setf (aref dr i) (+ (aref dr i) (* be (aref dc ic)))))))
      res)))

;;; ---- convolution -----------------------------------------------------------

(defun conv1d-attrs (node)
  (values (first (node-attr node "strides" '(1)))
          (first (node-attr node "dilations" '(1)))
          (node-attr node "pads" '(0 0))
          (node-attr node "group" 1)))

;;; The two kernels below are split out of their ops for one reason: they run at
;;; (speed 3) (safety 0) with every index typed, while the op around them keeps
;;; normal safety for the shape checks.  Before the split, the profile of one
;;; synthesis put 14.1% of the whole run in GENERIC-+ and 5.3% in GENERIC-* —
;;; the accumulation was dispatching on operand type per tap.
;;;
;;; Every offset variable is declared DIM, and the row bases are hoisted to the
;;; loop level where they stop changing, so no expression ever multiplies three
;;; extents together (that overflows a fixnum and puts the generic call back).
;;;
;;; Both kernels take a range of OUTPUT ROWS — the (batch item, output channel)
;;; pairs, flattened — rather than the whole tensor, because rows are what the
;;; worker pool splits on.  Rows write disjoint slices of DR, and the loops that
;;; accumulate into one output element (input channel, then kernel position) are
;;; still nested in that order, so a row's answer does not depend on how the
;;; range was cut.  Same summation order as before the split, therefore the same
;;; float result; inspect/node-gate.lisp is what says so.

;;; Both kernels bottom out in CONV-AXPY, and it is where the time is: measured
;;; alone on one core, the loop it replaces ran at 0.75 G multiply-adds/s and this
;;; one runs at 2.4.  Two things cost that:
;;;
;;;   the stride multiply — with STRIDE 1 (every Conv in this graph) the source
;;;     index is just a counter, but the general form recomputes I * STRIDE and
;;;     hides the fact that DX is being walked contiguously;
;;;   the loop test — one compare and branch per multiply-add, against a body
;;;     that is two loads, a multiply, an add and a store.  The accumulations are
;;;     into different elements and so are independent, which is what makes it
;;;     safe to do four per test and let them issue together.
;;;
;;; Unrolling does not reorder anything: each output element still gets its terms
;;; in the same sequence, so the floats come out bit-for-bit identical.

(declaim (inline conv-axpy))
(defun conv-axpy (dr dx weight dst src count stride)
  "DR[DST+i] += WEIGHT * DX[SRC + i*STRIDE], for i in [0, COUNT)."
  (declare (type (simple-array single-float (*)) dr dx)
           (type single-float weight)
           (type dim dst src count stride)
           (optimize (speed 3) (safety 0)))
  (if (= stride 1)
      (let ((wv (f32v-broadcast weight)))
        (do-vectorized (iv is count)
          (setf (f32v-ref dr (+ dst iv))
                (f32v+ (f32v-ref dr (+ dst iv)) (f32v* wv (f32v-ref dx (+ src iv)))))
          (incf (aref dr (+ dst is)) (* weight (aref dx (+ src is))))))
      ;; a strided read cannot be one load, and every Conv in this graph is
      ;; stride 1 — this arm is for the gate's sweep and for ConvTranspose
      (loop for i of-type dim from 0 below count
            do (incf (aref dr (+ dst i)) (* weight (aref dx (+ src (* i stride)))))))
  (values))

;;; CONV-AXPY still touches the output row once per kernel tap, and the decoder's
;;; heavy layers have kw 5 and 7 — so a row is read and written seven times per
;;; input channel to accumulate what is arithmetically one sum.  Once the loop
;;; overhead was gone that traffic became the ceiling: Conv gets 6.2x out of 16
;;; threads and 8.6x out of 64, which is what running out of memory bandwidth
;;; looks like.  (Two ways of reducing the OTHER traffic were measured and thrown
;;; away first — accumulating a row in an L1-sized tile, and cutting the time axis
;;; so each input channel is read once per tile instead of once per output
;;; channel.  Both were worth nothing, because the activations already fit in a
;;; 32 MB L3 and were never going to memory.  The repeated row is.)
;;;
;;; CONV-FUSED does all KW taps in one pass. The taps of one kernel read DX at a
;;; fixed spacing — DILATION — and land on the same output element, so the row is
;;; loaded and stored once per input channel instead of KW times, while the reads
;;; of DX are the same bytes the separate passes read anyway.
;;;
;;; The sum is written as one left-to-right (+ dr t0 t1 ...), which is the order
;;; the separate passes accumulated in, so the floats do not move.
;;;
;;; Only for stride 1 — every Conv in this graph — since that is what makes the
;;; window where all KW taps are inside the signal a single contiguous run.  The
;;; handful of taps at each end, where the padding bites, go the per-tap way.

(defmacro %conv-fused-body (kw)
  "The fused loop for a literal KW, so the tap count is a compile-time constant
and the weights and source bases hoist out of the loop.

Vectorized over the output tap, which is the choice that costs nothing: each
lane is a different element of the row, so every element's KW terms are still
summed one after another in the same order and the answer is the scalar answer
bit for bit.  Putting the lanes across the taps instead — the obvious reading of
\"fused\" — would have meant a horizontal sum at the end, which reassociates."
  (let ((ws (loop for k below kw collect (gensym "W")))
        (vs (loop for k below kw collect (gensym "V")))
        (ss (loop for k below kw collect (gensym "S")))
        (iv (gensym "IV")) (is (gensym "IS")))
    (flet ((chain (index scalarp)
             ;; ((dr + w0*x0) + w1*x1) + ... , left to right, which is the order
             ;; the separate per-tap passes accumulated in
             (loop with acc = (if scalarp
                                  `(aref dr (+ dst ,index))
                                  `(f32v-ref dr (+ dst ,index)))
                   for s in ss
                   for w in (if scalarp ws vs)
                   do (setf acc (if scalarp
                                    `(+ ,acc (* ,w (aref dx (+ ,s ,index))))
                                    `(f32v+ ,acc (f32v* ,w (f32v-ref dx (+ ,s ,index))))))
                   finally (return acc))))
      `(let (,@(loop for w in ws for k from 0 collect `(,w (aref dw (+ wrow ,k))))
             ,@(loop for s in ss for k from 0 collect `(,s (+ src (* ,k dilation)))))
         (declare (type single-float ,@ws) (type dim ,@ss))
         (let (,@(loop for v in vs for w in ws collect `(,v (f32v-broadcast ,w))))
           (do-vectorized (,iv ,is count)
             (setf (f32v-ref dr (+ dst ,iv)) ,(chain iv nil))
             (setf (aref dr (+ dst ,is)) ,(chain is t))))))))

(defun conv-fused (dr dx dw wrow kw dilation dst src count)
  "DR[DST+i] += sum over k in [0, KW) of DW[WROW+k] * DX[SRC + k*DILATION + i],
for i in [0, COUNT).  Every tap is assumed in range; the caller establishes that."
  (declare (type (simple-array single-float (*)) dr dx dw)
           (type dim wrow kw dilation dst src count)
           (optimize (speed 3) (safety 0)))
  (case kw
    (1 (%conv-fused-body 1)) (2 (%conv-fused-body 2)) (3 (%conv-fused-body 3))
    (4 (%conv-fused-body 4)) (5 (%conv-fused-body 5)) (6 (%conv-fused-body 6))
    (7 (%conv-fused-body 7)) (8 (%conv-fused-body 8))
    ;; wider kernels than this graph has: still one pass over the row, but the
    ;; tap loop stays a loop
    (t (loop for i of-type dim from 0 below count
             do (let ((acc (aref dr (+ dst i))))
                  (declare (type single-float acc))
                  (dotimes (k kw)
                    (declare (type dim k))
                    (setf acc (+ acc (* (aref dw (+ wrow k))
                                        (aref dx (+ src (* k dilation) i))))))
                  (setf (aref dr (+ dst i)) acc)))))
  (values))

(declaim (inline conv1d-taps))
(defun conv1d-taps (dr dx dw wrow orow xrow kw dilation pad-begin stride len
                    out-len range-lo range-hi)
  "Accumulate every kernel tap into output taps [RANGE-LO, RANGE-HI) of one row,
one pass per tap: the unfused path.  Used for the whole row when the fused one
does not apply, and for the padded ends when it does."
  (declare (type (simple-array single-float (*)) dr dx dw)
           (type dim wrow orow xrow kw dilation stride len out-len
                 range-lo range-hi)
           (type (signed-byte 30) pad-begin)
           (optimize (speed 3) (safety 0)))
  (dotimes (kk kw)
    (declare (type dim kk))
    (let ((weight (aref dw (+ wrow kk)))
          ;; the taps this kernel position contributes to are the ones whose
          ;; source sample is inside the signal; solve for that range once
          ;; instead of testing every tap
          (shift (- (* kk dilation) pad-begin)))
      (declare (type fixnum shift))
      (unless (zerop weight)
        ;; both ends are clamped into DIM: LO because a tap index is never
        ;; negative, HI because an empty range has to stay representable when
        ;; SHIFT runs off the end of the signal
        (let ((lo (max range-lo (max 0 (ceiling (- shift) stride))))
              (hi (min range-hi (max 0 (min out-len (ceiling (- len shift) stride))))))
          (declare (type dim lo hi))
          ;; the first tap in range starts at SHIFT + LO*STRIDE, which is what LO
          ;; was solved for, so the source index is non-negative from here on
          (when (< lo hi)
            (conv-axpy dr dx weight (+ orow lo) (+ xrow shift (* lo stride))
                       (- hi lo) stride))))))
  (values))

(declaim (inline conv1d-line))
(defun conv1d-line (dr dx dw wrow orow xrow kw dilation pad-begin stride len out-len
                    ilo ihi)
  "Convolve one input line into one output line: the padded ends one tap at a
time, the middle all at once.

ILO and IHI bound the run of output taps for which every kernel tap reads a real
sample, and the caller solves for them because they depend only on the geometry —
not on which line this is.  An empty run (ILO >= IHI, which is what a strided
convolution always gets, since the fused loop is a stride-1 walk) means the whole
line goes the per-tap way.

This is the whole of what the 1-D and 2-D convolutions share: a 2-D kernel row is
a 1-D kernel applied to one input row, so the tuned loops below serve both and
there is one summation order between them rather than two."
  (declare (type (simple-array single-float (*)) dr dx dw)
           (type dim wrow orow xrow kw dilation stride len out-len ilo ihi)
           (type (signed-byte 30) pad-begin)
           (optimize (speed 3) (safety 0)))
  (cond ((< ilo ihi)
         (conv1d-taps dr dx dw wrow orow xrow kw dilation pad-begin stride
                      len out-len 0 ilo)
         ;; the fused window's first source sample is ILO - PAD-BEGIN taps in,
         ;; which is exactly what ILO was solved for
         (conv-fused dr dx dw wrow kw dilation (+ orow ilo)
                     (+ xrow (- ilo pad-begin)) (- ihi ilo))
         (conv1d-taps dr dx dw wrow orow xrow kw dilation pad-begin stride
                      len out-len ihi out-len))
        (t
         (conv1d-taps dr dx dw wrow orow xrow kw dilation pad-begin stride
                      len out-len 0 out-len)))
  (values))

(declaim (inline fused-window))
(defun fused-window (out-len len kw dilation pad-begin stride)
  "Values: the first and last output tap of the run where every kernel tap reads
a real sample.  It starts once the leftmost tap clears the front padding and ends
when the rightmost runs off the signal.  Empty unless the stride is 1, which is
what makes that run a contiguous walk of the input."
  (if (= stride 1)
      (let ((lo (min out-len (max 0 pad-begin))))
        (values lo (max lo (min out-len
                                (max 0 (+ (- len (* (1- kw) dilation)) pad-begin))))))
      (values 0 0)))

(defun conv1d-core (dx dw dr row-lo row-hi cin len cout cgroup kw out-len
                    stride dilation pad-begin per-group-out)
  "Accumulate the 1-D convolution of DX by DW into rows [ROW-LO, ROW-HI) of DR,
which the caller has already filled with the bias (or with zeros)."
  (declare (type (simple-array single-float (*)) dx dw dr)
           (type dim row-lo row-hi cin len cout cgroup kw out-len stride dilation
                 per-group-out)
           (type (signed-byte 30) pad-begin)
           (optimize (speed 3) (safety 0)))
  (multiple-value-bind (ilo ihi) (fused-window out-len len kw dilation pad-begin stride)
   (let* ((xplane (* cin len))           ; elements per batch item, input
          (oplane (* cout out-len))      ;                          output
          (wplane (* cgroup kw)))        ; elements per output channel, weights
    (declare (type dim xplane oplane wplane ilo ihi))
    (loop for row of-type dim from row-lo below row-hi do
      (multiple-value-bind (nb m) (floor row cout)
        (declare (type dim nb m))
        (let ((xbase (* nb xplane))
              (orow (+ (* nb oplane) (* m out-len)))
              (wbase (* m wplane))
              (gr (floor m per-group-out)))
          (declare (type dim xbase orow wbase gr))
          (dotimes (c cgroup)
            (declare (type dim c))
            (let* ((chan (+ (* gr cgroup) c))
                   (xrow (+ xbase (* chan len)))
                   (wrow (+ wbase (* c kw))))
              (declare (type dim chan xrow wrow))
              (conv1d-line dr dx dw wrow orow xrow kw dilation pad-begin stride
                           len out-len ilo ihi))))))))
  (values))

;;; A 2-D convolution is a sum of 1-D ones.  Fix a kernel row and it reads a
;;; single input row with a single row of weights, along the innermost axis,
;;; which is exactly the loop above — so the whole of the new code is the walk
;;; that decides WHICH input row each kernel row lands on, and the arithmetic
;;; stays in the kernels that are already tuned and gated.
;;;
;;; An output element accumulates in the order (input channel, kernel row,
;;; kernel column), the same nesting the 1-D path uses with its middle loop
;;; absent, and the rows a thread owns are still disjoint — so splitting the
;;; work does not move a float here either.

(defun conv2d-core (dx dw dr row-lo row-hi cin ih iw cout cgroup kh kw oh ow
                    stride-h stride-w dil-h dil-w pad-h pad-w per-group-out)
  "Accumulate the 2-D convolution of DX by DW into rows [ROW-LO, ROW-HI) of DR,
which the caller has already filled with the bias.  A row is a (batch item,
output channel) pair, as in the 1-D case; here it holds OH lines of OW."
  (declare (type (simple-array single-float (*)) dx dw dr)
           (type dim row-lo row-hi cin ih iw cout cgroup kh kw oh ow
                 stride-h stride-w dil-h dil-w per-group-out)
           (type (signed-byte 30) pad-h pad-w)
           (optimize (speed 3) (safety 0)))
  (multiple-value-bind (ilo ihi) (fused-window ow iw kw dil-w pad-w stride-w)
    (let ((xplane (* cin ih iw))         ; elements per batch item, input
          (oplane (* cout oh ow))        ;                          output
          (wplane (* cgroup kh kw)))     ; elements per output channel, weights
      (declare (type dim xplane oplane wplane ilo ihi))
      (loop for row of-type dim from row-lo below row-hi do
        (multiple-value-bind (nb m) (floor row cout)
          (declare (type dim nb m))
          (let ((xbase (* nb xplane))
                (obase (+ (* nb oplane) (* m oh ow)))
                (wbase (* m wplane))
                (gr (floor m per-group-out)))
            (declare (type dim xbase obase wbase gr))
            (dotimes (c cgroup)
              (declare (type dim c))
              (let ((xchan (+ xbase (* (+ (* gr cgroup) c) ih iw)))
                    (wchan (+ wbase (* c kh kw))))
                (declare (type dim xchan wchan))
                (dotimes (y oh)
                  (declare (type dim y))
                  (let ((orow (+ obase (* y ow))))
                    (declare (type dim orow))
                    (dotimes (k kh)
                      (declare (type dim k))
                      ;; the input row this kernel row reads; outside the image
                      ;; it is padding, and padding contributes nothing
                      (let ((sy (+ (* y stride-h) (* k dil-h) (- pad-h))))
                        (declare (type fixnum sy))
                        (when (and (>= sy 0) (< sy ih))
                          (conv1d-line dr dx dw (+ wchan (* k kw)) orow
                                       (+ xchan (* sy iw)) kw dil-w pad-w
                                       stride-w iw ow ilo ihi)))))))))))))
  (values))

(defun fill-conv-bias (dr db batch cout row-len)
  "The bias is the initial value of every output tap, so it goes in before the
accumulation rather than inside it.  ROW-LEN is the taps per (batch, channel)
row — one line in 1-D, a whole image in 2-D."
  (declare (type (simple-array single-float (*)) dr db))
  (dotimes (nb batch)
    (dotimes (m cout)
      (let ((orow (* (+ (* nb cout) m) row-len))
            (b (aref db m)))
        (dotimes (o row-len) (setf (aref dr (+ orow o)) b))))))

(defun conv1d (node x w bias)
  (multiple-value-bind (stride dilation pads group) (conv1d-attrs node)
    (let* ((xs (tensor-shape x)) (ws (tensor-shape w))
           (batch (aref xs 0)) (cin (aref xs 1)) (len (aref xs 2))
           (cout (aref ws 0)) (cgroup (aref ws 1)) (kw (aref ws 2))
           (pad-begin (first pads)) (pad-end (second pads))
           (out-len (1+ (floor (- (+ len pad-begin pad-end) (* dilation (1- kw)) 1) stride)))
           (per-group-out (floor cout group))
           (res (make-tensor :f32 (vector batch cout out-len)))
           (dx (the (simple-array single-float (*)) (tensor-data x)))
           (dw (the (simple-array single-float (*)) (tensor-data w)))
           (dr (the (simple-array single-float (*)) (tensor-data res)))
           (db (and bias (the (simple-array single-float (*)) (tensor-data bias)))))
      (declare (type index len out-len kw))
      (unless (= cin (* cgroup group))
        (error "Conv: input has ~d channels but the weights want ~d x ~d"
               cin group cgroup))
      (when db (fill-conv-bias dr db batch cout out-len))
      ;; a row costs CGROUP * KW multiply-adds per output tap; hand a thread
      ;; work worth having, not a row that finishes before it wakes up
      (parallel-range (* batch cout)
                      (lambda (lo hi)
                        (conv1d-core dx dw dr lo hi cin len cout cgroup kw out-len
                                     stride dilation pad-begin per-group-out))
                      :min-chunk (max 1 (ceiling +parallel-grain+
                                                 (max 1 (* cgroup kw out-len)))))
      res)))

(defun conv2d (node x w bias)
  (let* ((strides (node-attr node "strides" '(1 1)))
         (dilations (node-attr node "dilations" '(1 1)))
         ;; ONNX writes pads as every axis's begin, then every axis's end
         (pads (node-attr node "pads" '(0 0 0 0)))
         (group (node-attr node "group" 1))
         (xs (tensor-shape x)) (ws (tensor-shape w))
         (batch (aref xs 0)) (cin (aref xs 1)) (ih (aref xs 2)) (iw (aref xs 3))
         (cout (aref ws 0)) (cgroup (aref ws 1)) (kh (aref ws 2)) (kw (aref ws 3))
         (stride-h (first strides)) (stride-w (second strides))
         (dil-h (first dilations)) (dil-w (second dilations))
         (pad-h (first pads)) (pad-w (second pads))
         (oh (1+ (floor (- (+ ih pad-h (third pads)) (* dil-h (1- kh)) 1) stride-h)))
         (ow (1+ (floor (- (+ iw pad-w (fourth pads)) (* dil-w (1- kw)) 1) stride-w)))
         (per-group-out (floor cout group))
         (res (make-tensor :f32 (vector batch cout oh ow)))
         (dx (the (simple-array single-float (*)) (tensor-data x)))
         (dw (the (simple-array single-float (*)) (tensor-data w)))
         (dr (the (simple-array single-float (*)) (tensor-data res)))
         (db (and bias (the (simple-array single-float (*)) (tensor-data bias)))))
    (declare (type index ih iw oh ow kh kw))
    (unless (= cin (* cgroup group))
      (error "Conv: input has ~d channels but the weights want ~d x ~d"
             cin group cgroup))
    (when db (fill-conv-bias dr db batch cout (* oh ow)))
    (parallel-range (* batch cout)
                    (lambda (lo hi)
                      (conv2d-core dx dw dr lo hi cin ih iw cout cgroup kh kw oh ow
                                   stride-h stride-w dil-h dil-w pad-h pad-w
                                   per-group-out))
                    :min-chunk (max 1 (ceiling +parallel-grain+
                                               (max 1 (* cgroup kh kw oh ow)))))
    res))

(defop "Conv" (node ins)
  (destructuring-bind (x w &optional bias) ins
    (let ((auto (node-attr node "auto_pad" "NOTSET"))
          (rank (- (length (tensor-shape w)) 2)))
      ;; SAME_UPPER and friends compute the padding from the input shape, and an
      ;; exporter that knows the shape writes the padding out instead — so this
      ;; has never been seen, and guessing at it is worse than saying so
      (unless (equal auto "NOTSET")
        (error "Conv ~a wants auto_pad ~a, and only explicit pads are implemented"
               (node-name node) auto))
      (case rank
        (1 (conv1d node x w bias))
        (2 (conv2d node x w bias))
        (t (error "Conv over ~d spatial dimensions is not implemented" rank))))))

;;; ConvTranspose's inner loop is CONV-AXPY read backwards: the contiguous walk
;;; is over the input and the stride lands on the output, so the multiply cannot
;;; be dropped — the upsampling layers here stride by 8.  Amortizing the loop test
;;; over four independent accumulations still applies, and is all that is left to
;;; take.
(declaim (inline conv-scatter))
(defun conv-scatter (dr dx weight dst src count stride)
  "DR[DST + i*STRIDE] += WEIGHT * DX[SRC+i], for i in [0, COUNT)."
  (declare (type (simple-array single-float (*)) dr dx)
           (type single-float weight)
           (type dim dst src count stride)
           (optimize (speed 3) (safety 0)))
  (let ((quad (logandc2 count 3))
        (s2 (* 2 stride)) (s3 (* 3 stride)) (s4 (* 4 stride)))
    (declare (type dim quad s2 s3 s4))
    (do ((i 0 (+ i 4)) (d dst (+ d s4)) (b src (+ b 4)))
        ((>= i quad))
      (declare (type dim i d b))
      (setf (aref dr d)          (+ (aref dr d)          (* weight (aref dx b)))
            (aref dr (+ d stride)) (+ (aref dr (+ d stride)) (* weight (aref dx (+ b 1))))
            (aref dr (+ d s2))   (+ (aref dr (+ d s2))   (* weight (aref dx (+ b 2))))
            (aref dr (+ d s3))   (+ (aref dr (+ d s3))   (* weight (aref dx (+ b 3))))))
    (loop for i of-type dim from quad below count
          do (incf (aref dr (+ dst (* i stride))) (* weight (aref dx (+ src i))))))
  (values))

;;; The scatter above is one output element per multiply and STRIDE elements
;;; apart, which is the worst thing a vector unit can be handed.  It does not
;;; have to be that way.  A single kernel position K reaches only the outputs
;;; J = L*STRIDE + K - PAD, and every one of those has the same J mod STRIDE —
;;; so a pass over the input for one K writes one residue class of the output
;;; row, and it writes it in order, one element per input sample.
;;;
;;; De-interleave the row into its STRIDE classes and every pass becomes an
;;; ordinary stride-1 CONV-AXPY: contiguous in, contiguous out, vectorizable.
;;; Nothing about which terms are added, or in what order, changes — this is the
;;; same loop over the same taps, writing to a permuted address — which is what
;;; lets inspect/conv-gate.lisp keep comparing exactly.  The permutation costs
;;; two passes over the row, against CGROUP-IN * KW passes of real work.
(declaim (inline transpose-deinterleave transpose-interleave))
(defun transpose-deinterleave (dr tmp orow out-len stride)
  "TMP gets row OROW of DR, one residue class after another."
  (declare (type (simple-array single-float (*)) dr tmp)
           (type dim orow out-len stride) (optimize (speed 3) (safety 0)))
  (let ((p 0))
    (declare (type dim p))
    (dotimes (r stride)
      (loop for j of-type dim from r below out-len by stride
            do (setf (aref tmp p) (aref dr (+ orow j)))
               (incf p))))
  (values))

(defun transpose-interleave (dr tmp orow out-len stride)
  "The inverse of TRANSPOSE-DEINTERLEAVE."
  (declare (type (simple-array single-float (*)) dr tmp)
           (type dim orow out-len stride) (optimize (speed 3) (safety 0)))
  (let ((p 0))
    (declare (type dim p))
    (dotimes (r stride)
      (loop for j of-type dim from r below out-len by stride
            do (setf (aref dr (+ orow j)) (aref tmp p))
               (incf p))))
  (values))

(defun conv-transpose1d-core (dx dw dr row-lo row-hi cin len cout cgroup-in
                              cgroup-out kw out-len stride dilation pad-begin)
  "Scatter DX through DW into rows [ROW-LO, ROW-HI) of DR, which the caller has
already filled with the bias (or with zeros).  Note the weight layout:
(in-channels, out/group, kernel) — the transpose of Conv's."
  (declare (type (simple-array single-float (*)) dx dw dr)
           (type dim row-lo row-hi cin len cout cgroup-in cgroup-out kw out-len
                 stride dilation)
           (type (signed-byte 30) pad-begin)
           (optimize (speed 3) (safety 0)))
  (let* ((xplane (* cin len))
         (oplane (* cout out-len))
         (wplane (* cgroup-out kw))     ; elements per INPUT channel here
         ;; the de-interleaved path needs one scratch row, and a dilated kernel
         ;; does not have the property it rests on (its taps skip residues)
         (classes (= dilation 1))
         (tmp (make-array (if classes out-len 0) :element-type 'single-float))
         ;; where each residue class starts in TMP
         (cstart (make-array (if classes stride 0) :element-type 'fixnum)))
    (declare (type dim xplane oplane wplane)
             (type (simple-array single-float (*)) tmp))
    (when classes
      (let ((p 0))
        (declare (type dim p))
        (dotimes (r stride)
          (setf (aref cstart r) p)
          (incf p (max 0 (ceiling (- out-len r) stride))))))
    (loop for row of-type dim from row-lo below row-hi do
      (multiple-value-bind (nb ochan) (floor row cout)
        (declare (type dim nb ochan))
        (multiple-value-bind (g m) (floor ochan cgroup-out)
          (declare (type dim g m))
          (let ((xbase (* nb xplane))
                (orow (+ (* nb oplane) (* ochan out-len)))
                (wcol (* m kw)))
            (declare (type dim xbase orow wcol))
            (when classes (transpose-deinterleave dr tmp orow out-len stride))
            (dotimes (ci cgroup-in)
              (declare (type dim ci))
              (let* ((c (+ (* g cgroup-in) ci))
                     (xrow (+ xbase (* c len)))
                     (wrow (+ (* c wplane) wcol)))
                (declare (type dim c xrow wrow))
                (dotimes (kk kw)
                  (declare (type dim kk))
                  (let ((weight (aref dw (+ wrow kk)))
                        ;; output tap for input sample l is l*stride + shift
                        (shift (- (* kk dilation) pad-begin)))
                    (declare (type fixnum shift))
                    (unless (zerop weight)
                      (let ((lo (max 0 (ceiling (- shift) stride)))
                            (hi (max 0 (min len (ceiling (- out-len shift)
                                                         stride)))))
                        (declare (type dim lo hi))
                        ;; LO was solved for exactly the point where the output
                        ;; tap stops being negative
                        (when (< lo hi)
                          (if classes
                              ;; this tap owns residue class SHIFT mod STRIDE,
                              ;; and input sample l is its element l + SHIFT div
                              ;; STRIDE — so the same pass, written contiguously
                              (conv-axpy tmp dx weight
                                         (the dim (+ (aref cstart
                                                           (mod shift stride))
                                                     lo (floor shift stride)))
                                         (+ xrow lo) (- hi lo) 1)
                              (conv-scatter dr dx weight
                                            (+ orow shift (* lo stride))
                                            (+ xrow lo)
                                            (- hi lo) stride)))))))))
            (when classes (transpose-interleave dr tmp orow out-len stride)))))))
  (values))

(defop "ConvTranspose" (node ins)
  (destructuring-bind (x w &optional bias) ins
    (let ((kernel (node-attr node "kernel_shape")))
      (unless (= 1 (length kernel))
        (error "ConvTranspose over ~d spatial dimensions is not implemented"
               (length kernel))))
    (multiple-value-bind (stride dilation pads group) (conv1d-attrs node)
      (let* ((xs (tensor-shape x)) (ws (tensor-shape w))
             (batch (aref xs 0)) (cin (aref xs 1)) (len (aref xs 2))
             ;; note the weight layout: (in-channels, out-channels/group, kernel)
             (cgroup-in (floor cin group)) (cgroup-out (aref ws 1)) (kw (aref ws 2))
             (cout (* cgroup-out group))
             (pad-begin (first pads)) (pad-end (second pads))
             (extra (first (node-attr node "output_padding" '(0))))
             (out-len (+ (* (1- len) stride) (- pad-begin) (- pad-end)
                         (* dilation (1- kw)) 1 extra))
             (res (make-tensor :f32 (vector batch cout out-len)))
             (dx (the (simple-array single-float (*)) (tensor-data x)))
             (dw (the (simple-array single-float (*)) (tensor-data w)))
             (dr (the (simple-array single-float (*)) (tensor-data res)))
             (db (and bias (the (simple-array single-float (*)) (tensor-data bias)))))
        (declare (type index len out-len kw))
        (when db
          (dotimes (nb batch)
            (dotimes (m cout)
              (let ((orow (+ (* nb cout out-len) (* m out-len)))
                    (b (aref db m)))
                (dotimes (o out-len) (setf (aref dr (+ orow o)) b))))))
        (parallel-range (* batch cout)
                        (lambda (lo hi)
                          (conv-transpose1d-core dx dw dr lo hi cin len cout
                                                 cgroup-in cgroup-out kw out-len
                                                 stride dilation pad-begin))
                        :min-chunk (max 1 (ceiling +parallel-grain+
                                                   (max 1 (* cgroup-in kw len)))))
        res))))

;;; ---- noise -----------------------------------------------------------------
;;;
;;; VITS draws Gaussian noise in two places: the stochastic duration predictor
;;; (scaled by noise_w) and the flow (scaled by noise_scale).  Both scales come
;;; in as inputs, so setting them to zero makes the whole graph a pure function
;;; — which is exactly what the fixture dump does, and why a golden value exists
;;; for all but three of its nodes.  With the scales at their voice defaults the
;;; noise is real, and this is where it comes from.

(defvar *noise-seed* 1
  "Seed for the graph's Gaussian noise.  Fixed by default: two runs of the same
text should give the same audio unless something asks for otherwise.")

(defvar *noise-state* nil)

(defun reset-noise (&optional (seed *noise-seed*))
  (setf *noise-state* (sb-ext:seed-random-state seed)))

(defun random-normal ()
  "One standard normal, Box-Muller.  Not the same stream as onnxruntime's — the
per-node gate skips the three values downstream of this, and everything past the
noise-scale multiply agrees again."
  (unless *noise-state* (reset-noise))
  (let ((u (max 1d-300 (random 1d0 *noise-state*)))
        (v (random 1d0 *noise-state*)))
    (* (sqrt (* -2d0 (log u))) (cos (* 2d0 pi v)))))

(defop "RandomNormalLike" (node ins)
  (let* ((a (first ins))
         (dtype (if (node-attr node "dtype") (onnx-dtype (node-attr node "dtype"))
                    (tensor-dtype a)))
         (mean (float (node-attr node "mean" 0.0d0) 1d0))
         (scale (float (node-attr node "scale" 1.0d0) 1d0))
         (res (make-tensor dtype (tensor-shape a))))
    (with-typed-data (dr res :dtypes (:f32 :f64))
      (dotimes (i (length dr))
        (setf (aref dr i) (coerce (+ mean (* scale (random-normal)))
                                  (array-element-type dr)))))
    res))
