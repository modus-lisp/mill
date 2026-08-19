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
;;; A row here is one (batch item, output row) pair, flattened.  The batch
;;; offsets are gathered into three vectors up front rather than walked with
;;; DO-BROADCAST inside the kernel: a broadcast walk is an odometer with a carry,
;;; which is inherently sequential, and there are only a handful of batch items
;;; to gather.
;;;
;;; The kernel is written around what these graphs actually ask for, which is not
;;; what "matrix multiply" suggests.  A streaming Zipformer's dominant MatMul is
;;; (4 1 384) x (384 768): FOUR rows against a matrix a megabyte wide.  Nothing
;;; about that is compute-bound — every element of B is read, used once, and
;;; thrown away — and widening the inner loop to AVX2 measured exactly nothing.
;;;
;;; That last result was the interesting one, because the reason was not the one
;;; it looked like.  The loop was not short of lanes and it was not short of
;;; bandwidth: it was paying an AVX-SSE transition on every pass over K, because
;;; reading the multiplier out of A with AREF is a legacy SSE instruction and
;;; everything around it is VEX.  The note in simd.lisp has the measurement —
;;; ~115 ns per transition, which is 46 us of a 243 us call here, and which
;;; becomes 90% of the run in any kernel that reads a scalar more than once per
;;; pass.  That is also why the obvious rewrite, accumulating in registers with K
;;; on the inside, came out FIFTY TIMES slower the first time it was tried, and
;;; why this file used to blame the column-major walk over B.  It was not the
;;; walk.  With the multiplier loaded by F32V-BROADCAST-REF the same rewrite is
;;; three times faster than the loop it replaces, and the stride over B — the
;;; thing that supposedly cost fifty times — measures the same as reading B
;;; straight through.
;;;
;;; So: a block of +MM-ROWS+ output rows by +MM-LANES+ lane groups of columns is
;;; held in vector registers while K runs on the inside, and B is read once per
;;; block.  Twelve accumulators, three B values and one multiplier is sixteen
;;; registers, which is what AVX2 has; 4x4 spills and measured twice as slow as
;;; 4x3, and 4x1 half as fast.
;;;
;;; The stride over B was innocent of the fifty times, but it is not innocent.
;;; It only shows up at the working set the model actually has: this encoder is
;;; 250 MB of weights and every one of them is read once per 0.16 s of audio, so
;;; nothing is ever warm.  A block walk reads twenty-four floats and jumps a row,
;;; and measured against a straight read of the same 203 MB that is 7.98 GB/s
;;; against 21.29 on one core and 20.60 against 30.41 on the pool.  At a
;;; megabyte, where the earlier measurements were taken, the two are the same
;;; speed and this whole paragraph is invisible.
;;;
;;; The fix is the one every BLAS makes: permute B once into the order the kernel
;;; visits it — column block outer, K inner, lanes contiguous — so the walk is a
;;; straight read.  It is worth doing here for a reason a BLAS does not have,
;;; which is that these B's are weights.  The same matrix is multiplied by a new
;;; row of audio sixty-four times a second for the length of the recording, so
;;; the permutation is paid once and read thousands of times.  MM-PACKED-B is
;;; what decides, and what it asks is whether B is a weight — CONSTANT-DATA-P,
;;; which LOAD-MODEL sets on the initializers and nothing else sets.  It is
;;; tempting to ask instead whether this B has been seen before, and that is
;;; wrong in exactly the case that matters: graph.lisp pools activation buffers,
;;; so the same data vector comes back holding somebody else's numbers, and a
;;; pack cached against it would be right the first time and silently stale
;;; after.  On the dominant shape at 203 MB that is 240 us a call against 47,
;;; and the answers are bit-identical because a permutation is not arithmetic.
;;;
;;; The zero skip is the one thing that loop cannot do, because testing for zero
;;; is a scalar compare and a scalar compare is the same 115 ns — measured, one
;;; ZEROP per row per pass cost more than every multiply in the block put
;;; together.  So the test moves out of the loop: each group of rows is scanned
;;; once before the vector work, and a group with a zero anywhere in it runs the
;;; older K-outside kernel, which skips per row and pays the transition once per
;;; pass rather than four times.  On this model that is 16% of MatMul calls and
;;; 3% of the multiplies, and 4.6% of the seconds MatMul spends.
;;;
;;; That kernel reads B where it lies, not the pack, which is the one place this
;;; file knowingly leaves the bad walk in — it is a few percent of a few percent,
;;; and the packed layout puts a group's K contiguously, so if it ever matters
;;; the fix is a read address and not an argument.
;;;
;;; Splitting for threads follows the shape rather than the tradition.  Rows are
;;; the natural unit and there are FOUR of them here, so a 16-thread pool over
;;; rows is a 4-thread pool; when rows are scarce the split is over columns
;;; instead, which are disjoint, plentiful, and give each thread its own block.
;;;
;;; Every output element is still a sum over K in increasing K order, with the
;;; same zero skip, so no float result moves: the node gate and the two element
;;; gates are what say so, and they are exact comparisons.

(defconstant +mm-rows+ 4
  "Output rows computed together off one pass over B.  Each costs an accumulator
register per lane group and saves a re-read of B; four is where the accumulators
still fit alongside the operands, and it is exactly the row count of this
model's largest MatMul.")

(defconstant +mm-lanes+ 3
  "Lane groups of columns held in registers at once.  R rows by C lane groups
costs R*C accumulators plus C values of B plus one multiplier, and sixteen is
all there is: 4x3 measured 75 us on the dominant shape, 4x2 84, 4x1 108, and 4x4
164 because it spills.")

(defconstant +mm-cols+ (* +mm-lanes+ +f32-lanes+)
  "Columns in a wide block, and the alignment of every column window a caller
hands the kernel.  A thread that started at a lane boundary instead of a block
boundary would visit the blocks the packed B does not have.")

(defparameter *mm-pack-floor* 65536
  "Elements of B below which packing is not worth it.  A quarter of a megabyte
still fits in L2, and a walk that stays in L2 is not paying for its stride —
measured, at a megabyte the blocked and straight reads run at the same speed and
the pack is a copy for nothing.  Set this above any weight in the model to turn
packing off; it costs a second copy of every matrix it applies to.")

(defvar *mm-packs*
  (make-hash-table :test 'eq :weakness :key :synchronized t)
  "B's data vector to its packed copy.

Weak on the key, so a packed copy lives exactly as long as the weights it holds —
a model that is dropped drops its packs with it.  Keyed on the data vector rather
than the tensor because a reshape shares the data and wants the same pack.")

(defun mm-pack (db k n)
  "DB permuted into the order %MM-FAST reads it: wide blocks of +MM-COLS+
columns, then single lane groups, then the columns that are not a lane, each run
holding all of K contiguously.

Every column occupies exactly K floats and the groups are in increasing column
order, so the run for a group starting at column COL begins at COL*K — which is
why the kernel needs no table to find it."
  (declare (type (simple-array single-float (*)) db) (type dim k n)
           (optimize (speed 3) (safety 0)))
  (let ((p (make-array (* k n) :element-type 'single-float))
        (o 0) (col 0))
    (declare (type dim o col))
    (flet ((group (w)
             (declare (type dim w))
             (dotimes (kk k)
               (let ((brow (the dim (+ (the dim (* kk n)) col))))
                 (dotimes (j w)
                   (setf (aref p o) (aref db (the dim (+ brow j))))
                   (incf o))))
             (incf col w)))
      (loop while (<= (+ col +mm-cols+) n) do (group +mm-cols+))
      (loop while (<= (+ col +f32-lanes+) n) do (group +f32-lanes+))
      (loop while (< col n) do (group 1)))
    p))

(defun mm-packed-b (db k n)
  "The packed copy of DB, building it the first time, or NIL to say the kernel
should read B where it lies.

Only a weight is packed, and CONSTANT-DATA-P is the only thing that says so —
not a use count, which would be wrong for exactly the case that matters: an
activation's vector goes back to the pool and comes out again holding something
else, so a copy kept against it is right until it silently is not.  A weight, by
contrast, comes round once per audio frame for the length of the recording and
pays for the permutation on the second call.

Only a whole single matrix qualifies: a batch of them shares one data vector and
would need a pack each."
  (declare (type (simple-array single-float (*)) db) (type dim k n))
  (when (and (= (length db) (* k n))
             (>= (* k n) *mm-pack-floor*)
             (constant-data-p db))
    (or (gethash db *mm-packs*)
        (setf (gethash db *mm-packs*) (mm-pack db k n)))))

(declaim (inline mm-dense-p))
(defun mm-dense-p (da arows rr k)
  "Is every one of the RR rows' K multipliers nonzero?

Asked once per group, outside the vector loop, because asking inside it is a
scalar compare and a scalar compare costs a transition — see simd.lisp.  A NO
sends the group to %MM-SPARSE, which is slower and does the skip properly."
  (declare (type (simple-array single-float (*)) da)
           (type (simple-array fixnum (*)) arows)
           (type dim rr k) (optimize (speed 3) (safety 0)))
  (dotimes (i rr t)
    (let ((base (aref arows i)))
      (declare (type dim base))
      (dotimes (kk k)
        (when (zerop (aref da (the dim (+ base kk))))
          (return-from mm-dense-p nil))))))

(defmacro %mm-block (r c packed)
  "Column blocks of C lane groups for R rows, accumulating in registers.

K is on the inside and the accumulators never touch memory until the block is
finished, so B is read once per block and the output once per block.  Advances
COL to the first column it did not cover, so the caller can follow this with a
narrower block and then the scalar remainder.

PACKED says which B this is reading.  The two differ in one expression: where a
row of the block sits.  In the graph's own layout that is BBASE + KK*N + COL, a
jump of a whole row of B per pass; in the packed one it is COL*K + KK*C*LANES,
which is the next thing in memory.

Free in the expansion: DA, DB, DP, DR, BBASE, K, N, COL, COL-HI and the
AROWi/OROWi the caller bound."
  (let ((accs (loop for i below r
                    collect (loop for j below c
                                  collect (intern (format nil "ACC~d~d" i j)))))
        (bs (loop for j below c collect (intern (format nil "BV~d" j))))
        (as (loop for i below r collect (intern (format nil "AROW~d" i))))
        (os (loop for i below r collect (intern (format nil "OROW~d" i)))))
    `(loop while (<= (+ col (* ,c +f32-lanes+)) col-hi) do
      (let ((bb (the dim ,(if packed `(* col k) `(+ bbase col))))
            ,@(loop for row in accs append
                    (loop for a in row collect `(,a (f32v-broadcast 0.0)))))
        (declare (type dim bb) (type f32v-pack ,@(reduce #'append accs)))
        (dotimes (kk k)
          (declare (type dim kk))
          (let* ((brow (the dim (+ bb (the dim (* kk ,(if packed `(* ,c +f32-lanes+) 'n))))))
                 ,@(loop for b in bs for j from 0
                         collect `(,b (f32v-ref ,(if packed 'dp 'db)
                                                (the dim (+ brow (* ,j +f32-lanes+)))))))
            (declare (type f32v-pack ,@bs))
            ,@(loop for i below r
                    collect `(let ((xv (f32v-broadcast-ref da (the dim (+ ,(nth i as) kk)))))
                               (declare (type f32v-pack xv))
                               (setf ,@(loop for j below c
                                             append `(,(nth j (nth i accs))
                                                      (f32v+ ,(nth j (nth i accs))
                                                             (f32v* xv ,(nth j bs))))))))))
        ;; the window is written, not added to: the caller left DR zeroed and no
        ;; other block touches these columns, so storing the register is the same
        ;; sum in the same order
        (setf ,@(loop for i below r
                      append (loop for j below c
                                   append `((f32v-ref dr (the dim (+ ,(nth i os) col
                                                                     (* ,j +f32-lanes+))))
                                            ,(nth j (nth i accs))))))
        (incf col (* ,c +f32-lanes+))))))

(defmacro %mm-fast (r packed)
  "R rows over [COL, COL-HI) with no multiplier equal to zero.

Wide blocks first, then one lane group at a time, then the columns that are not
a lane at all — which is also the order MM-PACK lays a packed B out in, so
PACKED changes where B is read and nothing else.  COL-LO is a multiple of
+MM-COLS+ for the same reason: a window that started anywhere else would ask for
blocks the packed B was not cut into.

The fence goes between the vector work and the remainder and nowhere else: the
remainder is scalar code, and so is whatever called this.

Free in the expansion, from the enclosing kernel: AROWS, OROWS, BBASE, DA, DB,
DP, DR, K, N, COL-LO and COL-HI."
  (let ((as (loop for i below r collect (intern (format nil "AROW~d" i))))
        (os (loop for i below r collect (intern (format nil "OROW~d" i)))))
    `(let ((col col-lo)
           ,@(loop for a in as for i from 0 collect `(,a (aref arows ,i)))
           ,@(loop for o in os for i from 0 collect `(,o (aref orows ,i))))
       (declare (type dim col ,@as ,@os))
       (%mm-block ,r ,+mm-lanes+ ,packed)
       (%mm-block ,r 1 ,packed)
       (f32v-done)
       ,(let ((sums (loop for i below r collect (intern (format nil "SUM~d" i)))))
          `(loop while (< col col-hi) do
            (let ,(loop for s in sums collect `(,s 0.0))
              (declare (type single-float ,@sums))
              (dotimes (kk k)
                (declare (type dim kk))
                (let ((bv ,(if packed
                               `(aref dp (the dim (+ (the dim (* col k)) kk)))
                               `(aref db (the dim (+ bbase (the dim (* kk n)) col))))))
                  ,@(loop for i below r
                          collect `(incf ,(nth i sums)
                                         (* (aref da (the dim (+ ,(nth i as) kk))) bv)))))
              (setf ,@(loop for i below r
                            append `((aref dr (the dim (+ ,(nth i os) col))) ,(nth i sums))))
              (incf col)))))))

(defmacro %mm-sparse (r)
  "The kernel for R rows where some multiplier is exactly zero.

A row whose multiplier is zero is skipped rather than multiplied: 0.0 times an
infinity is a NaN where the skip leaves a finite sum, and the gates compare
exactly.  That means a test per row per pass, and a test is a scalar compare, so
this loop keeps K on the outside where all R of them can be read and tested
together — one transition per pass instead of R.  It is three times slower than
%MM-FAST and it runs on 3% of the multiplies.

Free in the expansion, from the enclosing kernel: AROWS, OROWS, BBASE, DA, DB,
DR, K, N, COL-LO, COL-HI and CV-HI."
  (let ((xs (loop for i below r collect (intern (format nil "X~d" i))))
        (as (loop for i below r collect (intern (format nil "AROW~d" i))))
        (os (loop for i below r collect (intern (format nil "OROW~d" i)))))
    (labels ((pass (which)
               ;; the column loop for the rows in WHICH, as a list of indices
               `(do ((col col-lo (+ col +f32-lanes+))) ((>= col cv-hi))
                  (declare (type dim col))
                  (let ((bv (f32v-ref db (the dim (+ brow col)))))
                    ,@(loop for i in which
                            collect `(setf (f32v-ref dr (the dim (+ ,(nth i os) col)))
                                           (f32v+ (f32v-ref dr (the dim (+ ,(nth i os) col)))
                                                  (f32v* ,(intern (format nil "XV~d" i))
                                                         bv)))))))
             (tail (i)
               `(loop for col of-type dim from cv-hi below col-hi
                      do (incf (aref dr (the dim (+ ,(nth i os) col)))
                               (* ,(nth i xs) (aref db (the dim (+ brow col))))))))
      `(let ,(append (loop for a in as for i from 0 collect `(,a (aref arows ,i)))
                     (loop for o in os for i from 0 collect `(,o (aref orows ,i))))
         (declare (type dim ,@as ,@os))
         ;; the whole lanes, all of K.  The fence is out here and not inside the
         ;; K loop: at K=512 one VZEROUPPER per row of B costs more than the
         ;; multiplies do.  Splitting the columns this way is safe for the same
         ;; reason the thread split is — an output element still sums over K in
         ;; K order, whichever pass writes it.
         (dotimes (kk k)
           (declare (type dim kk))
           (let ((brow (the dim (+ bbase (* kk n))))
                 ,@(loop for x in xs for a in as
                         collect `(,x (aref da (the dim (+ ,a kk))))))
             (declare (type dim brow) (type single-float ,@xs))
             (if (and ,@(loop for x in xs collect `(not (zerop ,x))))
                 ;; every row wants this row of B: read it once, fan it out
                 (let ,(loop for x in xs for i from 0
                             collect `(,(intern (format nil "XV~d" i)) (f32v-broadcast ,x)))
                   (declare (type f32v-pack ,@(loop for i below r
                                                    collect (intern (format nil "XV~d" i)))))
                   ,(pass (loop for i below r collect i)))
                 ;; someone's multiplier is exactly zero, so their row is left
                 ;; alone rather than added to
                 (progn
                   ,@(loop for i below r
                           collect `(unless (zerop ,(nth i xs))
                                      (let ((,(intern (format nil "XV~d" i))
                                              (f32v-broadcast ,(nth i xs))))
                                        (declare (type f32v-pack
                                                       ,(intern (format nil "XV~d" i))))
                                        ,(pass (list i)))))))))
         (f32v-done)
         ;; and the columns that are not a whole lane, at width one
         (when (< cv-hi col-hi)
           (dotimes (kk k)
             (declare (type dim kk))
             (let ((brow (the dim (+ bbase (* kk n))))
                   ,@(loop for x in xs for a in as
                           collect `(,x (aref da (the dim (+ ,a kk))))))
               (declare (type dim brow) (type single-float ,@xs))
               ,@(loop for i below r
                       collect `(unless (zerop ,(nth i xs)) ,(tail i))))))))))

(defun matmul-core (da db dp dr ias ibs ios m k n row-lo row-hi col-lo col-hi)
  "Accumulate rows [ROW-LO, ROW-HI) by columns [COL-LO, COL-HI) of a batch of
matrix products into DR, which the caller has left zeroed.  IAS/IBS/IOS are the
per-batch-item element offsets into DA/DB/DR; row R belongs to batch item
(FLOOR R M).

DP is MM-PACKED-B's copy of B or NIL, and COL-LO must be a multiple of +MM-COLS+
when it is not NIL.  It is only ever produced for a single unbatched matrix, so
BBASE is zero wherever it is used.

Two calls may not overlap in the window they name, which is what makes both of
the caller's splits — by row and by column — safe."
  (declare (type (simple-array single-float (*)) da db dr)
           (type (or null (simple-array single-float (*))) dp)
           (type (simple-array fixnum (*)) ias ibs ios)
           (type dim m k n row-lo row-hi col-lo col-hi)
           (optimize (speed 3) (safety 0)))
  (let* ((width (the dim (- col-hi col-lo)))
         (cv-hi (the dim (+ col-lo (- width (mod width +f32-lanes+)))))
         (arows (make-array +mm-rows+ :element-type 'fixnum))
         (orows (make-array +mm-rows+ :element-type 'fixnum))
         (bbase 0)
         (row row-lo))
    (declare (dynamic-extent arows orows) (type dim bbase row))
    (loop while (< row row-hi) do
      ;; how many of the next rows share a B — all of them when B is broadcast
      ;; over the batch (the usual case) or when they are rows of one matrix,
      ;; and one at a time when neither holds
      (let ((rr 1))
        (declare (type dim rr))
        (multiple-value-bind (item r) (floor row m)
          (setf bbase (aref ibs item)
                (aref arows 0) (+ (aref ias item) (* r k))
                (aref orows 0) (+ (aref ios item) (* r n))))
        (loop for i of-type dim from 1 below (min +mm-rows+ (- row-hi row))
              do (multiple-value-bind (item r) (floor (+ row i) m)
                   (unless (= (aref ibs item) bbase) (return))
                   (setf (aref arows i) (+ (aref ias item) (* r k))
                         (aref orows i) (+ (aref ios item) (* r n)))
                   (incf rr)))
        (if (mm-dense-p da arows rr k)
            (if dp
                (let ((dp dp))
                  (declare (type (simple-array single-float (*)) dp))
                  (ecase rr
                    (1 (%mm-fast 1 t))
                    (2 (%mm-fast 2 t))
                    (3 (%mm-fast 3 t))
                    (4 (%mm-fast 4 t))))
                (ecase rr
                  (1 (%mm-fast 1 nil))
                  (2 (%mm-fast 2 nil))
                  (3 (%mm-fast 3 nil))
                  (4 (%mm-fast 4 nil))))
            (ecase rr
              (1 (%mm-sparse 1))
              (2 (%mm-sparse 2))
              (3 (%mm-sparse 3))
              (4 (%mm-sparse 4))))
        (incf row rr))))
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
               (dp (mm-packed-b db k n))
               (dr (the (simple-array single-float (*)) (tensor-data res))))
          (declare (type dim m k n))
          (let* ((nb (max 1 (shape-size batch)))
                 (ias (make-array nb :element-type 'fixnum :initial-element 0))
                 (ibs (make-array nb :element-type 'fixnum :initial-element 0))
                 (ios (make-array nb :element-type 'fixnum :initial-element 0)))
            (do-broadcast ((ia sa) (ib sb) (io so) batch)
              (setf (aref ias i) ia (aref ibs i) ib (aref ios i) io))
            ;; Rows first when there are enough of them to go round, because
            ;; splitting rows keeps each thread's B whole.  Otherwise columns:
            ;; this model's big products have four rows and a thousand columns,
            ;; and a four-way split of a sixteen-thread pool is the difference
            ;; between the kernel being fast and the op being fast.
            (let ((nrows (* nb m)))
              (declare (type dim nrows))
              (if (>= nrows (* 2 *worker-count*))
                  (parallel-range nrows
                                  (lambda (lo hi)
                                    (matmul-core da db dp dr ias ibs ios m k n lo hi 0 n))
                                  :min-chunk (max 1 (ceiling +parallel-grain+
                                                             (max 1 (* k n)))))
                  ;; a whole block per chunk, so no thread owns half a vector or
                  ;; starts in the middle of one of the packed B's blocks
                  (let ((tiles (ceiling n +mm-cols+)))
                    (parallel-range tiles
                                    (lambda (lo hi)
                                      (matmul-core da db dp dr ias ibs ios m k n 0 nrows
                                                   (* lo +mm-cols+)
                                                   (min n (* hi +mm-cols+))))
                                    :min-chunk (max 1 (ceiling +parallel-grain+
                                                               (max 1 (* nrows k
                                                                         +mm-cols+)))))))))
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

(defvar *gemm-transposes*
  (make-hash-table :test 'eq :weakness :key :synchronized t)
  "A weight's data vector to the transposed tensor Gemm needs of it.")

(defun transposed-operand (a)
  "TRANSPOSE-2D of A, kept if A is a weight.

transB is how a fully-connected layer arrives, so the matrix being transposed is
almost always the layer's weights, and transposing them again for every frame of
audio costs twice: once for the copy, and once more because a fresh tensor every
call is a data vector MM-PACKED-B has never seen and will never pack.  The
joiner's 500x512 projection was 8.4% of this model's run on those two counts
alone.

Keyed on the vector, so the shape is checked rather than assumed: a reshape of a
weight shares its data and wants a different transpose."
  (let* ((d (tensor-data a))
         (want (vector (aref (tensor-shape a) 1) (aref (tensor-shape a) 0))))
    (if (constant-data-p d)
        (let ((hit (gethash d *gemm-transposes*)))
          (if (and hit (equalp (tensor-shape hit) want))
              hit
              (setf (gethash d *gemm-transposes*) (note-constant (transpose-2d a)))))
        (transpose-2d a))))

(defop "Gemm" (node ins)
  ;; alpha * A' * B' + beta * C, where ' is an optional transpose and C
  ;; broadcasts over the result.
  (destructuring-bind (a b &optional c) ins
    (let* ((alpha (node-attr node "alpha" 1))
           (beta (node-attr node "beta" 1))
           (a (if (eql 1 (node-attr node "transA" 0)) (transposed-operand a) a))
           (b (if (eql 1 (node-attr node "transB" 0)) (transposed-operand b) b))
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
