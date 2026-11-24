;;;; ops-math.lisp — elementwise arithmetic, comparison, and activations.
;;;;
;;;; Every binary op here broadcasts, and broadcasting is done with strides of
;;;; zero rather than by materializing the stretched operand: a (1,1,30) mask
;;;; against a (1,192,30) tensor should not cost a 192x copy.
;;;;
;;;; The dtype dispatch is a macro rather than a generic function because the
;;;; whole point is that the inner loop sees a SIMPLE-ARRAY of a known element
;;;; type.  Without that SBCL boxes every float it touches, and this graph
;;;; touches a few hundred million of them per second of speech.
;;;;
;;;; Float32 gets a second, faster road through all of this — typed cores split
;;;; out of the ops, run over ranges of output elements by the worker pool — for
;;;; the reason ops-nn.lisp gives: the op keeps normal safety for its shape work
;;;; while the loop runs at (speed 3) (safety 0) with every index a DIM.  The
;;;; other dtypes stay on the general road, because in this graph they are
;;;; scalars and one-element shapes: the i64 arithmetic is all sequence lengths.

(in-package #:mill)

(deftype f32v () '(simple-array single-float (*)))

;;; ---- how big a chunk is worth a thread -------------------------------------
;;;
;;; PARALLEL-RANGE splits on output elements here, not on rows, so MIN-CHUNK is
;;; "how many elements still pay for waking a worker".  Two numbers, because an
;;; elementwise op is either memory-bound or transcendental and there is an order
;;; of magnitude between them.

(defconstant +flat-grain+ 65536
  "Output elements below which a memory-bound elementwise node is not worth
splitting.  An Add moves 12 bytes and costs a couple of nanoseconds an element,
so this is on the order of 100 us of work — a few dozen times what waking a
worker and joining it costs.")

(defconstant +heavy-grain+ 4096
  "The same number for an op whose element is a division or a transcendental
(Div, Pow, Erf, Tanh, Sigmoid).  Those run 10-50x an Add per element, so a much
smaller chunk still dwarfs the join.")

(defun make-f32-tensor-unfilled (shape)
  "A float32 tensor whose data has NOT been zeroed.  MAKE-TENSOR fills, and for
the decoder's (1,32,87808) Add outputs that fill is an 11 MB write pass thrown
away by the very next loop — 1.5% of a synthesis sat in
UB32-BASH-FILL-WITH-SINGLE-FLOAT doing it.  Only for kernels that assign every
element; anything that accumulates into its output still needs MAKE-TENSOR."
  (let ((n (shape-size shape)))
    (%make-tensor :f32 (coerce shape 'simple-vector)
                  (or (pool-take :f32 n)
                      (make-array n :element-type 'single-float)))))

;;; ---- broadcasting without an odometer --------------------------------------

(defun tile-pattern (in-shape out-shape)
  "How to read a tensor of IN-SHAPE while walking OUT-SHAPE's index space, as a
pair (P Q) meaning \"output element I reads input element (MOD (FLOOR I Q) P)\".
NIL when the broadcast is not of that form.

Every broadcast this graph performs is some leading axes stretched, then a
contiguous run of real axes, then some trailing axes stretched — a bias vector
(192) against (1,192,192), a per-frame scale (1,1,343) against (1,192,343), a
per-row mean (1,162,1) against (1,162,192), a scalar against anything.  All of
those are two counters.  DO-BROADCAST instead walks a fixnum odometer array per
element, which is where GENERIC-+, HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS and the
two OPTIMIZED-DATA-VECTOR-REFs in the profile were coming from: about 5% of the
whole run spent deciding which element to read next.

The derivation, for the reader checking it: over the live axes J1..J2-1 the
index is SUM coord_k * stride_k, the strides there are the output's own strides
divided by SP[J2], and SUM coord_k * ostride_k over a contiguous run is just
I mod SP[J1] minus I mod SP[J2].  Hence P = SP[J1]/SP[J2] and Q = SP[J2]."
  (let ((strides (broadcast-strides in-shape out-shape))
        (dims '()) (ss '()))
    ;; An axis of extent 1 never moves an index, so its stride is don't-care.
    ;; Dropping those first is what turns (1,2,162) read through (2,1) — whose
    ;; leading zero stride is an artifact of the leading 1 — into a plain block.
    (loop for k from (1- (length out-shape)) downto 0
          when (> (aref out-shape k) 1)
            do (push (aref out-shape k) dims)
               (push (aref strides k) ss))
    (let* ((m (length dims))
           (d (coerce dims 'simple-vector))
           (s (coerce ss 'simple-vector))
           (sp (make-array (1+ m))))       ; SP[k] = product of D[k..m-1]
      (setf (aref sp m) 1)
      (loop for k from (1- m) downto 0
            do (setf (aref sp k) (* (aref sp (1+ k)) (aref d k))))
      (let* ((j1 (or (position-if-not #'zerop s) m))
             (j2 (let ((p (position-if-not #'zerop s :from-end t)))
                   (if p (1+ p) m)))
             (q (aref sp j2)))
        ;; a zero anywhere inside the live run means the stretched axes are not
        ;; all at one end or the other, and two counters cannot express it
        (loop for k from j1 below j2
              unless (eql (aref s k) (floor (aref sp (1+ k)) q))
                do (return-from tile-pattern nil))
        (values (floor (aref sp j1) q) q)))))

;;; ---- binary ----------------------------------------------------------------

(defmacro def-f32-cores (name (xa xb) form &optional vec-form)
  "Three typed float32 cores for one elementwise op, split out of the op the same
way CONV1D-CORE is: (speed 3) (safety 0) with every index a DIM, while the op
around them keeps normal safety for the shape checks.

Each takes a range of OUTPUT elements, because that is what the worker pool
splits on.  An elementwise result does not depend on how the range was cut and
nothing is summed, so unlike the convolutions there is no ordering to preserve —
a split is bit-for-bit the unsplit loop.

FLAT is both operands walked by the output's own index — no broadcast at all,
and no counters to carry.  SCALAR is that with a one-element right operand
hoisted out of the loop, which is what Pow and the decoder's Divs are.  TILED is
the general (P Q) form; its counters are seeded from LO rather than 0 so that a
chunk starts reading where the whole loop would have.

VEC-FORM, when the op has one, is FORM in the vector vocabulary of simd.lisp,
and it vectorizes FLAT and SCALAR.  It is bit-for-bit FORM because a lane of a
packed add, subtract, multiply or divide is the IEEE operation the scalar
instruction does — the same rounding on the same two floats — and the lanes are
independent, so nothing is reassociated.  TILED is left scalar on purpose: its
two counters carry, which is sequential by construction, and it is the shape
that shows up on small tensors where the transition fence would cost more than
the loop."
  (let ((flat (intern (format nil "~:@(~a~)-FLAT-F32" name)))
        (scalar (intern (format nil "~:@(~a~)-SCALAR-F32" name)))
        (tiled (intern (format nil "~:@(~a~)-TILED-F32" name)))
        (n (gensym "N")) (iv (gensym "IV")) (is (gensym "IS")) (bv (gensym "BV")))
    `(progn
       (defun ,flat (da db dr lo hi)
         (declare (type f32v da db dr) (type dim lo hi)
                  (optimize (speed 3) (safety 0)))
         ,(if vec-form
              `(let ((,n (- hi lo)))
                 (declare (type dim ,n))
                 (do-vectorized (,iv ,is ,n)
                   (let ((,xa (f32v-ref da (+ lo ,iv))) (,xb (f32v-ref db (+ lo ,iv))))
                     (declare (ignorable ,xa ,xb))
                     (setf (f32v-ref dr (+ lo ,iv)) ,vec-form))
                   (let ((,xa (aref da (+ lo ,is))) (,xb (aref db (+ lo ,is))))
                     (declare (ignorable ,xa ,xb))
                     (setf (aref dr (+ lo ,is)) ,form))))
              `(loop for i of-type dim from lo below hi
                     do (let ((,xa (aref da i)) (,xb (aref db i)))
                          (declare (ignorable ,xa ,xb))
                          (setf (aref dr i) ,form))))
         (values))
       (defun ,scalar (da db dr lo hi)
         (declare (type f32v da db dr) (type dim lo hi)
                  (optimize (speed 3) (safety 0)))
         ,(if vec-form
              `(let* ((,xb (aref db 0))
                      (,n (- hi lo))
                      (,bv (f32v-broadcast ,xb)))
                 (declare (ignorable ,xb ,bv) (type dim ,n))
                 (do-vectorized (,iv ,is ,n)
                   (let ((,xa (f32v-ref da (+ lo ,iv))) (,xb ,bv))
                     (declare (ignorable ,xa ,xb))
                     (setf (f32v-ref dr (+ lo ,iv)) ,vec-form))
                   (let ((,xa (aref da (+ lo ,is))))
                     (declare (ignorable ,xa))
                     (setf (aref dr (+ lo ,is)) ,form))))
              `(let ((,xb (aref db 0)))
                 (declare (ignorable ,xb))
                 (loop for i of-type dim from lo below hi
                       do (let ((,xa (aref da i)))
                            (declare (ignorable ,xa))
                            (setf (aref dr i) ,form)))))
         (values))
       (defun ,tiled (da db dr lo hi pa qa pb qb)
         (declare (type f32v da db dr)
                  (type dim lo hi) (type (integer 1 #.(1- (expt 2 30))) pa qa pb qb)
                  (optimize (speed 3) (safety 0)))
         ;; KA counts to QA and then steps CA, which wraps at PA — the two
         ;; counters that TILE-PATTERN found, unrolled into the loop so that no
         ;; element pays for a division or an array of coordinates.
         (let ((ca (mod (floor lo qa) pa)) (ka (mod lo qa))
               (cb (mod (floor lo qb) pb)) (kb (mod lo qb)))
           (declare (type dim ca ka cb kb))
           (loop for i of-type dim from lo below hi
                 do (let ((,xa (aref da ca)) (,xb (aref db cb)))
                      (declare (ignorable ,xa ,xb))
                      (setf (aref dr i) ,form))
                    (incf ka)
                    (when (= ka qa)
                      (setf ka 0) (incf ca) (when (= ca pa) (setf ca 0)))
                    (incf kb)
                    (when (= kb qb)
                      (setf kb 0) (incf cb) (when (= cb pb) (setf cb 0)))))
         (values)))))

(defmacro def-binop (name (xa xb) &key form int-form f32-form vec-form (out :same)
                                       (dtypes '(:f32 :f64 :i64 :i32))
                                       (grain '+flat-grain+))
  "Define an elementwise binary op.  FORM computes the result from XA and XB;
INT-FORM replaces it for integer dtypes when the two differ (Div is the reason:
ONNX integer division truncates), and F32-FORM replaces it for float32 when a
cheaper expression is exact there (Pow is the reason).  VEC-FORM is the float32
expression in simd.lisp's vocabulary, for the ops that have one — see
DEF-F32-CORES for which cores it reaches and why it changes no result.  OUT is
:SAME or :BOOL.

A :SAME op over float32 gets the typed cores and the worker pool; everything
else keeps the general road below.  GRAIN is how many output elements are worth
waking a thread for, which the caller knows and PARALLEL-RANGE does not."
  (let* ((fastp (and (eq out :same) (member :f32 dtypes)))
         (f32-body (or f32-form form))
         (flat (intern (format nil "~:@(~a~)-FLAT-F32" name)))
         (scalar (intern (format nil "~:@(~a~)-SCALAR-F32" name)))
         (tiled (intern (format nil "~:@(~a~)-TILED-F32" name))))
    (flet ((general (dt)
             ;; the road every dtype took before float32 got its own: allocate,
             ;; then either walk all three tensors by one index or run the
             ;; odometer
             (let* ((et (dtype-element-type dt))
                    (ot (if (eq out :bool) '(unsigned-byte 8) et))
                    (body (cond ((and int-form (member dt '(:i64 :i32 :bool))) int-form)
                                ((eq dt :f32) f32-body)
                                (t form))))
               `(let ((res (make-tensor ,(if (eq out :bool) :bool dt) shape))
                      (da (the (simple-array ,et (*)) (tensor-data a)))
                      (db (the (simple-array ,et (*)) (tensor-data b))))
                  (let ((dr (the (simple-array ,ot (*)) (tensor-data res))))
                    (if (and (equalp (tensor-shape a) shape)
                             (equalp (tensor-shape b) shape))
                        ;; Nothing is being stretched, so all three tensors are
                        ;; walked by the same index and the odometer is pure
                        ;; overhead — which it was: an Add cost 20 ns an element
                        ;; with it, and this graph's Adds alone were 27% of a
                        ;; synthesis.
                        (locally (declare (optimize (speed 3) (safety 0)))
                          (dotimes (i (length dr))
                            (let ((,xa (aref da i)) (,xb (aref db i)))
                              (declare (ignorable ,xa ,xb))
                              (setf (aref dr i) ,body))))
                        (do-broadcast ((ia sa) (ib sb) shape)
                          (let ((,xa (aref da ia)) (,xb (aref db ib)))
                            (declare (ignorable ,xa ,xb))
                            (setf (aref dr i) ,body)))))
                  res))))
      `(progn
         ,@(when fastp `((def-f32-cores ,name (,xa ,xb) ,f32-body ,vec-form)))
         (defop ,name (node ins)
           (let* ((a (first ins)) (b (second ins))
                  (shape (broadcast-shapes (tensor-shape a) (tensor-shape b)))
                  (sa (broadcast-strides (tensor-shape a) shape))
                  (sb (broadcast-strides (tensor-shape b) shape)))
             (declare (ignorable sa sb))
             (unless (eq (tensor-dtype a) (tensor-dtype b))
               (error "~a: operands differ in dtype (~a vs ~a)" ,name
                      (tensor-dtype a) (tensor-dtype b)))
             (ecase (tensor-dtype a)
               ,@(loop for dt in dtypes
                       collect
                       (if (and fastp (eq dt :f32))
                           `(:f32
                             (let ((n (shape-size shape)))
                               (multiple-value-bind (pa qa)
                                   (tile-pattern (tensor-shape a) shape)
                                 (multiple-value-bind (pb qb)
                                     (tile-pattern (tensor-shape b) shape)
                                   ;; DIM is what the cores index with; a tensor
                                   ;; too big for it would silently wrap, so it
                                   ;; goes down the general road instead
                                   (if (and pa pb (typep n 'dim))
                                       (let* ((res (make-f32-tensor-unfilled shape))
                                              (da (the f32v (tensor-data a)))
                                              (db (the f32v (tensor-data b)))
                                              (dr (the f32v (tensor-data res))))
                                         (parallel-range
                                          n
                                          (cond ((and (= qa 1) (= pa n)
                                                      (= qb 1) (= pb n))
                                                 (lambda (lo hi) (,flat da db dr lo hi)))
                                                ((and (= qa 1) (= pa n) (= pb 1))
                                                 (lambda (lo hi) (,scalar da db dr lo hi)))
                                                (t
                                                 (lambda (lo hi)
                                                   (,tiled da db dr lo hi pa qa pb qb))))
                                          :min-chunk ,grain)
                                         res)
                                       ,(general :f32))))))
                           `(,dt ,(general dt)))))))))))

(defun pow-f (base exponent)
  "ONNX Pow on floats.  Computed in double and rounded back, which is at least as
close to the true value as a float32 pow; a negative base with a fractional
exponent is NaN, not a complex number."
  (let ((b (float base 1d0)) (e (float exponent 1d0)))
    (cond ((and (minusp b) (/= e (ffloor e)))
           (sb-kernel:make-double-float #x7FF80000 0))
          (t (expt b e)))))

(declaim (inline pow-f32))
(defun pow-f32 (b e)
  "Pow on float32.  Every one of this graph's 39 Pow nodes raises to a scalar
exponent of 2.0 — squaring, dressed up as a call that went out to libm's pow
through two boxed doubles and cost 2.8 ms a node.

The special case is not an approximation of the general one, it is the same
number: a float32 square is exact in a double (24 bits squared is 48, and a
double carries 53), so rounding that exact double back to single rounds once,
which is what (* B B) in single does.  Overflow, underflow to a subnormal and
negative bases all follow from that being a single rounding either way."
  (declare (type single-float b e))
  (if (= e 2.0f0)
      (* b b)
      (float (pow-f b e) b)))

(def-binop "Add" (x y) :form (+ x y) :vec-form (f32v+ x y)
                       :dtypes (:f32 :f64 :i64 :i32))
(def-binop "Sub" (x y) :form (- x y) :vec-form (f32v- x y)
                       :dtypes (:f32 :f64 :i64 :i32))
(def-binop "Mul" (x y) :form (* x y) :vec-form (f32v* x y)
                       :dtypes (:f32 :f64 :i64 :i32))
(def-binop "Div" (x y) :form (/ x y) :vec-form (f32v/ x y)
                       :int-form (values (truncate x y))
                       :dtypes (:f32 :f64 :i64 :i32) :grain +heavy-grain+)
(def-binop "Pow" (x y) :form (float (pow-f x y) x) :f32-form (pow-f32 x y)
                       :int-form (expt x y)
                       :dtypes (:f32 :f64 :i64 :i32) :grain +heavy-grain+)

(def-binop "Equal" (x y) :form (if (= x y) 1 0) :out :bool
                         :dtypes (:f32 :f64 :i64 :i32 :bool))
(def-binop "Less" (x y) :form (if (< x y) 1 0) :out :bool)
(def-binop "LessOrEqual" (x y) :form (if (<= x y) 1 0) :out :bool)
(def-binop "Greater" (x y) :form (if (> x y) 1 0) :out :bool)
(def-binop "GreaterOrEqual" (x y) :form (if (>= x y) 1 0) :out :bool)
(def-binop "And" (x y) :form (if (and (/= x 0) (/= y 0)) 1 0) :dtypes (:bool))
(def-binop "Or" (x y) :form (if (or (/= x 0) (/= y 0)) 1 0) :dtypes (:bool))

;;; ---- selection -------------------------------------------------------------

(defop "Where" (node ins)
  ;; All three operands broadcast against each other, which this graph relies on:
  ;; the condition is usually a (1,1,T) mask and the branches are full tensors.
  (destructuring-bind (cond-t a b) ins
    (let* ((shape (broadcast-shapes (broadcast-shapes (tensor-shape cond-t)
                                                      (tensor-shape a))
                                    (tensor-shape b)))
           (sc (broadcast-strides (tensor-shape cond-t) shape))
           (sa (broadcast-strides (tensor-shape a) shape))
           (sb (broadcast-strides (tensor-shape b) shape))
           (dc (the (simple-array (unsigned-byte 8) (*)) (tensor-data cond-t))))
      (unless (eq (tensor-dtype a) (tensor-dtype b))
        (error "Where: branches differ in dtype (~a vs ~a)"
               (tensor-dtype a) (tensor-dtype b)))
      (macrolet ((dispatch ()
                   `(ecase (tensor-dtype a)
                      ,@(loop for dt in '(:f32 :f64 :i64 :i32 :bool)
                              for et = (dtype-element-type dt)
                              collect
                              `(,dt (let ((res (make-tensor ,dt shape))
                                          (da (the (simple-array ,et (*)) (tensor-data a)))
                                          (db (the (simple-array ,et (*)) (tensor-data b))))
                                      (let ((dr (the (simple-array ,et (*)) (tensor-data res))))
                                        (do-broadcast ((ic sc) (ia sa) (ib sb) shape)
                                          (setf (aref dr i)
                                                (if (zerop (aref dc ic))
                                                    (aref db ib)
                                                    (aref da ia)))))
                                      res))))))
        (dispatch)))))

;;; ---- unary -----------------------------------------------------------------

(defmacro def-unop (name (x) &key form int-form f32-form (dtypes '(:f32 :f64))
                                  (grain '+flat-grain+))
  "Define an elementwise unary op.  Same shape as DEF-BINOP — including the
float32 fast road, which for a unary op needs no broadcast analysis at all: the
output is the input's shape, so the core is always the flat one."
  (let* ((fastp (member :f32 dtypes))
         (core (intern (format nil "~:@(~a~)-CORE-F32" name)))
         (f32-body (or f32-form form)))
    `(progn
       ,@(when fastp
           `((defun ,core (da dr lo hi)
               (declare (type f32v da dr) (type dim lo hi)
                        (optimize (speed 3) (safety 0)))
               (loop for i of-type dim from lo below hi
                     do (let ((,x (aref da i)))
                          (declare (ignorable ,x))
                          (setf (aref dr i) ,f32-body)))
               (values))))
       (defop ,name (node ins)
         (let ((a (first ins)))
           (ecase (tensor-dtype a)
             ,@(loop for dt in dtypes
                     for et = (dtype-element-type dt)
                     for body = (if (and int-form (member dt '(:i64 :i32 :bool)))
                                    int-form
                                    (if (eq dt :f32) f32-body form))
                     collect
                     (if (and fastp (eq dt :f32))
                         `(:f32
                           (let ((n (tensor-size a)))
                             (if (typep n 'dim)
                                 (let* ((res (make-f32-tensor-unfilled (tensor-shape a)))
                                        (da (the f32v (tensor-data a)))
                                        (dr (the f32v (tensor-data res))))
                                   (parallel-range n
                                                   (lambda (lo hi) (,core da dr lo hi))
                                                   :min-chunk ,grain)
                                   res)
                                 (let ((res (make-tensor :f32 (tensor-shape a)))
                                       (da (the f32v (tensor-data a))))
                                   (let ((dr (the f32v (tensor-data res))))
                                     (locally (declare (optimize (speed 3) (safety 0)))
                                       (dotimes (i (length da))
                                         (let ((,x (aref da i)))
                                           (declare (ignorable ,x))
                                           (setf (aref dr i) ,f32-body)))))
                                   res))))
                         `(,dt (let ((res (make-tensor ,dt (tensor-shape a)))
                                     (da (the (simple-array ,et (*)) (tensor-data a))))
                                 (let ((dr (the (simple-array ,et (*)) (tensor-data res))))
                                   (locally (declare (optimize (speed 3) (safety 0)))
                                     (dotimes (i (length da))
                                       (let ((,x (aref da i)))
                                         (declare (ignorable ,x))
                                         (setf (aref dr i) ,body)))))
                                 res)))))))
       ,name)))

;;; erf has no CL equivalent and GELU is 24 nodes of this graph, so it is worth
;;; getting right rather than approximating.  This is the confluent
;;; hypergeometric series
;;;
;;;     erf(x) = 2x/sqrt(pi) * e^(-x^2) * SUM (2x^2)^n / (1*3*5*...*(2n+1))
;;;
;;; whose terms are all positive — nothing cancels, so the relative error stays
;;; near machine epsilon instead of degrading the way the alternating Taylor
;;; series does.  Past |x|=6 the answer is +-1 to well under a double's
;;; resolution, let alone a float's.
(defconstant +2/sqrt-pi+ 1.1283791670955126d0)

(defun erf-d (x)
  (declare (type double-float x))
  (let ((ax (abs x)))
    (cond ((> ax 6d0) (float-sign x 1d0))
          ((zerop ax) x)
          (t (let* ((x2 (* ax ax))
                    (term 1d0)
                    (sum 1d0))
               (declare (type double-float term sum))
               (loop for n of-type fixnum from 1 below 300
                     do (setf term (/ (* term 2d0 x2) (+ (* 2d0 n) 1d0)))
                        (incf sum term)
                     while (> term (* 1d-18 sum)))
               (float-sign x (* +2/sqrt-pi+ ax (exp (- x2)) sum)))))))

;;; ...and it is worth getting right in double, which is the dtype the series is
;;; for.  Every Erf in THIS graph is float32, where the series' last ten digits
;;; are thrown away by the coercion back to single and the ~35 divisions it takes
;;; to produce them are not: it measured 210 ns a call, and Erf was 3% of a
;;; synthesis.  Abramowitz & Stegun 7.1.26 is one reciprocal, a degree-5 Horner
;;; and one exp, and its error is bounded at 1.5e-7 — a third of a float32 ulp at
;;; 1.0, and three orders under the node gate's 1e-4.  Measured against the
;;; series over [-6,6] at 1e-5 spacing the worst disagreement is 1.39e-7, or
;;; 1.79e-7 after both are rounded to single.
;;;
;;; The trap this avoids is coercion: ERF-APPROX is inlined so the core loop
;;; keeps the double unboxed.  Left as a non-inline call it costs more in boxing
;;; than the arithmetic saves.
(defconstant +erf-r+ 0.3275911d0)
(defconstant +erf-c1+ 0.254829592d0)
(defconstant +erf-c2+ -0.284496736d0)
(defconstant +erf-c3+ 1.421413741d0)
(defconstant +erf-c4+ -1.453152027d0)
(defconstant +erf-c5+ 1.061405429d0)

(declaim (inline erf-approx))
(defun erf-approx (x)
  (declare (type double-float x) (optimize (speed 3) (safety 0)))
  (let ((ax (abs x)))
    (if (> ax 6d0)
        (float-sign x 1d0)
        (let* ((v (/ 1d0 (+ 1d0 (* +erf-r+ ax))))
               (poly (* v (+ +erf-c1+
                             (* v (+ +erf-c2+
                                     (* v (+ +erf-c3+
                                             (* v (+ +erf-c4+
                                                     (* v +erf-c5+)))))))))))
          (float-sign x (- 1d0 (* poly (exp (- (* ax ax))))))))))

(declaim (inline sigmoid-d softplus-d))
(defun sigmoid-d (x) (/ 1d0 (+ 1d0 (exp (- (float x 1d0))))))
(defun softplus-d (x)
  ;; ln(1+e^x) overflows for large x and is just x there; the branch keeps the
  ;; big-input case exact instead of infinite.
  (let ((x (float x 1d0)))
    (if (> x 30d0) x (log (+ 1d0 (exp x))))))

(def-unop "Neg" (x) :form (- x) :dtypes (:f32 :f64 :i64 :i32))
(def-unop "Abs" (x) :form (abs x) :dtypes (:f32 :f64 :i64 :i32))
(def-unop "Sqrt" (x) :form (if (minusp x) (/ (- x x) (- x x)) (sqrt x))
                     :grain +heavy-grain+)
(def-unop "Exp" (x) :form (exp x) :grain +heavy-grain+)
(def-unop "Log" (x) :form (log x) :grain +heavy-grain+)
(def-unop "Ceil" (x) :form (fceiling x))
(def-unop "Floor" (x) :form (ffloor x))
(def-unop "Tanh" (x) :form (tanh x) :grain +heavy-grain+)
(def-unop "Relu" (x) :form (max x (- x x)) :int-form (max x 0) :dtypes (:f32 :f64 :i64 :i32))
(def-unop "Erf" (x) :form (float (erf-d (float x 1d0)) x)
                    :f32-form (float (erf-approx (float x 1d0)) x)
                    :grain +heavy-grain+)
(def-unop "Sigmoid" (x) :form (float (sigmoid-d x) x) :grain +heavy-grain+)
(def-unop "Softplus" (x) :form (float (softplus-d x) x) :grain +heavy-grain+)
(def-unop "Not" (x) :form (if (zerop x) 1 0) :int-form (if (zerop x) 1 0) :dtypes (:bool))

;;; Trigonometry.  Kokoro's duration predictor builds its positional encoding out of
;;; Sin and Cos, and ROUND appears once, turning predicted durations into frame counts.
;;; ONNX rounds halves to EVEN, which is exactly what CL's FROUND does — the naive
;;; "add a half and truncate" would bias every .5 upward and drift the alignment.
(def-unop "Sin" (x) :form (sin x) :grain +heavy-grain+)
(def-unop "Cos" (x) :form (cos x) :grain +heavy-grain+)
(def-unop "Atan" (x) :form (atan x) :grain +heavy-grain+)
(def-unop "Round" (x) :form (fround x))

;;; LeakyRelu is not a DEF-UNOP because its slope is a node attribute rather than
;;; a constant of the op.  It gets the same treatment by hand: 16 nodes, but they
;;; are the decoder's, so it moves 22.9 M elements — more than everything else in
;;; this file except Add.
(defun leaky-relu-core-f32 (da dr alpha lo hi)
  (declare (type f32v da dr) (type single-float alpha) (type dim lo hi)
           (optimize (speed 3) (safety 0)))
  (loop for i of-type dim from lo below hi
        do (let ((x (aref da i)))
             (setf (aref dr i) (if (minusp x) (* alpha x) x))))
  (values))

(defop "LeakyRelu" (node ins)
  (let* ((a (first ins))
         (alpha (float (node-attr node "alpha" 0.01d0) 1.0f0))
         (n (tensor-size a)))
    (if (typep n 'dim)
        (let* ((res (make-f32-tensor-unfilled (tensor-shape a)))
               (da (the f32v (tensor-data a)))
               (dr (the f32v (tensor-data res))))
          (parallel-range n
                          (lambda (lo hi) (leaky-relu-core-f32 da dr alpha lo hi))
                          :min-chunk +flat-grain+)
          res)
        (let* ((res (make-tensor (tensor-dtype a) (tensor-shape a)))
               (da (the f32v (tensor-data a)))
               (dr (the f32v (tensor-data res))))
          (dotimes (i (length da))
            (let ((x (aref da i)))
              (setf (aref dr i) (if (minusp x) (* alpha x) x))))
          res))))

(defop "Clip" (node ins)
  (destructuring-bind (a &optional lo hi) ins
    (let* ((res (make-tensor (tensor-dtype a) (tensor-shape a))))
      (with-two-typed-data (da a) (dr res)
        (let ((lo (and lo (plusp (tensor-size lo)) (tensor-scalar lo)))
              (hi (and hi (plusp (tensor-size hi)) (tensor-scalar hi))))
          (dotimes (i (length da))
            (let ((x (aref da i)))
              (when (and lo (< x lo)) (setf x lo))
              (when (and hi (> x hi)) (setf x hi))
              (setf (aref dr i) x)))))
      res)))
