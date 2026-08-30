;;;; tensor.lisp — the one data structure everything else moves around.
;;;;
;;;; A tensor is a flat typed vector plus a shape, row-major (C order), which is
;;;; the layout ONNX uses on the wire.  Keeping the data flat and the shape
;;;; separate means a reshape is free, a transpose is an explicit copy, and every
;;;; kernel can be written as a loop over a linear index — which is the shape a
;;;; SIMD version needs to be in later.  There is no stride field on purpose:
;;;; strided views make reshape cheap and every other kernel harder, and this
;;;; engine does far more arithmetic than it does reshaping.

(in-package #:mill)

(deftype index () '(unsigned-byte 62))

;; An extent, a coordinate, or an offset inside one tensor — as opposed to INDEX,
;; which is only bounded by what an array can hold.  The point of the narrower
;; type is headroom: a product of two DIMs is 60 bits and a sum of several such
;; products is still under the 63-bit fixnum, so a kernel that composes an offset
;; out of channel counts and lengths gets fixnum arithmetic throughout instead of
;; a generic call per term.  Nothing in a speech model comes within three orders
;; of magnitude of 2^30 — the largest tensor this model builds is a few million
;; elements.  A product of THREE dims does overflow, so kernels hoist the
;; plane-size products into their own DIM bindings rather than writing them out.
(deftype dim () '(unsigned-byte 30))

;; No struct copier: the one DEFSTRUCT would write shares the data vector, and a
;; "copy" that aliases its original is exactly the bug ScatterND would hit.
;; COPY-TENSOR in ops-shape.lisp copies the data too.
(defstruct (tensor (:constructor %make-tensor (dtype shape data)) (:copier nil))
  "DTYPE is one of :f32 :f64 :i64 :i32 :bool.  SHAPE is a fixnum vector, outermost
dimension first.  DATA is a flat SIMPLE-ARRAY of the matching element type, in
row-major order."
  (dtype :f32 :type keyword)
  (shape #() :type simple-vector)
  (data nil))

(declaim (inline dtype-element-type))
(defun dtype-element-type (dtype)
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    (:i64 '(signed-byte 64))
    (:i32 '(signed-byte 32))
    ;; ONNX bools are one byte each, and arithmetic on them (And, Not, Where's
    ;; condition) is easier on octets than on a bit vector.
    (:bool '(unsigned-byte 8))))

(declaim (inline dtype-zero))
(defun dtype-zero (dtype)
  (ecase dtype (:f32 0.0f0) (:f64 0.0d0) ((:i64 :i32 :bool) 0)))

(declaim (inline dtype-element-bytes))
(defun dtype-element-bytes (dtype)
  (ecase dtype (:f32 4) (:f64 8) (:i64 8) (:i32 4) (:bool 1)))

(defun shape-size (shape)
  (let ((n 1)) (declare (type index n))
    (loop for d across shape do (setf n (* n (the index d))))
    n))

;;; ---- reusing data vectors --------------------------------------------------
;;;
;;; A synthesis allocates ~240 MB of intermediates and collects nearly all of it,
;;; and the collection is not what costs: gencgc drops a generation that survived
;;; nothing almost for free.  What costs is the other end — a multi-megabyte
;;; vector is a large object, so every one of them is fresh pages the kernel has
;;; to map and the runtime has to zero, and the profile put ~9% of a run in
;;; ALLOCATE-VECTOR-WITH-WIDETAG and gc_alloc_large underneath it.
;;;
;;; RUN-MODEL already knows the moment a value is dead, because it computed every
;;; value's last reader before starting.  Binding *TENSOR-POOL* hands those dead
;;; vectors back here instead of to the collector, and MAKE-TENSOR takes one
;;; rather than asking for pages that are already warm in this process.  Outside
;;; RUN-MODEL the variable is NIL and none of this exists, which is what keeps it
;;; invisible to the gates and to any other caller.

(defvar *tensor-pool* nil
  "EQUAL hash of (dtype . length) -> a list of free data vectors, or NIL.
Only RUN-MODEL binds it, and only when it is dropping dead values at all.")

(defvar *tensor-pool-bytes* 0
  "How much the pool is holding, so it cannot grow into the retain-everything
memory profile the drop logic exists to avoid.")

(defparameter *tensor-pool-budget* (* 48 1024 1024)
  "Bytes the pool may hold on to.  Past this a dead vector is simply dropped —
the point is to recycle the working set, not to keep the whole graph alive.")

(defvar *const-data*
  (make-hash-table :test 'eq :weakness :key :synchronized t)
  "The data vectors that never change, as a set.

An op that wants to keep a derived copy of an operand — a transposed one, a
repacked one — needs to know that the operand will still say the same thing next
time, and in this engine that is not a property of a tensor.  It is a property of
the vector underneath, because POOL-TAKE hands the same vector to a new tensor
with new contents as soon as the old one is dropped.  A pack cached against a
recycled activation would be right once and wrong afterwards, which is the kind
of wrong that survives a gate.

So the rule is narrow: a vector is in here because LOAD-MODEL read it out of the
.bin, or because it is a Constant node's attribute, which is the same thing said
in the graph instead of the initializer table.  Weak on the key, so the set holds
nothing alive and a model that is dropped takes its entries with it.")

(defun note-constant (tensor)
  "Record TENSOR's data vector as never changing.  For initializers and for the
attributes of Constant nodes, which is all the model owns."
  (when (and (tensor-p tensor) (tensor-data tensor))
    (setf (gethash (tensor-data tensor) *const-data*) t))
  tensor)

(declaim (inline constant-data-p))
(defun constant-data-p (data)
  (values (gethash data *const-data*)))

(defun pool-take (dtype n)
  "A free data vector of N elements of DTYPE, or NIL.  Its contents are whatever
the last owner left; every caller either fills it or assigns every element."
  (let ((tbl *tensor-pool*))
    (when tbl
      (let* ((key (cons dtype n))
             (free (gethash key tbl)))
        (when free
          (setf (gethash key tbl) (cdr free))
          (decf *tensor-pool-bytes* (* n (dtype-element-bytes dtype)))
          (car free))))))

(defun pool-release (dtype data)
  "Offer DATA back to the pool.  Silently declines once the budget is reached,
and always declines what the model owns.

The second refusal is what lets a Constant node hand out its attribute instead of
a copy of it.  The pool is for scratch — vectors the run made and the run is done
with — and a weight is not scratch: recycling one would hand the next tensor a
buffer the model is still using to say what that constant is.  Asking here rather
than at every call site means the guarantee holds in the subgraph runners too,
which have their own loops and no notion of pinning."
  (let ((tbl *tensor-pool*)
        (n (length data)))
    (when (and tbl (plusp n) (not (constant-data-p data)))
      (let ((bytes (* n (dtype-element-bytes dtype))))
        (when (<= (+ *tensor-pool-bytes* bytes) *tensor-pool-budget*)
          (push data (gethash (cons dtype n) tbl))
          (incf *tensor-pool-bytes* bytes)))))
  (values))

(defun fill-typed (dtype data value)
  "FILL, with the element type known, so it becomes a word bash rather than a
generic loop over an array of unknown type."
  (ecase dtype
    (:f32 (fill (the (simple-array single-float (*)) data)
                (the single-float value)))
    (:f64 (fill (the (simple-array double-float (*)) data)
                (the double-float value)))
    (:i64 (fill (the (simple-array (signed-byte 64) (*)) data)
                (the (signed-byte 64) value)))
    (:i32 (fill (the (simple-array (signed-byte 32) (*)) data)
                (the (signed-byte 32) value)))
    (:bool (fill (the (simple-array (unsigned-byte 8) (*)) data)
                 (the (unsigned-byte 8) value)))))

(defun make-tensor (dtype shape &key (initial-element nil ie-p))
  "A new tensor of DTYPE.  SHAPE may be a list or a vector."
  (let* ((shape (coerce shape 'simple-vector))
         (n (shape-size shape))
         (et (dtype-element-type dtype))
         (value (if ie-p (coerce initial-element et) (dtype-zero dtype)))
         (reused (pool-take dtype n)))
    (%make-tensor dtype shape
                  (cond (reused (fill-typed dtype reused value) reused)
                        (t (make-array n :element-type et
                                         :initial-element value))))))

(defun make-tensor-unfilled (dtype shape)
  "A new tensor of DTYPE whose data has NOT been zeroed.  Only for kernels that
assign every element of it — anything that accumulates still needs MAKE-TENSOR.
A recycled vector holds the last owner's numbers and a fresh one holds whatever
the pages held, so a caller that skips an element leaks a value from elsewhere in
the graph rather than reading a nice diagnosable zero."
  (let* ((shape (coerce shape 'simple-vector))
         (n (shape-size shape)))
    (%make-tensor dtype shape
                  (or (pool-take dtype n)
                      (make-array n :element-type (dtype-element-type dtype))))))

(defun tensor-size (tensor) (shape-size (tensor-shape tensor)))
(defun tensor-rank (tensor) (length (tensor-shape tensor)))
(defun tensor-dim (tensor i) (aref (tensor-shape tensor) i))

(defun tensor-scalar (tensor)
  "The single element of a scalar (or one-element) tensor."
  (aref (tensor-data tensor) 0))

;;; ---- typed access ----------------------------------------------------------
;;;
;;; Kernels want a declared SIMPLE-ARRAY of a known element type or SBCL boxes
;;; every float it touches.  WITH-TYPED-DATA compiles its body once per dtype and
;;; picks at run time, so an inner loop written once still runs unboxed.

(defmacro with-typed-data ((var tensor &key (dtypes '(:f32 :f64 :i64 :i32 :bool))) &body body)
  "Bind VAR to TENSOR's data, declared as the SIMPLE-ARRAY its dtype implies."
  (let ((tn (gensym "T")))
    `(let ((,tn ,tensor))
       (ecase (tensor-dtype ,tn)
         ,@(loop for dt in dtypes
                 collect `(,dt (let ((,var (the (simple-array ,(dtype-element-type dt) (*))
                                                (tensor-data ,tn))))
                                 (declare (ignorable ,var))
                                 ,@body)))))))

(defmacro with-two-typed-data ((va ta) (vb tb &key (dtypes '(:f32 :f64 :i64 :i32 :bool)))
                              &body body)
  "Like WITH-TYPED-DATA over two tensors that share a dtype.  Mixed dtypes are a
bug in the caller — ONNX ops that combine types say so explicitly (Cast, Where) —
so this refuses rather than coercing quietly.

DTYPES narrows which branches are compiled, exactly as in WITH-TYPED-DATA, and
matters for more than code size.  NESTING two WITH-TYPED-DATA forms over a pair
compiles every CROSS combination — for {:f32,:f64} that is four branches, two of
which write a DOUBLE-FLOAT into a SINGLE-FLOAT array.  Those branches cannot run,
because every caller builds its output with (MAKE-TENSOR (TENSOR-DTYPE input) ...),
but they are compiled all the same, and a compiler is entitled to complain about
code it can see is wrong.  SBCL 2.2.9 does, with a full WARNING that ASDF turns
fatal; 2.5.6 derives the types differently and only prints a note.  Same source,
different verdict, and the older one is right.

Stating the invariant here rather than implying it through nesting removes those
branches from existence, and keeps the check that the tensors really do agree —
without it, the THE declarations would be a lie and a mismatch would be silent
arithmetic on the wrong element type instead of an error."
  (let ((tan (gensym)) (tbn (gensym)))
    `(let ((,tan ,ta) (,tbn ,tb))
       (unless (eq (tensor-dtype ,tan) (tensor-dtype ,tbn))
         (error "tensor dtypes differ: ~a vs ~a" (tensor-dtype ,tan) (tensor-dtype ,tbn)))
       (ecase (tensor-dtype ,tan)
         ,@(loop for dt in dtypes
                 collect `(,dt (let ((,va (the (simple-array ,(dtype-element-type dt) (*))
                                               (tensor-data ,tan)))
                                     (,vb (the (simple-array ,(dtype-element-type dt) (*))
                                               (tensor-data ,tbn))))
                                 (declare (ignorable ,va ,vb))
                                 ,@body)))))))

;;; ---- indexing --------------------------------------------------------------

(defun row-major-strides (shape)
  "Element strides for SHAPE, row-major.  Stride of the last axis is 1."
  (let* ((r (length shape))
         (s (make-array r :element-type 'fixnum :initial-element 1)))
    (loop for i from (- r 2) downto 0
          do (setf (aref s i) (* (aref s (1+ i)) (aref shape (1+ i)))))
    s))

(defun tensor-ref (tensor &rest subscripts)
  "Element at SUBSCRIPTS.  For tests and printing — kernels index linearly."
  (let ((strides (row-major-strides (tensor-shape tensor)))
        (at 0))
    (loop for s in subscripts for i from 0 do (incf at (* s (aref strides i))))
    (aref (tensor-data tensor) at)))

(defun tensor-from-list (dtype shape list)
  "A tensor of DTYPE holding LIST, row-major.  Mostly for tests and small
constants built in code."
  (let* ((tn (make-tensor dtype shape))
         (et (dtype-element-type dtype))
         (data (tensor-data tn)))
    (loop for x in list for i from 0 do (setf (aref data i) (coerce x et)))
    tn))

;;; ---- broadcasting ----------------------------------------------------------
;;;
;;; NumPy's rules, which ONNX inherits: right-align the shapes, and a dimension
;;; of 1 stretches to meet the other.  Implemented as strides of 0 on the
;;; stretched axes, so a broadcast input needs no copy — the kernel walks the
;;; output's index space and reads the input through zero strides.

(defun broadcast-shapes (a b)
  "The shape A and B broadcast to, or an error naming both if they do not."
  (let* ((ra (length a)) (rb (length b)) (r (max ra rb))
         (out (make-array r :initial-element 1)))
    (dotimes (i r out)
      (let* ((da (if (< (- r i) (1+ ra)) (aref a (- i (- r ra))) 1))
             (db (if (< (- r i) (1+ rb)) (aref b (- i (- r rb))) 1)))
        (cond ((or (= da db) (= db 1)) (setf (aref out i) da))
              ((= da 1) (setf (aref out i) db))
              (t (error "shapes ~a and ~a do not broadcast (axis ~d: ~d vs ~d)"
                        a b i da db)))))))

(defun broadcast-strides (shape out-shape)
  "Strides for reading a tensor of SHAPE while walking OUT-SHAPE's index space.
Axes that are being stretched get a stride of 0, so the same element is re-read."
  (let* ((r (length out-shape))
         (rs (length shape))
         (own (row-major-strides shape))
         (out (make-array r :element-type 'fixnum :initial-element 0)))
    (dotimes (i r out)
      (let ((j (- i (- r rs))))
        (when (>= j 0)
          (setf (aref out i) (if (= (aref shape j) 1) 0 (aref own j))))))))

(defmacro do-broadcast (specs &body body)
  "Walk an index space, binding one element index per input as it goes.

SPECS is any number of (VAR STRIDES) pairs followed by the shape to walk: each
VAR is the element index of an input read through STRIDES, and I is the linear
index into the space itself.  Strides of 0 are what make this broadcast — the
same input element is read for every step along a stretched axis.

The last axis is walked as a straight run rather than through the odometer, and
that is not a micro-optimization: this macro is under every elementwise op in the
engine, and the odometer costs several array accesses per element against the two
or three the body itself does.  Written the obvious way, Where, Log, Equal and Exp
together were 19% of a transcription, most of it spent counting.  A carry is only
possible once per run, so hoisting it out of the run is exact — the indices
visited, and the order, are the same as before.

The run advances each VAR by that input's own last-axis stride, which is 0 for an
input being stretched along it, and then rewinds so the leading axes carry from
where the run began."
  (let* ((out-shape (car (last specs)))
         (pairs (butlast specs))
         (vars (mapcar #'first pairs))
         (svars (loop for nil in pairs collect (gensym "STRIDES")))
         (steps (loop for nil in pairs collect (gensym "STEP")))
         (sh (gensym)) (r (gensym)) (n (gensym)) (idx (gensym)) (k (gensym))
         (inner (gensym "INNER")) (outer (gensym "OUTER")) (jj (gensym)))
    `(let* ((,sh ,out-shape)
            (,r (length ,sh))
            (,n (shape-size ,sh))
            (,inner (if (plusp ,r) (aref ,sh (1- ,r)) 1))
            (,outer (if (plusp ,inner) (floor ,n ,inner) 0))
            (,idx (make-array (max 1 ,r) :element-type 'fixnum :initial-element 0))
            ,@(loop for s in svars for p in pairs collect `(,s ,(second p)))
            ,@(loop for st in steps for s in svars
                    collect `(,st (if (plusp ,r) (aref ,s (1- ,r)) 0)))
            ,@(loop for v in vars collect `(,v 0))
            (i 0))
       (declare (type index ,n i) (type fixnum ,inner ,outer ,@steps ,@vars)
                (ignorable i ,@vars))
       (dotimes (,jj ,outer)
         (declare (ignorable ,jj))
         (dotimes (,k ,inner)
           (declare (ignorable ,k))
           ,@body
           (incf i)
           ,@(loop for v in vars for st in steps collect `(incf ,v ,st)))
         ;; back to where the run started, so the leading axes carry from there
         ,@(loop for v in vars for st in steps collect `(decf ,v (* ,inner ,st)))
         ;; odometer over everything but the last axis, which the run just did
         (loop for ,k from (- ,r 2) downto 0
               do ,@(loop for v in vars for s in svars
                          collect `(incf ,v (aref ,s ,k)))
                  (incf (aref ,idx ,k))
                  (if (< (aref ,idx ,k) (aref ,sh ,k))
                      (return)
                      (progn
                        ,@(loop for v in vars for s in svars
                                collect `(decf ,v (* (aref ,idx ,k) (aref ,s ,k))))
                        (setf (aref ,idx ,k) 0))))))))

;;; ---- comparison ------------------------------------------------------------

(defun tensor-max-abs-diff (a b)
  "Largest absolute difference between two same-shaped tensors, as a double.
NIL if the shapes differ.  This is the number every gate in the project reports:
a port of a numeric graph is never bit-exact against another implementation's
kernels, so 'equal' is the wrong question and 'how far off' is the right one."
  (unless (equalp (tensor-shape a) (tensor-shape b))
    (return-from tensor-max-abs-diff nil))
  (let ((worst 0d0))
    (with-two-typed-data (da a) (db b)
      (dotimes (i (length da))
        (let ((d (abs (- (float (aref da i) 1d0) (float (aref db i) 1d0)))))
          (when (> d worst) (setf worst d)))))
    worst))

(defun tensor-equal-p (a b &optional (tolerance 0d0))
  (let ((d (tensor-max-abs-diff a b)))
    (and d (<= d tolerance))))

(defun print-tensor (tensor &key (stream *standard-output*) (limit 8))
  "A one-line summary: dtype, shape, and the first few elements."
  (format stream "~&<~a ~a" (tensor-dtype tensor) (coerce (tensor-shape tensor) 'list))
  (let ((n (min limit (tensor-size tensor)))
        (data (tensor-data tensor)))
    (format stream " [~{~a~^ ~}~:[~; ...~]]>~%"
            (loop for i below n collect (aref data i))
            (> (tensor-size tensor) n)))
  tensor)
