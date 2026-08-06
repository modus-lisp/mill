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
  "Offer DATA back to the pool.  Silently declines once the budget is reached."
  (let ((tbl *tensor-pool*)
        (n (length data)))
    (when (and tbl (plusp n))
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

(defmacro with-two-typed-data ((va ta) (vb tb) &body body)
  "Like WITH-TYPED-DATA over two tensors that share a dtype.  Mixed dtypes are a
bug in the caller — ONNX ops that combine types say so explicitly (Cast, Where) —
so this refuses rather than coercing quietly."
  (let ((tan (gensym)) (tbn (gensym)))
    `(let ((,tan ,ta) (,tbn ,tb))
       (unless (eq (tensor-dtype ,tan) (tensor-dtype ,tbn))
         (error "tensor dtypes differ: ~a vs ~a" (tensor-dtype ,tan) (tensor-dtype ,tbn)))
       (ecase (tensor-dtype ,tan)
         ,@(loop for dt in '(:f32 :f64 :i64 :i32 :bool)
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
same input element is read for every step along a stretched axis."
  (let* ((out-shape (car (last specs)))
         (pairs (butlast specs))
         (vars (mapcar #'first pairs))
         (svars (loop for nil in pairs collect (gensym "STRIDES")))
         (sh (gensym)) (r (gensym)) (n (gensym)) (idx (gensym)) (k (gensym)))
    `(let* ((,sh ,out-shape)
            (,r (length ,sh))
            (,n (shape-size ,sh))
            (,idx (make-array (max 1 ,r) :element-type 'fixnum :initial-element 0))
            ,@(loop for s in svars for p in pairs collect `(,s ,(second p)))
            ,@(loop for v in vars collect `(,v 0)))
       (declare (type index ,n) (type fixnum ,@vars) (ignorable ,@vars))
       (dotimes (i ,n)
         (declare (type index i))
         ,@body
         ;; odometer: bump the last axis, carry leftward
         (loop for ,k from (1- ,r) downto 0
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
