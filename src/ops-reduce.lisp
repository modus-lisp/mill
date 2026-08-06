;;;; ops-reduce.lisp — reductions and the ops that scan along one axis.
;;;;
;;;; Sums accumulate in double even when the tensor is float32.  onnxruntime
;;;; accumulates in the element type, so this is deliberately not bit-identical:
;;;; it is more accurate, and the difference (~1e-7 relative on a 192-wide layer
;;;; norm) is far below the tolerance the per-node gate uses.  Being closer to
;;;; the true value than the reference is a defensible place to differ; being
;;;; further away is not.

(in-package #:mill)

(defun axis-view (shape axis)
  "SHAPE seen as (OUTER DIM INNER) around AXIS: everything left of the axis, the
axis itself, everything right of it.  Both the scan ops want this."
  (values (shape-size (subseq shape 0 axis))
          (aref shape axis)
          (shape-size (subseq shape (1+ axis)))))

(defun reduction-plan (shape axes keepdims)
  "Values: the output shape, and the strides to walk the INPUT with while
accumulating into the output — zero on every reduced axis, so all the elements
being folded together land on the same output index."
  (let* ((rank (length shape))
         (axes (mapcar (lambda (a) (normalize-axis a rank)) axes))
         (compact (coerce (loop for d across shape for i from 0
                                unless (member i axes) collect d)
                          'simple-vector))
         (cstrides (row-major-strides compact))
         (strides (make-array (max 1 rank) :element-type 'fixnum :initial-element 0))
         (c 0))
    (dotimes (i rank)
      (unless (member i axes)
        (setf (aref strides i) (aref cstrides c))
        (incf c)))
    (values (if keepdims
                (let ((s (copy-seq shape)))
                  (dolist (a axes s) (setf (aref s a) 1)))
                compact)
            strides)))

(defmacro def-reduction (name (acc x) &key init int-init step finish int-finish
                                          (dtypes '(:f32 :f64)))
  "Define a reduction.  INIT starts the accumulator, STEP folds one element X
into ACC, and FINISH turns the accumulator into the stored value (it also sees
COUNT, the number of elements folded into that cell)."
  (let ((cells (gensym "CELLS")))
    `(defun ,(intern (concatenate 'string "%REDUCE-" (string-upcase name)))
         (a axes keepdims)
       (multiple-value-bind (out-shape strides)
           (reduction-plan (tensor-shape a) axes keepdims)
         (ecase (tensor-dtype a)
           ,@(loop for dt in dtypes
                   for et = (dtype-element-type dt)
                   ;; floats accumulate in double; integers accumulate as
                   ;; integers, because rounding an index is not a small error
                   for float-p = (member dt '(:f32 :f64))
                   for at = (if float-p 'double-float '(signed-byte 64))
                   for start = (if float-p init int-init)
                   for done = (if float-p finish (or int-finish finish))
                   collect
                   `(,dt (let* ((res (make-tensor ,dt out-shape))
                                (n (tensor-size res))
                                (da (the (simple-array ,et (*)) (tensor-data a)))
                                (dr (the (simple-array ,et (*)) (tensor-data res)))
                                (,cells (make-array (max 1 n) :element-type ',at
                                                              :initial-element ,start))
                                (count (if (zerop n) 0 (floor (tensor-size a) n))))
                           (declare (ignorable count)
                                    (type (simple-array ,at (*)) ,cells))
                           (do-broadcast ((io strides) (tensor-shape a))
                             (let ((,acc (aref ,cells io))
                                   (,x ,(if float-p `(float (aref da i) 1d0) `(aref da i))))
                               (declare (type ,at ,acc ,x) (ignorable ,acc ,x))
                               (setf (aref ,cells io) ,step)))
                           (dotimes (j n)
                             (let ((,acc (aref ,cells j)))
                               (declare (type ,at ,acc) (ignorable ,acc))
                               (setf (aref dr j) (coerce ,done ',et))))
                           res))))))))

(def-reduction "sum" (acc x) :init 0d0 :int-init 0 :step (+ acc x) :finish acc
                             :dtypes (:f32 :f64 :i64 :i32))
(def-reduction "mean" (acc x) :init 0d0 :int-init 0 :step (+ acc x)
                              :finish (/ acc count) :int-finish (truncate acc count)
                              :dtypes (:f32 :f64 :i64 :i32))
(def-reduction "max" (acc x) :init sb-ext:double-float-negative-infinity
                             :int-init most-negative-fixnum
                             :step (max acc x) :finish acc
                             :dtypes (:f32 :f64 :i64 :i32))
(def-reduction "min" (acc x) :init sb-ext:double-float-positive-infinity
                             :int-init most-positive-fixnum
                             :step (min acc x) :finish acc
                             :dtypes (:f32 :f64 :i64 :i32))

(defun all-axes (a) (loop for i below (tensor-rank a) collect i))

(defop "ReduceMean" (node ins)
  (let ((a (first ins)))
    (%reduce-mean a (or (node-attr node "axes")
                        (and (second ins) (tensor-integers (second ins)))
                        (all-axes a))
                  (not (eql 0 (node-attr node "keepdims" 1))))))

(defop "ReduceSum" (node ins)
  (let ((a (first ins)))
    (%reduce-sum a (or (node-attr node "axes")
                       (and (second ins) (tensor-integers (second ins)))
                       (all-axes a))
                 (not (eql 0 (node-attr node "keepdims" 1))))))

(defop "ReduceMax" (node ins)
  (let ((a (first ins)))
    (%reduce-max a (or (node-attr node "axes")
                       (and (second ins) (tensor-integers (second ins)))
                       (all-axes a))
                 (not (eql 0 (node-attr node "keepdims" 1))))))

(defop "ReduceMin" (node ins)
  (let ((a (first ins)))
    (%reduce-min a (or (node-attr node "axes")
                       (and (second ins) (tensor-integers (second ins)))
                       (all-axes a))
                 (not (eql 0 (node-attr node "keepdims" 1))))))

;;; ---- scans -----------------------------------------------------------------

(defop "Softmax" (node ins)
  ;; opset 13 semantics: softmax along AXIS alone, not over the flattened tail
  ;; the way opset 11 did.  This graph is all axis=-1 attention weights.
  (let* ((a (first ins))
         (shape (tensor-shape a))
         (axis (normalize-axis (node-attr node "axis" -1) (length shape)))
         (res (make-tensor (tensor-dtype a) shape)))
    (multiple-value-bind (outer dim inner) (axis-view shape axis)
      (let ((da (the (simple-array single-float (*)) (tensor-data a)))
            (dr (the (simple-array single-float (*)) (tensor-data res))))
        (dotimes (o outer)
          (dotimes (n inner)
            (let ((base (+ (* o dim inner) n))
                  (top sb-ext:single-float-negative-infinity)
                  (sum 0d0))
              (declare (type single-float top) (type double-float sum))
              (dotimes (k dim)
                (let ((x (aref da (+ base (* k inner))))) (when (> x top) (setf top x))))
              (dotimes (k dim)
                (let ((e (exp (- (float (aref da (+ base (* k inner))) 1d0) top))))
                  (incf sum e)
                  (setf (aref dr (+ base (* k inner))) (float e 1.0f0))))
              (dotimes (k dim)
                (setf (aref dr (+ base (* k inner)))
                      (float (/ (float (aref dr (+ base (* k inner))) 1d0) sum) 1.0f0))))))))
    res))

(defop "CumSum" (node ins)
  (let* ((a (first ins))
         (shape (tensor-shape a))
         (axis (normalize-axis (tensor-scalar (second ins)) (length shape)))
         (exclusive (eql 1 (node-attr node "exclusive" 0)))
         (reverse (eql 1 (node-attr node "reverse" 0)))
         (res (make-tensor (tensor-dtype a) shape)))
    (multiple-value-bind (outer dim inner) (axis-view shape axis)
      (with-two-typed-data (da a) (dr res)
        (dotimes (o outer)
          (dotimes (n inner)
            (let ((base (+ (* o dim inner) n))
                  (run (dtype-zero (tensor-dtype a))))
              (dotimes (step dim)
                (let* ((k (if reverse (- dim 1 step) step))
                       (at (+ base (* k inner))))
                  (cond (exclusive (setf (aref dr at) run) (incf run (aref da at)))
                        (t (incf run (aref da at)) (setf (aref dr at) run))))))))))
    res))
