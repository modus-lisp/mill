;;;; ops-index.lisp — gathering and scattering.
;;;;
;;;; This is where VITS builds its alignment: the duration predictor produces a
;;;; path, NonZero turns a mask into coordinates, and GatherND/ScatterND move
;;;; encoder frames into the places the path says they go.  Every one of these
;;;; ops takes an index tensor, and every one of them accepts negative indices,
;;;; which is a cheap way to be off by a whole dimension.

(in-package #:mill)

(declaim (inline wrap-index))
(defun wrap-index (i dim)
  "ONNX index semantics: negative counts from the end."
  (if (minusp i) (+ i dim) i))

(defop "Gather" (node ins)
  (destructuring-bind (a indices) ins
    (let* ((shape (tensor-shape a))
           (axis (normalize-axis (node-attr node "axis" 0) (length shape)))
           (ishape (tensor-shape indices))
           (out-shape (concatenate 'simple-vector
                                   (subseq shape 0 axis) ishape (subseq shape (1+ axis))))
           (dim (aref shape axis))
           (outer (shape-size (subseq shape 0 axis)))
           (inner (shape-size (subseq shape (1+ axis))))
           (ni (tensor-size indices))
           (idx (tensor-data indices))
           (res (make-tensor (tensor-dtype a) out-shape))
           (da (tensor-data a))
           (dr (tensor-data res)))
      (dotimes (o outer)
        (dotimes (j ni)
          (let ((src (* (+ (* o dim) (wrap-index (aref idx j) dim)) inner)))
            (replace dr da :start1 (* (+ (* o ni) j) inner)
                           :start2 src :end2 (+ src inner)))))
      res)))

(defop "GatherElements" (node ins)
  (destructuring-bind (a indices) ins
    (let* ((shape (tensor-shape a))
           (axis (normalize-axis (node-attr node "axis" 0) (length shape)))
           (ishape (tensor-shape indices))
           (dstrides (row-major-strides shape))
           (dim (aref shape axis))
           (axis-stride (aref dstrides axis))
           ;; walk the INDEX tensor's space; every axis but the gathered one
           ;; contributes its own coordinate, so zero that one out and add the
           ;; looked-up index by hand
           (strides (copy-seq dstrides))
           (idx (tensor-data indices))
           (res (make-tensor (tensor-dtype a) ishape)))
      (setf (aref strides axis) 0)
      (with-two-typed-data (da a) (dr res)
        (do-broadcast ((base strides) ishape)
          (setf (aref dr i)
                (aref da (+ base (* axis-stride (wrap-index (aref idx i) dim)))))))
      res)))

(defun gather-nd-plan (data indices)
  "Values: index-tuple width K, number of index rows, and the block size each row
selects.  GatherND and ScatterND agree on all three; only the direction differs."
  (let* ((ishape (tensor-shape indices))
         (k (aref ishape (1- (length ishape))))
         (rows (shape-size (subseq ishape 0 (1- (length ishape)))))
         (block-size (shape-size (subseq (tensor-shape data) k))))
    (values k rows block-size)))

(defop "GatherND" (node ins)
  (destructuring-bind (a indices) ins
    (multiple-value-bind (k rows block-size) (gather-nd-plan a indices)
      (let* ((shape (tensor-shape a))
             (ishape (tensor-shape indices))
             (dstrides (row-major-strides shape))
             (idx (tensor-data indices))
             (out-shape (concatenate 'simple-vector
                                     (subseq ishape 0 (1- (length ishape)))
                                     (subseq shape k)))
             (res (make-tensor (tensor-dtype a) out-shape))
             (da (tensor-data a))
             (dr (tensor-data res)))
        (dotimes (r rows)
          (let ((src 0))
            (dotimes (j k)
              (incf src (* (aref dstrides j)
                           (wrap-index (aref idx (+ (* r k) j)) (aref shape j)))))
            (replace dr da :start1 (* r block-size)
                           :start2 src :end2 (+ src block-size))))
        res))))

(defop "ScatterND" (node ins)
  (destructuring-bind (a indices updates) ins
    (multiple-value-bind (k rows block-size) (gather-nd-plan a indices)
      ;; ScatterND is the one op that writes into its data operand, so it copies
      ;; first — every other op in mill treats its inputs as read-only, and the
      ;; runner shares tensors between nodes on that promise.
      (let* ((res (copy-tensor a))
             (shape (tensor-shape a))
             (dstrides (row-major-strides shape))
             (idx (tensor-data indices))
             (du (tensor-data updates))
             (dr (tensor-data res)))
        (dotimes (r rows)
          (let ((dst 0)
                (src (* r block-size)))
            (dotimes (j k)
              (incf dst (* (aref dstrides j)
                           (wrap-index (aref idx (+ (* r k) j)) (aref shape j)))))
            (replace dr du :start1 dst :start2 src :end2 (+ src block-size))))
        res))))

(defop "NonZero" (node ins)
  ;; Output is (rank, count): one ROW per dimension, one COLUMN per nonzero
  ;; element, in row-major order.  The transposed layout is what trips people up
  ;; — coordinates are read down the columns, not across.
  (let* ((a (first ins))
         (shape (tensor-shape a))
         (rank (length shape))
         (strides (row-major-strides shape))
         (count 0))
    (with-typed-data (da a)
      (dotimes (i (length da)) (unless (zerop (aref da i)) (incf count)))
      (let* ((res (make-tensor :i64 (vector rank count)))
             (dr (tensor-data res))
             (col 0))
        (dotimes (i (length da))
          (unless (zerop (aref da i))
            (let ((rest i))
              (dotimes (d rank)
                (setf (aref dr (+ (* d count) col)) (floor rest (aref strides d)))
                (setf rest (mod rest (aref strides d)))))
            (incf col)))
        res))))
