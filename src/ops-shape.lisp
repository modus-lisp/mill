;;;; ops-shape.lisp — the ops that move data without doing arithmetic to it.
;;;;
;;;; Two thirds of this graph's nodes are these: Shape, Reshape, Unsqueeze,
;;;; Slice, Concat, Transpose, Expand.  They are where a port goes wrong quietly
;;;; — an off-by-one in a slice produces a tensor of the right shape full of the
;;;; wrong numbers, and the audio at the end still sounds like speech.  Hence the
;;;; per-node gate.

(in-package #:mill)

;;; ---- small shared helpers --------------------------------------------------

(defun tensor-integers (tensor)
  "The elements of an integer tensor as a list.  ONNX passes shapes, axes,
starts, and ends as i64 tensors, so this is how most of these ops read their
arguments."
  (let ((data (tensor-data tensor)))
    (loop for i below (tensor-size tensor) collect (aref data i))))

(defun normalize-axis (axis rank)
  (if (minusp axis) (+ axis rank) axis))

(defun copy-tensor (a)
  (%make-tensor (tensor-dtype a) (copy-seq (tensor-shape a)) (copy-seq (tensor-data a))))

(defun reshape-view (a shape)
  "A tensor with the same data as A and a new SHAPE.  The data is shared, which
is safe because no op in mill writes into its inputs — the ones that would
(ScatterND, Pad) copy first."
  (let ((shape (coerce shape 'simple-vector)))
    (assert (= (shape-size shape) (tensor-size a)) ()
            "reshape ~a -> ~a changes the element count"
            (coerce (tensor-shape a) 'list) (coerce shape 'list))
    (%make-tensor (tensor-dtype a) shape (tensor-data a))))

(defun zero-strides (rank)
  (make-array (max 1 rank) :element-type 'fixnum :initial-element 0))

(defun collapse-gather-axes (out-shape strides)
  "OUT-SHAPE and STRIDES rewritten with the fewest axes that describe the same
walk, outermost first.  Two values, both simple FIXNUM vectors: extents and
strides.

Two rules, and between them they are most of what makes this copy fast.  An axis
of extent 1 is one step of no size, so it is dropped.  And two adjacent axes
where the outer one's stride is exactly the inner one's span — STRIDE-OUT =
STRIDE-IN * EXTENT-IN — visit a contiguous block of source indices in ascending
order, which is one axis of their product.  That is the common case rather than a
corner one: a Slice or a Split that only cuts an outer axis leaves everything to
its right untouched, so the whole tail collapses to a single run and the copy
becomes one block move.

The output is dense row-major either way, so this only ever reasons about the
source side; the destination is written straight through."
  (let ((ext '()) (st '()))
    (loop for k of-type fixnum from 0 below (length out-shape)
          for e of-type fixnum = (aref out-shape k)
          for s of-type fixnum = (aref strides k)
          do (cond ((= e 1))
                   ;; EXT's head is the innermost axis so far, which is the one
                   ;; this axis sits inside of
                   ((and ext (= (the fixnum (first st)) (* s e)))
                    (setf (first ext) (* (the fixnum (first ext)) e)
                          (first st) s))
                   (t (push e ext) (push s st))))
    (let* ((r (length ext))
           (ev (make-array (max 1 r) :element-type 'fixnum :initial-element 1))
           (sv (make-array (max 1 r) :element-type 'fixnum :initial-element 0)))
      ;; EXT was built innermost-first, and the walk wants outermost-first
      (loop for e in ext for s in st for k downfrom (1- r)
            do (setf (aref ev k) e (aref sv k) s))
      (values ev sv r))))

(defun tensor-gather-strided (a base strides out-shape)
  "Read A through STRIDES starting at BASE, walking OUT-SHAPE.  This is the one
copy loop behind Slice, Transpose, Expand and Split — they differ only in the
strides they hand it, and on this graph it is a fifth of a single-threaded
synthesis, so it is written as three cases rather than one.

After COLLAPSE-GATHER-AXES the innermost axis is the run, and the axes outside it
are an odometer that runs once per run instead of once per element.  A run of
stride 1 is a block move; SBCL's REPLACE between two simple arrays of the same
element type is a memmove, which is the whole point of collapsing.  A run of any
other stride is a typed gather — that is what a real Transpose leaves behind, and
even there the odometer is off the hot path.

The output gets every element assigned, so it starts unfilled: MAKE-TENSOR would
zero a buffer this loop is about to overwrite completely, and these buffers are
hundreds of kilobytes."
  (let* ((out (make-tensor-unfilled (tensor-dtype a) out-shape))
         (n (tensor-size out)))
    (declare (type index n))
    (when (plusp n)
      (multiple-value-bind (ext st r) (collapse-gather-axes out-shape strides)
        (declare (type (simple-array fixnum (*)) ext st) (type fixnum r))
        (with-two-typed-data (da a) (dr out)
          (locally (declare (optimize (speed 3) (safety 0)))
            (let* ((inner (if (plusp r) (aref ext (1- r)) 1))
                   (istep (if (plusp r) (aref st (1- r)) 1))
                   (outer (floor n inner))
                   (idx (make-array (max 1 r) :element-type 'fixnum
                                              :initial-element 0))
                   (src base)
                   (dst 0))
              (declare (type index inner outer dst) (type fixnum istep src))
              (dotimes (o outer)
                (declare (type index o))
                (if (= istep 1)
                    (replace dr da :start1 dst :end1 (+ dst inner) :start2 src)
                    (loop for k of-type index from 0 below inner
                          do (setf (aref dr (+ dst k))
                                   (aref da (+ src (* k istep))))))
                (incf dst inner)
                ;; odometer over everything outside the run, innermost first
                (loop for k of-type fixnum from (- r 2) downto 0
                      do (incf src (aref st k))
                         (incf (aref idx k))
                         (when (< (aref idx k) (aref ext k)) (return))
                         (decf src (* (aref idx k) (aref st k)))
                         (setf (aref idx k) 0))))))))
    out))

;;; ---- shape reporting and reshaping -----------------------------------------

(defop "Shape" (node ins)
  (let* ((shape (tensor-shape (first ins)))
         (out (make-tensor :i64 (vector (length shape)))))
    (loop for d across shape for i from 0 do (setf (aref (tensor-data out) i) d))
    out))

(defop "Reshape" (node ins)
  (let* ((a (first ins))
         (want (tensor-integers (second ins)))
         (allowzero (eql 1 (node-attr node "allowzero")))
         (in-shape (tensor-shape a))
         ;; 0 means "keep the input's dimension here" unless allowzero says the
         ;; caller really meant an empty axis; -1 means "whatever is left over".
         (dims (loop for d in want for i from 0
                     collect (cond ((and (zerop d) (not allowzero)) (aref in-shape i))
                                   (t d))))
         (rest (reduce #'* (remove -1 dims)))
         (dims (substitute (if (zerop rest) 0 (floor (tensor-size a) rest)) -1 dims)))
    (reshape-view a dims)))

(defop "Unsqueeze" (node ins)
  ;; opset 13 moved the axes from an attribute to an input; they are relative to
  ;; the OUTPUT rank, which is why they are normalized after the new rank is known.
  (let* ((a (first ins))
         (in-shape (tensor-shape a))
         (rank (+ (length in-shape) (tensor-size (second ins))))
         (axes (sort (mapcar (lambda (x) (normalize-axis x rank))
                             (tensor-integers (second ins)))
                     #'<))
         (out (make-array rank :initial-element nil))
         (src 0))
    (dolist (ax axes) (setf (aref out ax) 1))
    (dotimes (i rank)
      (unless (aref out i) (setf (aref out i) (aref in-shape src)) (incf src)))
    (reshape-view a out)))

(defop "Squeeze" (node ins)
  (let* ((a (first ins))
         (in-shape (tensor-shape a))
         (rank (length in-shape))
         (axes (if (second ins)
                   (mapcar (lambda (x) (normalize-axis x rank)) (tensor-integers (second ins)))
                   (loop for d across in-shape for i from 0 when (= d 1) collect i))))
    (reshape-view a (loop for d across in-shape for i from 0
                          unless (member i axes) collect d))))

(defop "Transpose" (node ins)
  (let* ((a (first ins))
         (shape (tensor-shape a))
         (rank (length shape))
         (perm (or (node-attr node "perm") (loop for i from (1- rank) downto 0 collect i)))
         (perm (coerce perm 'vector))
         (istrides (row-major-strides shape))
         (out-shape (map 'vector (lambda (p) (aref shape p)) perm))
         ;; moving one step along output axis k moves ISTRIDES[perm[k]] in the input
         (strides (make-array (max 1 rank) :element-type 'fixnum :initial-element 0)))
    (dotimes (k rank) (setf (aref strides k) (aref istrides (aref perm k))))
    (tensor-gather-strided a 0 strides out-shape)))

(defop "Expand" (node ins)
  (let* ((a (first ins))
         ;; Expand broadcasts BOTH ways: the result is the broadcast of the input
         ;; shape with the requested one, not the requested one outright.
         (out-shape (broadcast-shapes (tensor-shape a)
                                      (coerce (tensor-integers (second ins)) 'vector))))
    (tensor-gather-strided a 0 (broadcast-strides (tensor-shape a) out-shape) out-shape)))

;;; ---- slicing and joining ---------------------------------------------------

(defop "Slice" (node ins)
  (destructuring-bind (a starts ends &optional axes steps) ins
    (let* ((shape (tensor-shape a))
           (rank (length shape))
           (istrides (row-major-strides shape))
           (starts (tensor-integers starts))
           (ends (tensor-integers ends))
           (axes (if axes
                     (mapcar (lambda (x) (normalize-axis x rank)) (tensor-integers axes))
                     (loop for i below (length starts) collect i)))
           (steps (if steps (tensor-integers steps) (make-list (length starts) :initial-element 1)))
           (out-shape (copy-seq shape))
           (strides (make-array (max 1 rank) :element-type 'fixnum :initial-element 0))
           (base 0))
      (dotimes (k rank) (setf (aref strides k) (aref istrides k)))
      (loop for ax in axes for st in starts for en in ends for sp in steps
            do (let* ((dim (aref shape ax))
                      (st (if (minusp st) (+ st dim) st))
                      (en (if (minusp en) (+ en dim) en))
                      ;; the clamps differ by direction: a backward slice may end
                      ;; at -1 (meaning "past the front"), a forward one at dim.
                      (st (if (plusp sp) (min (max st 0) dim) (min (max st 0) (1- dim))))
                      (en (if (plusp sp) (min (max en 0) dim) (min (max en -1) (1- dim))))
                      (n (max 0 (ceiling (- en st) sp))))
                 (setf (aref out-shape ax) n)
                 (incf base (* st (aref istrides ax)))
                 (setf (aref strides ax) (* sp (aref istrides ax)))))
      (tensor-gather-strided a base strides out-shape))))

(defop "Concat" (node ins)
  (let* ((rank (length (tensor-shape (first ins))))
         (axis (normalize-axis (node-attr node "axis" 0) rank))
         (shape (copy-seq (tensor-shape (first ins))))
         (total (loop for a in ins sum (aref (tensor-shape a) axis)))
         ;; everything left of the axis iterates, everything right of it is a
         ;; contiguous run that can be copied wholesale
         (outer (shape-size (subseq shape 0 axis)))
         (inner (shape-size (subseq shape (1+ axis)))))
    (setf (aref shape axis) total)
    (let* ((out (make-tensor (tensor-dtype (first ins)) shape))
           (dr (tensor-data out))
           (at 0))
      (declare (type index at))
      ;; REPLACE rather than a typed loop: the inputs are a runtime list, so a
      ;; per-dtype dispatch would have to happen inside the loop anyway, and
      ;; SBCL's REPLACE on two simple arrays of the same element type is a block
      ;; move — which is all this op is.
      (dotimes (o outer)
        (dolist (a ins)
          (let* ((chunk (* (aref (tensor-shape a) axis) inner))
                 (from (* o chunk)))
            (replace dr (tensor-data a) :start1 at :start2 from :end2 (+ from chunk))
            (incf at chunk))))
      out)))

(defop "Split" (node ins)
  (let* ((a (first ins))
         (shape (tensor-shape a))
         (rank (length shape))
         (axis (normalize-axis (node-attr node "axis" 0) rank))
         (sizes (if (second ins)
                    (tensor-integers (second ins))
                    (let ((n (length (node-outputs node))))
                      (make-list n :initial-element (floor (aref shape axis) n)))))
         (istrides (row-major-strides shape))
         (at 0))
    (loop for size in sizes
          collect (let ((out-shape (copy-seq shape))
                        (strides (make-array (max 1 rank) :element-type 'fixnum
                                                          :initial-element 0)))
                    (dotimes (k rank) (setf (aref strides k) (aref istrides k)))
                    (setf (aref out-shape axis) size)
                    (prog1 (tensor-gather-strided a (* at (aref istrides axis))
                                                  strides out-shape)
                      (incf at size))))))

;;; ---- filling and padding ---------------------------------------------------

(defop "Constant" (node ins)
  ;; A node with no inputs that hands back an attribute.  The result is a COPY:
  ;; the attribute belongs to the model and outlives the run, while the value
  ;; bound here is an ordinary intermediate the runner may hand to the tensor
  ;; pool the moment its last reader is done — which would quietly rewrite the
  ;; constant under a second run of the same model.
  (let ((tensor (node-attr node "value")))
    (cond (tensor (copy-tensor tensor))
          ((node-attr node "value_int")
           (tensor-from-list :i64 #() (list (node-attr node "value_int"))))
          ((node-attr node "value_ints")
           (let ((xs (node-attr node "value_ints")))
             (tensor-from-list :i64 (vector (length xs)) xs)))
          ((node-attr node "value_float")
           (tensor-from-list :f32 #() (list (node-attr node "value_float"))))
          ((node-attr node "value_floats")
           (let ((xs (node-attr node "value_floats")))
             (tensor-from-list :f32 (vector (length xs)) xs)))
          (t (error "Constant ~a has no value attribute this understands (~{~a~^, ~})"
                    (node-name node) (mapcar #'first (node-attrs node)))))))

(defop "Tile" (node ins)
  (destructuring-bind (a repeats) ins
    (let* ((shape (tensor-shape a))
           (reps (tensor-integers repeats)))
      (unless (= (length reps) (length shape))
        (error "Tile wants one repeat per axis: ~a for shape ~a"
               reps (coerce shape 'list)))
      ;; Tiling is a broadcast in disguise, once the axes are split.  Give the
      ;; input a fresh axis of extent 1 in front of each of its own, stretch
      ;; those to the repeat counts, and fold each pair back together: output
      ;; element r*d + k along an axis reads input element k, which is exactly
      ;; what a repeat means in row-major order.  So this costs one strided copy
      ;; and no new kernel.
      (let* ((wide (coerce (loop for d across shape append (list 1 d)) 'vector))
             (split (coerce (loop for d across shape for r in reps append (list r d))
                            'vector))
             (view (reshape-view a wide)))
        (reshape-view (tensor-gather-strided view 0 (broadcast-strides wide split) split)
                      (map 'vector #'* shape (coerce reps 'vector)))))))

(defop "ConstantOfShape" (node ins)
  (let* ((value (node-attr node "value"))
         (dtype (if value (tensor-dtype value) :f32))
         (out (make-tensor dtype (coerce (tensor-integers (first ins)) 'vector))))
    (when value
      (let ((x (tensor-scalar value)))
        (with-typed-data (dr out)
          (dotimes (i (length dr)) (setf (aref dr i) x)))))
    out))

(defop "Pad" (node ins)
  (destructuring-bind (a pads &optional value axes) ins
    (declare (ignore axes))
    (let* ((mode (node-attr node "mode" "constant"))
           (shape (tensor-shape a))
           (rank (length shape))
           (pads (tensor-integers pads))
           (begin (subseq pads 0 rank))
           (end (subseq pads rank))
           (out-shape (map 'vector #'+ shape (coerce begin 'vector) (coerce end 'vector)))
           (out (make-tensor (tensor-dtype a) out-shape))
           (ostrides (row-major-strides out-shape))
           (fill-value (if (and value (plusp (tensor-size value))) (tensor-scalar value) nil)))
      (unless (string= mode "constant")
        (error "Pad mode ~s is not implemented (this graph only uses constant)" mode))
      ;; Negative pads mean "crop", which ONNX allows and this graph does not use;
      ;; refusing is better than silently producing a plausible wrong tensor.
      (when (some #'minusp pads)
        (error "Pad with negative pads (cropping) is not implemented"))
      (with-two-typed-data (da a) (dr out)
        (when (and fill-value (not (eql fill-value (dtype-zero (tensor-dtype a)))))
          (dotimes (i (length dr)) (setf (aref dr i) fill-value)))
        ;; walk the INPUT's index space; the output offset moves by the output's
        ;; strides, starting at the corner the begin-pads put us in
        (let ((base 0)
              (strides (make-array (max 1 rank) :element-type 'fixnum :initial-element 0)))
          (loop for b in begin for k from 0
                do (incf base (* b (aref ostrides k)))
                   (setf (aref strides k) (aref ostrides k)))
          (let ((zero (zero-strides rank)))
            (do-broadcast ((io strides) (ignored zero) shape)
              (progn ignored (setf (aref dr (+ base io)) (aref da i)))))))
      out)))

;;; ---- type and sequence -----------------------------------------------------

(defun onnx-dtype (code)
  "ONNX TensorProto element type codes."
  (case code
    (1 :f32) (6 :i32) (7 :i64) (9 :bool) (11 :f64)
    (t (error "Cast to ONNX dtype ~d is not implemented" code))))

(defmacro %convert (x from to)
  "The conversion ONNX means by Cast, with both types known at compile time."
  (let ((float-p (member to '(:f32 :f64)))
        (int-from (member from '(:i64 :i32 :bool))))
    (cond ((eq to :bool) `(if (zerop ,x) 0 1))
          (float-p `(float ,x ,(if (eq to :f32) 1.0f0 1.0d0)))
          ;; float -> int truncates toward zero, per the spec's C-cast semantics
          (int-from `(the ,(dtype-element-type to) ,x))
          (t `(the ,(dtype-element-type to) (truncate ,x))))))

(defop "Cast" (node ins)
  (let* ((a (first ins))
         (to (onnx-dtype (node-attr node "to")))
         (out (make-tensor to (tensor-shape a))))
    (if (eq to (tensor-dtype a))
        (copy-tensor a)
        (macrolet ((dispatch ()
                     `(ecase (tensor-dtype a)
                        ,@(loop for from in '(:f32 :f64 :i64 :i32 :bool)
                                collect `(,from
                                          (let ((da (the (simple-array ,(dtype-element-type from) (*))
                                                         (tensor-data a))))
                                            (ecase to
                                              ,@(loop for dst in '(:f32 :f64 :i64 :i32 :bool)
                                                      collect `(,dst
                                                                (let ((dr (the (simple-array ,(dtype-element-type dst) (*))
                                                                               (tensor-data out))))
                                                                  (dotimes (i (length da))
                                                                    (setf (aref dr i)
                                                                          (%convert (aref da i) ,from ,dst)))))))))))))
          (dispatch)
          out))))

(defop "Range" (node ins)
  (destructuring-bind (start limit delta) ins
    (let* ((s (tensor-scalar start)) (l (tensor-scalar limit)) (d (tensor-scalar delta))
           (n (max 0 (ceiling (- l s) d)))
           (out (make-tensor (tensor-dtype start) (vector n))))
      (with-typed-data (dr out)
        (dotimes (i n) (setf (aref dr i) (+ s (* i d)))))
      out)))

(defop "Identity" (node ins) (first ins))
