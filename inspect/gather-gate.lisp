;;;; gather-gate.lisp — the strided copy against its definition.
;;;;
;;;;     sbcl --non-interactive --load inspect/gather-gate.lisp
;;;;
;;;; TENSOR-GATHER-STRIDED is the one copy loop behind Slice, Transpose, Expand
;;;; and Split, and it is written as three cases — a block move, a typed strided
;;;; gather, and an odometer over whatever is left — chosen by
;;;; COLLAPSE-GATHER-AXES rewriting the walk into fewer axes.  That rewrite is
;;;; the part worth a gate.  It claims two things: an axis of extent 1 can be
;;;; dropped, and two adjacent axes can be fused when the outer one's stride is
;;;; exactly the inner one's span.  Both are easy to believe and easy to get
;;;; wrong by one axis, and the failure mode is the one ops-shape.lisp warns
;;;; about at the top of the file — a tensor of the right shape full of numbers
;;;; from the wrong places, which still sounds like speech.
;;;;
;;;; The node gate cannot cover this.  It runs one voice, so it only ever sees
;;;; the handful of (shape, strides) triples that voice happens to use — seven
;;;; distinct transposes and a dozen slices.  Nothing exercises a negative
;;;; stride into a collapsed axis, or a broadcast stride of 0 next to a
;;;; contiguous tail, or rank 5.
;;;;
;;;; The reference is the definition itself: walk the output's multi-index and
;;;; take BASE + sum of index*stride.  No collapsing, no cases.  Comparison is
;;;; EXACT and there is no tolerance argument to have — this op copies numbers
;;;; and does no arithmetic to them, so a single differing element is a bug.

(require :asdf)
(asdf:load-system :mill)

(in-package #:mill)

;; seeded, so a failure reported by this gate is a failure someone else can get
(defparameter *rng* (sb-ext:seed-random-state 20260806))

(defun rnd (n) (random n *rng*))

(defun reference-gather (a base strides out-shape)
  "A + BASE read through STRIDES, straight from the definition: the element at
output multi-index (i0 i1 ...) is A[BASE + i0*s0 + i1*s1 + ...]."
  (let* ((rank (length out-shape))
         (n (shape-size (coerce out-shape 'simple-vector)))
         (out (make-tensor (tensor-dtype a) out-shape))
         (idx (make-list rank :initial-element 0))
         (da (tensor-data a))
         (dr (tensor-data out)))
    (dotimes (i n)
      (setf (aref dr i)
            (aref da (+ base (loop for k below rank
                                   sum (* (nth k idx) (aref strides k))))))
      ;; odometer, last axis fastest
      (loop for k downfrom (1- rank) to 0
            do (incf (nth k idx))
               (if (< (nth k idx) (aref out-shape k)) (return) (setf (nth k idx) 0))))
    out))

(defun reachable-p (size base strides out-shape)
  "Does every index this walk produces land inside a SIZE-element vector?  The
extremes are the only ones that can miss: each axis contributes its whole span in
one direction or the other."
  (let ((lo base) (hi base))
    (dotimes (k (length out-shape))
      (let ((span (* (aref strides k) (1- (aref out-shape k)))))
        (if (minusp span) (incf lo span) (incf hi span))))
    (and (>= lo 0) (< hi size))))

(defun random-case ()
  "An input tensor and a (base, strides, out-shape) walk over it that is in
bounds.  Strides are drawn from what the four ops actually produce — a row-major
stride times a step, 0 for a broadcast axis, negative for a reversed slice — but
the axes are shuffled and the extents are free, so most of what comes out is a
walk no voice would ever ask for."
  (loop
    (let* ((rank (+ 1 (rnd 4)))
           (in-shape (coerce (loop repeat rank collect (+ 1 (rnd 5))) 'vector))
           (size (shape-size (coerce in-shape 'simple-vector)))
           (istrides (row-major-strides in-shape))
           (orank (+ 1 (rnd 4)))
           (out-shape (make-array orank :element-type 'fixnum))
           (strides (make-array orank :element-type 'fixnum)))
      (dotimes (k orank)
        (let ((src (rnd rank)))
          (setf (aref out-shape k) (+ 1 (rnd 5))
                (aref strides k) (case (rnd 5)
                                   (0 0)                       ; broadcast
                                   (1 (- (aref istrides src))) ; reversed
                                   (2 (* 2 (aref istrides src))) ; step 2
                                   (t (aref istrides src))))))
      (let ((base (rnd size))
            (a (make-tensor :f32 in-shape)))
        (when (reachable-p size base strides out-shape)
          ;; distinct values, so a copy from the wrong place cannot look right
          (dotimes (i size) (setf (aref (tensor-data a) i) (float (+ 1 i) 1f0)))
          (return (list a base strides out-shape)))))))

(defun same-p (x y)
  (and (equalp (tensor-shape x) (tensor-shape y))
       (let ((dx (tensor-data x)) (dy (tensor-data y)))
         (dotimes (i (length dx) t)
           (unless (= (aref dx i) (aref dy i)) (return nil))))))

;;; ---- the sweep --------------------------------------------------------------

(defparameter *cases* 20000)

(let ((bad 0) (collapsed 0) (blocked 0))
  (dotimes (c *cases*)
    (destructuring-bind (a base strides out-shape) (random-case)
      (multiple-value-bind (ext st r) (collapse-gather-axes out-shape strides)
        (declare (ignore ext))
        (when (< r (length out-shape)) (incf collapsed))
        (when (and (plusp r) (= 1 (aref st (1- r)))) (incf blocked)))
      (let ((got (tensor-gather-strided a base strides out-shape))
            (want (reference-gather a base strides out-shape)))
        (unless (same-p got want)
          (incf bad)
          (when (<= bad 5)
            (format t "~&MISMATCH  in ~a base ~a strides ~a -> out ~a~%  got  ~a~%  want ~a~%"
                    (coerce (tensor-shape a) 'list) base
                    (coerce strides 'list) (coerce out-shape 'list)
                    (coerce (tensor-data got) 'list)
                    (coerce (tensor-data want) 'list)))))))
  (format t "~&~%ran ~d random walks: ~d collapsed to fewer axes, ~d ended in a~
             ~% block move~%" *cases* collapsed blocked)
  ;; the fixed cases are the shapes this voice actually runs, kept by name so a
  ;; regression on the ones that matter is named rather than buried in a count
  (format t "~%the shapes en_US-lessac-medium uses:~%")
  (dolist (spec '(((1 384 388) 0 (1 192 388) (148992 388 1)   "Split, channel axis")
                  ((1 384 388) 74496 (1 192 388) (148992 388 1) "Split, upper half")
                  ((1 192 198) 0 (1 198 192) (38016 1 198)    "Transpose 0 2 1")
                  ((1 2 198 396) 0 (1 2 198 395) (156816 78408 396 1) "Slice, last axis")
                  ;; a reversed slice starts at the far end and walks back, so
                  ;; BASE is the last row rather than 0 — getting that wrong is
                  ;; how this list was wrong the first time it ran
                  ((1 192 388) 74108 (1 192 388) (74496 -388 1) "Slice, reversed")
                  ((1 388 192) 0 (1 192 388) (74496 1 192)    "Transpose, wide")
                  ((1 1 198 388) 0 (1 1 388 198) (76824 76824 1 388) "Transpose rank 4")))
    (destructuring-bind (ish base osh st name) spec
      (let* ((a (make-tensor :f32 (coerce ish 'vector)))
             (n (tensor-size a))
             (strides (make-array (length st) :element-type 'fixnum
                                              :initial-contents st))
             (out-shape (coerce osh 'vector)))
        (dotimes (i n) (setf (aref (tensor-data a) i) (float (+ 1 i) 1f0)))
        (cond ((not (reachable-p n base strides out-shape))
               (incf bad)
               (format t "  BAD SPEC  ~a — walks outside the input~%" name))
              (t
               (let ((ok (same-p (tensor-gather-strided a base strides out-shape)
                                 (reference-gather a base strides out-shape))))
                 (unless ok (incf bad))
                 (format t "  ~:[FAIL~;ok  ~]  ~a~%" ok name)))))))
  (format t "~%~:[GATE GREEN — every walk copies exactly what the definition says~;~:*~d WALKS DIFFER~]~%"
          (if (zerop bad) nil bad)))

(sb-ext:exit)
