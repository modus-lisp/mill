;;;; conv-gate.lisp — Conv and ConvTranspose against a naive direct convolution.
;;;;
;;;;     sbcl --dynamic-space-size 4096 --load inspect/conv-gate.lisp
;;;;
;;;; inspect/node-gate.lisp already checks these ops against onnxruntime, but only
;;;; at the shapes this one voice happens to use: stride 1, a handful of kernel
;;;; widths, symmetric padding.  The kernel has fast paths that switch on exactly
;;;; those attributes — the fused path needs stride 1 and splits the output into a
;;;; padded end, an interior where every tap reads a real sample, and another
;;;; padded end — so the combinations the graph does not contain are the ones with
;;;; no coverage at all, and they are where an off-by-one in the split would hide.
;;;;
;;;; This sweeps attributes instead of trusting the graph: strides, dilations,
;;;; asymmetric and zero and oversized padding, kernel widths on both sides of the
;;;; fused specializations, and grouped convolution.  The reference is the
;;;; definition written out with a bounds test per tap, which is slow and obviously
;;;; correct.
;;;;
;;;; The comparison is exact.  Both sides accumulate in the same order, so any
;;;; difference at all is a bug rather than rounding, and saying so here is worth
;;;; more than a tolerance that would also swallow a real one.

(require :asdf)
(asdf:load-system :mill)

(in-package #:mill)

(defun ref-conv1d (x w bias batch cin len cout cgroup kw out-len
                   stride dilation pad-begin group)
  "The definition of a 1-D convolution, one output element at a time."
  (let ((r (make-array (* batch cout out-len) :element-type 'single-float
                                              :initial-element 0.0))
        (per-group-out (floor cout group)))
    (dotimes (nb batch r)
      (dotimes (m cout)
        (let ((gr (floor m per-group-out)))
          (dotimes (o out-len)
            (let ((acc (if bias (aref bias m) 0.0)))
              (declare (type single-float acc))
              (dotimes (c cgroup)
                (dotimes (kk kw)
                  (let ((src (+ (* o stride) (* kk dilation) (- pad-begin))))
                    (when (and (>= src 0) (< src len))
                      (incf acc (* (aref w (+ (* m cgroup kw) (* c kw) kk))
                                   (aref x (+ (* nb cin len)
                                              (* (+ (* gr cgroup) c) len)
                                              src))))))))
              (setf (aref r (+ (* nb cout out-len) (* m out-len) o)) acc))))))))

(defun ref-conv-transpose1d (x w bias batch cin len cout cgroup-in cgroup-out kw
                             out-len stride dilation pad-begin group)
  "A 1-D transposed convolution, one output element at a time.

Written as a gather rather than the scatter the operator is usually defined as,
for the sake of the comparison: the terms are the same either way, but a scatter
adds them to an output element in whatever order it happens to walk the input,
and floating-point addition is not associative.  Inverting the mapping — for a
given output tap and kernel position there is at most one input sample, the one
at (o + pad - k*dilation)/stride when that divides — pins the order to (input
channel, then kernel position), which is the order CONV-TRANSPOSE1D-CORE
accumulates in.  That is what lets this be compared exactly."
  (declare (ignore group))
  (let ((r (make-array (* batch cout out-len) :element-type 'single-float
                                              :initial-element 0.0)))
    (dotimes (nb batch r)
      (dotimes (ochan cout)
        (multiple-value-bind (g mm) (floor ochan cgroup-out)
          (dotimes (o out-len)
            (let ((acc (if bias (aref bias ochan) 0.0)))
              (declare (type single-float acc))
              (dotimes (ci cgroup-in)
                (let ((c (+ (* g cgroup-in) ci)))
                  (dotimes (kk kw)
                    (let ((num (- (+ o pad-begin) (* kk dilation))))
                      (when (and (>= num 0) (zerop (mod num stride)))
                        (let ((l (floor num stride)))
                          (when (< l len)
                            (incf acc (* (aref w (+ (* c cgroup-out kw) (* mm kw) kk))
                                         (aref x (+ (* nb cin len) (* c len) l)))))))))))
              (setf (aref r (+ (* nb cout out-len) (* ochan out-len) o)) acc))))))))

(defun filled (n seed)
  "A deterministic spread of values, including exact zeros so the zero-weight
skip in the unfused path is exercised."
  (let ((a (make-array n :element-type 'single-float))
        (s seed))
    (dotimes (i n a)
      (setf s (mod (+ (* s 1103515245) 12345) 2147483648))
      (setf (aref a i) (if (zerop (mod s 11))
                           0.0
                           (- (/ (float (mod s 2000) 1.0) 1000.0) 1.0))))))

(defun run-case (op batch cin len cout kw stride dilation pad-begin pad-end group
                 bias-p seed)
  "Build one node, run the real op and the reference, and return their mismatch
count and a description."
  (let* ((transposep (string= op "ConvTranspose"))
         (cgroup (floor cin group))
         (per-group-out (floor cout group))
         (out-len (if transposep
                      (+ (* (1- len) stride) (* dilation (1- kw)) 1
                         (- pad-begin) (- pad-end))
                      (1+ (floor (- (+ len pad-begin pad-end) (* dilation (1- kw)) 1)
                                 stride))))
         (desc (format nil "~a cin=~d cout=~d len=~d kw=~d s=~d d=~d pads=(~d ~d) g=~d~:[~; +bias~]"
                       op cin cout len kw stride dilation pad-begin pad-end group bias-p)))
    (when (< out-len 1) (return-from run-case (values :skipped desc)))
    (let* ((wn (if transposep (* cin per-group-out kw) (* cout cgroup kw)))
           (xd (filled (* batch cin len) seed))
           (wd (filled wn (+ seed 7919)))
           (bd (and bias-p (filled cout (+ seed 104729))))
           (x (make-tensor :f32 (vector batch cin len)))
           (w (make-tensor :f32 (if transposep
                                    (vector cin per-group-out kw)
                                    (vector cout cgroup kw))))
           (b (and bias-p (make-tensor :f32 (vector cout))))
           (node (%make-node op "t" #() #()
                             (list (list "kernel_shape" :ints (list kw))
                                   (list "strides" :ints (list stride))
                                   (list "dilations" :ints (list dilation))
                                   (list "pads" :ints (list pad-begin pad-end))
                                   (list "group" :int group)))))
      (replace (tensor-data x) xd)
      (replace (tensor-data w) wd)
      (when b (replace (tensor-data b) bd))
      (let* ((got (tensor-data
                   (first (sb-int:with-float-traps-masked
                              (:invalid :divide-by-zero :overflow :underflow)
                            (funcall (gethash op *op-table*)
                                     node (if b (list x w b) (list x w)))))))
             (want (if transposep
                       (ref-conv-transpose1d xd wd bd batch cin len cout
                                             cgroup per-group-out kw out-len
                                             stride dilation pad-begin group)
                       (ref-conv1d xd wd bd batch cin len cout cgroup kw out-len
                                   stride dilation pad-begin group)))
             (bad 0))
        (unless (= (length got) (length want))
          (return-from run-case
            (values (length want) (format nil "~a — LENGTH ~d, expected ~d"
                                          desc (length got) (length want)))))
        (dotimes (i (length want))
          (unless (= (aref got i) (aref want i)) (incf bad)))
        (values bad desc)))))

(defun main ()
  (let ((cases 0) (skipped 0) (failures '()) (seed 1))
    (dolist (op '("Conv" "ConvTranspose"))
      (dolist (kw '(1 2 3 5 7 8 9 11))
        (dolist (stride '(1 2 3 8))
          (dolist (dilation '(1 2 3 12))
            (dolist (pads '((0 0) (1 1) (3 3) (0 4) (5 1) (9 9) (36 36)))
              (dolist (group '(1 2))
                (dolist (len '(1 7 40 97))
                  (let ((cin (* group 3)) (cout (* group 2)))
                    (multiple-value-bind (bad desc)
                        (run-case op 2 cin len cout kw stride dilation
                                  (first pads) (second pads) group
                                  (oddp seed) (incf seed 13))
                      (cond ((eq bad :skipped) (incf skipped))
                            (t (incf cases)
                               (when (plusp bad)
                                 (push (format nil "~a — ~d elements differ" desc bad)
                                       failures)))))))))))))
    (format t "~&~%ran ~d cases, ~d skipped as having no output~%" cases skipped)
    (cond ((null failures)
           (format t "~%GATE GREEN — every case matches the definition exactly~%"))
          (t
           (format t "~%GATE RED — ~d cases differ:~%" (length failures))
           (dolist (f (reverse (subseq failures 0 (min 20 (length failures)))))
             (format t "  ~a~%" f))
           (when (> (length failures) 20)
             (format t "  ... and ~d more~%" (- (length failures) 20)))))
    (finish-output)
    (sb-ext:exit :code (if failures 1 0))))

(main)
