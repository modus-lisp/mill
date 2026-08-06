;;;; gemm-gate.lisp — Gemm against the definition, at shapes that can disagree.
;;;;
;;;;     sbcl --non-interactive --load inspect/gemm-gate.lisp
;;;;
;;;; Gemm is alpha*A'*B' + beta*C, with either operand optionally transposed and
;;;; C broadcast over the result.  mill writes it as MatMul plus that fold, so
;;;; what is worth checking is not the matrix product — the node gates cover that
;;;; on two real models — but the fold: the transposes, the two scalars, and the
;;;; shape C is allowed to arrive in.
;;;;
;;;; Every case here is NON-SQUARE, and that is the whole design of the sweep.  A
;;;; transpose written with the wrong extent is still a transpose when the matrix
;;;; is square: it indexes the same elements in the same order, and every square
;;;; test passes.  mill shipped exactly that bug.  It survived the zipformer
;;;; decoder, whose projection is 512x512, and died on the joiner's 500x512 —
;;;; twenty logits out of place, which is a different word.
;;;;
;;;; The comparison is exact.  The reference accumulates in the same order as
;;;; MATMUL-CORE (over k, with the zero-skip that changes nothing arithmetically),
;;;; so any difference is a wrong element rather than rounding.

(require :asdf)
(asdf:load-system :mill)

(in-package #:mill)

(defun spread (n seed)
  "A deterministic spread of values.  Distinct per position, so an index that is
off by anything at all lands on a different number."
  (let ((a (make-array n :element-type 'single-float))
        (s seed))
    (dotimes (i n a)
      (setf s (mod (+ (* s 1103515245) 12345) 2147483648))
      (setf (aref a i) (- (/ (float (mod s 4001) 1.0) 1000.0) 2.0)))))

(defun ref-gemm (a b c m k n alpha beta trans-a trans-b c-shape)
  "The definition, one output element at a time.  A is (M K) unless TRANS-A, in
which case it arrived as (K M); likewise B.  C broadcasts over the (M N) result
from whatever shape C-SHAPE is."
  (let ((r (make-array (* m n) :element-type 'single-float)))
    (dotimes (i m r)
      (dotimes (j n)
        (let ((acc 0.0))
          (declare (type single-float acc))
          (dotimes (kk k)
            (let ((av (aref a (if trans-a (+ (* kk m) i) (+ (* i k) kk))))
                  (bv (aref b (if trans-b (+ (* j k) kk) (+ (* kk n) j)))))
              ;; MATMUL-CORE skips a zero row of A, which drops a term that would
              ;; have been zero; the sum is the same either way
              (unless (zerop av) (setf acc (+ acc (* av bv))))))
          (setf (aref r (+ (* i n) j)) (* alpha acc))
          (when c
            (let ((ci (ecase (length c-shape)
                        (0 0)
                        (1 (if (= (aref c-shape 0) 1) 0 j))
                        (2 (+ (* (if (= (aref c-shape 0) 1) 0 i) (aref c-shape 1))
                              (if (= (aref c-shape 1) 1) 0 j))))))
              (incf (aref r (+ (* i n) j)) (* beta (aref c ci))))))))))

(defun run-gemm (m k n alpha beta trans-a trans-b c-shape seed)
  ;; masked throughout, not just around the op: a wrong index can read a NaN out
  ;; of a recycled buffer, and a gate that dies on the comparison instead of
  ;; reporting it is a gate that cannot tell you what went wrong
  (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
    (let* ((ashape (if trans-a (vector k m) (vector m k)))
           (bshape (if trans-b (vector n k) (vector k n)))
           (ad (spread (* m k) seed))
           (bd (spread (* k n) (+ seed 7919)))
           (cd (and c-shape (spread (max 1 (shape-size c-shape)) (+ seed 104729))))
           (a (make-tensor :f32 ashape))
           (b (make-tensor :f32 bshape))
           (c (and c-shape (make-tensor :f32 c-shape)))
           (node (%make-node "Gemm" "t" #() #()
                             (list (list "alpha" :float alpha)
                                   (list "beta" :float beta)
                                   (list "transA" :int (if trans-a 1 0))
                                   (list "transB" :int (if trans-b 1 0)))))
           (desc (format nil "Gemm ~dx~d * ~dx~d~:[~;, A'~]~:[~;, B'~] alpha=~a beta=~a C=~a"
                         m k k n trans-a trans-b alpha beta
                         (and c-shape (coerce c-shape 'list)))))
      (replace (tensor-data a) ad)
      (replace (tensor-data b) bd)
      (when c (replace (tensor-data c) cd))
      (let ((got (first (funcall (gethash "Gemm" *op-table*)
                                 node (if c (list a b c) (list a b)))))
            (want (ref-gemm ad bd cd m k n alpha beta trans-a trans-b c-shape)))
        (cond ((not (equalp (tensor-shape got) (vector m n)))
               (values (* m n) (format nil "~a — SHAPE ~a, expected (~d ~d)"
                                       desc (coerce (tensor-shape got) 'list) m n)))
              (t (let ((bad 0))
                   (dotimes (i (* m n))
                     (unless (= (aref (tensor-data got) i) (aref want i)) (incf bad)))
                   (values bad desc))))))))

(defun main ()
  (let ((cases 0) (failures '()) (seed 1))
    ;; No M = K = N anywhere: the three extents differ in every case, so an index
    ;; expression that confuses two of them cannot come out right by luck
    (dolist (dims '((1 512 500) (3 512 500) (500 512 3) (2 5 7) (7 5 2)
                    (1 1 9) (9 1 1) (4 9 1) (13 3 8)))
      (dolist (trans '((nil nil) (t nil) (nil t) (t t)))
        (dolist (scal '((1.0 1.0) (0.5 1.0) (1.0 2.0) (2.5 -0.25) (1.0 0.0)))
          (destructuring-bind (m k n) dims
            (dolist (c-shape (list nil (vector n) (vector 1) (vector m n)
                                   (vector m 1) (vector 1 n) #()))
              (multiple-value-bind (bad desc)
                  (run-gemm m k n (first scal) (second scal)
                            (first trans) (second trans) c-shape (incf seed 13))
                (incf cases)
                (when (plusp bad)
                  (push (format nil "~a — ~d elements differ" desc bad) failures))))))))
    (format t "~&~%ran ~d cases~%" cases)
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
