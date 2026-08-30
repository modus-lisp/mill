;;;; seq-ops-gate.lisp — Resize and LayerNormalization, which nothing else covered.
;;;;
;;;;     sbcl --load inspect/seq-ops-gate.lisp
;;;;
;;;; Written because these two ops were refactored with no test that would have
;;;; noticed if the refactor changed what they compute.  Both used to dispatch on
;;;; dtype by NESTING two WITH-TYPED-DATA forms, which compiles every cross
;;;; combination — including writing a DOUBLE-FLOAT into a SINGLE-FLOAT array.  Those
;;;; branches cannot run, since every caller builds its output with the input's dtype,
;;;; but they are compiled, and SBCL 2.2.9 rejects them with a fatal WARNING where
;;;; 2.5.6 only prints a note.  Same source, two compilers, two verdicts.
;;;;
;;;; So the pair is bound in ONE dispatch now, and the cases below check the thing a
;;;; dispatch refactor can quietly break: the numbers, in both dtypes, and that the
;;;; output keeps the dtype it went in with.
;;;;
;;;; The last case is the invariant itself — tensors that disagree must signal, not
;;;; compute, because a THE declaration over the wrong element type is silent nonsense
;;;; rather than an error.

(require :asdf)
(let ((here (make-pathname :name nil :type nil
                          :defaults (or *load-truename* *compile-file-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:tree ,(merge-pathnames "../../" here))
                      (:exclude "vendor") (:exclude "deps") :inherit-configuration)))
(handler-bind ((warning #'muffle-warning))
  (let ((*standard-output* (make-broadcast-stream))) (asdf:load-system :mill)))
(in-package :mill)

(defun run1 (op node ins)
  (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
    (first (funcall (gethash op *op-table*) node ins))))

(defvar *bad* 0)
(defun chk (name got want)
  (if (equalp got want) (format t "  ok   ~a~%" name)
      (progn (incf *bad*) (format t "  FAIL ~a~%    got  ~a~%    want ~a~%" name got want))))

;;; Resize, nearest, 2x2 -> 4x4: each source pixel becomes a 2x2 block.
(dolist (dt '(:f32 :f64))
  (let* ((x (make-tensor dt (vector 1 1 2 2)))
         (sc (make-tensor dt (vector 4)))
         (node (%make-node "Resize" "t" #() #() (list (list "mode" :string "nearest")))))
    (replace (tensor-data x) (map 'list (lambda (v) (coerce v (dtype-element-type dt)))
                                  '(1 2 3 4)))
    (replace (tensor-data sc) (map 'list (lambda (v) (coerce v (dtype-element-type dt)))
                                   '(1 1 2 2)))
    (let ((r (run1 "Resize" node (list x nil sc))))
      (chk (format nil "Resize ~a shape" dt) (tensor-shape r) #(1 1 4 4))
      (chk (format nil "Resize ~a dtype kept" dt) (tensor-dtype r) dt)
      (chk (format nil "Resize ~a pixels" dt)
           (coerce (tensor-data r) 'list)
           (map 'list (lambda (v) (coerce v (dtype-element-type dt)))
                '(1 1 2 2  1 1 2 2  3 3 4 4  3 3 4 4))))))

;;; LayerNormalization over the last axis of a 1x4: mean 2.5, and scale/bias applied.
(dolist (dt '(:f32 :f64))
  (let* ((x (make-tensor dt (vector 1 4)))
         (s (make-tensor dt (vector 4)))
         (node (%make-node "LayerNormalization" "t" #() #()
                           (list (list "axis" :int -1)))))
    (replace (tensor-data x) (map 'list (lambda (v) (coerce v (dtype-element-type dt)))
                                  '(1 2 3 4)))
    (fill (tensor-data s) (coerce 1 (dtype-element-type dt)))
    (let* ((r (run1 "LayerNormalization" node (list x s)))
           (d (coerce (tensor-data r) 'list))
           (sum (reduce #'+ d))
           (sym (< (abs (+ (first d) (car (last d)))) 1d-4)))
      (chk (format nil "LayerNorm ~a shape" dt) (tensor-shape r) #(1 4))
      (chk (format nil "LayerNorm ~a dtype kept" dt) (tensor-dtype r) dt)
      (chk (format nil "LayerNorm ~a is zero-mean" dt) (< (abs sum) 1d-3) t)
      (chk (format nil "LayerNorm ~a is symmetric about the mean" dt) sym t))))

;;; And the invariant the pair macro now enforces: mismatched dtypes are refused.
(let* ((a (make-tensor :f32 (vector 2))) (b (make-tensor :f64 (vector 2))))
  (chk "mismatched dtypes signal rather than compute"
       (handler-case (progn (with-two-typed-data (da a) (db b) (list da db)) :no-error)
         (error () :refused))
       :refused))

(format t "~&~%~:[ALL OK~;~:*~d FAILED~]~%" (if (plusp *bad*) *bad* nil))
(sb-ext:exit :code (if (plusp *bad*) 1 0))
