;;;; subgraph-gate.lisp — the scope chain, and the two hazards it creates.
;;;;
;;;;     sbcl --non-interactive --load inspect/subgraph-gate.lisp
;;;;
;;;; An If runs a graph out of its own attributes, and that graph reads whatever
;;;; it did not compute itself straight out of the scope containing the If — by
;;;; name, with nothing declared.  Two things in the runner have to know that,
;;;; and neither is visible in the answer a model gives:
;;;;
;;;;   The lifetimes.  A value's last reader may be inside a branch, several
;;;;   nodes after the last plain node that mentioned it.  Count only the plain
;;;;   readers and the runner drops the value — hands its buffer to the pool —
;;;;   before the branch that needs it runs.  This is not hypothetical: it is
;;;;   what the zipformer decoder did the first time it ran here.
;;;;
;;;;   The constants.  Constant hands back an attribute of the model, and an
;;;;   attribute outlives the run while the value bound to it is an ordinary
;;;;   intermediate the pool may recycle.  Return the attribute itself and the
;;;;   first run quietly rewrites the model for the second.
;;;;
;;;; Neither shows up on a model whose branches happen to be shallow and whose
;;;; constants happen to outlive their last reader, which is most of them — so
;;;; the graphs here are built to make both fire.  The check is exact: these are
;;;; questions of which bytes are where, not of summation order.

(require :asdf)
(asdf:load-system :mill)

(in-package #:mill)

(defun mk (op name ins outs &optional attrs)
  (%make-node op name (coerce ins 'simple-vector) (coerce outs 'simple-vector) attrs))

(defun const-attr (tensor) (list (list "value" :tensor tensor)))

(defun build-model (nodes &key inputs outputs initializers)
  (let ((m (%make-model)))
    (setf (model-inputs m) inputs
          (model-outputs m) outputs
          (model-nodes m) (coerce nodes 'simple-vector))
    (loop for (name . tensor) in initializers
          do (setf (gethash name (model-initializers m)) tensor))
    m))

(defparameter *failures* 0)

(defun check (name got want)
  (let ((ok (and (tensor-p got) (equalp (tensor-shape got) (tensor-shape want))
                 (eql 0d0 (tensor-max-abs-diff got want)))))
    (unless ok (incf *failures*))
    (format t "  ~:[FAIL~;ok  ~]  ~a~%" ok name)
    (unless ok
      (format t "          got:    ") (print-tensor got :limit 8)
      (format t "          wanted: ") (print-tensor want :limit 8))
    ok))

(defun check-true (name value)
  (unless value (incf *failures*))
  (format t "  ~:[FAIL~;ok  ~]  ~a~%" (and value t) name))

;;; ---- a model whose branch reads a value the plain nodes are done with ------
;;;
;;; "a" is produced by node 0 and last mentioned by a plain node at node 1.  The
;;; If is node 3, and both of its branches read "a".  A runner that retires "a"
;;; after node 1 cannot run either branch.

(defun branching-model ()
  (let ((k (tensor-from-list :f32 #(3) '(100 200 300))))
    (build-model
     (list (mk "Relu" "relu" '("x") '("a"))
           ;; the last PLAIN reader of "a" — and nothing here wants its answer
           (mk "Shape" "shape" '("a") '("ignored"))
           (mk "Constant" "one" '() '("one") (const-attr (tensor-from-list :f32 #() '(1))))
           (mk "If" "branch" '("cond") '("out")
               (list (list "then_branch" :graph
                           (%make-subgraph
                            (vector (mk "Identity" "b/id" '("a") '("t0")))
                            '("t0") (make-hash-table :test #'equal)))
                     (list "else_branch" :graph
                           ;; reads an outer value, an outer constant, and an
                           ;; initializer of its own, all by name
                           (%make-subgraph
                            (vector (mk "Add" "b/add" '("a" "k") '("e0"))
                                    (mk "Mul" "b/mul" '("e0" "one") '("e1")))
                            '("e1")
                            (let ((h (make-hash-table :test #'equal)))
                              (setf (gethash "k" h) k) h))))))
     :inputs '(("x" :f32 #(3)) ("cond" :bool #()))
     :outputs '(("out" :f32 #(3))))))

(defun run-branch (cond-value)
  (let* ((m (branching-model))
         (x (tensor-from-list :f32 #(3) '(-1 2 -3)))
         (c (tensor-from-list :bool #() (list cond-value))))
    (run-model m (list (cons "x" x) (cons "cond" c)))))

;;; ---- a model that runs a constant into the pool and then reuses it ---------
;;;
;;; "c" is dead the moment "sum" is computed, so the runner offers its buffer to
;;; the pool, and the next tensor of that size and type takes it.  If "c" was the
;;; attribute rather than a copy of it, the second run reads whatever the first
;;; run left there.

(defun constant-model ()
  (build-model
   (list (mk "Constant" "c" '() '("c") (const-attr (tensor-from-list :f32 #(4) '(1 -2 3 -4))))
         ;; "c" is dead here, and its buffer is the only free one of this size
         (mk "Add" "sum" '("x" "c") '("sum"))
         ;; so this lands in it — and squaring is chosen so that what lands is
         ;; nothing like what was there.  With RELU it would have written the
         ;; constant back over itself and the run below would have agreed for
         ;; the wrong reason.
         (mk "Mul" "square" '("sum" "sum") '("square"))
         (mk "Identity" "out" '("square") '("out")))
   :inputs '(("x" :f32 #(4)))
   :outputs '(("out" :f32 #(4)))))

(defun run-constant ()
  (let ((m (constant-model))
        (x (tensor-from-list :f32 #(4) '(0 0 0 0))))
    ;; The same model twice, which is the whole point.  Copied on the way out:
    ;; a run's tensors are only the caller's until the next run, which may hand
    ;; their buffers to something else — and an answer that changes underneath
    ;; the comparison would make the two runs agree for no reason.
    (list (copy-tensor (model-output-tensor m (run-model m (list (cons "x" x)))))
          (copy-tensor (model-output-tensor m (run-model m (list (cons "x" x))))))))

;;; ---- Tile, against the definition ------------------------------------------

(defun reference-tile (a reps)
  (let* ((shape (tensor-shape a))
         (out-shape (map 'vector #'* shape (coerce reps 'vector)))
         (out (make-tensor (tensor-dtype a) out-shape))
         (istrides (row-major-strides shape))
         (rank (length shape))
         (idx (make-array (max 1 rank) :element-type 'fixnum :initial-element 0)))
    ;; walk the output's multi-index; each coordinate wraps into the input
    (dotimes (i (tensor-size out) out)
      (let ((src 0))
        (dotimes (k rank) (incf src (* (mod (aref idx k) (aref shape k)) (aref istrides k))))
        (setf (aref (tensor-data out) i) (aref (tensor-data a) src)))
      (loop for k from (1- rank) downto 0
            do (incf (aref idx k))
               (if (< (aref idx k) (aref out-shape k)) (return) (setf (aref idx k) 0))))))

(defun check-tile (shape reps)
  (let* ((a (make-tensor :f32 (coerce shape 'vector)))
         (node (mk "Tile" "tile" '("a" "reps") '("out"))))
    (dotimes (i (tensor-size a)) (setf (aref (tensor-data a) i) (float (+ 1 i) 1f0)))
    (check (format nil "Tile ~a by ~a" shape reps)
           (op-tile node (list a (tensor-from-list :i64 (vector (length reps)) reps)))
           (reference-tile a reps))))

;;; ---- run -------------------------------------------------------------------

(format t "~&scope chain~%")
(let ((then-vals (run-branch 1))
      (else-vals (run-branch 0))
      (x (tensor-from-list :f32 #(3) '(-1 2 -3))))
  (declare (ignore x))
  (check "then branch reads an outer value retired by the plain nodes"
         (gethash "out" then-vals)
         (tensor-from-list :f32 #(3) '(0 2 0)))
  (check "else branch reads an outer value, an outer constant, and its own initializer"
         (gethash "out" else-vals)
         (tensor-from-list :f32 #(3) '(100 202 300)))
  (check-true "the branch's own names do not escape into the enclosing scope"
              (and (null (gethash "t0" then-vals))
                   (null (gethash "e0" else-vals))
                   (null (gethash "e1" else-vals))))
  (check-true "the branch not taken leaves nothing behind"
              (null (gethash "e0" then-vals))))

(format t "~&constants~%")
(destructuring-bind (first-run second-run) (run-constant)
  (check "a second run of the same model answers the same"
         second-run first-run)
  (check "and answers what the constant says" first-run
         (tensor-from-list :f32 #(4) '(1 4 9 16))))


(format t "~&Tile~%")
(check-tile '(3) '(4))
(check-tile '(2 3) '(3 1))
(check-tile '(2 3) '(1 2))
(check-tile '(2 1 3) '(2 4 2))
(check-tile '(1 1) '(5 5))

(format t "~%~:[GATE GREEN — scopes, lifetimes and constants all hold~;~:*~d CHECK~:P FAILED~]~%"
        (if (zerop *failures*) nil *failures*))
(sb-ext:exit :code (if (zerop *failures*) 0 1))
