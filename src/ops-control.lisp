;;;; ops-control.lisp — the ops that run a graph instead of doing arithmetic.
;;;;
;;;; Everything else in mill takes tensors and returns tensors.  These take a
;;;; graph out of their own attributes and run it, which is why they are apart:
;;;; they are the only ops that reach back into the interpreter, and the only
;;;; ones whose cost is a whole subgraph rather than a kernel.
;;;;
;;;; They show up in exported models wherever the tracer met a Python branch it
;;;; could not fold away — a decoder that squeezes a batch axis when the batch is
;;;; one and leaves it alone otherwise, say.  The branch it takes is decided at
;;;; run time by a value the graph computed, so the shape it produces is too.

(in-package #:mill)

(defop "If" (node ins)
  (let* ((which (if (zerop (tensor-scalar (first ins))) "else_branch" "then_branch"))
         (branch (or (node-attr node which)
                     (error "If ~a has no ~a" (node-name node) which)))
         ;; the branch reads the enclosing scope by name and returns tensors;
         ;; everything else it computed goes away with its scope
         (outs (run-subgraph branch)))
    (unless (= (length outs) (length (node-outputs node)))
      (error "If ~a's ~a returned ~d value~:p, but the node declares ~d output~:p"
             (node-name node) which (length outs) (length (node-outputs node))))
    outs))
