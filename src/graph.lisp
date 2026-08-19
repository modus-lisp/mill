;;;; graph.lisp — execute a model, node for node, in the order it was exported.
;;;;
;;;; ONNX graphs are already topologically sorted, so "execution" is a loop: look
;;;; up each node's inputs by name, call the op, bind its outputs by name.  There
;;;; is no scheduler and no fusion pass, and that is the point — the graph mill
;;;; runs is the graph onnxruntime ran, so every intermediate value has a golden
;;;; counterpart and a wrong answer names the exact node that first diverged.
;;;; Optimizing comes later, and it comes with that gate still in place.

(in-package #:mill)

(define-condition unimplemented-op (error)
  ((op :initarg :op :reader unimplemented-op-op)
   (node :initarg :node :initform nil :reader unimplemented-op-node))
  (:report (lambda (c s)
             (format s "no implementation for ONNX op ~a~@[ (node ~a)~]"
                     (unimplemented-op-op c)
                     (let ((n (unimplemented-op-node c))) (and n (node-name n)))))))

(defvar *op-table* (make-hash-table :test #'equal)
  "ONNX op type string -> (lambda (node inputs) ...) returning a list of tensors.
INPUTS holds one entry per declared input, NIL where the model omitted an
optional one; the list is as long as the node says, so ops can index positionally
the way the ONNX spec describes them.")

(defmacro defop (name (node inputs) &body body)
  "Define the implementation of ONNX op NAME.  BODY returns a tensor, or a list
of tensors for a multi-output op."
  (let ((fn (intern (concatenate 'string "OP-" (string-upcase name)))))
    `(progn
       (defun ,fn (,node ,inputs)
         (declare (ignorable ,node ,inputs))
         ,@body)
       (setf (gethash ,name *op-table*)
             (lambda (n i) (let ((r (,fn n i))) (if (listp r) r (list r)))))
       ,name)))

(defun op-implemented-p (op) (nth-value 1 (gethash op *op-table*)))

(defun model-missing-ops (model)
  "Op types the model uses that nothing implements yet, with how many nodes each
accounts for.  The work list, in the order worth doing it."
  (let ((counts (make-hash-table :test #'equal)))
    (loop for n across (model-nodes model)
          unless (op-implemented-p (node-op n))
            do (incf (gethash (node-op n) counts 0)))
    (sort (loop for op being the hash-keys of counts using (hash-value c)
                collect (cons op c))
          #'> :key #'cdr)))

;;; ---- scopes ----------------------------------------------------------------
;;;
;;; A flat graph needs no scope at all: every value has a name and the names are
;;; unique.  A subgraph does, because ONNX nests one graph inside another node's
;;; attributes and lets the inner one read the outer one's values by name without
;;; declaring anything — an If's branches are written against whatever is in
;;; scope where the If sits.  So the runner carries a chain of tables, innermost
;;; first: a lookup walks outward, and a binding always lands in the innermost,
;;; which is what makes a branch's intermediates disappear when it finishes.

(defvar *value-env* '()
  "Scopes, innermost first.  Bound by RUN-MODEL and pushed by RUN-SUBGRAPH.")

(defun env-get (name)
  "Values: the tensor bound to NAME in the nearest scope that has one, and
whether any scope did."
  (dolist (scope *value-env* (values nil nil))
    (multiple-value-bind (v hit) (gethash name scope)
      (when hit (return (values v t))))))

(defun apply-node (node i)
  "Run one node: look its inputs up in the scope chain, call the op, return the
list of outputs.  I is only ever used to say which node it was."
  (let ((fn (or (gethash (node-op node) *op-table*)
                (error 'unimplemented-op :op (node-op node) :node node)))
        (ins (map 'list
                  (lambda (name)
                    (if (string= name "")
                        nil
                        (multiple-value-bind (v hit) (env-get name)
                          (unless hit
                            (error "node ~a wants ~s, which no node has produced"
                                   (node-name node) name))
                          v)))
                  (node-inputs node))))
    (handler-case (funcall fn node ins)
      (unimplemented-op (c) (error c))
      (error (c)
        (error "~a (node ~d, ~a ~a): ~a"
               (type-of c) i (node-op node) (node-name node) c)))))

(defun run-subgraph (sub)
  "Run SUB in a fresh scope pushed onto the enclosing one, and return the tensors
its outputs name.  The scope goes away on the way out, and with it everything the
branch computed but did not return.

None of the lifetime machinery below applies here: a branch is a handful of
nodes, so its intermediates are left to the collector rather than pooled, and the
values that matter are the ones it hands back to the node that ran it."
  (let* ((scope (make-hash-table :test #'equal))
         (*value-env* (cons scope *value-env*)))
    (maphash (lambda (k v) (setf (gethash k scope) v)) (subgraph-initializers sub))
    (loop for node across (subgraph-nodes sub)
          for i from 0
          do (loop for name across (node-outputs node)
                   for o in (apply-node node i)
                   unless (string= name "") do (setf (gethash name scope) o)))
    (mapcar (lambda (name)
              (multiple-value-bind (v hit) (env-get name)
                (unless hit
                  (error "subgraph output ~s was never produced" name))
                v))
            (subgraph-outputs sub))))

;;; ---- lifetimes -------------------------------------------------------------
;;;
;;; A 2755-node graph holds ~110 MB of intermediates if nothing is ever dropped,
;;; which is fine on this box and is not fine on a Pi.  Every value's last use is
;;; known before the run starts, so the runner drops each one the moment the last
;;; node that reads it has run.  KEEP names the values to hold on to anyway (the
;;; gate keeps everything; synthesis keeps only the audio).

(defun node-subgraph-reads (node)
  "Value names read by the graphs nested in NODE's attributes.

A branch reads the enclosing scope by name, and those reads belong to the node
that owns the branch — not to the last ordinary node that happened to mention the
value.  Without this the runner drops a value at its last plain use and the
branch, running later, finds nothing there.

The list over-approximates: it includes the names a branch produces and consumes
entirely within itself, which exist in no outer scope.  That costs a few dead
entries in a table and nothing else, and the alternative — deciding which names
escape — is the same question the scope chain already answers at run time.

NIL for every op but If, after one pass over an attribute list."
  (let ((names '()))
    (labels ((walk (n)
               (loop for in across (node-inputs n)
                     unless (string= in "") do (push in names))
               (subgraphs n))
             (subgraphs (n)
               (loop for (nil kind value) in (node-attrs n)
                     when (eq kind :graph)
                       do (map nil #'walk (subgraph-nodes value)))))
      (subgraphs node))
    names))

(defun last-use-table (model)
  "Value name -> index of the last node that reads it."
  (let ((last (make-hash-table :test #'equal)))
    (loop for n across (model-nodes model)
          for i from 0
          do (loop for in across (node-inputs n)
                   unless (string= in "") do (setf (gethash in last) i))
             (dolist (in (node-subgraph-reads n)) (setf (gethash in last) i)))
    last))

;;; ---- the loop --------------------------------------------------------------

(defun run-model (model feeds &key keep-all (keep '()) trace (on-node nil))
  "Run MODEL over FEEDS, an alist or hash table of input name -> tensor.
Returns a hash table of value name -> tensor holding the graph outputs, plus
whatever KEEP names, or every intermediate when KEEP-ALL.  ON-NODE, if given, is
called with (index node outputs) as each node finishes — which is how the gate
compares against onnxruntime without the runner knowing anything about fixtures."
  (let* ((vals (make-hash-table :test #'equal :size 4096))
         (last (unless keep-all (last-use-table model)))
         (pin (make-hash-table :test #'equal))
         ;; How many live names hold each data vector.  Names, not tensors:
         ;; RESHAPE-VIEW and the identity-shaped ops hand the same vector to a
         ;; second name, and a vector is only free once every name holding it is
         ;; gone.  Pinned names are never dropped, so the weights and the graph
         ;; outputs simply never come back down to zero.
         (refs (make-hash-table :test #'eq :size 4096))
         (*tensor-pool* (unless keep-all (make-hash-table :test #'equal)))
         (*tensor-pool-bytes* 0)
         ;; the outermost scope, and the only one until a node runs a subgraph
         (*value-env* (list vals)))
    (flet ((bind (name tensor)
             (setf (gethash name vals) tensor)
             (when (tensor-p tensor)
               (incf (gethash (tensor-data tensor) refs 0))))
           (unbind (name)
             (multiple-value-bind (v hit) (gethash name vals)
               (when hit
                 (remhash name vals)
                 (when (tensor-p v)
                   (let ((d (tensor-data v)))
                     (when (zerop (decf (gethash d refs 1)))
                       (remhash d refs)
                       (pool-release (tensor-dtype v) d))))))))
      (dolist (name keep) (setf (gethash name pin) t))
      (dolist (o (model-outputs model)) (setf (gethash (first o) pin) t))
      ;; initializers are shared and never dropped: they are the weights
      (maphash (lambda (k v) (setf (gethash k pin) t) (bind k v))
               (model-initializers model))
      (flet ((feed (name tensor)
               (let ((want (find name (model-inputs model) :key #'first :test #'string=)))
                 (unless want (error "~s is not an input of this model" name))
                 (unless (eq (tensor-dtype tensor) (second want))
                   (error "input ~s wants ~a, got ~a" name (second want)
                          (tensor-dtype tensor))))
               (setf (gethash name pin) t)
               (bind name tensor)))
        (if (hash-table-p feeds)
            (maphash #'feed feeds)
            (loop for (name . tensor) in feeds do (feed name tensor))))
    ;; onnxruntime computes with IEEE defaults: dividing by zero gives an
    ;; infinity and the graph carries on.  SBCL traps by default, so a masked-off
    ;; region is what "the same arithmetic" actually means here.
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
      (loop for node across (model-nodes model)
          for i from 0
          do (let ((outs (if *op-times*
                             ;; two clock reads, and only when someone asked
                             (let ((entry (profile-entry-for node))
                                   (t0 (get-internal-real-time)))
                               (multiple-value-prog1 (apply-node node i)
                                 (incf (pe-ticks entry) (- (get-internal-real-time) t0))
                                 (incf (pe-calls entry))))
                             (apply-node node i))))
               (loop for name across (node-outputs node)
                     for o in outs
                     unless (string= name "")
                       do (bind name o))
               (when trace
                 (format *trace-output* "~&~5d ~a ~a -> ~{~a~^, ~}~%" i (node-op node)
                         (node-name node)
                         (loop for o in outs
                               collect (format nil "~a~a" (tensor-dtype o)
                                               (coerce (tensor-shape o) 'list)))))
               (when on-node (funcall on-node i node outs))
               (unless keep-all
                 ;; drop everything whose last reader was this node, a branch's
                 ;; reads included — a name a branch invented is not in VALS and
                 ;; UNBIND passes over it
                 (flet ((retire (name)
                          (unless (or (string= name "") (gethash name pin))
                            (let ((lu (gethash name last)))
                              (when (and lu (= lu i)) (unbind name))))))
                   (map nil #'retire (node-inputs node))
                   (mapc #'retire (node-subgraph-reads node))))))))
    vals))

(defun model-output-tensor (model vals &optional (which 0))
  (gethash (first (nth which (model-outputs model))) vals))
