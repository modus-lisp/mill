;;;; packages.lisp — mill, an ONNX graph interpreter in pure Common Lisp.
(defpackage #:mill
  (:use #:cl)
  (:export
   ;; tensors
   #:tensor #:tensor-p #:make-tensor #:tensor-data #:tensor-shape #:tensor-dtype
   #:tensor-size #:tensor-rank #:tensor-dim #:tensor-ref #:tensor-from-list
   #:tensor-scalar #:tensor-integers
   #:tensor-equal-p #:tensor-max-abs-diff #:print-tensor
   ;; reach a tensor's data at its own element type, which is what anything
   ;; writing numeric code against a tensor ends up needing
   #:with-typed-data
   ;; how many cores the kernels use, how much memory the pool may hold onto,
   ;; and how to give the cores back
   #:*worker-count* #:*tensor-pool-budget* #:shutdown-workers
   ;; models
   #:model #:load-model #:model-nodes #:model-initializers #:model-inputs #:model-outputs
   #:model-config #:model-opset
   #:node #:node-op #:node-name #:node-inputs #:node-outputs #:node-attrs #:node-attr
   ;; execution
   #:run-model #:model-output-tensor #:*op-table* #:defop #:op-implemented-p
   #:unimplemented-op #:model-missing-ops
   #:*noise-seed* #:reset-noise #:random-normal
   ;; only bound if the mill/npy system was loaded; named here so a gate in
   ;; another package can reach it without reaching into mill's internals
   #:read-npy))
