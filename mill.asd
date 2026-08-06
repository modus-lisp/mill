;;;; mill — an ONNX graph interpreter in pure Common Lisp.
;;;;
;;;; No FFI, no foreign libraries, no C.  A graph exported by tools/export-onnx.py
;;;; is executed node for node, in the order it was exported, so that every one of
;;;; its intermediate values can be compared against onnxruntime's answer for the
;;;; same node.  That is what makes the arithmetic in here checkable at all.

(asdf:defsystem #:mill
  :description "An ONNX graph interpreter in pure Common Lisp."
  :author "ynniv"
  :license "MIT"
  :serial t
  :components ((:module "src"
                :serial t
                :components ((:file "packages")
                             (:file "tensor")
                             (:file "simd")
                             (:file "parallel")
                             (:file "model")
                             (:file "graph")
                             (:file "ops-shape")
                             (:file "ops-math")
                             (:file "ops-reduce")
                             (:file "ops-index")
                             (:file "ops-nn")))))

;;; A .npy reader, so a gate can compare against values numpy wrote.  Its own
;;; system because nothing in the engine needs it — only the things that check
;;; the engine do, and they live in other repositories.
(asdf:defsystem #:mill/npy
  :description "Read numpy .npy files as mill tensors."
  :author "ynniv"
  :license "MIT"
  :depends-on (#:mill)
  :serial t
  :components ((:module "src" :components ((:file "npy")))))
