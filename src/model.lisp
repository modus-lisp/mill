;;;; model.lisp — read what tools/export-onnx.py wrote.
;;;;
;;;; Two files: a .graph of s-expressions (topology, attributes, and where each
;;;; tensor lives) and a .bin of concatenated little-endian tensor data.  The
;;;; split is deliberate — the topology is small and worth being able to read, and
;;;; the weights are large and worth being able to load without parsing anything.

(in-package #:mill)

(defstruct (node (:constructor %make-node (op name inputs outputs attrs)))
  "One graph node, as the exporter emitted it.  OP is the ONNX op type string;
INPUTS and OUTPUTS are value names.  An empty input name means 'omitted optional
argument', which ONNX uses positionally and several ops care about."
  (op "" :type string)
  (name "" :type string)
  (inputs #() :type simple-vector)
  (outputs #() :type simple-vector)
  (attrs '() :type list))

(defun node-attr (node name &optional default)
  "Attribute NAME of NODE, or DEFAULT.  Values arrive already decoded: :ints is a
list, :tensor is a TENSOR."
  (let ((hit (assoc name (node-attrs node) :test #'string=)))
    (if hit (third hit) default)))

(defstruct (subgraph (:constructor %make-subgraph (nodes outputs initializers)))
  "A graph nested inside a node's attributes — the two branches of an If.

It declares no inputs, and that is not an omission: an ONNX subgraph of this form
reads whatever it did not compute itself straight out of the scope that contains
it, by name.  So there is nothing to bind on the way in, and OUTPUTS is the whole
of its interface on the way out."
  (nodes #() :type simple-vector)
  (outputs '() :type list)
  (initializers (make-hash-table :test #'equal) :type hash-table))

(defstruct (model (:constructor %make-model))
  (inputs '() :type list)               ; (name dtype shape) — shape may hold strings
  (outputs '() :type list)
  (initializers (make-hash-table :test #'equal) :type hash-table)  ; name -> tensor
  (nodes #() :type simple-vector)
  (opset 0)
  (config nil))                         ; the voice's JSON, when one shipped with it

;;; ---- decoding the blob -----------------------------------------------------
;;;
;;; The bytes are little-endian, which every machine this runs on already is; the
;;; decode is written out rather than blitted so that it stays correct on one that
;;; is not, and because at 15M parameters it costs a fraction of a second either
;;; way.  Load time is not the hot path — inference is.

(declaim (inline u32-at))
(defun u32-at (bytes at)
  (declare (type (simple-array (unsigned-byte 8) (*)) bytes) (type index at))
  (logior (aref bytes at)
          (ash (aref bytes (+ at 1)) 8)
          (ash (aref bytes (+ at 2)) 16)
          (ash (aref bytes (+ at 3)) 24)))

(defun decode-tensor (dtype shape bytes offset)
  (let* ((tn (make-tensor dtype shape))
         (n (tensor-size tn))
         (data (tensor-data tn)))
    (declare (type (simple-array (unsigned-byte 8) (*)) bytes))
    (ecase dtype
      (:f32 (let ((d (the (simple-array single-float (*)) data)))
              (dotimes (i n)
                (let ((u (u32-at bytes (+ offset (* 4 i)))))
                  (setf (aref d i)
                        (sb-kernel:make-single-float
                         (if (>= u #x80000000) (- u #x100000000) u)))))))
      (:f64 (let ((d (the (simple-array double-float (*)) data)))
              (dotimes (i n)
                (let* ((at (+ offset (* 8 i)))
                       (lo (u32-at bytes at))
                       (hi (u32-at bytes (+ at 4))))
                  (setf (aref d i)
                        (sb-kernel:make-double-float
                         (if (>= hi #x80000000) (- hi #x100000000) hi) lo))))))
      (:i64 (let ((d (the (simple-array (signed-byte 64) (*)) data)))
              (dotimes (i n)
                (let* ((at (+ offset (* 8 i)))
                       (u (logior (u32-at bytes at) (ash (u32-at bytes (+ at 4)) 32))))
                  (setf (aref d i) (if (>= u #x8000000000000000)
                                       (- u #x10000000000000000) u))))))
      (:i32 (let ((d (the (simple-array (signed-byte 32) (*)) data)))
              (dotimes (i n)
                (let ((u (u32-at bytes (+ offset (* 4 i)))))
                  (setf (aref d i) (if (>= u #x80000000) (- u #x100000000) u))))))
      (:bool (let ((d (the (simple-array (unsigned-byte 8) (*)) data)))
               (dotimes (i n) (setf (aref d i) (aref bytes (+ offset i)))))))
    tn))

(defun read-file-bytes (path)
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((b (make-array (file-length s) :element-type '(unsigned-byte 8))))
      (read-sequence b s)
      b)))

;;; ---- reading the graph -----------------------------------------------------

(defun parse-dtype (s)
  (cond ((string= s "f32") :f32) ((string= s "f64") :f64)
        ((string= s "i64") :i64) ((string= s "i32") :i32)
        ((string= s "bool") :bool)
        (t (error "unknown dtype ~s in graph file" s))))

(defun read-graph-form (path)
  "READ the graph file.  *READ-EVAL* is off: this is data, and a graph file is
something a build step produced, not something to hand the evaluator."
  (with-open-file (s path :external-format :utf-8)
    (let ((*read-eval* nil)
          (*package* (find-package '#:mill))
          (*read-default-float-format* 'double-float))
      (read s))))

(defun decode-attr (attr bytes)
  "(name kind value) from the graph file.  :tensor values become tensors and
:graph values become subgraphs; the rest — ints, floats, strings — are already
the Lisp objects the op wants."
  (destructuring-bind (name kind value) attr
    (list name kind
          (case kind
            (:tensor (destructuring-bind (tname dt shape offset nbytes) value
                       (declare (ignore tname nbytes))
                       (decode-tensor (parse-dtype dt) (coerce shape 'vector)
                                      bytes offset)))
            (:graph (decode-subgraph value bytes))
            (t value)))))

(defun decode-node (form bytes)
  "One (op name inputs outputs attrs) form as a NODE.  Shared with subgraphs,
which are nodes all the way down."
  (destructuring-bind (op name inputs outputs attrs) form
    (%make-node op name (coerce inputs 'simple-vector) (coerce outputs 'simple-vector)
                (mapcar (lambda (a) (decode-attr a bytes)) attrs))))

(defun decode-subgraph (form bytes)
  (destructuring-bind (&key initializers nodes outputs) form
    (let ((inits (make-hash-table :test #'equal)))
      (loop for (name dt shape offset nbytes) in initializers
            do (progn nbytes
                      (setf (gethash name inits)
                            (decode-tensor (parse-dtype dt) (coerce shape 'vector)
                                           bytes offset))))
      (%make-subgraph (map 'simple-vector (lambda (n) (decode-node n bytes)) nodes)
                      (coerce outputs 'list) inits))))

(defun load-model (graph-path &key bin-path config-path)
  "Load the model exported to GRAPH-PATH (and the .bin beside it)."
  (let* ((graph-path (pathname graph-path))
         (bin-path (or bin-path (make-pathname :type "bin" :defaults graph-path)))
         (config-path (or config-path
                          (let ((p (make-pathname
                                    :name (concatenate 'string (pathname-name graph-path)
                                                       ".config")
                                    :type "json" :defaults graph-path)))
                            (and (probe-file p) p))))
         (form (read-graph-form graph-path))
         (bytes (read-file-bytes bin-path))
         (m (%make-model)))
    (setf (model-opset m) (getf form :opset))
    (flet ((io (specs)
             (loop for (name dt shape) in specs
                   collect (list name (parse-dtype dt) (coerce shape 'vector)))))
      (setf (model-inputs m) (io (getf form :inputs))
            (model-outputs m) (io (getf form :outputs))))
    (loop for (name dt shape offset nbytes) in (getf form :initializers)
          do (progn nbytes
                    (setf (gethash name (model-initializers m))
                          (decode-tensor (parse-dtype dt) (coerce shape 'vector)
                                         bytes offset))))
    (setf (model-nodes m)
          (map 'simple-vector (lambda (n) (decode-node n bytes)) (getf form :nodes)))
    (when config-path
      (setf (model-config m) (read-json-file config-path)))
    m))

;;; ---- just enough JSON ------------------------------------------------------
;;;
;;; A model may ship a JSON config beside its graph — a Piper voice carries its
;;; phoneme id map, sample rate and inference defaults that way, and whoever
;;; loads the model is the one who knows what the keys mean.  A dependency for
;;; that is not worth it, and the reader is small enough to be obviously right.
;;; Objects become hash tables, arrays become vectors.

(defun read-json-file (path)
  (with-open-file (s path :external-format :utf-8)
    (let ((text (make-string (file-length s))))
      (json-parse (subseq text 0 (read-sequence text s))))))

(defun json-parse (text)
  (let ((pos 0))
    (labels ((peek-ch () (when (< pos (length text)) (char text pos)))
             (next () (prog1 (char text pos) (incf pos)))
             (skip () (loop while (and (< pos (length text))
                                       (member (char text pos) '(#\Space #\Tab #\Newline #\Return)))
                            do (incf pos)))
             (expect (c) (skip) (unless (eql (next) c) (error "JSON: expected ~a at ~d" c pos)))
             (value ()
               (skip)
               (case (peek-ch)
                 (#\{ (object)) (#\[ (array)) (#\" (jstring))
                 (#\t (incf pos 4) t) (#\f (incf pos 5) nil) (#\n (incf pos 4) :null)
                 (t (number))))
             (object ()
               (expect #\{)
               (let ((h (make-hash-table :test #'equal)))
                 (skip)
                 (when (eql (peek-ch) #\}) (next) (return-from object h))
                 (loop (skip)
                       (let ((k (jstring)))
                         (expect #\:)
                         (setf (gethash k h) (value)))
                       (skip)
                       (if (eql (peek-ch) #\,) (next) (progn (expect #\}) (return h))))))
             (array ()
               (expect #\[)
               (let ((xs '()))
                 (skip)
                 (when (eql (peek-ch) #\]) (next) (return-from array #()))
                 (loop (push (value) xs)
                       (skip)
                       (if (eql (peek-ch) #\,) (next)
                           (progn (expect #\]) (return (coerce (nreverse xs) 'vector)))))))
             (jstring ()
               (expect #\")
               (with-output-to-string (out)
                 (loop for c = (next)
                       until (eql c #\")
                       do (if (eql c #\\)
                              (let ((e (next)))
                                (case e
                                  (#\n (write-char #\Newline out))
                                  (#\t (write-char #\Tab out))
                                  (#\r (write-char #\Return out))
                                  (#\b (write-char #\Backspace out))
                                  (#\f (write-char #\Page out))
                                  (#\u (let ((code (parse-integer text :start pos :end (+ pos 4)
                                                                      :radix 16)))
                                         (incf pos 4)
                                         (write-char (code-char code) out)))
                                  (t (write-char e out))))
                              (write-char c out)))))
             (number ()
               (let ((start pos))
                 (loop while (and (< pos (length text))
                                  (find (char text pos) "-+.eE0123456789"))
                       do (incf pos))
                 (let ((s (subseq text start pos)))
                   (if (find-if (lambda (c) (find c ".eE")) s)
                       (let ((*read-default-float-format* 'double-float)
                             (*read-eval* nil))
                         (read-from-string s))
                       (parse-integer s))))))
      (prog1 (value) (skip)))))
