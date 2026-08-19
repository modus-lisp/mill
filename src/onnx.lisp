;;;; onnx.lisp — read a .onnx directly, in Lisp.
;;;;
;;;; tools/export-onnx.py exists so the Lisp side never had to parse protobuf, on the reasoning that
;;;; conversion runs once per model at build time, on a workstation.  That reasoning holds right up
;;;; until the target has no Python — which is modus, where the whole point is that there is nothing
;;;; underneath the Lisp.  So: protobuf, here, and the exporter becomes an optimization rather than a
;;;; dependency.  The two paths produce the same MODEL, and the gates do not know which one ran.
;;;;
;;;; This reads the subset of ONNX that a model actually carries — ModelProto, GraphProto, NodeProto,
;;;; AttributeProto, TensorProto, ValueInfoProto — and skips every field it does not know rather than
;;;; failing on it, because protobuf is designed for exactly that and a graph will always have fields
;;;; this does not care about (docstrings, training metadata, sparse initializers).
;;;;
;;;; Wire format, in full: a tag is (field-number << 3) | wire-type, and there are four wire types
;;;; still in use — varint, 64-bit, length-delimited, 32-bit.  Everything else is composition.

(in-package #:mill)

;;; ---- the wire format ---------------------------------------------------------

(defstruct (pb (:constructor make-pb (data pos end)))
  (data #() :type (simple-array (unsigned-byte 8) (*)))
  (pos 0 :type fixnum)
  (end 0 :type fixnum))

(defun pb-varint (b)
  "A base-128 varint.  Returns the unsigned value; a signed field reinterprets it."
  (declare (type pb b) (optimize (speed 3) (safety 1)))
  (let ((v 0) (shift 0) (d (pb-data b)))
    (loop
      (when (>= (pb-pos b) (pb-end b)) (error "onnx: truncated varint"))
      (let ((byte (aref d (pb-pos b))))
        (incf (pb-pos b))
        (setf v (logior v (ash (logand byte #x7f) shift)))
        (when (zerop (logand byte #x80)) (return v))
        (incf shift 7)
        (when (> shift 70) (error "onnx: varint too long"))))))

(defun pb-signed (u)
  "A varint field declared int64 is two's complement, not zigzag — ONNX uses no sint fields."
  (if (logbitp 63 u) (- u (ash 1 64)) u))

(defun pb-fixed (b n)
  (let ((v 0) (d (pb-data b)))
    (dotimes (i n v)
      (setf v (logior v (ash (aref d (+ (pb-pos b) i)) (* 8 i))))
      (when (= i (1- n)) (incf (pb-pos b) n)))))

(defun pb-bytes (b)
  "A length-delimited field, as a displaced-free copy."
  (let* ((len (pb-varint b))
         (out (make-array len :element-type '(unsigned-byte 8))))
    (replace out (pb-data b) :start2 (pb-pos b) :end2 (+ (pb-pos b) len))
    (incf (pb-pos b) len)
    out))

(defun pb-string (b)
  (let ((bytes (pb-bytes b)))
    (map 'string #'code-char bytes)))          ; ONNX names are ASCII in practice

(defun pb-sub (b)
  "A nested message: a sub-reader over the same buffer, without copying."
  (let* ((len (pb-varint b))
         (sub (make-pb (pb-data b) (pb-pos b) (+ (pb-pos b) len))))
    (incf (pb-pos b) len)
    sub))

(defun pb-skip (b wire)
  (ecase wire
    (0 (pb-varint b))
    (1 (pb-fixed b 8))
    (2 (let ((len (pb-varint b))) (incf (pb-pos b) len)))
    (5 (pb-fixed b 4))))

(defmacro do-fields ((field wire b) &body clauses)
  "Walk a message.  BODY is an ECASE-shaped list on FIELD; anything unmatched is skipped, which is
what makes this forward-compatible with ONNX versions this predates."
  (let ((tag (gensym)) (blk (gensym)))
    `(loop while (< (pb-pos ,b) (pb-end ,b))
           do (let* ((,tag (pb-varint ,b))
                     (,field (ash ,tag -3))
                     (,wire (logand ,tag 7)))
                (declare (ignorable ,field ,wire))
                (block ,blk
                  (case ,field
                    ,@clauses
                    (t (pb-skip ,b ,wire))))))))

;;; ---- ONNX messages -----------------------------------------------------------

(defun %onnx-dtype (code)
  (case code (1 :f32) (6 :i32) (7 :i64) (9 :bool) (11 :f64)
        (t (error "onnx: tensor element type ~d is not supported" code))))

(defun %raw->tensor (dtype dims raw)
  "RAW_DATA is little-endian packed elements — the form every real exporter writes."
  (let* ((shape (coerce dims 'vector))
         (tensor (make-tensor dtype shape))
         (d (tensor-data tensor))
         (n (length d)))
    (ecase dtype
      (:f32 (dotimes (i n) (setf (aref d i) (%bits->f32 (%le raw (* 4 i) 4)))))
      (:f64 (dotimes (i n) (setf (aref d i) (%bits->f64 (%le raw (* 8 i) 8)))))
      (:i64 (dotimes (i n) (setf (aref d i) (pb-signed (%le raw (* 8 i) 8)))))
      (:i32 (dotimes (i n) (let ((u (%le raw (* 4 i) 4)))
                             (setf (aref d i) (if (logbitp 31 u) (- u (ash 1 32)) u)))))
      (:bool (dotimes (i n) (setf (aref d i) (aref raw i)))))
    tensor))

(defun %le (bytes off n)
  (let ((v 0)) (dotimes (i n v) (setf v (logior v (ash (aref bytes (+ off i)) (* 8 i)))))))

(defun %bits->f32 (bits) (sb-kernel:make-single-float (if (logbitp 31 bits) (- bits (ash 1 32)) bits)))
(defun %bits->f64 (bits)
  (sb-kernel:make-double-float (let ((hi (ash bits -32))) (if (logbitp 31 hi) (- hi (ash 1 32)) hi))
                               (logand bits #xffffffff)))

(defun %read-tensor (b)
  "TensorProto -> (values name tensor)."
  (let ((dims '()) (dtype nil) (name "") (raw nil)
        (f32 '()) (i64 '()) (i32 '()) (f64 '()))
    (do-fields (f w b)
      (1 (if (= w 2)                                   ; packed dims
             (let ((s (pb-sub b)))
               (loop while (< (pb-pos s) (pb-end s)) do (push (pb-signed (pb-varint s)) dims)))
             (push (pb-signed (pb-varint b)) dims)))
      (2 (setf dtype (%onnx-dtype (pb-varint b))))
      (4 (if (= w 2)
             (let ((s (pb-sub b)))
               (loop while (< (pb-pos s) (pb-end s)) do (push (%bits->f32 (pb-fixed s 4)) f32)))
             (push (%bits->f32 (pb-fixed b 4)) f32)))
      (5 (if (= w 2)
             (let ((s (pb-sub b)))
               (loop while (< (pb-pos s) (pb-end s)) do (push (pb-varint s) i32)))
             (push (pb-varint b) i32)))
      (7 (if (= w 2)
             (let ((s (pb-sub b)))
               (loop while (< (pb-pos s) (pb-end s)) do (push (pb-signed (pb-varint s)) i64)))
             (push (pb-signed (pb-varint b)) i64)))
      (8 (setf name (pb-string b)))
      (9 (setf raw (pb-bytes b)))
      (10 (if (= w 2)
              (let ((s (pb-sub b)))
                (loop while (< (pb-pos s) (pb-end s)) do (push (%bits->f64 (pb-fixed s 8)) f64)))
              (push (%bits->f64 (pb-fixed b 8)) f64)))
      (13 (error "onnx: ~a uses external data, which this reader does not follow" name)))
    (setf dims (nreverse dims))
    (let ((dtype (or dtype :f32)))
      (values name
              (cond (raw (%raw->tensor dtype dims raw))
                    ;; the typed_data forms: rarer, but constants often arrive this way
                    (t (let* ((vals (cond (f32 (nreverse f32)) (i64 (nreverse i64))
                                          (i32 (nreverse i32)) (f64 (nreverse f64))
                                          (t '()))))
                         (tensor-from-list dtype (coerce dims 'vector) vals))))))))

(defun %read-attr (b)
  "AttributeProto -> (name key value), matching what the .graph reader produces."
  (let ((name "") (type nil) (i nil) (f nil) (s nil) (ints '()) (floats '()) (strings '()) (tens nil))
    (do-fields (fl w b)
      (1 (setf name (pb-string b)))
      (2 (setf f (%bits->f32 (pb-fixed b 4))))
      (3 (setf i (pb-signed (pb-varint b))))
      (4 (setf s (pb-string b)))
      (5 (setf tens (nth-value 1 (%read-tensor (pb-sub b)))))
      (7 (if (= w 2)
             (let ((sub (pb-sub b)))
               (loop while (< (pb-pos sub) (pb-end sub)) do (push (%bits->f32 (pb-fixed sub 4)) floats)))
             (push (%bits->f32 (pb-fixed b 4)) floats)))
      (8 (if (= w 2)
             (let ((sub (pb-sub b)))
               (loop while (< (pb-pos sub) (pb-end sub)) do (push (pb-signed (pb-varint sub)) ints)))
             (push (pb-signed (pb-varint b)) ints)))
      (9 (push (pb-string b) strings))
      (20 (setf type (pb-varint b))))
    ;; AttributeType: 1 FLOAT 2 INT 3 STRING 4 TENSOR 6 FLOATS 7 INTS 8 STRINGS
    (list name
          (case type
            (1 :float) (2 :int) (3 :string) (4 :tensor)
            (6 :floats) (7 :ints) (8 :strings)
            (t (cond (f :float) (i :int) (s :string) (tens :tensor)
                     (floats :floats) (ints :ints) (strings :strings) (t :int))))
          (case type
            (1 (float (or f 0) 1d0)) (2 (or i 0)) (3 (or s "")) (4 tens)
            (6 (mapcar (lambda (x) (float x 1d0)) (nreverse floats)))
            (7 (nreverse ints)) (8 (nreverse strings))
            (t (cond (f (float f 1d0)) (i i) (s s) (tens tens)
                     (floats (mapcar (lambda (x) (float x 1d0)) (nreverse floats)))
                     (ints (nreverse ints)) (strings (nreverse strings)) (t 0)))))))

(defun %read-node (b)
  (let ((ins '()) (outs '()) (name "") (op "") (attrs '()))
    (do-fields (f w b)
      (1 (push (pb-string b) ins))
      (2 (push (pb-string b) outs))
      (3 (setf name (pb-string b)))
      (4 (setf op (pb-string b)))
      (5 (push (%read-attr (pb-sub b)) attrs)))
    (%make-node op name (coerce (nreverse ins) 'simple-vector)
                (coerce (nreverse outs) 'simple-vector)
                ;; the .graph reader hands ops (name key value); keep that exactly
                (mapcar (lambda (a) (list (first a) (second a) (third a))) (nreverse attrs)))))

(defun %read-value-info (b)
  "ValueInfoProto -> (name dtype shape), shape holding integers or dimension names."
  (let ((name "") (dtype nil) (dims '()))
    (do-fields (f w b)
      (1 (setf name (pb-string b)))
      (2 (let ((ty (pb-sub b)))                                  ; TypeProto
           (do-fields (tf tw ty)
             (1 (let ((tt (pb-sub ty)))                          ; TypeProto.Tensor
                  (do-fields (ef ew tt)
                    (1 (setf dtype (ignore-errors (%onnx-dtype (pb-varint tt)))))
                    (2 (let ((sh (pb-sub tt)))                   ; TensorShapeProto
                         (do-fields (sf sw sh)
                           (1 (let ((dim (pb-sub sh)))
                                (let ((val nil) (param nil))
                                  (do-fields (df dw dim)
                                    (1 (setf val (pb-signed (pb-varint dim))))
                                    (2 (setf param (pb-string dim))))
                                  (push (or val param "?") dims))))))))))))))
    (list name (or dtype :f32) (coerce (nreverse dims) 'vector))))

(defun %read-graph (b)
  (let ((nodes '()) (inits (make-hash-table :test #'equal)) (ins '()) (outs '()))
    (do-fields (f w b)
      (1 (push (%read-node (pb-sub b)) nodes))
      (5 (multiple-value-bind (name tensor) (%read-tensor (pb-sub b))
           (setf (gethash name inits) tensor)))
      (11 (push (%read-value-info (pb-sub b)) ins))
      (12 (push (%read-value-info (pb-sub b)) outs)))
    (values (coerce (nreverse nodes) 'simple-vector) inits (nreverse ins) (nreverse outs))))

(defun %onnx-config (path)
  "Piper ships <voice>.onnx.json beside the weights; mill's exporter renames it .config.json.
Accept either, so a voice fetched straight from upstream needs no rearranging."
  (let ((candidates (list (concatenate 'string (namestring path) ".json")
                          (namestring (make-pathname :type "json" :defaults path))
                          (namestring (make-pathname :type "json"
                                                     :name (concatenate 'string
                                                                        (pathname-name path) ".config")
                                                     :defaults path)))))
    (dolist (c candidates)
      (when (probe-file c) (return (read-json-file c))))))

(defun load-onnx (path &key (config nil config-p))
  "Read a .onnx file into a MODEL, in Lisp, with no exporter and no protobuf library.

The graph's declared inputs include its initializers in some producers' output, so anything with a
weight of the same name is dropped from the input list — those are constants, not arguments."
  (let* ((bytes (with-open-file (s path :element-type '(unsigned-byte 8))
                  (let ((v (make-array (file-length s) :element-type '(unsigned-byte 8))))
                    (read-sequence v s) v)))
         (b (make-pb bytes 0 (length bytes)))
         (opset 0) (meta '()) nodes inits ins outs)
    (do-fields (f w b)
      (7 (multiple-value-setq (nodes inits ins outs) (%read-graph (pb-sub b))))
      (8 (let ((imp (pb-sub b)))                                  ; OperatorSetIdProto
           (do-fields (if_ iw imp)
             (2 (setf opset (max opset (pb-varint imp)))))))
      (14 (let ((entry (pb-sub b)) (k "") (v ""))                 ; metadata_props
            (do-fields (mf mw entry)
              (1 (setf k (pb-string entry)))
              (2 (setf v (pb-string entry))))
            (push (cons k v) meta))))
    (unless nodes (error "onnx: no graph in ~a" path))
    (%make-model :inputs (remove-if (lambda (i) (gethash (first i) inits)) ins)
                 :outputs outs
                 :initializers inits
                 :nodes nodes
                 :opset opset
                 :metadata (nreverse meta)
                 :config (if config-p config (%onnx-config path)))))

;;; ---- .onnx -> .graph + .bin ---------------------------------------------------
;;; Which is where protobuf belongs: ONCE per model, not on every load.
;;;
;;; The runtime format is an s-expression and a blob of little-endian floats — a form Lisp READs
;;; and a vector it mmaps in spirit — and it loads about three times faster than walking protobuf
;;; does.  Keeping the reader above is still what makes this possible without Python; it just
;;; belongs at build time, exactly where tools/export-onnx.py always said it did.

(defun %dtype-name (dtype)
  (ecase dtype (:f32 "f32") (:f64 "f64") (:i64 "i64") (:i32 "i32") (:bool "bool")))

(defun %dtype-width (dtype)
  (ecase dtype (:f32 4) (:f64 8) (:i64 8) (:i32 4) (:bool 1)))

(defun %emit-tensor (tensor out)
  "Append TENSOR's elements to OUT (a fill-pointer octet vector).  Returns (offset . nbytes)."
  (let* ((offset (fill-pointer out))
         (d (tensor-data tensor))
         (dtype (tensor-dtype tensor))
         (w (%dtype-width dtype)))
    (dotimes (i (length d))
      (let ((bits (ecase dtype
                    (:f32 (logand (sb-kernel:single-float-bits (aref d i)) #xffffffff))
                    (:f64 (logior (ash (logand (sb-kernel:double-float-high-bits (aref d i))
                                               #xffffffff) 32)
                                  (sb-kernel:double-float-low-bits (aref d i))))
                    ((:i64 :i32) (logand (aref d i) (1- (ash 1 (* 8 w)))))
                    (:bool (aref d i)))))
        (dotimes (b w) (vector-push-extend (ldb (byte 8 (* 8 b)) bits) out))))
    (cons offset (- (fill-pointer out) offset))))

(defun onnx->graph (onnx-path out-dir)
  "Convert ONNX-PATH into the .graph + .bin pair mill loads, writing both into OUT-DIR.
Returns the graph path.  This is tools/export-onnx.py, in Lisp and without the dependency."
  (let* ((model (load-onnx onnx-path :config nil))
         (stem (pathname-name onnx-path))
         (graph-path (merge-pathnames (concatenate 'string stem ".graph") out-dir))
         (bin-path (merge-pathnames (concatenate 'string stem ".bin") out-dir))
         (blob (make-array (* 1024 1024) :element-type '(unsigned-byte 8)
                                         :adjustable t :fill-pointer 0))
         (inits '()))
    (ensure-directories-exist out-dir)
    (maphash (lambda (name tensor)
               (let ((where (%emit-tensor tensor blob)))
                 (push (list name (%dtype-name (tensor-dtype tensor))
                             (coerce (tensor-shape tensor) 'list) (car where) (cdr where))
                       inits)))
             (model-initializers model))
    (with-open-file (o bin-path :direction :output :element-type '(unsigned-byte 8)
                                :if-exists :supersede :if-does-not-exist :create)
      (write-sequence blob o))
    (with-open-file (o graph-path :direction :output :if-exists :supersede
                                  :if-does-not-exist :create)
      (format o ";;; Generated by mill's ONNX reader — do not edit.~%;;; source: ~a~%"
              (file-namestring onnx-path))
      (format o "(:version 1~% :producer \"mill\"~% :opset ~a~% :metadata (~{~s~^ ~})~%"
              (model-opset model)
              (loop for (k . v) in (model-metadata model) collect (list k v)))
      (flet ((io (specs)
               (format nil "(~{~a~^~%           ~})"
                       (loop for (name dt shape) in specs
                             collect (format nil "(~s ~s ~a)" name (%dtype-name dt)
                                             (coerce shape 'list))))))
        (format o " :inputs ~a~% :outputs ~a~%" (io (model-inputs model)) (io (model-outputs model))))
      (format o " :initializers (~%~{  ~a~%~})~%"
              (loop for (name dt shape off n) in (nreverse inits)
                    collect (format nil "(~s ~s ~a ~a ~a)" name dt shape off n)))
      (format o " :nodes (~%")
      (loop for node across (model-nodes model)
            do (format o "  (~s ~s (~{~s~^ ~}) (~{~s~^ ~}) (~{~a~^ ~}))~%"
                       (node-op node) (node-name node)
                       (coerce (node-inputs node) 'list) (coerce (node-outputs node) 'list)
                       (loop for (aname kind value) in (node-attrs node)
                             collect (if (eq kind :tensor)
                                         (let ((where (%emit-tensor value blob)))
                                           (format nil "(~s :tensor (~s ~s ~a ~a ~a))" aname aname
                                                   (%dtype-name (tensor-dtype value))
                                                   (coerce (tensor-shape value) 'list)
                                                   (car where) (cdr where)))
                                         (format nil "(~s ~s ~a)" aname kind
                                                 (typecase value
                                                   (string (format nil "~s" value))
                                                   (list (format nil "~a" value))
                                                   (t value)))))))
      (format o " ))~%"))
    ;; tensor-valued attributes may have appended to the blob after it was written; rewrite it
    (with-open-file (o bin-path :direction :output :element-type '(unsigned-byte 8)
                                :if-exists :supersede)
      (write-sequence blob o))
    (namestring graph-path)))
