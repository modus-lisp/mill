;;;; npy.lisp — read NumPy's .npy files, for the gates only.
;;;;
;;;; This is not part of the engine.  It exists so the per-node gate can read
;;;; what onnxruntime produced without a Python process in the loop, and so the
;;;; comparison is against bytes on disk that can be re-inspected later rather
;;;; than against a number that scrolled past once.
;;;;
;;;; The format: a 6-byte magic, a version, a header length, and a Python dict
;;;; literal naming the dtype, the element order, and the shape — then the raw
;;;; elements.  Only the little-endian C-order cases NumPy actually writes here
;;;; are handled; anything else stops the gate rather than being guessed at.

(in-package #:mill)

(defun npy-header-value (header key)
  "The text following 'KEY': in a .npy header dict, up to the next comma or brace."
  (let* ((at (search (format nil "'~a':" key) header)))
    (unless at (error "no ~a in .npy header ~s" key header))
    (let* ((start (position #\Space header :start (+ at (length key) 3) :test #'char/=))
           ;; the shape is a tuple, so its commas are not the delimiter
           (end (if (char= (char header start) #\()
                    (1+ (position #\) header :start start))
                    (position-if (lambda (c) (member c '(#\, #\}))) header :start start))))
      (string-trim " " (subseq header start (or end (length header)))))))

(defun npy-dtype (descr)
  (let ((d (string-trim "'\"" descr)))
    (cond ((member d '("<f4" "=f4" "f4") :test #'string=) :f32)
          ((member d '("<f8" "=f8" "f8") :test #'string=) :f64)
          ((member d '("<i8" "=i8" "i8") :test #'string=) :i64)
          ((member d '("<i4" "=i4" "i4") :test #'string=) :i32)
          ((member d '("|b1" "b1" "?") :test #'string=) :bool)
          (t (error "unsupported .npy dtype ~s" descr)))))

(defun npy-shape (text)
  "Parse a Python tuple literal: (), (5,), (1, 30, 192)."
  (let ((dims '()) (at 0))
    (loop
      (let ((start (position-if #'digit-char-p text :start at)))
        (unless start (return))
        (multiple-value-bind (n next) (parse-integer text :start start :junk-allowed t)
          (push n dims)
          (setf at next))))
    (coerce (nreverse dims) 'simple-vector)))

(defun read-npy (path)
  "The array in PATH as a mill tensor."
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((magic (make-array 6 :element-type '(unsigned-byte 8))))
      (read-sequence magic s)
      (unless (and (= (aref magic 0) #x93)
                   (string= "NUMPY" (map 'string #'code-char (subseq magic 1))))
        (error "~a is not a .npy file" path)))
    (let* ((major (read-byte s))
           (minor (read-byte s))
           (hlen (if (= major 1)
                     (logior (read-byte s) (ash (read-byte s) 8))
                     (logior (read-byte s) (ash (read-byte s) 8)
                             (ash (read-byte s) 16) (ash (read-byte s) 24))))
           (hbytes (make-array hlen :element-type '(unsigned-byte 8))))
      (declare (ignore minor))
      (read-sequence hbytes s)
      (let* ((header (map 'string #'code-char hbytes))
             (dtype (npy-dtype (npy-header-value header "descr")))
             (fortran (npy-header-value header "fortran_order"))
             (shape (npy-shape (npy-header-value header "shape")))
             (n (shape-size shape))
             (width (ecase dtype (:f32 4) (:f64 8) (:i64 8) (:i32 4) (:bool 1)))
             (raw (make-array (* n width) :element-type '(unsigned-byte 8))))
        (when (search "True" fortran)
          (error "~a is in Fortran order; mill's tensors are C order" path))
        (read-sequence raw s)
        (decode-tensor dtype shape raw 0)))))
