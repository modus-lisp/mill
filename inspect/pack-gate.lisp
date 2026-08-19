;;;; pack-gate.lisp — the packed B is a permutation, so it had better be exact.
;;;;
;;;;     sbcl --non-interactive --load inspect/pack-gate.lisp
;;;;
;;;; MM-PACKED-B hands MATMUL-CORE a copy of B laid out in the order the kernel
;;;; reads it, and the claim that makes it safe is that permuting the operand
;;;; changes nothing: the same products are summed over K in the same order, so
;;;; every output bit is the same.  This gate holds that claim to the letter.
;;;;
;;;; Each case is run twice: once with B left as an ordinary tensor, which takes
;;;; the unpacked kernel, and once with the same B declared a weight, which gets
;;;; the pack.  The two results are compared to each other bit for bit and both
;;;; to the definition, and the gate refuses to pass a case where the pack was
;;;; not actually built — a threshold or a shape test that quietly turned packing
;;;; off would otherwise make this file green by doing nothing.
;;;;
;;;; What the sweep is looking for:
;;;;
;;;;   - N that is not a whole number of blocks, of lane groups, or of either,
;;;;     because the packed layout has three regions and the kernel walks them in
;;;;     one order that both sides have to agree on;
;;;;   - a column split across threads, because a window that did not start on a
;;;;     block boundary would read the wrong region of the pack;
;;;;   - zeros in A, which send a group of rows to the sparse kernel — which
;;;;     reads B where it lies even when a pack exists, and would be caught here
;;;;     if it ever read the wrong one;
;;;;   - a batched B, which shares one data vector between matrices and must not
;;;;     be packed at all.

(require :asdf)
(asdf:load-system :mill)

(in-package #:mill)

(defun spread (n seed &key (zero-every 0))
  "A deterministic spread of values, distinct per position.  ZERO-EVERY plants an
exact 0.0 at that period, which is what puts a group of rows on the sparse path."
  (let ((a (make-array n :element-type 'single-float))
        (s seed))
    (dotimes (i n a)
      (setf s (mod (+ (* s 1103515245) 12345) 2147483648))
      (setf (aref a i)
            (if (and (plusp zero-every) (zerop (mod i zero-every)))
                0.0
                (- (/ (float (mod s 4001) 1.0) 1000.0) 2.0))))))

(defun ref-matmul (ad bd nb m k n)
  "The definition: NB independent (M K) x (K N) products, summed over K in
increasing K, skipping a zero multiplier the way MATMUL-CORE does."
  (let ((r (make-array (* nb m n) :element-type 'single-float)))
    (dotimes (item nb r)
      (dotimes (i m)
        (dotimes (j n)
          (let ((acc 0.0))
            (declare (type single-float acc))
            (dotimes (kk k)
              (let ((av (aref ad (+ (* item m k) (* i k) kk))))
                (unless (zerop av)
                  (setf acc (+ acc (* av (aref bd (+ (* kk n) j))))))))
            (setf (aref r (+ (* item m n) (* i n) j)) acc)))))))

(defun run-case (nb m k n zero-every threads)
  (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
    (let* ((seed (+ (* nb 131) (* m 17) (* k 7) n 1))
           (ad (spread (* nb m k) seed :zero-every zero-every))
           (bd (spread (* k n) (+ seed 7919)))
           (a (make-tensor :f32 (vector nb m k)))
           (b (make-tensor :f32 (vector k n)))
           (fn (gethash "MatMul" *op-table*))
           (desc (format nil "(~d ~d ~d)x(~d ~d)~:[~*~; zeros every ~d~], ~d thread~:p"
                         nb m k k n (plusp zero-every) zero-every threads)))
      (replace (tensor-data a) ad)
      (replace (tensor-data b) bd)
      (let* ((cold (copy-seq (tensor-data (first (funcall fn nil (list a b))))))
             (warm (progn (note-constant b)
                          (tensor-data (first (funcall fn nil (list a b))))))
             (pack (gethash (tensor-data b) *mm-packs*))
             (want (ref-matmul ad bd nb m k n))
             (problems '()))
        (unless (typep pack '(simple-array single-float (*)))
          (push "B was never packed — the case proves nothing" problems))
        (let ((bad (loop for i below (length cold)
                         count (/= (aref cold i) (aref warm i)))))
          (when (plusp bad)
            (push (format nil "packed and unpacked differ in ~d of ~d elements"
                          bad (length cold))
                  problems)))
        (let ((bad (loop for i below (length warm)
                         count (/= (aref warm i) (aref want i)))))
          (when (plusp bad)
            (push (format nil "packed differs from the definition in ~d of ~d elements"
                          bad (length warm))
                  problems)))
        (values problems desc)))))

(defun run-batched-b (nb m k n)
  "A B that is a batch of matrices shares one data vector, and packing it would
give every matrix the first one's weights.  It must be left alone."
  (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
    (let* ((ad (spread (* nb m k) 3))
           (bd (spread (* nb k n) 5))
           (a (make-tensor :f32 (vector nb m k)))
           (b (make-tensor :f32 (vector nb k n)))
           (fn (gethash "MatMul" *op-table*))
           (desc (format nil "batched B (~d ~d ~d)x(~d ~d ~d)" nb m k nb k n)))
      (replace (tensor-data a) ad)
      (replace (tensor-data b) bd)
      (note-constant b)
      (let ((got (tensor-data (first (funcall fn nil (list a b)))))
            (problems '()))
        (when (typep (gethash (tensor-data b) *mm-packs*)
                     '(simple-array single-float (*)))
          (push "a batch of matrices was packed as though it were one" problems))
        (dotimes (item nb)
          (let ((want (ref-matmul (subseq ad (* item m k) (* (1+ item) m k))
                                  (subseq bd (* item k n) (* (1+ item) k n))
                                  1 m k n)))
            (let ((bad (loop for i below (* m n)
                             count (/= (aref got (+ (* item m n) i)) (aref want i)))))
              (when (plusp bad)
                (push (format nil "batch item ~d differs in ~d of ~d elements"
                              item bad (* m n))
                      problems)))))
        (values problems desc)))))

(defun main ()
  ;; every B in the sweep packs, whatever its size: the shapes here are chosen to
  ;; break the layout, not to be big, and the real threshold would exclude them all
  (setf *mm-pack-floor* 0)
  (let ((cases 0) (failures '()))
    (flet ((check (problems desc)
             (incf cases)
             (dolist (p problems) (push (format nil "~a — ~a" desc p) failures))))
      (dolist (threads '(1 8))
        (shutdown-workers)
        (setf *worker-count* threads)
        (dolist (n '(24 25 31 32 47 48 8 7 96 100 768))
          (dolist (dims '((1 1) (4 1) (1 5) (3 4)))
            (destructuring-bind (nb m) dims
              (dolist (k '(1 3 37))
                (dolist (zero-every '(0 5 1))
                  (multiple-value-bind (problems desc)
                      (run-case nb m k n zero-every threads)
                    (check problems desc))))))))
      (shutdown-workers)
      (setf *worker-count* 8)
      (multiple-value-bind (problems desc) (run-batched-b 3 2 37 96)
        (check problems desc)))
    (format t "~&~%ran ~d cases~%" cases)
    (cond ((null failures)
           (format t "~%GATE GREEN — the packed B is the same arithmetic to the bit~%"))
          (t
           (format t "~%GATE RED — ~d complaints:~%" (length failures))
           (dolist (f (reverse (subseq failures 0 (min 20 (length failures)))))
             (format t "  ~a~%" f))
           (when (> (length failures) 20)
             (format t "  ... and ~d more~%" (- (length failures) 20)))))
    (finish-output)
    (sb-ext:exit :code (if failures 1 0))))

(main)
