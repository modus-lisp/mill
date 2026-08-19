;;;; pool-gate.lisp — the worker pool under CONCURRENT dispatchers.
;;;;
;;;;     sbcl --dynamic-space-size 4096 --load inspect/pool-gate.lisp
;;;;
;;;; parallel.lisp has one *JOB* slot and one SEQ/ACK pair per worker, so it has room
;;;; for exactly one dispatcher at a time.  Nothing enforced that.  Two threads calling
;;;; PARALLEL-RANGE at once would interleave a SEQ bump between another caller's read of
;;;; SEQ and its wait for the matching ACK, and JOIN-WORKER waits for that ACK by
;;;; SPINNING with no deadline — so the failure is not an error and not a wrong number,
;;;; it is threads at 100% of a core forever.
;;;;
;;;; One process running two models at once is ordinary: a desktop that speaks while it
;;;; listens has chord in PARALLEL-RANGE on one thread and stave in it on another.  That
;;;; is where this was found, four hours into a wedged desktop.
;;;;
;;;; So this gate is about two things a single-threaded gate cannot see:
;;;;   (1) LIVENESS.  Many threads dispatching at once must all finish.  A watchdog
;;;;       fails the gate rather than letting it hang, because a hang in CI is a hang
;;;;       nobody reads.
;;;;   (2) EXACTNESS UNDER CONTENTION.  Every thread's output must be bit-identical to
;;;;       the same job run serially.  A pool that hands two dispatchers each other's
;;;;       ranges would produce gaps and double-writes, which liveness alone would miss.
;;;; Plus the nesting case (a job that itself splits), which used to be forbidden by a
;;;; comment and is now just one thread's worth of work.

(require :asdf)
(asdf:load-system :mill)
(in-package #:mill)

(defparameter *threads* 6 "Concurrent dispatchers — more than the pool has workers.")
(defparameter *rounds* 40)
(defparameter *n* 20000)
(defparameter *deadline* 90 "Seconds before we call it wedged.")

(defvar *fail* 0)
(defun check (ok fmt &rest args)
  (format t "  [~:[FAIL~;pass~]] ~?~%" ok fmt args) (unless ok (incf *fail*)))

(defun job (out lo hi)
  "Deterministic, order-independent, and slow enough per element to overlap."
  (declare (type (simple-array double-float (*)) out) (type fixnum lo hi))
  (loop for i of-type fixnum from lo below hi
        do (setf (aref out i) (* (sin (float i 1d0)) (cos (float (* 3 i) 1d0))))))

(defun run-split (out)
  (parallel-range (length out) (lambda (lo hi) (job out lo hi)) :min-chunk 64))

(defun serial-reference ()
  (let ((out (make-array *n* :element-type 'double-float)))
    (job out 0 *n*) out))

;;; the watchdog: a wedge is the failure this gate exists for, so it must be REPORTED
;;; rather than waited on.  Prints who was still running, which is what you want when
;;; the wedge comes back.
(defvar *done* nil)
(sb-thread:make-thread
 (lambda ()
   (let ((end (+ (get-internal-real-time) (* *deadline* internal-time-units-per-second))))
     (loop until *done*
           do (sleep 1/4)
              (when (> (get-internal-real-time) end)
                (format t "~&  [FAIL] WEDGED — ~d s with no progress; live threads:~%~{    ~a~%~}"
                        *deadline*
                        (mapcar #'sb-thread:thread-name (sb-thread:list-all-threads)))
                (format t "~%=> FAIL (wedged)~%") (finish-output)
                (sb-ext:exit :code 1 :abort t)))))
 :name "pool-gate-watchdog")

(format t "~&[mill pool: ~d dispatchers x ~d rounds over ~d workers]~%"
        *threads* *rounds* (1- *worker-count*))

(let* ((ref (serial-reference))
       (outs (loop repeat *threads* collect (make-array *n* :element-type 'double-float)))
       (errs (make-array *threads* :initial-element nil))
       (start (get-internal-real-time))
       (threads (loop for out in outs
                      for id from 0
                      collect (let ((out out) (id id))
                                (sb-thread:make-thread
                                 (lambda ()
                                   (handler-case (dotimes (r *rounds*) (run-split out))
                                     (serious-condition (e) (setf (aref errs id) (princ-to-string e)))))
                                 :name (format nil "pool-gate-~d" id))))))
  (mapc #'sb-thread:join-thread threads)
  (setf *done* t)
  (let ((secs (/ (float (- (get-internal-real-time) start)) internal-time-units-per-second)))
    (check t "~d concurrent dispatchers all returned (~,1f s)" *threads* secs))
  (check (every #'null (coerce errs 'list)) "no dispatcher signalled: ~a" (coerce errs 'list))
  (check (every (lambda (o) (equalp o ref)) outs)
         "every thread's output is bit-identical to the serial reference")
  ;; a job that splits again: the inner call finds the pool held by its own thread and
  ;; runs the range itself.  Used to be undefined behaviour; must now be exact.
  (let ((out (make-array *n* :element-type 'double-float)))
    (parallel-range 8 (lambda (lo hi)
                        (declare (ignore lo hi))
                        (parallel-range *n* (lambda (a b) (job out a b)) :min-chunk 64))
                    :min-chunk 1)
    (check (equalp out ref) "a split nested inside a split still computes the whole range"))
  ;; and the pool still works normally afterwards — no worker left mid-job
  (let ((out (make-array *n* :element-type 'double-float)))
    (run-split out)
    (check (equalp out ref) "the pool is still correct after all of that")))

(format t "~%=> ~:[PASS~;FAIL (~d)~]~%" (plusp *fail*) *fail*)
(finish-output)
(sb-ext:exit :code (if (plusp *fail*) 1 0))
