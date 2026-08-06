;;;; parallel.lisp — a fixed worker pool, and one way to use it.
;;;;
;;;; The convolutions are 86% of a synthesis and their output rows are
;;;; independent, so the cheapest remaining speedup is to run rows on different
;;;; cores.  This is deliberately the smallest thing that does that: a pool of
;;;; threads created once, a single job at a time, and no queue.  There is no
;;;; work stealing and no nesting — a kernel splits its rows into as many chunks
;;;; as there are threads, everyone runs one chunk, and the caller waits.  That
;;;; is enough because every chunk of a convolution costs the same.
;;;;
;;;; Threads are the portable lever: sb-thread exists on ARM, so this helps the
;;;; Pi this engine is aimed at.  SIMD would help more on this desktop and not at
;;;; all there, which is why it is not what got built first.
;;;;
;;;; Without :sb-thread the whole file degrades to calling the function once.

(in-package #:mill)

(defun detect-cores ()
  "Cores according to /proc/cpuinfo, or 1 if that cannot be read.  There is no
portable way to ask, and guessing high is worse than guessing low."
  (max 1 (or (ignore-errors
               (with-open-file (s "/proc/cpuinfo" :if-does-not-exist nil)
                 (and s (loop for line = (read-line s nil)
                              while line
                              count (and (>= (length line) 9)
                                         (string= "processor" line :end2 9))))))
             1)))

(defparameter *worker-count*
  (let ((env (or (sb-ext:posix-getenv "MILL_THREADS")
                 (sb-ext:posix-getenv "CHORD_THREADS"))))
    (or (and env (ignore-errors (parse-integer env)))
        (max 1 (min 16 (detect-cores)))))
  "Threads to spread a kernel over, counting the calling thread.  MILL_THREADS
overrides it, and CHORD_THREADS still does too — this pool was chord's before it
was mill's, and a knob that has been sitting in someone's shell profile deserves
to keep working.  1 disables the pool entirely.  Changing it after the first run
needs SHUTDOWN-WORKERS to take effect.

The cap is where the measured curve flattens, not a guess.  On a 116-core EPYC,
one sentence synthesized at 0.30x realtime on 1 thread, 1.01x on 8, 1.31x on 16,
and 1.56x on 64 — so the last 48 threads buy 19% while the first 15 buy 340%.
What is left over is the ops that are not convolutions plus the join at the end
of each of the graph's 132 conv nodes, and neither gets better with more cores.
Taking every core on a shared machine to win a fifth is not a good default.")

(defconstant +parallel-grain+ 100000
  "Multiply-adds below which a chunk is not worth waking a thread for.  Waking
one and joining it costs a few microseconds; this is roughly a hundred times
that, so the split is never the reason a small node got slower.")

#+sb-thread
(progn

(defstruct (worker (:constructor %make-worker))
  (thread nil)
  (go-sem (sb-thread:make-semaphore) :type sb-thread:semaphore)
  (done-sem (sb-thread:make-semaphore) :type sb-thread:semaphore)
  (lo 0 :type fixnum)
  (hi 0 :type fixnum))

(defvar *workers* nil "Simple vector of *WORKER-COUNT* - 1 workers, or NIL.")
(defvar *job* nil "The function the workers are to run, set before waking them.")
(defvar *job-error* nil "The first condition a worker signalled, re-signalled by the caller.")

;;; *JOB* is read from other threads, so it is assigned rather than bound — a LET
;;; here would be invisible to them.  That is also why a job may not itself call
;;; PARALLEL-RANGE: there is one slot, not a stack of them.

(defun run-worker (w)
  (loop
    (sb-thread:wait-on-semaphore (worker-go-sem w))
    ;; float traps are per thread, and RUN-MODEL's masking happened in the
    ;; caller's; without this a denormal in the flow decoder kills a worker
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
      (handler-case (funcall (the function *job*) (worker-lo w) (worker-hi w))
        (serious-condition (e) (setf *job-error* e))))
    (sb-thread:signal-semaphore (worker-done-sem w))))

(defun ensure-workers ()
  (or *workers*
      (setf *workers*
            (coerce (loop for i from 1 below (max 1 *worker-count*)
                          collect (let ((w (%make-worker)))
                                    (setf (worker-thread w)
                                          (sb-thread:make-thread
                                           (lambda () (run-worker w))
                                           :name (format nil "mill worker ~d" i)))
                                    w))
                    'simple-vector))))

(defun shutdown-workers ()
  "Stop the pool.  Only for changing *WORKER-COUNT* and for measurement — a
process that is synthesizing wants the threads to stay."
  (when *workers*
    (map nil (lambda (w) (sb-thread:terminate-thread (worker-thread w))) *workers*)
    (setf *workers* nil))
  (values))

(defun parallel-range (n fn &key (min-chunk 1))
  "Call FN on disjoint subranges (LO HI) that together cover [0, N).

MIN-CHUNK is how few units are still worth a thread; the caller knows what a
unit costs and this file does not.  FN runs in the calling thread too, so a
one-chunk split is just a call, and small work never touches the pool."
  (declare (type function fn) (type fixnum n))
  (let* ((pool (if (and (> *worker-count* 1) (>= n (* 2 min-chunk))) (ensure-workers) nil))
         (k (if pool (min (1+ (length pool)) (max 1 (floor n min-chunk))) 1)))
    (if (< k 2)
        (funcall fn 0 n)
        (progn
          (setf *job* fn *job-error* nil)
          (loop for i from 1 below k
                for w = (svref pool (1- i))
                do (setf (worker-lo w) (floor (* i n) k)
                         (worker-hi w) (floor (* (1+ i) n) k))
                   (sb-thread:signal-semaphore (worker-go-sem w)))
          (unwind-protect (funcall fn 0 (floor n k))
            ;; every worker that was woken must be joined even if the caller's
            ;; own chunk threw, or the next job would see stale semaphore counts
            (loop for i from 1 below k
                  do (sb-thread:wait-on-semaphore (worker-done-sem (svref pool (1- i))))))
          (when *job-error* (error *job-error*))
          (values)))))

) ; #+sb-thread

#-sb-thread
(progn
  (defun shutdown-workers () (values))
  (defun parallel-range (n fn &key (min-chunk 1))
    (declare (ignore min-chunk))
    (funcall fn 0 n)))
