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

(defparameter *worker-spin* 30000
  "How long a worker looks for its next job before it goes to sleep, in spins.

This is the difference between a pool that costs 27 microseconds to dispatch to
and one that costs under two.  A semaphore hands the thread to the kernel, and
getting it back is a scheduler round trip — which was fine when a job was a
whole convolution and is not fine here, where a graph is thousands of small ops
in a row and the transducer asks for one per audio frame.  Between two ops of
the same graph the next job is microseconds away, so the worker waits for it
awake.

The sleep is still there underneath, and it is what makes this safe to leave on:
a pool with nothing to do stops burning cores after one spin budget, so an idle
process is idle.  Set it to 0 to go back to sleeping immediately.")

#+sb-thread
(progn

;;; A job is announced by bumping SEQ and finished by copying it into ACK, so
;;; both sides can tell "there is work" and "the work is done" by reading one
;;; fixnum — no kernel call on either end when the pool is already awake.  The
;;; semaphore is only the way back from sleep.
;;;
;;; SLEEPING and SEQ are the two halves of a Dekker handshake: the dispatcher
;;; writes SEQ then reads SLEEPING, the worker writes SLEEPING then re-reads
;;; SEQ, and the memory barriers between are what make it impossible for both to
;;; miss the other.  Whichever way the race falls the worker runs the job: it
;;; either sees the new SEQ and never sleeps, or it sleeps and gets signalled.
;;; A signal that arrives when the worker is already awake is not lost, it is
;;; just a count the next WAIT-ON-SEMAPHORE returns from immediately — which is
;;; why that wait is a loop around the SEQ test rather than a single call.

(defstruct (worker (:constructor %make-worker))
  (thread nil)
  (go-sem (sb-thread:make-semaphore) :type sb-thread:semaphore)
  (seq 0 :type fixnum)
  (ack 0 :type fixnum)
  (sleeping 0 :type fixnum)
  (lo 0 :type fixnum)
  (hi 0 :type fixnum))

(defvar *workers* nil "Simple vector of *WORKER-COUNT* - 1 workers, or NIL.")
(defvar *job* nil "The function the workers are to run, set before waking them.")
(defvar *job-error* nil "The first condition a worker signalled, re-signalled by the caller.")

;;; *JOB* is read from other threads, so it is assigned rather than bound — a LET
;;; here would be invisible to them.  That is also why a job may not itself call
;;; PARALLEL-RANGE: there is one slot, not a stack of them.

(defvar *pool-lock* (sb-thread:make-mutex :name "mill-pool")
  "Held for the length of a dispatch.  There is ONE *JOB* slot and one SEQ/ACK pair
per worker, so the pool has room for exactly one dispatcher at a time — which was
true from the first line of this file and was enforced by nothing.

Two threads dispatching at once do not merely interleave: the second one's SEQ bump
lands between the first one's read of SEQ and its wait for ACK, and JOIN-WORKER then
waits for an acknowledgement of a job number that no longer exists.  It waits by
SPINNING, with no deadline, so the symptom is not an error or a wrong answer — it is
two threads at 100% of a core forever and a machine that looks like it crashed.
That is reachable from one process running two models at once, which is an ordinary
thing to do: a desktop that speaks while it listens has chord in this function on one
thread and stave in it on another, and the ear and the voice wedge each other.

A caller that cannot have the pool runs its range ITSELF rather than waiting for it.
Waiting would be correct and would also mean an audio thread blocking on an unrelated
model's convolution; running it here costs that one call its threads and nothing else,
which is the same deal FN already accepts when N is small or the pool is off.")

(declaim (inline worker-wait))
(defun worker-wait (w seen)
  "Block until W's SEQ leaves SEEN, spinning first."
  (declare (type worker w) (type fixnum seen))
  (loop repeat *worker-spin*
        do (when (/= (worker-seq w) seen) (return-from worker-wait))
           (sb-ext:spin-loop-hint)
           (sb-thread:barrier (:read)))
  ;; out of patience: announce the sleep, then look once more, because the
  ;; dispatcher may have posted between the last spin and here
  (setf (worker-sleeping w) 1)
  (sb-thread:barrier (:memory))
  (loop until (/= (worker-seq w) seen)
        do (sb-thread:wait-on-semaphore (worker-go-sem w))
           (sb-thread:barrier (:read)))
  (setf (worker-sleeping w) 0))

(defun run-worker (w)
  (let ((seen 0))
    (declare (type fixnum seen))
    (loop
      (worker-wait w seen)
      (setf seen (worker-seq w))
      ;; float traps are per thread, and RUN-MODEL's masking happened in the
      ;; caller's; without this a denormal in the flow decoder kills a worker
      (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
        (handler-case (funcall (the function *job*) (worker-lo w) (worker-hi w))
          (serious-condition (e) (setf *job-error* e))))
      (sb-thread:barrier (:memory))
      (setf (worker-ack w) seen))))

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

(defun join-worker (w)
  "Wait for W to finish the job it was last given.

The caller is here because it ran out of its own chunk, so there is nothing
useful to do instead and the spin is not stealing anything.  It yields after a
while anyway: a pool wider than the machine means someone is waiting for a core
rather than for work, and spinning at that point is how a 16-thread split loses
to a 1-thread one."
  (declare (type worker w))
  (let ((want (worker-seq w)))
    (loop until (= (worker-ack w) want)
          for spins of-type fixnum from 0
          do (sb-ext:spin-loop-hint)
             (sb-thread:barrier (:read))
             (when (> spins *worker-spin*) (sb-thread:thread-yield)))
    (sb-thread:barrier (:read))))

(defun parallel-range (n fn &key (min-chunk 1))
  "Call FN on disjoint subranges (LO HI) that together cover [0, N).

MIN-CHUNK is how few units are still worth a thread; the caller knows what a
unit costs and this file does not.  FN runs in the calling thread too, so a
one-chunk split is just a call, and small work never touches the pool."
  (declare (type function fn) (type fixnum n))
  ;; Whoever already holds the pool is either this job's own dispatcher (FN nesting a
  ;; split inside a split) or, on a worker thread, the dispatcher it is running for.
  ;; Both used to be corruption; both are now simply one thread's worth of work.
  (when (or (<= *worker-count* 1) (< n (* 2 min-chunk))
            (sb-thread:holding-mutex-p *pool-lock*))
    (funcall fn 0 n)
    (return-from parallel-range (values)))
  (unless (%parallel-pooled n fn min-chunk)
    (funcall fn 0 n))                     ; the pool was another graph's; run it here
  (values))

(defun %parallel-pooled (n fn min-chunk)
  "Try to run FN's split ON THE POOL.  T if it ran there, NIL if the pool was taken."
  (declare (type function fn) (type fixnum n))
  (sb-thread:with-mutex (*pool-lock* :wait-p nil)
    (let* ((pool (ensure-workers))
           (k (min (1+ (length pool)) (max 1 (floor n min-chunk)))))
      (cond
        ((< k 2) (funcall fn 0 n))
        (t
         (setf *job* fn *job-error* nil)
         (loop for i from 1 below k
               for w = (svref pool (1- i))
               do (setf (worker-lo w) (floor (* i n) k)
                        (worker-hi w) (floor (* (1+ i) n) k))
                  ;; the range has to be visible before the sequence number
                  ;; that says to go read it
                  (sb-thread:barrier (:memory))
                  (incf (worker-seq w))
                  (sb-thread:barrier (:memory))
                  (unless (zerop (worker-sleeping w))
                    (sb-thread:signal-semaphore (worker-go-sem w))))
         (unwind-protect (funcall fn 0 (floor n k))
           ;; every worker that was woken must be joined even if the caller's
           ;; own chunk threw, or the next job would run against a live one
           (loop for i from 1 below k
                 do (join-worker (svref pool (1- i)))))
         (when *job-error* (error *job-error*))))
      ;; ran here, on the pool — the caller must not run the range a second time.
      ;; NIL is reserved for "the lock was not free", which WITH-MUTEX returns for us.
      t)))

) ; #+sb-thread

#-sb-thread
(progn
  (defun shutdown-workers () (values))
  (defun parallel-range (n fn &key (min-chunk 1))
    (declare (ignore min-chunk))
    (funcall fn 0 n)))
