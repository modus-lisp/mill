;;;; profile.lisp — where the time went, per node and per op.
;;;;
;;;; The runner already knows how to say what each node computed; this is the
;;;; other half of the same question, which nothing could answer until now: how
;;;; long each node took.  Before this, "MatMul is the bottleneck" was an opinion
;;;; formed by reading the source — a good opinion, possibly, but the same kind
;;;; of thing as guessing which line of a program is hot.
;;;;
;;;; Accounting is per NODE, keyed on the node object itself, and summed per op
;;;; only when the report is printed.  A graph is not uniform: a Zipformer has
;;;; MatMuls two orders of magnitude apart in size, and an op-level total that
;;;; says "MatMul, 60%" does not say whether that is one node or four hundred.
;;;; The node table answers that; the op table is what you optimize against.
;;;;
;;;; Cost when off is one special-variable read per node.  Cost when on is two
;;;; clock reads per node — microsecond resolution, so a node under a microsecond
;;;; reads as zero and is, for our purposes, free.  Nothing here is subtracted
;;;; for overhead: the number is wall time inside the op, measured the same way
;;;; for every op, and the comparison between ops is what it exists to support.

(in-package #:mill)

(defvar *op-times* nil
  "When non-NIL, an EQ hash table node -> PROFILE-ENTRY that RUN-MODEL fills in.
Bound by WITH-PROFILING; NIL everywhere else, which is what makes the runner pay
nothing for this.")

(defstruct (profile-entry (:conc-name pe-))
  (op "" :type string)
  (name "" :type string)
  (calls 0 :type fixnum)
  (ticks 0 :type fixnum))

(declaim (inline profile-entry-for))
(defun profile-entry-for (node)
  (or (gethash node *op-times*)
      (setf (gethash node *op-times*)
            (make-profile-entry :op (node-op node) :name (node-name node)))))

(defmacro with-profiling (&body body)
  "Run BODY with per-node timing on, and return what BODY returned.  The table is
left in *OP-TIMES* for the caller to report on — nest this around a whole
workload, not around one call, since the interesting thing is the total."
  `(let ((*op-times* (make-hash-table :test #'eq :size 4096)))
     (multiple-value-prog1 (progn ,@body)
       (setf *last-profile* *op-times*))))

(defvar *last-profile* nil
  "The table the most recent WITH-PROFILING filled, so a script can report after
the fact without threading it through.")

(defun profile-seconds (ticks)
  (/ (float ticks 1d0) internal-time-units-per-second))

(defun op-profile (&optional (table *last-profile*))
  "(op calls seconds) per op type, slowest first."
  (let ((by-op (make-hash-table :test #'equal)))
    (maphash (lambda (node pe)
               (declare (ignore node))
               (let ((cell (or (gethash (pe-op pe) by-op)
                               (setf (gethash (pe-op pe) by-op) (list 0 0 0)))))
                 (incf (first cell) (pe-calls pe))
                 (incf (second cell) (pe-ticks pe))
                 (incf (third cell))))
             (or table (make-hash-table)))
    (sort (loop for op being the hash-keys of by-op using (hash-value cell)
                collect (list op (first cell) (profile-seconds (second cell))
                              (third cell)))
          #'> :key #'third)))

(defun report-profile (&key (table *last-profile*) (nodes 12) (stream *standard-output*))
  "Print where the time went: every op, then the individual nodes that cost most.

The percentages are of profiled time, not of the process — everything outside
RUN-MODEL (feature extraction, the search, loading) is not in here at all, and a
workload that is 40% search will still show its ops summing to 100%."
  (unless table
    (format stream "~&nothing profiled~%")
    (return-from report-profile))
  (let* ((rows (op-profile table))
         (total (reduce #'+ rows :key #'third :initial-value 0d0)))
    (format stream "~&~,3f s in ~d op types over ~d nodes~2%"
            total (length rows)
            (hash-table-count table))
    (format stream "~&  ~16a ~8@a ~10@a ~9@a ~10@a~%"
            "op" "nodes" "calls" "seconds" "share")
    (dolist (r rows)
      (destructuring-bind (op calls seconds node-count) r
        (when (> seconds (* 0.0005d0 total))
          (format stream "  ~16a ~8d ~10d ~9,3f ~9,1f%~%"
                  op node-count calls seconds
                  (if (zerop total) 0 (* 100 (/ seconds total)))))))
    (when (plusp nodes)
      (format stream "~%  the ~d most expensive individual nodes:~%" nodes)
      (let ((all (sort (loop for pe being the hash-values of table collect pe)
                       #'> :key #'pe-ticks)))
        (loop for pe in all
              repeat nodes
              do (format stream "  ~16a ~8d ~9,3f ~9,1f%   ~a~%"
                         (pe-op pe) (pe-calls pe) (profile-seconds (pe-ticks pe))
                         (if (zerop total) 0 (* 100 (/ (profile-seconds (pe-ticks pe)) total)))
                         (pe-name pe)))))
    (finish-output stream)
    total))
