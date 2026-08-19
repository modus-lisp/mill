;;;; ops-seq.lisp — the ops a speech decoder needs that a vision model does not.
;;;;
;;;; LayerNormalization, Resize, LSTM and STFT.  They arrived together because Kokoro needs them
;;;; together: a transformer text encoder normalizes per token, a duration predictor runs a
;;;; bidirectional LSTM over the sequence, the decoder upsamples time with Resize, and the vocoder
;;;; ends in a short-time Fourier transform.  Nothing here is speech-specific in itself — an LSTM is
;;;; an LSTM — but this is the set that a VITS voice never asks for and a modern one always does.
;;;;
;;;; Each is written against the ONNX operator spec rather than against what one model happens to
;;;; use, EXCEPT where the spec has modes nobody here exercises: those signal rather than guess, so
;;;; a model that wants cubic Resize gets an error naming the mode instead of silently-wrong audio.

(in-package #:mill)

;;; ---- LayerNormalization ------------------------------------------------------
;;; Normalize each slice along AXIS..rank, then scale and shift.  Distinct from BatchNorm in that
;;; the statistics are per-sample and computed live, so there is nothing to fold at export time.

(defop "LayerNormalization" (node ins)
  (let* ((x (first ins)) (scale (second ins)) (bias (third ins))
         (rank (tensor-rank x))
         (axis (let ((a (node-attr node "axis" -1))) (if (minusp a) (+ rank a) a)))
         (eps (float (node-attr node "epsilon" 1.0d-5) 1d0))
         (inner (let ((n 1)) (loop for i from axis below rank do (setf n (* n (tensor-dim x i)))) n))
         (outer (floor (tensor-size x) (max 1 inner)))
         (res (make-tensor (tensor-dtype x) (tensor-shape x))))
    (when (zerop inner) (error "LayerNormalization: empty normalization axis"))
    (with-typed-data (dx x :dtypes (:f32 :f64))
      (with-typed-data (ds scale :dtypes (:f32 :f64))
        (with-typed-data (dr res :dtypes (:f32 :f64))
          (let ((db (and bias (tensor-data bias))))
            (dotimes (o outer)
              (let ((base (* o inner)) (mean 0d0) (var 0d0))
                (loop for i below inner do (incf mean (aref dx (+ base i))))
                (setf mean (/ mean inner))
                (loop for i below inner
                      do (let ((d (- (aref dx (+ base i)) mean))) (incf var (* d d))))
                (setf var (/ var inner))
                (let ((inv (/ 1d0 (sqrt (+ var eps)))))
                  (loop for i below inner
                        do (setf (aref dr (+ base i))
                                 (coerce (+ (* (- (aref dx (+ base i)) mean) inv (aref ds i))
                                            (if db (aref db i) 0d0))
                                         (array-element-type dr)))))))))))
    res))

;;; ---- Resize ------------------------------------------------------------------
;;; Scales-driven only, which is what an upsampling decoder emits; a SIZES-driven Resize would be
;;; the same loop with the scale derived per axis, but nothing here produces one, so it errors.

(defun %resize-src-index (dst scale in-len mode coord)
  "Where output index DST reads from, under COORD's convention.  Returns a double."
  (declare (ignorable in-len))
  (let ((s (if (zerop scale) 1d0 scale)))
    (cond ((string= coord "asymmetric") (/ (float dst 1d0) s))
          ((string= coord "half_pixel") (- (/ (+ (float dst 1d0) 0.5d0) s) 0.5d0))
          ((string= coord "align_corners")
           (if (<= (round (* in-len s)) 1) 0d0
               (/ (* (float dst 1d0) (- in-len 1)) (- (round (* in-len s)) 1))))
          (t (error "Resize: unsupported coordinate_transformation_mode ~s (~a)" coord mode)))))

(defop "Resize" (node ins)
  (let* ((x (first ins))
         (scales-t (third ins))
         (mode (node-attr node "mode" "nearest"))
         (coord (node-attr node "coordinate_transformation_mode" "half_pixel"))
         (nearest-mode (node-attr node "nearest_mode" "round_prefer_floor"))
         (rank (tensor-rank x)))
    (unless scales-t
      (error "Resize: only the scales form is implemented; this node gave sizes"))
    (unless (member mode '("nearest" "linear") :test #'string=)
      (error "Resize: unsupported mode ~s" mode))
    (let* ((sd (tensor-data scales-t))
           (scales (make-array rank :element-type 'double-float))
           (out-shape (make-array rank :element-type 'fixnum)))
      (dotimes (i rank)
        (setf (aref scales i) (float (aref sd i) 1d0)
              (aref out-shape i) (max 1 (floor (* (tensor-dim x i) (aref scales i))))))
      (let ((res (make-tensor (tensor-dtype x) out-shape)))
        (with-typed-data (dx x :dtypes (:f32 :f64))
          (with-typed-data (dr res :dtypes (:f32 :f64))
            ;; row-major strides for both sides, computed once
            (let ((istride (make-array rank :element-type 'fixnum))
                  (ostride (make-array rank :element-type 'fixnum))
                  (idx (make-array rank :element-type 'fixnum :initial-element 0)))
              (let ((si 1) (so 1))
                (loop for i from (1- rank) downto 0
                      do (setf (aref istride i) si (aref ostride i) so
                               si (* si (tensor-dim x i))
                               so (* so (aref out-shape i)))))
              (dotimes (o (length dr))
                ;; unflatten the output index
                (let ((rem o))
                  (dotimes (i rank)
                    (setf (aref idx i) (floor rem (aref ostride i))
                          rem (mod rem (aref ostride i)))))
                (if (string= mode "nearest")
                    (let ((off 0))
                      (dotimes (i rank)
                        (let* ((f (%resize-src-index (aref idx i) (aref scales i)
                                                     (tensor-dim x i) mode coord))
                               (s (cond ((string= nearest-mode "floor") (floor f))
                                        ((string= nearest-mode "ceil") (ceiling f))
                                        (t (round f)))))
                          (incf off (* (min (1- (tensor-dim x i)) (max 0 s)) (aref istride i)))))
                      (setf (aref dr o) (aref dx off)))
                    ;; linear: interpolate along every axis whose scale is not 1, which in practice
                    ;; is one axis — so this walks the 2^k corners of only the axes that move
                    (let ((axes '()))
                      (dotimes (i rank)
                        (unless (= 1d0 (aref scales i)) (push i axes)))
                      (let ((acc 0d0))
                        (dotimes (corner (ash 1 (length axes)))
                          (let ((off 0) (w 1d0) (bit 0))
                            (dotimes (i rank)
                              (if (member i axes)
                                  (let* ((f (%resize-src-index (aref idx i) (aref scales i)
                                                               (tensor-dim x i) mode coord))
                                         (f (max 0d0 (min (float (1- (tensor-dim x i)) 1d0) f)))
                                         (lo (floor f)) (frac (- f lo))
                                         (up (logbitp bit corner))
                                         (s (min (1- (tensor-dim x i)) (+ lo (if up 1 0)))))
                                    (setf w (* w (if up frac (- 1d0 frac))))
                                    (incf off (* s (aref istride i)))
                                    (incf bit))
                                  (incf off (* (aref idx i) (aref istride i)))))
                            (unless (zerop w) (incf acc (* w (aref dx off))))))
                        (setf (aref dr o) (coerce acc (array-element-type dr))))))))))
        res))))

;;; ---- LSTM --------------------------------------------------------------------
;;; ONNX packs the four gates in the order i, o, f, c — NOT the i, f, g, o that most papers and most
;;; other frameworks use.  Getting that wrong produces a model that runs, converges to plausible
;;; magnitudes, and says the wrong words, so it is the one thing here worth stating twice.

(defun %lstm-pack (data base rows cols)
  "Transpose the (ROWS, COLS) block at BASE into a fresh (COLS, ROWS) single-float array.

Done once per direction, so the per-timestep loop can walk a gate row CONTIGUOUSLY.  Without it
the natural loop reads W[g,k] for fixed g — a dot product, which needs a horizontal sum, which is
the one thing the vector vocabulary here deliberately does not have.  Transposed, the same
arithmetic becomes broadcast-and-accumulate across the 4H gates: eight at a time, no reduction."
  (declare (type dim base rows cols) (optimize (speed 3) (safety 0)))
  (let ((out (make-array (* rows cols) :element-type 'single-float)))
    (declare (type f32v out))
    (dotimes (r rows out)
      (let ((src (+ base (* r cols))))
        (declare (type dim src))
        (dotimes (c cols)
          (setf (aref out (+ (* c rows) r))
                (coerce (aref data (+ src c)) 'single-float)))))))

(defun %lstm-axpy (acc wt off n xv)
  "ACC[0,N) += XV * WT[OFF, OFF+N).  The whole inner loop of an LSTM, vectorized."
  (declare (type f32v acc wt) (type dim off n) (type single-float xv)
           (optimize (speed 3) (safety 0)))
  (let ((i 0) (xb (f32v-broadcast xv)))
    (declare (type dim i))
    (loop while (<= (+ i +f32-lanes+) n)
          do (setf (f32v-ref acc i)
                   (f32v+ (f32v-ref acc i) (f32v* xb (f32v-ref wt (+ off i)))))
             (incf i +f32-lanes+))
    (f32v-done)
    (loop while (< i n)
          do (setf (aref acc i) (+ (aref acc i) (* xv (aref wt (+ off i)))))
             (incf i))))

(defun %lstm-direction (dx dw dr db seq batch input hidden dir num-dir dh dc dy backward)
  "Run one direction in place.  DY is (seq, num-dir, batch, hidden); DH/DC are (num-dir, batch, hidden)."
  (declare (type fixnum seq batch input hidden dir num-dir)
           (optimize (speed 3) (safety 1)))
  (let* ((g4 (* 4 hidden))
         (wt (%lstm-pack dw (* dir g4 input) g4 input))      ; (input, 4H)
         (rt (%lstm-pack dr (* dir g4 hidden) g4 hidden))    ; (hidden, 4H)
         (bbase (* dir 8 hidden))
         (bias (make-array g4 :element-type 'single-float))
         (acc (make-array g4 :element-type 'single-float))
         (h (make-array (* batch hidden) :element-type 'single-float :initial-element 0.0))
         (c (make-array (* batch hidden) :element-type 'single-float :initial-element 0.0)))
    (declare (type f32v wt rt bias acc h c))
    (dotimes (g g4)
      (setf (aref bias g) (+ (coerce (aref db (+ bbase g)) 'single-float)
                             (coerce (aref db (+ bbase g4 g)) 'single-float))))
    (when dh (dotimes (i (* batch hidden))
               (setf (aref h i) (coerce (aref dh (+ (* dir batch hidden) i)) 'single-float))))
    (when dc (dotimes (i (* batch hidden))
               (setf (aref c i) (coerce (aref dc (+ (* dir batch hidden) i)) 'single-float))))
    (dotimes (step seq)
      (let ((tt (if backward (- seq 1 step) step)))
        (declare (type fixnum tt))
        (dotimes (b batch)
          (replace acc bias)
          (let ((xrow (+ (* tt batch input) (* b input))))
            (declare (type dim xrow))
            (dotimes (k input)
              (let ((xv (coerce (aref dx (+ xrow k)) 'single-float)))
                (unless (zerop xv) (%lstm-axpy acc wt (* k g4) g4 xv)))))
          (let ((hrow (* b hidden)))
            (declare (type dim hrow))
            (dotimes (k hidden)
              (let ((hv (aref h (+ hrow k))))
                (unless (zerop hv) (%lstm-axpy acc rt (* k g4) g4 hv)))))
          ;; ONNX gate order: i, o, f, c
          (dotimes (k hidden)
            (let* ((i-g (/ 1.0 (+ 1.0 (exp (- (aref acc k))))))
                   (o-g (/ 1.0 (+ 1.0 (exp (- (aref acc (+ hidden k)))))))
                   (f-g (/ 1.0 (+ 1.0 (exp (- (aref acc (+ (* 2 hidden) k)))))))
                   (c-g (tanh (aref acc (+ (* 3 hidden) k))))
                   (idx (+ (* b hidden) k))
                   (cc (+ (* f-g (aref c idx)) (* i-g c-g))))
              (setf (aref c idx) cc
                    (aref h idx) (* o-g (tanh cc)))
              (setf (aref dy (+ (* tt num-dir batch hidden) (* dir batch hidden) idx))
                    (coerce (aref h idx) (array-element-type dy))))))))
    (values h c)))

(defop "LSTM" (node ins)
  (let* ((x (first ins)) (w (second ins)) (r (third ins)) (b (fourth ins))
         (initial-h (sixth ins)) (initial-c (seventh ins))
         (direction (node-attr node "direction" "forward"))
         (num-dir (if (string= direction "bidirectional") 2 1))
         (seq (tensor-dim x 0)) (batch (tensor-dim x 1)) (input (tensor-dim x 2))
         (hidden (node-attr node "hidden_size" (floor (tensor-dim w 1) 4)))
         (y (make-tensor :f32 (vector seq num-dir batch hidden)))
         (yh (make-tensor :f32 (vector num-dir batch hidden)))
         (yc (make-tensor :f32 (vector num-dir batch hidden))))
    (unless (member direction '("forward" "bidirectional" "reverse") :test #'string=)
      (error "LSTM: unsupported direction ~s" direction))
    (let ((dx (tensor-data x)) (dw (tensor-data w)) (drr (tensor-data r))
          (db (if b (tensor-data b)
                  (make-array (* num-dir 8 hidden) :element-type 'single-float
                                                   :initial-element 0.0)))
          (dh (and initial-h (tensor-data initial-h)))
          (dc (and initial-c (tensor-data initial-c)))
          (dy (tensor-data y)) (dyh (tensor-data yh)) (dyc (tensor-data yc)))
      (dotimes (dir num-dir)
        (multiple-value-bind (h c)
            (%lstm-direction dx dw drr db seq batch input hidden dir num-dir dh dc dy
                             (or (string= direction "reverse")
                                 (and (= num-dir 2) (= dir 1))))
          (dotimes (i (* batch hidden))
            (setf (aref dyh (+ (* dir batch hidden) i)) (coerce (aref h i) 'single-float)
                  (aref dyc (+ (* dir batch hidden) i)) (coerce (aref c i) 'single-float))))))
    (list y yh yc)))

;;; ---- STFT --------------------------------------------------------------------
;;; A direct DFT per frame.  The vocoder runs this once, on one signal, so an FFT would buy
;;; correctness nothing and cost the reader the one place in this file where the maths is plain.

(defop "STFT" (node ins)
  (let* ((signal (first ins)) (frame-step-t (second ins))
         (window (third ins)) (frame-length-t (fourth ins))
         (onesided (eql 1 (node-attr node "onesided" 1)))
         (batch (tensor-dim signal 0))
         (siglen (tensor-dim signal 1))
         (step (round (aref (tensor-data frame-step-t) 0)))
         (dwin (and window (tensor-data window)))
         (flen (cond (frame-length-t (round (aref (tensor-data frame-length-t) 0)))
                     (window (tensor-dim window 0))
                     (t (error "STFT: neither frame_length nor a window was given"))))
         (nfreq (if onesided (1+ (floor flen 2)) flen))
         (frames (if (< siglen flen) 0 (1+ (floor (- siglen flen) step))))
         (res (make-tensor :f32 (vector batch frames nfreq 2)))
         (ds (tensor-data signal)) (dr (tensor-data res)))
    (dotimes (bt batch)
      (dotimes (f frames)
        (let ((off (+ (* bt siglen) (* f step))))
          (dotimes (k nfreq)
            (let ((re 0d0) (im 0d0)
                  (w (/ (* -2d0 pi k) flen)))
              (dotimes (n flen)
                (let ((v (* (float (aref ds (+ off n)) 1d0)
                            (if dwin (float (aref dwin n) 1d0) 1d0)))
                      (ang (* w n)))
                  (incf re (* v (cos ang)))
                  (incf im (* v (sin ang)))))
              (let ((base (* 2 (+ (* bt frames nfreq) (* f nfreq) k))))
                (setf (aref dr base) (coerce re 'single-float)
                      (aref dr (1+ base)) (coerce im 'single-float))))))))
    res))
