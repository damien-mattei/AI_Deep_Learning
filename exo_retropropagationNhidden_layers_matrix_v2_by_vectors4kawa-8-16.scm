(require Scheme+)

(require array)

(require matrix)

(import (only (kawa base) (+ orig+) (display orig-display)))

(define-overload-existing-operator + orig+)

(define-overload-procedure random)

(define (random-int n) (integer (* n (random))))

(overload-existing-operator + vector-append (vector? vector?))

(define d0 (->double 0.0))

(define d1 (->double 1.0))

(overload-procedure random (lambda () (->double (java.lang.Math:random)))
 ())

(overload-procedure random random-int (integer?))

(define (uniform-dummy dummy1 dummy2)
 (->double ($nfx$ -1.0 + (random) * 2.0)))

(define (uniform-interval inf :: double sup :: double)
 (define gap :: double (- sup inf)) ($nfx$ inf + gap * (random)))

(define (σ z̃ :: double) (/ d1 (+ d1 (exp (- z̃)))))

(define (der_tanh z :: double z̃ :: double) ($nfx$ d1 - z ** 2))

(define (der_σ z :: double z̃ :: double) (* z (- d1 z)))

(define (der_atan z :: double z̃ :: double) (/ 1 ($nfx$ d1 + z̃ ** 2)))

(define
 (modification_des_poids M_i_o η :: double z_input z_output z̃_output
  ᐁ_i_o მzⳆმz̃)
 (<+ (len_layer_output len_layer_input_plus1forBias) (M_i_o:dim))
 ($nfx$ len_layer_input <+ len_layer_input_plus1forBias - 1)
 (for-each-in (j (in-range len_layer_output))
  (for-each-in (i (in-range len_layer_input))
   ($nfx$ (bracket-apply M_i_o j (+ i 1)) <- (bracket-apply M_i_o j (+ i 1)) -
    (- η) * (bracket-apply z_input i) *
    (მzⳆმz̃ (bracket-apply z_output j) (bracket-apply z̃_output j)) *
    (bracket-apply ᐁ_i_o j)))
  ($nfx$ (bracket-apply M_i_o j 0) <- (bracket-apply M_i_o j 0) - (- η) * 1.0
   * (მzⳆმz̃ (bracket-apply z_output j) (bracket-apply z̃_output j)) *
   (bracket-apply ᐁ_i_o j))))

(define-simple-class ReseauRetroPropagation () (nbiter init-value: 3)
 (activation_function_hidden_layer) (activation_function_output_layer)
 (activation_function_hidden_layer_derivative)
 (activation_function_output_layer_derivative) (ηₛ :: double 1.0) (z) (z̃)
 (M) (ᐁ) (eror :: double 0.0)
 ((*init* nc nbiter0 ηₛ0 :: double activation_function_hidden_layer0
   activation_function_output_layer0
   activation_function_hidden_layer_derivative0
   activation_function_output_layer_derivative0)
  (display "*init* : nc=") (display nc) (newline) (<- nbiter nbiter0)
  (<- ηₛ ηₛ0)
  (<- activation_function_hidden_layer activation_function_hidden_layer0)
  (<- activation_function_output_layer activation_function_output_layer0)
  (<- activation_function_hidden_layer_derivative
   activation_function_hidden_layer_derivative0)
  (<- activation_function_output_layer_derivative
   activation_function_output_layer_derivative0)
  (<+ lnc (vector-length nc)) (define (make-vector-z lg) (make-vector lg d0))
  (<- z (vector-map make-vector-z nc)) (display "z=") (display z) (newline)
  (<- z̃ (vector-map make-vector-z nc)) (display "z̃=") (display z̃)
  (newline)
  (<- M
   (vector-map
    (lambda (n)
     (create-matrix-f64-by-function uniform-dummy (bracket-apply nc n + 1)
      (+ (bracket-apply nc n) 1)))
    ($bracket-list$ 0 <: (- lnc 1))))
  (display "M=") (display M) (newline) (<- ᐁ (vector-map make-vector-z nc))
  (display "ᐁ=") (display ᐁ) (newline) (display "nbiter=") (display nbiter)
  (newline))
 ((display-retro-propag) (display "vector of matrices  M=") (newline)
  (for-each (lambda (mt) (mt:display-matrix) (newline)) M) (newline))
 ((display . L)
  (cond ((null? L) (display-retro-propag))
   ((= (length L) 2) (orig-display (car L) (cadr L)))
   (else (orig-display (car L)))))
 ((accepte_et_propage x)
  (when (≠ (vector-length x) (vector-length (bracket-apply z 0)))
   (display "Mauvais nombre d'entrées !") (newline) (exit #f))
  (<- (bracket-apply z 0) x) (<+ n (vector-length z)) (declare z_1) (declare i)
  (for ((<- i 0) ($nfx$ i < n - 2) ($nfx$ i <- i + 1))
   ($nfx$ z_1 <- #(1) + (bracket-apply z i))
   ($nfx$ (bracket-apply z̃ i + 1) <- (bracket-apply M i) * z_1)
   (<- (bracket-apply z i + 1)
    (vector-map activation_function_hidden_layer (bracket-apply z̃ i + 1))))
  ($nfx$ z_1 <- #(1) + (bracket-apply z i))
  ($nfx$ (bracket-apply z̃ i + 1) <- (bracket-apply M i) * z_1)
  (<- (bracket-apply z i + 1)
   (vector-map activation_function_output_layer (bracket-apply z̃ i + 1))))
 ((apprentissage Lexemples) (<+ ip 0) (declare x y)
  (for-each-in (it (in-range nbiter))
   (if ($nfx$ it % 1000 = 0) then (display it) (newline))
   (<- x (car (bracket-apply Lexemples ip)))
   (<- y (cdr (bracket-apply Lexemples ip))) (accepte_et_propage x)
   ($nfx$ i <+ i_output_layer <+ (vector-length z) - 1)
   (<+ ns (vector-length (bracket-apply z i)))
   (for-each-in (k (in-range ns))
    ($nfx$ (bracket-apply (bracket-apply ᐁ i) k) <- (bracket-apply y k) -
     (bracket-apply (bracket-apply z i) k)))
   (<+ მzⳆმz̃ activation_function_output_layer_derivative)
   (modification_des_poids (bracket-apply M i - 1) ηₛ (bracket-apply z i - 1)
    (bracket-apply z i) (bracket-apply z̃ i) (bracket-apply ᐁ i) მzⳆმz̃)
   (<- მzⳆმz̃ activation_function_hidden_layer_derivative)
   (for-each-in (i (reversed (in-range 1 i_output_layer)))
    (<+ nc (vector-length (bracket-apply z i)))
    (<+ ns (vector-length (bracket-apply z i + 1)))
    (for-each-in (j (in-range nc))
     (<- (bracket-apply (bracket-apply ᐁ i) j) d0)
     (for-each-in (k (in-range ns))
      ($nfx$ (bracket-apply (bracket-apply ᐁ i) j) <-
       (bracket-apply (bracket-apply ᐁ i) j) +
       (მzⳆმz̃ (bracket-apply (bracket-apply z i + 1) k)
        (bracket-apply (bracket-apply z̃ i + 1) k))
       * (bracket-apply (bracket-apply M i) k (+ j 1)) *
       (bracket-apply (bracket-apply ᐁ i + 1) k))))
    (modification_des_poids (bracket-apply M i - 1) ηₛ
     (bracket-apply z i - 1) (bracket-apply z i) (bracket-apply z̃ i)
     (bracket-apply ᐁ i) მzⳆმz̃))
   (<- ip (random (vector-length Lexemples)))))
 ((test Lexemples) (display "Test des exemples :") (newline) (<+ err d0)
  (declare entree sortie_attendue ᐁ)
  (for-each-in (entree-sortie_attendue Lexemples)
   (<- entree (car entree-sortie_attendue))
   (<- sortie_attendue (cdr entree-sortie_attendue))
   (accepte_et_propage entree)
   (format #t "~a --> ~a : on attendait ~a~%" entree
    (bracket-apply z (vector-length z) - 1) sortie_attendue)
   ($nfx$ ᐁ <- (bracket-apply sortie_attendue 0) -
    (bracket-apply (bracket-apply z (vector-length z) - 1) 0))
   ($nfx$ err <- err + ᐁ ** 2))
  ($nfx$ err <- err * (->double 0.5)) (display "Error on examples=")
  (display err) (newline)))

(display "################## NOT ##################")

(newline)

(<+ r1 (ReseauRetroPropagation #(1 2 1) 5000 10 σ σ der_σ der_σ))

(<+ Lexemples1
 (vector (cons (vector d1) (vector d0)) (cons (vector d0) (vector d1))))

(r1:apprentissage Lexemples1)

(newline)

(r1:display)

(newline)

(r1:test Lexemples1)

(define precision (->double 10.0))

(display "precision=")

(display precision)

(newline)

(define (trunc x :: double) (->double (/ (round (* precision x)) precision)))

(define (trunc-matrix mt) (mt:apply trunc))

(for-each trunc-matrix r1:M)

(display "Matrix vector modified r1:M=")

(newline)

(r1:display-retro-propag)

(newline)

(r1:test Lexemples1)

(newline)

(display "################## XOR ##################")

(newline)

(<+ r2 (ReseauRetroPropagation #(2 8 1) 250000 0.1 σ σ der_σ der_σ))

(<+ Lexemples2
 (vector (cons (vector d1 d0) (vector d1)) (cons (vector d0 d0) (vector d0))
  (cons (vector d0 d1) (vector d1)) (cons (vector d1 d1) (vector d0))))

(r2:apprentissage Lexemples2)

(r2:test Lexemples2)

(newline)

(newline)

(r2:display-retro-propag)

(newline)

(for-each trunc-matrix r2:M)

(display "Matrix vector modified r2:M=")

(newline)

(r2:display-retro-propag)

(newline)

(r2:test Lexemples2)

(newline)

(display "################## SINE ##################")

(newline)

(<+ r3
 (ReseauRetroPropagation #(1 70 70 1) 50000 0.01 atan tanh der_atan der_tanh))

(declare pi)

($nfx$ pi <- 4 * (atan 1))

(<+ Llearning
 (vector-map (lambda (x) (cons (vector x) (vector (sin x))))
  (list->vector
   (map (lambda (n) (uniform-interval (- pi) pi)) (in-range 10000)))))

(<+ Ltest
 (vector-map (lambda (x) (cons (vector x) (vector (sin x))))
  (list->vector
   (map (lambda (n) (uniform-interval (/ (- pi) 2) (/ pi 2)))
    (in-range 10000)))))

(r3:apprentissage Llearning)

(r3:test Ltest)

(newline)

