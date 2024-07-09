(use-modules
  (Scheme+)
  (matrix+)
  ((srfi srfi-42)
   #:renamer
   (lambda (sym) (if (eq? sym ':) 's42-: sym)))
  (oop goops)
  (srfi srfi-43))


(define-method
  (+ (x <vector>) (y <vector>))
  (vector-append x y))


(overload-square-brackets
  matrix-ref
  matrix-set!
  (matrix? number? number?))


(overload-square-brackets
  matrix-line-ref
  matrix-line-set!
  (matrix? number?))


(define (uniform-dummy dummy1 dummy2)
  (+ -1 (random 2.0)))


(define (uniform-interval inf sup)
  (<+ gap (- sup inf))
  (+ inf (random (* gap 1.0))))


(define (σ z̃) (/ 1 (+ 1 (exp (- z̃)))))


(define (der_tanh z z̃) (- 1 (** z 2)))


(define (der_σ z z̃) (* z (- 1 z)))


(define (der_atan z z̃) (/ 1 (+ 1 (** z̃ 2))))


(define (modification_des_poids
         M_i_o
         η
         z_input
         z_output
         z̃_output
         ᐁ_i_o
         მzⳆმz̃)
  (<+ (len_layer_output len_layer_input_plus1forBias)
      (dim-matrix M_i_o))
  (<+ len_layer_input
      (- len_layer_input_plus1forBias 1))
  (for-each-in
    (j (in-range len_layer_output))
    (for-each-in
      (i (in-range len_layer_input))
      (<- ($bracket-apply$next M_i_o j (+ i 1))
          (+ ($bracket-apply$next M_i_o j (+ i 1))
             (* η
                ($bracket-apply$next z_input i)
                (მzⳆმz̃
                  ($bracket-apply$next z_output j)
                  ($bracket-apply$next z̃_output j))
                ($bracket-apply$next ᐁ_i_o j)))))
    (<- ($bracket-apply$next M_i_o j 0)
        (+ ($bracket-apply$next M_i_o j 0)
           (* η
              1.0
              (მzⳆმz̃
                ($bracket-apply$next z_output j)
                ($bracket-apply$next z̃_output j))
              ($bracket-apply$next ᐁ_i_o j))))))


(define-class
  ReseauRetroPropagation
  ()
  (nbiter
    #:init-value
    3
    #:init-keyword
    #:nbiter
    #:getter
    nbp-get-nbiter
    #:setter
    nbp-set-nbiter!)
  (activation_function_hidden_layer
    #:init-keyword
    #:activation_function_hidden_layer
    #:getter
    nbp-get-activation_function_hidden_layer)
  (activation_function_output_layer
    #:init-keyword
    #:activation_function_output_layer
    #:getter
    nbp-get-activation_function_output_layer)
  (activation_function_hidden_layer_derivative
    #:init-keyword
    #:activation_function_hidden_layer_derivative
    #:getter
    nbp-get-activation_function_hidden_layer_derivative)
  (activation_function_output_layer_derivative
    #:init-keyword
    #:activation_function_output_layer_derivative
    #:getter
    nbp-get-activation_function_output_layer_derivative)
  (ηₛ #:init-value
      1.0
      #:init-keyword
      #:ηₛ
      #:getter
      nbp-get-ηₛ)
  (z #:getter nbp-get-z #:setter nbp-set-z!)
  (z̃ #:getter nbp-get-z̃ #:setter nbp-set-z̃!)
  (M #:getter nbp-get-M #:setter nbp-set-M!)
  (ᐁ #:getter nbp-get-ᐁ #:setter nbp-set-ᐁ!)
  (eror #:init-value 0))


(define (*init* nc nbp)
  (display "*init* : nc=")
  (display nc)
  (newline)
  (<+ lnc (vector-length nc))
  (define (make-vector-zero i lg)
    (make-vector lg 0))
  (declare z z̃ M ᐁ)
  (<+ nbiter (nbp-get-nbiter nbp))
  (<- z (vector-map make-vector-zero nc))
  (display "z=")
  (display z)
  (newline)
  (<- z̃ (vector-map make-vector-zero nc))
  (display "z̃=")
  (display z̃)
  (newline)
  (<- M
      (vector-ec
        (s42-: n (- lnc 1))
        (create-matrix-by-function
          uniform-dummy
          ($bracket-apply$next nc (+ n 1))
          (+ ($bracket-apply$next nc n) 1))))
  (display "M=")
  (display M)
  (newline)
  (<- ᐁ (vector-map make-vector-zero nc))
  (display "ᐁ=")
  (display ᐁ)
  (newline)
  (display "nbiter=")
  (display nbiter)
  (newline)
  (nbp-set-z! nbp z)
  (nbp-set-z̃! nbp z̃)
  (nbp-set-M! nbp M)
  (nbp-set-ᐁ! nbp ᐁ))


(define (accepte_et_propage x nbp)
  (<+ z (nbp-get-z nbp))
  (when (≠ (vector-length x)
           (vector-length ($bracket-apply$next z 0)))
        (display "Mauvais nombre d'entrées !")
        (newline)
        (exit #f))
  (<- ($bracket-apply$next z 0) x)
  (<+ n (vector-length z))
  (declare z_1)
  (<+ z̃ (nbp-get-z̃ nbp))
  (<+ M (nbp-get-M nbp))
  (<+ activation_function_hidden_layer
      (nbp-get-activation_function_hidden_layer nbp))
  (define (activation_function_hidden_layer_indexed i z̃)
    (activation_function_hidden_layer z̃))
  (<+ activation_function_output_layer
      (nbp-get-activation_function_output_layer nbp))
  (define (activation_function_output_layer_indexed i z̃)
    (activation_function_output_layer z̃))
  (declare i)
  (for ((<- i 0) (< i (- n 2)) (<- i (+ i 1)))
       (<- z_1 (+ #(1) ($bracket-apply$next z i)))
       (<- ($bracket-apply$next z̃ (+ i 1))
           (* ($bracket-apply$next M i) z_1))
       (<- ($bracket-apply$next z (+ i 1))
           (vector-map
             activation_function_hidden_layer_indexed
             ($bracket-apply$next z̃ (+ i 1)))))
  (<- z_1 (+ #(1) ($bracket-apply$next z i)))
  (<- ($bracket-apply$next z̃ (+ i 1))
      (* ($bracket-apply$next M i) z_1))
  (<- ($bracket-apply$next z (+ i 1))
      (vector-map
        activation_function_output_layer_indexed
        ($bracket-apply$next z̃ (+ i 1))))
  (nbp-set-z! nbp z)
  (nbp-set-z̃! nbp z̃))


(define (apprentissage Lexemples nbp)
  (<+ ip 0)
  (<+ z (nbp-get-z nbp))
  (<+ z̃ (nbp-get-z̃ nbp))
  (<+ M (nbp-get-M nbp))
  (<+ ᐁ (nbp-get-ᐁ nbp))
  (<+ activation_function_output_layer_derivative
      (nbp-get-activation_function_output_layer_derivative
        nbp))
  (<+ activation_function_hidden_layer_derivative
      (nbp-get-activation_function_hidden_layer_derivative
        nbp))
  (<+ ηₛ (nbp-get-ηₛ nbp))
  (<+ nbiter (nbp-get-nbiter nbp))
  (declare x y)
  (for-each-in
    (it (in-range nbiter))
    (when (= (% it 1000) 0) (display it) (newline))
    (<- x (car ($bracket-apply$next Lexemples ip)))
    (<- y (cdr ($bracket-apply$next Lexemples ip)))
    (accepte_et_propage x nbp)
    (<+ i i_output_layer (- (vector-length z) 1))
    (<+ ns (vector-length ($bracket-apply$next z i)))
    (for-each-in
      (k (in-range ns))
      (<- ($bracket-apply$next ($bracket-apply$next ᐁ i) k)
          (- ($bracket-apply$next y k)
             ($bracket-apply$next ($bracket-apply$next z i) k))))
    (<+ მzⳆმz̃
        activation_function_output_layer_derivative)
    (modification_des_poids
      ($bracket-apply$next M (- i 1))
      ηₛ
      ($bracket-apply$next z (- i 1))
      ($bracket-apply$next z i)
      ($bracket-apply$next z̃ i)
      ($bracket-apply$next ᐁ i)
      მzⳆმz̃)
    (<- მzⳆმz̃
        activation_function_hidden_layer_derivative)
    (for-each-in
      (i (reversed (in-range 1 i_output_layer)))
      (<+ nc (vector-length ($bracket-apply$next z i)))
      (<+ ns
          (vector-length ($bracket-apply$next z (+ i 1))))
      (for-each-in
        (j (in-range nc))
        (<+ k 0)
        (<- ($bracket-apply$next ($bracket-apply$next ᐁ i) j)
            ($+> (<+ sum 0)
                 (for-each-in
                   (k (in-range ns))
                   (<- sum
                       (+ sum
                          (* (მzⳆმz̃
                               ($bracket-apply$next
                                 ($bracket-apply$next z (+ i 1))
                                 k)
                               ($bracket-apply$next
                                 ($bracket-apply$next z̃ (+ i 1))
                                 k))
                             ($bracket-apply$next
                               ($bracket-apply$next M i)
                               k
                               (+ j 1))
                             ($bracket-apply$next
                               ($bracket-apply$next ᐁ (+ i 1))
                               k)))))
                 sum)))
      (modification_des_poids
        ($bracket-apply$next M (- i 1))
        ηₛ
        ($bracket-apply$next z (- i 1))
        ($bracket-apply$next z i)
        ($bracket-apply$next z̃ i)
        ($bracket-apply$next ᐁ i)
        მzⳆმz̃))
    (<- ip (random (vector-length Lexemples))))
  (nbp-set-z! nbp z)
  (nbp-set-z̃! nbp z̃)
  (nbp-set-M! nbp M)
  (nbp-set-ᐁ! nbp ᐁ))


(define (test Lexemples nbp)
  (<+ z (nbp-get-z nbp))
  (display "Test des exemples :")
  (newline)
  (<+ err 0)
  (declare entree sortie_attendue ᐁ)
  (for-each-in
    (entree-sortie_attendue (vector->list Lexemples))
    (<- entree (car entree-sortie_attendue))
    (<- sortie_attendue (cdr entree-sortie_attendue))
    (accepte_et_propage entree nbp)
    (format
      #t
      "~a --> ~a : on attendait ~a~%"
      entree
      ($bracket-apply$next z (- (vector-length z) 1))
      sortie_attendue)
    (<- ᐁ
        (- ($bracket-apply$next sortie_attendue 0)
           ($bracket-apply$next
             ($bracket-apply$next z (- (vector-length z) 1))
             0)))
    (<- err (+ err (** ᐁ 2))))
  (<- err (* err 0.5))
  (display "Error on examples=")
  (display err)
  (newline))


(display
  "################## NOT ##################")


(newline)


(<+ r1
    (make ReseauRetroPropagation
          #:nbiter
          5000
          #:ηₛ
          10
          #:activation_function_hidden_layer
          σ
          #:activation_function_output_layer
          σ
          #:activation_function_hidden_layer_derivative
          der_σ
          #:activation_function_output_layer_derivative
          der_σ))


(<+ Lexemples1 #((#(1) . #(0)) (#(0) . #(1))))


(*init* #(1 2 1) r1)


(apprentissage Lexemples1 r1)


(test Lexemples1 r1)


(newline)


(display
  "################## XOR ##################")


(newline)


(<+ r2
    (make ReseauRetroPropagation
          #:nbiter
          250000
          #:ηₛ
          0.1
          #:activation_function_hidden_layer
          σ
          #:activation_function_output_layer
          σ
          #:activation_function_hidden_layer_derivative
          der_σ
          #:activation_function_output_layer_derivative
          der_σ))


(<+ Lexemples2
    #((#(1 0) . #(1))
      (#(0 0) . #(0))
      (#(0 1) . #(1))
      (#(1 1) . #(0))))


(*init* #(2 8 1) r2)


(apprentissage Lexemples2 r2)


(test Lexemples2 r2)


(newline)


(display
  "################## SINE ##################")


(newline)


(<+ r3
    (make ReseauRetroPropagation
          #:nbiter
          50000
          #:ηₛ
          0.01
          #:activation_function_hidden_layer
          atan
          #:activation_function_output_layer
          tanh
          #:activation_function_hidden_layer_derivative
          der_atan
          #:activation_function_output_layer_derivative
          der_tanh))


(declare pi)


(<- pi (* 4 (atan 1)))


(<+ Llearning
    (vector-map
      (lambda (i x) (cons (vector x) (vector (sin x))))
      (list->vector
        (map (lambda (n) (uniform-interval (- pi) pi))
             (in-range 10000)))))


(*init* #(1 70 70 1) r3)


(<+ Ltest
    (vector-map
      (lambda (i x) (cons (vector x) (vector (sin x))))
      (list->vector
        (map (lambda (n)
               (uniform-interval (/ (- pi) 2) (/ pi 2)))
             (in-range 10000)))))


(apprentissage Llearning r3)


(test Ltest r3)


(newline)


