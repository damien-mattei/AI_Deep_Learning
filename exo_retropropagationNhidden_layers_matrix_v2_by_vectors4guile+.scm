
; Deep Learning : back propagation, gradient descent, neural network with N hidden layers

; L'algorithme de rétro-propagation du gradient dans un
; réseau de neurones avec N couches cachées.

;  D. Mattei	


; use MacVim to show ALL the characters of this file (not Emacs, not Aquamacs)
; jeu de couleurs: Torte ou Koehler

; use: (load "exo_retropropagationNhidden_layers_matrix_v2_by_vectors4guile+.scm")


(use-modules (Scheme+)
	     (matrix+)
	     (srfi srfi-42) ; Eager Comprehensions
	     (oop goops)
	     (srfi srfi-43)) ; vectors


;scheme@(guile-user)> {#(1 2) + #(3 4 5)}
;$1 = #(1 2 3 4 5)
;scheme@(guile-user)> {#(1 2) + #(3 4 5) + #(6 7)}
;$2 = #(1 2 3 4 5 6 7)
(define-method (+ (x <vector>) (y <vector>)) (vector-append x y))

;; ; first stage overloading
;; (define-overload-existing-operator +)

;; ; second stage overloading
;; (overload-existing-operator + vector-append (vector? vector?))



;; overload [ ] 
(overload-square-brackets matrix-ref matrix-set!  (matrix? number? number?))
(overload-square-brackets matrix-line-ref matrix-line-set! (matrix? number?))

(define (uniform-dummy dummy1 dummy2) {-1 + (random 2.0)}) 

; return a random number between [inf, sup]
(define (uniform-interval inf sup)
  {gap <+ {sup - inf}}
  {inf + (random {gap * 1.0})})

; sigmoïde
(define (σ z̃) 
  {1 / {1 + (exp (- z̃))}})

; some derivatives
(define (der_tanh z z̃)
  {1 - z ** 2})	

(define (der_σ z z̃)
    {z * {1 - z}})

(define (der_atan z z̃)
  {1 / {1 + z̃ ** 2}})



#| this is a Scheme multi line comment,
but will it works with Scheme+ parser?
|#

; modify coefficients layer
(define (modification_des_poids M_i_o η z_input z_output z̃_output ᐁ_i_o მzⳆმz̃) ; derivative of activation function of the layer
	 
	  ; the length of output and input layer with coeff. used for bias update
	  {(len_layer_output len_layer_input_plus1forBias) <+ (dim-matrix M_i_o)} ; use values and define-values to create bindings
        
	  {len_layer_input <+ {len_layer_input_plus1forBias - 1}}

	  (for-each-in (j (in-range len_layer_output)) ; line
		(for-each-in (i (in-range len_layer_input)) ; column , parcours les colonnes de la ligne sauf le bias
		    {M_i_o[j {i + 1}]  <-  M_i_o[j {i + 1}] - {(- η) * z_input[i] * მzⳆმz̃(z_output[j] z̃_output[j]) * ᐁ_i_o[j]}})

		; and update the bias
            	{M_i_o[j 0]  <-  M_i_o[j 0] - {(- η) * 1.0 * მzⳆმz̃(z_output[j] z̃_output[j]) * ᐁ_i_o[j]}}))
	





