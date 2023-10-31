
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


;; overload [ ] 
(overload-square-brackets matrix-ref matrix-set!  (matrix? number? number?))
(overload-square-brackets matrix-line-ref matrix-line-set! (matrix? number?))

(for ({i <+ 0} {i < 3} {i <- i + 1})
     (display i) (newline))


