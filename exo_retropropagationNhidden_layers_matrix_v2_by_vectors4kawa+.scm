
; Deep Learning : back propagation, gradient descent, neural network with N hidden layers

; L'algorithme de rétro-propagation du gradient dans un
; réseau de neurones avec N couches cachées.

;  D. Mattei	


; use MacVim to show ALL the characters of this file (not Emacs, not Aquamacs)
; jeu de couleurs: Torte ou Koehler

; kawa -Dkawa.import.path=".:/Users/mattei/Dropbox/git/Scheme-PLUS-for-Kawa:./kawa"
 

; use: (load "exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa+.scm")

;;(include "../Scheme-PLUS-for-Kawa/Scheme+.scm")

(require 'srfi-1) ; any,every
(require 'srfi-69) ;; hash table

;;(require matrix)

(require overload)


;; try include , use include-relative if problems
(include "../Scheme-PLUS-for-Kawa/rec.scm") ; rec does  not exist in Kawa (no SRFI 31)
(include "../Scheme-PLUS-for-Kawa/def.scm")
(include "../Scheme-PLUS-for-Kawa/set-values-plus.scm")
(include "../Scheme-PLUS-for-Kawa/for_next_step.scm")
(include "../Scheme-PLUS-for-Kawa/declare.scm")
(include "../Scheme-PLUS-for-Kawa/condx.scm")
(include "../Scheme-PLUS-for-Kawa/block.scm")
(include "../Scheme-PLUS-for-Kawa/not-equal.scm")
(include "../Scheme-PLUS-for-Kawa/exponential.scm")
(include "../Scheme-PLUS-for-Kawa/while-do-when-unless.scm")
(include "../Scheme-PLUS-for-Kawa/repeat-until.scm")
(include "../Scheme-PLUS-for-Kawa/modulo.scm")
(include "../Scheme-PLUS-for-Kawa/bitwise.scm")


(include "../Scheme-PLUS-for-Kawa/slice.scm")




; first stage overloading
;(define-overload-existing-operator +)
;(define-overload-existing-operator *)



; second stage overloading
(overload-existing-operator + vector-append (vector? vector?))

(include "../Scheme-PLUS-for-Kawa/scheme-infix.scm")

(include "../Scheme-PLUS-for-Kawa/assignment.scm")
(include "../Scheme-PLUS-for-Kawa/apply-square-brackets.scm")


(include "../Scheme-PLUS-for-Kawa/array.scm")

(include "kawa/matrix.scm")

(overload-existing-operator * multiply-matrix-matrix (matrix? matrix?))
(overload-existing-operator * multiply-matrix-vector (matrix? vector?))




(define M1 (create-matrix-by-function (lambda (i j) (+ i j)) 2 3))
(define M2 (create-matrix-by-function (lambda (i j) (+ i j)) 3 2))
(define M1*M2 (* M1 M2))
;(define M1*M2 (multiply-matrix-matrix M1 M2))
(define vr (matrix-v M1*M2))
(display vr)




