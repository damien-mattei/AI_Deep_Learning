;; matrix

;; Kawa version

;; kawa curly-infix2prefix.scm ../AI_Deep_Learning/kawa/matrix+.scm > ../AI_Deep_Learning/kawa/matrix.scm

;; use with Scheme+: (require matrix)

;;(module-name "matrix")

;; (export multiply-matrix-matrix
;; 	multiply-matrix-vector
;; 	matrix
;; 	matrix-v
;; 	create-matrix-by-function
;; 	dim-matrix
;; 	matrix-ref
;; 	matrix-set!
;; 	matrix-line-ref
;; 	matrix-line-set!
;; 	vector->matrix-column
;; 	matrix-column->vector)

(require 'srfi-1) ; any,every
(require 'srfi-69) ;; hash table

;; try include , use include-relative if problems
;; (include "../../Scheme-PLUS-for-Kawa/rec.scm") ; rec does  not exist in Kawa (no SRFI 31)
;; (include "../../Scheme-PLUS-for-Kawa/def.scm")
;; (include "../../Scheme-PLUS-for-Kawa/set-values-plus.scm")
;; (include "../../Scheme-PLUS-for-Kawa/for_next_step.scm")
;; (include "../../Scheme-PLUS-for-Kawa/declare.scm")
;; (include "../../Scheme-PLUS-for-Kawa/condx.scm")
;; (include "../../Scheme-PLUS-for-Kawa/block.scm")
;; (include "../../Scheme-PLUS-for-Kawa/not-equal.scm")
;; (include "../../Scheme-PLUS-for-Kawa/exponential.scm")
;; (include "../../Scheme-PLUS-for-Kawa/while-do-when-unless.scm")
;; (include "../../Scheme-PLUS-for-Kawa/repeat-until.scm")
;; (include "../../Scheme-PLUS-for-Kawa/modulo.scm")
;; (include "../../Scheme-PLUS-for-Kawa/bitwise.scm")


;; (include "../../Scheme-PLUS-for-Kawa/slice.scm")

;; (include "../../Scheme-PLUS-for-Kawa/scheme-infix.scm")

;; (include "../../Scheme-PLUS-for-Kawa/assignment.scm")
;; (include "../../Scheme-PLUS-for-Kawa/apply-square-brackets.scm")

;; (include "../../Scheme-PLUS-for-Kawa/array.scm")


;; (matrix #(1 2 3))
;; matrix@4612b856


(define-simple-class matrix ()

  (v :: vector)

  ((*init* (vParam :: vector)) 
   (set! v vParam))

  )


;; (define M (create-matrix-by-function (lambda (i j) (+ i j)) 2 3))
(define (create-matrix-by-function fct lin col)
  (matrix (create-vector-2d fct lin col)))


;; return the line and column values of dimension
;; (dim-matrix M)
;; 2 3
(define (dim-matrix M)

  (when (not (matrix? M))
	(error "argument is not of type matrix"))
  
  {v <+ (matrix-v M)}
  {lin <+ (vector-length v)}
  {col <+ (vector-length {v[0]})}
  (values lin col))




;; #|kawa:85|# (define M1 (create-matrix-by-function (lambda (i j) (+ i j)) 2 3))
;; #|kawa:86|# (define M2 (create-matrix-by-function (lambda (i j) (+ i j)) 3 2))
;; #|kawa:87|# (multiply-matrix-matrix M1 M2)
;; matrix@3fc1abf
;; #|kawa:88|# (define M1*M2 (multiply-matrix-matrix M1 M2))
;; #|kawa:89|# M1*M2
;; matrix@3bf5911d
;; #|kawa:90|# (matrix-v M1*M2)
;; #(#(5 8) #(8 14))
;; #|kawa:91|# (matrix-v M1)
;; #(#(0 1 2) #(1 2 3))
;; #|kawa:92|# (matrix-v M2)
;; #(#(0 1) #(1 2) #(2 3))
(define (multiply-matrix-matrix M1 M2)

  {(n1 p1) <+ (dim-matrix M1)}
  {(n2 p2) <+ (dim-matrix M2)}
  
  (when {p1 ≠ n2} (error "matrix.* : matrix product impossible, incompatible dimensions"))
  
  {v1 <+ (matrix-v M1)}
  {v2 <+ (matrix-v M2)}
  
  (define (res i j)
    {sum <+ 0}
    (for ({k <+ 0} {k < p1} {k <- k + 1})
	 {sum <- sum + v1[i][k] * v2[k][j]}))
	
  {v <+ (create-vector-2d res n1 p2)}
  
  (matrix v))



;; (matrix-v M)
;;#(#(0 1 2)
;;  #(1 2 3))
(define (matrix-v M)
  (slot-ref M 'v))



(define (vector->matrix-column v)
  (matrix (vector-map (lambda (x) (make-vector 1 x))
		      v)))

(define (matrix-column->vector Mc)
  {v <+ (matrix-v Mc)}
  (vector-map (lambda (v2) {v2[0]})
	      v))


(define (multiply-matrix-vector M v) ;; args: matrix ,vector ;  return vector
  {Mc <+ (vector->matrix-column v)}
  ;;(matrix-column->vector (multiply-matrix-matrix M Mc)))
  (matrix-column->vector {M * Mc}))


;; define getter,setter
;; (matrix-ref M 1 1)
;; 5
(define (matrix-ref M lin col)
  {v <+ (matrix-v M)}
  {v[lin][col]})


;; (matrix-set! M 0 1 -7)
;; -7
;; #|kawa:63|# (matrix-v M)
;; #(#(0 -7 2) #(4 5 6))
(define (matrix-set! M lin col x)
  {v <+ (matrix-v M)}
  {v[lin][col] <- x})



;; (matrix-line-ref M 1)
;; #(1 2 3)
(define (matrix-line-ref M lin)
  {v <+ (matrix-v M)}
  {v[lin]})


;; (matrix-v M)
;; #(#(0 1 2) #(1 2 3))
;; #|kawa:51|# (matrix-line-set! M 1 #(4 5 6))
;; #(4 5 6)
;; #|kawa:52|# (matrix-v M)
;; #(#(0 1 2) #(4 5 6))
(define (matrix-line-set! M lin vect-line)
  {v <+ (matrix-v M)}
  {v[lin] <- vect-line})



;; overload [ ] 
(overload-square-brackets matrix-ref matrix-set!  (matrix? number? number?))
(overload-square-brackets matrix-line-ref matrix-line-set! (matrix? number?))


