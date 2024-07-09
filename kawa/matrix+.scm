;; matrix

;; typed version for Kawa/Java types

;; Kawa version


; make -f Makefile.Kawa all

; kawa curly-infix2prefix4kawa.scm  --srfi-105 ../AI_Deep_Learning/kawa/matrix+.scm | tr -d '|'  > ../AI_Deep_Learning/kawa/module_directory/matrix.scm

; kawa -Dkawa.import.path=".:/Users/mattei/Scheme-PLUS-for-Kawa:./kawa/module_directory"

;; use with Scheme+:
;; (require Scheme+)
;; (require array)
;;(require matrix)


;(module-name "matrix") ; change in R7RS fail

;(define-library (matrix) ; R7RS

(import (Scheme+)
        (array))

;; (require Scheme+)
;; (require array)


(export multiply-matrix-matrix
	multiply-matrix-matrix-float
	multiply-matrix-matrix-double
	
	multiply-matrix-vector
	
	matrix
	matrix-float
	matrix-double
	matrix-f64
	matrix-f32
	
	matrix-v
	
	create-matrix-by-function
	create-matrix-float-by-function
	create-matrix-double-by-function
	create-matrix-f64-by-function
	create-matrix-f32-by-function

	
	dim-matrix
	
	matrix-ref
	matrix-set!
	
	matrix-f64-ref
	matrix-f64-set!

	matrix-f32-ref
	matrix-f32-set!
	 
	matrix-line-ref
	matrix-line-set!

	matrix-f64-line-ref
	matrix-f64-line-set!

	matrix-f32-line-ref
	matrix-f32-line-set!
	
	vector->matrix-column
	matrix-column->vector

	* ; note that i export here the overloaded * operator

	;;display-matrix
	)

	
;; first stage overloading
;(define orig* *)
(import (only (kawa base) (* orig*)))

(define-overload-existing-operator * orig*)


;; (matrix #(1 2 3))
;; matrix@4612b856


(define-simple-class matrix ()

  (v); :: vector)

  ((*init* vParam); :: vector)
   ;(display "Constructor matrix") (newline)
   ;(display (this)) (newline)
   ;(define that (this)) ;  that avoid (this):v to be transformed by scheme+ parser in (this) :v which bugs  
   (set! v vParam)
   ;(display that:v) (newline)
   )

  ;; Need a default constructor as well.
  ((*init*) (values)); #!void bugs scheme+ parser: but (values) = #!void

  ((display-matrix)
   (define that (this)) ;  that avoid (this):v to be transformed by scheme+ parser in (this) :v which bugs
   (display "Matrix:") (newline)
   (display (this)) (newline)
   (display that:v) (newline))

  ((dim)
   (dim-matrix (this)))

  ((apply f)
   (apply-matrix f (this)))

  ) ; end class

(define-simple-class matrix-float (matrix)
  ;; A constructor which calls the superclass constructor.
  ((*init* vParam); :: vector)
   (invoke-special matrix (this) '*init* vParam))) ;  class inheritance

(define-simple-class matrix-double (matrix)
  ;; A constructor which calls the superclass constructor.
  ((*init* vParam); :: vector);(vParam :: vector))
   (invoke-special matrix (this) '*init* vParam))) ;  class inheritance


(define-simple-class matrix-f64 (matrix)
  ;; A constructor which calls the superclass constructor.
  ((*init* vParam) ;(vParam :: f64vector))
   (invoke-special matrix (this) '*init* vParam)
   ;;(display "Constructor matrix-f64") (newline)
   ) ;  class inheritance

  ;; ((display-matrix)
  ;;  (define that (this)) ;  that avoid (this):v to be transformed by scheme+ parser in (this) :v which bugs
  ;;  (display "Matrix-f64:") (newline)
  ;;  (display (this)) (newline)
  ;;  (display that:v) (newline))
  )

(define-simple-class matrix-f32 (matrix)
  ;; A constructor which calls the superclass constructor.
  ((*init* vParam) 
   (invoke-special matrix (this) '*init* vParam)
   ;;(display "Constructor matrix-f32") (newline)
   )) ;  class inheritance










;; (define (display-matrix M)
;;  (display "Matrix:") (display M) (newline)
;;  (display (matrix-v M)) (newline))



(define (matrix-scheme? M)
  {(matrix? M) and (not (matrix-float? M)) and (not (matrix-double? M)) and (not (matrix-f64? M)) and (not (matrix-f32? M))})


;; (define M (create-matrix-by-function (lambda (i j) (+ i j)) 2 3))
(define (create-matrix-by-function fct lin col)
  (matrix (create-vector-2d fct lin col)))


(define (create-matrix-float-by-function fct lin col)
  (matrix-float (create-vector-2d fct lin col)))

(define (create-matrix-double-by-function fct lin col)
  (matrix-double (create-vector-2d fct lin col)))

(define (create-matrix-f64-by-function fct lin col)
  (display "create-matrix-f64-by-function") (newline)
  (define m (create-f64vector-2d fct lin col))
  (display "m created") (newline)
  (matrix-f64 m))
					;(matrix-f64 (create-f64vector-2d fct lin col)))


(define (create-matrix-f32-by-function fct lin col)
  (display "create-matrix-f32-by-function") (newline)
  (define m (create-f32vector-2d fct lin col))
  (display "m created") (newline)
  (matrix-f32 m))
  ;(matrix-f32 (create-f32vector-2d fct lin col)))



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
;; (define M1*M2 (* M1 M2))
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

  ;(display "matrix+.scm : multiply-matrix-matrix") (newline)

  {(n1 p1) <+ (dim-matrix M1)}
  {(n2 p2) <+ (dim-matrix M2)}
  
  (when {p1 ≠ n2} (error "matrix.* : matrix product impossible, incompatible dimensions"))
  
  {v1 <+ (matrix-v M1)}
  {v2 <+ (matrix-v M2)}
  
  (define (res i j)
    {sum <+ 0}
    (for ({k <+ 0} {k < p1} {k <- k + 1})
	 {sum <- sum + v1[i][k] * v2[k][j]})
	 ;(display "sum=")(display sum) (newline)
    sum)

	
  {v <+ (create-vector-2d res n1 p2)}
  
  (matrix v))

;; (define M (matrix-float (create-vector-2d (lambda (i j) (* 1.0 (+ i j))) 2 3)))
;; #|kawa:6|# (matrix-float? M)
;; #t
;; #|kawa:7|# (matrix? M)
;; #t
;; #|kawa:8|# (matrix-v M)
;; #(#(0.0 1.0 2.0) #(1.0 2.0 3.0))
(define (multiply-matrix-matrix-float M1 M2)

  ;;(display "matrix+.scm : multiply-matrix-matrix-float") (newline)

  {(n1 p1) <+ (dim-matrix M1)}
  {(n2 p2) <+ (dim-matrix M2)}
  
  (when {p1 ≠ n2} (error "matrix.* : matrix product impossible, incompatible dimensions"))
  
  {v1 <+ (matrix-v M1)}
  {v2 <+ (matrix-v M2)}
  
  (define (res i j)
    (define sum :: float 0.0)
    (for ({k <+ 0} {k < p1} {k <- k + 1})
	 {sum <- sum + v1[i][k] * v2[k][j]})
	 ;(display "sum=")(display sum) (newline)
    sum)

	
  {v <+ (create-vector-2d res n1 p2)}
  
  (matrix-float v))


(define (multiply-matrix-matrix-double M1 M2)

  ;(display "matrix+.scm : multiply-matrix-matrix-double") (newline)

  {(n1 p1) <+ (dim-matrix M1)}
  {(n2 p2) <+ (dim-matrix M2)}
  
  (when {p1 ≠ n2} (error "matrix.* : matrix product impossible, incompatible dimensions"))
  
  {v1 <+ (matrix-v M1)}
  {v2 <+ (matrix-v M2)}
  
  (define (res i j)
    (define sum :: double 0.0)
    (for ({k <+ 0} {k < p1} {k <- k + 1})
	 {sum <- sum + v1[i][k] * v2[k][j]})
	 ;(display "sum=")(display sum) (newline)
    sum)

	
  {v <+ (create-vector-2d res n1 p2)}
  
  (matrix-double v))




(define (multiply-matrix-matrix-f64 M1 M2)

  ;(display "matrix+.scm : multiply-matrix-matrix-f64") (newline)

  {(n1 p1) <+ (dim-matrix M1)}
  {(n2 p2) <+ (dim-matrix M2)}
  
  (when {p1 ≠ n2} (error "matrix.* : matrix product impossible, incompatible dimensions"))
  
  ;;{v1 <+ (matrix-v M1)}
  ;;{v2 <+ (matrix-v M2)}
  ;; {v1 <+ M1:v}
  ;; {v2 <+ M2:v}
  
  (define (res i j)
    (define sum :: double 0.0)
    (for ({k <+ 0} {k < p1} {k <- k + 1})
	 {sum <- sum + M1:v[i][k] * M2:v[k][j]}) ; Kawa M:v syntax
	 ;(display "sum=")(display sum) (newline)
    sum)

	
  {v <+ (create-f64vector-2d res n1 p2)}
  
  (matrix-f64 v))


(define (multiply-matrix-matrix-f32 M1 M2)

  ;(display "matrix+.scm : multiply-matrix-matrix-f32") (newline)

  {(n1 p1) <+ (dim-matrix M1)}
  {(n2 p2) <+ (dim-matrix M2)}
  
  (when {p1 ≠ n2} (error "matrix.* : matrix product impossible, incompatible dimensions"))
  
  {v1 <+ (matrix-v M1)}
  {v2 <+ (matrix-v M2)}
  
  (define (res i j)
    (define sum :: float 0.0)
    (for ({k <+ 0} {k < p1} {k <- k + 1})
	 {sum <- sum + v1[i][k] * v2[k][j]})
	 ;(display "sum=")(display sum) (newline)
    sum)

	
  {v <+ (create-f32vector-2d res n1 p2)}
  
  (matrix-f32 v))


;; second stage overloading
(overload-existing-operator * multiply-matrix-matrix (matrix? matrix?))
(overload-existing-operator * multiply-matrix-matrix-float (matrix-float? matrix-float?))
;; note: the order is important as a matrix-float is also a matrix but the overloading stage store in a list the overloaded functions
;; so this must me stored with specific matrix first (float, double) and general matrix after.
;; other wise one should use specialized predicate (see: matrix-scheme?)
(overload-existing-operator * multiply-matrix-matrix-double (matrix-double? matrix-double?))
;; as lisp/scheme construct the lists by adding at head (not tail) the data the overloading order must be: matrix,float,double or matrix,double,float
(overload-existing-operator * multiply-matrix-matrix-f64 (matrix-f64? matrix-f64?))
(overload-existing-operator * multiply-matrix-matrix-f32 (matrix-f32? matrix-f32?))

(display "$ovrld-ht$=")
(display $ovrld-ht$)
(newline)


;; (matrix-v M)
;;#(#(0 1 2)
;;  #(1 2 3))
(define (matrix-v M)
  (slot-ref M 'v))



(define (vector->matrix-column v)
  (matrix (vector-map (lambda (x) (make-vector 1 x))
		      v)))




(define (vector->matrix-float-column v)
  (matrix-float (vector-map (lambda (x) (make-vector 1 (->float x)))
			    v)))

(define (vector->matrix-double-column v)
  (matrix-double (vector-map (lambda (x) (make-vector 1 (->double x)))
			     v)))


(define (vector->matrix-f64-column v)
  (matrix-f64 (vector-map (lambda (x) (make-f64vector 1 (->double x)))
			     v)))


(define (vector->matrix-f32-column v)
  (matrix-f32 (vector-map (lambda (x) (make-f32vector 1 (->float x)))
			     v)))


(define (matrix-column->vector Mc)
  {v <+ (matrix-v Mc)}
  (vector-map (lambda (v2) {v2[0]})
	      v))


(define (multiply-matrix-vector M v) ;; args: matrix ,vector ;  return vector
  {Mc <+ (vector->matrix-column v)}
  ;;(matrix-column->vector (multiply-matrix-matrix M Mc)))
  (matrix-column->vector {M * Mc}))

(define (multiply-matrix-float-vector M v) ;; args: matrix ,vector ;  return vector
  {Mc <+ (vector->matrix-float-column v)}
  (matrix-column->vector {M * Mc}))

(define (multiply-matrix-double-vector M v) ;; args: matrix ,vector ;  return vector
  {Mc <+ (vector->matrix-double-column v)}
  (matrix-column->vector {M * Mc}))


(define (multiply-matrix-f64-vector M v) ;; args: matrix ,vector ;  return vector
  {Mc <+ (vector->matrix-f64-column v)}
  (matrix-column->vector {M * Mc}))


(define (multiply-matrix-f32-vector M v) ;; args: matrix ,vector ;  return vector
  {Mc <+ (vector->matrix-f32-column v)}
  (matrix-column->vector {M * Mc}))


(overload-existing-operator * multiply-matrix-vector (matrix? vector?))
(overload-existing-operator * multiply-matrix-float-vector (matrix-float? vector?))
(overload-existing-operator * multiply-matrix-double-vector (matrix-double? vector?))
(overload-existing-operator * multiply-matrix-f64-vector (matrix-f64? vector?))
(overload-existing-operator * multiply-matrix-f32-vector (matrix-f32? vector?))

;; define getter,setter
;; (matrix-ref M 1 1)
;; 5
(define (matrix-ref M lin col)
  {v <+ (matrix-v M)}
  {v[lin][col]})

(define (matrix-f64-ref M lin col)
  {v <+ (matrix-v M)}
  ({v[lin]} col))


(define (matrix-f32-ref M lin col)
  {v <+ (matrix-v M)}
  ({v[lin]} col))


;; (matrix-set! M 0 1 -7)
;; -7
;; #|kawa:63|# (matrix-v M)
;; #(#(0 -7 2) #(4 5 6))
(define (matrix-set! M lin col x)
  {v <+ (matrix-v M)}
  {v[lin][col] <- x})


(define (matrix-f64-set! M lin col x)
  {v <+ (matrix-v M)}
  (f64vector-set! {v[lin]} col x))


(define (matrix-f32-set! M lin col x)
  {v <+ (matrix-v M)}
  (f32vector-set! {v[lin]} col x))


;; (matrix-line-ref M 1)
;; #(1 2 3)
(define (matrix-line-ref M lin)
  {v <+ (matrix-v M)}
  {v[lin]})

(define (matrix-f64-line-ref M lin)
  {v <+ (matrix-v M)}
  {v[lin]})

(define (matrix-f32-line-ref M lin)
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


(define (matrix-f64-line-set! M lin vect-line)
  {v <+ (matrix-v M)}
  {v[lin] <- vect-line})


(define (matrix-f32-line-set! M lin vect-line)
  {v <+ (matrix-v M)}
  {v[lin] <- vect-line})


;; overload [ ] 
(overload-square-brackets matrix-ref matrix-set!  (matrix? number? number?))
(overload-square-brackets matrix-line-ref matrix-line-set! (matrix? number?))

(overload-square-brackets matrix-f64-ref matrix-f64-set!  (matrix-f64? number? number?))
(overload-square-brackets matrix-f64-line-ref matrix-f64-line-set! (matrix-f64? number?))

(overload-square-brackets matrix-f32-ref matrix-f32-set!  (matrix-f32? number? number?))
(overload-square-brackets matrix-f32-line-ref matrix-f32-line-set! (matrix-f32? number?))


; apply a function to each elements of the matrix
(define (apply-matrix f M)
  (vector-map (lambda (v) (vector-map f v))
	      M:v))


;) ; end module


