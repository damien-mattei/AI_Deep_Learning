(import (Scheme+) (array))

(export multiply-matrix-matrix multiply-matrix-matrix-float
 multiply-matrix-matrix-double multiply-matrix-vector matrix matrix-float
 matrix-double matrix-f64 matrix-f32 matrix-v create-matrix-by-function
 create-matrix-float-by-function create-matrix-double-by-function
 create-matrix-f64-by-function create-matrix-f32-by-function dim-matrix
 matrix-ref matrix-set! matrix-f64-ref matrix-f64-set! matrix-f32-ref
 matrix-f32-set! matrix-line-ref matrix-line-set! matrix-f64-line-ref
 matrix-f64-line-set! matrix-f32-line-ref matrix-f32-line-set!
 vector->matrix-column matrix-column->vector *)

(import (only (kawa base) (* orig*)))

(define-overload-existing-operator * orig*)

(define-simple-class matrix () (v) ((*init* vParam) (set! v vParam))
 ((*init*) (values))
 ((display-matrix) (define that (this)) (display "Matrix:") (newline)
  (display (this)) (newline) (display that:v) (newline))
 ((dim) (dim-matrix (this))) ((apply f) (apply-matrix f (this))))

(define-simple-class matrix-float (matrix)
 ((*init* vParam) (invoke-special matrix (this) (quote *init*) vParam)))

(define-simple-class matrix-double (matrix)
 ((*init* vParam) (invoke-special matrix (this) (quote *init*) vParam)))

(define-simple-class matrix-f64 (matrix)
 ((*init* vParam) (invoke-special matrix (this) (quote *init*) vParam)))

(define-simple-class matrix-f32 (matrix)
 ((*init* vParam) (invoke-special matrix (this) (quote *init*) vParam)))

(define (matrix-scheme? M)
 (and (matrix? M) (not (matrix-float? M)) (not (matrix-double? M))
  (not (matrix-f64? M)) (not (matrix-f32? M))))

(define (create-matrix-by-function fct lin col)
 (matrix (create-vector-2d fct lin col)))

(define (create-matrix-float-by-function fct lin col)
 (matrix-float (create-vector-2d fct lin col)))

(define (create-matrix-double-by-function fct lin col)
 (matrix-double (create-vector-2d fct lin col)))

(define (create-matrix-f64-by-function fct lin col)
 (display "create-matrix-f64-by-function") (newline)
 (define m (create-f64vector-2d fct lin col)) (display "m created") (newline)
 (matrix-f64 m))

(define (create-matrix-f32-by-function fct lin col)
 (display "create-matrix-f32-by-function") (newline)
 (define m (create-f32vector-2d fct lin col)) (display "m created") (newline)
 (matrix-f32 m))

(define (dim-matrix M)
 (when (not (matrix? M)) (error "argument is not of type matrix"))
 (<+ v (matrix-v M)) (<+ lin (vector-length v))
 (<+ col (vector-length (bracket-apply v 0))) (values lin col))

(define (multiply-matrix-matrix M1 M2) (<+ (n1 p1) (dim-matrix M1))
 (<+ (n2 p2) (dim-matrix M2))
 (when (≠ p1 n2)
  (error "matrix.* : matrix product impossible, incompatible dimensions"))
 (<+ v1 (matrix-v M1)) (<+ v2 (matrix-v M2))
 (define (res i j) (<+ sum 0)
  (for ((<+ k 0) (< k p1) ($nfx$ k <- k + 1))
   ($nfx$ sum <- sum + (bracket-apply (bracket-apply v1 i) k) *
    (bracket-apply (bracket-apply v2 k) j)))
  sum)
 (<+ v (create-vector-2d res n1 p2)) (matrix v))

(define (multiply-matrix-matrix-float M1 M2) (<+ (n1 p1) (dim-matrix M1))
 (<+ (n2 p2) (dim-matrix M2))
 (when (≠ p1 n2)
  (error "matrix.* : matrix product impossible, incompatible dimensions"))
 (<+ v1 (matrix-v M1)) (<+ v2 (matrix-v M2))
 (define (res i j) (define sum :: float 0.0)
  (for ((<+ k 0) (< k p1) ($nfx$ k <- k + 1))
   ($nfx$ sum <- sum + (bracket-apply (bracket-apply v1 i) k) *
    (bracket-apply (bracket-apply v2 k) j)))
  sum)
 (<+ v (create-vector-2d res n1 p2)) (matrix-float v))

(define (multiply-matrix-matrix-double M1 M2) (<+ (n1 p1) (dim-matrix M1))
 (<+ (n2 p2) (dim-matrix M2))
 (when (≠ p1 n2)
  (error "matrix.* : matrix product impossible, incompatible dimensions"))
 (<+ v1 (matrix-v M1)) (<+ v2 (matrix-v M2))
 (define (res i j) (define sum :: double 0.0)
  (for ((<+ k 0) (< k p1) ($nfx$ k <- k + 1))
   ($nfx$ sum <- sum + (bracket-apply (bracket-apply v1 i) k) *
    (bracket-apply (bracket-apply v2 k) j)))
  sum)
 (<+ v (create-vector-2d res n1 p2)) (matrix-double v))

(define (multiply-matrix-matrix-f64 M1 M2) (<+ (n1 p1) (dim-matrix M1))
 (<+ (n2 p2) (dim-matrix M2))
 (when (≠ p1 n2)
  (error "matrix.* : matrix product impossible, incompatible dimensions"))
 (define (res i j) (define sum :: double 0.0)
  (for ((<+ k 0) (< k p1) ($nfx$ k <- k + 1))
   ($nfx$ sum <- sum + (bracket-apply (bracket-apply M1:v i) k) *
    (bracket-apply (bracket-apply M2:v k) j)))
  sum)
 (<+ v (create-f64vector-2d res n1 p2)) (matrix-f64 v))

(define (multiply-matrix-matrix-f32 M1 M2) (<+ (n1 p1) (dim-matrix M1))
 (<+ (n2 p2) (dim-matrix M2))
 (when (≠ p1 n2)
  (error "matrix.* : matrix product impossible, incompatible dimensions"))
 (<+ v1 (matrix-v M1)) (<+ v2 (matrix-v M2))
 (define (res i j) (define sum :: float 0.0)
  (for ((<+ k 0) (< k p1) ($nfx$ k <- k + 1))
   ($nfx$ sum <- sum + (bracket-apply (bracket-apply v1 i) k) *
    (bracket-apply (bracket-apply v2 k) j)))
  sum)
 (<+ v (create-f32vector-2d res n1 p2)) (matrix-f32 v))

(overload-existing-operator * multiply-matrix-matrix (matrix? matrix?))

(overload-existing-operator * multiply-matrix-matrix-float
 (matrix-float? matrix-float?))

(overload-existing-operator * multiply-matrix-matrix-double
 (matrix-double? matrix-double?))

(overload-existing-operator * multiply-matrix-matrix-f64
 (matrix-f64? matrix-f64?))

(overload-existing-operator * multiply-matrix-matrix-f32
 (matrix-f32? matrix-f32?))

(display "$ovrld-ht$=")

(display $ovrld-ht$)

(newline)

(define (matrix-v M) (slot-ref M (quote v)))

(define (vector->matrix-column v)
 (matrix (vector-map (lambda (x) (make-vector 1 x)) v)))

(define (vector->matrix-float-column v)
 (matrix-float (vector-map (lambda (x) (make-vector 1 (->float x))) v)))

(define (vector->matrix-double-column v)
 (matrix-double (vector-map (lambda (x) (make-vector 1 (->double x))) v)))

(define (vector->matrix-f64-column v)
 (matrix-f64 (vector-map (lambda (x) (make-f64vector 1 (->double x))) v)))

(define (vector->matrix-f32-column v)
 (matrix-f32 (vector-map (lambda (x) (make-f32vector 1 (->float x))) v)))

(define (matrix-column->vector Mc) (<+ v (matrix-v Mc))
 (vector-map (lambda (v2) (bracket-apply v2 0)) v))

(define (multiply-matrix-vector M v) (<+ Mc (vector->matrix-column v))
 (matrix-column->vector (* M Mc)))

(define (multiply-matrix-float-vector M v)
 (<+ Mc (vector->matrix-float-column v)) (matrix-column->vector (* M Mc)))

(define (multiply-matrix-double-vector M v)
 (<+ Mc (vector->matrix-double-column v)) (matrix-column->vector (* M Mc)))

(define (multiply-matrix-f64-vector M v) (<+ Mc (vector->matrix-f64-column v))
 (matrix-column->vector (* M Mc)))

(define (multiply-matrix-f32-vector M v) (<+ Mc (vector->matrix-f32-column v))
 (matrix-column->vector (* M Mc)))

(overload-existing-operator * multiply-matrix-vector (matrix? vector?))

(overload-existing-operator * multiply-matrix-float-vector
 (matrix-float? vector?))

(overload-existing-operator * multiply-matrix-double-vector
 (matrix-double? vector?))

(overload-existing-operator * multiply-matrix-f64-vector (matrix-f64? vector?))

(overload-existing-operator * multiply-matrix-f32-vector (matrix-f32? vector?))

(define (matrix-ref M lin col) (<+ v (matrix-v M))
 (bracket-apply (bracket-apply v lin) col))

(define (matrix-f64-ref M lin col) (<+ v (matrix-v M))
 ((bracket-apply v lin) col))

(define (matrix-f32-ref M lin col) (<+ v (matrix-v M))
 ((bracket-apply v lin) col))

(define (matrix-set! M lin col x) (<+ v (matrix-v M))
 (<- (bracket-apply (bracket-apply v lin) col) x))

(define (matrix-f64-set! M lin col x) (<+ v (matrix-v M))
 (f64vector-set! (bracket-apply v lin) col x))

(define (matrix-f32-set! M lin col x) (<+ v (matrix-v M))
 (f32vector-set! (bracket-apply v lin) col x))

(define (matrix-line-ref M lin) (<+ v (matrix-v M)) (bracket-apply v lin))

(define (matrix-f64-line-ref M lin) (<+ v (matrix-v M)) (bracket-apply v lin))

(define (matrix-f32-line-ref M lin) (<+ v (matrix-v M)) (bracket-apply v lin))

(define (matrix-line-set! M lin vect-line) (<+ v (matrix-v M))
 (<- (bracket-apply v lin) vect-line))

(define (matrix-f64-line-set! M lin vect-line) (<+ v (matrix-v M))
 (<- (bracket-apply v lin) vect-line))

(define (matrix-f32-line-set! M lin vect-line) (<+ v (matrix-v M))
 (<- (bracket-apply v lin) vect-line))

(overload-square-brackets matrix-ref matrix-set! (matrix? number? number?))

(overload-square-brackets matrix-line-ref matrix-line-set! (matrix? number?))

(overload-square-brackets matrix-f64-ref matrix-f64-set!
 (matrix-f64? number? number?))

(overload-square-brackets matrix-f64-line-ref matrix-f64-line-set!
 (matrix-f64? number?))

(overload-square-brackets matrix-f32-ref matrix-f32-set!
 (matrix-f32? number? number?))

(overload-square-brackets matrix-f32-line-ref matrix-f32-line-set!
 (matrix-f32? number?))

(define (apply-matrix f M) (vector-map (lambda (v) (vector-map f v)) M:v))

