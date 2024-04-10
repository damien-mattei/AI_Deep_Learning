
; Deep Learning : back propagation, gradient descent, neural network with N hidden layers

; L'algorithme de rétro-propagation du gradient dans un
; réseau de neurones avec N couches cachées.

;  D. Mattei	


; use MacVim to show ALL the characters of this file (not Emacs, not Aquamacs)
; jeu de couleurs: Torte ou Koehler


; kawa curly-infix2prefix4kawa.scm  --kawa ../AI_Deep_Learning/exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa+.scm | tr -d '|' > ../AI_Deep_Learning/exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa.scm

; or:

;kawa ../curly-infix2prefix4kawa.scm  --kawa exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa+.scm | tr -d '|' > exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa.scm
; 

; kawa -Dkawa.import.path=".:/Users/mattei/Scheme-PLUS-for-Kawa:./kawa"

; (load "exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa.scm")




(require matrix)

(require Scheme+)

(require array)


;; first stage overloading
(import (only (scheme base) (+ orig+))) ; (* orig*)))

;(define orig+ +)
;(define orig* *)

;(define-overload-existing-operator * orig*)
(define-overload-existing-operator + orig+)

(define-overload-procedure random)


(define (random-int n)
  (integer {n * (random)})) 


; second stage overloading
;(overload-existing-operator * multiply-matrix-matrix (matrix? matrix?))
;(overload-existing-operator * multiply-matrix-vector (matrix? vector?))

;(define * (make-procedure method: (lambda (x ::number y ::number) (orig* x y))
;			  method: (lambda (x ::matrix y ::matrix) (multiply-matrix-matrix  x y))
;			  method: (lambda (x ::matrix y ::vector) (multiply-matrix-vector  x y))
;			  method: (lambda lyst (apply orig* lyst))))


;(insert-operator! orig* *)


(overload-existing-operator + vector-append (vector? vector?))

;(define + (make-procedure method: (lambda (x ::number y ::number) (orig+ x y))
;			  method: (lambda (x ::vector y ::vector) (vector-append x y))
;			  method: (lambda lyst (apply orig+ lyst))))


(insert-operator! orig+ +)


(overload-procedure random java.lang.Math:random ())
(overload-procedure random random-int (integer?))


(define (uniform-dummy dummy1 dummy2) {-1 + (random) * 2})

; return a random number between [inf, sup]
(define (uniform-interval inf sup)
  {gap <+ sup - inf}
  {inf + gap * (random)})


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


; modify coefficients layer
(define (modification_des_poids M_i_o η z_input z_output z̃_output ᐁ_i_o მzⳆმz̃) ; derivative of activation function of the layer
	 
	  ; the length of output and input layer with coeff. used for bias update
	  {(len_layer_output len_layer_input_plus1forBias) <+ (dim-matrix M_i_o)} ; use values and define-values to create bindings
        
	  {len_layer_input <+ len_layer_input_plus1forBias - 1}

	  (for-each-in (j (in-range len_layer_output)) ; line
		(for-each-in (i (in-range len_layer_input)) ; column , parcours les colonnes de la ligne sauf le bias
		    {M_i_o[j {i + 1}]  <-  M_i_o[j {i + 1}] - (- η) * z_input[i] * მzⳆმz̃(z_output[j] z̃_output[j]) * ᐁ_i_o[j]})

		; and update the bias
            	{M_i_o[j 0]  <-  M_i_o[j 0] - (- η) * 1.0 * მzⳆმz̃(z_output[j] z̃_output[j]) * ᐁ_i_o[j]}))
	



;; (define net (ReseauRetroPropagation #(2 3 1) 250000 10 σ σ der_σ der_σ))

(define-simple-class ReseauRetroPropagation ()  ; network back propagation
  
  (nbiter init-value: 3)
  (activation_function_hidden_layer)
  (activation_function_output_layer)
  (activation_function_hidden_layer_derivative)
  (activation_function_output_layer_derivative)

  (ηₛ 1.0)

  (z)
  (z̃)

  (M)

  (ᐁ)

  (eror 0)

  ((*init* nc nbiter0 ηₛ0 activation_function_hidden_layer0
	   	          activation_function_output_layer0
			  activation_function_hidden_layer_derivative0
			  activation_function_output_layer_derivative0)
   
   (display "*init* : nc=") (display nc) (newline)
   {nbiter <- nbiter0}
   {ηₛ <- ηₛ0}
   {activation_function_hidden_layer <- activation_function_hidden_layer0}
   {activation_function_output_layer <- activation_function_output_layer0}
   {activation_function_hidden_layer_derivative <- activation_function_hidden_layer_derivative0}
   {activation_function_output_layer_derivative <- activation_function_output_layer_derivative0}

   {lnc <+ (vector-length nc)}

   (define (make-vector-z lg) (make-vector lg 0))

   {z <- (vector-map make-vector-z nc)}
   (display "z=") (display z) (newline)

   ; z̃[0] is not used as z[0] is x, the initial data
	 
   {z̃ <- (vector-map make-vector-z nc)}

   (display "z̃=") (display z̃) (newline)

   {M <- (vector-map (lambda (n) create-matrix-by-function(uniform-dummy nc[n + 1] {nc[n] + 1})) ;; Matrix-vect
 		     [0 <: (- lnc 1)])}	; in Kawa special syntax we can not use infix expression

   (display "M=") (display M) (newline)

   {ᐁ <- (vector-map make-vector-z nc)}

   (display "ᐁ=") (display ᐁ) (newline)

   (display "nbiter=") (display nbiter) (newline)

  ) ;; end *init*

  ; forward propagation
    
  ; z_* sans le coef. 1 constant pour le bias



((accepte_et_propage x) ; on entre des entrées et on les propage
  
  (when {vector-length(x) ≠ vector-length(z[0])} 
	(display "Mauvais nombre d'entrées !") (newline)
	(exit #f))

  {z[0] <- x} ; on ne touche pas au biais

  ;; propagation des entrées vers la sortie

  {n <+ vector-length(z)}
  ;;(display "n=") (display n) (newline)

  ;; hidden layers
  (declare z_1)

  (declare i) ; because the variable will be used outside the 'for' loop too
  
  (for ({i <- 0} {i < n - 2} {i <- i + 1}) ; personnal 'for' definition as in Javascript,C,C++,Java

       ;; calcul des stimuli reçus par la couche cachée d'indice i+1 à-partir de la précedente

       ;; create an array with 1 in front for the bias coefficient
       
       {z_1 <- #(1) + z[i]} ; + operator has been overloaded to append scheme vectors

       ;;(display "z_1 = ") (display z_1) (newline)

       {z̃[i + 1] <- M[i] * z_1} ; z̃ = matrix * vector , return a vector

       ;;(display "z̃[i + 1] = ") (display {z̃[i + 1]}) (newline)

       #| calcul des réponses des neurones cachés
       
       i also use Neoteric Expression :https://sourceforge.net/p/readable/wiki/Rationale-neoteric/
       example: {map(sin '(0.2 0.7 0.3))}
       '(0.19866933079506122 0.644217687237691 0.29552020666133955)
       
       i also use Neoteric Expression to easily port Python code to Scheme+
       
       the original Python code was:
       z[i+1] = list(map(self.activation_function_hidden_layer,z̃[i+1]))
       the Scheme+ port is below: |#
       
       {z[i + 1] <- vector-map(activation_function_hidden_layer z̃[i + 1])}

       ;;(display "z[i + 1] = ") (display {z[i + 1]}) (newline)

       ) ; end for


  ;; output layer
  
  ;; calcul des stimuli reçus par la couche cachée d'indice i+1 à-partir de la précedente

  ;; create a list with 1 in front for the bias coefficient
  {z_1 <- #(1) + z[i]}

  {z̃[i + 1] <- M[i] * z_1} ; z̃ = matrix * vector , return a vector

  ;; calcul des réponses des neurones de la couche de sortie
  {z[i + 1] <- vector-map(activation_function_output_layer z̃[i + 1])}
  ;;(display "z[i + 1] = ") (display {z[i + 1]}) (newline)
  
  ) ; end method



((apprentissage Lexemples) ; apprentissage des poids par une liste d'exemples
  
  {ip <+ 0} ; numéro de l'exemple courant

  (declare x y)
  (for-each-in (it (in-range nbiter)) ; le nombre d'itérations est fixé !
		 
	       	 (when {it % 1000 = 0}
		       (display it)(newline))

		 ;;(display it)(newline)
		 
		 ;{err <+ 0.0} ; l'erreur totale pour cet exemple

		 {(x y) <- Lexemples[ip]}         ; un nouvel exemple à apprendre

		 ;; PROPAGATION VERS L'AVANT
		 (accepte_et_propage x)       ; sorties obtenues sur l'exemple courant, self.z_k et z_j sont mis à jour

		 ;; RETRO_PROPAGATION VERS L'ARRIERE, EN DEUX TEMPS

		 {i <+ i_output_layer <+ vector-length(z) - 1} ; start at index i of the ouput layer

		 {ns <+ vector-length(z[i])}
		 

		 ;; TEMPS 1. calcul des gradients locaux sur la couche k de sortie (les erreurs commises)
		 (for-each-in (k (in-range ns))
				{ᐁ[i][k] <- y[k] - z[i][k]})     ; gradient sur un neurone de sortie (erreur locale)
				;{err <- err + ᐁ[i][k] ** 2})    ; l'erreur quadratique totale

		 ;{err <- err * 0.5}

		 ;(when {it = nbiter - 1}
		 ;      {eror <- err})               ; mémorisation de l'erreur totale à la dernière itération


		 ;; modification des poids de la matrice de transition de la derniére couche de neurones cachés à la couche de sortie

		 {მzⳆმz̃ <+ activation_function_output_layer_derivative}

		 {modification_des_poids(M[i - 1] ηₛ z[i - 1] z[i] z̃[i] ᐁ[i] მzⳆმz̃)}

		 ;; TEMPS 2. calcul des gradients locaux sur les couches cachées (rétro-propagation), sauf pour le bias constant

		 {მzⳆმz̃ <- activation_function_hidden_layer_derivative}

		 (for-each-in (i (reversed (in-range 1 i_output_layer)))
				{nc <+ vector-length(z[i])}
				{ns <+ vector-length(z[i + 1])}
				(for-each-in (j (in-range nc))
					{ᐁ[i][j] <- ($+>
							{sum <+ 0}  
							(for-each-in (k (in-range ns))
							     {sum <- sum + მzⳆმz̃(z[i + 1][k] z̃[i + 1][k]) * M[i][k {j + 1}] * ᐁ[i + 1][k]})
							sum)})
				;; modification des poids de la matrice de transition de la couche i-1 à i
				{modification_des_poids(M[i - 1] ηₛ  z[i - 1] z[i] z̃[i] ᐁ[i] მzⳆმz̃)})

		 ;; et l'on passe à l'exemple suivant
		 
		 {ip <- random(vector-length(Lexemples))}

	       ) ; end for it
  ) ; end define/public

  



  ((test Lexemples)

          (display "Test des exemples :") (newline)
          {err <+ 0}

	  (declare entree sortie_attendue ᐁ)
	  (for-each-in (entree-sortie_attendue Lexemples)
		{(entree sortie_attendue) <- entree-sortie_attendue} ; use pairs in Scheme instead of tuples and vectors in Python
		(accepte_et_propage entree)
		(format #t "~a --> ~a : on attendait ~a~%" entree {z[vector-length(z) - 1]} sortie_attendue)  ; ~% is(newline)
		{ᐁ <- sortie_attendue[0] - z[vector-length(z) - 1][0]} ; erreur sur un element
		{err <- err + ᐁ ** 2}) ; l'erreur quadratique totale
		
	  {err <- err * 0.5}
	  (display "Error on examples=") (display err) (newline))

) ; end class	



;; ################## NOT ##################
;; *init* : nc=#(1 2 1)
;; z=#(#(0) #(0 0) #(0))
;; z̃=#(#(0) #(0 0) #(0))
;; M=#(matrix@2f6bcf87 matrix@58f174d9)
;; ᐁ=#(#(0) #(0 0) #(0))
;; nbiter=5000
;; exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa.scm:138:2: warning - no known slot 'apprentissage' in java.lang.Object
;; 0
;; 1000
;; 2000
;; 3000
;; 4000
;; exo_retropropagationNhidden_layers_matrix_v2_by_vectors4kawa.scm:139:2: warning - no known slot 'test' in java.lang.Object
;; Test des exemples :
;; #(1) --> #(0.006583904270400075) : on attendait #(0)
;; #(0) --> #(0.9926139128833222) : on attendait #(1)
;; Error on examples=1.1963304682059438E-4


(display "################## NOT ##################")
(newline)

{r1 <+ (ReseauRetroPropagation  #(1 2 1) 5000 10 σ σ der_σ der_σ)}

{Lexemples1 <+ #((#(1) . #(0)) (#(0) . #(1)))}  ; use pairs in Scheme instead of vectors in Python

(r1:apprentissage Lexemples1)

(r1:test Lexemples1)

(newline)


;; ################## XOR ##################
;; *init* : nc=#(2 3 1)
;; z=#(#(0 0) #(0 0 0) #(0))
;; z̃=#(#(0 0) #(0 0 0) #(0))
;; M=#(matrix@23cbe174 matrix@2018ac47)
;; ᐁ=#(#(0 0) #(0 0 0) #(0))
;; nbiter=250000
;; ...
;; Test des exemples :
;; #(1 0) --> #(0.9982626600542213) : on attendait #(1)
;; #(0 0) --> #(4.7118633474893784E-4) : on attendait #(0)
;; #(0 1) --> #(0.9982856989092453) : on attendait #(1)
;; #(1 1) --> #(0.0021302684113227318) : on attendait #(0)
;; Error on examples=1.082825413028618E-5



(display "################## XOR ##################")
(newline)

{r2 <+ (ReseauRetroPropagation  #(2 8 1) 250000 0.1 σ σ der_σ der_σ)} ; 3' 22"

{Lexemples2 <+ #( (#(1 0) . #(1))  (#(0 0) . #(0))  (#(0 1) . #(1))  (#(1 1) . #(0)))}  ; use pairs in Scheme instead of vectors in Python

(r2:apprentissage Lexemples2)

(r2:test Lexemples2)

(newline)



(display "################## SINE ##################")
(newline)

{r3 <+ (ReseauRetroPropagation #(1 70 70 1) 50000 0.01 atan tanh der_atan der_tanh)}

(declare pi)
{pi <- 4 * atan(1)}
;(display pi)
;(newline)
{Llearning <+ (vector-map (lambda (x) (cons (vector x) (vector (sin x))))  ; use pairs in Scheme instead of vectors in Python
			  (list->vector (map (lambda (n) (uniform-interval (- pi) pi))
					(in-range 10000))))}

;(display "Llearning=")  (display Llearning) (newline)

{Ltest <+ (vector-map (lambda (x) (cons (vector x) (vector (sin x))))  ; use pairs in Scheme instead of vectors in Python
		      (list->vector (map (lambda (n) (uniform-interval {(- pi) / 2} {pi / 2}))
					 (in-range 10000))))}




(r3:apprentissage Llearning)

(r3:test Ltest)

(newline)


