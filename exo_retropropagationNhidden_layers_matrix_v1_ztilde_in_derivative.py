# L'algorithme de rétro-propagation du gradient dans un
# réseau de neurones avec N couches cachées.

#  D. Mattei

# python3.7 -O exo_retropropagationNhidden_layers_matrix_ztilde_in_derivative.py

# use MacVim to show ALL the characters of this file (not Emacs, not Aquamacs)

from random import seed, uniform,randint
seed(1789)     # si vous voulez avoir les mêmes tirages aléatoires à chaque exécution du fichier !
from math import exp, pow, pi, sin , tanh , atan
from Matrix import Matrix
from time import time


# sigmoïde
def σ(z̃):
    try:
        s = 1/(1+ exp(-z̃))
    except OverflowError as e:
        # Somehow no exception is caught here...
        #print('OverflowError...')
        #print("x=",x)
        #sys.exit(1)
        s = 0
    except Exception as e:
        print(e)
    
    return s


# not used
def σࠤ(z):
    return σ(z)*(1-σ(z))

# not used
def tanhࠤ(x):
    return 1 - tanh(x)**2

def der_tanh(z,z̃):
    return 1 - z**2

def der_σ(z,z̃):
    return z*(1-z)


def leaky_RELU(z̃):
    return max(0.01*z̃,z̃)

def der_leaky_RELU(z,z̃):
    return 1 if z>=0 else 0.01

def RELU(z̃):
    return max(0,z̃)

def der_RELU(z,z̃):
    return 1 if z>=0 else 0

def swish(z̃):
    return z̃ * σ(z̃)

def der_swish(z,z̃):
    return z + σ(z̃) * (1 - z)

def der_atan(z,z̃):
    return 1 / (1 + pow(z̃,2))



class ReseauRetroPropagation():
    
    def __init__(self,nc=[2,3,1], nbiter=3, ηₛ=1.0 , ηₑ=0.0001 ,
                 activation_function_hidden_layer=tanh,
                 activation_function_output_layer=tanh,
                 activation_function_hidden_layer_derivative=der_tanh,
                 activation_function_output_layer_derivative=der_tanh):
        
        '''Construit un réseau de neurones avec plusieurs couches cachées. Il y a des entrées (+ biais),
        des neurones dans les couches cachées (+ biais) et des neurones en sortie dont les nombres sont définies dans nc.'''

        lnc = len(nc) # the total of all layer including input, output and hidden layers
        
        # on crée le tableau des couches du réseau
        # >>> nc=[2,3,1]
        # >>> [ [0] * n for n in nc ]
        # [[0, 0], [0, 0, 0], [0]]      
        self.z = [ [0] * n for n in nc ] # les entrées concrètes seront fournies avec la méthode accepte
        self.z̃ = [ [0] * n for n in nc ] # z̃[0] is not used as z[0] is x, the initial data 
               
        # nc[n] + 1 in the matrix size because we add one column of bias in the matrix for each hidden neuron of the hidden layer "c"

        # >>> M = [ Matrix( lambda j,i: uniform(-1,1) , nc[n+1] , nc[n] + 1 ) for n in range(len(nc)-1) ]
        # # Matrix constructor Matrix (function,Numeric,Numeric) #
        # # Matrix constructor Matrix (function,Numeric,Numeric) #
        # [[[ 0.30891168 -0.06324858 -0.77054504]
        #  [ 0.56467559  0.4090438  -0.6001032 ]
        #  [ 0.04621124 -0.4736403   0.52908111]]
        # Matrix @ 0x7f14c2dfa090 
        # , [[-0.17710618 -0.32836366 -0.81737218  0.65399886]]
        # Matrix @ 0x7f14c2d9bad0 
        # ]
  
        # >>> print(M[0])
        # [[ 0.0865122   0.48109634 -0.88726825]
        #  [-0.62196803 -0.02562076 -0.12770346]
        #  [-0.19076204 -0.38836422 -0.91260862]]
        
        # use with M[0][1][2]  notation
        #M[i][j][k] == poids k->j from layer i to layer i+1
        self.M = [ Matrix( lambda j,i: uniform(-1,1) , nc[n+1] , nc[n] + 1 )   for n in range(lnc - 1) ]
        # for n in range(lnc - 1):
        #     print("self.M[",n,"]=");print(self.M[n])

        # >>> ᐁ = [ [0] * n for n in nc ]
        # >>> ᐁ
        # [[0, 0], [0, 0, 0], [0]]
        self.ᐁ = [ [0] * n for n in nc ]    # gradients locaux des neurones cachés et gradient sur la couche de sortie
        # ᐁ[0] is useless but keep same index with z
        
        self.nbiter = nbiter

        # "learning rate" 
        self.ηₛ = ηₛ
        self.ηₑ = ηₑ
        self.error = 0

        self.activation_function_hidden_layer = activation_function_hidden_layer
        self.activation_function_output_layer = activation_function_output_layer
        self.activation_function_hidden_layer_derivative = activation_function_hidden_layer_derivative
        self.activation_function_output_layer_derivative = activation_function_output_layer_derivative


    # forward propagation
    
    # fusionne accept et propage
    # z_* sans le coef. 1 constant pour le bias
    def accepte_et_propage(self,x):         # on entre des entrées et on les propage

        # note: i just reference the variables for code readness (hide all the self keyword)
        z = self.z
        z̃ = self.z̃ 
        M = self.M
        
        if len(x) != len(z[0]):
            raise ValueError("Mauvais nombre d'entrées !")
        
        z[0] = x       # on ne touche pas au biais
        self.z[0] = z[0]
        
        # propagation des entrées vers la sortie

        n = len(z)

        # hidden layers
        for i in range(n-2) :
            
            # calcul des stimuli reçus par la couche cachée d'indice i+1 à-partir de la précedente

            # create a list with 1 in front for the bias coefficient
            z_1 = [1] + z[i]
            
            z̃[i+1] = M[i] * z_1 # z̃ = matrix * iterable (list here)
            
            # calcul des réponses des neurones cachés
            #z[i+1] = list(map(σ,z̃))
            #z[i+1] = list(map(tanh,z̃))
            z[i+1] = list(map(self.activation_function_hidden_layer,z̃[i+1])) 

            # update the variable when necessary
            self.z[i+1] = z[i+1]
            self.z̃[i+1] = z̃[i+1]


        # output layer
        i = i + 1

        # calcul des stimuli reçus par la couche cachée d'indice i+1 à-partir de la précedente

        # create a list with 1 in front for the bias coefficient
        z_1 = [1] + z[i]
        
        z̃[i+1] = M[i] * z_1 # z̃ = matrix * iterable (list here)
        
        # calcul des réponses des neurones de la couche de sortie
        z[i+1] = list(map(self.activation_function_output_layer,z̃[i+1])) 
        
        # update the variable when necessary
        self.z[i+1] = z[i+1]
        self.z̃[i+1] = z̃[i+1]
    

        #print("accepte_et_propage : self.z[i+1] ="); print(self.z[i+1])
        #return self.z[i+1]              # et retour des sorties


    def print_matrix_elements(self,M):
        
        for e in M:
            print(e)

        print()
    
    def apprentissage(self,Lexemples):  # apprentissage des poids par une liste d'exemples

        nbiter = self.nbiter

        ip = 0                          # numéro de l'exemple courant

        # TODO: take in account the error as stop point
        for it in range(nbiter):   # le nombre d'itérations est fixé !

            error = 0.0                     # l'erreur totale pour cet exemple
            
            (x,y) = Lexemples[ip]         # un nouvel exemple à apprendre
                      
            # PROPAGATION VERS L'AVANT
            self.accepte_et_propage(x)       # sorties obtenues sur l'exemple courant, self.z_k et z_j sont mis à jour
              
            # RETRO_PROPAGATION VERS L'ARRIERE, EN DEUX TEMPS

            # note: i just use local reference for the variables for code readness (hide all the self keyword)
            z = self.z
            z̃ = self.z̃

            i = i_output_layer = len(z) - 1 # start at index i of the ouput layer

            ᐁ = self.ᐁ
            
            ns = len(z[i])
            
            # TEMPS 1. calcul des gradients locaux sur la couche k de sortie (les erreurs commises)
            for k in range(ns):
                ᐁ[i][k] = y[k] - z[i][k]       # gradient sur un neurone de sortie (erreur locale)
                error += pow(ᐁ[i][k],2)                              # l'erreur quadratique totale
                
            error *= 0.5
            
            if it == nbiter-1 : self.error = error                # mémorisation de l'erreur totale à la dernière itération

            # modification des poids de la matrice de transition de la derniére couche de neurones cachés à la couche de sortie
            M = self.M # read/write data

            # because i dislike self keyword in my mathematical expressions i recopy the variables
            ηₛ = self.ηₛ
            ηₑ = self.ηₑ
            
            η = ηₛ
                        
            #η = ((ηₑ - ηₛ) / nbiter) * it + ηₛ
            #print(η)
                    
            # (test fait: modifier la matrice apres le calcul du gradient de la couche j (maintenant i-1) , conclusion: ne change pas la convergence de l'algo)

            მzⳆმz̃ = self.activation_function_output_layer_derivative
            
            self.modification_des_poids(M[i-1],η,z[i-1],z[i],z̃[i],ᐁ[i],მzⳆმz̃)

            #self.print_matrix_elements(M)
            
                        
            # TEMPS 2. calcul des gradients locaux sur les couches cachées (rétro-propagation), sauf pour le bias constant

            მzⳆმz̃ = self.activation_function_hidden_layer_derivative

            for i in reversed(range(1,i_output_layer)) :

                nc = len(z[i])
                ns = len(z[i+1])
                for j in range(nc):
                    
                    ᐁ[i][j] = sum(მzⳆმz̃(z[i+1][k],z̃[i+1][k]) * M[i][k][j+1] * ᐁ[i+1][k] for k in range(ns))

                # modification des poids de la matrice de transition de la couche i-1 à i
         
                self.modification_des_poids(M[i-1],η,z[i-1],z[i],z̃[i],ᐁ[i],მzⳆმz̃)

            #self.print_matrix_elements(M)

            # et l'on passe à l'exemple suivant
            
            #ip = (ip + 1) % len(Lexemples)      # parcours des exemples en ordre circulaire
            ip = randint(0,len(Lexemples) - 1)
           

    

            
    # modify coefficients layer
    def modification_des_poids(self,M_i_o,η,z_input,z_output,z̃_output,ᐁ_i_o,მzⳆმz̃): # derivative of activation function of the layer
        #print(z̃_output)
        
        # the length of output and input layer with coeff. used for bias update             
        (len_layer_output, len_layer_input_plus1forBias) = M_i_o.dim()
        
        len_layer_input = len_layer_input_plus1forBias - 1
       
        for j in range(len_layer_output):  # line
            
            for i in range(len_layer_input): # column , parcours les colonnes de la ligne sauf le bias

                M_i_o[j][i+1] -= -η * z_input[i] * მzⳆმz̃(z_output[j],z̃_output[j]) * ᐁ_i_o[j]

            # and update the bias
            M_i_o[j][0] -= -η * 1.0 * მzⳆმz̃(z_output[j],z̃_output[j]) * ᐁ_i_o[j]
                
            
                
    def dump(self,n,msg):     # dump du réseau en entrant dans l'itération numéro n
        print('---------- DUMP',msg,'itération numéro',n)
        print('M :') ; print(self.M)
        print('z  :',self.z)
        print('ᐁ :',self.ᐁ)
        print()

    def test(self,Lexemples):
        print('Test des exemples :')
        for (entree,sortie_attendue) in Lexemples:
            self.accepte_et_propage(entree)
            print(entree,'-->',self.z[len(self.z)-1],': on attendait',sortie_attendue)



            
if __name__ == '__main__':
    

    print('################## NOT ##################')
    r1 = ReseauRetroPropagation([1,2,1],50000,0.1,0.001,σ,σ,der_σ,der_σ)
    Lexemples1 = [[[1],[0]],[[0],[1]]]
    START = time() ; r1.apprentissage(Lexemples1) ; END = time()
    r1.test(Lexemples1)
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r1.nbiter,END-START))
    print()
    print("Error=") ; print(r1.error)
    
    print('################## XOR ##################')
    # 2 entrées (+ bias), 1 neurone en sortie
    r2 = ReseauRetroPropagation([2,3,1],250000,0.1,0.001,σ,σ,der_σ,der_σ) 
    #r2 = ReseauRetroPropagation([2,8,10,7,1],50000,0.1,0.001,σ,σ,der_σ,der_σ) 
    Lexemples2 = [[[1,0],[1]], [[0,0],[0]], [[0,1],[1]], [[1,1],[0]]]
    START = time() ; r2.apprentissage(Lexemples2) ; END = time()
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r2.nbiter,END-START))
    r2.test(Lexemples2)
    print("Error=") ; print(r2.error)
    #print("r2.mat_ij=",r2.mat_ij)

    print('################## SINUS ##################')
    #r3 = ReseauRetroPropagation([1,30,30,30,1],50000,0.01,0.000001,tanh,tanh,der_tanh,der_tanh)
    #r3 = ReseauRetroPropagation([1,30,30,30,1],5000,0.01,0.000001,tanh,tanh,der_tanh,der_tanh)
    #r3 = ReseauRetroPropagation([1,30,30,30,1],250000,0.01,0.000001,atan,tanh,der_atan,der_tanh)
    #r3 = ReseauRetroPropagation([1,30,30,30,1],50000,0.01,0.000001,leaky_RELU,tanh,der_leaky_RELU,der_tanh)
    r3 = ReseauRetroPropagation([1,70,70,1],nbiter=50000,ηₛ=0.01,ηₑ=0.000001) 
    Llearning = [ [[x],[sin(x)]] for x in [ uniform(-pi,pi) for n in range(10000)] ]
    Ltest = [ [[x],[sin(x)]] for x in [ uniform(-pi/2,pi/2) for n in range(10)] ]
    START = time() ; r3.apprentissage(Llearning) ; END = time()
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r3.nbiter,END-START))
    r3.test(Ltest)
    print("Error=") ; print(r3.error)
    
    # COMPLEMENTS EN LIGNE
    #from webbrowser import open as browse
    # Beaucoup de matériel sur la page Web de Geoffrey Hinton à Toronto, en particulier
    # l'article paru dans la revue "Nature" de 2015, et l'ancien MOOC de 2012 :
    #browse('https://www.cs.toronto.edu/~hinton/')
    # Si vous voulez vous lancer vraiment (avec TensorFlow de Google) :
    #browse('https://developers.google.com/machine-learning/crash-course/?hl=fr')
    # Et pour situer la place du "machine learning" dans l'IA :
    #browse('https://fr.wikipedia.org/wiki/Intelligence_artificielle')
    
    
    
    
