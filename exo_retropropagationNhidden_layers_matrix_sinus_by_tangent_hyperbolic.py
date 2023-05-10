# L'algorithme de rétro-propagation du gradient dans un
# réseau de neurones avec N couches cachées.

#  D. Mattei

# python3.7 -O exo_retropropagationNhidden_layers_matrix_sinus.py

from random import seed, uniform,randint
#seed(1789)     # si vous voulez avoir les mêmes tirages aléatoires à chaque exécution du fichier !
from math import exp, pow, pi, sin , tanh
from Matrix import Matrix
from time import time


# sigmoïde
def σ(x):
    return 1/(1+ exp(-x))

def σࠤ(z):
    return σ(z)*(1-σ(z))


def tanhࠤ(x):
     return 1 - tanh(x)**2

class ReseauRetroPropagation():
    
    def __init__(self,nc=[2,3,1],nbiter=3,ηₛ=1.0,ηₑ=0.0001):
        '''Construit un réseau de neurones avec une couche cachée. Il y a ne entrées (+ biais),
        nc neurones dans la couche cachée (+ biais) et ns neurones en sortie.'''

        lnc = len(nc) # the total of all layer including input, output and hidden layers
        
        # on crée le tableau des couches du réseau
        # >>> nc=[2,3,1]
        # >>> [ [0] * n for n in nc ]
        # [[0, 0], [0, 0, 0], [0]]      
        self.z = [ [0] * n for n in nc ] # les entrées concrètes seront fournies avec la méthode accepte
               
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


        
    # fusionne accept et propage
    # z_* sans le coef. 1 constant pour le bias
    def accepte_et_propage(self,x):         # on entre des entrées et on les propage

        # note: i just reference the variables for code readness (hide all the self keyword)
        z = self.z
        M = self.M
        
        if len(x) != len(z[0]):
            raise ValueError("Mauvais nombre d'entrées !")
        
        z[0] = x       # on ne touche pas au biais
        self.z[0] = z[0]
        
        # propagation des entrées vers la sortie

        n = len(z)
        
        for i in range(n-1) :
            
            # calcul des stimuli reçus par la couche cachée d'indice i+1 à-partir de la précedente

            # create a list with 1 in front for the bias coefficient
            z_1 = [1] + z[i]
            
            z̃ = M[i] * z_1 # z̃ = matrix * iterable (list here)
            
            # calcul des réponses des neurones cachés ou de la couche de sortie
            #z[i+1] = list(map(σ,z̃))
            z[i+1] = list(map(tanh,z̃)) 

            # update the variable when necessary
            self.z[i+1] = z[i+1]

        #print("accepte_et_propage : self.z[i+1] ="); print(self.z[i+1])
        #return self.z[i+1]              # et retour des sorties


    
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

            i = i_output_layer = len(z) - 1 # start at index i of the ouput layer

            ᐁ = self.ᐁ
            
            ns = len(z[i])
            
            # TEMPS 1. calcul des gradients locaux sur la couche k de sortie (les erreurs commises)
            for k in range(ns):
                ᐁ[i][k] = y[k] - z[i][k]       # gradient sur un neurone de sortie (erreur locale)
                error += pow(ᐁ[i][k],2)                              # l'erreur quadratique totale
                
            error *= 0.5
            #print(it)
            #print(error)
            if it == nbiter-1 : self.error = error                # mémorisation de l'erreur totale à la dernière itération

            # modification des poids de la matrice de transition de la derniére couche de neurones cachés à la couche de sortie
            M = self.M # read/write data

            # beacause i dislike self keyword in my mathematical expressions i recopy the variables
            ηₛ = self.ηₛ
            ηₑ = self.ηₑ
            
            η = ηₛ
                        
            #η = ((ηₑ - ηₛ) / nbiter) * it + ηₛ
            #print(η)
                    
            # (test fait: modifier la matrice apres le calcul du gradient de la couche j (maintenant i-1) , conclusion: ne change pas la convergence de l'algo)

            self.modification_des_poids(M[i-1],η,z[i-1],z[i],ᐁ[i],tanhࠤ)

            #print(M[i-1])
                        
            # TEMPS 2. calcul des gradients locaux sur les couches cachées (rétro-propagation), sauf pour le bias constant

            #print(i_output_layer)
            for i in reversed(range(1,i_output_layer)) :

                nc = len(z[i])
                ns = len(z[i+1])
                for j in range(nc):
                    #ᐁ[i][j] = sum(z[i+1][k] * (1 - z[i+1][k]) * M[i][k][j+1] * ᐁ[i+1][k] for k in range(ns)) # sigmoide
                    ᐁ[i][j] = sum(tanhࠤ(z[i+1][k]) * M[i][k][j+1] * ᐁ[i+1][k] for k in range(ns))

                #print("ᐁ[i]=",ᐁ[i])
                
                # modification des poids de la matrice de transition de la couche i-1 à i
         
                self.modification_des_poids(M[i-1],η,z[i-1],z[i],ᐁ[i],tanhࠤ)

              
           
            # et l'on passe à l'exemple suivant
            
            #ip = (ip + 1) % len(Lexemples)      # parcours des exemples en ordre circulaire
            ip = randint(0,len(Lexemples) - 1)
           

    # modify coefficients layer
    def modification_des_poids(self,M_i_o,η,z_input,z_output,ᐁ_i_o,deriv): # derivative of activation function of the layer
        # the length of output and input layer with coeff. used for bias update             
        (len_layer_output, len_layer_input_plus1forBias) = M_i_o.dim()
        
        len_layer_input = len_layer_input_plus1forBias - 1
       
        for j in range(len_layer_output):  # line
            
            for i in range(len_layer_input): # column , parcours les colonnes de la ligne sauf le bias
                #M_i_o[j][i+1] -= -η * z_input[i] * z_output[j] * (1 - z_output[j]) * ᐁ_i_o[j] # sigmoid
                M_i_o[j][i+1] -= -η * z_input[i] * deriv(z_output[j]) * ᐁ_i_o[j]

            # and update the bias
            #M_i_o[j][0] -= -η * 1.0 * z_output[j] * (1 - z_output[j]) * ᐁ_i_o[j]
            M_i_o[j][0] -= -η * 1.0 * deriv(z_output[j]) * ᐁ_i_o[j]
                
            
                
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
    r1 = ReseauRetroPropagation([1,2,1],nbiter=10000,ηₛ=0.1)
    Lexemples1 = [[[1],[0]],[[0],[1]]]
    START = time() ; r1.apprentissage(Lexemples1) ; END = time()
    r1.test(Lexemples1)
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r1.nbiter,END-START))
    print()
    print("Error=") ; print(r1.error)
    
    print('################## XOR ##################')
    r2 = ReseauRetroPropagation([2,8,10,7,1],nbiter=50000,ηₛ=0.1)    # 2 entrées (+ bias), 1 neurone en sortie
    Lexemples2 = [[[1,0],[1]], [[0,0],[0]], [[0,1],[1]], [[1,1],[0]]]
    START = time() ; r2.apprentissage(Lexemples2) ; END = time()
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r2.nbiter,END-START))
    r2.test(Lexemples2)
    print("Error=") ; print(r2.error)
    #print("r2.mat_ij=",r2.mat_ij)

    print('################## SINUS ##################')
    r3 = ReseauRetroPropagation([1,30,30,30,1],nbiter=50000,ηₛ=0.01,ηₑ=0.000001)
    #r3 = ReseauRetroPropagation([1,70,70,1],nbiter=50000,ηₛ=0.01,ηₑ=0.000001) 
    Llearning = [ [[x],[sin(x)]] for x in [ uniform(-pi,pi) for n in range(1000)] ]
    Ltest = [ [[x],[sin(x)]] for x in [ uniform(-pi/2,pi/2) for n in range(10)] ]
    START = time() ; r3.apprentissage(Llearning) ; END = time()
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r3.nbiter,END-START))
    r3.test(Ltest)
    print("Error=") ; print(r3.error)
    
    # COMPLEMENTS EN LIGNE
    from webbrowser import open as browse
    # Beaucoup de matériel sur la page Web de Geoffrey Hinton à Toronto, en particulier
    # l'article paru dans la revue "Nature" de 2015, et l'ancien MOOC de 2012 :
    #browse('https://www.cs.toronto.edu/~hinton/')
    # Si vous voulez vous lancer vraiment (avec TensorFlow de Google) :
    #browse('https://developers.google.com/machine-learning/crash-course/?hl=fr')
    # Et pour situer la place du "machine learning" dans l'IA :
    #browse('https://fr.wikipedia.org/wiki/Intelligence_artificielle')
    
    
    
    
