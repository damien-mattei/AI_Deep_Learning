# L'algorithme de rétro-propagation du gradient dans un
# réseau de neurones avec N couches cachées.

# modifications par D. Mattei

from random import seed, uniform
seed(1789)     # si vous voulez avoir les mêmes tirages aléatoires à chaque exécution du fichier !
from math import exp, pow
from MatrixNumPy import MatrixNumPy
from time import time


# sigmoïde
def sig(x):
    return 1/(1+ exp(-x))

class ReseauRetroPropagation():
    
    def __init__(self,nc=[2,3,1],nbiter=3,eta=1):
        '''Construit un réseau de neurones avec une couche cachée. Il y a ne entrées (+ biais),
        nc neurones dans la couche cachée (+ biais) et ns neurones en sortie.'''

        lnc = len(nc) # the total of all layer including input, output and hidden layers
        
        # on crée le tableau des couches du réseau
        # >>> nc=[2,3,1]
        # >>> [ [0] * n for n in nc ]
        # [[0, 0], [0, 0, 0], [0]]      
        self.z = [ [0] * n for n in nc ] # les entrées concrètes seront fournies avec la méthode accepte
        
        
        # nc[n] + 1 in the matrix size because we add one column of bias in the matrix for each hidden neuron of the hidden layer "c"

        # >>> mat = [ MatrixNumPy( lambda j,i: uniform(-1,1) , nc[n+1] , nc[n] + 1 ) for n in range(len(nc)-1) ]
        # # MatrixNumPy constructor MatrixNumPy (function,Numeric,Numeric) #
        # # MatrixNumPy constructor MatrixNumPy (function,Numeric,Numeric) #
        # [[[ 0.30891168 -0.06324858 -0.77054504]
        #  [ 0.56467559  0.4090438  -0.6001032 ]
        #  [ 0.04621124 -0.4736403   0.52908111]]
        # MatrixNumPy @ 0x7f14c2dfa090 
        # , [[-0.17710618 -0.32836366 -0.81737218  0.65399886]]
        # MatrixNumPy @ 0x7f14c2d9bad0 
        # ]

        # use with mat[0][1][2] or mat[0][1,2] notation
        #mat[i][j][k] == poids k->j from layer i to layer i+1
        self.mat =[ MatrixNumPy( lambda j,i: uniform(-1,1) , nc[n+1] , nc[n] + 1 ) for n in range(len(nc) - 1) ] 
        
        self.z_j = nc * [0]     # valeurs z_j des neurones cachés
        self.grad_j = nc * [0]    # gradients locaux des neurones cachés

        # nc+1 in the matrix size because with add one column of bias in the matrix for each neuron of the output layer "k"
        self.mat_jk = MatrixNumPy(lambda k,j: uniform(-1,1),ns,nc+1)  # self.mat_jk[k][j] == poids j->k

        self.z_k = ns * [0]     # valeurs z_k des neurones de sortie
        self.grad_k = ns * [0]    # gradients locaux des neurones de sortie
        
        self.nbiter = nbiter
        self.eta = eta                  # "learning rate" 
        self.error = 0


        
    # fusionne accept et propage
    # z_* sans le coef. 1 constant
    def accepte_et_propage(self,Lentrees):         # on entre des entrées et on les propage
        
        if len(Lentrees) != len(self.z_i):
            raise ValueError("Mauvais nombre d'entrées !")
        self.z_i = Lentrees       # on ne touche pas au biais
        
        # propagation des entrées vers la sortie
        
        # calcul des stimuli reçus par la couche cachée à-partir des entrées

        # note: i just reference the variables for code readness (hide all the self keyword)
        mat_ij = self.mat_ij
        z_i = self.z_i

        # create a list with 1 in front
        z_i_1 = [1] + z_i
        
        z̃_j = mat_ij * z_i_1 # z̃_i = matrix * iterable (list here)

        # calcul des réponses des neurones cachés
        z_j = list(map(sig,z̃_j)) 
            
        # calcul des stimuli reçus par la couche de sortie
        mat_jk = self.mat_jk

        # create a list with 1 in front
        z_j_1 = [1] + z_j
        
        z̃_k = mat_jk * z_j_1 # matrix * iterable (list here)

        # calcul des réponses des neurones cachés
        z_k = list(map(sig,z̃_k))
        
        # update the variable when necessary
        self.z_j = z_j
        self.z_k = z_k
        
        #return self.z_k               # et retour des sorties


    
    def apprentissage(self,Lexemples):  # apprentissage des poids par une liste d'exemples

        nbiter = self.nbiter

        ip = 0                          # numéro de l'exemple courant

        # TODO: take in account the error as stop point
        for it in range(nbiter):   # le nombre d'itérations est fixé !
            
            error = 0.0                     # l'erreur totale pour cet exemple
            if trace and (it in (0,nbiter-1)) and (ip == len(Lexemples)-1):
                self.dump(it,'entrée')
            (entrees,sorties_attendues) = Lexemples[ip]         # un nouvel exemple à apprendre
            if trace_full : print('\nExemple à apprendre :',entrees,'-->',sorties_attendues)
            
            # PROPAGATION VERS L'AVANT
            self.accepte_et_propage(entrees)       # sorties obtenues sur l'exemple courant, self.z_k et z_j sont mis à jour
              
            # RETRO_PROPAGATION VERS L'ARRIERE, EN DEUX TEMPS

            # note: i just reference the variables for code readness (hide all the self keyword)
            z_k = self.z_k # read-only variable
            grad_k = self.grad_k

            ns = len(z_k)
            
            # TEMPS 1. calcul des gradients locaux sur la couche k de sortie (les erreurs commises)
            for k in range(ns):
                grad_k[k] = sorties_attendues[k] - z_k[k]       # gradient sur un neurone de sortie (erreur locale)
                error += pow(grad_k[k],2)                              # l'erreur quadratique totale
                
            error *= 0.5
            #print(it)
            #print(error)
            if it == nbiter-1 : self.error = error                     # mémorisation de l'erreur totale à la dernière itération

            # modification des poids j->k
            mat_jk = self.mat_jk # read/write data

            z_i = self.z_i
            ne = len(z_i)
            z_j = self.z_j
            nc = len(z_j)
            eta = self.eta
            
           
            # (test fait: modifier la matrice apres le calcul du gradient de la couche j , conclusion: ne change pas la convergence de l'algo)

            self.modification_des_poids(mat_jk,eta,z_j,z_k,grad_k)
                       
            # for k in range(ns): # line
            #     for j in range(nc): # column , parcours les colonnes de la ligne sauf le bias
            #         mat_jk[k][j+1] -= - eta * z_j[j] * z_k[k] * (1 - z_k[k]) * grad_k[k]
            #     # and update the bias
            #     mat_jk[k][0] -= - eta * 1.0 * z_k[k] * (1 - z_k[k]) * grad_k[k]
                                
            # Réponse à la question "b4" : T_{jk} = z_k * (1-z_k) * w_{jk}


            
            # TEMPS 2. calcul des gradients locaux sur la couche j cachée (rétro-propagation), sauf pour le bias constant
            grad_j = self.grad_j
            
            for j in range(nc):
                grad_j[j] = sum(z_k[k] * (1 - z_k[k]) * mat_jk[k][j+1] * grad_k[k] for k in range(ns))
                
            
            # modification des poids i->j
            mat_ij = self.mat_ij
             
            self.modification_des_poids(mat_ij,eta,z_i,z_j,grad_j)
            
            # for j in range(nc):  # line
                
            #     for i in range(ne): # column , parcours les colonnes de la ligne sauf le bias
            #         mat_ij[j][i+1] -= -eta * z_i[i] * z_j[j] * (1 - z_j[j]) * grad_j[j]
                    
            #     # and update the bias
            #     mat_ij[j][0] -= -eta * 1.0 * z_j[j] * (1 - z_j[j]) * grad_j[j]
                
          
           
            # et l'on passe à l'exemple suivant
            if trace and (it in (0,nbiter-1)) and (ip == len(Lexemples)-1):
                self.dump(it,'sortie')
                
            ip = (ip + 1) % len(Lexemples)      # parcours des exemples en ordre circulaire

            self.grad_k = grad_k
            self.mat_jk = mat_jk
            self.grad_j = grad_j
            self.mat_ij = mat_ij


            
    def modification_des_poids(self,M_i_o,eta,z_input,z_output,grad_i_o):
        # the length of output and input layer with coeff. used for bias update             
        (len_layer_output, len_layer_input_plus1forBias) = M_i_o.dim()
        
        len_layer_input = len_layer_input_plus1forBias - 1

        
        for j in range(len_layer_output):  # line
            
            for i in range(len_layer_input): # column , parcours les colonnes de la ligne sauf le bias
                M_i_o[j][i+1] -= -eta * z_input[i] * z_output[j] * (1 - z_output[j]) * grad_i_o[j]

            # and update the bias
            M_i_o[j][0] -= -eta * 1.0 * z_output[j] * (1 - z_output[j]) * grad_i_o[j]
                
            
                
    def dump(self,n,msg):     # dump du réseau en entrant dans l'itération numéro n
        print('---------- DUMP',msg,'itération numéro',n)
        print('mat_ij :') ; print(self.mat_ij)
        print('z_j  :',self.z_j)
        print('grad_j :',self.grad_j)
        print('mat_jk :') ; print(self.mat_jk)
        print('z_k  :',self.z_k)
        print('grad_k :',self.grad_k)
        print()

    def test(self,Lexemples):
        print('Test des exemples :')
        for (entree,sortie_attendue) in Lexemples:
            self.accepte_et_propage(entree)
            print(entree,'-->',self.z_k,': on attendait',sortie_attendue)



            
if __name__ == '__main__':
    
    trace = False
    trace_full = False

    print('################## NOT ##################')
    r1 = ReseauRetroPropagation(1,2,1,nbiter=10000,eta=0.5)
    Lexemples1 = [[[1],[0]],[[0],[1]]]
    START = time() ; r1.apprentissage(Lexemples1) ; END = time()
    r1.test(Lexemples1)
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r1.nbiter,END-START))
    print()
    
    print('################## XOR ##################')
    r2 = ReseauRetroPropagation(2,3,1,nbiter=50000,eta=0.1)    # 2 entrées (+ bias), 3 neurones cachés (+ bias), 1 neurone en sortie
    Lexemples2 = [[[1,0],[1]], [[0,0],[0]], [[0,1],[1]], [[1,1],[0]]]
    START = time() ; r2.apprentissage(Lexemples2) ; END = time()
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r2.nbiter,END-START))
    r2.test(Lexemples2)
    print("Error=") ; print(r2.error)
    #print("r2.mat_ij=",r2.mat_ij)

    # COMPLEMENTS EN LIGNE
    from webbrowser import open as browse
    # Beaucoup de matériel sur la page Web de Geoffrey Hinton à Toronto, en particulier
    # l'article paru dans la revue "Nature" de 2015, et l'ancien MOOC de 2012 :
    #browse('https://www.cs.toronto.edu/~hinton/')
    # Si vous voulez vous lancer vraiment (avec TensorFlow de Google) :
    #browse('https://developers.google.com/machine-learning/crash-course/?hl=fr')
    # Et pour situer la place du "machine learning" dans l'IA :
    #browse('https://fr.wikipedia.org/wiki/Intelligence_artificielle')
    
    
    
    
