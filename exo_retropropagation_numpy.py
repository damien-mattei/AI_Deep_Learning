# exo_retro_propagation.py
# L'algorithme de rétro-propagation du gradient dans un
# réseau de neurones avec 1 couche cachée.

#  launch with: python3.11 -O exo_retropropagation_numpy.py

from random import seed, uniform
seed(1789)     # si vous voulez avoir les mêmes tirages aléatoires à chaque exécution du fichier !
from math import exp, pow
#from Matrix import Matrix
from time import time

import numpy as np


# >>> np.matrix(np.fromfunction(lambda j,i: i + j, (2,3)))
# matrix([[0., 1., 2.],
#         [1., 2., 3.]])

# >>> matrix_from_function(lambda i,j: i + j, (2,2))
# matrix_from_function :

# dm = (2, 2)

# res =
# [[0. 1.]
#  [1. 2.]]
# matrix([[0., 1.],
#         [1., 2.]])
# >>> 
def matrix_from_function(lb,dm): # lambda, dimension as tuple

    if debug:
            print("matrix_from_function :\n")
            print("dm = {}\n".format(dm))

    res = np.matrix(np.fromfunction(np.vectorize(lb),dm))
    if debug:
        print("res =")
        print(res)
    return res

                     
# >>> t=[1,2,3,4]
# >>> np.matrix(np.column_stack((t,)))
# matrix([[1],
#         [2],
#         [3],
#         [4]])
                  
def column_matrix_from_list(L):

    return np.matrix(np.column_stack((L,)))

# >>> m1col
# matrix([[1],
#         [2],
#         [3],
#         [4]])
# >>> m1col.T.tolist()[0]
# [1, 7, 8, 9]
def column_matrix_to_list(Mcol):

    return Mcol.T.tolist()[0]

                     
debug = False               

# sigmoïde
def sig(x):
    return 1/(1+ exp(-x))

class ReseauRetroPropagation():
    
    def __init__(self,ne=2,nc=3,ns=1,nbiter=3,eta=1):
        '''Construit un réseau de neurones avec une couche cachée. Il y a ne entrées (+ biais),
        nc neurones dans la couche cachée (+ biais) et ns neurones en sortie.'''
        self.ne = ne # entrée /  input
        self.nc = nc # caché / hidden
        self.ns = ns # sortie / output
        print(self.ne,'entrées(+1),',self.nc,'neurones cachés(+1) et',self.ns,'en sortie.')
        # le réseau calcule sur 7 vecteurs et 2 matrices
        self.act_i = [1] + ne * [0]     # les entrées concrètes seront fournies avec la méthode accepte
        self.mat_ij = matrix_from_function(lambda j,i: uniform(-1,1), (nc,ne+1))  # self.mat_ij[j,i] == poids i->j
        if debug:
            print("ReseauRetroPropagation : __init__ :\n")
            print("ne = {}\n".format(ne))
            print("nc = {}\n".format(nc))
            print("self.mat_ij = \n")
            print(self.mat_ij)
        self.stim_j = [1] + nc * [0]    # les stimuli \tilde{z}_j = Z̃_j = z̃_j reçus par les neurones cachés
        self.act_j = [1] + nc * [0]     # valeurs z_j des neurones cachés
        self.grad_j = [0] + nc * [0]    # gradients locaux des neurones cachés
        self.mat_jk = matrix_from_function(lambda k,j: uniform(-1,1), (ns,nc+1))  # self.mat_jk[k,j] == poids j->k
        self.stim_k = ns * [0]          # les stimuli \tilde{z}_j reçus par les neurones cachés
        self.act_k = ns * [0]     # valeurs z_k des neurones de sortie
        self.grad_k = ns * [0]    # gradients locaux des neurones de sortie
        self.nbiter = nbiter
        self.eta = eta                  # "learning rate" 
        self.error = 0

        
    def accepte(self,Lentrees):         # on entre des entrées et on les propage
        
        if len(Lentrees) != len(self.act_i) - 1:
            raise ValueError("Mauvais nombre d'entrées !")
        self.act_i[1:] = Lentrees       # on ne touche pas au biais
        if trace_full : print('self.act_i =',self.act_i)
        self.propage()                  # propagation des entrées
        return self.act_k               # et retour des sorties

    
    def propage(self) :                # propagation des entrées vers la sortie
        
        # calcul des stimuli reçus par la couche cachée à partir des entrées
        # i dislike using self in all variables,but python has no reference like C++ so i must copy the variables
        # and anyway i need more variables for numpy
        stim_j = self.stim_j
        mat_ij = self.mat_ij
        act_i = self.act_i
                     
        MATCOLact_i = column_matrix_from_list(self.act_i) # Matrix column

        if debug:
            print("ReseauRetroPropagation : propage :\n")
            print("mat_ij = \n")
            print(mat_ij)
            print("MATCOLact_i =\n")
            print(MATCOLact_i)
            
        MATCOLstim_j = mat_ij * MATCOLact_i # Z̃_j , this is an application, a Matrix multiplication of a Matrix with a Matrix column

        stim_j = column_matrix_to_list(MATCOLstim_j)
                     
        if trace_full : print('z̃_j = stim_j =',stim_j)       

        # calcul des réponses des neurones cachés
        act_j = [1] + list(map(sig,stim_j))
        if trace_full : print('z_j = act_j =',act_j)
        
        # calcul des stimuli reçus par la couche de sortie
                     
        mat_jk = self.mat_jk
        
        MATCOLact_j = column_matrix_from_list(act_j) # Matrix column
        
        MATCOLstim_k = mat_jk * MATCOLact_j

        stim_k = column_matrix_to_list(MATCOLstim_k)

        if trace_full : print('z̃_k = stim_k =',stim_k)
        
        # calcul des réponses des neurones cachés
        act_k = list(map(sig,stim_k))
        
        if trace_full : print('act_k =',act_k)
        if trace_full : print('forward prop. finished and seems OK')

        # after computation without self i need to copy back the variables 
        self.stim_j = stim_j
        self.act_j = act_j
        self.stim_k = stim_k
        self.act_k = act_k


        
    def apprentissage(self,Lexemples):  # apprentissage des poids par une liste d'exemples
        ip = 0                          # numéro de l'exemple courant
        
        for it in range(self.nbiter):   # le nombre d'itérations est fixé !
            error = 0.0                     # l'erreur totale pour cet exemple
            if trace and (it in (0,self.nbiter-1)) and (ip == len(Lexemples)-1):
                self.dump(it,'entrée')
            (entrees,sorties_attendues) = Lexemples[ip]         # un nouvel exemple à apprendre
            if trace_full : print('\nExemple à apprendre :',entrees,'-->',sorties_attendues)
            
            # PROPAGATION VERS L'AVANT
            self.accepte(entrees)       # sorties obtenues sur l'exemple courant, self.act_k est mis à jour
            if trace_full: print('act_k =',self.act_k)
            
            # RETRO_PROPAGATION VERS L'ARRIERE, EN DEUX TEMPS
            
            # TEMPS 1. calcul des gradients locaux sur la couche k de sortie (les erreurs commises)
            for k in range(self.ns):
                self.grad_k[k] = sorties_attendues[k] - self.act_k[k]       # gradient sur un neurone de sortie (erreur locale)
                error += pow(self.grad_k[k],2)                              # l'erreur quadratique totale
            error *= 0.5
            if it == self.nbiter-1 : self.error = error                     # mémorisation de l'erreur totale à la dernière itération

            # modification des poids j->k
            if trace_full:
                print('self.mat_jk=',self.mat_jk)
                print('self.nc=',self.nc)
                print('self.ns=',self.ns)
                
            for j in range(self.nc+1):
                if trace_full:
                        print("j=",j)
                for k in range(self.ns):
                    if trace_full:
                        print("k=",k)
                        print("self.mat_jk[k,j]=",self.mat_jk[k,j])
                    #self.mat_jk.L[k][j] -= -self.eta*self.act_j[j]*self.act_k[k]*(1-self.act_k[k])*self.grad_k[k]
                    self.mat_jk[k,j] -= -self.eta*self.act_j[j]*self.act_k[k]*(1-self.act_k[k])*self.grad_k[k]
                    
            # Réponse à la question "b4" : T_{jk} = z_k * (1-z_k) * w_{jk}

            # TEMPS 2. calcul des gradients locaux sur la couche j cachée (rétro-propagation), sauf pour le bias constant
            for j in range(1,self.nc+1):
                #self.grad_j[j] = sum(self.act_k[k]*(1-self.act_k[k])*self.mat_jk.L[k][j]*self.grad_k[k] for k in range(self.ns))
                self.grad_j[j] = sum(self.act_k[k]*(1-self.act_k[k])*self.mat_jk[k,j]*self.grad_k[k] for k in range(self.ns))
                
            if trace_full: print('gradients sur la couche j :',self.grad_j)
            
            # modification des poids i->j
            for i in range(self.ne+1):
                for j in range(1,self.nc+1):
                    #self.mat_ij.L[j-1][i] -= -self.eta*self.act_i[i]*self.act_j[j]*(1-self.act_j[j])*self.grad_j[j]
                    self.mat_ij[j-1,i] -= -self.eta*self.act_i[i]*self.act_j[j]*(1-self.act_j[j])*self.grad_j[j]
                    
            # et l'on passe à l'exemple suivant
            if trace and (it in (0,self.nbiter-1)) and (ip == len(Lexemples)-1):
                self.dump(it,'sortie')
            ip = (ip + 1) % len(Lexemples)      # parcours des exemples en ordre circulaire



                
    def dump(self,n,msg):     # dump du réseau en entrant dans l'itération numéro n
        print('---------- DUMP',msg,'itération numéro',n)
        print('mat_ij :') ; print(self.mat_ij)
        print('act_j  :',self.act_j)
        print('grad_j :',self.grad_j)
        print('mat_jk :') ; print(self.mat_jk)
        print('act_k  :',self.act_k)
        print('grad_k :',self.grad_k)
        print()

    def test(self,Lexemples):
        print('Test des exemples :')
        for (entree,sortie_attendue) in Lexemples:
            self.accepte(entree)
            print(entree,'-->',self.act_k,': on attendait',sortie_attendue)
        
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
    r2 = ReseauRetroPropagation(2,3,1,nbiter=200000,eta=0.1)    # 2 entrées (+ bias), 3 neurones cachés (+ bias), 1 neurone en sortie
    Lexemples2 = [[[1,0],[1]], [[0,0],[0]], [[0,1],[1]], [[1,1],[0]]]
    START = time() ; r2.apprentissage(Lexemples2) ; END = time()
    print('APPRENTISSAGE sur {} itérations, time = {:.2f}s'.format(r2.nbiter,END-START))
    r2.test(Lexemples2)

    # COMPLEMENTS EN LIGNE
    from webbrowser import open as browse
    # Beaucoup de matériel sur la page Web de Geoffrey Hinton à Toronto, en particulier
    # l'article paru dans la revue "Nature" de 2015, et l'ancien MOOC de 2012 :
    #browse('https://www.cs.toronto.edu/~hinton/')
    # Si vous voulez vous lancer vraiment (avec TensorFlow de Google) :
    #browse('https://developers.google.com/machine-learning/crash-course/?hl=fr')
    # Et pour situer la place du "machine learning" dans l'IA :
    #browse('https://fr.wikipedia.org/wiki/Intelligence_artificielle')
    
    
    
    
