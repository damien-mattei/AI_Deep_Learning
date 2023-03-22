# exo_retro_propagation.py
# L'algorithme de rétro-propagation du gradient dans un
# réseau de neurones avec 1 couche cachée.

from random import seed, uniform
seed(1789)     # si vous voulez avoir les mêmes tirages aléatoires à chaque exécution du fichier !
from math import exp, pow
from MatrixNumPy import MatrixNumPy
from time import time

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
        self.z_i = [1] + ne * [0]     # les entrées concrètes seront fournies avec la méthode accepte,on a rajouté un 1, coefficient pour le biais
        self.mat_ij = MatrixNumPy(lambda j,i: uniform(-1,1),nc,ne+1)  # self.mat_ij[j][i] == poids i->j
        
        self.z̃_j = [1] + nc * [0]    # les stimuli \tilde{z}_j z̃_j reçus par les neurones cachés
        #print("__init__ : len(self.z̃_j) =",len(self.z̃_j))
        
        self.z_j = [1] + nc * [0]     # valeurs z_j des neurones cachés
        self.grad_j = [0] + nc * [0]    # gradients locaux des neurones cachés
        self.mat_jk = MatrixNumPy(lambda k,j: uniform(-1,1),ns,nc+1)  # self.mat_jk[k][j] == poids j->k
        
        self.z̃_k = ns * [0]          # les stimuli \tilde{z}_k z̃_k reçus par les neurones de sortie
        self.z_k = ns * [0]     # valeurs z_k des neurones de sortie
        self.grad_k = ns * [0]    # gradients locaux des neurones de sortie
        
        self.nbiter = nbiter
        self.eta = eta                  # "learning rate" 
        self.error = 0


    # todo: fusionner accept et propage
    # faire des z_i sans le coef. 1 constant
    def accepte(self,Lentrees):         # on entre des entrées et on les propage
        
        if len(Lentrees) != len(self.z_i) - 1:
            raise ValueError("Mauvais nombre d'entrées !")
        self.z_i[1:] = Lentrees       # on ne touche pas au biais
        if trace_full : print('self.z_i =',self.z_i)
        self.propage()                  # propagation des entrées
        return self.z_k               # et retour des sorties

    
    def propage(self) :                # propagation des entrées vers la sortie
        
        # calcul des stimuli reçus par la couche cachée à-partir des entrées

        # note: i just reference the variables for code readness (hide all the self keyword)
        mat_ij = self.mat_ij
        z_i = self.z_i # z_i

        z̃_j = mat_ij * z_i # z̃_i = matrix * iterable (list here) = Mij * z_i
        
        if trace_full : print('z̃_j =',z̃_j)
        

        # calcul des réponses des neurones cachés
        z_j = [1] + list(map(sig,z̃_j))    # z_j with 1 to compute bias 
        if trace_full : print('z_j =',z_j)
        
        # calcul des stimuli reçus par la couche de sortie
        mat_jk = self.mat_jk
        
        z̃_k = mat_jk * z_j # matrix * iterable (list here)

        if trace_full : print('z̃_k =',z̃_k)
        
        # calcul des réponses des neurones cachés
        z_k = list(map(sig,z̃_k))
        
        if trace_full : print('z_k =',z_k)
        if trace_full : print('forward prop. finished and seems OK')

        # update the variable when necessary
        self.z̃_j = z̃_j
        self.z_j = z_j
        self.z̃_k = z̃_k
        self.z_k = z_k
        #print("propage : len(self.z̃_j) =",len(self.z̃_j))


        
    def apprentissage(self,Lexemples):  # apprentissage des poids par une liste d'exemples

        nbiter = self.nbiter

        ip = 0                          # numéro de l'exemple courant

        # TODO: take in account the error as stopper
        for it in range(nbiter):   # le nombre d'itérations est fixé !
            
            error = 0.0                     # l'erreur totale pour cet exemple
            if trace and (it in (0,nbiter-1)) and (ip == len(Lexemples)-1):
                self.dump(it,'entrée')
            (entrees,sorties_attendues) = Lexemples[ip]         # un nouvel exemple à apprendre
            if trace_full : print('\nExemple à apprendre :',entrees,'-->',sorties_attendues)
            
            # PROPAGATION VERS L'AVANT
            self.accepte(entrees)       # sorties obtenues sur l'exemple courant, self.z_k et z_j sont mis à jour
            if trace_full: print('z_k =',self.z_k)
            
            # RETRO_PROPAGATION VERS L'ARRIERE, EN DEUX TEMPS

            # note: i just reference the variables for code readness (hide all the self keyword)
            z_k = self.z_k # read-only variable
            grad_k = self.grad_k
            
            # TEMPS 1. calcul des gradients locaux sur la couche k de sortie (les erreurs commises)
            for k in range(self.ns):
                grad_k[k] = sorties_attendues[k] - z_k[k]       # gradient sur un neurone de sortie (erreur locale)
                error += pow(grad_k[k],2)                              # l'erreur quadratique totale
                
            error *= 0.5
            #print(it)
            #print(error)
            if it == nbiter-1 : self.error = error                     # mémorisation de l'erreur totale à la dernière itération

            # modification des poids j->k
            mat_jk = self.mat_jk # read/write data
            nc = self.nc
            ns = self.ns
            ne = self.ne
            z_j = self.z_j
            eta = self.eta
            
            if trace_full:
                print('mat_jk=',mat_jk)
                print('nc=',nc)
                print('ns=',ns)

            # (test fait: modifier la matrice apres le calcul du gradient de la couche j , conclusion: ne change pas la convergence de l'algo)

            #self.modification_des_poids(mat_jk,eta,z_j,z_k,grad_k)
            for j in range(nc+1):
                for k in range(ns):
                    mat_jk[k][j] -= - eta * z_j[j] * z_k[k] * (1 - z_k[k]) * grad_k[k]
                                
            # Réponse à la question "b4" : T_{jk} = z_k * (1-z_k) * w_{jk}

            
            # TEMPS 2. calcul des gradients locaux sur la couche j cachée (rétro-propagation), sauf pour le bias constant
            grad_j = self.grad_j
            
            for j in range(1,nc+1): # grad_j est indexé à partir de 1 !
                grad_j[j] = sum(z_k[k] * (1 - z_k[k]) * mat_jk[k][j] * grad_k[k] for k in range(ns))
                
            if trace_full: print('gradients sur la couche j :',grad_j)
          
            z_i = self.z_i
            
            # modification des poids i->j
            mat_ij = self.mat_ij
            
            for i in range(ne+1):
                for j in range(1,nc+1):  # car on a  indexé à partir de 1 grad_j à cause des z_j !
                    mat_ij[j-1][i] -= -eta * z_i[i] * z_j[j] * (1 - z_j[j]) * grad_j[j]
                    
                # for j in range(1,nc): # original version
                #     mat_ij[j-1][i] -= -eta * z_i[i] * z_j[j] * (1 - z_j[j]) * grad_j[j]
                
                    
            # et l'on passe à l'exemple suivant
            if trace and (it in (0,nbiter-1)) and (ip == len(Lexemples)-1):
                self.dump(it,'sortie')
                
            ip = (ip + 1) % len(Lexemples)      # parcours des exemples en ordre circulaire

            self.grad_k = grad_k
            self.mat_jk = mat_jk
            self.grad_j = grad_j
            self.mat_ij = mat_ij

            

    def modification_des_poids(self,M,eta,z_input,z_output,grad_i_o):
        # the length of input layer and bias
        #print("modification_des_poids")
        (len_layer_input_plus1forBias, len_layer_output) = M.dim()
        print("len_layer_output=",len_layer_output)
        print("len_layer_input_plus1forBias=",len_layer_input_plus1forBias)
        
        for i in range(len_layer_output):
            print("i=",i)
            for j in range(1,len_layer_input_plus1forBias):
                    M[j-1][i] -= -eta * z_input[i] * z_output[j] * (1 - z_output[j]) * grad_i_o[j]
                    print("j=",j)
                  
            
            
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
            self.accepte(entree)
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

    # COMPLEMENTS EN LIGNE
    from webbrowser import open as browse
    # Beaucoup de matériel sur la page Web de Geoffrey Hinton à Toronto, en particulier
    # l'article paru dans la revue "Nature" de 2015, et l'ancien MOOC de 2012 :
    #browse('https://www.cs.toronto.edu/~hinton/')
    # Si vous voulez vous lancer vraiment (avec TensorFlow de Google) :
    #browse('https://developers.google.com/machine-learning/crash-course/?hl=fr')
    # Et pour situer la place du "machine learning" dans l'IA :
    #browse('https://fr.wikipedia.org/wiki/Intelligence_artificielle')
    
    
    
    
