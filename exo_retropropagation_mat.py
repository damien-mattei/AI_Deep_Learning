# exo_retro_propagation.py
# L'algorithme de rétro-propagation du gradient dans un
# réseau de neurones avec 1 couche cachée.

# launch with: python3.11 -O exo_retropropagation_mat.py

from random import seed, uniform
seed(1789)     # si vous voulez avoir les mêmes tirages aléatoires à chaque exécution du fichier !
from math import exp, pow
from exo_mat2 import Mat
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
        self.act_i = [1] + ne * [0]     # les entrées concrètes seront fournies avec la méthode accepte
        self.mat_ij = Mat(lambda j,i: uniform(-1,1),nc,ne+1)  # self.mat_ij[j][i] == poids i->j
        self.stim_j = [1] + nc * [0]    # les stimuli \tilde{z}_j reçus par les neurones cachés
        self.act_j = [1] + nc * [0]     # valeurs z_j des neurones cachés
        self.grad_j = [0] + nc * [0]    # gradients locaux des neurones cachés
        self.mat_jk = Mat(lambda k,j: uniform(-1,1),ns,nc+1)  # self.mat_jk[k][j] == poids j->k
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
        # calcul des stimuli reçus par la couche cachée à-partir des entrées
        self.stim_j = self.mat_ij.app(self.act_i)
        if trace_full : print('self.stim_j =',self.stim_j)
        # calcul des réponses des neurones cachés
        self.act_j = [1] + list(map(sig,self.stim_j))
        if trace_full : print('self.act_j =',self.act_j)
        # calcul des stimuli reçus par la couche de sortie
        self.stim_k = self.mat_jk.app(self.act_j)
        if trace_full : print('self.stim_k =',self.stim_k)
        # calcul des réponses des neurones cachés
        self.act_k = list(map(sig,self.stim_k))
        #if trace_full : print('self.act_k =',self.act_k)
        if trace_full : print('forward prop. finished and seems OK')

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
            if trace_full: print('self.mat_jk.L=',self.mat_jk.L)
            for j in range(self.nc+1):
                for k in range(self.ns):
                    #self.mat_jk.L[k][j] -= -self.eta*self.act_j[j]*self.act_k[k]*(1-self.act_k[k])*self.grad_k[k]
                    self.mat_jk[k][j] -= -self.eta*self.act_j[j]*self.act_k[k]*(1-self.act_k[k])*self.grad_k[k]
                    
            # Réponse à la question "b4" : T_{jk} = z_k * (1-z_k) * w_{jk}

            # TEMPS 2. calcul des gradients locaux sur la couche j cachée (rétro-propagation), sauf pour le bias constant
            for j in range(1,self.nc+1):
                #self.grad_j[j] = sum(self.act_k[k]*(1-self.act_k[k])*self.mat_jk.L[k][j]*self.grad_k[k] for k in range(self.ns))
                self.grad_j[j] = sum(self.act_k[k]*(1-self.act_k[k])*self.mat_jk[k][j]*self.grad_k[k] for k in range(self.ns))
                
            if trace_full: print('gradients sur la couche j :',self.grad_j)
            
            # modification des poids i->j
            for i in range(self.ne+1):
                for j in range(1,self.nc+1):
                    #self.mat_ij.L[j-1][i] -= -self.eta*self.act_i[i]*self.act_j[j]*(1-self.act_j[j])*self.grad_j[j]
                    self.mat_ij[j-1][i] -= -self.eta*self.act_i[i]*self.act_j[j]*(1-self.act_j[j])*self.grad_j[j]
                    
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
    
    
    
    
