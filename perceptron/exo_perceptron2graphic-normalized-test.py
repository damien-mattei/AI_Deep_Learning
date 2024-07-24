#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exo_perceptron2.py
L'algorithme d'apprentissage des poids par le "perceptron".
Le premier poids est celui de l'entrée biaisée mise à 1.
"""

import random

from exo_perceptron1 import Perceptron

import globals

from toolboxGraphic import *

from time import *


# for graphics with Tk
import tkinter
from math import *

from logger import create_logger

LOG = create_logger()


# >>> V=[1,2,3]
# >>> norm(V)
# 3.7416573867739413
def norm(V):

    squares = list(map(lambda x: x * x , V))

    return sqrt(sum(squares))


    
class PerceptronApprentissage(Perceptron):
    
    '''Construit un perceptron capable de calculer les poids permettant de
    satisfaire un "training set".'''
    
    def __init__(self,window_title,Lexemples,timeout=120,unite=30,graphic_mode=False,demo_mode=False,display_line=False,normalize=False):
        
        '''La liste 'Lexemples' représente le training set : ce que l'on souhaire obtenir en réponse.
        Les poids sont initialisés aleatoirement. Abandonne le calcul au bout de 'timeout' secondes.'''

        self.demo_mode = demo_mode
        
        self.n_dim = len(Lexemples[0][0])

        self.normalize = normalize

        self.norm = 1

        self.prod_norms = 1 # produit des normes

        # correction vector computed by learning algorithm
        self.Lcorr=[0 for i in range(self.n_dim + 1)]
        
        # call the masters class init
        #super().__init__(Lpoids=[0 for i in range(n_dim + 1)])
        Perceptron.__init__(self,Lpoids=[0 for i in range(self.n_dim + 1)],demo_mode=demo_mode) # plus un pour le biais constant en premier
        
        self.Lexemples = [(L,v) for (L,v) in Lexemples]         
        self.timeout = timeout
        print('Training set :',self.Lexemples)
        print('Initialisation : bias = {} et poids = {}'.format(self.Lpoids[0],self.Lpoids[1:]))
        print('Lpoids = {}'.format(self.Lpoids))

        print("self.demo_mode={}".format(self.demo_mode))
        print("demo_mode={}".format(demo_mode))
        
        # graphic init
        if graphic_mode and self.n_dim == 2: # we only can plot 2 dimensions perceptron inputs

            self.graphic_mode = True
            self.graph=toolboxGraphic(window_title,unite,demo_mode=demo_mode)

            if display_line:
                # plot ideal separation line
                y1 = yd(xmin)
                y2 = yd(xmax)
                self.graph.ligne(xmin,y1,xmax,y2,color='cyan')
                
            #  Lexemples = [ ([x,y,z,...], val) , ... ]
            Lpoints = [pt for (pt,v) in Lexemples]
            for pt in Lpoints:
                x = pt[0]
                y = pt[1]
                self.graph.flash_disque(x,y,5,'orange')
                self.graph.root.update()
                
            self.cx = 0
            self.cy = 0

        else :
            self.graphic_mode = False
            
        self.apprentissage()


        
    def apprentissage(self):
        
        print("J'apprends ({} sec max)...".format(self.timeout))
        start = time()
        cpt = 0
        ip = 0                                                  # numéro du premier exemple

        
        while not self.apprentissage_fini():                    # tant que TOUT n'est pas appris :
            
            LOG.debug(self.__class__.__name__)
            
            cpt=cpt+1
            print('Itération: {} '.format(cpt))
            
            
            # couple input-output
            ces=self.Lexemples[ip]
            
            (entrees,sortie_attendue) = ces      # un nouvel exemple à apprendre
            
            print("ip={}".format(ip))        
            print("entrees={}".format(entrees))
            print('Lpoids = {}'.format(self.Lpoids))

            if self.graphic_mode :
                self.plot_input(entrees,sortie_attendue)

            sortie_obtenue = self.reponse(entrees)              # sortie obtenue sur l'exemple courant
            print("sortie_obtenue={}".format(sortie_obtenue))
            print("sortie_attendue={}".format(sortie_attendue))

            if sortie_obtenue != sortie_attendue :              # rectification de l'erreur
                
                print("La sortie obtenue ne correspond pas à celle attendue => Modifications poids...")
                erreur = self.modifier_poids(entrees,sortie_obtenue,sortie_attendue)   # learn !

                print("self.Lpoids = {}".format(self.Lpoids))

                if self.normalize:
                        self.normalize_weights()
                
                if self.graphic_mode :
                    self.draw_line_to_data(entrees)
                    self.draw_correction(erreur)
                    self.graph.check_idle()
                   
                    self.draw_normal_vector_and_correction()
                    self.draw_distance() # distance extremity
                    self.draw_normal()
                    
                    self.draw_separation_line()

                    self.graph.root.update()
                    #sleep(globals.tempo)

                    
                # affiche et vérifie que on a bien appris cet entrée?
                print("Verification sur l'entrée courante:")
                sortie_obtenue = self.reponse(entrees)              # sortie obtenue sur l'exemple courant
                print("sortie_obtenue={}".format(sortie_obtenue))
                print("sortie_attendue={}".format(sortie_attendue))
                if sortie_obtenue == sortie_attendue :
                    print("Ca marche sur l'entrée courante.")

            else:
                print("sorties obtenue et attendue sont identiques => pas de modification des poids.")
                if self.graphic_mode:
                    (c,a,b) = self.Lpoids # ax + by + c, equation droite
                    if a==0 :
                        if b==0 :
                            if c==0 :
                                print("No line to plot because defined set is the whole cartesian plane !")
                            else :
                                print("No line to plot because defined set is empty !")                    
                    #sleep(globals.tempo)
                
            

            if self.n_dim == 2:
                (c,a,b) = self.Lpoids # ax + by + c, equation droite
                print("{}*x + {}*y + {} = 0".format(a,b,c))

            print("self.Lpoids = {}".format(self.Lpoids))
            
            # on va passer à un autre couple entrees-sortie
            ip = (ip + 1) % len(self.Lexemples)                  # circulaire

            tim = time()
            if (tim - start > self.timeout):
                print("time = {}".format(tim))
                print("start = {}".format(start))
                print("self.timeout = {}".format(self.timeout))
                print("Trop long, j'abandonne l'apprentissage !")
                return

            if self.graphic_mode :
                self.graph.root.update()
                
            print("globals.idle= {}".format(globals.idle))
            while globals.idle :
                sleep(1)
                self.timeout = self.timeout + 1 # pour ne pas être à cours de temps
                self.graph.root.update() # indispensable pour eviter de bloquer l'IHM
                
            print();print()
            
            if self.graphic_mode:
                sleep(globals.tempo)
            
        print('Apprentissage terminé en {} itérations'.format(cpt))
        print('Apprentissage OK. Bias = {} et poids = {}'.format(self.Lpoids[0],self.Lpoids[1:]))


        
    def modifier_poids(self,entrees,sortie_obtenue,sortie_attendue) :     # mise à jour des poids

        LOG.debug(self.__class__.__name__)
        print("self.Lpoids = {}".format(self.Lpoids))

        if not(self.demo_mode) :
            entrees = [1] + entrees
        else : # en mode demo on fait disparaitre le bias pour passer par l'origine
            entrees = [0] + entrees

        erreur = sortie_attendue - sortie_obtenue # généralement -1 ou +1, va donner le sens de la correction
        
        print("erreur = sortie_attendue - sortie_obtenue = {}".format(erreur))
        print("erreur * entrees[i]",end='')
        print(" / prod_norms") if self.normalize else print()    
        
        for i in range(len(self.Lpoids)) :

            print("{} -> ".format(self.Lpoids[i]),end = '')
            #self.Lpoids[i] += erreur * entrees[i]

            # note: l'erreur pouvant être négative on ajoute ou soustrait le poids de l'erreur multiplié par l'entree[i]

            # si on a normalisé le vecteur normal on doit aussi mettre à l'échelle le vecteur correction (sinon la géometrie n'est pas conservée!)
            # ce n'est pas l'erreur (qui de toutes façon ici ne donne que le sens de la correction) que on met à l'echelle mais l'entrée
            # même si le calcul menant à l'erreur est déjà normalisé  l'erreur sera ici toujours 1 ou -1 donc il faut "normaliser" l'entrée aussi.
            #self.Lcorr[i] = erreur * entrees[i] / self.prod_norms if self.normalize and self.prod_norms !=0 and i > 0 else (erreur * entrees[i])
            # en fait ce serait plus simple avec des entrées normalisées (ce qui n'est pas le cas)

            self.Lcorr[i] = erreur * entrees[i] / self.prod_norms if self.normalize and self.prod_norms !=0 else (erreur * entrees[i]) 
            self.Lpoids[i] = self.Lpoids[i] + self.Lcorr[i]
            
            # généralement on corrige le vecteur Poids (normal) de la valeur de l'entrée dans le sens de l'erreur
            # ou de maniére proportionnelle dans le cas général

            print("{} modif par {} * {} ".format(self.Lpoids[i],erreur,entrees[i]),end ='')
            #if self.normalize and self.prod_norms !=0 and i > 0: print(" / {}".format(self.prod_norms))
            if self.normalize and self.prod_norms !=0 : print(" / {}".format(self.prod_norms))      
            print("  (Bias)")  if i==0 else print()

        print("self.Lcorr = {}".format(self.Lcorr))
            
        return erreur

    

    # normalize weights, so the norm of normal vector is unity and bias should be distance from origin 
    def normalize_weights(self):

        LOG.debug(self.__class__.__name__)
        
        V_normal = self.Lpoids[1:] # on exclu le Bias car on veux juste le vecteur normal à l'hyperplan
        self.norm = norm(V_normal)

        print("norm = {}".format(self.norm))

        self.prod_norms = self.prod_norms * self.norm

        print("prod_norms = {}".format(self.prod_norms))

        if self.norm != 0 :
            #self.Lpoids[1:] = list(map(lambda x:x / self.norm , self.Lpoids[1:]))
            self.Lpoids = list(map(lambda x:x / self.norm , self.Lpoids)) # note: on normalise aussi le Bias
            # car on veux que le Bias soit la distance de l'hyperplan à l'origine O

        
        print("self.Lpoids = {}".format(self.Lpoids))
        
    
            
    def apprentissage_fini(self):   # la leçon est-elle apprise ?
        LOG.debug(self.__class__.__name__)
        nb_appris = 0
        rv = True
        for i in range(len(self.Lexemples)):
            (entrees,sortie_attendue) = self.Lexemples[i]
            if self.reponse(entrees) != sortie_attendue :
                print("reponse precedente pas bonne => apprentissage pas fini.")
                rv = False
                #return False
            else :
                nb_appris = nb_appris + 1
        print("Nombre d'entrées apprisent par le Perceptron: {} sur {}".format(nb_appris,len(self.Lexemples)))
        print("sortie de apprentissage_fini(self)")
        return rv


    def plot_input(self,entrees,sortie_attendue):
        
        (x,y)=entrees
        LOG.debug(self.__class__.__name__)
        print(" x = {} , y = {} ".format(x,y))
        if sortie_attendue == 1:
            color='green'
        else:
            color='red'
        self.graph.flash_disque(x,y,5,color)

        self.graph.root.update_idletasks()

        
    def draw_separation_line(self):
        
        (c,a,b) = self.Lpoids # ax + by + c, equation droite
        self.graph.draw_cartesian_line(a,b,c)

            
    def draw_normal_vector_and_correction(self):
        
        LOG.debug(self.__class__.__name__)
                
        # draw normal vector and correction
        (cb,cx,cy) = self.Lcorr
        self.graph.draw_normal_vector_and_correction(cx,cy) # a , b are normal coords

        
    def draw_distance(self) :

        LOG.debug(self.__class__.__name__)
        # distance is the absolut value of Bias if normal vector (nx,ny) as 1 as length (vector normalized)
        (bias,nx,ny) = self.Lpoids

        if nx==0 :
            if ny==0 :
                if bias==0 :
                    print("No distance to plot because defined set is the whole cartesian plane !")
                    return
                else :
                    print("No distanceto plot because defined set is empty !")
                    return
                
        if self.normalize :
            distance = abs(bias)
            self.graph.draw_distance(distance,nx,ny)
        else :
            n = norm(self.Lpoids[1:])
            distance = abs(bias) / n
            self.graph.draw_distance(distance,nx / n,ny / n) # on aura besoin des coordonnées d'un vecteur normalisé
            
        

            
    def draw_normal(self) :

        LOG.debug(self.__class__.__name__)
        (c,a,b) = self.Lpoids # ax + by + c, equation droite
        if a==0 :
            if b==0 :
                if c==0 :
                    print("No normal vector to plot because defined set is the whole cartesian plane !")
                    return
                else :
                    print("No normal vector to plot because defined set is empty !")
                    return
      
        #if self.normalize :
        self.graph.draw_normal(a,b)

            
    # draw correction vector from origin O
    def draw_correction(self,erreur):

        LOG.debug(self.__class__.__name__)
        (cb,cx,cy) = self.Lcorr
        print("(cb,cx,cy) = {}".format(self.Lcorr))
        
        # if erreur < 0:
        #    self.graph.draw_anti_correction(cx,cy)
           
        self.graph.draw_correction(cx,cy)
        
    def draw_line_to_data(self,data):

        LOG.debug(self.__class__.__name__)
        (x,y)=data
        self.graph.draw_line_to_data(x,y)

        
        
if __name__ == '__main__':

    LOG.debug("exo_perceptron2graphic.py")
    globals.initialize()
    print("globals.idle".format( globals.idle )) # print the initial value

    display_line = False
    
    # un perceptron qui apprend le OU
    # L1=[([1,1],1),([0,0],0),([1,0],1),([0,1],1)]
    # p1 = PerceptronApprentissage("OR",Lexemples=L1,unite=90,timeout=1200,graphic_mode=True,display_line=display_line,normalize=False)   # ce que l'on veut
    # print('Vérifions si le perceptron p1 a appris la formule (x1 ou x2) :')
    # for x in (0,1) :
    #     for y in (0,1) :
    #         print('{} ou {} --> {}'.format(x,y,p1.reponse([x,y])))
   
    # if not p1.verify(L1):
    #     print("ERREUR le Perceptron n'a pas bien appris !")
    # else:
    #     print("Perceptron bien vérifié sur tout l'ensemble d'apprentissage.")
        
    # print()

    # sleep(5)
    
    # un perceptron qui apprend le ET
    # L2=[([1,1],1),([0,0],0),([1,0],0),([0,1],0)]
    # p2 = PerceptronApprentissage("AND",Lexemples=L2,unite=70,timeout=1200,graphic_mode=True,display_line=display_line,normalize=True)   # ce que l'on veut
    # print('Vérifions si le perceptron p1 a appris la formule (x1 et x2) :')
    # for x in (0,1) :
    #     for y in (0,1) :
    #         print('{} et {} --> {}'.format(x,y,p2.reponse([x,y])))

    # if not p2.verify(L2):
    #     print("ERREUR le Perceptron n'a pas bien appris !")
    # else:
    #     print("Perceptron bien vérifié sur tout l'ensemble d'apprentissage.")
    # print()

    # # un perceptron qui apprend le NON
    # L3=[([1],0),([0],1)]
    # p3 = PerceptronApprentissage(Lexemples=L3,normalize=True)   # ce que l'on veut
    # print('Vérifions si le perceptron p3 a appris la formule (non x1) :')
    # for x in (0,1) :
    #         print('non {} --> {}'.format(x,p3.reponse([x])))
   
    # p3.displayVerify(L3)
    # print()
    
    # # un perceptron qui apprend la formule logique (x1 et (x2 ou (non x3)))
    # L4=[([1,0,0],1),([1,0,1],0),([1,1,0],1),([1,1,1],1),
    #     ([0,0,0],0),([0,0,1],0),([0,1,0],0),([0,1,1],0)]
    # p4 = PerceptronApprentissage(Lexemples=L4,normalize=True)
    # print('Vérifions si le perceptron p4 a appris la formule (x1 et (x2 ou (non x3))) :')
    # for x in (0,1) :
    #     for y in (0,1) :
    #         for z in (0,1) :
    #             print('{} et ({} ou (non {})) -> {}'.format(x,y,z,p4.reponse([x,y,z])))
  
    # p4.displayVerify(L4)

    # print()

    
    # un perceptron qui apprend le "ou exclusif" XOR.
    # print('Une tentative avec le "ou exclusif", Ctrl-C pour stopper !')
    # p5 = PerceptronApprentissage(Lexemples=[([1,1],0),([0,1],1),([1,0],1),([0,0],0)], timeout=30)
    # print('Vérifions si le perceptron p5 a appris la formule (x1 XOR x2) :')
    # for x in (0,1) :
    #     for y in (0,1) :
    #         print('{} oux {} --> {}'.format(x,y,p5.reponse([x,y])))
    # print()

    demo_mode = False

    print("demo_mode={}".format(demo_mode))
    
    if not(demo_mode) :
        b = 2.3 + random.uniform(-0.5,0.5)
    else : # en mode demonstration on passe par l'origine pour une compréhension plus simple
        b = 0
        
    a = 0.4 + random.uniform(-0.2,0.2)

    
    #a = 0.4313933964737387
    
    
    #b = 2.7537892102696104
    
    print("y = {}.x + {}".format(a,b))
    
    # droite
    def yd(x):
        # coef. dir : (1,a)
        return a*x+b

    # droite cartesienne
    def dcart(x,y):
        # (1,a) = (-bp,ap)
        bp = -1
        ap = a
        cp = -b * bp
        return ap*x + bp*y + cp 

    xmin = -4
    xmax = 4

    ymin = -3
    print("ymin= {}".format(ymin))
    
    ymax = 3

    Lrand = [ [random.uniform(xmin,xmax),random.uniform(ymin + b,ymax + b)] for x in range(20)]

    
    L7 = [ ([x , y], 1 if dcart(x,y) <= 0 else 0) for [x , y] in Lrand]

    # to force recomputation of a previous data set
    

    #L7 = [([-1.9480624671882776, 5.03116331450617], 1), ([3.61196594114164, 5.607172614045583], 1), ([0.9279995797858636, 3.4686925230134205], 1), ([2.402108428816798, 5.37926892388518], 1), ([-1.1384186374165495, 1.821592963578646], 0), ([3.5742398135767957, 1.3894258880036408], 0), ([-2.6020773363243146, 2.1169165329993596], 1), ([-3.267886874183885, 0.6484831947279937], 0), ([0.22745848768929466, 4.018220659879464], 1), ([3.592335489020673, 3.36097032647568], 0), ([-1.482823136631966, 1.9842761471703168], 0), ([-1.9194201778891609, 3.800685884390733], 1), ([2.428553955800658, -0.2282873920923505], 0), ([-3.124553512062059, 0.47031001437314135], 0), ([1.1380808387651324, 0.5084082198335229], 0), ([-0.7999818254530968, 0.8057860428803123], 0), ([0.45473533965732305, 5.173825404509627], 1), ([-3.06572805329962, 1.8908580017940317], 1), ([-0.6937612148802685, 1.4180641922305732], 0), ([-3.88607748124244, 1.7370330644200969], 1)]
    # 425 iterations (normalisé ou pas )
    print("L7 = {}".format(L7))

    #p7 = PerceptronApprentissage(Lexemples=L7,unite=50,timeout=360000,demo_mode=demo_mode,graphic_mode=True,display_line=True,normalize=True)   # ce que l'on veut

    p7 = PerceptronApprentissage("Linear separator",Lexemples=L7,unite=50,timeout=360000,demo_mode=demo_mode,graphic_mode=True,display_line=True,normalize=False)   # ce que l'on veut

    if not p7.verify(L7):
        print("ERREUR le Perceptron n'a pas bien appris !")
    else:
        print("Perceptron bien vérifié sur tout l'ensemble d'apprentissage.")
        

    ##    from webbrowser import open as browse
    ##    browse('https://en.wikipedia.org/wiki/Perceptron')      # meilleur que la version française
    ##    browse('https://www.youtube.com/watch?v=5w-jolCle6g')
    ##    browse('https://fr.coursera.org/learn/neural-networks') # par Geoffrey Hinton !
    ##    browse('https://www.irif.fr/~kesner/enseignement/iup/cours81.pdf')
    ##    browse('https://en.wikipedia.org/wiki/History_of_artificial_intelligence#Perceptrons_and_the_dark_age_of_connectionism')
    

