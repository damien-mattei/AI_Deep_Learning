#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exo_perceptron1.py
Le "perceptron" de base, avec sa fonction d'activation Y de Heaviside.
On choisit à la main les poids dont celui du biais. Le biais est constant (1)
et forcé en première entrée du perceptron.
"""





# --------------- question a) ---------------------------------------

def Y(x):       # la fonction d'Heaviside
    return 1 if x >= 0 else 0

class Perceptron:
    '''Construit un perceptron dont on donne la liste des poids Lpoids,
    le premier poids étant celui du biais.'''
    
    def __init__(self,Lpoids,demo_mode=False):
        self.Lpoids = Lpoids
        self.demo_mode = demo_mode

        
   
    def reponse(self,Lentrees):
        '''La liste Lentrees représente les données concrètes, sans le biais.'''

        print("Perceptron :: reponse")
        print('Lpoids = {}'.format(self.Lpoids))
        print("Lentrees = {}".format(Lentrees))
        if len(Lentrees) != len(self.Lpoids) - 1:
            raise ValueError("Mauvais nombre d'entrées !")

        if not(self.demo_mode) :
            Lentrees = [1] + Lentrees         # le biais est constant en tête et vaut 1
            #Lentrees = [-1] + Lentrees         # le biais est constant en tête et vaut -1
        else : # en mode demo on fait disparaitre le bias pour passer par l'origine
            Lentrees = [0] + Lentrees
            
        lg = len(Lentrees)
        
        # affiche le détail du calcul
        for i in range(lg): 
            print("Le[{}] * Lp[{}]".format(i,i),end='')
            if i != lg - 1 :
                print(" + ",end='')
        print(" =")
        for i in range(lg):      
            print("({} * {})".format(Lentrees[i], self.Lpoids[i]),end='')
            if i != lg - 1 :
                print(" + ",end='')

        somme_ponderee = sum(Lentrees[i] * self.Lpoids[i] for i in range(len(Lentrees)))

        print(" = {}".format(somme_ponderee))

        # apply Heaviside function
        Ys = Y(somme_ponderee)
        print("Y({}) = {}".format(somme_ponderee,Ys))

        return Ys

    
    # verify the Perceptron on a given set of input and output
    # note: cette fonctionalité existe déjà dans la classe d'apprentissage du perceptron
    def verify(self,Lexemples):
        for (input,output) in Lexemples:
            if self.reponse(input) != output:
                return False
        return True


    def displayVerify(self,Lexemples):
        if not self.verify(Lexemples):
            print("ERREUR le Perceptron n'a pas bien appris !")
        else:
            print("Perceptron bien vérifié sur tout l'ensemble d'apprentissage.")


            
            
if __name__ == '__main__':
    print('Perceptron p1 : x1 OU x2')
    p1 = Perceptron([-1,1,1])
    for x in (0,1):
        for y in (0,1):
            print('{} ou {} --> {}'.format(x,y,p1.reponse([x,y])))
    print()

    print('Perceptron p2 : x1 ET x2')
    p2 = Perceptron([-3,1,2])       # le biais vaut -3, les deux poids sont 1 et 2
    for x in (0,1):
        for y in (0,1):
            print('{} et {} --> {}'.format(x,y,p2.reponse([x,y])))
    print()

    print('Perceptron p3 : NON x1')
    p3 = Perceptron([0,-1])         # le biais vaut 0, le poids de l'unique entrée est -1
    for x in (0,1):
        print('non {} --> {}'.format(x,p3.reponse([x])))
    print()

    print('Perceptron p4 : x1 ET (x2 OU (NON x3))')
    p4 = Perceptron([-2,2,1,-1])    # le biais vaut -2, les poids sont 2, 1 et -1
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                print('{} et ({} ou (non {})) --> {}'.format(x,y,z,p4.reponse([x,y,z])))
    print()

# --------------- question b) ---------------------------------------

# Par l'absurde, supposons qu'il existe un perceptron réalisant le XOR (ou exclusif).
# p5 = Perceptron([w0,w1,w2]) où w0 est le biais. La réponse sur les entrées [x1,x2]
# est égale à Y(w0 + w1*x1 + w2*x2).
# On doit donc avoir d'après la table de vérité du XOR :
#   x1=0, x2=1 --> Y(w0 + w2) = 1 ==> w0 + w2 >= 0
#   x1=1, x2=0 --> Y(w0 + w1) = 1 ==> w0 + w1 >= 0
#   x1=0, x2=0 --> Y(w0) = 0 ==> w0 < 0
#   x1=1, x2=1 --> Y(w0 + w1 + w2) = 0 ==> w0 + w1 + w2 < 0
#
# Je vous laisse vérifier que ces 4 inéquations sont incompatibles. CQFD

# D.Mattei :
# 2W0 + W1 + W2 >= 0
# => W0 + W1 + W2 >= -W0

# w0 < 0 => -w0 > 0
# donc w0 + w1 + w2 >= 0
# contradiction la derniere inequation du dessus !








