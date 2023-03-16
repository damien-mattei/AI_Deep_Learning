#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MatrixNumPy.py
The class MatrixNumPy.
Derived from:
exo_mat2.py
La classe MatrixNumPy : algèbre des matrices de format quelconque, avec numpy
"""

# D. Mattei

from multimethod import multimethod

from typing import Union,Callable

from collections.abc import Iterable

import numpy


class MatError(Exception):     # juste pour la lisibilité des exceptions
    pass


Numeric = Union[float, int]



class MatrixNumPy:
    '''Construct an object MatrixNumPy.'''

    # >>> m1=MatrixNumPy(2,3)
    @multimethod
    def __init__(self,n : Numeric,p : Numeric): # lines, columns
        '''Construit un objet matrice de type MatrixNumPy, d'attributs le format self.dim
        et le tableau architecturé en liste de listes de même longueur. Exemples :
            m = MatrixNumPy([[1,3],[-2,4],[0,-1]])  à 3 lignes et 2 colonnes
            m = MatrixNumPy(lambda i,j: i+j,3,5)    à 3 lignes et 5 colonnes'''

        if __debug__:
            print("# MatrixNumPy constructor MatrixNumPy (Numeric,Numeric) #")

        __init__(lambda i,j: 0,n,p) # return a Zero matrix


   
    @multimethod
    def __init__(self,f : Callable,n : Numeric,p : Numeric):

        if __debug__:
            print("# MatrixNumPy constructor MatrixNumPy (function,Numeric,Numeric) #")

        self.A = numpy.array([[f(i,j) for j in range(p)] for i in range(n)])

    
    @multimethod
    def __init__(self,Af : list):  # la liste qui contient les éléments de matrice

        if __debug__:
            print("# MatrixNumPy constructor MatrixNumPy,list #")

        if any(map(lambda x:type(x) != list,Af)) :
            raise MatError('MatrixNumPy : on attend une liste de listes !')
        p = len(Af[0])
        if any(map(lambda x:len(x)!=p,Af)) :
            raise MatError('MatrixNumPy : on attend une liste de listes de même longueur !')
        self.A = numpy.array(Af)         # l'array qui contient les éléments de matrice
        


    @multimethod
    def __init__(self,Arr : numpy.ndarray):

        if __debug__:
            print("# MatrixNumPy constructor MatrixNumPy,numpy.ndarray #")

        self.A = Arr

        

    def dim(self):
        '''Retourne le format de la matrice courante.'''

        return self.A.shape

    

    # m1=MatrixNumPy(lambda i,j : i+j, 5,2)
    # # MatrixNumPy constructor MatrixNumPy (function,Numeric,Numeric) #
    # m1
    #       0.00	      1.00
    #       1.00	      2.00
    #       2.00	      3.00
    #       3.00	      4.00
    #       4.00	      5.00
    # MatrixNumPy @ 0x105ae03d0 

    # print(m1)
    #       0.00	      1.00
    #       1.00	      2.00
    #       2.00	      3.00
    #       3.00	      4.00
    #       4.00	      5.00
    def __repr__(self):
        '''Retourne une chaine formatée avec colonnes alignées représentant
        la matrice m.'''
        
        return self.__str__() + '\nMatrixNumPy @ {} \n'.format(hex(id(self)))
        


    # >>> print(m)
    def __str__(self):

        '''Retourne une chaine formatée avec colonnes alignées représentant
        la matrice m.'''

        return self.A.__str__()

    

    def __getitem__(self,i):        # pour pouvoir écrire m[i] pour la ligne i
        return self.A[i]            # et m[i][j] pour l'élément en ligne i et colonne j

    def lig(self,i):                # m.lig(i) <==> m[i]
        '''Retourne la ligne i >= 0 de la matrice sous forme de liste plate.'''
        return self.A[i].tolist()

    def col(self,j):
        '''Retourne la colonne j >= 0 de la matrice sous forme de liste plate.'''
        (n,_) = self.dim()
        return [self.A[i][j] for i in range(n)]

    
    def __add__(self,m2):
        '''Retourne la somme de la matrice courante et d'une matrice m2
        de même format.'''
        (n,p) = self.dim()
        if m2.dim() != (n,p):
            raise MatError('mat_sum : Mauvais formats de matrices !')
        A = self.A ; A2 = m2.A
        AplusA2 = numpy.add(A,A2)
        return MatrixNumPy(AplusA2)
    

    def __sub__(self,m2):
        '''Retourne la différence entre la matrice courante et une matrice
        m2 de même format.'''
        return MatrixNumPy(numpy.substract(self.A,m2.A))


    def mul(self,k):
        '''Retourne le produit externe du nombre k par la matrice m.'''
        (n,p) = self.dim()
        return MatrixNumPy(lambda i,j : k*self.A[i][j],n,p)
    

    # R  : multiplicand
    # matrix multiplication by number
    
    @multimethod
    def __rmul__(self, m : Numeric): #  self is at RIGHT of multiplication operand : m * self
        '''Retourne le produit externe du nombre par la matrice'''
        if __debug__:
            print("MatrixNumPy.py : __rmul__(MatrixNumPy,Numeric)")

        return self.mul(m)
        
    
    def app(self,v):                           # v = [a,b,c,d]
        '''Retourne l'application de la matrice self au vecteur v vu comme une liste
        plate. Le résultat est aussi une liste plate.'''
        # transformation de la liste v en matrice uni-colonne
        mv = MatrixNumPy(list(map(lambda x:[x],v)))          # mv = [[a],[b],[c],[d]]
        # l'application n'est autre qu'un produit de matrices
        res = self * mv         # objet de type MatrixNumPy car produit de 2 matrices
        res = res.A             # objet de type Array
        # et on ré-aplatit la liste
        return list(map(lambda A:A[0],res))


    
    # R  : multiplicand
    # m1=MatrixNumPy(lambda i,j : i+j, 5,2)
    # # MatrixNumPy constructor MatrixNumPy (function,Numeric,Numeric) #
    # m1*(-2,-3.5)
    # MatrixNumPy.py : __mul__(MatrixNumPy,Iterable)
    # # MatrixNumPy constructor MatrixNumPy,list #
    # MatrixNumPy.py : __mul__(MatrixNumPy,MatrixNumPy)
    # # MatrixNumPy constructor MatrixNumPy (function,Numeric,Numeric) #
    # [-3.5, -9.0, -14.5, -20.0, -25.5]
    @multimethod
    def __mul__(self, R : Iterable): #  self is at LEFT of multiplication operand : self * R = MatrixNumPy * R, R is at Right

        if __debug__:
            print("MatrixNumPy.py : __mul__(MatrixNumPy,Iterable)")

        return self.app(R)
            

    
    # R  : multiplicand
    # matrix multiplication
    # m2=MatrixNumPy([[-2],[-3.5]])
    
    # m1*m2
    #     >>> m2
    # [[-2. ]
    #  [-3.5]]
    # MatrixNumPy @ 0x7f48a430ee10 

    # >>> m1*m2
    # MatrixNumPy.py : __mul__(MatrixNumPy,MatrixNumPy)
    # # MatrixNumPy constructor MatrixNumPy,numpy.ndarray #
    # [[ -3.5]
    #  [ -9. ]
    #  [-14.5]
    #  [-20. ]
    #  [-25.5]]
    #MatrixNumPy @ 0x7f48a4362590 
    @multimethod
    def __mul__(self, m2 : object): #  self is at LEFT of multiplication operand : self * m2 = MatrixNumPy * m2 = MatrixNumPy * MatrixNumPy, m2 is at Right of operator

        if __debug__:
            print("MatrixNumPy.py : __mul__(MatrixNumPy,MatrixNumPy)")

        (n1,p1) = self.dim()
        (n2,p2) = m2.dim()
        if p1 != n2 : raise MatError('Produit de matrices impossible !')
      
        # le produit aura pour format (n1,p2)
        return MatrixNumPy(numpy.matmul(self.A,m2.A))

        

        
