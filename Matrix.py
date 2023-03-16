#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix.py
The class Matrix.
Derived from:
exo_mat2.py
La classe Matrix : algèbre des matrices de format quelconque, sans numpy
"""


from multimethod import multimethod

from typing import Union,Callable

from collections.abc import Iterable


class MatError(Exception):     # juste pour la lisibilité des exceptions
    pass


Numeric = Union[float, int]



class Matrix:
    '''Construct an object Matrix.'''

    # >>> m1=Matrix(2,3)
    @multimethod
    def __init__(self,n : Numeric,p : Numeric): # lines, columns
        '''Construit un objet matrice de type Matrix, d'attributs le format self.dim
        et la liste architecturée en liste de listes de même longueur. Exemples :
            m = Matrix([[1,3],[-2,4],[0,-1]])  à 3 lignes et 2 colonnes
            m = Matrix(lambda i,j: i+j,3,5)    à 3 lignes et 5 colonnes'''

        if __debug__:
            print("# Matrix constructor Matrix (Numeric,Numeric) #")

        __init__(lambda i,j: 0,n,p) # return a Zero matrix


   
    @multimethod
    def __init__(self,f : Callable,n : Numeric,p : Numeric):

        if __debug__:
            print("# Matrix constructor Matrix (function,Numeric,Numeric) #")

        self.L = [[f(i,j) for j in range(p)] for i in range(n)]

    
    @multimethod
    def __init__(self,Lf : list):

        if __debug__:
            print("# Matrix constructor Matrix,list #")

        if any(map(lambda x:type(x) != list,Lf)) :
            raise MatError('Matrix : on attend une liste de listes !')
        p = len(Lf[0])
        if any(map(lambda x:len(x)!=p,Lf)) :
            raise MatError('Matrix : on attend une liste de listes de même longueur !')
        self.L = Lf         # la liste qui contient les éléments de matrice
        


    def dim(self):
        '''Retourne le format de la matrice courante.'''
        n = len(self.L)
        if n == 0:
            raise MatError('Matrice vide !')
        return (n,len(self.L[0]))

    

    # m1=Matrix(lambda i,j : i+j, 5,2)
    # # Matrix constructor Matrix (function,Numeric,Numeric) #
    # m1
    #       0.00	      1.00
    #       1.00	      2.00
    #       2.00	      3.00
    #       3.00	      4.00
    #       4.00	      5.00
    # Matrix @ 0x105ae03d0 

    # print(m1)
    #       0.00	      1.00
    #       1.00	      2.00
    #       2.00	      3.00
    #       3.00	      4.00
    #       4.00	      5.00
    def __repr__(self):
        '''Retourne une chaine formatée avec colonnes alignées représentant
        la matrice m.'''

        return self.__str__() + '\nMatrix @ {} \n'.format(hex(id(self)))
        


    # >>> print(m)
    def __str__(self):

        '''Retourne une chaine formatée avec colonnes alignées représentant
        la matrice m.'''

        def texte(x):
            return '{:10.2f}'.format(x)   # précision limitée à l'affichage...
            
        (n,p) = self.dim()
        L = ['\t'.join(list(map(texte,self.L[i]))) for i in range(n)]
        
        return '\n'.join(L)

    

    def __getitem__(self,i):        # pour pouvoir écrire m[i] pour la ligne i
        return self.L[i]            # et m[i][j] pour l'élément en ligne i et colonne j

    def lig(self,i):                # m.lig(i) <==> m[i]
        '''Retourne la ligne i >= 0 de la matrice sous forme de liste plate.'''
        return self.L[i]

    def col(self,j):
        '''Retourne la colonne j >= 0 de la matrice sous forme de liste plate.'''
        (n,_) = self.dim()
        return [self.L[i][j] for i in range(n)]

    
    def __add__(self,m2):
        '''Retourne la somme de la matrice courante et d'une matrice m2
        de même format.'''
        (n,p) = self.dim()
        if m2.dim() != (n,p):
            raise MatError('mat_sum : Mauvais formats de matrices !')
        L = self.L ; L2 = m2.L
        return Matrix(lambda i,j : L[i][j] + L2[i][j],n,p)

    def __sub__(self,m2):
        '''Retourne la différence entre la matrice courante et une matrice
        m2 de même format.'''
        return self + m2.mul(-1)


    def mul(self,k):
        '''Retourne le produit externe du nombre k par la matrice m.'''
        (n,p) = self.dim()
        return Matrix(lambda i,j : k*self.L[i][j],n,p)
    

    # R  : multiplicand
    # matrix multiplication by number
    
    @multimethod
    def __rmul__(self, m : Numeric): #  self is at RIGHT of multiplication operand : m * self
        '''Retourne le produit externe du nombre par la matrice'''
        if __debug__:
            print("Matrix.py : __rmul__(Matrix,Numeric)")

        return self.mul(m)
        
    
    def app(self,v):                           # v = [a,b,c,d]
        '''Retourne l'application de la matrice self au vecteur v vu comme une liste
        plate. Le résultat est aussi une liste plate.'''
        # transformation de la liste v en matrice uni-colonne
        mv = Matrix(list(map(lambda x:[x],v)))          # mv = [[a],[b],[c],[d]]
        # l'application n'est autre qu'un produit de matrices
        res = self * mv         # objet de type Mat
        #print("app : res =\n"); print(res)
        res = res.L             # objet de type list
        # et on ré-aplatit la liste
        return list(map(lambda L:L[0],res))


    
    # R  : multiplicand
    # m1=Matrix(lambda i,j : i+j, 5,2)
    # # Matrix constructor Matrix (function,Numeric,Numeric) #
    # m1*(-2,-3.5)
    # Matrix.py : __mul__(Matrix,Iterable)
    # # Matrix constructor Matrix,list #
    # Matrix.py : __mul__(Matrix,Matrix)
    # # Matrix constructor Matrix (function,Numeric,Numeric) #
    # [-3.5, -9.0, -14.5, -20.0, -25.5]
    @multimethod
    def __mul__(self, R : Iterable): #  self is at LEFT of multiplication operand : self * R = Matrix * R, R is at Right

        if __debug__:
            print("Matrix.py : __mul__(Matrix,Iterable)")

        return self.app(R)
            

    
    # R  : multiplicand
    # matrix multiplication
    # m2=Matrix([[-2],[-3.5]])
    # # Matrix constructor Matrix,list #
    # m2
    #      -2.00
    #      -3.50
    # Matrix @ 0x10127f490 

    # m1*m2
    # Matrix.py : __mul__(Matrix,Matrix)
    # # Matrix constructor Matrix (function,Numeric,Numeric) #
    #      -3.50
    #      -9.00
    #     -14.50
    #     -20.00
    #     -25.50
    # Matrix @ 0x1012a7810
    @multimethod
    def __mul__(self, m2 : object): #  self is at LEFT of multiplication operand : self * m2 = Matrix * m2 = Matrix * Matrix, m2 is at Right of operator

        if __debug__:
            print("Matrix.py : __mul__(Matrix,Matrix)")

        (n1,p1) = self.dim()
        (n2,p2) = m2.dim()
        if p1 != n2 : raise MatError('Produit de matrices impossible !')
        def res(i,j) :              # l'élément en ligne i et colonne j du résultat
            return sum(self.L[i][k] * m2.L[k][j] for k in range(p1))
        # le produit aura pour format (n1,p2)
        return Matrix(res,n1,p2)

        

        
