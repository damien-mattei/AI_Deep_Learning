#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exo_mat2.py
La classe Mat : algèbre des matrices de format quelconque, sans numpy
"""

class MatError(Exception):     # juste pour la lisibilité des exceptions
    pass

class Mat:
    def __init__(self,Lf,*args):
        '''Construit un objet matrice de type Mat, d'attributs le format self.dim
        et la liste architecturée en liste de listes de même longueur. Exemples :
            m = Mat([[1,3],[-2,4],[0,-1]])  à 3 lignes et 2 colonnes
            m = Mat(lambda i,j: i+j,3,5)    à 3 lignes et 5 colonnes'''
        if type(Lf) == list:    # on attend une liste de listes de même longueur
            if any(map(lambda x:type(x) != list,Lf)) :
                raise MatError('Mat : on attend une liste de listes !')
            p = len(Lf[0])
            if any(map(lambda x:len(x)!=p,Lf)) :
                raise MatError('Mat : on attend une liste de listes de même longueur !')
            self.L = Lf         # la liste qui contient les éléments de matrice
        elif callable(Lf):      # Lf est une fonction
            if len(args) != 2:
                raise MatError('Utilisez Mat(f,n,p) ?')
            (n,p) = args
            self.L = [[Lf(i,j) for j in range(p)] for i in range(n)]
        else:
            raise MatError('Mauvaise utilisation de la classe Mat')

    def dim(self):
        '''Retourne le format de la matrice courante.'''
        n = len(self.L)
        if n == 0:
            raise MatError('Matrice vide !')
        return (n,len(self.L[0]))

    def __repr__(self):
        '''Retourne une chaine formatée avec colonnes alignées représentant
        la matrice m.'''
        def texte(x):
            return '{:10.2f}'.format(x)   # précision limitée à l'affichage...
        (n,p) = self.dim()
        L = ['\t'.join(list(map(texte,self.L[i]))) for i in range(n)]
        return '\n'.join(L)

    def __add__(self,m2):
        '''Retourne la somme de la matrice courante et d'une matrice m2
        de même format.'''
        (n,p) = self.dim()
        if m2.dim() != (n,p):
            raise MatError('mat_sum : Mauvais formats de matrices !')
        L = self.L ; L2 = m2.L
        return Mat(lambda i,j : L[i][j] + L2[i][j],n,p)

    def __sub__(self,m2):
        '''Retourne la différence entre la matrice courante et une matrice
        m2 de même format.'''
        return self + m2.mul(-1)

    def mul(self,k):
        '''Retourne le produit externe du nombre k par la matrice m.'''
        (n,p) = self.dim()
        return Mat(lambda i,j : k*self.L[i][j],n,p)

    def __mul__(self,m2):
        (n1,p1) = self.dim()
        (n2,p2) = m2.dim()
        if p1 != n2 : raise MatError('Produit de matrices impossible !')
        def res(i,j) :              # l'élément en ligne i et colonne j du résultat
            return sum(self.L[i][k] * m2.L[k][j] for k in range(p1))
        # le produit aura pour format (n1,p2)
        return Mat(res,n1,p2)

    def __getitem__(self,i):        # pour pouvoir écrire m[i] pour la ligne i
        return self.L[i]            # et m[i][j] pour l'élément en ligne i et colonne j

    def lig(self,i):                # m.lig(i) <==> m[i]
        '''Retourne la ligne i >= 0 de la matrice sous forme de liste plate.'''
        return self.L[i]

    def col(self,j):
        '''Retourne la colonne j >= 0 de la matrice sous forme de liste plate.'''
        (n,_) = self.dim()
        return [self.L[i][j] for i in range(n)]

    def app(self,v):                           # v = [a,b,c,d]
        '''Retourne l'application de la matrice self au vecteur v vu comme une liste
        plate. Le résultat est aussi une liste plate.'''
        # transformation de la liste v en matrice uni-colonne
        mv = Mat(list(map(lambda x:[x],v)))          # mv = [[a],[b],[c],[d]]
        # l'application n'est autre qu'un produit de matrices
        res = self * mv         # objet de type Mat
        res = res.L             # objet de type list
        # et on ré-aplatit la liste
        return list(map(lambda L:L[0],res))

    def __eq__(self,m2):
        '''Retourne True si les matrices ont des éléments égaux
        au sens de =='''
        (n,p) = self.dim()
        if m2.dim() != self.dim(): return False
        for i in range(n):
            for j in range(p):
                if self[i][j] != m2[i][j]: return False
        return True

if __name__ == '__main__':
    def test(str):
        print('>>>',str)
        print(eval(str),'\n')

    h = Mat(lambda i,j: i+j, 3, 3)
    ZERO3 = Mat(lambda i,j: 0, 3, 3)
    test('h')
    test('h.dim()')
    test('h[1][2]')
    test('h.lig(1)')
    test('h.col(2)')
    test('h - h.mul(2)')
    test('h - h')
    test('h + h - h.mul(2) == ZERO3')
    test('h * h')
    test('h.app([1,2,3])')



