#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graphic class to plot Perceptron
"""


from time import time

# for graphics with Tk
import tkinter
from math import *

from time import *

from logger import create_logger

import globals

LOG = create_logger()



# the click on mouse button is used to pause the Perceptron learning

def click(item):

    #global idle

    #print()
    LOG.debug("perceptronGraphic.py")
    #print("globals.idle= {}".format(globals.idle))
    print("Mouse button clicked !")
    
    if not globals.idle :
        globals.idle = True
    else :
        globals.idle = False

    
        
class toolboxGraphic():

    '''plot the perceptron.'''

    
    def __init__(self,window_title,unite=70,demo_mode=False):

        self.demo_mode = demo_mode
        self.root = tkinter.Tk()
        self.root.title("Learning Perceptron : "+window_title)
        print('unite={}'.format(unite))

        self.unite=unite


        # canvas size
        self.hauteur = 700 #500
        self.largeur = 900 #700
    
        # center O
        self.XO = self.largeur//2
        self.YO = self.hauteur//2

        # middle point of line
        self.middle_point = (0,0)
        self.old_middle_point = (0,0)

        # correction flag
        self.corrFlag = False

        # extremity of distance
        self.ed = (0,0)
        
        # extremity of normal
        self.xe = 0
        self.ye = 0
        
        self.Dessin = tkinter.Canvas(self.root,height=self.hauteur,width=self.largeur,bg='white')
        self.Dessin.pack()

        # draw axis
        self.Dessin.create_line((0,self.YO),(self.largeur,self.YO),fill='black')
        self.Dessin.create_line((self.XO,0),(self.XO,self.hauteur),fill='black')

        # drawing units on axis
        for i in range(-10,10):
            self.ligne(i,-5/unite,i,5/unite) # keep the mark of unit on axis independent of unit
            self.ligne(-5/unite,i,5/unite,i)

        # the click on mouse button is used to pause the Perceptron learning
        self.Dessin.bind('<Button-1>', click)
        
        self.root.update()
        
        #self.root.mainloop()
    
        
    def canvas_coord(self,x,y):
        cx = self.XO+x*self.unite
        cy = self.YO-y*self.unite
        LOG.debug(self.__class__.__name__)
        print(" x = {} , y = {} ".format(x,y))
        print(" cx = {} , cy = {} ".format(cx,cy))
        return cx,cy

    def point(self,x,y):
        (cx,cy)=self.canvas_coord(x,y)
        self.Dessin.create_rectangle(cx,cy,cx,cy,fill='black',outline='')


    def ligne(self,x1,y1,x2,y2,color='black',dash=None,arrow=None,width=1):
        (cx1,cy1)=self.canvas_coord(x1,y1)
        (cx2,cy2)=self.canvas_coord(x2,y2)
        return self.Dessin.create_line((cx1,cy1),(cx2,cy2),fill=color,dash=dash,arrow=arrow,width=width)

    #Draw an Oval in the canvas
    # r in pixels
    def disque(self,x,y,r): 
        (cx,cy)=self.canvas_coord(x,y)
        self.Dessin.create_oval(cx-r,cy-r,cx+r,cy+r,fill='black')

    def flash_disque(self,x,y,r,color):
        LOG.debug(self.__class__.__name__)
        print(" x = {} , y = {} ".format(x,y))
        (cx,cy)=self.canvas_coord(x,y)
        self.Dessin.create_oval(cx-r,cy-r,cx+r,cy+r,fill='yellow',tag="tempo")
        self.root.update()
        sleep(globals.tempo_flash)
        self.Dessin.delete("tempo")
        self.Dessin.create_oval(cx-r,cy-r,cx+r,cy+r,fill=color)
  


        
    def draw_vertical_line(self,x_abs):

        LOG.debug(self.__class__.__name__)

        self.Dessin.delete("separe")
        (cx,cy)=self.canvas_coord(x_abs,0)
        it=self.Dessin.create_line((cx,0),(cx,self.hauteur-1),tag="separe",fill='orange',width=2)
        self.Dessin.itemconfig(it, tag="separe")
        self.root.update()
        sleep(globals.tempo)
        

    def draw_horizontal_line(self,y_ord):

        LOG.debug(self.__class__.__name__)
        
        self.Dessin.delete("separe")
        (cx,cy)=self.canvas_coord(0,y_ord)
        it=self.Dessin.create_line((0,cy),(self.largeur-1,cy),tag="separe",fill='orange',width=2)
        self.Dessin.itemconfig(it, tag="separe")
     


    def draw_line(self,a,b):

        LOG.debug(self.__class__.__name__)
        print("y = {}*x + {}".format(a,b))
        x1=-self.XO//self.unite
        y1=a*x1+b
        x2=(self.largeur-self.XO)//self.unite
        y2=a*x2+b
        print("[(x1,y1),(x2,y2)] : [({},{}),({},{})]".format(x1,y1,x2,y2))
        self.old_middle_point = self.middle_point
        print("self.demo_mode=".format(self.demo_mode))
        # if not self.demo_mode :
        #     self.middle_point = self.compute_middle(x1,y1,x2,y2)
        
        self.Dessin.delete("separe")

        it=self.ligne(x1,y1,x2,y2,'orange',width=2)
        self.Dessin.itemconfig(it, tag="separe")
        

    def draw_cartesian_line(self,a,b,c): # ax + by + c, equation droite
        if a==0 :
            if b==0 :
                LOG.debug(self.__class__.__name__)
                if c==0 :
                    print("No line to plot because defined set is the whole cartesian plane !")
                else :
                    print("No line to plot because defined set is empty !")
            else :
                self.draw_horizontal_line(-c/b)
        elif b==0 :
            self.draw_vertical_line(-c/a)
        else :
            self.draw_line(-a/b,-c/b)


    def compute_middle(self,x1,y1,x2,y2):
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return (xm,ym)
    

    def draw_normal_vector_and_correction(self,cx,cy):
        
        LOG.debug(self.__class__.__name__)
        
        if self.corrFlag: # have we a normal to correct yet?

           
            # draw correction vector at the tip of old normal
            xec = self.xe + cx # extremite ancienne normale + correction
            yec = self.ye + cy
            it=self.ligne(self.xe,self.ye,xec,yec,'brown',arrow=tkinter.LAST)    
            self.Dessin.itemconfig(it, tag="correction_near_normal")
            
            print("draw correction vector at the tip of old normal")

            self.root.update()
            sleep(globals.tempo_normal)
            self.root.update()
            self.check_idle()

            (xd,yd) = self.ed
            
            # this draw what will be the new normal near the old one
            it=self.ligne(xd,yd,xec,yec,'Slategray3',arrow=tkinter.LAST,width=2)
            self.Dessin.itemconfig(it, tag="new_normal_near_old")
            self.root.update()
            print("draw what will be the new normal near the old one")
            sleep(globals.tempo_new_normal)
            self.root.update()
            print("globals.idle= {}".format(globals.idle))
            while globals.idle :
                sleep(1)
                self.root.update() # indispensable pour eviter de bloquer l'IHM
            
            self.Dessin.delete("correction_near_normal")
            self.Dessin.delete("new_normal_near_old")

        self.Dessin.delete("normal")
        self.Dessin.delete("distance")
        self.Dessin.delete("correction")
        self.Dessin.delete("anti_correction")


        
    def draw_normal(self,nx,ny):
        
        # draw new normal
        (xd,yd) = self.ed
        self.xe = xd + nx
        self.ye = yd + ny
        it=self.ligne(xd,yd,self.xe,self.ye,'red',arrow=tkinter.LAST,width=2)
        self.Dessin.itemconfig(it, tag="normal")
        print("draw new normal")
        self.root.update()
        self.corrFlag = True
        

    def draw_distance(self,distance,nx,ny) :

        LOG.debug(self.__class__.__name__)
        # distance extremity
        xd = nx * distance
        yd = ny * distance
        it=self.ligne(0,0,xd,yd,'DarkOrchid1',width=2)
        self.Dessin.itemconfig(it, tag="distance")
        print("draw distance")
        self.root.update()
        self.ed = (xd,yd)

        
    # the idle loop
    def check_idle(self):
        self.root.update()
        print("globals.idle= {}".format(globals.idle))
        while globals.idle :
            sleep(1)
            self.root.update() # indispensable pour eviter de bloquer l'IHM

    # draw correction vector from origin O
    def draw_correction(self,cx,cy):

        LOG.debug(self.__class__.__name__)
        print("cx = {} , cy = {}".format(cx,cy))
        
        it=self.ligne(0,0,cx,cy,'brown',arrow=tkinter.LAST)
        self.Dessin.itemconfig(it, tag="correction")
        
        self.root.update()
        sleep(3*globals.tempo_normal)
        self.Dessin.delete("data_line")
        self.root.update()

    # draw a dashed line showing data where correction belongs from
    def draw_anti_correction(self,cx,cy):

        LOG.debug(self.__class__.__name__)
        print("cx = {} , cy = {}".format(cx,cy))
        
        it=self.ligne(0,0,-cx,-cy,'brown',dash=(3, 3))
        self.Dessin.itemconfig(it, tag="anti_correction")
        self.root.update()
        sleep(2*globals.tempo_normal)
        

    # draw a dashed line showing element
    def draw_line_to_data(self,cx,cy):

        LOG.debug(self.__class__.__name__)
        print("cx = {} , cy = {}".format(cx,cy))
        
        it=self.ligne(0,0,cx,cy,'brown',dash=(3, 3))
        self.Dessin.itemconfig(it, tag="data_line")
        self.root.update()
        sleep(2*globals.tempo_normal)
