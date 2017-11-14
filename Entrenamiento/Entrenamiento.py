# -*- coding: utf-8 -*-
"""
Script entrenamiento para Starstim

Este archivo contiene dos tipos de entrenamiento del sistema. 
Tipo=1
Corresponde a las etapas de concentración y relajación
Tipo = 2
Corresponde a que la persona mueva una extremidad físicamente y además se imagine el movimiento.

@author: Dayán Mendez Vasquez,
         Jorge Silva Correa,
"""
import pygame
import random
import time 
import numpy as np
import scipy.io as sio
import winsound         # for sound  
from pylsl import StreamInlet, resolve_stream

class game(object):

    def __init__ (self, id_s = "unknown",rept = 1,tipo = 1, width = 800, height = 600, fps = 30):
        """Initialize pygame, window, background, font,...
        """
        pygame.init()
        pygame.display.set_caption("VIP: BCI")
        self.tipo = tipo 
        self.width = width
        self.height = height
        self.id_s = str(id_s)
        self.rept = int(rept)
        self.dimensions = (self.width, self.height)
        #self.screen = pygame.display.set_mode(self.dimensions, pygame.DOUBLEBUF)
        self.screen = pygame.display.set_mode(self.dimensions, pygame.FULLSCREEN)
        if (self.tipo == 2):
            self.imagenR= pygame.image.load("R1.png").convert()
            self.imagenL= pygame.image.load("R2.png").convert()
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.screen.fill((255,255,255))#Fondo blanco
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.playtime = 0.0
        self.font = pygame.font.SysFont('mono', 40, bold=True)
        size_screen= self.screen.get_size();
        self.x_center = size_screen[0]/2.0 - 210
        self.y_center = size_screen[1]/2.0 - 210
        
    def run(self):
        """The mainloop
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        print ("[!] Preparation stage started")
                        self.preparation()
                        print ("[!] Preparation stage Finished")
            milliseconds = self.clock.tick(self.fps)
            self.playtime += milliseconds / 1000.0
            self.draw_text("Entrenamiento sistema BCI")
            pygame.display.flip()
            self.screen.blit(self.background, (0, 0))
        pygame.quit()

    def draw_text(self, text, color = (100, 255, 100), dw = 0, dh = 0):
        """Center text in window"""
        fw, fh = self.font.size(text) # fw: font width,  fh: font height
        surface = self.font.render(text, True, color)
        # // makes integer division in python3
        self.screen.blit(surface, ((self.width - fw - dw) // 2, (self.height - dh) // 2))

    def  preparation(self):
        """Definition of the tasks of the training.
        """
        ntrial = 0
        t = 5 # Geting data time
        rs = 3 # Rest time 
        sound = 1 #Sound time 
        frequencySound = 800 # Frequency sound 
        timeOut = 3 #  time without getting data during the activity, at the beginning and the end
        if self.tipo == 1:
            datac = np.zeros((self.rept,t*500,8));#Concentration data
            datar = np.zeros((self.rept,t*500,8));#Relaxantion data                             
            while(ntrial < self.rept):
                self.rest(rs)
                winsound.Beep(frequencySound ,sound*1000)
                j1 = self.concentration(t,timeOut)
                #-------------------------
                self.rest(rs)

                winsound.Beep(frequencySound ,sound*1000)
                j2 = self.relaxation(t,timeOut)
                #-------------------------
                self.Loading()
                datac[ntrial]=j1
                datar[ntrial]=j2
                ntrial+=1     
            self.saveDataDB(self.id_s,self.tipo,datac,datar,0,0)

        elif self.tipo == 2:
            dataRI = np.zeros((self.rept,t*500,8)); #Right hand data - Mind 
 
            dataLI = np.zeros((self.rept,t*500,8)); #Left hand data- Mind
            while(ntrial < self.rept):
                self.rest(rs)
                winsound.Beep(frequencySound , sound*1000)
                j2 = self.rigthMind(t,timeOut)
                #-------------------------
                self.rest(rs)
                winsound.Beep(frequencySound , sound*1000)
                j4 = self.leftMind(t,timeOut)
                #------------------------
                self.Loading()
                dataRI[ntrial]=j2
                dataLI[ntrial]=j4
                ntrial+=1    
                
            self.saveDataDB(self.id_s,self.tipo,0,dataRI,0,dataLI)
            pass
            
    def rigthReal(self,t,timeOut):
        self.draw_text("Mueva",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(2)
        self.screen.blit(self.imagenR, [self.x_center, self.y_center])
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def rigthMind(self,t,timeOut):
        self.screen.blit(self.background, (0, 0))
        time.sleep(2)
        self.screen.blit(self.imagenR, [self.x_center, self.y_center])
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def leftReal(self,t,timeOut):
        self.draw_text("Mueva",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(2)
        self.screen.blit(self.imagenL, [self.x_center, self.y_center])
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def leftMind(self,t,timeOut):
        self.screen.blit(self.background, (0, 0))
        time.sleep(2)
        self.screen.blit(self.imagenL, [self.x_center, self.y_center])
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def concentration(self,t,timeOut):
        g=random.randint(100,200)
        self.draw_text(str(g),(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def relaxation(self,t,timeOut):
        self.draw_text("[+]",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d

    def rest(self,t): 
        self.draw_text("Descanse",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(t)
    
    def Loading(self):
        self.draw_text("Cargando...",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(1)

    def getDataO(self, tm):
        stream_name = 'NIC'
        streams = resolve_stream('type', 'EEG')
        fs = 500  # Frecuencia de muestreo
        N = fs * tm  # Numero de muestras
        c = 0;
        muestras = []
        try:
            for i in range(len(streams)):

                if streams[i].name() == stream_name:
                    index = i
                    print ("NIC stream available")

            print ("Connecting to NIC stream... \n")
            inlet = StreamInlet(streams[index])

        except NameError:
            print ("Error: NIC stream not available\n\n\n")
        data_time = np.zeros((N,8))
        while c < N:
            sample, timestamp = inlet.pull_sample()
            muestras.append(sample)
            c += 1

        # Diccionario con los datos de los electrodos
        data_time = np.array(muestras)       
        return data_time

    def saveDataDB(self,name,tipo,d1,d2,d3,d4):
        
        if tipo==1:
            datac = np.transpose(d1,(1,2,0))
            datar = np.transpose(d2,(1,2,0))
            sio.savemat(name+'.mat',{'conc':datac, 'rel':datar})
        elif tipo == 2:
           # dataRR = np.transpose(d1,(1,2,0))
            dataRI= np.transpose(d2,(1,2,0))
          #  dataLR= np.transpose(d3,(1,2,0))
            dataLI= np.transpose(d4,(1,2,0))
            sio.savemat(name+'.mat',{'izq':dataRI,'der':dataLI})
            pass
            
if __name__ == '__main__':
    #raw_input("[!] Digite la cantidad de pruebas a realizar: ")
    #raw_input("[!] Seleccione el tipo de entrenamiento[1=con-rela][2=real- imag]:")
    id_s = raw_input("[!] Digite el identificador del sujeto: ")
    rept = 30
    tipo = 1 # Concentración relajación.
    game(id_s,rept,int(tipo), 800, 600).run()


