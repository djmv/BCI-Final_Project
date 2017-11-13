# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:32:42 2017

Este script corresponde a la prueba de realimentación voluntaria.
Aquí el sujeto escoge que actividad desea realizar y se ejecuta en la pantalla.

@author: Dayán Mendez Vasquez,
         Jorge Silva Correa,
"""

import pygame
import time 
from scipy import signal
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from pylsl import StreamInlet, resolve_stream
import random
import winsound
from scipy.signal import hilbert

# Definimos algunos colores
VERDE   = (0, 255, 150)
#-------------------------------------------------------------------------------------

class game(object):

    def __init__ (self,ID="unknown", width=800, height=600, fps=60):
        """Initialize pygame, window, background, font,..."""
        pygame.init()
        pygame.display.set_caption("VIP: BCI")
        self.width = width
        self.height = height
        self.dimensions = (self.width, self.height)
        #self.screen = pygame.display.set_mode(self.dimensions, pygame.DOUBLEBUF)
        self.screen = pygame.display.set_mode(self.dimensions, pygame.FULLSCREEN)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('mono', 40, bold=True)
        self.fps = fps
        self.playtime = 0.0
        self.r=5
        self.ID=ID
        self.rept = 10
        
    def run(self):
        """The mainloop"""
        #Entrenamiento del clasificador
        [data_ext, label]= self.getDataExt(self.ID)
        feats = self.processing(data_ext)
        """
        Clasificación
        """
        # Se crea el clasificador
        clf = svm.SVC(kernel='linear', C=1).fit(feats, label)
        #Se entrena el clasificador con las mejores características
        #Seleccionar los indicies de las características importantes -----------
        ntrial = 0
        #----------------------------------------------------------------------
        hecho = False
        #----------------------------------------------------------------------
        self.draw_text("BCI Game")
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(1)
        #----------------------------------------------------------------------
        self.draw_text("Instrucciones ",VERDE,  0, 300)
        self.draw_text("-Solo usa tu mente-",VERDE, 0 ,100)
        self.draw_text(".:COMIENZA:.",VERDE, 0 ,0)
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(1)
        #----------------------------------------------------------------------
        while not hecho:
            # --- Bucle principal de eventos            
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT: 
                    print("Se presionó el boton cerrar")
                    hecho = True
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_ESCAPE:
                        hecho = True
                        print("Se presionó la techa escape")
                    if evento.key == pygame.K_s:
                        print ("[!] Preparation stage started")
                        #----------------------------------------       
                        ntrial = 0
                        t = 5; #tiempo de muestra
                        datac = np.zeros((self.rept,8,t*500));
                        datar = np.zeros((self.rept,8,t*500));
                        labels1 = np.zeros((self.rept))
                        labels2 = np.zeros((self.rept))
                        while (ntrial < self.rept):
                            dtTem1 = self.concentration(5,3)
                            FEATS1 = self.processing(dtTem1)
                            lab1 = 0
                            #--------------------------------------------------------------------
                            winsound.Beep(800,1000)
                            cls=clf.predict(FEATS1)
                            print str(cls) +"tem1"
                            if (cls == 1):
                                self.good()
                                time.sleep(2)
                                lab1 = 1
                            elif (cls == 2):
                                self.bad()
                                time.sleep(2)
                                
                            self.rest(3)
                            #----------------------------------------
                            dtTem2 = self.relaxation(5,3)
                            FEATS2 = self.processing(dtTem2)
                            #--------------------------------------------------------------------
                            lab2=0
                            cls=clf.predict(FEATS2)
                
                            winsound.Beep(800,1000)
                            if (cls == 1):
                                self.bad()
                                time.sleep(2)
                                
                            elif (cls == 2):
                                self.good()
                                time.sleep(2)
                                lab2=1
                                
                            datac[ntrial]=dtTem1
                            datar[ntrial]=dtTem2
                            labels1[ntrial] = lab1
                            labels2[ntrial] = lab2
                            ntrial+=1
                        self.saveDataDB(self.ID+str(ntrial),datac,datar,labels1,labels2)
            #            pygame.draw.rect(self.screen,VERDE, [x,y, 100, 100])
            #            pygame.display.flip()
            #            self.screen.blit(self.background, (0, 0))
                        milliseconds = self.clock.tick(self.fps)
                        self.playtime += milliseconds / 1000.0
                        hecho = True
                        print ("[!] Preparation stage Finished")
        pygame.quit()
         
    def draw_text(self, text, color = VERDE, dw = 0, dh = 0):
        """Center text in window"""
        fw, fh = self.font.size(text) # fw: font width,  fh: font height
        surface = self.font.render(text, True, color)
        # // makes integer division in python3
        self.screen.blit(surface, ((self.width - fw - dw) // 2, (self.height - dh) // 2))
        
    def concentration(self,t,timeOut):
        g=random.randint(100,200)
        self.draw_text(str(g),(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def good(self):
        self.draw_text("Bien :) ",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))

    def bad(self):
        self.draw_text("Mal :( ",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
   
    def relaxation(self,t,timeOut):
        self.draw_text("[+]",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(timeOut)
        d = self.getDataO(t)
        return d
    
    def saveDataDB(self,name,d1,d2,label1,label2):        
        datac = np.transpose(d1,(1,2,0))
        datar = np.transpose(d2,(1,2,0))
        sio.savemat(name+'ret3.mat',{'conc':datac,'rel':datar})
        sio.savemat(name+'label3.mat',{'conc':datac, 'rel':datar,'lab1':label1,'lab2':label2})
        
    def rest(self,t): 
        self.draw_text("Descanse",(100,255,100))
        pygame.display.flip()
        self.screen.blit(self.background, (0, 0))
        time.sleep(t)
    
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
        data_time = np.transpose(data_time,(1,0))
        """
        data_time = np.random.random((8,tm*500))
        #data_time = np.transpose(data_time,(1,0))
        """
        return data_time
    
    def getDataExt(self,id_s):
        """importa los datos del dataset.--------------------------------------
        """
        mat_contents = sio.loadmat(id_s)
        conc = mat_contents['conc']
        rel = mat_contents['rel']
        dim=len(rel.shape)
        if (dim==3):
            conc = np.transpose(conc, (2, 1, 0))
            rel = np.transpose(rel, (2, 1, 0))
            data_time = np.zeros((len(conc)*2, len(conc[0]), len(conc[0][0])))
            data_time[0:len(conc)] = conc
            data_time[len(conc)::] = rel
            
        if (dim ==2):
            data_time = np.zeros((2,len(conc), len(conc[0])))
            data_time[0:len(conc)] = conc
            data_time[len(conc)::] = rel
        
        LABELS = np.zeros((data_time.shape[0]))
        LABELS[0:len(LABELS)/2] = 1
        LABELS[len(LABELS)/2::] = 2
        return data_time, LABELS

    def processing(self,dataTime):
        Datar = self.car(dataTime)
        [D_DC, m_v] = self.remove_dc(Datar) # Datos sin DC
        Y = self.butter_filter(D_DC) # Se filtra en 8 y 12 Hz (Alfa)
        if (len(Y.shape) == 3):
            Y = Y[:,:,100::] #Se eliminan 0,2 s por transistorio
        if (len(Y.shape)==2):
            Y = Y[:,100::]
        SCALE = 1
        DIV = 500.0/SCALE
        SUB_SIGNAL = self.down_sampling(Y, int(SCALE), DIV) # Sub muestreo
        sigan = hilbert(SUB_SIGNAL)
        signalhilt = np.abs(sigan) #Envolvente de la señal 
        nyq = 0.5*500
        [b_c, a_c] = signal.butter(4, [2/nyq], btype='low') #Se filtra a 2 Hz
        sighilt = signal.lfilter(b_c, a_c, signalhilt)
        if (len(Y.shape) == 3):
            sighiltm1 = np.mean(sighilt[:,:,500::], axis=2) # Valor promedio por electrodo 
        if (len(Y.shape)==2):
            sighiltm1 = np.mean(sighilt[:,500::], axis=1)
        
        ff = sighiltm1.shape # Tamaño del array
        dimen = len(ff)
        if (dimen == 3):
            M_F = sighiltm1 # Potencia promedio para cada frecuencia
            FEATS = np.reshape(M_F, (ff[0], M_F.shape[2]*ff[1]))
        elif (dimen == 2):
            M_F = sighiltm1 # Potencia promedio para cada frecuencia
            FEATS = M_F
        elif (dimen == 1):
            M_F = np.zeros((1,len(sighiltm1)))
            M_F[0,:] = sighiltm1.T # Potencia promedio para cada frecuencia
            print M_F
            FEATS = M_F
        return FEATS
        
    def butter_filter(self,data,lowcut = 8, highcut=13, fqc=500, order=6):
        """Filtro pasabajas.
        """
        nyq = 0.5*fqc
        high = highcut/nyq
        low = lowcut/nyq;
        [b_c, a_c] = signal.butter(order, [low, high], btype='band')
        filt_sig = signal.lfilter(b_c, a_c, data)
        return filt_sig
    
    def remove_dc(self,data):
        """ Remueve el DC de una señal."""
        
        dim = len(data.shape)
        ndata = np.zeros(np.shape(data))
        if (dim == 3):
            mean_v = np.mean(data, axis=2)
            for trial in range(0, len(data)):
                for electrode in range(0, len(data[trial])):
                    # Señal original -  señal DC
                    v_trial = (data[trial][electrode] - mean_v[trial][electrode])
                    ndata[trial][electrode] = v_trial # guardamos señal sin DC
        
        elif (dim == 2):
            mean_v = np.mean(data, axis=1)
            for electrode in range(0, len(data)):
                    # Señal original -  señal DC
                    v_trial = (data[electrode] - mean_v[electrode])
                    ndata[electrode] = v_trial # guardamos señal sin DC
        return ndata, mean_v
     
    def down_sampling(self,data, sc_v, div):
        """Reduce la frecuencia de muestreo de una señal.
        """
        dim = len(data.shape)
        if (dim == 3):
            if ((div % 2) != 0):
                sub_signals = np.zeros((len(data), len(data[0]), len(data[0][0])/sc_v+1))
            else:
                sub_signals = np.zeros((len(data), len(data[0]), len(data[0][0])/sc_v))
        
            for trial in range(0, len(data)):
                for electrode in range(0, len(data[trial])):
                    sub_signals[trial][electrode] = data[trial][electrode][::sc_v]
        
        elif (dim == 2):    
            if ((div % 2) != 0):
                sub_signals = np.zeros((len(data), len(data[0])/sc_v+1))
            else:
                sub_signals = np.zeros((len(data), len(data[0])/sc_v))
            for number in range(0, len(data)):
                    sub_signals[number] = data[number][::sc_v]
        return sub_signals
    
    def modelInd(self, S_ID):
        indm = np.load('ind'+S_ID+'.npy')
        return indm

    def car(self,data):
        """Re-referencia los datos teniendo el cuenta el promedio de todos los electrodos,
        Se resta el promedio común a cada uno.
        """
        dim = len(data.shape)
        ndata = np.zeros(np.shape(data))
        if (dim==3):
            for trial in range(0,len(data)):
                mean_v = np.mean(data[trial],axis=0)
                dtrial = data[trial] - mean_v
                ndata[trial] = dtrial
        if (dim == 2):
            mean_v = np.mean(data,0)
            ndata = data - mean_v
        return ndata

if __name__ == '__main__':
    # call with width of window and fps
    #S_ID = raw_input("[!] Digite el identificador del sujeto: ")
    S_ID="edgard1909" # Nombre de la prueba de entrenamiento
    game(S_ID).run()
    
