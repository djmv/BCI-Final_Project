# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:08:35 2017

Este script entrena un clasificador SVM con el metodo del espectrograma 
Luego realiza predicciones con datos nuevos realimentados.  

@author: Dayán Mendez Vasquez,
         Jorge Silva Correa,
"""
from scipy import signal
import scipy.io as sio
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def get_data_db(id_s):
    """importa los datos del dataset(.mat).
    """
    mat_contents = sio.loadmat(id_s)
    conc = mat_contents['conc']
    rel = mat_contents['rel']
    dim=len(rel.shape)
    if (dim==3):
        if (len(conc)>len(conc[0][0])):
              conc = np.transpose(conc, (2, 1, 0))
              rel = np.transpose(rel, (2, 1, 0))
        data_time = np.zeros((len(conc)*2, len(conc[0]), len(conc[0][0])))
        data_time[0:len(conc)] = conc
        data_time[len(conc)::] = rel
        
    if (dim ==2):
        data_time = np.zeros((2,len(conc), len(conc[0])))
        data_time[0:len(conc)] = conc
        data_time[len(conc)::] = rel
    return data_time


def butter_filter(data, highcut=31, fqc=500, order=6):
    """Filtro pasabajas.
    """
    nyq = 0.5*fqc
    high = highcut/nyq
    [b_c, a_c] = signal.butter(order, high, btype='low')
    filt_sig = signal.filtfilt(b_c, a_c, data)
    return filt_sig

def remove_dc(data):
    """ Remueve el DC de una señal.
    """
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
                ndata[trial] = v_trial # guardamos señal sin DC
                
    return ndata, mean_v

def down_sampling(data, sc_v, div):
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

def artifact_regression(data, reference):
    """Remueve los artifacts de una señal usando regresión. """
    reference = np.transpose(np.matrix(reference))
    data = np.transpose(np.matrix(data))
    op1 = (np.transpose(reference)*reference)/int(reference.shape[0]-1)
    op2 = (np.transpose(reference)*data)/int(data.shape[0]-1)
    coeff, _, _, _ = np.linalg.lstsq(op1, op2)
    data = data - reference*coeff
    data = np.transpose(data)
    return data

def car(data):
    """Referencia los datos teniendo el cuenta el promedio de todos los electrodos,
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

"""Codigo principal - Entrenamiento espectrograma y prueba de predicción """
"Main Code"
plt.close('all')
S_ID = "S01" # Nombre del archivo del sujeto
DATA = get_data_db(S_ID) # Obtiene los datos del .mat
Datar = car(DATA) # Referencia la señal 
[D_DC, m_v] = remove_dc(Datar) # Datos sin DC
Y = butter_filter(D_DC) #Filtra la señal a 30 Hz
SCALE = 1
FS = 500/SCALE # esto es porque fue submuestreado 
DIV = 500.0/SCALE
Electrode = ['AF3','AF4','F3','F4','C3','C4','C1','C2'];

TS = 1.0/FS
Wsize = round(0.7*FS)
Wnover = round(0.6*FS)
Windw = signal.hamming(Wsize)
TIME = np.arange(0, len(D_DC[0][0]) * TS, TS)
F, T, S = signal.spectrogram(Y, fs=FS, window=Windw,nperseg=Wsize, noverlap=Wnover)

frang = ((F>=8)*(F<=13))# Rango de análisis alfa,
#Si se desea agregar Beta solo se debe cambiar el rango ((F>=16)*(F<=31))
S_prom = np.mean(S[:,:,frang,:], axis=3) # Promedio en el tiempo para cada frecuencia
S_shape = np.shape(S_prom)
M_f = S_prom # Potencia promedio para cada frecuencia
Feats_a = np.reshape(M_f, (S_shape[0], M_f.shape[2]*S_shape[1])) 
# Se crea la matriz caracteristica 

LABELS = np.zeros((Y.shape[0])) # se crean las etiquetas para cada clase
LABELS[0:len(LABELS)/2] = 1
LABELS[len(LABELS)/2::] = 2
# Se crea un clasificador SVM 
# Se entrena con las caracteristicas y las etiquetas.
clf = svm.SVC(kernel='linear', C=1).fit(Feats_a, LABELS) 

"""Procesamiento datos de realimentación"""

S_ID2 = "S01_R" # Nombre archivo realimentación
DATA = get_data_db(S_ID2)
Datar = car(DATA)
[D_DC, m_v] = remove_dc(Datar) # Datos sin DC
Y = butter_filter(D_DC)
SUB_SIGNAL = Y

TS = 1.0/FS
Wsize = int(0.7*FS)
Wnover = int(0.6*FS)
Windw = signal.hamming(Wsize)
TIME = np.arange(0, len(D_DC[0][0]) * TS, TS)
F, T, S = signal.spectrogram(Y, fs=FS, window=Windw,nperseg=Wsize, noverlap=Wnover)      
S_prom = np.mean(S[:,:,frang,:], axis=3)
S_shape = np.shape(S_prom)
M_f = S_prom # Potencia promedio para cada frecuencia
Feats_a = np.reshape(M_f, (S_shape[0], M_f.shape[2]*S_shape[1]))
test_feats = Feats_a

y_pred = clf.predict(test_feats) # Predicción sobre datos de realimentación

y_true = LABELS # Etiquetas de la acción realizada en las pruebas de realimentación
precision = accuracy_score(y_true, y_pred) # Precisión del clasificador.
print precision 
conf = confusion_matrix(LABELS, y_pred) # Matriz de confusión
print conf # Matriz de confusión
