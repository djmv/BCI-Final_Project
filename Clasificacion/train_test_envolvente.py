# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:08:35 2017

Este script entrena un clasificador SVM con el metodo de la envolvente 
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
#from sklearn import decomposition
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import learning_curve
from scipy import stats

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
   # conc = np.reshape(conc, (len(conc)*5,8,2500))
   # rel = np.reshape(rel, (len(rel)*5,8,2500))
    return data_time


def butter_filter(data,lowcut = 8, highcut=13, fqc=500, order=6):
    """Filtro pasabajas.
    """
    nyq = 0.5*fqc
    high = highcut/nyq
    low = lowcut/nyq;
    [b_c, a_c] = signal.butter(order, [low,high], btype='band')
    filt_sig = signal.filtfilt(b_c, a_c, data)
    #filt_sig = signal.filtfilt(b_c, a_c,  data)
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

def split_overlap(array,size,overlap):
    result = []
    while True:
        if len(array) <= size:
            result.append(array)
            return result
        else:
            result.append(array[:size])
            array = array[size-overlap:]


"Main Code"
plt.close('all')
S_ID = "S01" # Nombre del archivo del sujeto
DATA = get_data_db(S_ID) # Obtiene los datos del .mat
Datar = car(DATA) # Referencia la señal 
[D_DC, m_v] = remove_dc(Datar) # Datos sin DC
Y = butter_filter(D_DC) #Filtra la señal entre 8-13 Hz
SCALE = 1
FS = 500/SCALE # esto es porque fue submuestreado 
DIV = 500.0/SCALE
Electrode = ['AF3','AF4','F3','F4','C3','C4','C1','C2'];

TS = 1.0/FS
TIME = np.arange(0, len(Y[0][0]) * TS, TS)
sigan = hilbert(Y)
signalhilt = np.abs(sigan) #Envolvente de la señal 
nyq = 0.5*500
[b_c, a_c] = signal.butter(4, [2/nyq], btype='low') #Se filtra a 2 Hz
sighilt = signal.filtfilt(b_c, a_c, signalhilt)
hilb1 = sighilt[0:len(Y)/2,:,500:-500] # Se eliminan los transistorios 
hilb2 = sighilt[len(Y)/2::,:,500:-500]
sighilt1 = np.mean(hilb1, axis=0) #Envolvente promedio clase 1
sighilt2 = np.mean(hilb2, axis=0) #Envolvente promedio clase 2

sighiltm1_1 = np.mean(sighilt[:,:,500:-500], axis=2) # Valor promedio por electrodo 
sighiltm1 = sighiltm1_1 #Solo alfa
#%%
ff = sighiltm1.shape # Tamaño del array
dimen = len(ff)

if (dimen == 3):
  M_F = sighiltm1 # Potencia promedio para cada frecuencia
  FEATS = np.reshape(M_F, (ff[0], M_F.shape[2]*ff[1]))

elif (dimen == 2):    
  M_F = sighiltm1 # Potencia promedio para cada frecuencia
  FEATS = M_F
elif (dimen == 1):    
  M_F = sighiltm1 # Potencia promedio para cada frecuencia
  FEATS = M_F
  
LABELS = np.zeros((FEATS.shape[0]))
LABELS[0:len(LABELS)/2] = 1
LABELS[len(LABELS)/2::] = 2

clf = svm.SVC(kernel='linear', C=1).fit(FEATS,LABELS)


S_ID2 = "S01_R" # Nombre del archivo del sujeto
DATA = get_data_db(S_ID2) # Obtiene los datos del .mat
Datar = car(DATA) # Referencia la señal 
[D_DC, m_v] = remove_dc(Datar) # Datos sin DC
Y = butter_filter(D_DC) #Filtra la señal entre 8-13 Hz
TS = 1.0/FS
TIME = np.arange(0, len(Y[0][0]) * TS, TS)
sigan = hilbert(Y)
signalhilt = np.abs(sigan) #Envolvente de la señal 
nyq = 0.5*500
[b_c, a_c] = signal.butter(4, [2/nyq], btype='low') #Se filtra a 2 Hz
sighilt = signal.lfilter(b_c, a_c, signalhilt)
hilb1 = sighilt[0:len(Y)/2,:,500::] # Se eliminan los transistorios 
hilb2 = sighilt[len(Y)/2::,:,500::]
sighilt1 = np.mean(hilb1, axis=0) #Envolvente promedio clase 1
sighilt2 = np.mean(hilb2, axis=0) #Envolvente promedio clase 2
LABELS = np.zeros((Y.shape[0]))
LABELS[0:len(LABELS)/2] = 1
LABELS[len(LABELS)/2::] = 2
sighiltm1_1 = np.mean(sighilt[:,:,500:-500], axis=2) # Valor promedio por electrodo 
test_feats = sighiltm1_1 #Caracteristicas de prueba

y_pred = clf.predict(test_feats)
y_true = LABELS
sc = accuracy_score(y_true, y_pred)
print sc
conf = confusion_matrix(LABELS, y_pred)
print conf

