# BCI-Final_Project
Aquí se implementa una interfaz cerebro-computadora basada en ritmos cerebrales alfa para el control de una animación 

## Getting Started

Este proyecto se desarrolló con la finalidad, de que puede replicarse el proyecto en cualquier parte.

### Pre-requisitos

Los siguientes programas se necesitan instalar
```
Python 2.7 NIC
```
Además, las librerías de Python.
Estas librerías pueden ser instaladas via pip.
```
Numpy Scipy Matplotlib Pygame Scikit-learn Pylsl winsound
```

## Pruebas

Los scripts están dividos en tres carpetas:

### Clasificación

Aquí se encuentran los scripts con los métodos de envolvente y espectrograma. En estos scripts se entrena un clasificador y luego se prueba el clasificador con los datos de realimentación del mismo sujeto

Comando para ejecutar espectrograma:
```
python train_test_espectro.py
```

Comando para ejecutar envolvente:
```
python train_test_envolvente.py
```
### Entrenamiento

Se encuentra el codigo de entrenamiento para concentracion y relajación 

```
python Entrenamiento.py
```

### Realimentación

Se encuentran los codigo de realimentación voluntaria y no voluntaria para concentracion y relajación. 
En ambos algoritmos se deben ejecutar despues del entrenamiento.

Realimentación No voluntaria:
```
python Realimentacion_no_voluntaria.py
```

Realimentación voluntaria:
```
python Realimentacion_voluntaria.py
```

## Authors

* **Dayán Méndez** - [Perfil](https://github.com/djmv)

* **Jorge Silva** - [Perfil](https://github.com/JorgeluissilvaC)

## Acknowledgments

* BSPAI-Lab
* A todas las personas que colaboraron en nuestra investigación y etapa de aprendizaje.
