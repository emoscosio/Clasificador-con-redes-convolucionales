# -*- coding: utf-8 -*-
"""
# Experimentos de entrenamiento, validación y test, para la red de clasificación de señales de tráfico

Realizado por Eduardo Moscosio Navarro. 

Ingeniería Electrónica, Robótica y Mecatrónica. Universidad de Sevilla
"""
"""
Esta celda se ejecuta solamente en Google Colab.
"""

# Linkamos con nuestro Drive para tener disponibles los archivos:
from google.colab import drive
drive.mount('/content/drive/')

# Se instala keras:
!pip install -q keras

"""En primer lugar, se cargan todas las librerías que se van a usar:"""

# Cargamos librerías a usar
import keras

import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import copy

"""Definición de funciones para preprocesado de imágenes:"""

# Funciones:

# Cambia a gris y redimensiona al tamaño deseado:
def image_gray_resize(images, width = None, height = None, inter = cv2.INTER_AREA):
    dim = (width,height)
    for k in range(0, len(images)):
      # RGB to Gray image conversion
      gray = cv2.cvtColor(images[k], cv2.COLOR_BGR2GRAY)
      # resize the image
      images[k] = cv2.resize(gray, dim, interpolation = inter)
      # return the resized image
    return 0

# Normaliza la imagen:
def image_normalize(images, coef = None):
    for k in range(0, len(images)):
      images[k].astype('float32') # Convierte a float32
      images[k] = images[k]/coef # Escalado
    return 0

# Muestra las imágenes: Esto es porque al convertir a Gray la imagen RGB,
# no la muestra directamente gris sino en tonalidades azules, porque no la 
# interpreta como tal, por eso, se reconvierte a 3 canales, de modo que ya si
# se vea. solo se usa para representar
def show_im(img):
    imagen = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imagen)

"""Se cargan las imágenes del dataset guardadas como arrays de numpy con anterioridad. Para ello, hay que poner en cada uno la dirección en la que se encuentra esa parte del dataset:"""

dir_train_labels = '' ## PONER AQUÍ DIRECCIÓN DE LAS ETIQUETAS DEL CONJUNTO DE TRAIN ENTRE ''
dir_train_img = '' ## PONER AQUÍ DIRECCIÓN DE LAS IMÁGENES DEL CONJUNTO DE TRAIN ENTRE ''
dir_valid_labels = '' ## PONER AQUÍ DIRECCIÓN DE LAS ETIQUETAS DEL CONJUNTO DE VALIDATION ENTRE ''
dir_valid_img = '' ## PONER AQUÍ DIRECCIÓN DE LAS IMÁGENES DEL CONJUNTO DE VALIDATION ENTRE ''
dir_test_labels = '' ## PONER AQUÍ DIRECCIÓN DE LAS ETIQUETAS DEL CONJUNTO DE TEST ENTRE ''
dir_test_img = '' ## PONER AQUÍ DIRECCIÓN DE LAS IMÁGENES DEL CONJUNTO DE TEST ENTRE ''

train_lab = np.load(dir_train_labels, allow_pickle=True) # Etiquetas de entrenamiento
train_im = np.load(dir_train_img, allow_pickle=True) # Imágenes de entrenamiento
valid_lab = np.load(dir_valid_labels, allow_pickle=True) # Etiquetas de validación
valid_im = np.load(dir_valid_img, allow_pickle=True) # Imágenes de validación
test_lab = np.load(dir_test_labels, allow_pickle=True) # Etiquetas de test
test_im = np.load(dir_test_img, allow_pickle=True) # Imágenes de test

# Imprime en pantalla el número de imágenes de cada parte
print("Total de imágenes: ", len(train_lab)+len(valid_lab)+len(test_lab))
print("Imágenes de Training: ", len(train_lab))
print("Imágenes de Validation: ", len(valid_lab))
print("Imágenes de Test: ", len(test_lab))

"""Visualización de una imagen de cada clase para ver que todo está correcto:"""

print("TRAIN:")
print("etiquetas: ", len(train_lab), "// imagenes: ", len(train_im))
print("Etiqueta del ejemplo: ", train_lab[100])
plt.figure()
plt.imshow(train_im[100])


print("VALIDATION:")
print("etiquetas: ", len(valid_lab), "// imagenes: ", len(valid_im))
print("Etiqueta del ejemplo: ", valid_lab[500])
plt.figure()
plt.imshow(valid_im[500])


print("TEST:")
print("etiquetas: ", len(test_lab), "// imagenes: ", len(test_im))
print("Etiqueta del ejemplo: ", test_lab[1000])
plt.figure()
plt.imshow(test_im[1000])

"""Se preprocesa la imagen, empezando con un cambio a escala de grises y luego una redimensión al tamaño deseado, en este caso, 64x64 píxeles:"""

ancho = 64
alto = 64
image_gray_resize(train_im, width = ancho, height = alto)
image_gray_resize(valid_im, width = ancho, height = alto)
image_gray_resize(test_im, width = ancho, height = alto)

"""Visualización de una imagen de cada clase para ver que todo está correcto:"""

print("TRAIN:")
print("etiquetas: ", len(train_lab), "// imagenes: ", len(train_im))
print("Etiqueta del ejemplo: ", train_lab[100])
print("Tamaño del ejemplo: ", train_im[100].shape)
plt.figure()
show_im(train_im[100])

print("VALIDATION:")
print("etiquetas: ", len(valid_lab), "// imagenes: ", len(valid_im))
print("Etiqueta del ejemplo: ", valid_lab[500])
print("Tamaño del ejemplo: ", valid_im[500].shape)
plt.figure()
show_im(valid_im[500])

print("TEST:")
print("etiquetas: ", len(test_lab), "// imagenes: ", len(test_im))
print("Etiqueta del ejemplo: ", test_lab[1000])
print("Tamaño del ejemplo: ", test_im[1000].shape)
plt.figure()
show_im(test_im[1000])

"""Normalización de las imágenes en un rango de 0 a 1 para que sea procesable por la red. Originalmente están en un rango de 0 a 255."""

image_normalize(train_im, coef = 255)
image_normalize(valid_im, coef = 255)
image_normalize(test_im, coef = 255)

"""Visualización de una imagen de cada clase para ver que todo está correcto:"""

print("TRAIN:")
print("etiquetas: ", len(train_lab), "// imagenes: ", len(train_im))
print("Etiqueta del ejemplo: ", train_lab[100])
print("Tamaño del ejemplo: ", train_im[100].shape)
print(train_im[100])

print("VALIDATION:")
print("etiquetas: ", len(valid_lab), "// imagenes: ", len(valid_im))
print("Etiqueta del ejemplo: ", valid_lab[500])
print("Tamaño del ejemplo: ", valid_im[500].shape)
print(valid_im[500])

print("TEST:")
print("etiquetas: ", len(test_lab), "// imagenes: ", len(test_im))
print("Etiqueta del ejemplo: ", test_lab[1000])
print("Tamaño del ejemplo: ", test_im[1000].shape)
print(test_im[1000])

"""Por último, se redimensionan las imágenes como un tensor de la forma [-1, alto, ancho, 1] para que la red pueda adquirir los datos a la entrada. Además, se pasan las etiquetas a codificación "one-hot", ya que se usará más adelante:"""

# Primero se pasa cada imagen a una lista de train, valid y test:
train_list = []
for k in range(0,len(train_im)):
  train_list.append(train_im[k])
valid_list = []
for k in range(0,len(valid_im)):
  valid_list.append(valid_im[k])
test_list = []
for k in range(0,len(test_im)):
  test_list.append(test_im[k])

# Una vez se tienen las listas, ya se puede redimensionar de la manera deseada
# (Si se hacía directamente con los vectores daba un error y no se podía redimensionar)
new_train = np.array(train_list)
new_validation = np.array(valid_list)
new_train = new_train.reshape([-1,alto, ancho,1])
new_validation = new_validation.reshape([-1,alto, ancho,1])
test = np.array(test_list)
test = test.reshape([-1,alto, ancho,1])

# Por último, se pasan las etiquetas a categorical para el entrenamiento:
# Pasamos las etiquetas a codificación "one-hot",es decir, se pone '1' en la posición de clase mientras
# las demás se ponen a '0':
from keras.utils import to_categorical
train_lab = to_categorical(train_lab)
valid_lab = to_categorical(valid_lab)
test_labels = to_categorical(test_lab)

"""EMPIEZA EL DISEÑO DE LA RED:

A continuación hay 3 celdas cada una con un modelo declarado para un experimento concreto. En primer lugar, se encuentran los experimentos con la red original sin ningñun añadido. A continuación, en la celda siguiente, se encuentran los modelos para los experimentos realizados con dropout. Y finalmente, se encuentra la celda con los experimentos realizados para ver el efecto de modificar la red original en la obtención de resultados. Hay que ejecutar cada una y elegir en dicha celda el modelo deseado según el experimento que se quiera hacer.

PRIMEROS EXPERIMENTOS
"""

############# Código principal de red convolucional ####################################
from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))

model.add(layers.Conv2D(64, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))

model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))

model.add(layers.Conv2D(256, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu', input_shape=(1024,)))
model.add(layers.Dense(512, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dense(128, activation='relu', name="Dense_Hidden_Layer3"))
model.add(layers.Dense(43, activation='softmax'))

model.summary() # Muestra la estructura de la red

"""EXPERIMENTOS CON DROPOUT (DESCOMENTAR EL QUE PROCEDA)

*   Dropout de tipo 1: Se usó una probabilidad de de activación de neuronas del 80% en las capas convolucionales (que se representa aquí con un 20% de neuronas desactivadas), mientras que para las capas densas se usó una probabilidad de activación de solamente el 30% (representado por el 70% en el código en Keras)
*   Dropout de tipo 2: Es como el tipo 1 pero se pusieron las capas densas con una probabilidad de activación de neuronas del 50%, mientras que en las capas convolucionales se puso una probabilidad de activación del 75% (25% de desactivación) respecto al tipo 1 donde era del 80%.
*   Dropout de tipo 3: Las probabilidades usadas son como las del tipo 2, con la diferencia de que ahora no se hace Dropout entre la última capa oculta de la etapa densamente conectada y la capa de salida Softmax.
"""

############# Código principal de red convolucional ####################################
from keras import layers
from keras import models

# Experimentos de Dropout: 
# En Keras, el parámetro de probabilidad que maneja el Dropout indica el 
# porcentaje de neuronas que quedan desactivadas de manera aleatoria en cada época de entrenamiento. 


#####################
#     DROPOUT 1     #

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))
model.add(layers.Dropout(0.20, name="Dropout1_CNN_Layer")) 
model.add(layers.Conv2D(64, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))
model.add(layers.Dropout(0.20, name="Dropout2_CNN_Layer"))
model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))
model.add(layers.Dropout(0.20, name="Dropout3_CNN_Layer"))
model.add(layers.Conv2D(256, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))
model.add(layers.Dropout(0.20, name="Dropout4_CNN_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu', input_shape=(1024,)))
model.add(layers.Dropout(0.7, name="Dropout1_Dense_Layer"))
model.add(layers.Dense(512, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dropout(0.7, name="Dropout2_Dense_Layer"))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dropout(0.7, name="Dropout3_Dense_Layer"))
model.add(layers.Dense(128, activation='relu', name="Dense_Hidden_Layer3"))
model.add(layers.Dropout(0.7, name="Dropout4_Dense_Layer"))
model.add(layers.Dense(43, activation='softmax'))

#####################

"""
#####################
#     DROPOUT 2     #

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))
model.add(layers.Dropout(0.25, name="Dropout1_CNN_Layer")) 
model.add(layers.Conv2D(64, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))
model.add(layers.Dropout(0.25, name="Dropout2_CNN_Layer"))
model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))
model.add(layers.Dropout(0.25, name="Dropout3_CNN_Layer"))
model.add(layers.Conv2D(256, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))
model.add(layers.Dropout(0.25, name="Dropout4_CNN_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu', input_shape=(1024,)))
model.add(layers.Dropout(0.5, name="Dropout1_Dense_Layer"))
model.add(layers.Dense(512, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dropout(0.5, name="Dropout2_Dense_Layer"))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dropout(0.5, name="Dropout3_Dense_Layer"))
model.add(layers.Dense(128, activation='relu', name="Dense_Hidden_Layer3"))
model.add(layers.Dropout(0.5, name="Dropout4_Dense_Layer"))
model.add(layers.Dense(43, activation='softmax'))

#####################
"""
"""
#####################
#     DROPOUT 3     #

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))
model.add(layers.Dropout(0.25, name="Dropout1_CNN_Layer")) 
model.add(layers.Conv2D(64, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))
model.add(layers.Dropout(0.25, name="Dropout2_CNN_Layer"))
model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))
model.add(layers.Dropout(0.25, name="Dropout3_CNN_Layer"))
model.add(layers.Conv2D(256, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))
model.add(layers.Dropout(0.25, name="Dropout4_CNN_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu', input_shape=(1024,)))
model.add(layers.Dropout(0.5, name="Dropout1_Dense_Layer"))
model.add(layers.Dense(512, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dropout(0.5, name="Dropout2_Dense_Layer"))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dropout(0.5, name="Dropout3_Dense_Layer"))
model.add(layers.Dense(128, activation='relu', name="Dense_Hidden_Layer3"))
model.add(layers.Dense(43, activation='softmax'))

#####################
"""

model.summary()

"""EXPERIMENTOS DE MODIFICACIÓN DE LA RED ORIGINAL (DESCOMENTAR EL QUE PROCEDA)"""

############# Código principal de red convolucional ####################################
# Diseñamos la red:
from keras import layers
from keras import models
# Parametros en cada capa = Nº filtros*(Area filtro + 1). Por ejemplo:
# model.add(layers.Conv2D(32,(6,6),activation='relu', input_shape=(32,32,1))) ===> 32*(6*6 + 1) = 1184

##########################
#     MODIFICACIÓN 1     #

model = models.Sequential()

model.add(layers.Conv2D(64,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))

model.add(layers.Conv2D(128, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))

model.add(layers.Conv2D(256, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))

model.add(layers.Conv2D(512, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(2048, activation='relu', input_shape=(2048,)))
model.add(layers.Dense(1024, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dense(512, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer3"))
model.add(layers.Dense(128, activation='relu', name="Dense_Hidden_Layer4"))
model.add(layers.Dense(64, activation='relu', name="Dense_Hidden_Layer5"))
model.add(layers.Dense(43, activation='softmax'))
##########################

"""
##########################
#     MODIFICACIÓN 2     #

model = models.Sequential()

model.add(layers.Conv2D(16,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))

model.add(layers.Conv2D(32, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))

model.add(layers.Conv2D(64, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))

model.add(layers.Conv2D(128, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu', input_shape=(512,)))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dense(128, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dense(43, activation='softmax'))
##########################
"""
"""
##########################
#     MODIFICACIÓN 3     #

model = models.Sequential()

model.add(layers.Conv2D(64,(3,3),activation='relu', input_shape=(64,64,1), name="Conv1_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling1_Layer"))
model.add(layers.Dropout(0.25, name="Dropout1_CNN_Layer")) 
model.add(layers.Conv2D(128, (2,2), activation='relu', name="Conv2_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling2_Layer"))
model.add(layers.Dropout(0.25, name="Dropout2_CNN_Layer"))
model.add(layers.Conv2D(256, (3,3), activation='relu', name="Conv3_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling3_Layer"))
model.add(layers.Dropout(0.25, name="Dropout3_CNN_Layer"))
model.add(layers.Conv2D(512, (3,3), activation='relu', name="Conv4_Layer"))
model.add(layers.MaxPooling2D((2, 2), name="Pooling4_Layer"))
model.add(layers.Dropout(0.25, name="Dropout4_CNN_Layer"))

model.add(layers.Flatten())

model.add(layers.Dense(2048, activation='relu', input_shape=(2048,)))
model.add(layers.Dense(4096, activation='relu', name="Dense_Hidden_Layer1"))
model.add(layers.Dropout(0.25, name="Dropout1_Dense_Layer"))
model.add(layers.Dense(2048, activation='relu', name="Dense_Hidden_Layer2"))
model.add(layers.Dropout(0.25, name="Dropout2_Dense_Layer"))
model.add(layers.Dense(1024, activation='relu', name="Dense_Hidden_Layer3"))
model.add(layers.Dropout(0.25, name="Dropout3_Dense_Layer"))
model.add(layers.Dense(512, activation='relu', name="Dense_Hidden_Layer4"))
model.add(layers.Dropout(0.25, name="Dropout4_Dense_Layer"))
model.add(layers.Dense(256, activation='relu', name="Dense_Hidden_Layer5"))
model.add(layers.Dense(43, activation='softmax'))

##########################
"""

model.summary()

"""ENTRENAMIENTO Y VALIDACIÓN:

Se harán varios experimentos, usando en cada uno un optimizador diferente:
- SGD
- RMSprop
- Adam

Hay que descomentar el que se desee probar, y definir en cada uno el número épocas y el batch de imágenes deseado.
"""

# Entrenamiento de la red:

###############
#     SGD     #

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
batch = 200
epocas = 50
###############

"""
###################
#     RMSprop     #
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])
batch = 200
epocas = 50
###################
"""
"""
################
#     Adam     #
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch = 200
epocas = 50
################
"""

# Entrenamiento y validación del modelo con el optimizador y métricas elegidas:

snn = model.fit(new_train, 
                train_lab, 
                batch_size=batch, 
                epochs=epocas, 
                validation_data=(new_validation, valid_lab), 
                shuffle=True, 
                verbose=2)

"""ANÁLISIS DE RESULTADOS DE ENTRENAMIENTO Y VALIDACIÓN:"""

# Para pintar gráficas de accuracy:
import matplotlib.pyplot as plt

plt.figure()
plt.plot(snn.history['accuracy'],'r')  
plt.plot(snn.history['val_accuracy'],'g')  
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 100
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.grid(True)
plt.savefig("Train_acc_vs_Val_acc.jpg", bbox_inches='tight')

# Para pintar gráficas de loss:
import matplotlib.pyplot as plt

plt.figure() 
plt.plot(snn.history['loss'],'r')  
plt.plot(snn.history['val_loss'],'g')
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 100
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])
plt.grid(True)
plt.savefig("Train_loss_vs_Val_loss.jpg", bbox_inches='tight')

"""ANÁLISIS DE TEST:

Se obtendrá el accuracy de test del modelo entrenado, la matriz de confusión, y las métricas de precisió, recall y F1 para cada clase del dataset.
"""

test_loss, test_acc= model.evaluate(test, test_labels)
print ("Test Accuracy:", test_acc)
print ("Test Loss:", test_loss)

# Generación de predicciones (o TEST): Le damos imágenes sin las etiquetas, diferentes a las usadas ya, y a ver que sale:
# Se comprueba antes de hacer la predicción:
#plt.imshow(x_test[11], cmap=plt.cm.binary) # Es un '6'
# Ahora se hace la predicción y se mira si coincide con lo que debe salir:
#predictions = model.predict(np.array(test), batch_size=32, verbose=1)
predictions = model.predict(test, batch_size=batch, verbose=1)
pred_max = np.argmax(predictions, axis=1)

# Matriz de confusión:
conf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), pred_max)
# Se visualiza:
import pandas as pd
show_matrix = pd.DataFrame(conf_matrix, range(43), range(43))
show_matrix

resultados = classification_report(np.argmax(test_labels, axis=1), pred_max)

# Se imprimen por un txt y luego se sacan por pantalla:

import sys
orig_stdout = sys.stdout # Guarda la dirección actual de escritura
sys.stdout = open('Test.txt','wt') # Cambia la salida de datos al archivo.txt
print(resultados) # Imprime en el archivo.txt

sys.stdout.close() # Se cierra el archivo
sys.stdout=orig_stdout # Se retorna la salida de datos aquí"""
print(resultados) # Escribe aquí en la celda

"""Finalmente, se guardan la estructura y los pesos de la red obtenida,dándole el nombre deseado. En este caso, se le da un nombre con los accuracy de los 3 análisis, el optimizador empleado, el número de épocas, y el número de imágenes en el batch."""

# Guardar el modelo de red:
# Nombre del archivo:
nombre = "" ## PONER NOMBRE QUE SE LE QUIERA DAR AL MODELO ENTRE ""
# serializar el modelo a JSON
model_json = model.to_json()
with open(nombre + ".json", "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
model.save_weights(nombre + ".h5")
print("Modelo Guardado!")
