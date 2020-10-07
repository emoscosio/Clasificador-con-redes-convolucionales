# -*- coding: utf-8 -*-
"""
# Simulador de implementación de la red

Realizado por Eduardo Moscosio Navarro. 

Ingeniería Electrónica, Robótica y Mecatrónica. Universidad de Sevilla
"""
"""
Esta celda solo se ejecuta al usar Google Colab.
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
import cv2
import numpy as np
import seaborn as sn
import shutil # Para copiar a otra carpeta
import copy

"""Definición de funciones para preprocesado de imágenes:

Incluye la misma función de preprocesado usada en los experimentos, y una función que asocia al número de clase predicho por la red el nombre real de la señal que pertenece a dicha clase.
"""

# Funciones de preprocesado:
def image_preproc(img, coef = None, width = None, height = None, inter = cv2.INTER_AREA):
    dim = (width,height)
    # RGB to Gray image conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize the image
    img_prep = cv2.resize(gray, dim, interpolation = inter)
    # rescale the image
    img_prep.astype('float32') # Convierte a float32
    img_prep = img_prep/coef # Escalado
    # return the resized image
    return img_prep

# Funciones de resultados:
def signal_type(prediction):
  # Vector con los nombres de las señales:
  signal = np.array(["Velocidad máxima 20 Km/h", "Velocidad máxima 30 Km/h", "Velocidad máxima 50 Km/h", "Velocidad máxima 60 Km/h", "Velocidad máxima 70 Km/h",
            "Velocidad máxima 80 Km/h", "Fin de limitación de velocidad máxima 80 Km/h", "Velocidad máxima 100 Km/h", "Velocidad máxima 120 Km/h", "Adelantamiento prohibido",
            "Adelantamiento prohibido para camiones", "Intersección con prioridad", "Calzada con prioridad", "Ceda el paso", "STOP", "Circulación prohibida en ambos sentidos",
            "Prohibición de acceso a vehículos destinados a transporte de mercancías", "Entrada prohibida", "Otros peligros", "Curva peligrosa hacia la izquierda",
            "Curva peligrosa hacia la derecha", "Curvas peligrosas hacia la izquierda", "Perfil irregular", "Pavimento deslizante", "Estrechamiento de calzada por la derecha",
            "Obras", "Semáforo", "Peatones", "Niños", "Ciclistas", "Pavimento deslizante por hielo o nieve", "Paso de animales en libertad", "Fin de prohibiciones",
            "Sentido obligatorio derecha", "Sentido obligatorio izquierda", "Sentido obligatorio recto", "Recto y derecha únicas direcciones permitidas",
            "Recto e izquierda únicas direcciones permitidas", "Paso obligatorio derecha", "Paso obligatorio izquierda", "Intersección de sentido giratorio-obligatorio",
            "Fin de prohibición de adelantamiento", "Fin de prohibición de adelantamiento para camiones"])
  # Se asocia el número obtenido en la predicción con el nombre de la señal:
  if len(prediction) > 0:
    for k in range(0,len(prediction)):
      if prediction[k] < 10:
        print(str(prediction[k]) + "   ==>  " + str(signal[prediction[k]]))
      else:
        print(str(prediction[k]) + "  ==>  " + str(signal[prediction[k]]))
  else:
    print(str(prediction) + "  ==>  " + str(signal[prediciton]))

"""Se cargan las imágenes del dataset guardadas como arrays de numpy en un archivo .npy con anterioridad. Para ello, hay que poner en cada uno la dirección en la que se encuentra esa parte del dataset. Ahora solamente se usarán las de test, ya que simularán las señales de tráfico.

Además, hay que cargar el modelo de la red desarrollado, por lo que hay que introducir la dirección en la que se encuentre dicho modelo. Hay que cargar los archivos .json y .h5, que en este caso tenían el mismo nombre por comodidad.
"""

# Cargamos el dataset, tanto imágenes como sus etiquetas:
dir_labels = '' ## PONER AQUÍ DIRECCIÓN DE LAS ETIQUETAS DEL CONJUNTO DE TEST ENTRE ''
dir_img = '' ## PONER AQUÍ DIRECCIÓN DE LAS IMÁGENES DEL CONJUNTO DE TEST ENTRE ''

test_lab = np.load(dir_labels, allow_pickle=True)
test_im = np.load(dir_img, allow_pickle=True)

print("Numero de imágenes: ", len(test_im))

# Carga el modelo con los pesos de la red:
from keras import layers
from keras import models

# cargar json y crear el modelo
# Nombre del archivo:
dir_mod = "" ## PONER DIRECCIÓN DEL MODELO ENTRE ""

json_file = open(dir_mod + ".json", 'r')
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)

# cargar pesos al nuevo modelo
model.load_weights(dir_mod + ".h5")
print("Cargado modelo desde disco.")
model.summary() # Para ver como es la red

"""Se aportan 3 tipos diferentes de simuladores:

- Simulador 1: Recibe imágenes en formato jpg, png, etc., que no pertencen al conjunto de test previamente cargado. Para ello, únicamente hay que poner la dirección de la imagen donde corresponde. Como salida, sacará por pantalla el porcentaje predicho para cada una de las 43 señales, la predicción realizada por la red, elegida como la que tiene mayor porcentaje, y el nombre de la señal asociado a esa predicción. Por último, saca por pantalla la imagen que se analizó con el fin de verificar si el resultado es o no correcto.
- Simulador 2: Se usa el conjunto de test cargado anteriormente, pero solo con una imagen que se elija. Para ello, hay que elegir el número de la imagen del dataset y ponerlo donde corresponda. Por lo demás, el funcionamiento es igual que el anterior, solo que ahora saca también la etiqueta de la imagen del dataset, pudiendo ver si coincide con la predicha.
- Simulador 3: Se usa de nuevo el dataset, pero ahora se hace un número de análisis de imágenes aleatorias para calcular el tiempo medio de respuesta de la red. Ese número de análisis es elegido por el usuario. Saca a la salida el tiempo que tarda para cada imagen, el tiempo medio, las predicciones de la red, y el tipo de señal que son realmente.
"""

# SIMULADOR 1:
from time import time

# Elige la imagen:
dir = "" ## PONER DIRECCIÓN DE IMAGEN A CLASIFICAR ENTRE ""

img_prueba = plt.imread(dir)
pred = []
# Le llega una imagen:
signal = copy.copy(img_prueba)
# Empieza el proceso:
start_time = time() # Tiempo de ejecución comienza

# Preprocesado:
ancho = 64
alto = 64
signal_prep = image_preproc(signal, coef = 255, width = ancho, height = alto)
test = signal_prep.reshape([-1,ancho, alto,1])

# Clasificación:
predictions = model.predict(test, batch_size=1, verbose=0) # Obtiene los 43 porcentajes para la imagen
pred_max = np.argmax(predictions, axis=-1) # Se queda con la que tiene mayor porcentaje

# Termina el proceso:
elapsed_time = time() - start_time # Tiempo de ejecución termina
print("Tiempo empleado: %.10f seconds." % elapsed_time) # Imprime el tiempo que ha tardado
pred.append(pred_max)


# Imprime la clase predicha y la imagen original:
print("Las predicciones son: ")
print(predictions)
print("La señal predicha es de la clase: ")
signal_type(pred)

plt.imshow(signal)

# SIMULADOR 2:
from time import time

# Elige la imagen:
num = 100 ## PONER NÚMERO DE LA IMAGEN DEL DATASET A CLASIFICAR
real_label = []
pred = []
# Le llega una imagen:
signal = copy.copy(test_im[num])
# Empieza el proceso:
start_time = time() # Tiempo de ejecución comienza

# Preprocesado:
ancho = 64
alto = 64
signal_prep = image_preproc(signal, coef = 255, width = ancho, height = alto)
test = signal_prep.reshape([-1,ancho, alto,1])

# Clasificación:
predictions = model.predict(test, batch_size=1, verbose=0) # Obtiene los 43 porcentajes para la imagen
pred_max = np.argmax(predictions, axis=-1) # Se queda con la que tiene mayor porcentaje

# Termina el proceso:
elapsed_time = time() - start_time # Tiempo de ejecución termina
print("Tiempo empleado: %.10f seconds." % elapsed_time) # Imprime el tiempo que ha tardado
pred.append(pred_max)
real_label.append(int(test_lab[num]))


# Imprime la clase predicha y la imagen original:
print("Las predicciones son: ")
print(predictions)
print("La señal predicha es de la clase: ")
signal_type(pred)
print("La señal pertence realmente a la clase: ")
signal_type(real_label)
plt.imshow(signal)

# SIMULADOR 3:
from time import time
from random import randint

# Elige la imagen:
numero_imagenes = 10 ## PONER NÚMERO DE IMÁGENES ALEATORIAS QUE SE TOMARÁN DEL DATASET
real_label = []
elapsed = []
pred = []
for k in range(0,numero_imagenes):
  # Le llega una imagen:
  num = randint(0,len(test_lab)-1) # Número aleatorio entre 0 y el número de imágenes - 1
  signal = copy.copy(test_im[num])
  # Empieza el proceso:
  start_time = time() # Tiempo de ejecución comienza

  # Preprocesado:
  ancho = 64
  alto = 64
  signal_prep = image_preproc(signal, coef = 255, width = ancho, height = alto)
  test = signal_prep.reshape([-1,ancho, alto,1])

  # Clasificación:
  predictions = model.predict(test, batch_size=1, verbose=0) # Obtiene los 43 porcentajes para la imagen
  pred_max = np.argmax(predictions, axis=-1) # Se queda con la que tiene mayor porcentaje
  # Termina el proceso:
  elapsed_time = time() - start_time # Tiempo de ejecución 
  
  pred.append(pred_max)
  elapsed.append(elapsed_time)
  real_label.append(int(test_lab[num]))
  print("Tiempo empleado: %.10f seconds." % elapsed_time) # Imprime el tiempo que ha tardado

# Imprime la clase predicha y la imagen original:
print("Tiempo medio que tarda en segundos: ", sum(elapsed)/numero_imagenes)
print("La señal predicha es de la clase: ")
signal_type(pred)
print("La señal pertence realmente a la clase: ")
signal_type(real_label)
