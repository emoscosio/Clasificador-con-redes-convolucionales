# Clasificador-con-redes-convolucionales
Este proyecto ha sido desarrollado como TFG para la Universidad de Sevilla por el alumno Eduardo Moscosio Navarro de Ingeniería Electrónica, Robótica y Mecatrónica de la ETSI.

Se ha desarrollado una red neuronal convolucional capaz de identificar 43 señales de tráfico diferentes, usando para su entrenamiento el dataset "German Traffic Sign" del enlace http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset. 
Se proporcionarán diferentes códigos de Python que implementan todos los experimentos realizados para el correcto diseño de la red, así como un simulador que carga el modelo de red para identificar las imágenes que se le pasen.

Por último, se implementó dicha red en un sistema embebido Raspberry Pi 3 Model B, por lo que se aportará también el código que se usó para implementar la red y usar una cámara conectado a dicho dispositivo en el archivo "imlementa_RPI3.py".

Además de los códigos comentados anteriormente, se aportan también las imágenes y etiquetas del dataset ya guardados como matrices y vectores de numpy en los archivos.npy para agilizar el proceso, y se proporciona el modelo de red neuronal elegido como definitivo para resolver el problema, indicando en su nombre el valor de accuracy obtenido en el entrenamiento, validación y test.
