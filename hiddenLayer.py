'''
Codigo de la CAPA OCULTA: Entrenamiento

En esta capa se hace el entrenamiento de las fotos.

Las redes neuronales tienen 3 capas:

1) Capa de Entrada.
2) Capa Oculta. 
3) Capa de Salida.

@author: Nicolas Gaston Rodriguez Perez
'''


from cProfile import label
import cv2 as cv
import os
import numpy as np
from time import time

data_path = 'capturas/'
listaData = os.listdir(data_path)
print('User Data: ', listaData)

# Inits
people = []
rostrosData = []
id = 0

'''

    Inicio de etiquetado de la informacion. Etiqueto la informacion para cada persona.

'''

print('\n>>>>>>>>>>>>> Procesando las fotos de las personas <<<<<<<<<<<<<<<<<<<')
initial_time = time()

for carpeta in listaData:
    carpeta_usuario = '{}{}'.format(data_path, carpeta)
    print('\n############ Etiquetando fotos en la carpeta: {} ############'.format(carpeta))

    for archivo in os.listdir(carpeta_usuario):

        print('{}: imagen {}'.format(carpeta_usuario, archivo))

        people.append(id)

        # Guardo las fotos en el array
        rostrosData.append(
            cv.imread(
                '{}/{}'.format(carpeta_usuario, archivo),
                0  # Este ultimo 0 transforma la imagen a escala de grises
            )
        )

    id = + 1

final_time = time()
total_time = final_time - initial_time
print('\nTardo {} segundos en etiquetar las imagenes. \n\n'.format(total_time))


'''

Aca empieza el entrenamiento.

'''

print('>>>>>>>>>>>>>>>>>>>>>>>>> Empieza el entrenamiento <<<<<<<<<<<<<<<<<<<<<\n')
initial_time = time()

#entrenamientoLBPHEigenFaceRecognizer = cv.face.LBPHEigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.train(rostrosData, np.array(people))

# Guardo el entrenamiento en un archivo .xml

entrenamientoEigenFaceRecognizer.write('entrenamientoEigenFaceRecognizer.xml')


final_time = time()
total_time = final_time - initial_time
print('Termino el entrenamiento.')
print('Tardo {} segundos en realizar el entrenamiento.\n'.format(total_time))
