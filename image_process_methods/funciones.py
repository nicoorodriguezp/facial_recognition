'''
Este modulo tiene el fin de tener todas las operaciones relacionadas con los filtros y procesos aplicados
en las imagenes.

La idea es importar este modulo y tener todas los metodos a mano.

@author: Nicolas Gaston Rodriguez Perez
'''

import cv2 as cv
import numpy as np


def get_greyscale(image):
    # Obteno y retorno la imagen en escala de grises
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def remove_noise_gaussianBlur(grey_image, gaussValue=1):
    '''
    Hago un suavizado de la imagen por si esta pixeleada o borrosa(ruidos de la imagen).

    Si la imagen esta pixeleada, lo que hace es desenfocarla para que se puedan notar mejor los bordes
    de cada objeto con GaussianBlur.

    Reduce los pixeles que tiene una imagen.'''
    return cv.GaussianBlur(
        grey_image,
        # parametros para cambiar el desenfoque. siempre ambos valores tiene que ser el mismo.
        (gaussValue, gaussValue),
        0
    )


def remove_noise_medianBlur(image):
    # Remuevo el ruido
    return cv.medianBlur(image, 5)


def thresholding(image):
    # Realizo la umbralizacion
    return cv.threshold(
        image,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OUTSU
    )[1]


'''

>>>>>>>>> Operaciones de Morfologia para eliminar el ruido segun donde se encuentre <<<<<<
            https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html

* Erosion:  Todos los píxeles cercanos al límite se descartarán según el tamaño del kernel.
            Es útil para eliminar pequeños ruidos blancos (como hemos visto en el capítulo de 
            espacio de color), separar dos objetos conectados, etc.

* Dilatacion:   Es justo lo contrario de la erosión. Por lo tanto, aumenta la región blanca en la imagen 
                o aumenta el tamaño del objeto en primer plano. Normalmente, en casos como la eliminación 
                de ruido, la erosión es seguida por la dilatación. orque la erosión elimina los ruidos 
                blancos, pero también encoge nuestro objeto. Entonces lo dilatamos. Dado que el ruido 
                se ha ido, no volverán, pero el área de nuestro objeto aumenta. 
                También es útil para unir partes rotas de un objeto.

* Apertura: La apertura es solo otro nombre de la erosión seguida de la dilatación . 
            Es útil para eliminar el ruido, como explicamos anteriormente. 
            BASICAMENTE: Cuando el ruido esta por fuera del kernel(objeto)

* Cierre:   El cierre es el reverso de la apertura, la dilatación seguida de la erosión . 
            Es útil para cerrar pequeños agujeros dentro de los objetos de primer plano o pequeños 
            puntos negros en el objeto. BASICAMENTE cuando el ruido esta dentro del kernel(objeto).

* Gradiente morfológico:    Es la diferencia entre dilatación y erosión de una imagen. 
                            BASICAMENTE: El resultado se verá como el contorno/marco del objeto.


El kernel value vendria a ser el objeto al cual estamos analizando.

Se tiene que ir cambiando y probando hasta encontrar cual es el que buscamos.
'''


def erosion(image, kernelValue=5):
    kernel = np.ones((kernelValue, kernelValue), np.uint8)
    return cv.erode(image, kernel, interations=1)


def dilatation(image, kernelValue=5):
    kernel = np.ones((kernelValue, kernelValue), np.uint8)
    return cv.dilate(image, kernel, interations=1)


def opening(image, kernelValue=5):
    # Erosion seguida por la dilatacion
    kernel = np.ones((kernelValue, kernelValue), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)


def close(image, kernelValue=5):
    # Erosion seguida por la dilatacion
    kernel = np.ones((kernelValue, kernelValue), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)


def morphologicalGradient(image, kernelValue=5):
    # Erosion seguida por la dilatacion
    kernel = np.ones((kernelValue, kernelValue), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)


def canny(image):
    # Deteccion de bordes
    return cv.Canny(image, 100, 200)


def deskew(image):
    # Correcion de sesgo(skew)
    coords = np.column_stack(np.where(image > 0))
    angle = cv.minAreaRect((coords)[-1])

    if angle < 45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]

    center = (w//2, h//2)

    M = cv.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv.warpAffine(
        image,
        M,
        (w, h),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE
    )

    return rotated


def match_template(image, template):
    # Matcheando Plantillas
    return cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
