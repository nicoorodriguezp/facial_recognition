'''
Codigo de la CAPA DE SALIDA: Reconocimiento Facial

En esta capa se hace el reconocimiento de las fotos.

Las redes neuronales tienen 3 capas:

1) Capa de Entrada.
2) Capa Oculta.
3) Capa de Salida.

@author: Nicolas Gaston Rodriguez Perez
'''

import cv2 as cv
import os
import image_process_methods.funciones as image

data_path = 'capturas/'
listaData = os.listdir(data_path)
print('User Data: ', listaData)

entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('entrenamientoEigenFaceRecognizer.xml')
ruidos = cv.CascadeClassifier(
    'face_noices/haarcascade_frontalface_default.xml'
)


# abro la camara interna
# camara = cv.VideoCapture(0)
camara = cv.VideoCapture('messi.mp4')

# Verifico que se abrio alguna camara.
if not camara.isOpened():
    print("No se encontro una camara.")
    exit()

id = 0  # id de las capturas

while True:

    respuesta, captura = camara.read()

    try:
        # Convierto la captura de video a escala de grises
        grises = image.get_greyscale(captura)
        # Copio las propiedades de la captura procesada
        idcaptura = grises.copy()

    except:
        print("Se termino el video. No hay mas para analizar.")
        break

    cara = ruidos.detectMultiScale(grises, 1.3, 5)

    for(x, y, e1, e2) in cara:
        # y: y mas la altura, x = x + el ancho
        # Establezco en donde va a realizar la captura de la foto
        rostrocapturado = idcaptura[y:y+e2, x:x+e1]

        # Modifico el tamano de la captura
        rostrocapturado = cv.resize(
            rostrocapturado,
            (160, 160),
            interpolation=cv.INTER_CUBIC
        )

        '''
        Hago la prediccion de a quien le pertenece la cara.

        predict devuelve dos valores: resultado[0]: es la etiqueta -  resultado[1]: es la prediccion
        '''
        resultado = entrenamientoEigenFaceRecognizer.predict(rostrocapturado)

        # Agrego el resultado en el video:
        # si la prediccion me da debajo de 3000. Mientras mas chico el numero, mas se acerca al resultado.
        if resultado[1] < 4000:
            cv.putText(
                captura,  # Donde muestra el texto: Durante la captura de video
                '{}'.format(listaData[resultado[0]]),  # EL texto a mostrar
                (x, y-30),  # Ubicacion del texto
                1, 1.3,  # Escala del texto
                (0, 255, 0),  # Color del texto
                1,  # Grosor de la caja de texto
                cv.LINE_AA  # Figura geometrica del texto: cuadrada
            )
            cv.putText(captura, '{}'.format(resultado), (x, y-5),
                       1, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)
        else:
            cv.putText(captura, 'No reconocido.', (x, y-30),
                       1, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(captura, '{}'.format(resultado), (x, y-5),
                       1, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)

    # Muestro/Actualizo el video con el resultado.
    cv.imshow('Resultado', captura)

    if cv.waitKey(1) == ord('q'):
        break

camara.release()
cv.destroyAllWindows()
