'''
Codigo de la CAPA DE ENTRADA: Entrada de informacion.

Las redes neuronales tienen 3 capas:

1) Capa de Entrada.
2) Capa Oculta.
3) Capa de salida.

@author: Nicolas Gaston Rodriguez Perez
'''
import cv2 as cv
import os
import shutil
import image_process_methods.funciones as image


def createFolder(capture_path):
    if os.path.exists(capture_path):

        shutil.rmtree(capture_path)
        os.makedirs(capture_path)

    elif not os.path.exists(capture_path):
        os.makedirs(capture_path)


user_name = 'mama'
ruta = "capturas/{}_model".format(user_name)
cantidadCapturas = 300

createFolder(ruta)


# Obtengo todos los ruidos que puede tener una cara a su alrededor
ruidos = cv.CascadeClassifier(
    'face_noices/haarcascade_frontalface_default.xml'
)

# # abro la camara interna
# camara = cv.VideoCapture(0)
camara = cv.VideoCapture('messi.mp4')

# Verifico que se abrio alguna camara.
if not camara.isOpened():
    print("No se encontro una camara.\n")
    exit()

id = 0  # id de las capturas
print('\n>>>>>>>>>>>>> Inicio de captura de Rostro <<<<<<<<<<<<<<<<<<\n')

while True:

    respuesta, captura = camara.read()

    try:
        # Convierto la captura de video a escala de grises
        grises = image.get_greyscale(captura)
        # Copio las propiedades de la captura procesada
        idcaptura = grises.copy()

    except:
        print(
            "Warning: El video no tiene la duracion suficiente para sacar {} fotos. \n"
            .format(cantidadCapturas)
        )
        print("Las fotos han sido capturadas correctamente durante toda la duracion del video.")
        print("Puede proceder a entrenar las {} fotos obtenidas.\n".format(id+1))
        break

    '''
    Aca comparo los ruidos con lo que vendria a ser una cara.
    Las otras escalas representan en porcentaje el tamano que tendria una cara. Esto ayuda a identificar
    cuales podrian ser caras.

    1.0 = 100%
    1.5 = 50%
    1.3 = 30%
    '''

    cara = ruidos.detectMultiScale(grises, 1.3, 5)

    '''
    coordenadas x e y
    e1 = esquina superior izquierda
    e2 = esquina inferior derecha

    Recorro de esquina a esquina(diagonal), de izquierda a derecha y, de arriba a abajo.
    '''
    for(x, y, e1, e2) in cara:
        # Dibujo un rectangulo alrededor de la cara
        # rectangulo: captura, ancho y largo, las esquinas, color del rectangulo, grosor del marco
        cv.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)

        # y: y mas la altura, x = x + el ancho
        # Establezco en donde va a realizar la captura de la foto
        rostrocapturado = idcaptura[y:y+e2, x:x+e1]

        # Modifico el tamano de la captura
        rostrocapturado = cv.resize(
            rostrocapturado,
            (160, 160),
            interpolation=cv.INTER_CUBIC
        )

        # Guardo la captura y cambio el index para la proxima

        cv.imwrite(
            # donde lo guardo + nombre archivo
            ruta+'/img_{}.jpg'.format(id),
            rostrocapturado  # que es lo que guardo
        )
        id = id+1

    # Muestro la Captura
    cv.imshow("Resultado Rostro", captura)

    # Una vez que la cantidad de captura llegue a la que queremos, se termine el bucle.
    if id == cantidadCapturas:
        print('Las fotos han sido capturadas correctamente.')
        print("Puede proceder a entrenar las fotos obtenidas.\n")
        break
camara.release()
cv.destroyAllWindows()
