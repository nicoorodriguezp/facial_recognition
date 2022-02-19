# Facial Recognition using Python, CV2, Numpy
* Developed by: Nicolas Gaston Rodriguez Perez

Source code to perform a facial recognition in python, capturing the faces of people and training the machine to recognize who is each one of them.

You can see examples of the results at the end of this document.



### Three layers can be visualized in the code:

1. Input Layer: where the faces of the people are captured.

2. Hidden Layer: Where the machine is taught to identify people.

3. Output Layer: Where the result of the machine learning is shown and it can be visualized how it identifies the different faces.




### You can also find other folders:

1. face_noices: here are the .xml files containing already learned data for recognizing faces and removing noise from other objects.

2. image_process_methods: Here you can find a small module that contains methods that can be applied to change the morphology and the number of pixels of the photos.

3. capturas: Here are stored the folders containing the captured images of people.


The code will generate the file "trainingEigenFaceRecognizer.xml", which contains everything learned by the machine about people's faces.

You can process faces from the camera or from an mp4 video.

# Console
<h3> python inputLayer.py </h3>

![alt text](https://github.com/nicoorodriguezp/facial_recognition/blob/main/EjemplosResultados/inputLayer.png)

<h3> python hiddenLayer.py </h3>

![alt text](https://github.com/nicoorodriguezp/facial_recognition/blob/main/EjemplosResultados/hiddenLayer1.png)
![alt text](https://github.com/nicoorodriguezp/facial_recognition/blob/main/EjemplosResultados/hiddenLayer2.png)

# Results
![alt text](https://github.com/nicoorodriguezp/facial_recognition/blob/main/EjemplosResultados/outputLayer.png)
![Nicolas Gaston Rodriguez Perez](https://github.com/nicoorodriguezp/facial_recognition/blob/main/EjemplosResultados/outputLayer2.png)

# Docs

https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html

https://numpy.org/doc/stable/user/index.html#user

# Contact me
https://www.linkedin.com/in/nicogrp/
