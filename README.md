# Machine Learning Pipeline para reconocimiento de objetos.

Este repositorio contiene un pipeline generado para poder identificar productos en estanterias de un supermercado, a partir de imagenes capturadas por un robot. 

## Análisis del problema
Se desea poder identificar  distintos productos que se encuentran en una estanteria de supermercado. Para esto, primero se realizó una implementación de un robot que realiza la tarea de recorrer la gondola, capturando imagenes. Estas imagenes son unidas con un algoritmo que permite generar una unica imagen final, que contendrá todos los objetos a reconocer.

Una primer aproximación para el reconocimiento de objetos sería la de utilizar implementaciones como YOLO (basadas en una CNN supervisada). Pero existe un inconveniente debido a que los productos cambian su packaging constantemente, se dificulta entrenar una red neuronal convolucional para que detecte y clasifique los mismos, ya que deberíamos estar entrenandola en cada cambio que se haga sobre los envases de los productos. Además, deberiamos tener una gran cantidad de imagenes por producto, lo cual es una tarea costosa que deseamos evitar.

Luego de realizar una investigación, se optó por elegir una solución planteada por Alessio Tonioni, la cual se centra en evitar la gran cantidad de imagenes que requiere un detector como YOLO. 

Primero se implementa una Mask R-CNN que permite identificar formas de productos (Botellas, Cajas, Latas, etc). Esta red no tendrá el objetivo de realizar una clasificación minusiosa de cada producto en particular, si no que deberá identificar objetos y determinar la forma que poseen.

Cada objeto identificado, será comparado con imagenes de referencia que se tendrá de los productos (aproximadamente 10). De esta forma, la red no tendrá que ser reeentrenada, ni tampoco tendremos que utilizar una gran cantidad de imagenes por producto, ya que con 10 es suficiente para realizar las detecciones.

## Extraccion de features


```python
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.resnet50 import ResNet50

model = VGG16(weights='imagenet', include_top=False)

def obtainFeatures (img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)

    return vgg16_feature
    
```



## Comparacion de features
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def takeSecond(elem):
    return elem[1]

def comparator(featureToCompare):
	files=[]
	ranking=[]

	pathFeatures= "./refRobot/"+featureToCompare[1]+"/"

	#features resnet50
	#pathFeatures= "./imgRef/"+featureToCompare[1]+"/"
	for r, d, f in os.walk(pathFeatures):
		for file in f:
			if '.npy' in file:
				files.append(os.path.join(r, file))

	maximo=[0,0]

	for f in files:
		reference= np.load(f)
		cos=0

		a=featureToCompare[0][0].flatten('F')
		b= reference[0].flatten('F')
		dot = np.dot(a, b)
		norma = np.linalg.norm(a)
		normb = np.linalg.norm(b)
		cos = dot / (norma * normb)	
		
		ranking.append([f[f.rfind("/")+1:f.find("-")],cos])
	ranking.sort(key=takeSecond,reverse=True)
	return ranking
```
