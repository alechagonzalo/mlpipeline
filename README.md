
# Machine Learning Pipeline para reconocimiento de objetos.

Este repositorio contiene un pipeline generado para poder identificar productos en estanterias de un supermercado, a partir de imagenes capturadas por un robot. Es parte del Proyecto Integrador **"Sistema inteligente de relevamiento de stock"**, de la carrera de Ingeniería en Computación de los alumnos Gonzalo Alecha  y Martin Ferreiro, en la Facultad de Cs. Exactas, Fisicas y Naturales (UNC).

## Análisis del problema
Se desea poder identificar  distintos productos que se encuentran en una estanteria de supermercado. Para esto, primero se realizó una implementación de un robot que realiza la tarea de recorrer la gondola, capturando imagenes. Estas imagenes son unidas con un algoritmo que permite generar una unica imagen final, que contendrá todos los objetos a reconocer.

Una primer aproximación para el reconocimiento de objetos sería la de utilizar implementaciones como YOLO (basadas en una CNN supervisada). Pero existe un inconveniente debido a que los productos cambian su packaging constantemente, se dificulta entrenar una red neuronal convolucional para que detecte y clasifique los mismos, ya que deberíamos estar entrenandola en cada cambio que se haga sobre los envases de los productos. Además, deberiamos tener una gran cantidad de imagenes por producto, lo cual es una tarea costosa que deseamos evitar.

Luego de realizar una investigación, se optó por elegir una solución planteada por Alessio Tonioni en su trabajo *["A deep learning pipeline for product recognition on store shelves"](https://www.researchgate.net/publication/328063285_A_deep_learning_pipeline_for_product_recognition_in_store_shelves)*, la cual se centra en evitar la gran cantidad de imagenes que requiere una red totalmente supervisada. 

Primero se implementa una Mask R-CNN que permite identificar formas de productos (Botellas, Cajas, Latas, etc). Esta red no tendrá el objetivo de realizar una clasificación minusiosa de cada producto en particular, si no que deberá identificar objetos y determinar la forma que poseen.

Cada objeto identificado, será comparado con imagenes de referencia que se tendrá de los productos (aproximadamente 10). De esta forma, la red no tendrá que ser reeentrenada, ni tampoco tendremos que utilizar una gran cantidad de imagenes por producto, ya que con 10 es suficiente para realizar las detecciones.

## Implementación de detector de formas
Para  realizar la detección de formas, se utilizó una implementación de Mask-RCNN hecha por [Matterport](https://github.com/matterport/Mask_RCNN), la cual hace uso de Python 3, Keras y TensorFlow. El modelo genera cuadros delimitadores y máscaras de segmentación para cada instancia de un objeto en la imagen. Se basa en Feature Pyramid Network (FPN) y una red troncal ResNet101.

En primer lugar, se generó el codigo  `Mask_RCNN/samples/Box/box.py `  que nos permitirá entrenar la red para que tenga la capacidad de detectar dos clases: *box* y *bottle*. Dentro de esta carpeta se encuentra el dataset usado para el entrenamiento, el cual consta de imagenes de distintas cajas y botellas, que fueron etiquetadas con la herramienta [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/).

![Etiquetado de objetos con VIA tool](https://i.ibb.co/K20jBv4/Screenshot-from-2019-08-15-20-40-35.png)

Luego de 30 epochs, el detector de formas adquirió la capacidad de reconocer productos en estas dos categorías. En la imagen a continuación generada con `Mask_RCNN/regiones.py `, se observa que algunos productos son reconocidos bajo la clase *box* cuando realmente son sprays, pero esto no significa un problema ya que actualmente trabajaremos con botellas y cajas.

![Detección de productos con Mask-RCNN luego de 30 épocas](https://i.ibb.co/x6rpn8t/Screenshot-from-2019-08-16-00-46-38.png)
A partir de este momento, somos capaces de reconocer objetos con estas dos formas en la imagen de entrada, y de extraer las regiones de los mismos, para su posterior procesamiento.
Por ejemplo, en la siguiente imagen de entrada tenemos todos los productos a reconocer por nuestro sistema: Botella de Pepsi Zero, Botella de Pepsi Regular, Caja de Raspberri Pi, Caja de Freescale, Botella de Mr. Musculo y Caja de UHU.

![](https://i.ibb.co/Dt0wZTS/pepsi-C003751.jpg)
Luego de pasar por el detector, y extraer las regiones de cada objeto tendremos:

![](https://i.ibb.co/2sGTpdB/New-Project-23.png)
Por cada producto se generará una imagen en particular, para una mejor comprensión se muestran todas juntas.

## Generación de imagenes de referencia

Para proceder en la creación de nuestro pipeline, debemos generar las imágenes de referencia que representen a cada producto a identificar, las cuales se utilizaran para ser comparadas con las regiones obtenidas en el detector. 
Debemos tener en cuenta que estas imágenes no pueden tener una calidad totalmente diferente a las que ingresan al detector, ya que necesitamos que exista la mayor similitud posible, para que el clasificador funcione correctamente.
Por cada producto se obtienen aproximadamente 10 imágenes de referencia, eliminando el fondo de la misma como se ve a continuación:

![](https://i.ibb.co/C25yf4G/New-Project-2.jpg)

## Filtro de color

La confusión de una botella de Pepsi Común con Pepsi Zero fue el error más frecuente. Este error eracomprensible, ya que ambos tienen la misma forma, y mismo color (en gran parte de su superficie); el único cambio notable, es la diferencia de color de su etiqueta y la de su tapa.
Para solucionar esto, y evitar cualquiera falsa detección de la misma naturaleza, se planteo una solución basada en la composición de colores RGB de los objetos detectados.
Lo que se realizo, fue separar las imágenes de referencia teniendo en cuenta el color RGB predominante en cada una. Es decir que la base de datos de referencia quedo divida en seis grupos: una primera división, en cajas y botellas; y una segunda, en tres subgrupos de acuerdo al color rojo, verde
o azul que predomina.
De esta manera, a la hora de realizar las detecciones, se obtiene el color RGB predominante de cada uno de los elementos obtenidos de la red MASK R-CNNN. Teniendo en cuenta el tipo de objeto (caja o botella) y del color RGB, se realiza la comparación de la feature obtenida, con el grupo de referencia correspondiente.
Para la obtención del color RGB dominante, se utilizo la librería de Python Scikit-Image. 

```python

    img = io.imread("imagen.png")
    average = img.mean(axis=0).mean(axis=0)
    average=list(average)
    if len(average)==4:
        average=average[:-1]

    # maximo=0 -> Red, maximo=1 -> Green,  maximo=2 -> Blue  
    maximo=average.index(max(average))

```

## Extracción de features
Para poder comparar dos imágenes, necesitamos caracterizar las mismas de forma matemática. Esto se logra a través de las features, las cuales son patrones simples, a partir de los cuales podemos describir lo que vemos en la imagen. Por ejemplo, el ojo de gato será una característica en una imagen de un gato. El papel principal de las features es transformar la información visual en el espacio vectorial. Esto nos da la posibilidad de realizar operaciones matemáticas en ellas, por ejemplo, encontrar vectores similares.

Continuando con la arquitectura elegida, el extractor de features se implementará a través de una red *VGG16* preentrenada con *ImageNet*. Esta red tiene una capa de entrada que toma una imagen de 224x224x3, y una capa de salida que es una predicción *softmax* de 1000 clases.
Este proceso se realizó también con un modelo *ResNet 50*, obteniendo resultados similares a *VGG16*, por lo que se determinó utilizar este último.

Como nosotros no queremos realizar una predicción, lo que hacemos es obtener la salida de la ultima capa de *max polling*, la cual contendrá las features de la imagen ingresada, con un tamaño de 7x7x512.
La extracción de las features se lleva a cabo en `Mask_RCNN/vgg.py `.

```python
model = VGG16(weights='imagenet', include_top=False)

def obtainFeatures (img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    return vgg16_feature
```
En definitiva, por cada imagen de referencia se tendrá un archivo *.npy* (Numpy) que contendrá la feature asociada a esa imagen.

## Comparacion de features

Obtenidas las features de las imágenes de referencia, junto con las features de las regiones candidatas a contener un producto, se procede a realizar la comparación de las mismas.
Como las features serán vectores de 25088 elementos, la comparación se puede realizar a través de la similitud de coseno. Matemáticamente, esta hace referencia a la similitud existente entre dos vectores, en un espacio que posee un producto interior con el que se evalúa el valor del coseno del
ángulo comprendido entre ellos. 

Para cada feature de una región se calcula la similitud de coseno con cada uno de las features de referencia. Un detalle a tener en cuenta, es que las imágenes de referencia están clasificadas en dos subclases (Box o Bottle). De esta manera, la comparación se llevará a cabo a partir de la forma reconocida en la Mask R-CNN, la cual determinará la clase del producto.

Una vez comparada la feature de una región con todas las features de referencia, se genera un ranking descendente, que contiene la clase a la cual pertenece la feature de referencia, y el valor de similitud (1-distancia calculada). Luego, se seleccionan los 3 elementos posicionados en la cima del ranking, y se evalúa la ocurrencia de la clase. Se puede dar el caso que una clase se repita 2/3 de las veces, y será determinada como la clase del producto. En el caso que las tres primeras posiciones se encuentren 3 clases distintas, se seleccionará la que tenga mayor valor de similitud (posición mas alta en el ranking).

El proceso anteriormente nombrado se denomina K-NN (K-Nearest Neighbor), el cual es un método que posibilita clasificar objetos con un aprendizaje "vago", ya que su entrenamiento solo requiere de ejemplos cercanos en el espacio de los elementos. El valor K determinará con cuantos vecinos será
comparado el elemento del cual se quiere obtener la clase. En nuestra implementación se trabajó con un valor de K=3.

Es necesario destacar que el porcentaje de similitud que se obtiene dependerá de que tan similar sea la imagen de referencia con la región detectada. Por ese mismo motivo, las imágenes de referencia deben ser tomadas en condiciones de luz similares, y con una cámara similar a la que luego obtendrá
fotos de las góndolas.

```python
def takeSecond(elem):
    return elem[1]

def comparator(featureToCompare,color,aspect):
	files=[]
	ranking=[]
	# obtain names of reference features


	# si existe filtro de color, comparar con todas las features
	if color !=False:
		pathFeatures= "./refRobot/"+featureToCompare[1]+"/"+str(color)+"/"
		for r, d, f in os.walk(pathFeatures):
			for file in f:
				if '.npy' in file:
					files.append(os.path.join(r, file))		

	else:
		pathFeatures= "./refRobot/"+featureToCompare[1]+"2/"
		for r, d, f in os.walk(pathFeatures):
			for file in f:
				if '.npy' in file:
					files.append(os.path.join(r, file))	

	if len(files)==0:
		return False
		
	maximo=[0,0]

	#print(files)

	# obtain simility value between reference features
	for f in files:
		reference= np.load(f)
		# obtengo la relacion de aspecto de la feature
		rel=f[f.rfind("-")+1:f.rfind(".")]
		
		cos=0

		a=featureToCompare[0][0].flatten('F')
		b= reference[0].flatten('F')
		dot = np.dot(a, b)
		norma = np.linalg.norm(a)
		normb = np.linalg.norm(b)
		cos = dot / (norma * normb)
	

		# guardo el nombre de la feature, su similitud y su relacion de aspecto
		ranking.append([f[f.rfind("/")+1:f.find("-")],cos,float(rel)])

	#ordeno el ranking de acuerdo a la similitud	
	ranking.sort(key=takeSecond,reverse=True)

	return ranking
```

## Ejecucion

Para ejecutar el programa (utilizando el filtro de color) se debe ejecutar el siguiente comando `python3 detector.py detector --color=True`.
Una vez que se ejecuta el programa pide el ingreso de una imagen a reconocer. Luego de unos segundos se realiza la detección, entregando los resultados de la misma, y además, se genera una imagen (reconocimiento.png) con las regiones detectadas por el detector de formas.


## Resultados obtenidos
A continuación se muestran pruebas realizadas con distintas imágenes, y los resultados obtenidos de la detección. Se presenta la imagen de entrada capturada por el robot, las detecciones realizadas por el detector de formas, y por ultimo, la salida en consola con las detecciones de los productos.

### Prueba A

![](https://i.ibb.co/2sGTpdB/New-Project-23.png)
![](https://i.ibb.co/hLG26XM/3751.png)
![](https://i.ibb.co/Dtpk04n/3751mask.png)
![](https://i.ibb.co/16F2JXg/3751cons.png)

### Prueba B

![](https://i.ibb.co/SrWL3zQ/4200.png)
![](https://i.ibb.co/gT0Bw05/4200mask.png)
![](https://i.ibb.co/TqsT18Y/4200cons.png)

### Prueba C

![](https://i.ibb.co/hBZLT3q/4613.png)
![](https://i.ibb.co/pfygBw0/4613mask.png)
![](https://i.ibb.co/yFtHPjV/4613cons.png)

### Prueba D

![](https://i.ibb.co/3s55YNv/4624.png)
![](https://i.ibb.co/JsFP536/4624cons.png)
![](https://i.ibb.co/WfddV2n/4624mask.png)

### Prueba E

![](https://i.ibb.co/RgpHvK7/5215.png)
![](https://i.ibb.co/LSDVrTc/5215cons.png)
![](https://i.ibb.co/GdM53BC/5215mask.png)

