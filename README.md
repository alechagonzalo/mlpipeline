
# Machine Learning Pipeline para reconocimiento de objetos.

Este repositorio contiene un pipeline generado para poder identificar productos en estanterias de un supermercado, a partir de imagenes capturadas por un robot. Es parte del Proyecto Integrador **"Robot de análisis y control de stock"**, de la carrera de Ingeniería en Computación de los alumnos Gonzalo Alecha  y Martin Ferreiro, en la Facultad de Cs. Exactas, Fisicas y Naturales (UNC).

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


## Resultados obtenidos


## Trabajos a futuro
