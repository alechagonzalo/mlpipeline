# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import argparse
from PIL import Image
import datetime

import skimage.draw
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from vgg import obtainFeatures
from comparador import comparator
import os


def takeSecond(elem):
    return elem[1]

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
               (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# define 81 classes that the coco model knowns about
class_names = ['BG', 'box', 'bottle']

classes = ["Raspberry", "Pepsi Zero", "FreeScale", "UHU",
    "Mr Musculo", "Blem Electrica", "Pepsi Comun"]

# define the test configuration


class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 2


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image',
                metavar="path or URL to image",
                help='Image to apply the color splash effect on')

parser.add_argument('--output',
                metavar="path to save image",
                help='Path where save the output file')
args = parser.parse_args()

# Validate arguments

umbralDetection = 0.75

assert args.image
"Se requiere imagen de entrada"
assert args.output
"Se requiere path de salida"

image = args.image
output = args.output
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('trainingW/mask_rcnn_object_0050.h5', by_name=True)
# load photograph
img = load_img(image)
img = img_to_array(img)
# make prediction
print("Realizando detecciones....")
results = rcnn.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
N = r['rois'].shape[0]


imagenConNegro = display_instances(
    img, r['rois'], r['masks'], r['class_ids'], class_names, output+image, r['scores'])

imagenn = Image.fromarray(imagenConNegro.astype('uint8'), 'RGB')
# plt.imshow(imagenn)
# plt.savefig('foo.png')
# plt.show()
for i in range(N-1, -1, -1):
     if float(r['scores'][i]) < umbralDetection:
          print("elimino deteccion con "+str(float(r['scores'][i])))
          r['scores']= np.delete(r['scores'],i)
          r['class_ids']=np.delete(r['class_ids'],i)
          r['rois']=np.delete(r['rois'],i,axis=0)
          r['masks']=np.delete(r['masks'],i,axis=0)

N=r['scores'].shape[0]

print("Generando "+str(N)+" recortes....")
printProgressBar(0, N, prefix = 'Progreso:', suffix = 'Completo', length = 50)

pathsave="recortes/"+image+"-{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())


try:
    os.mkdir(pathsave)
except OSError:
    print ("Creation of the directory %s failed" % pathsave)
else:
    print ("Successfully created the directory %s \n" % pathsave)

for i in range (N):
     file_name = pathsave+"/"+class_names[r['class_ids'][i]]+"-score:"+  str(r['scores'][i]) +"-{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
     imagenn.crop((r['rois'][i][1], r['rois'][i][0],r['rois'][i][3],r['rois'][i][2])).save(file_name)
     printProgressBar(i + 1, N, prefix = 'Progreso:', suffix = 'Completo', length = 50)

print("\nObteniendo features de recortes")

path = pathsave+"/"
files = []
for r, d, f in os.walk(path):
     for file in f:
          if '.png' in file:
               files.append(os.path.join(r, file))

# vector que contendra las features
features=[]
printProgressBar(0, len(files), prefix = 'Progreso:', suffix = 'Completo', length = 50)
for index,f in enumerate(files):
     # np.save("./ReferenceFeatures/"+f[37:]+".npy", obtainFeatures(f))
     f2= f.replace(path,"")
     feature= [obtainFeatures(f),f2[:f2.find("-")]]
     features.append([f2,feature]) 
     printProgressBar(index + 1, len(files), prefix = 'Feature '+ str(index+ 1)+"/"+str(len(files)), suffix = 'Completo', length = 50)

print("\nClasificando productos")
printProgressBar(0, len(features), prefix = 'Progreso:', suffix = 'Completo', length = 50)

detecciones=[]

for index,featureToCompare in enumerate(features):
     
     detecciones.append([featureToCompare[0],comparator(featureToCompare[1])])
     printProgressBar(i + 1, len(features), prefix = 'Producto '+ str(index + 1)+"/"+str(len(features)), suffix = 'Completo', length = 50)


print("\nLa imagen ingresada contiene:")
k=3
for index,deteccion in enumerate(detecciones):
# eleccion de K vecinos más cercanos
     #print(str(index+1)+" "+classes[detecciones[index][1]]+" con "+str(round(detecciones[index][0]*100,2))+"%")
     print("\nPara imagen "+deteccion[0]+":")
     print('\n'.join('{}: {}'.format(*k) for k in enumerate(deteccion[1][:k])))
     rank = deteccion[1][:k]
     knns=[]

     for i in range(len(rank)):
          knns.append(rank[i][0])
     
     print(knns)
     value = max(knns,key=knns.count)
     print("Con k="+str(k)+" tenemos que el recorte es: "+classes[int(value)])
     

