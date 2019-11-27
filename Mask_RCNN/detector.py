# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import blackbackground,display_instances2
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import argparse
from PIL import Image,ImageEnhance
import datetime

from PIL import ImageFont
from PIL import ImageDraw 
import skimage.draw
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from vgg import obtainFeatures
from comparador import comparator
import os
import sys
import cv2
from skimage import io

import time

# define 81 classes that the coco model knowns about
class_names = ['BG', 'box', 'bottle']

classes = ["Raspberry", "Pepsi Zero", "FreeScale", "UHU",
    "Mr Musculo", "NO ESPECIFICO", "Pepsi Comun"]

# define the test configuration


global filtrocolor

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 2
     #IMAGE_MAX_DIM = 640
     #IMAGE_MIN_DIM = 256
     DETECTION_MIN_CONFIDENCE = 0.75
     RPN_ANCHOR_RATIOS = [0.2, 0.5, 1]


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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def deteccion(modelo,image):

     #im = Image.open(image)
     #enhancer = ImageEnhance.Sharpness(im)
     #enhanced_im = enhancer.enhance(20.0)
     #image2= image[:image.rfind(".")]+"2"+image[image.rfind("."):]
     #enhanced_im.save(image2)



     # load photograph
     img = load_img(image)
     img = img.resize((int(img.size[0]*0.5),int(img.size[1]*0.5)),Image.ANTIALIAS)
     img = img_to_array(img)
     # make prediction
     print("Realizando detecciones....")
     results = modelo.detect([img], verbose=0)
     
     # get dictionary for first prediction
     r = results[0]
     display_instances2(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
     N = r['rois'].shape[0]

     #guardar imagen reconocida
     #display_instances2(img, r['rois'], r['masks'], r['class_ids'], class_names,r['scores'])

     N=r['scores'].shape[0]

     print("Generando "+str(N)+" recortes....")
     printProgressBar(0, N, prefix = 'Progreso:', suffix = 'Completo', length = 50)

     pathsave="recortes/"+image[image.rfind("/")+1:]+"-{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())


     try:
          os.mkdir(pathsave)
     except OSError:
          print ("Creation of the directory %s failed" % pathsave)
     else:
          print ("Successfully created the directory %s \n" % pathsave)
     
     ratios=[]

     
     
     for i in range (N):
          
          ImgParam=np.copy(img)
          imagen=blackbackground(ImgParam,r['rois'][i],r['masks'][:,:,i])
          x1=r['rois'][i][1]
          y1=r['rois'][i][0]
          x2=r['rois'][i][3]
          y2=r['rois'][i][2]
          width=x2-x1
          height=y2-y1
          ratio= round(float(width/height),3)
          file_name = pathsave+"/"+class_names[r['class_ids'][i]]+"-score:"+str(r['scores'][i]) +"-{:%Y%m%dT%H%M%S}-".format(datetime.datetime.now())+str(ratio) +".png"
          imagenn = Image.fromarray(imagen.astype('uint8'), 'RGB')
          imagenn.crop((r['rois'][i][1], r['rois'][i][0],r['rois'][i][3],r['rois'][i][2])).save(file_name)
          printProgressBar(i + 1, N, prefix = 'Progreso:', suffix = 'Completo', length = 50)
          
     

     return pathsave

def knn(pathsave):

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
          rel=f[f.rfind("-")+1:f.rfind(".")]
          f2= f.replace(path,"")
          feature= [obtainFeatures(f),f2[:f2.find("-")]]

          print(filtrocolor)

          if filtrocolor==True:
               img = io.imread(f)
               average = img.mean(axis=0).mean(axis=0)
               average=list(average)
               if len(average)==4:
                    average=average[:-1]
               maximo=average.index(max(average))

               if maximo == 0:
                    color="r"
               elif maximo == 1:
                    color="g"
               elif maximo==2:
                    color="b"    
          
               features.append([f2,feature,color,float(rel)]) 
          else:
               features.append([f2,feature,False,float(rel)]) 


          printProgressBar(index + 1, len(files), prefix = 'Feature '+ str(index+ 1)+"/"+str(len(files)), suffix = 'Completo', length = 50)

     print("\nClasificando productos")
     printProgressBar(0, len(features), prefix = 'Progreso:', suffix = 'Completo', length = 50)

     detecciones=[]
   
     for index,featureToCompare in enumerate(features):

          detecciones.append([featureToCompare[0],comparator(featureToCompare[1],featureToCompare[2],featureToCompare[3]),featureToCompare[2]])
          printProgressBar(index + 1, len(features), prefix = 'Producto '+ str(index + 1)+"/"+str(len(features)), suffix = 'Completo', length = 50)

     k=3

     productos=[]
     for index,deteccion in enumerate(detecciones):
     # eleccion de K vecinos más cercanos
          #print(str(index+1)+" "+classes[detecciones[index][1]]+" con "+str(round(detecciones[index][0]*100,2))+"%")
          print("\nPara imagen "+deteccion[0]+":")
          #img = Image.open(path+deteccion[0])
          #draw = ImageDraw.Draw(img)
          #font = ImageFont.truetype('./Roboto-Bold.ttf', 16)

          if deteccion[1] != False:
               print('\n'.join('{}: {}'.format(*k) for k in enumerate(deteccion[1][:k])))
               rank = deteccion[1][:k]
               knns=[]

               for i in range(len(rank)):
                    knns.append(rank[i][0])
               
               value = max(knns,key=knns.count)
               print()
               print("Con k="+str(k)+" tenemos que el recorte es: "+bcolors.OKGREEN +classes[int(value)]+ bcolors.ENDC)
               productos.append(int(value))

               recorteratio=float(deteccion[0][deteccion[0].rfind("-")+1:deteccion[0].rfind(".")])
               knnratio=float(deteccion[1][0][2])

               if abs(recorteratio-knnratio) > 0.3:
                    print(bcolors.FAIL + "\nAtención: Es posible que existan más productos en la detección" 
      + bcolors.ENDC)
                    #draw.text((0, 0),str(int(value))+": "+classes[int(value)]+", ATENCION",(255,255,255),font=font)
               #else:
                    #draw.text((0, 0),str(int(value))+": "+classes[int(value)],(255,255,255),font=font)
          else:
               print("No existen features para el color dominante en el recorte")
               #draw.text((0, 0),"No features",(255,255,255),font=font)

          #img.save(path+deteccion[0])





     return productos


# Lee archivo de texto y obtiene los objetos que componen cada imagen. 
# Luego, detecta cada imagen del path, y compara las detecciones realizadas
# con las especificadas en el archivo.
# Con esto obtenemos la precisión de nuestro sistema

def test(rcnn,path):
     f = open(path+"/pruebas.txt", "r")

     testimages=[]


     fl =f.readlines()
     for x in fl:
          filename= path+x[:x.find("-")]
          fileproducs=[]

          x = x[x.find("-"):]
          while x.find("-") != -1:
               x = x[x.find("-")+1:]
               clase = x[:x.find("-")]
               x = x[x.find("-"):]
               fileproducs.append(int(clase))
          testimages.append([filename,fileproducs])
     

     precision=0
     recall=0

     for i in range(len(testimages)):
          print("\n============ Deteccion de imagen =================")
          path = deteccion(rcnn,testimages[i][0])
          detecciones = knn(path)
     return



if __name__ == '__main__':
     # Parse command line arguments
     start_time = time.time()

     parser = argparse.ArgumentParser()
     parser.add_argument("command",
                        metavar="<command>",
                        help="'detector' or 'test'")
     parser.add_argument('--image',
                    metavar="path or URL to image",
                    help='Image to apply the color splash effect on')
     parser.add_argument('--folder',
                    metavar="path or URL to image",
                    help='Image to apply the color splash effect on')

     parser.add_argument('--color',
                    metavar="path or URL to image",
                    help='')

     args = parser.parse_args()

     
     # define the model
     rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
     # load coco model weights
     weightpath='trainingW/mask_rcnn_object_0060.h5'
     
     rcnn.load_weights(weightpath, by_name=True)
     print("%s s. --- Pesos cargados" % (time.time() - start_time))

     if args.color =="True":
          filtrocolor=True
     else:
          filtrocolor=False

     if args.command == "detector":
          while True:
               print('Ingrese nombre de imagen:')
               image = input()
               if os.path.isfile(image) == True:
                    recortespath = deteccion(rcnn,image)
                    prods= knn(recortespath)
                    print("%s s. --- La imagen ingresada contiene:" % (time.time() - start_time))

                    for p in prods:
                         print(bcolors.BOLD + str(" "+classes[p])+ bcolors.ENDC)
               else:
                    print("No existe imagen con ese nombre")
     

     elif args.command == "test":
          folder = args.folder
          test(rcnn,folder)

     print("\n \n")
     #print("Tiempo de ejecución: "+ str(round(end-start,2))+" seg.")
