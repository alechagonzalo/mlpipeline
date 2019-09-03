# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import blackbackground
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
import sys
import cv2
from skimage import io

import time

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


def deteccion(modelo,image):
     start2=time.time()
     umbralDetection = 0.75
     # load photograph
     img = load_img(image)
     img = img.resize((int(img.size[0]*0.5),int(img.size[1]*0.5)),Image.ANTIALIAS)
     img = img_to_array(img)
     # make prediction
     print("Realizando detecciones....")
     results = modelo.detect([img], verbose=0)
     # get dictionary for first prediction
     r = results[0]
     N = r['rois'].shape[0]
     end2=time.time()     
     print(end2-start2)

     #imagenConNegro = display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
     #plt.imshow(imagenn)
     #plt.savefig('foo.png')
     #plt.show()

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

     pathsave="recortes/"+image[image.rfind("/")+1:]+"-{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())


     try:
          os.mkdir(pathsave)
     except OSError:
          print ("Creation of the directory %s failed" % pathsave)
     else:
          print ("Successfully created the directory %s \n" % pathsave)

     
     for i in range (N):
          
          ImgParam=np.copy(img)
          imagen=blackbackground(ImgParam,r['rois'][i],r['masks'][:,:,i])
          file_name = pathsave+"/"+class_names[r['class_ids'][i]]+"-score:"+  str(r['scores'][i]) +"-{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
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
          f2= f.replace(path,"")
          feature= [obtainFeatures(f),f2[:f2.find("-")]]

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

          features.append([f2,feature,color]) 
          printProgressBar(index + 1, len(files), prefix = 'Feature '+ str(index+ 1)+"/"+str(len(files)), suffix = 'Completo', length = 50)

     print("\nClasificando productos")
     printProgressBar(0, len(features), prefix = 'Progreso:', suffix = 'Completo', length = 50)

     detecciones=[]
   
     for index,featureToCompare in enumerate(features):

          detecciones.append([featureToCompare[0],comparator(featureToCompare[1],featureToCompare[2]),featureToCompare[2]])
          printProgressBar(index + 1, len(features), prefix = 'Producto '+ str(index + 1)+"/"+str(len(features)), suffix = 'Completo', length = 50)


     print("\nLa imagen ingresada contiene:")
     k=3
     productos=[]
     for index,deteccion in enumerate(detecciones):
     # eleccion de K vecinos más cercanos
          #print(str(index+1)+" "+classes[detecciones[index][1]]+" con "+str(round(detecciones[index][0]*100,2))+"%")
          print("\nPara imagen "+deteccion[0]+":")
          if deteccion[1] != False:
               print('\n'.join('{}: {}'.format(*k) for k in enumerate(deteccion[1][:k])))
               rank = deteccion[1][:k]
               knns=[]

               for i in range(len(rank)):
                    knns.append(rank[i][0])
               
               value = max(knns,key=knns.count)
               print("Con k="+str(k)+" tenemos que el recorte es: "+classes[int(value)])
               productos.append(int(value))
          else:
               print("No existen features para el color dominante en el recorte")
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
          filename= path+"/"+x[:x.find("-")]
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
          totalVisibles=len(testimages[i][1])
          detectadas=len(detecciones)
          errores=0
          correctas=0
          print("\n============ Test de imagen =================")
          print(detecciones)
          for ele in testimages[i][1]: 
               if ele in detecciones:    
                    detecciones.remove(ele)
                    correctas=correctas+1
               else:
                    errores = errores+1     

          print("De "+str(totalVisibles)+" se detectaron "+str(detectadas)+". Se cometieron "+str(errores)+" errores.")
          print("Presicion:"+str(correctas/detectadas))
          print("Recall:"+str(correctas/(correctas+errores)))

          precision=precision+(correctas/detectadas)
          recall=recall+(correctas/(correctas+errores))
     precision=precision/len(testimages)
     recall=recall/len(testimages)
     fmeasure=(precision+recall)/2
     print("============= Evaluacion del sistema =============")
     print("Con "+str(len(testimages))+" imagenes obtenemos:")
     print("Presición: "+str(precision))
     print("Recall: "+str(recall))
     print("F-Measure: "+str(fmeasure))

     return



if __name__ == '__main__':
     # Parse command line arguments
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

     args = parser.parse_args()

     if args.command == "detector":
          assert args.image
          "Se requiere imagen de entrada"
     elif args.command == "detector":
          assert args.folder
          "Se requiere imagen de entrada"

     
     
     # define the model
     rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
     # load coco model weights
     rcnn.load_weights('trainingW/mask_rcnn_object_0050.h5', by_name=True)
     
     start = time.time()
     if args.command == "detector":
          image = args.image
          recortespath = deteccion(rcnn,image)
          knn(recortespath)

     elif args.command == "test":
          folder = args.folder
          test(rcnn,folder)

     end = time.time()
     print("Tiempo de ejecución: "+ str(round(end-start,2))+" seg.")