import cv2
import numpy as np
from skimage import io
import os
import shutil

img_path="./refRobot/bottle/"

files=[]

for r, d, f in os.walk(img_path):
    for file in f:
        if '.png' in file:
            if not '.npy' in file:
                files.append(os.path.join(r, file))
            

for f in files:
    img = io.imread(f)
    average = img.mean(axis=0).mean(axis=0)
    average=list(average)
    if len(average)==4:
        average=average[:-1]

    maximo=average.index(max(average))
    print(f+ " "+ str(maximo))

    if maximo==0:
        #rojo
        name=f
        newpath=f[:f.rfind("/")+1]+"r"+f[f.rfind("/"):]
        os.rename(f, newpath)

        newpath=f[:f.rfind("/")+1]+"r"+f[f.rfind("/"):]+".npy"
        os.rename(f+".npy", newpath)

    elif maximo==1:
        #green
        newpath=f[:f.rfind("/")+1]+"g"+f[f.rfind("/"):]
        os.rename(f, newpath)


        newpath=f[:f.rfind("/")+1]+"g"+f[f.rfind("/"):]+".npy"
        os.rename(f+".npy", newpath)

    elif maximo==2:
        #blue
        newpath=f[:f.rfind("/")+1]+"b"+f[f.rfind("/"):]
        os.rename(f, newpath)


        newpath=f[:f.rfind("/")+1]+"b"+f[f.rfind("/"):]+".npy"
        os.rename(f+".npy", newpath)

    else:
        print("error")



