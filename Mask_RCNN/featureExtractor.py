from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.resnet50 import ResNet50
import os
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False)
#model.summary()

#model = ResNet50(weights='imagenet', pooling=max,include_top = False) 
model.summary()

img_path="./refRobot/rasp/"

files=[]

for r, d, f in os.walk(img_path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))
            


for f in files:
    img = image.load_img(f, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    np.save(f+".npy",vgg16_feature)
    print("feature guardada "+f)