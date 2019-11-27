from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.resnet50 import ResNet50
import os
model = VGG16(weights='imagenet', include_top=False)
#model = ResNet50(weights='imagenet', pooling=max,include_top = False) 
#model.summary()


img_path="./refRobot/bottle/r/"

files=[]

for r, d, f in os.walk(img_path):
    for file in f:
        if '.png' in file:
            if not '.npy' in file:
                files.append(os.path.join(r, file))



for f in files:

    img = image.load_img(f)
    img_data = image.img_to_array(img)
    print(img_data.shape)
    width=img_data.shape[1]
    height=img_data.shape[0]
    ratio= round(float(width/height),3)

    del img, img_data

    img = image.load_img(f, target_size=(224, 224))
    
    img_data = image.img_to_array(img)
    print(img_data.shape)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)

    np.save(f+"-"+str(ratio)+".npy",vgg16_feature)
