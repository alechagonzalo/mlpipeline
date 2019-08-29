from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.resnet50 import ResNet50

model = VGG16(weights='imagenet', include_top=False)
#model = ResNet50(weights='imagenet', pooling=max,include_top = False) 
#model.summary()



def obtainFeatures (img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)

    return vgg16_feature
