from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.resnet50 import ResNet50
import os
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False)
#model.summary()
import matplotlib.pyplot as pyplot
from keras.models import Model

ixs = [ 17]
outputs=[model.layers[i].output for i in ixs]
#mostrar mapa de features
model = Model(inputs=model.inputs, outputs=outputs)

#model = ResNet50(weights='imagenet', pooling=max,include_top = False) 
model.summary()

# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

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
    #np.save(f+".npy",vgg16_feature)
    print("feature guardada "+f)

    square = 8
    for fmap in vgg16_feature:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        pyplot.show()