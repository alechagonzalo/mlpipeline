
# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances2
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import argparse
# define 81 classes that the coco model knowns about
# define 81 classes that the coco model knowns about
class_names = ['BG', 'box', 'bottle']

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 2

parser = argparse.ArgumentParser()
parser.add_argument('--image',
                metavar="path or URL to image",
                help='Image to apply the color splash effect on')

args = parser.parse_args()


assert args.image
"Se requiere imagen de entrada"

image = args.image

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights

rcnn.load_weights('trainingW/mask_rcnn_object_0050.h5', by_name=True)
#rcnn.load_weights('trainingW/mask_rcnn_coco.h5', by_name=True)
# load photograph
img = load_img(image)
img = img_to_array(img)
# make prediction
results = rcnn.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances2(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
