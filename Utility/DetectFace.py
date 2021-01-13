import os
import time
import tensorflow as tf
import tarfile
import shutil

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder


#######################################################
# EXTRACT THE MODEL TAR FILE
#######################################################
# Mode up a dir
os.chdir('..')

# print(os.getcwd())
print(os.path.join(os.getcwd(),'exported-models'))
if os.path.exists(os.path.join(os.getcwd(),'exported-models')) == False:
    fname = os.path.join(os.getcwd(),'trained_model.tar.gz')
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()
    # MOVEL EXPORTED-MODEL DIR OUTSIDE
    shutil.move(os.path.join(os.getcwd(),'content','exported-models'),os.getcwd())
    shutil.rmtree(os.path.join(os.getcwd(),'content'))



#######################################################
# CONFIGURE & LOAD MODEL
#######################################################
start_time = time.time()
MODEL_DIR = os.path.join(os.getcwd(),'exported-models','checkpoint')
PATH_TO_LABELS = os.path.join(os.getcwd(),'face_label.pbtxt')
PATH_TO_CONFIG = os.path.join(os.getcwd(),'exported-models','pipeline.config')

configs = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG)
model_configs = configs['model']

detection_model = model_builder.build(
    model_config=model_configs,
    is_training=False
)

# Restore the checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MODEL_DIR,'ckpt-0')).expect_partial()

def get_model_detection_function(model):
    """
    return a tf.function for detection
    """

    @tf.function
    def detect_fn(image):
        """ Detect objects im image"""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])
    
    return detect_fn

detect_fn = get_model_detection_function(detection_model)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Model Compiled!!! @ {detect_fn} in Time : {elapsed_time}')



# #######################################################
# Load label map for ploting
# #######################################################
# label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
catergories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True
)
category_index = label_map_util.create_category_index(catergories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)



# A function to convert img into an input tensor
def load_image_into_numpy_array(path):
  
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



# #######################################################
# PLOTTING
# #######################################################

from PIL import Image
from six import BytesIO
import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


## Change the path to your testing image below!!!
image_path = 'F:\DetectFace\pic.jpg'
image_np = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np,0), dtype=tf.float32
)
detections, predictions_dict, shape = detect_fn(input_tensor)
label_id_offset = 1


image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=100,
    min_score_thresh=0.5,
    agnostic_mode=False
)

plt.figure(figsize=(12,8))
plt.imshow(image_np_with_detections)
plt.show()