import sys
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam

import numpy as np
import matplotlib
from numpy import count_nonzero
matplotlib.use('agg')
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

#load model
# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = '../data/checkpoints/ssd300_heavy_machine/ssd300_heavy_machine_noAugOneImage_epoch-03_loss-3.0880_val_loss-1.5296.h5' # ssd300_heavy_machine_noAugOneImage_epoch-01_loss-2.0257_val_loss-0.5414.h5'# bad results repeated detections in every image ssd300_heavy_machine_epoch-21_loss-5.2081_val_loss-5.1517.h5'

# model_path = '../data/checkpoints/ssd300_heavy_machine/ssd300_heavy_machine_epoch-02_loss-6.1302_val_loss-6.1017.h5'# bad results repeated detections in every image ssd300_heavy_machine_epoch-21_loss-5.2081_val_loss-5.1517.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

print('!!!!!!!!!!!!!!!!!!!!!! loading model')
try:
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                'L2Normalization': L2Normalization,
                                                'DecodeDetections': DecodeDetections,
                                                'compute_loss': ssd_loss.compute_loss})
except ValueError:
    print("invalid model file specified:",model_path)
    sys.exit(1)
except ImportError:
    print('import error! for ', model_path)
    sys.exit(1)

print('!!!!!!!!!!!!!!!!!!!!!!!! done loading model')
# sys.exit(1)

# img_height = 300 # Height of the input images
# img_width = 300 # Width of the input images
# img_channels = 3 # Number of color channels of the input images
# subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
# swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
# # TODO: Set the number of classes.
# n_classes = 8 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
# # scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
# two_boxes_for_ar1 = True
# steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
# clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
# variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
normalize_coords = True

# new_img_height = 1500
# new_img_width = 2048
# new_model = ssd_300(image_size=(new_img_height, new_img_width, img_channels),
#                 n_classes=n_classes,
#                 mode='inference',
#                 l2_regularization=0.0005,
#                 scales=scales,
#                 aspect_ratios_per_layer=aspect_ratios,
#                 two_boxes_for_ar1=two_boxes_for_ar1,
#                 steps=steps,
#                 offsets=offsets,
#                 clip_boxes=clip_boxes,
#                 variances=variances,
#                 normalize_coords=normalize_coords,
#                 subtract_mean=subtract_mean,
#                 divide_by_stddev=None,
#                 swap_channels=swap_channels,
#                 confidence_thresh=0.1,
#                 iou_threshold=0.45,
#                 top_k=200,
#                 nms_max_output_size=400,
#                 return_predictor_sizes=False)

# for new_layer, layer in zip(new_model.layers[1:], model.layers[1:]):
#     new_layer.set_weights(layer.get_weights())
# /home/linuxosprey/ow_keras_ssd/data/checkpoints/ssd300_heavy_machine/ssd300_heavy_machine_noAugOneImage_epoch-03_loss-3.0880_val_loss-1.5296.h5
#load  data set
dataset = DataGenerator(hdf5_dataset_path='dataset_heavy_machine_train_min.h5')

img_width = 2048
img_height = 1500
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

generator = dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)
#predict
for n in range(0,100):
    print('n=',n)
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator)

    print('batch_images shape = ', batch_images.shape)

    print('predict call')
    y_pred = model.predict(batch_images)
    print('decode_detections call')
    print('y_pred_shape = ' , y_pred.shape)
    y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.2,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
    print('y_pred_decoded, len, shape = ' , len(y_pred_decoded), y_pred_decoded[0].shape)
    print('apply inverse transforms call')

    y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
    i = 0
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded_inv[i])

    


    # Visualize the predictions.


  
    fig = plt.figure(figsize=(20,12))
    plt.imshow(batch_original_images[i])

    current_axis = plt.gca()

    classes = ['background', 'thing', 'truck', 'pedestrian', 'bicyclist',
            'traffic_light', 'motorcycle', 'bus', 'stop_sign'] # Just so we can print class names onto the image instead of IDs

    # Draw the predicted boxes in blue
    for box in batch_original_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in y_pred_decoded_inv[i]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]

        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})


    img_filename = 'ssd3_after_scaled_scales_epochs_'+ str(n) +'.png'
    print('saving image ', img_filename)
    fig.savefig(img_filename)