import json
import os
import sys
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger

from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from keras.models import save_model

from math import ceil

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_patch_sampling_ops import RandomMaxCropFixedAR
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


train_session = {}


model_name = 'ssd_300_heavy_machinery'
train_session['model_name'] = model_name
session_suffix = 'single_image_repro_270_208_no_frozen_layers_100stepsPerEpoch'
train_session['session_suffix'] = session_suffix

output_dir = os.path.join('..','output',model_name,session_suffix)
os.makedirs(output_dir,exist_ok=True)
train_session['train_output_dir'] = output_dir
K.clear_session()  # Clear previous models from memory.

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_fraction =0.4
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
# subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset#for original data set (COCO)

# The per-channel mean of the images in the dataset
subtract_mean = [123,117,104]
train_session['bgr_mean'] = subtract_mean
# The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
swap_channels = [2, 1, 0]
# TODO: Set the number of classes.
n_classes = 8  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
# scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
# The space between two adjacent anchor box center points for each predictor layer.
steps = [8, 16, 32, 64, 100, 300]
# The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
# The variances by which the encoded target coordinates are scaled as in the original implementation
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True


# 1: Build the Keras model
model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=subtract_mean,
                swap_channels=swap_channels)

new_img_height = 1500
new_img_width = 2048

scales_scale = float(img_height)/float(new_img_height)

scaled_scales = [elem*scales_scale for elem in scales]

print('scales = ', scales)
print('scaled scales = ', scaled_scales)
train_session['scales'] = scales

new_model = ssd_300(image_size=(new_img_height, new_img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scaled_scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=subtract_mean,
                    swap_channels=swap_channels)

print("Model built.")

# new_model.summary()

# 2: Load the sub-sampled weights into the model.
weights_destination_path = '/home/linuxosprey/ow_ssd_keras/data/VGG_coco_SSD_300x300_iter_400000_subsampled_8_classes2.h5'

train_session["pre_trained_weights_path"] = weights_destination_path
# Load the weights that we've just created via sub-sampling.
weights_path = weights_destination_path

model.load_weights(weights_path, by_name=True)


for new_layer, layer in zip(new_model.layers[1:], model.layers[1:]):
    new_layer.set_weights(layer.get_weights())


print("Weights file loaded:", weights_path)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model.



# freeze certain layers.
new_model.summary()
classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']

freeze_layers = False
train_session['freeze_layers'] = freeze_layers
if freeze_layers:
    layers_to_be_frozen = classifier_names
    for layer in new_model.layers:
        if layer.name in layers_to_be_frozen:
            layer.trainable = True
            print('training layer ', layer.trainable, ' , ', layer.name)
            # print('layer input = ', layer.input)
        else:
            layer_freezed = False
            layer_input = str(layer.input)
            # print('layer input = ' , layer_input)
            for l in layers_to_be_frozen:

                if layer_input.find(l) >= 0:
                    layer.trainable = True
                    layer_freezed = True
                    layers_to_be_frozen.append(layer.name)
                    break
            if not layer_freezed:
                layer.trainable = False

    for layer in new_model.layers:
        if layer.name in layers_to_be_frozen and layer.trainable == False:
            print(' layer not frozen and should be: ', layer.name)
            sys.exit(-1)

    for layer in new_model.layers:
        print('POST layer ', layer.trainable, ' , ', layer.name)

    train_session['frozen_layers'] = layers_to_be_frozen


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
# compile the model
new_model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


train_dataset = DataGenerator(
    load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(
    load_images_into_memory=False, hdf5_dataset_path=None)

# Images

images_dir = '/home/linuxosprey/ow_ssd_keras/data/datasets/training_data_no270_no276_oxnard_suburbanAU_hr_combed_again/'

train_session['input_data_dir'] = images_dir
train_labels_name = 'hm_270_single_repro_old'
val_labels_name = 'hm_270_single_repro_old'
# Ground truth
# all cranes heavy machines from a incomplete version of all tagged data (bad data set for training..i think)  ../data_hm/heavy_machine_labels_train.csv'
train_labels_filename = images_dir + train_labels_name +'.csv'
val_labels_filename = images_dir + train_labels_name + '.csv'

train_session['train_labels_filename'] = train_labels_filename
train_session['val_labels_filename'] = val_labels_filename

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        input_format=['image_name', 'xmin',
                                      'xmax', 'ymin', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin',
                                    'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

train_data_path = os.path.join(images_dir,train_labels_name+'.h5')
val_data_path = os.path.join(images_dir,val_labels_name+'.h5')
train_session['train_h5_data_path'] = train_data_path
train_session['val_h5_data_path'] = val_data_path

if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    train_dataset.create_hdf5_dataset(file_path=train_data_path,
                                      resize=False,
                                      variable_image_size=True,
                                      verbose=True)

    val_dataset.create_hdf5_dataset(file_path=val_data_path,
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)


# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(
    train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(
    val_dataset_size))


# 3: Set the batch size.

# Change the batch size if you like, or if you run into GPU memory issues.
batch_size = 1

# 4: Set the image transformations for pre-processing and data augmentation options.
# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=new_img_height, width=new_img_width)

# For the training generator:
augment_type = 'vanilla'
# if augment_type == 'ssd':

#     data_augmentation = SSDDataAugmentation(img_height=new_img_height,
#                                             img_width=new_img_width,
#                                             background=subtract_mean)
# elif augment_type == 'photo_flip_resize':
#     data_augmentation = PhotoFlipResizeDataAugmentation(img_height=new_img_height,
#                                                         img_width=new_img_width,
#                                                         background=subtract_mean)

train_session['data_aug_type'] = augment_type


# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [new_model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   new_model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   new_model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   new_model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   new_model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   new_model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=new_img_height,
                                    img_width=new_img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scaled_scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
# transformations=[ssd_data_augmentation],

if augment_type =='vanilla':
  train_generator = train_dataset.generate(batch_size=batch_size,
                                          shuffle=True,
                                          transformations=[convert_to_3_channels,
                                                      resize],
                                          label_encoder=ssd_input_encoder,
                                          returns={'processed_images',
                                                    'encoded_labels'},
                                          keep_images_without_gt=False)

# else:
#   train_generator = train_dataset.generate(batch_size=batch_size,
#                                           shuffle=True,
#                                           transformations=[data_augmentation],
#                                           label_encoder=ssd_input_encoder,
#                                           returns={'processed_images',
#                                                     'encoded_labels'},
#                                           keep_images_without_gt=False)


val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(
    train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(
    val_dataset_size))

train_session['train_dataset_size'] = train_dataset_size
train_session['val_dataset_size'] = val_dataset_size


# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 25:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.

os.makedirs(os.path.join(output_dir,'checkpoints'),exist_ok=True)

model_checkpoint_path = os.path.join(
    output_dir, 'checkpoints', model_name +'_'+ session_suffix+'_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5')
train_session['checkpoint_path'] = model_checkpoint_path
model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_path,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
# model_checkpoint.best =

log_filename = os.path.join(output_dir,model_name + '_' + session_suffix + '_training_log.csv')
csv_logger = CSVLogger(filename=log_filename,
                       separator=',',
                       append=True)

train_session['log_filename'] = log_filename

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 200
steps_per_epoch = train_dataset.dataset_size*100
print('num steps for epoch = ', steps_per_epoch)

train_session['initial_epoch'] = initial_epoch
train_session['final_epoch'] = final_epoch
train_session['steps_per_epoch'] = steps_per_epoch

cnt=0
while os.path.exists(os.path.join(output_dir, 'train_session' + str(cnt)+'.json')):
  cnt = cnt+1

with open(os.path.join(output_dir, 'train_session' + str(cnt)+'.json'), 'w') as output_file:
    output_file.write(json.dumps(train_session,indent=4,sort_keys=True))

history = new_model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=final_epoch,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(
                                      val_dataset_size/batch_size),
                                  initial_epoch=initial_epoch)