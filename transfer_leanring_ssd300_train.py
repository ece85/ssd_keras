import sys
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger

from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model

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

import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session


K.clear_session() # Clear previous models from memory.


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_fraction =0.4
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    print('!!!!!!!!!!!!!!! setting session!')
   
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
#     except ImportError:
#         print('error setting gpu memory limit!')


weights_destination_path = '../data/VGG_coco_SSD_300x300_iter_400000_subsampled_8_classes2.h5'

img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 3 # Number of color channels of the input images
subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
# TODO: Set the number of classes.
n_classes = 8 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
# scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
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

##JF 01 24
new_img_height = 1500
new_img_width = 2048
new_model = ssd_300(image_size=(new_img_height, new_img_width, img_channels),
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
##END JF
print("Model built.")

# new_model.summary()
# sys.exit(1)
# 2: Load the sub-sampled weights into the model.

# Load the weights that we've just created via sub-sampling.
weights_path = weights_destination_path

model.load_weights(weights_path, by_name=True)

##JF 01 24
# Replace the weights in the inference model with imagenet weights
for new_layer, layer in zip(new_model.layers[1:], model.layers[1:]):
    new_layer.set_weights(layer.get_weights())
## END JF

print("Weights file loaded:", weights_path)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

new_model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


load_images_in_mem_flag = False
train_dataset = DataGenerator(load_images_into_memory=load_images_in_mem_flag, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=load_images_in_mem_flag, hdf5_dataset_path=None)

#Images
images_dir = '../data_hm/'

# Ground truth
train_labels_filename = '../data_hm/heavy_machine_labels_train.csv'
val_labels_filename   = '../data_hm/heavy_machine_labels_val.csv'

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.
if load_images_in_mem_flag == False:
    train_dataset.create_hdf5_dataset(file_path='dataset_heavy_machine_train.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

    val_dataset.create_hdf5_dataset(file_path='dataset_heavy_machine_val.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)


# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))







# 3: Set the batch size.

batch_size = 1 # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=new_img_height,
                                            img_width=new_img_width,
                                            background=subtract_mean)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=new_img_height, width=new_img_width)

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
                                    scales=scales,
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

sys.exit(1)
# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

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
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))



# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='../data/checkpoints/ssd300_heavy_machine/ssd300_heavy_machine_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best = 

csv_logger = CSVLogger(filename='ssd300_heavy_machine_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]







# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 100
steps_per_epoch = 1000

history = new_model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)



# 1: Set the generator for the predictions.

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)


# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))






# 3: Make predictions.

y_pred = new_model.predict(batch_images)



# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=new_img_height,
                                   img_width=new_img_width)

# 5: Convert the predictions for the original image.

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])


# 5: Draw the predicted boxes onto the image

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['heavy_machine']

fig = plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})





    fig.savefig('ssd3_transfer_learning5.png')






# dataset = DataGenerator()

# # TODO: Set the paths to your dataset here.
# images_path = '../data_hm/'
# labels_path = '../data_hm/heavy_machine_labels.csv'
# # images_path = '../data/udacity_driving_datasets/'
# # labels_path = '../data/udacity_driving_datasets/labels_val.csv'

# dataset.parse_csv(images_dir=images_path,
#                   labels_filename=labels_path,
#                   input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
#                   include_classes='all',
#                   random_sample=False)

# print("Number of images in the dataset:", dataset.get_dataset_size())

# convert_to_3_channels = ConvertTo3Channels()
# random_max_crop = RandomMaxCropFixedAR(patch_aspect_ratio=img_width/img_height)
# resize = Resize(height=new_img_height, width=new_img_width)

# # generator = dataset.generate(batch_size=1,
# #                              shuffle=False,
# #                              transformations=[convert_to_3_channels,
# #                                               random_max_crop,
# #                                               resize],
# #                              returns={'processed_images',
# #                                       'processed_labels',
# #                                       'filenames'},
# #                              keep_images_without_gt=False)
# generator = dataset.generate(batch_size=1,
#                              shuffle=False,
#                              transformations=[resize],
#                              returns={'processed_images',
#                                       'processed_labels',
#                                       'filenames'},
#                              keep_images_without_gt=False)


# # Generate samples

# batch_images, batch_labels, batch_filenames = next(generator)

# i = 0 # Which batch item to look at

# print("Image:", batch_filenames[i])
# print()
# print("Ground truth boxes:\n")
# print(batch_labels[i])


# # Make a prediction

# y_pred = new_model.predict(batch_images)


# # Decode the raw prediction.

# i = 0

# confidence_threshold = 0.5

# y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('    class    conf  xmin    ymin    xmax    ymax')
# print(y_pred_thresh[0])


# # Visualize the predictions.
# import matplotlib
# matplotlib.use('agg')
# from matplotlib import pyplot as plt


# fig = plt.figure(figsize=(20,12))
# plt.imshow(batch_images[i])

# current_axis = plt.gca()

# classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist',
#            'traffic_light', 'motorcycle', 'bus', 'stop_sign'] # Just so we can print class names onto the image instead of IDs

# # Draw the predicted boxes in blue
# for box in y_pred_thresh[i]:
#     class_id = box[0]
#     confidence = box[1]
#     xmin = box[2]
#     ymin = box[3]
#     xmax = box[4]
#     ymax = box[5]
#     label = '{}: {:.2f}'.format(classes[int(class_id)], confidence)
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

# # Draw the ground truth boxes in green (omit the label for more clarity)
# for box in batch_labels[i]:
#     class_id = box[0]
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(class_id)])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
#     #current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

#     fig.savefig('ssd3_transfer_learning5.png')