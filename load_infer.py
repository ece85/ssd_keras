import sys
sys.path.append('../')
import buffered_precision_recall as bpr
import json
import os
from eval_utils.average_precision_evaluator import Evaluator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd300 import ssd_300
from distutils import extension
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam

import numpy as np
from numpy import count_nonzero

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
K.clear_session()  # Clear previous models from memory.

# load model
session_record_path = '/home/linuxosprey/ow_ssd_keras/output/ssd_300_heavy_machinery/no270_no276_w359_90_496_combed_HeavyMachine_Tractor/train_session0.json'

eval_session_suffix = 'enfored_split_per_image_train_val_split_VAL'#used to prefix some output product names and folder
if not os.path.exists(session_record_path):
    print('input session record: ', session_record_path, ' does not exist')
    sys.exit(-1)

file = open(session_record_path)
contents = file.read()
session_record = json.loads(contents)
session_record['eval_session_suffix'] = eval_session_suffix

checkpoint_filename_lowest_loss = '../data/checkpoints/ssd300_heavy_machine/ssd300_hm_tractor_perImageEnforced_epoch-02_loss-8.8385_val_loss-9.7547.h5'
#trained on old per iamge split data: '../data/checkpoints/ssd300_heavy_machine/ssd300_heavy_machine_tractor_enhanced_combed_again_hm_epoch-01_loss-9.4464_val_loss-7.6128.h5'
#heavy machine only '../data/checkpoints/ssd300_heavy_machine/ssd300_heavy_machine_enhanced_combed_again_hm_epoch-04_loss-8.0991_val_loss-7.3224.h5'
# min_val_loss = 1e9
# checkpoints_dir = os.path.dirname(session_record['checkpoint_path'])
# for f in os.listdir(checkpoints_dir):
#     if f.find('.h5') >= 0:
#         # split filename into two (using val-loss- delimeter, grab the second one, trim last three chars)
#         val_loss = float(f.split('val_loss-')[1][:-3])
#         if val_loss < min_val_loss:
#             min_val_loss = val_loss
#             checkpoint_filename_lowest_loss = os.path.join(checkpoints_dir, f)


model_path = checkpoint_filename_lowest_loss

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
try:
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
except ValueError:
    print("invalid model file specified:", model_path)
    sys.exit(1)
except ImportError:
    print('import error! for ', model_path)
    sys.exit(1)


img_width = 300
img_height = 300
img_channels = 3  # Number of color channels of the input images
# The per-channel mean of the images in the dataset
subtract_mean = [93, 100, 103]#this should be the mean bgr of the train data this is calculated when splitting all_labels.csv files to their train and val derivative products
session_record['bgr_mean_eval'] = subtract_mean
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

session_record['matching_iou_threshold'] = 0.2

model_mode = 'inference'
new_img_height = 1500
new_img_width = 2048

scales_scale = float(img_height)/float(new_img_height)

scaled_scales = [elem*scales_scale for elem in scales]


new_model = ssd_300(image_size=(new_img_height, new_img_width, img_channels),
                    n_classes=n_classes,
                    mode=model_mode,
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
                    divide_by_stddev=None,
                    swap_channels=swap_channels,
                    confidence_thresh=0.1,
                    iou_threshold=session_record['matching_iou_threshold'],
                    top_k=200,
                    nms_max_output_size=400,
                    return_predictor_sizes=False)

for new_layer, layer in zip(new_model.layers[1:], model.layers[1:]):
    new_layer.set_weights(layer.get_weights())


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss2 = SSDLoss(neg_pos_ratio=3, alpha=1.0)


new_model.compile(optimizer=adam, loss=ssd_loss2.compute_loss)

dataset_path = 'HeavyMachine_tractor_val_perImageEnforced.h5'
#session_record['val_h5_data_path']# can specify path to a different dataset here. use convert_create_spoof_training_data.py to 
# '/home/linuxosprey/ow_ssd_keras/ssd_keras/labels_HeavyMachine_val_oldScript.h5'#
print('dataset path = ' , dataset_path)
dataset = DataGenerator(hdf5_dataset_path=dataset_path)
dataset_plot = DataGenerator(hdf5_dataset_path=dataset_path)
session_record['eval_dataset_path'] = dataset_path

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
                                      'inverse_transform',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)

generator_plot = dataset_plot.generate(batch_size=1,
                                       shuffle=False,
                                       transformations=[convert_to_3_channels,
                                                        resize],
                                       label_encoder=None,
                                       returns={'processed_images',
                                                'inverse_transform',
                                                'original_images',
                                                'original_labels',
                                                'image_ids'},
                                       keep_images_without_gt=False)

# batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels, batch_image_ids = next(


n_classes = 8
evaluator = Evaluator(model=new_model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=1,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=session_record['matching_iou_threshold'],
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

print('mean_avg prec = ', mean_average_precision)
print('average_precisions prec = ', average_precisions)
# print('precisions prec = ', np.array(precisions[1]).shape)
# print('recalls prec = ', np.array(recalls[1]).shape)

# print('mean_avg prec 1= ', mean_average_precision)
# print('average_precisions prec1 = ', average_precisions)
# print('precisions prec1 = ', precisions)
# print('recalls prec1 = ', recalls)

session_record['mean_average_precision'] = mean_average_precision
session_record['average_precisions'] = list(average_precisions)
session_record['precisions'] = list(precisions[1])
session_record['recalls'] = list(recalls[1])

classes = ['hm']
print('average precions len = ', len(average_precisions))
print(' precisions =', np.array(precisions).shape)

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format('class:', str(i),
          ' AP', round(average_precisions[i], 3)))
    # print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))

print()
print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))


m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8, m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes:
            break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1],
                         color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0, 1, 11))
        cells[i, j].set_yticks(np.linspace(0, 1, 11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(
            n*i+j+1, average_precisions[n*i+j+1]), fontsize=16)


cnt = 0
while os.path.exists(os.path.join(session_record['train_output_dir'], 'evaluation_' + eval_session_suffix + '_' + str(cnt))):
    cnt = cnt+1

output_dir = os.path.join(
    session_record['train_output_dir'], 'evaluation_' + eval_session_suffix + '_' + str(cnt))
os.makedirs(output_dir)
session_record['eval_output_dir'] = output_dir
session_record['precision_recall_figure_path'] = os.path.join(
    output_dir, session_record['model_name'] + '_' + session_record['session_suffix'] + 'precision_recall_curve.png')

fig.savefig(session_record['precision_recall_figure_path'])


with open(os.path.join(output_dir, 'evaluation_record.json'), 'w') as output_file:
    output_file.write(json.dumps(session_record, indent=4, sort_keys=True))

session_record['plot_confidence_thresh'] = 0.1
# predict

detections_filename = os.path.join(output_dir, 'detections.csv')
detections_file = open(detections_filename, 'w')
detections_file.write('frame,xmin,xmax,ymin,ymax,class_id,score\n')

truth_labels_filename = os.path.join(output_dir, 'predict_truth.csv')
truth_labels = open(truth_labels_filename, 'w')
truth_labels.write('frame,xmin,xmax,ymin,ymax,class_id\n')

for n in range(dataset.dataset_size):

    print('n=', n)
    batch_images, batch_image_ids,  batch_inverse_transforms, batch_original_images, batch_original_labels, = next(
        generator_plot)
    # print('batch_images shape = ', batch_images.shape)
    # print('batch_original_images shape = ', (batch_original_images))
    # print('batch_inverse_transforms shape = ', (batch_inverse_transforms))
    # print('batch_original_labels shape = ', (batch_original_labels))
    # print('batch_image_ids = ' , batch_image_ids)
    # print('batch_images = ' , type(batch_images))

    print('predict call')
    y_pred = new_model.predict(batch_images)
    print('decode_detections call')
    print('y_pred_shape = ', y_pred.shape)
    plot_images = True
    if model_mode == 'training':
        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=session_record['plot_confidence_thresh'],
                                           iou_threshold=session_record['matching_iou_threshold'],
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)

        y_pred_decoded_inv = apply_inverse_transforms(
            y_pred_decoded, batch_inverse_transforms)
        i = 0
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_decoded_inv[i])

        # Visualize the predictions.

        fig = plt.figure(figsize=(20, 12))
        plt.imshow(batch_original_images[i])

        current_axis = plt.gca()

        classes = ['background', 'hm', 'crane', 'unknown', 'tractor',
                   'traffic_light', 'motorcycle', 'bus', 'stop_sign']  # Just so we can print class names onto the image instead of IDs

        # Draw the predicted boxes in green
        for box in batch_original_labels[i]:
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            label = '{}'.format(classes[int(box[0])])
            current_axis.add_patch(plt.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=1))
            current_axis.text(xmax, ymax, label, size='x-small',
                              color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

        # Draw the ground truth boxes in blue (omit the label for more clarity)
        for box in y_pred_decoded_inv[i]:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]

            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=1))
            current_axis.text(xmax+50, ymax+50, label, size='x-small',
                              color='white', bbox={'facecolor': 'blue', 'alpha': 1.0})

    elif model_mode == 'inference':
        confidence_threshold = 0.2
        session_record['pr_confidence_thresh'] = confidence_threshold
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                         for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

        # Display the image and draw the predicted boxes onto it.

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        classes = ['background',
                   'hm', 'crane', 'unknown', 'tractor',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

        if plot_images:
            fig = plt.figure(figsize=(20, 12))
            plt.imshow(batch_original_images[0])

            current_axis = plt.gca()

        # Draw the ground truth boxes in green
        for box in batch_original_labels[0]:
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            class_id = int(box[0])
            label = '{}'.format(classes[class_id])
            if plot_images:
              current_axis.add_patch(plt.Rectangle(
                  (xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=1))
              # current_axis.text(xmax+50, ymax+50, label, size='x-small',
              #                   color='white', bbox={'facecolor': 'green', 'alpha': 1.0})
            truth_labels.write(batch_image_ids[0] + '.jpg' + ',' + str(xmin) + ',' + str(
                xmax) + ',' + str(ymin) + ',' + str(ymax) + ',' + str(class_id) + '\n')

        # draw predicted boxes
        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            class_id = int(box[0])
            score = float(box[1])

            xmin = box[2] * batch_original_images[0].shape[1] / img_width
            ymin = box[3] * batch_original_images[0].shape[0] / img_height
            xmax = box[4] * batch_original_images[0].shape[1] / img_width
            ymax = box[5] * batch_original_images[0].shape[0] / img_height
            detections_file.write(batch_image_ids[0] + '.jpg' + ',' + str(xmin) + ',' + str(
                xmax) + ',' + str(ymin) + ',' + str(ymax) + ',' + str(class_id) + ',' + str(score) + '\n')

            if plot_images:    
              color = colors[int(box[0])]
              label = '{}: {:.2f}'.format(classes[class_id], box[1])
              current_axis.add_patch(plt.Rectangle(
                  (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=1))
              # current_axis.text(xmax+50, ymax+50, label, size='x-small',
              #                   color='white', bbox={'facecolor': color, 'alpha': 1.0})

            

    if plot_images:
        img_filename = os.path.join(
            output_dir, session_record['model_name'] + '_' + session_record['session_suffix'] + '_predictions_' +
            str(batch_image_ids[0]) + '.png')

        print('saving image ', img_filename)
        fig.savefig(img_filename)

detections_file.close()

# detections_filename = '/home/linuxosprey/ow_ssd_keras/output/ssd_300_heavy_machinery/no270_no276_w359_90_496_combed_Heavy_machine_only_moreTrain_lessVal/evaluation_validation_set_correct_h5_file_0/detections.csv'
labels_filename = session_record['val_labels_filename']
session_record['detections_filename'] = detections_filename
test_detections = bpr.parse_detections(detections_filename)
truth_labels = bpr.parse_labels(labels_filename)

for class_id, det_list in test_detections.items():
    print('found ', len(det_list), ' in class id', class_id)

for class_id, label_list in truth_labels.items():
    print('found ', len(label_list), ' in class id', class_id)


buffer = 64
score_board = bpr.get_image_scoreboards(test_detections, truth_labels, buffer)

bpr.plot_scoreboards(output_dir,score_board)