# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
from tqdm import tqdm

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois, roi_logits = net.test_image(sess, blobs['data'], blobs['im_info'])
  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes, roi_logits

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes


def function(pth):
  if not os.path.exists(pth):
    dic = {}
    dic['now'] = 0
    with open('../data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt') as f:
      lines = f.readlines()
    img_list = ['{}.jpg'.format(line.replace('\n', '')) for line in lines]
    dic['img_list'] = img_list
    dic['num_images'] = len(img_list)
    dic['all_boxes'] = [[[] for _ in range(dic['num_images'])] for _ in range(12)]
  else:
    print('Load imfo...')
    dic = pickle.load(open(pth, 'r'))
    dic['now'] += 1
  return dic


def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):

  np.random.seed(cfg.RNG_SEED)

  pre_pth = '../output/results/'
  img_pre_pth = '../data/VOCdevkit2007/VOC2007/JPEGImages/'
  dic = function(os.path.join(pre_pth, 'imfo.pkl'))

  all_boxes = dic['all_boxes']

  results = {'results': {}, 'features': {}, 'rois': {}}
  start = dic['now']

  for i in range(dic['now'], dic['num_images']):
    img_pth = os.path.join(img_pre_pth, dic['img_list'][i])
    im = cv2.imread(img_pth)
    img_name = img_pth.split('/')[-1]

    scores, boxes, roi_logits = im_detect(sess, net, im)

    feature_indexes = []

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      feature_indexes.append(keep)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                      for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
          if len(keep) == 1:
            prediction = all_boxes[j][i][keep, :][0]
            pred_str = '{}\t{}\t{:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, j-1, prediction[0], prediction[1], prediction[2], prediction[3])

            feature_index = feature_indexes[j-1][keep[0]]
            results['results'][img_name] = pred_str
            results['features'][img_name] = roi_logits[feature_index, :]
            # results['rois'][img_name] = roi_poolings[feature_index, :, :, :]

    dic['now'] = i
    print('{}/{}'.format(i, dic['num_images']))

    if (i+1) % 25000 == 0:
      print('saving results...')
      with open(os.path.join(pre_pth, 'imfo.pkl'), 'w') as f:
        pickle.dump(dic, f)
      with open(os.path.join(pre_pth, '{}_{}.pkl'.format(start, i)), 'w') as f:
        pickle.dump(results, f)
      exit(0)

  print('Done!')
  # with open(os.path.join(pre_pth, 'imfo.pkl'), 'w') as f:
  #   pickle.dump(dic, f)
  with open(os.path.join(pre_pth, '{}_{}.pkl'.format(start, dic['now'])), 'w') as f:
    pickle.dump(results, f)

  # with open(weights_filename+'prediction_trainval.csv', 'w') as f:
  #   f.writelines(results)
  #
  # with open(weights_filename + 'features_vectors_trainval.pkl', 'w') as f:
  #   pickle.dump(features, f)
  #
  # with open(weights_filename + 'features_roi_trainval.pkl', 'w') as f:
  #   pickle.dump(roi, f)
