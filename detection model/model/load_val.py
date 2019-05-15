import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import numpy.random as npr
import scipy.sparse
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


class Validation():
    def __init__(self):
        self._data_path = '../data/VOCdevkit2007/VOC2007/'
        self._classes = ('__background__',  # always index 0
                         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'matlab_eval': False,
                       'use_diff': False,
                       'rpn_file': None}

    def _load_pascal_annotation(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1) * (y2 - y1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'image': os.path.join(self._data_path, 'JPEGImages', index + '.jpg'),
                'seg_areas': seg_areas}

    def load_val(self):
        print('Loading validation data...')

        with open(os.path.join(self._data_path, 'ImageSets/Main/test.txt'), 'r') as f:
            lines = f.readlines()
        indexes = [l.replace('\n','').replace('\r','') for l in lines]
        roidb = [self._load_pascal_annotation(index)
                    for index in indexes]

        num_images = len(indexes)
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
        # assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        #     'num_images ({}) must divide BATCH_SIZE ({})'. \
        #         format(num_images, cfg.TRAIN.BATCH_SIZE)

        im_blob, im_scales = self._get_image_blob(roidb, random_scale_inds)

        blobs = {'data': im_blob}

        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"

        if cfg.TRAIN.USE_ALL_GT:
            # Include all ground truth boxes
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        else:
            # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
            gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
            dtype=np.float32)
        return blobs

    def _get_image_blob(self, roidb, scale_inds):
        num_images = len(roidb)
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            im_scales.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, im_scales


if __name__ == '__main__':
    val = Validation()
    blobs = val.load_val()