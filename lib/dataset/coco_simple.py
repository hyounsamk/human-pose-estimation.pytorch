# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import copy
import cv2
import time

from torch.utils.data import Dataset

import numpy as np
from pycocotools.coco import COCO

from utils.transforms import get_affine_transform


logger = logging.getLogger(__name__)

'''
"keypoints": {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
},
"skeleton": [
    [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
    [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
'''

class DeltaTime(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.t0 = 0
    def update(self):
        self.t0 = time.time()
    def check_(self):
        return time.time() - self.t0
    def lap_(self):
        t = self.t0
        self.t0 = time.time()
        return self.t0 - t
    def lap(self, prefix):
        t = self.lap_()
        #print(prefix, t)

class COCOSimpleDataset(Dataset):
    def __init__(self, cfg, root, kps_file, transform):
        #_deltatime = DeltaTime()
        #_deltatime.update()

        self.root = root
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.transform = transform

        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.coco = COCO(kps_file)
        #_deltatime.lap("DELTA: COCOSimple(): COCO")

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))
        #_deltatime.lap("DELTA: COCOSimple(): INIT DONE")

    def _load_image_set_index(self):
        """ image id: int """
        _deltatime = DeltaTime()
        _deltatime.update()
        image_ids = self.coco.getImgIds()
        _deltatime.lap("DELTA: COCOSimple(): coco.getImgIds")
        return image_ids

    def _get_db(self):
        # use ground truth bbox
        """ ground truth bbox and keypoints """
        _deltatime = DeltaTime()
        _deltatime.update()
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
            _deltatime.lap("DELTA: COCOSimple(): _load_coco_keypoint_annotation_kernal")
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        rec = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                center, scale = self._box2cs(obj['clean_bbox'][:4])
                rec.append({
                    'image': self.image_path_from_index(index),
                    'center': center,
                    'scale': scale,
                    'filename': '',
                    'imgnum': 0,
                })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def __len__(self,):
        logger.info('=> len() ')
        return len(self.db)

    def __getitem__(self, idx):
        _deltatime = DeltaTime()
        _deltatime.update()

        db_rec = copy.deepcopy(self.db[idx])
        #_deltatime.lap("DELTA: COCOSimple(): deepcopy()")

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else 0

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #_deltatime.lap("DELTA: COCOSimple(): cv2.imread()")

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        #_deltatime.lap("DELTA: COCOSimple(): get_affine_transform()")
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        #_deltatime.lap("DELTA: COCOSimple(): cv2.warpAffine()")

        if self.transform:
            input = self.transform(input)
        #_deltatime.lap("DELTA: COCOSimple(): transform()")

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, meta


    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        image_path = os.path.join(
            self.root, file_name)
        return image_path

    def image_path_from_file_name(self, file_name):
        """ example: images / train2017 / 000000119993.jpg """
        return os.path.join(
            self.root, file_name)


