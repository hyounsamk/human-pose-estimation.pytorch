# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import cv2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from utils.utils import create_logger

import dataset
import models

def default_args(args):
    if not args:
        args = {}
    args["frequent"] = 50
    args["workers"] = 0
    args["cfg"] = "resnet152/384x288_d256x3_adam_lr1e-3.yaml"
    args["model"] = "pose_resnet_152_384x288.pth.tar"
    
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    args.cfg = os.path.join('experiments/coco_simple', args.cfg)
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--model',
                        help='model state file',
                        type=str)
    parser.add_argument('--dataset-dir',
                        help='dataset directory',
                        type=str)
    parser.add_argument('--coco-kps-file',
                        help='coco-like keypoints file',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        default=0,
                        type=int)

    parser.add_argument('--DEBUG',
                        help='enable/disable debug',
                        default='FALSE',
                        type=str)
    parser.add_argument('--DEBUG_SAVE_BATCH_IMAGES_GT',
                        help='SAVE_BATCH_IMAGES_GT',
                        default='FALSE',
                        type=str)
    parser.add_argument('--DEBUG_SAVE_BATCH_IMAGES_PRED',
                        help='SAVE_BATCH_IMAGES_PRED',
                        default='FALSE',
                        type=str)
    parser.add_argument('--DEBUG_SAVE_HEATMAPS_GT',
                        help='SAVE_HEATMAPS_GT',
                        default='FALSE',
                        type=str)
    parser.add_argument('--DEBUG_SAVE_HEATMAPS_PRED',
                        help='SAVE_HEATMAPS_PRED',
                        default='FALSE',
                        type=str)
    """
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    """

    args = parser.parse_args()

    return args

def arg2bool(val):
    return True if val == 'TRUE' or val == 'true' or val == 'True' else False

def reset_config(config, args):
    if args.frequent:
        config.PRINT_FREQ = args.frequent
    if args.gpus:
        config.GPUS = args.gpus
    if args.model:
        config.TEST.MODEL_FILE = os.path.join('models/pytorch/pose_coco', args.model)
    if args.coco_kps_file:
        config.TEST.COCO_KPS_FILE = args.coco_kps_file
    if args.dataset_dir:
        config.DATASET.ROOT = args.dataset_dir
    if args.workers:
        config.WORKERS = args.workers
    # debug options
    if args.DEBUG:
        config.DEBUG.DEBUG = arg2bool(args.DEBUG)
    if args.DEBUG_SAVE_BATCH_IMAGES_GT:
        config.DEBUG.SAVE_BATCH_IMAGES_GT = arg2bool(args.DEBUG_SAVE_BATCH_IMAGES_GT)
    if args.DEBUG_SAVE_BATCH_IMAGES_PRED:
        config.DEBUG.SAVE_BATCH_IMAGES_PRED = arg2bool(args.DEBUG_SAVE_BATCH_IMAGES_PRED)
    if args.DEBUG_SAVE_HEATMAPS_GT:
        config.DEBUG.SAVE_HEATMAPS_GT = arg2bool(args.DEBUG_SAVE_HEATMAPS_GT)
    if args.DEBUG_SAVE_HEATMAPS_PRED:
        config.DEBUG.SAVE_HEATMAPS_PRED = arg2bool(args.DEBUG_SAVE_HEATMAPS_PRED)
    """
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file
    """

import time
import numpy as np

from core.inference import get_max_preds
from core.inference import get_final_preds
from utils.vis import save_debug_images

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def load_model(args):
    reset_config(config, args)

    assert config.TEST.MODEL_FILE != None, \
        'args.model_file should be specified!!!'

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    return config, model, gpus, logger, final_output_dir


def evaluate(config, data_loader, data_set, model, output_dir, logger):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(data_set)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    #all_boxes = np.zeros((num_samples, 6))
    #meta = {}
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(data_loader):
            # compute output
            output = model(input)

            num_images = input.size(0)
            if config.DEBUG.DEBUG:
                dbg_pred, dbg_maxvals = get_max_preds(output.cpu().numpy())
                # pred는 각 포인트들의 좌표값(x, y) array이다. array size는 dataset마다 차이가 있다.
                # maxvals는 각 포인트들의 confidence(0과 1사이)값 array이다. pred의 같은 갯수이다.

                dbg_maxvals = np.round_(dbg_maxvals) # 0.5 이상은 1로 반올림. 
                meta['joints_vis'] = dbg_maxvals # visualization 시에는 값이 1인 포인트만 표시한다.

            # 각 포인트들의 최종 좌표값들을 가져온다.
            # 최종 좌표값은 ...
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            #preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
            preds, maxvals = get_final_preds(config, output.cpu().numpy(), c, s)

            # 반환할 all_preds list에 저장한다.
            # 각 point 별 좌표와 confidence value [x, y, confidence]
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            idx += num_images

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})' \
                        .format(i, len(data_loader), batch_time=batch_time)
                logger.info(msg)
                if config.DEBUG.DEBUG:
                    prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                    save_debug_images(config, input, meta, None, dbg_pred*4, None, prefix)

    return all_preds


def main():
    args = parse_args()

    config, model, gpus, logger, final_output_dir = load_model(args)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_set = dataset.coco_simple(
        config,
        config.DATASET.ROOT,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate
    all_preds = evaluate(config, data_loader, data_set, model, final_output_dir, logger)

    # 예측된 결과를 포함한다.
    print(all_preds.shape)
    # print(all_preds)
    for i, pred in enumerate(all_preds):
        if i > 10: break
        print('%02d predection result: ' % i)
        for j, point in enumerate(pred):
            if point[2] < 0.7: continue # confidence가 0.7 이상인 경우만 프린트 한다.
            print('\t %02d) x: %6.2f, y: %6.2f, confidence: %.3f' % (j, point[0], point[1], point[2]))


if __name__ == '__main__':
    main()
