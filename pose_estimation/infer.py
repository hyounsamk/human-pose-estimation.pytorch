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


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
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
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
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

def evaluate(config, data_loader, model, output_dir, logger):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #meta = {}
    with torch.no_grad():
        end = time.time()
        for i, (input, _, __, meta) in enumerate(data_loader):
            # compute output
            output = model(input)

            if config.DEBUG.DEBUG:
                pred, maxvals = get_max_preds(output.cpu().numpy())
                # pred는 각 포인트들의 좌표값(x, y) array이다. array size는 dataset마다 차이가 있다.
                # maxvals는 각 포인트들의 confidence(0과 1사이)값 array이다. pred의 같은 갯수이다.

                maxvals = np.round_(maxvals) # 0.5 이상은 1로 반올림. 
                meta['joints_vis'] = maxvals # visualization 시에는 값이 1인 포인트만 표시한다.

                # 각 포인트들의 최종 좌표값들을 가져온다.
                # 최종 좌표값은 ...
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                pred, maxvals = get_final_preds(config, output.cpu().numpy(), c, s)

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
                    save_debug_images(config, input, meta, None, pred*4, None, prefix)


def main():
    args = parse_args()
    reset_config(config, args)

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

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # evaluate
    evaluate(config, data_loader, model, final_output_dir, logger)


if __name__ == '__main__':
    main()
