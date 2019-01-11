# Human Pose Estimation

```sh
# mpii
python pose_estimation/valid.py \
    --frequent 1 \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar


python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet152/384x384_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_152_384x384.pth.tar

python pose_estimation/infer.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar \
    --im-file data/mpii/images/000001163.jpg

```

```sh
# coco_custom
python pose_estimation/valid.py \
    --frequent 1 \
    --cfg experiments/coco_custom/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar
# coco
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar

```

## Inference

```sh
# coco_custom resnet_50_256x192
python pose_estimation/infer.py \
    --frequent 1 \
    --cfg experiments/coco_custom/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar
# coco_custom resnet_152_384x288
python pose_estimation/infer.py \
    --frequent 1 \
    --cfg experiments/coco_custom/resnet152/384x288_d256x3_adam_lr1e-3.yaml \
    --model-file models/pytorch/pose_coco/pose_resnet_152_384x288.pth.tar

```