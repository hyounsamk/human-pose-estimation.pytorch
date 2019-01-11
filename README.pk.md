# Human Pose Estimation

```sh
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
