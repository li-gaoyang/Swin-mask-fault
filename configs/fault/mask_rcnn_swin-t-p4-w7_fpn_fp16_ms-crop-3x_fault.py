# -*- coding=utf-8 -*-


_base_ = '../swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))


# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))



# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('fault',)
data = dict(
    train=dict(
        img_prefix='./data/fault/train_val_swin_12_12/train/',
        classes=classes,
        ann_file='./data/fault/train_val_swin_12_12/annotations/train.json'),
    val=dict(
        img_prefix='./data/fault/train_val_swin_12_12/val/',
        classes=classes,
        ann_file='./data/fault/train_val_swin_12_12/annotations/val.json'),
    # test=dict(
    #     img_prefix='./data/fault/test/',
    #     classes=classes,
    #     ann_file='./data/fault/annotations/test.json')
      )
# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能

#python tools/train.py configs/fault/mask_rcnn_swin-t-p4-w7_fpn_1x_fault.py --work-dir work_dirs/mask_rcnn_swin-t-p4-w7_fpn_1x_fault --resume-from work_dirs/mask_rcnn_swin-t-p4-w7_fpn_1x_fault/latest.pth

#python tools/train.py configs/fault/mask_rcnn_swin-t-p4-w7_fpn_1x_fault.py

lr_config = dict(step=[50])
runner = dict(max_epochs=5000)

 




