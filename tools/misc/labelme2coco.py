
import os.path as osp
import mmcv
import os
import sys
sys.path.append(os.getcwd())
def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []
        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'fault'}])
    mmcv.dump(coco_format_json, out_file)
    
    

# 对验证集数据进行处理是，将下面路径中的train 替换成val 即可
# 注意数据集 balloon 的路径自行调整

# ann_file = 'data/balloon/train/via_region_data.json'
# out_file = 'data/balloon/train/annotation_coco.json'
# image_prefix = 'data/balloon/train'

# 路径方式---
ann_file = 'data/fault/train/via_region_data.json'
out_file = 'data/fault/annotations/annotation_fault_train.json'
image_prefix = 'data/fault/train'

# 执行程序
convert_balloon_to_coco(ann_file, out_file, image_prefix)