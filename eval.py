import os
import time
import torch
import yaml
import json

import cv2 as cv
import numpy as np
from tqdm import tqdm
from nets.mask_rcnn import MaskRCNN
from datasets.coco import coco_ids, rgb_mean, rgb_std, make_divisible, colors, coco_names
from utils.augmentations import RandScaleMinMax, BoxSegInfo
from utils.model_utils import AverageLogger
from pycocotools import mask as maskUtils


def coco_eavl(anno_path="/home/huffman/data/annotations/instances_val2017.json",
              pred_path="predicts.json",
              type="bbox"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(anno_path)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, type)
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


@torch.no_grad()
def eval_model(weight_path="weights/mask_rcnn_resnet50_last.pth", device="cuda:0"):
    from pycocotools.coco import COCO
    device = torch.device(device)
    with open("config/maskrcnn.yaml", 'r') as rf:
        cfg = yaml.safe_load(rf)
    net = MaskRCNN(**{**cfg['model'], 'pretrained': False})
    net.load_state_dict(torch.load(weight_path, map_location="cpu")['ema'])
    net.to(device)
    net.eval()
    data_cfg = cfg['data']
    basic_transform = RandScaleMinMax(min_threshes=[640], max_thresh=data_cfg['max_thresh'])
    coco = COCO(data_cfg['val_annotation_path'])
    coco_predict_list = list()
    time_logger = AverageLogger()
    pbar = tqdm(coco.imgs.keys())
    for img_id in pbar:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(data_cfg['val_img_root'], file_name)
        img = cv.imread(img_path)
        h, w, _ = img.shape
        img, ratio = basic_transform.scale_img(img,
                                               min_thresh=640)
        h_, w_ = img.shape[:2]
        padding_size = make_divisible(max(h_, w_), 64)
        img_inp = np.ones((padding_size, padding_size, 3)) * np.array((103, 116, 123))
        img_inp[:h_, :w_, :] = img
        img_inp = (img_inp[:, :, ::-1] / 255.0 - np.array(rgb_mean)) / np.array(rgb_std)
        img_inp = torch.from_numpy(img_inp).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float().to(device)
        tic = time.time()
        predict = net(img_inp, valid_size=[(padding_size, padding_size)])[0][0]
        duration = time.time() - tic
        time_logger.update(duration)
        pbar.set_description("fps:{:4.2f}".format(1 / time_logger.avg()))
        if predict is None:
            continue
        predict[:, [0, 2]] = (predict[:, [0, 2]] / ratio).clamp(min=0, max=w)
        predict[:, [1, 3]] = (predict[:, [1, 3]] / ratio).clamp(min=0, max=h)
        box = predict.cpu().numpy()
        coco_box = box[:, :4]
        coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
        for p, b in zip(box.tolist(), coco_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)
    coco_eavl(anno_path=data_cfg['val_annotation_path'], pred_path="predicts.json")


def visualize_model(weight_path="weights/mask_rcnn_resnet50_last.pth", device="cuda:0"):
    from pycocotools.coco import COCO
    device = torch.device(device)
    with open("config/maskrcnn.yaml", 'r') as rf:
        cfg = yaml.safe_load(rf)
    #     "box_score_thresh": 0.8
    net = MaskRCNN(**{**cfg['model'], 'pretrained': False, })
    net.load_state_dict(torch.load(weight_path, map_location="cpu")['ema'])
    net.to(device)
    net.eval()
    data_cfg = cfg['data']
    basic_transform = RandScaleMinMax(min_threshes=[640], max_thresh=data_cfg['max_thresh'])
    coco = COCO(data_cfg['val_annotation_path'])
    coco_predict_list = list()
    time_logger = AverageLogger()
    pbar = tqdm(coco.imgs.keys())
    i = 0
    for img_id in pbar:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(data_cfg['val_img_root'], file_name)
        img = cv.imread(img_path)
        # ori_img = img.copy()
        h, w, _ = img.shape
        img, ratio = basic_transform.scale_img(img,
                                               min_thresh=640)
        h_, w_ = img.shape[:2]
        padding_size = make_divisible(max(h_, w_), 64)
        img_inp = np.ones((padding_size, padding_size, 3)) * np.array((103, 116, 123))
        img_inp[:h_, :w_, :] = img
        img_inp = (img_inp[:, :, ::-1] / 255.0 - np.array(rgb_mean)) / np.array(rgb_std)
        img_inp = torch.from_numpy(img_inp).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float().to(device)
        tic = time.time()
        predict = net(img_inp, valid_size=[(padding_size, padding_size)])
        duration = time.time() - tic
        time_logger.update(duration)
        pbar.set_description("fps:{:4.2f}".format(1 / time_logger.avg()))
        ret = net.project_to_image(predict, [(w, h, ratio)])[0]
        box, mask = ret
        if len(box) == 0:
            continue
        box = box.cpu().numpy()
        mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)
        # mask = mask.cpu().numpy()
        for p, m in zip(box.tolist(), mask):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      # 'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5),
                                      'segmentation': maskUtils.encode(np.asfortranarray(m))})
        # box_seg_info = BoxSegInfo(img=ori_img, shape=(w, h), boxes=box[:, :4], labels=box[:, -1], mask=mask)
        # ret_img = box_seg_info.draw_mask(colors, coco_names)
        # import uuid
        # file_name = str(uuid.uuid4()).replace("-", "")
        # cv.imwrite("{:s}.jpg".format(file_name), ret_img)
        # i += 1
        # if i == 20:
        #     break
        # with open("predicts.json", 'w') as file:
        #     json.dump(coco_predict_list, file)
    coco_eavl(anno_path=data_cfg['val_annotation_path'], pred_path=coco_predict_list, type="segm")


if __name__ == '__main__':
    visualize_model()
#     26.09
#     23.64
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.390
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.598
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.436
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.530
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.508
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.534
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.330
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.705

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.557
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.353
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.128
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.530
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.443
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651
