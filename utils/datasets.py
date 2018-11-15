import glob
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image

from skimage.transform import resize

import logging
import copy
import utils.boxes as box_utils
# COCO API
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))

        # Repeat gray image in 3 channels
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)



class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        # with open(list_path, 'r') as file:
        #     self.img_files = file.readlines()
        # self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]

        #########################################################################
        # Load dataset in COCO format
        #########################################################################
        self.COCO = COCO(list_path)
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        # get all images
        self.image_ids = self.COCO.getImgIds()
        self.image_ids.sort()
        self.roidb = copy.deepcopy(self.COCO.loadImgs((self.image_ids)))

        # filter images(entries) without bboxes
        num = len(self.roidb)
        self._filter_roidb()
        num_after =len(self.roidb)
        logger.info("Filtered {} roidb entries: {} -> {}".
                    format(num-num_after, num, num_after))
        # get labels(cls+bbox)
        for entry in self.roidb:
            entry['boxes'] = np.empty((0, 4), dtype=np.float32)
            entry['gt_classes'] = np.empty((0), dtype=np.int32)
            self._add_gt_boxes(entry)

        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):
        #---------
        #  Image
        #---------

        #img_path = self.img_files[index % len(self.img_files)].rstrip()
        image_idx = index % len(self.roidb)
        img_path = self.roidb[image_idx]['file_name']
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        if len(img.shape) != 3:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
            # logger.info("image shape after repeat along channels: {}".format(img.shape))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------
        gt_boxes = self.roidb[image_idx]['boxes']   # [N, x_yolo, y_yolo, w_yolo, h_yolo]
        gt_classes = self.roidb[image_idx]['gt_classes'].reshape((-1, 1))

        labels = np.concatenate((gt_classes, gt_boxes), axis=1)
        # Extract coordinates for unpadded + unscaled image
        x1 = w * (labels[:, 1] - labels[:, 3]/2)
        y1 = h * (labels[:, 2] - labels[:, 4]/2)
        x2 = w * (labels[:, 1] + labels[:, 3]/2)
        y2 = h * (labels[:, 2] + labels[:, 4]/2)

        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates, relative values
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def _add_gt_boxes(self, entry):
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        valid_objs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # Convert from (x1, y1, w, h) to (x_center/img_w, y_center/img_h, w/img_w, h/img_h)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            x_yolo, y_yolo, w_yolo, h_yolo = box_utils.xyxy_to_xywh_yolo([x1, y1, x2, y2], height, width)

            if obj['area']>0 and x2>x1 and y2>y1:
                obj['clean_bbox'] = [x_yolo, y_yolo, w_yolo, h_yolo]
                valid_objs.append(obj)
        num_valid_objs = len(valid_objs)
        gt_boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            gt_boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
        entry['boxes'] = np.append(entry['boxes'], gt_boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)

    def _filter_roidb(self):
        # remove images(entries) that don't have bboxes
        valid_roidb = [entry for entry in self.roidb \
                       if len(self.COCO.getAnnIds(imgIds=[entry['id']])) > 0]
        self.roidb = valid_roidb

    def __len__(self):
        return len(self.roidb)
