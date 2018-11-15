from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import logging
from utils.logging import setup_logging

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('yolov3.train').setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/radio.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
args = parser.parse_args()
print(args)

cuda = torch.cuda.is_available() and args.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# classes = load_classes(args.class_path)

# Get data configuration
data_config = parse_data_config(args.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(args.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(args.model_config_path)
model.apply(weights_init_normal)
model.load_weights(args.weights_path)

if cuda:
    model = model.cuda()

model.train()

########################################################################
# Prepare Dataset
########################################################################
# from datasets.roidb import combined_roidb_for_training
# from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
# from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
# from utils.timer import Timer

#
# if args.dataset.startswith("radio"):
#     cfg.TRAIN.DATASETS = (args.dataset,)
#     cfg.MODEL.NUM_CLASSES = 2
# else:
#     raise ValueError("Unexpected args.dataset: {}".format(args.dataset))
# cfg_from_file(args.cfg_file)
# if args.set_cfgs is not None:
#     cfg_from_list(args.set_cfgs)
#
# ### Adaptively adjust some configs ###
# original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
# original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
# original_num_gpus = cfg.NUM_GPUS
# if args.batch_size is None:
#     args.batch_size = original_batch_size
# cfg.NUM_GPUS = torch.cuda.device_count()
# assert (args.batch_size % cfg.NUM_GPUS) == 0, \
#     'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
# cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
# effective_batch_size = args.iter_size * args.batch_size
# logger.info('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))
#
# logger.info('Adaptive config changes:')
# logger.info('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
# logger.info('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
# logger.info('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))
#
# if args.n_cpu is not None:
#     cfg.DATA_LOADER.NUM_THREADS = args.n_cpu
# print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)
#
# timers = defaultdict(Timer)
# timers['roidb'].tic()
# roidb, ratio_list, ratio_index = combined_roidb_for_training(
#     cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
# timers['roidb'].toc()
# roidb_size = len(roidb)
# logger.info('{:d} roidb entries'.format(roidb_size))
# logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)
#
# # Effective training sample size for one epoch
# train_size = roidb_size // args.batch_size * args.batch_size
#
# batchSampler = BatchSampler(
#     sampler=MinibatchSampler(ratio_list, ratio_index),
#     batch_size=args.batch_size,
#     drop_last=True
# )
#
# dataset = RoiDataLoader(
#     roidb,
#     cfg.MODEL.NUM_CLASSES,
#     training=True)
#
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_sampler=batchSampler,
#     num_workers=cfg.DATA_LOADER.NUM_THREADS,
#     collate_fn=collate_minibatch)

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(args.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        logger.info("batch imgs: {}".format(imgs.size()))
        logger.info("batch targets: {}".format(targets.size()))

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        logger.info(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                args.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % args.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch))
