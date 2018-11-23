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

parser.add_argument("--epochs", type=int, default=51, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/radio.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/radio.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_official_pytorch.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/radio.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
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
print(model)

use_numpy = False
model.load_weights(args.weights_path, from_numpy=use_numpy, change_num_classes=True)
#model.save_weights("weights/yolov3_original.pth", as_numpy=False)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# Get current epoch
pattern = re.compile(r"(?<=/)\d{2,5}(?=\.weights)")
res = pattern.search(args.weights_path)
if res:
    start_epoch = int(res.group(0)) + 1
else:
    start_epoch = 0

print("     epoch starts from {} to {}".format(start_epoch, start_epoch + args.epochs))

for epoch in range(start_epoch, start_epoch + args.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        # logger.info("batch imgs: {}".format(imgs.size()))
        # logger.info("batch targets: {}".format(targets.size()))

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        if batch_i % 10 == 0:
            logger.info(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    start_epoch + args.epochs,
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
        model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch), as_numpy=use_numpy)
