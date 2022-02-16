# general use libraries
import os
import sys
from tqdm import tqdm, trange
import numpy as np
from skimage import io
from skimage.draw import rectangle_perimeter, set_color
from kmeans_pytorch import kmeans

# Brevitas ad PyTorch libraries
import torch
import torch.utils.tensorboard
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import (
    Module,
    Sequential,
    BatchNorm2d,
    Conv2d,
    ReLU,
    MaxPool2d,
    MSELoss,
    BCELoss,
)
from brevitas.nn import (
    QuantIdentity,
    QuantConv2d,
    QuantReLU,
    QuantMaxPool2d,
)
from brevitas.inject.defaults import *
from brevitas.core.restrict_val import RestrictValueType
from brevitas.onnx import export_finn_onnx as exportONNX

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

#############################################
#               Configurations              #
#############################################


# size of output layer
O_SIZE = 5  # bb_x, bb_y, bb_w, bb_h, bb_conf

# bounding boxes grid output shape
GRID_SIZE = torch.tensor([20, 20])  # (y, x)

# Flag to use QuantTensor or not
QUANT_TENSOR = True

#############################################
#               Dataset                     #
#############################################


class YOLO_dataset(Dataset):
    def __init__(
        self, img_dir, lbl_dir, len_lim=-1, transform=None, grid_size=GRID_SIZE
    ):
        self.img_dir = img_dir
        self.imgs = sorted(os.listdir(self.img_dir))[:len_lim]
        self.lbl_dir = lbl_dir
        self.lbls = sorted(os.listdir(self.lbl_dir))[:len_lim]
        self.transform = transform
        self.grid_size = grid_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(os.path.join(self.img_dir, self.imgs[idx]))

        with open(os.path.join(self.lbl_dir, self.lbls[idx])) as f:
            dataline = f.readlines()[1]
            lbl_data = [data.strip() for data in dataline.split("\t")]
            lbl = np.array(lbl_data).astype(float)
            f.close()

        sample = [img, lbl]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        img, lbl = sample

        img = img.transpose((2, 0, 1))
        return [torch.from_numpy(img), torch.from_numpy(lbl)]


class Normalize(object):
    def __call__(self, sample, mean=0.5, std=0.5):
        img, lbl = sample

        img = ((img / 255) - mean) / std

        return [img, lbl]


#############################################
#               Quantizers                  #
#############################################

from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver


class PerTensorFloatScaling(ExtendedInjector):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP


class WeightQuant(IntQuant, MaxStatsScaling, PerTensorFloatScaling, WeightQuantSolver):
    pass


class ActQuant(
    UintQuant, ParamFromRuntimePercentileScaling, PerTensorFloatScaling, ActQuantSolver
):
    pass


#############################################
#               Network                     #
#############################################


class QTinyYOLOv2(Module):
    def __init__(
        self, n_anchors, weight_bit_width=8, act_bit_width=8, quant_tensor=QUANT_TENSOR,
    ):
        super(QTinyYOLOv2, self).__init__()
        self.weight_bit_width = int(np.clip(weight_bit_width, 1, 8))
        self.act_bit_width = int(np.clip(act_bit_width, 1, 8))
        self.n_anchors = n_anchors

        self.input = QuantIdentity(
            act_quant=Int8ActPerTensorFloatMinMaxInit,
            min_val=-1.0,
            max_val=1.0 - 2.0 ** (-7),
            signed=True,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            return_quant_tensor=quant_tensor,
        )
        self.conv1 = Sequential(
            QuantConv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=8,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(16),
            QuantReLU(
                act_quant=ActQuant, bit_width=8, return_quant_tensor=quant_tensor
            ),
            QuantMaxPool2d(2, 2, return_quant_tensor=quant_tensor),
        )
        self.conv2 = Sequential(
            QuantConv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(32),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            QuantMaxPool2d(2, 2, return_quant_tensor=quant_tensor),
        )
        self.conv3 = Sequential(
            QuantConv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(64),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            QuantMaxPool2d(2, 2, return_quant_tensor=quant_tensor),
        )
        self.conv4 = Sequential(
            QuantConv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(128),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            QuantMaxPool2d(2, 2, return_quant_tensor=quant_tensor),
        )
        self.conv5 = Sequential(
            QuantConv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(256),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            QuantMaxPool2d(2, 2, return_quant_tensor=quant_tensor),
        )
        self.conv6 = Sequential(
            QuantConv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(512),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            # QuantMaxPool2d(2, 2, return_quant_tensor=quant_tensor),
        )
        self.conv7 = Sequential(
            QuantConv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(512),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
        )
        self.conv8 = Sequential(
            QuantConv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                weight_quant=WeightQuant,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(512),
            QuantReLU(
                act_quant=ActQuant,
                bit_width=self.act_bit_width,
                return_quant_tensor=quant_tensor,
            ),
        )
        self.conv9 = QuantConv2d(
            in_channels=512,
            out_channels=self.n_anchors * O_SIZE,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=WeightQuant,
            weight_bit_width=8,
            return_quant_tensor=quant_tensor,
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        return x


class TinyYOLOv2(Module):
    def __init__(self):
        super(TinyYOLOv2, self).__init__()
        self.n_anchors = n_anchors

        self.conv1 = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.conv2 = Sequential(
            Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.conv3 = Sequential(
            Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.conv4 = Sequential(
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.conv5 = Sequential(
            Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.conv6 = Sequential(
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(512),
            ReLU(),
            # MaxPool2d(2, 2),
        )
        self.conv7 = Sequential(
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(512),
            ReLU(),
        )
        self.conv8 = Sequential(
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(512),
            ReLU(),
        )
        self.conv9 = Conv2d(
            in_channels=512,
            out_channels=self.n_anchors * O_SIZE,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        return x


def YOLOout(output, anchors, device, findBB):

    output = output.flatten(-2, -1)
    output = output.transpose(-2, -1)
    output = output.view(output.size(0), GRID_SIZE.prod(), anchors.size(0), O_SIZE)

    gx = (
        (
            (
                torch.arange(GRID_SIZE[1]).repeat_interleave(
                    GRID_SIZE[0] * anchors.size(0)
                )
            )
            / GRID_SIZE[1]
        ).view(GRID_SIZE.prod(), anchors.size(0))
    ).to(device)
    gy = (
        (
            (torch.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])).repeat_interleave(
                anchors.size(0)
            )
            / GRID_SIZE[0]
        ).view(GRID_SIZE.prod(), anchors.size(0))
    ).to(device)
    output[..., 0] = (torch.sigmoid(output[..., 0]) / GRID_SIZE[0]) + gx
    output[..., 1] = (torch.sigmoid(output[..., 1]) / GRID_SIZE[1]) + gy
    output[..., 2] = torch.exp(output[..., 2]) * anchors[:, 0]
    output[..., 3] = torch.exp(output[..., 3]) * anchors[:, 1]
    output[..., 4] = torch.sigmoid(output[..., 4])

    # find the most probable grid and box
    if findBB:
        # localizing most probable bounding box
        bb_conf, bb_idx = torch.max(output[..., -1], -1)
        g_idx = bb_conf.argmax(-1)
        output = output[
            torch.arange(output.size(0)),
            g_idx,
            bb_idx[torch.arange(output.size(0)), g_idx],
            :4,
        ]

    return output


#############################################
#               Loss func                   #
#############################################


class YOLOLoss(Module):
    def __init__(
        self,
        anchors,
        device,
        loss_fnc="yolo",
        l_coor_obj=5.0,
        l_coor_noobj=5.0,
        l_conf_obj=1.0,
        l_conf_noobj=0.5,
    ):
        super().__init__()
        self.anchors = anchors.to(device)
        self.device = device
        self.loss_fnc = loss_fnc
        self.l_coor_obj = l_coor_obj
        self.l_coor_noobj = l_coor_noobj
        self.l_conf_obj = l_conf_obj
        self.l_conf_noobj = l_conf_noobj
        self.mse = MSELoss()
        self.bce = BCELoss()
        self.i = 0

    def forward(self, pred_, label):
        global logger
        # locate bounding box location and
        idx_x = (label[:, 0] * GRID_SIZE[1]).floor()
        idx_y = (label[:, 1] * GRID_SIZE[0]).floor()
        idx = (idx_x * GRID_SIZE[0] + idx_y).type(torch.int64)
        # convert predictions to label style
        pred = YOLOout(pred_, self.anchors, self.device, False)
        # find closest anchor
        anchor_mask = (
            (
                (
                    label[:, 2:4].unsqueeze(1).repeat((1, self.anchors.size(0), 1))
                    - self.anchors.unsqueeze(0).repeat((label.size(0), 1, 1))
                )
                ** 2.0
            ).sum(2)
        ).argmax(1)
        # create anchors grid
        anchors_grid = torch.ones_like(pred[..., :4])
        anchors_grid[..., 0] = (
            (
                torch.arange(GRID_SIZE[1]).repeat_interleave(
                    GRID_SIZE[0] * self.anchors.size(0)
                )
                + 0.5
            )
            / GRID_SIZE[1]
        ).view(GRID_SIZE.prod(), self.anchors.size(0))
        anchors_grid[..., 1] = (
            (
                (torch.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])).repeat_interleave(
                    self.anchors.size(0)
                )
                + 0.5
            )
            / GRID_SIZE[0]
        ).view(GRID_SIZE.prod(), self.anchors.size(0))
        anchors_grid[..., 2:4] = self.anchors
        # mask of obj and noobj
        obj_mask = (torch.zeros_like(pred)).type(torch.bool)
        for i in range(obj_mask.size(0)):
            obj_mask[i, idx[i], anchor_mask[i], :] = True
        noobj_mask = ~obj_mask
        # prepare loss calculation parts
        pred_obj = pred[obj_mask].view(pred.shape[0], 5)
        pred_noobj = pred[noobj_mask].view(pred.shape[0], -1, 5)
        anchors_grid = anchors_grid[noobj_mask[..., :4]].view(pred.shape[0], -1, 4)
        pred_obj = pred_obj.to(self.device)
        pred_noobj = pred_noobj.to(self.device)
        anchors_grid = anchors_grid.to(self.device)

        if self.loss_fnc == "yolo":
            # coordination loss
            coor_l_obj = self.mse(pred_obj[:, :2], label[:, :2]) + self.mse(
                pred_obj[:, 2:4].sqrt(), label[:, 2:].sqrt()
            )
            # confidence loss
            conf_l_obj = self.mse(
                pred_obj[:, 4], torch.ones_like(pred_obj[:, 4], device=self.device)
            )
            conf_l_noobj = self.mse(
                pred_noobj[..., 4],
                torch.zeros_like(pred_noobj[..., 4], device=self.device),
            )
            # log loss parts
            logger.add_scalar("LossParts/coor_l_obj", coor_l_obj, self.i)
            logger.add_scalar("LossParts/conf_l_obj", conf_l_obj, self.i)
            logger.add_scalar("LossParts/conf_l_noobj", conf_l_noobj, self.i)
            self.i += 1
            # return loss
            return (
                coor_l_obj * self.l_coor_obj
                + conf_l_obj * self.l_conf_obj
                + conf_l_noobj * self.l_conf_noobj
            )

        elif self.loss_fnc == "yolov2":
            # coordination loss
            coor_l_obj = self.mse(pred_obj[:, :4], label)
            coor_l_noobj = self.mse(pred_noobj[..., :4], anchors_grid)
            # confidence loss
            conf_l_obj = self.mse(pred_obj[:, 4], IoU_calc(pred_obj, label))
            conf_l_noobj = self.mse(
                pred_noobj[..., 4],
                torch.zeros_like(pred_noobj[..., 4], device=self.device),
            )
            # log loss parts
            logger.add_scalar("LossParts/coor_l_obj", coor_l_obj, self.i)
            logger.add_scalar("LossParts/coor_l_noobj", coor_l_noobj, self.i)
            logger.add_scalar("LossParts/conf_l_obj", conf_l_obj, self.i)
            logger.add_scalar("LossParts/conf_l_noobj", conf_l_noobj, self.i)
            self.i += 1
            # return loss
            return (
                coor_l_obj * self.l_coor_obj
                + coor_l_noobj * self.l_coor_noobj
                + conf_l_obj * self.l_conf_obj
                + conf_l_noobj * self.l_conf_noobj
            )

        elif self.loss_fnc == "yolov3":
            # coordination loss
            coor_l_obj = self.bce(pred_obj[:, :2], label[:, :2]) + self.mse(
                pred_obj[:, 2:4], label[:, 2:]
            )
            # confidence loss
            conf_l_obj = self.bce(
                pred_obj[:, 4], torch.ones_like(pred_obj[:, 4], device=self.device)
            )
            conf_l_noobj = self.bce(
                pred_noobj[..., 4],
                torch.zeros_like(pred_noobj[..., 4], device=self.device),
            )
            # log loss parts
            logger.add_scalar("LossParts/coor_l_obj", coor_l_obj, self.i)
            logger.add_scalar("LossParts/coor_l_noobj", coor_l_noobj, self.i)
            logger.add_scalar("LossParts/conf_l_obj", conf_l_obj, self.i)
            logger.add_scalar("LossParts/conf_l_noobj", conf_l_noobj, self.i)
            self.i += 1
            # return loss
            return (
                coor_l_obj * self.l_coor_obj
                + conf_l_obj * self.l_conf_obj
                + conf_l_noobj * self.l_conf_noobj
            )

        else:
            print("User error: Unsupported loss function")
            exit()


#############################################
#               Util funcs                  #
#############################################


def getAnchors(dataset, n_anchors, device):
    datapoints = False
    # collect labels data
    for (_, label) in tqdm(
        dataset, total=len(dataset), desc="kmeans - data read", unit="batch"
    ):
        data = label[:, 2:]
        if torch.is_tensor(datapoints):
            datapoints = torch.vstack([datapoints, data])
        else:
            datapoints = data
    # k-means clustering
    kmean_idx, anchors = kmeans(
        X=datapoints, num_clusters=n_anchors, distance="euclidean", device=device
    )
    print(f"Anchors for k={n_anchors}:")
    [print(f"[{anchor[0]: .8f}, {anchor[1]: .8f}]") for anchor in anchors]
    return anchors


def IoU_calc(pred, label):
    # xmin, ymin, xmax, ymax
    label_bb = torch.stack(
        [
            torch.max(label[:, 0] - (label[:, 2] / 2), torch.tensor(0.0)),
            torch.max(label[:, 1] - (label[:, 3] / 2), torch.tensor(0.0)),
            torch.min(label[:, 0] + (label[:, 2] / 2), torch.tensor(1.0)),
            torch.min(label[:, 1] + (label[:, 3] / 2), torch.tensor(1.0)),
        ],
        1,
    )
    pred_bb = torch.stack(
        [
            torch.max(pred[:, 0] - (pred[:, 2] / 2), torch.tensor(0.0)),
            torch.max(pred[:, 1] - (pred[:, 3] / 2), torch.tensor(0.0)),
            torch.min(pred[:, 0] + (pred[:, 2] / 2), torch.tensor(1.0)),
            torch.min(pred[:, 1] + (pred[:, 3] / 2), torch.tensor(1.0)),
        ],
        1,
    )
    inter_bb = torch.stack(
        [
            torch.max(label_bb[:, 0], pred_bb[:, 0]),
            torch.max(label_bb[:, 1], pred_bb[:, 1]),
            torch.min(label_bb[:, 2], pred_bb[:, 2]),
            torch.min(label_bb[:, 3], pred_bb[:, 3]),
        ],
        1,
    )
    # calculate IoU
    label_area = label[:, 2] * label[:, 3]
    pred_area = pred[:, 2] * pred[:, 3]
    inter_area = torch.max(
        inter_bb[:, 2] - inter_bb[:, 0], torch.tensor(0.0)
    ) * torch.max(inter_bb[:, 3] - inter_bb[:, 1], torch.tensor(0.0))
    # return IoU
    return inter_area / (label_area + pred_area - inter_area)


def AssessBBDiffs(pred, label):
    # calculate Centres Distance
    CentDist = (
        (pred[:, 0] - label[:, 0]) ** 2.0 + (pred[:, 1] - label[:, 1]) ** 2.0
    ).sqrt()
    # calculate Size Ratio - pred/true
    SizeRatio = pred[:, 0:2].prod(-1) / label[:, 0:2].prod(-1)
    # return Centres Distance, and Size Ratio
    return [CentDist, SizeRatio]


def getXYminmax(pred, label, w=640, h=640):
    pred_ = torch.stack(
        [
            (
                torch.max(pred[..., 0] - (pred[..., 2] / 2), torch.tensor(0.0)) * w
            ).round(),
            (
                torch.max(pred[..., 1] - (pred[..., 3] / 2), torch.tensor(0.0)) * h
            ).round(),
            (
                torch.min(pred[..., 0] + (pred[..., 2] / 2), torch.tensor(1.0)) * w
            ).round(),
            (
                torch.min(pred[..., 1] + (pred[..., 3] / 2), torch.tensor(1.0)) * h
            ).round(),
        ],
        -1,
    )
    label_ = torch.stack(
        [
            (
                torch.max(label[..., 0] - (label[..., 2] / 2), torch.tensor(0.0)) * w
            ).round(),
            (
                torch.max(label[..., 1] - (label[..., 3] / 2), torch.tensor(0.0)) * h
            ).round(),
            (
                torch.min(label[..., 0] + (label[..., 2] / 2), torch.tensor(1.0)) * w
            ).round(),
            (
                torch.min(label[..., 1] + (label[..., 3] / 2), torch.tensor(1.0)) * h
            ).round(),
        ],
        -1,
    )
    return torch.stack([pred_, label_], 0)


#############################################
#               Training                    #
#############################################


def train(
    img_dir,
    lbl_dir,
    weight_bit_width=8,
    act_bit_width=8,
    anchors=False,
    n_anchors=5,
    n_epochs=100,
    batch_size=1,
    len_lim=-1,
    loss_fnc="yolo",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trainig on: {device}")

    # logger
    global logger
    logger = torch.utils.tensorboard.SummaryWriter(
        comment=f"W{weight_bit_width}A{act_bit_width}a{n_anchors}"
    )

    # dataset
    transformers = transforms.Compose([ToTensor(), Normalize()])
    dataset = YOLO_dataset(img_dir, lbl_dir, len_lim=len_lim, transform=transformers)
    # split dataset to train:valid:test - 60:20:20 ratio
    # [Train - Test - Train - Valid - Train ...]
    idx = np.arange(len(dataset), dtype=int)
    test_idx = idx[1::5].tolist()
    valid_idx = idx[3::5].tolist()
    train_idx = np.delete(idx, np.append(test_idx, valid_idx))
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)
    valid_set = Subset(dataset, valid_idx)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # get anchors
    save_anchors = True
    if torch.is_tensor(anchors):
        anchors = anchors.to(device)
        n_anchors = anchors.size(0)
        save_anchors = False
    else:
        print("Calculating Anchors")
        anchors = (getAnchors(train_loader, n_anchors, device)).to(device)

    # network setup
    net = QTinyYOLOv2(n_anchors, weight_bit_width, act_bit_width)
    net = net.to(device)
    loss_func = YOLOLoss(anchors, device, loss_fnc=loss_fnc)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=100, threshold=1e-5
    )

    # train network
    print("Training Start")
    for epoch in trange(n_epochs, desc="epoch", unit="epoch"):
        # train + train loss
        net.train()
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc="train loss",
            unit="batch",
        ):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            if QUANT_TENSOR:
                loss = loss_func(outputs.value.float(), labels.float())
            else:
                loss = loss_func(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            logger.add_scalar(
                "Loss/train_continuous", loss.item(), i + epoch * len(train_loader)
            )
            scheduler.step(loss)
        # log loss statistics
        logger.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
        # valid loss
        net.eval()
        with torch.no_grad():
            for i, data in tqdm(
                enumerate(valid_loader, 0),
                total=len(valid_loader),
                desc="valid loss",
                unit="batch",
            ):
                valid_images, valid_labels = data[0].to(device), data[1].to(device)
                valid_outputs = net(valid_images)
                if QUANT_TENSOR:
                    t_loss = loss_func(
                        valid_outputs.value.float(), valid_labels.float()
                    )
                else:
                    t_loss = loss_func(valid_outputs.float(), valid_labels.float())
                valid_loss += t_loss.item()
                logger.add_scalar(
                    "Loss/valid_continuous",
                    t_loss.item(),
                    i + epoch * len(valid_loader),
                )
        # log loss statistics
        logger.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)

        # train accuracy
        with torch.no_grad():
            train_miou = 0.0
            train_AP50 = 0.0
            train_AP75 = 0.0
            train_ctrdist = 0.0
            train_ratio = 0.0
            train_total = 0
            for i, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc="train accuracy",
                unit="batch",
            ):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                if QUANT_TENSOR:
                    bb_outputs = YOLOout(outputs.value, anchors, device, True)
                else:
                    bb_outputs = YOLOout(outputs, anchors, device, True)
                iou = IoU_calc(bb_outputs, labels)
                ctrDist, ratio = AssessBBDiffs(bb_outputs, labels)
                train_total += labels.size(0)
                train_miou += iou.sum()
                train_AP50 += (iou >= 0.5).sum()
                train_AP75 += (iou >= 0.75).sum()
                train_ctrdist += ctrDist.sum()
                train_ratio += ratio.sum()
                # sample images
                if i % (len(train_loader) // 2) == 0:
                    logger.add_image_with_boxes(
                        "TrainingResults",
                        images[0],
                        getXYminmax(bb_outputs[0], labels[0]),
                        labels=["prediction", "true"],
                    )
            # log accuracy statistics
            logger.add_scalar(
                "TrainingAcc/meanIoU", train_miou / len(train_loader), epoch
            )
            logger.add_scalar(
                "TrainingAcc/meanAP50", train_AP50 / len(train_loader), epoch
            )
            logger.add_scalar(
                "TrainingAcc/meanAP75", train_AP75 / len(train_loader), epoch
            )
            logger.add_scalar(
                "TrainingAcc/CtrDist", train_ctrdist / len(train_loader), epoch,
            )
            logger.add_scalar(
                "TrainingAcc/SizesRatio", train_ratio / len(train_loader), epoch
            )
        # valid accuracy
        with torch.no_grad():
            valid_miou = 0.0
            valid_AP50 = 0.0
            valid_AP75 = 0.0
            valid_ctrdist = 0.0
            valid_ratio = 0.0
            valid_total = 0
            for i, data in tqdm(
                enumerate(valid_loader),
                total=len(valid_loader),
                desc="valid accuracy",
                unit="batch",
            ):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                if QUANT_TENSOR:
                    bb_outputs = YOLOout(outputs.value, anchors, device, True)
                else:
                    bb_outputs = YOLOout(outputs, anchors, device, True)
                iou = IoU_calc(bb_outputs, labels)
                ctrDist, ratio = AssessBBDiffs(bb_outputs, labels)
                valid_total += labels.size(0)
                valid_miou += iou.sum()
                valid_AP50 += (iou >= 0.5).sum()
                valid_AP75 += (iou >= 0.75).sum()
                valid_ctrdist += ctrDist.sum()
                valid_ratio += ratio.sum()
                # sample images
                if i % (len(valid_loader) // 2) == 0:
                    logger.add_image_with_boxes(
                        "ValidationResults",
                        images[0],
                        getXYminmax(bb_outputs[0], labels[0]),
                        labels=["prediction", "true"],
                    )
            # log accuracy statistics
            logger.add_scalar(
                "ValidationAcc/meanIoU", valid_miou / len(valid_loader), epoch
            )
            logger.add_scalar(
                "ValidationAcc/meanAP50", valid_AP50 / len(valid_loader), epoch
            )
            logger.add_scalar(
                "ValidationAcc/meanAP75", valid_AP75 / len(valid_loader), epoch
            )
            logger.add_scalar(
                "ValidationAcc/CtrDist", valid_ctrdist / len(valid_loader), epoch,
            )
            logger.add_scalar(
                "ValidationAcc/SizesRatio", valid_ratio / len(valid_loader), epoch
            )

    # save network
    os.makedirs("./train_out", exist_ok=True)

    # export network ONNX
    onnx_path = (
        f"./train_out/trained_net_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.onnx"
    )
    exportONNX(net, (1, 3, 640, 640), onnx_path)

    # save network
    net_path = (
        f"./train_out/trained_net_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.pth"
    )
    torch.save(net.state_dict(), net_path)

    # save anchors
    if save_anchors:
        anchors_path = (
            f"./train_out/anchors_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.txt"
        )
        f = open(anchors_path, "a")
        for anchor in anchors:
            f.write(f"{anchor[0]: .8f}, {anchor[1]: .8f}\n")
        f.close()

    return [net, anchors]


# ------------------------------------------------------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    # asses input args
    img_dir = sys.argv[1]
    lbl_dir = sys.argv[2]
    weight_bit_width = int(sys.argv[3])
    act_bit_width = int(sys.argv[4])
    n_anchors = int(sys.argv[5])
    n_epochs = int(sys.argv[6])
    batch_size = int(sys.argv[7])

    anchors, net = train(
        img_dir,
        lbl_dir,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        anchors=False,
        n_anchors=n_anchors,
        n_epochs=n_epochs,
        batch_size=batch_size,
        len_lim=-1,
        loss_fnc="yolo",
    )

