# general use libraries
import os
from tqdm import tqdm, trange
import numpy as np
from skimage import io
import torch.utils.tensorboard
from kmeans_pytorch import kmeans

# Brevitas ad PyTorch libraries
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torchvision import transforms
from torch.nn import (
    Module,
    Sequential,
    BatchNorm2d,
    Conv2d,
    ReLU,
    MaxPool2d,
    MSELoss,
    BCELoss,
    DataParallel,
)
from brevitas.nn import (
    QuantIdentity,
    QuantConv2d,
    QuantReLU,
    QuantMaxPool2d,
)
from brevitas.inject.defaults import *
from brevitas.core.restrict_val import RestrictValueType
from brevitas.onnx import export_brevitas_onnx as exportONNX

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

#############################################
#               Configurations              #
#############################################


# size of output layer
# bb_x, bb_y, bb_w, bb_h, bb_conf

# image, input and bounding boxes grid output shapes
IMG_SHP = torch.tensor([360, 640])  # (h, w)
INPUT_SHP = torch.tensor([640, 640])  # (h, w)
GRID_SIZE = torch.tensor([20, 20])  # (y, x)
RATIO = INPUT_SHP / GRID_SIZE  # ratio between input size and grid size
LBL_COEF = torch.tensor(
    [GRID_SIZE[1], GRID_SIZE[0], INPUT_SHP[1], INPUT_SHP[0]]
)  # x, y, w, h

# Flag to use QuantTensor or not
QUANT_TENSOR = False
# Flag to use anchors averaging or not
ANCHOR_AVE = True

# global grid offset values placeholder
gx = torch.tensor(0.0)
gy = torch.tensor(0.0)

#############################################
#               Dataset                     #
#############################################


class YOLO_dataset(Dataset):
    def __init__(
        self, img_dir, lbl_dir, len_lim=-1, transform=None
    ):
        self.img_dir = img_dir
        self.imgs = sorted(os.listdir(self.img_dir))[:len_lim]
        self.lbl_dir = lbl_dir
        self.lbls = sorted(os.listdir(self.lbl_dir))[:len_lim]
        self.transform = transform

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
            out_channels=self.n_anchors * 5,
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
    def __init__(self, n_anchors):
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
            out_channels=self.n_anchors * 5,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
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

    # # set the outputs of the padded area to the values the loss sends them
    # pad_cut = int(
    #     (GRID_SIZE[0] - np.ceil(GRID_SIZE[0] * IMG_SHP[0] / IMG_SHP[1])) / 2.0
    # )
    # output[:, :, :pad_cut] = 0.0
    # output[:, :, -pad_cut:] = 0.0

    # reshape output to [batch, grid_boxes, anchors, predictions]
    output = output.flatten(-2, -1)
    output = output.transpose(-2, -1)
    output = output.view(output.size(0), GRID_SIZE.prod(), anchors.size(0), 5)
    output[..., 4] = torch.sigmoid(output[..., 4])
    if ANCHOR_AVE:
        output = output.mean(-2)

    gx_ = gx.to(device)
    gy_ = gy.to(device)
    output[..., 0] = torch.sigmoid(output[..., 0]) + gx_
    output[..., 1] = torch.sigmoid(output[..., 1]) + gy_
    if ANCHOR_AVE:
        output[..., 2] = torch.exp(output[..., 2]) * anchors[0, 0] * INPUT_SHP[0]
        output[..., 3] = torch.exp(output[..., 3]) * anchors[0, 1] * INPUT_SHP[1]
    else:
        output[..., 2] = torch.exp(output[..., 2]) * anchors[:, 0] * INPUT_SHP[0]
        output[..., 3] = torch.exp(output[..., 3]) * anchors[:, 1] * INPUT_SHP[1]
    # find the most probable grid and box
    if findBB:
        if ANCHOR_AVE:
            amax = output[..., 4].argmax(-1)
            output = output[torch.arange(output.size(0)), amax, :4]
        else:
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
        l_coor_obj=2.0,
        l_coor_noobj=1.0,
        l_conf_obj=5.0,
        l_conf_noobj=1.0,
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

    def forward(self, pred_, label, is_training=True):
        global logger, gx, gy
        # locate bounding box location and
        idx_x = (label[:, 0] * GRID_SIZE[1]).floor()
        idx_y = (label[:, 1] * GRID_SIZE[0]).floor()
        idx = (idx_x * GRID_SIZE[0] + idx_y).type(torch.int64)
        # convert predictions to label style
        pred = YOLOout(pred_, self.anchors, self.device, False)
        # find closest anchor
        if not ANCHOR_AVE:
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
        anchors_grid = torch.zeros_like(pred[..., :4])
        anchors_grid[..., 0] = gx + 0.5
        anchors_grid[..., 1] = gy + 0.5
        if ANCHOR_AVE:
            anchors_grid[..., 2:4] = self.anchors[0] * INPUT_SHP
        else:
            anchors_grid[..., 2:4] = self.anchors * INPUT_SHP
        # mask of obj and noobj
        obj_mask = (torch.zeros_like(pred)).type(torch.bool)
        for i in range(obj_mask.size(0)):
            if ANCHOR_AVE:
                obj_mask[i, idx[i], :] = True
            else:
                obj_mask[i, idx[i], anchor_mask[i], :] = True
        noobj_mask = ~obj_mask
        # prepare loss calculation parts
        pred_obj = pred[obj_mask].view(pred.shape[0], 5)
        pred_noobj = pred[noobj_mask].view(pred.shape[0], -1, 5)
        anchors_grid = anchors_grid[noobj_mask[..., :4]].view(pred.shape[0], -1, 4)
        pred_obj = pred_obj.to(self.device)
        pred_noobj = pred_noobj.to(self.device)
        anchors_grid = anchors_grid.to(self.device)
        iou = IoU_calc(pred_obj, label)

        if self.loss_fnc == "yolo":
            # coordination loss
            coor_l_obj = self.mse(
                pred_obj[:, :2], label[:, :2] * LBL_COEF[:2]
            ) + self.mse(pred_obj[:, 2:4].sqrt(), (label[:, 2:] * LBL_COEF[2:]).sqrt())
            coor_l_noobj = self.mse(pred_noobj[..., :4], anchors_grid)
            # confidence loss
            # conf_l_obj = self.mse(
            #     pred_obj[:, 4], torch.ones_like(pred_obj[:, 4], device=self.device)
            # )
            conf_l_obj = self.mse(pred_obj[:, 4], iou)
            conf_l_noobj = self.mse(
                pred_noobj[..., 4],
                torch.zeros_like(pred_noobj[..., 4], device=self.device),
            )
            # log loss parts
            if is_training:
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
        elif self.loss_fnc == "yolov2":
            # coordination loss
            coor_l_obj = self.mse(pred_obj[:, :4], label * LBL_COEF)
            coor_l_noobj = self.mse(pred_noobj[..., :4], anchors_grid)
            # confidence loss
            conf_l_obj = self.mse(pred_obj[:, 4], iou)
            conf_l_noobj = self.mse(
                pred_noobj[..., 4],
                torch.zeros_like(pred_noobj[..., 4], device=self.device),
            )
            # log loss parts
            if is_training:
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

        elif is_training == "yolov3":
            # coordination loss
            coor_l_obj = self.bce(
                pred_obj[:, :2], label[:, :2] * LBL_COEF[:2]
            ) + self.mse(pred_obj[:, 2:4], label[:, 2:] * LBL_COEF[2:])
            # confidence loss
            conf_l_obj = self.bce(
                pred_obj[:, 4], torch.ones_like(pred_obj[:, 4], device=self.device)
            )
            conf_l_noobj = self.bce(
                pred_noobj[..., 4],
                torch.zeros_like(pred_noobj[..., 4], device=self.device),
            )
            # log loss parts
            if self.training:
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
    if ANCHOR_AVE:
        anchors = datapoints.mean(0, True).repeat(n_anchors, 1)
    else:
        kmean_idx, anchors = kmeans(
            X=datapoints, num_clusters=n_anchors, distance="euclidean", device=device
        )
    print(f"Anchors for k={n_anchors}:")
    [print(f"[{anchor[0]: .8f}, {anchor[1]: .8f}]") for anchor in anchors]
    return anchors


def IoU_calc(pred, label_):
    label = label_ * LBL_COEF
    # xmin, ymin, xmax, ymax
    label_bb = torch.stack(
        [
            torch.max(
                label[..., 0] * RATIO[1] - (label[..., 2] / 2), torch.tensor(0.0)
            ),
            torch.max(
                label[..., 1] * RATIO[0] - (label[..., 3] / 2), torch.tensor(0.0)
            ),
            torch.min(label[..., 0] * RATIO[1] + (label[..., 2] / 2), INPUT_SHP[1]),
            torch.min(label[..., 1] * RATIO[0] + (label[..., 3] / 2), INPUT_SHP[0]),
        ],
        1,
    )
    pred_bb = torch.stack(
        [
            torch.max(pred[..., 0] * RATIO[1] - (pred[..., 2] / 2), torch.tensor(0.0)),
            torch.max(pred[..., 1] * RATIO[0] - (pred[..., 3] / 2), torch.tensor(0.0)),
            torch.min(pred[..., 0] * RATIO[1] + (pred[..., 2] / 2), INPUT_SHP[1]),
            torch.min(pred[..., 1] * RATIO[0] + (pred[..., 3] / 2), INPUT_SHP[0]),
        ],
        1,
    )
    inter_bb = torch.stack(
        [
            torch.max(label_bb[..., 0], pred_bb[..., 0]),
            torch.max(label_bb[..., 1], pred_bb[..., 1]),
            torch.min(label_bb[..., 2], pred_bb[..., 2]),
            torch.min(label_bb[..., 3], pred_bb[..., 3]),
        ],
        1,
    )
    # calculate IoU
    label_area = label[:, 2:4].prod(-1)
    pred_area = pred[:, 2:4].prod(-1)
    inter_area = torch.max(
        inter_bb[:, 2] - inter_bb[:, 0], torch.tensor(0.0)
    ) * torch.max(inter_bb[:, 3] - inter_bb[:, 1], torch.tensor(0.0))
    # return IoU
    return inter_area / (label_area + pred_area - inter_area)


def AssessBBDiffs(pred, label):
    # calculate Centres Distance
    CentDist = (
        (((pred[:, 0] * RATIO[1]) - (label[:, 0]) * INPUT_SHP[1])) ** 2.0
        + (((pred[:, 1] * RATIO[0]) - (label[:, 1]) * INPUT_SHP[0])) ** 2.0
    ).sqrt()
    # calculate Size Ratio - pred/true
    SizeRatio = pred[:, 2:4].prod(-1) / (label[:, 2:4].prod(-1) * INPUT_SHP.prod())
    # return Centres Distance, and Size Ratio
    return [CentDist, SizeRatio]


def getXYminmax(pred_, label_, device):
    label = label_ * LBL_COEF
    label = torch.stack(
        [
            torch.max(
                label[..., 0] * RATIO[1] - (label[..., 2] / 2), torch.tensor(0.0)
            ).round(),
            torch.max(
                label[..., 1] * RATIO[0] - (label[..., 3] / 2), torch.tensor(0.0)
            ).round(),
            torch.min(
                label[..., 0] * RATIO[1] + (label[..., 2] / 2), INPUT_SHP[1]
            ).round(),
            torch.min(
                label[..., 1] * RATIO[0] + (label[..., 3] / 2), INPUT_SHP[0]
            ).round(),
        ],
        1,
    ).view(1, 4)
    pred = torch.stack(
        [
            torch.max(
                pred_[..., 0] * RATIO[1] - (pred_[..., 2] / 2), torch.tensor(0.0)
            ).round(),
            torch.max(
                pred_[..., 1] * RATIO[0] - (pred_[..., 3] / 2), torch.tensor(0.0)
            ).round(),
            torch.min(
                pred_[..., 0] * RATIO[1] + (pred_[..., 2] / 2), INPUT_SHP[1]
            ).round(),
            torch.min(
                pred_[..., 1] * RATIO[0] + (pred_[..., 3] / 2), INPUT_SHP[0]
            ).round(),
        ],
        1,
    ).view(1, 4)
    iou_print_bb = torch.tensor([0, 0, 90, 15], device=device).view(1,4)
    return torch.stack([pred, label, iou_print_bb], 0).view(3, 4)


def set_grids_mats(n_anchors):
    global gx, gy

    if ANCHOR_AVE:
        gx = torch.arange(GRID_SIZE[1]).repeat_interleave(GRID_SIZE[0])

        gy = torch.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])
    else:
        gx = (
            (torch.arange(GRID_SIZE[1]).repeat_interleave(GRID_SIZE[0] * n_anchors))
        ).view(GRID_SIZE.prod(), n_anchors)

        gy = (
            (torch.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])).repeat_interleave(
                n_anchors
            )
        ).view(GRID_SIZE.prod(), n_anchors)


def exportTestset(testset, path):
    from torchvision.transforms import ToPILImage

    images = []
    os.makedirs(path, exist_ok=True)
    f = open(os.path.join(path, "testset_labels.txt"), "w")
    for i, [img, lbl] in tqdm(enumerate(testset),
                total=len(testset),
                desc="testset export",
                unit="images",):
        img = np.array(ToPILImage()((img + 1.0) / 2.0))
        # io.imsave(os.path.join(path, "images", f"{i:09d}.jpg"), img)
        images.append(img)
        f.write(f"{lbl[0]:.8f}\t{lbl[1]:.8f}\t{lbl[2]:.8f}\t{lbl[3]:.8f}\n")
    f.close()
    np.savez(os.path.join(path, "testset_images.npz"), images)


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
    lr_start=1 * 10 ** -4,
    lr_end=1 * 10 ** -6,
    lr_max=1 * 10 ** -3,
    len_lim=-1,
    img_samples=6,
    loss_fnc="yolo",
    schdlr="OneCycleLR",
    quantized=True,
    export_testset=False,
):
    if not quantized:
        global QUANT_TENSOR
        QUANT_TENSOR = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # send macros to device
    global INPUT_SHP, LBL_COEF
    INPUT_SHP = INPUT_SHP.to(device)
    LBL_COEF = LBL_COEF.to(device)

    if torch.is_tensor(anchors):
        n_anchors = anchors.size(0)
    set_grids_mats(n_anchors)

    print(f"Trainig on: {device}")
    if quantized:
        print(
            f"Network Configurations: \n\tQuantized - W{weight_bit_width}A{act_bit_width} with {n_anchors} anchors"
        )
    else:
        print(f"Network Configurations: \n\tPytorch - with {n_anchors} anchors")
    print(f"\tEpochs: {n_epochs}, Batch size: {batch_size}")

    # logger
    global logger
    if quantized:
        logger = torch.utils.tensorboard.SummaryWriter(
            comment=f"W{weight_bit_width}A{act_bit_width}a{n_anchors}"
        )
    else:
        logger = torch.utils.tensorboard.SummaryWriter(comment=f"PyTorch_a{n_anchors}")

    # dataset
    transformers = transforms.Compose([ToTensor(), Normalize()])
    dataset = YOLO_dataset(img_dir, lbl_dir, len_lim=len_lim, transform=transformers)
    # split dataset to train:valid:test - 60:20:20 ratio
    # [Train - Test - Train - Valid - Train ...]
    idx = np.arange(len(dataset), dtype=int)
    test_idx = idx[1::5].tolist()
    valid_idx = idx[3::5].tolist()
    train_idx = np.delete(idx, np.append(test_idx, valid_idx))
    train_len = len(train_idx)
    valid_len = len(valid_idx)
    test_len = len(test_idx)
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set = Subset(dataset, test_idx)
    if export_testset:
        exportTestset(test_set, "./Testset/")
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
    if torch.is_tensor(anchors):
        anchors = anchors.to(device)
    else:
        print("Calculating Anchors")
        anchors = (getAnchors(train_loader, n_anchors, device)).to(device)

    # network setup
    if quantized:
        net = QTinyYOLOv2(n_anchors, weight_bit_width, act_bit_width)
    else:
        net = TinyYOLOv2(n_anchors)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)
    net = net.to(device)
    loss_func = YOLOLoss(anchors, device, loss_fnc=loss_fnc)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=1e-4)
    if schdlr == "StepLR":
        scheduler = StepLR(
            optimizer, step_size=1, gamma=(lr_end / lr_start) ** (1 / n_epochs)
        )
    elif schdlr == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr_max,
            total_steps=n_epochs * len(train_loader),
            anneal_strategy="cos",
            pct_start=0.2,
            div_factor=lr_max / lr_start,
            final_div_factor=lr_start / lr_end,
        )

    # train network
    print("Training Start")
    for epoch in trange(n_epochs, desc="epoch", unit="epoch"):
        # train
        net.train()
        train_total = 0
        train_loss = 0.0
        train_miou = 0.0
        train_AP50 = 0.0
        train_AP75 = 0.0
        train_ctrdist = 0.0
        train_ratio = 0.0
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
                bb_outputs = YOLOout(outputs.value, anchors, device, True)
            else:
                loss = loss_func(outputs.float(), labels.float())
                bb_outputs = YOLOout(outputs, anchors, device, True)
            loss.backward()
            optimizer.step()
            iou = IoU_calc(bb_outputs, labels)
            ctrDist, ratio = AssessBBDiffs(bb_outputs, labels)
            train_total += labels.size(0)
            train_loss += loss.item()
            logger.add_scalar(
                "Loss/train_continuous", loss.item(), i + epoch * len(train_loader)
            )
            logger.add_scalar(
                "LossParts/learning_rate",
                scheduler.get_last_lr()[0],
                i + epoch * len(train_loader),
            )
            train_miou += iou.sum()
            train_AP50 += (iou >= 0.5).sum()
            train_AP75 += (iou >= 0.75).sum()
            train_ctrdist += ctrDist.sum()
            train_ratio += ratio.sum()
            if schdlr == "OneCycleLR":
                scheduler.step()
        if schdlr == "StepLR":
            scheduler.step()
        # log loss statistics
        logger.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
        # log accuracy statistics
        logger.add_scalar("TrainingAcc/meanIoU", train_miou / train_len, epoch)
        logger.add_scalar("TrainingAcc/meanAP50", train_AP50 / train_len, epoch)
        logger.add_scalar("TrainingAcc/meanAP75", train_AP75 / train_len, epoch)
        logger.add_scalar(
            "TrainingAcc/CtrDist", train_ctrdist / train_len, epoch,
        )
        logger.add_scalar("TrainingAcc/SizesRatio", train_ratio / train_len, epoch)

        # validation
        net.eval()
        valid_total = 0
        valid_loss = 0.0
        valid_miou = 0.0
        valid_AP50 = 0.0
        valid_AP75 = 0.0
        valid_ctrdist = 0.0
        valid_ratio = 0.0
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
                        valid_outputs.value.float(), valid_labels.float(), False
                    )
                    bb_outputs = YOLOout(valid_outputs.value, anchors, device, True)
                else:
                    t_loss = loss_func(
                        valid_outputs.float(), valid_labels.float(), False
                    )
                    bb_outputs = YOLOout(valid_outputs, anchors, device, True)
                iou = IoU_calc(bb_outputs, valid_labels)
                ctrDist, ratio = AssessBBDiffs(bb_outputs, valid_labels)
                valid_total += valid_labels.size(0)
                valid_loss += t_loss.item()
                valid_miou += iou.sum()
                valid_AP50 += (iou >= 0.5).sum()
                valid_AP75 += (iou >= 0.75).sum()
                valid_ctrdist += ctrDist.sum()
                valid_ratio += ratio.sum()
        # log loss statistics
        logger.add_scalar("Loss/valid", valid_loss / len(valid_loader), epoch)
        # log accuracy statistics
        logger.add_scalar("ValidationAcc/meanIoU", valid_miou / valid_len, epoch)
        logger.add_scalar("ValidationAcc/meanAP50", valid_AP50 / valid_len, epoch)
        logger.add_scalar("ValidationAcc/meanAP75", valid_AP75 / valid_len, epoch)
        logger.add_scalar(
            "ValidationAcc/CtrDist", valid_ctrdist / valid_len, epoch,
        )
        logger.add_scalar("ValidationAcc/SizesRatio", valid_ratio / valid_len, epoch)

        # sample images bb for logger
        net.eval()
        with torch.no_grad():
            # sample train images
            if train_len <= img_samples:
                step = 1
            else:
                step = int(np.ceil(train_len / img_samples))
            train_smp_set = Subset(train_set, np.arange(train_len, step=step))
            for n, [train_smp_img, train_smp_lbl] in enumerate(train_smp_set):
                train_smp_img = (train_smp_img.unsqueeze(0)).to(device)
                train_smp_lbl = (train_smp_lbl.unsqueeze(0)).to(device)
                train_smp_out = net(train_smp_img)
                if QUANT_TENSOR:
                    train_smp_bbout = YOLOout(
                        train_smp_out.value, anchors, device, True
                    )
                else:
                    train_smp_bbout = YOLOout(train_smp_out, anchors, device, True)
                iou = (IoU_calc(train_smp_bbout, train_smp_lbl).squeeze()).cpu().numpy()
                logger.add_image_with_boxes(
                    f"TrainingResults/img_{n}",
                    (train_smp_img + 1.0) / 2.0,
                    getXYminmax(train_smp_bbout, train_smp_lbl, device),
                    labels=["prediction", "true",  f"iou: {iou:.8f}"],
                    global_step=epoch,
                    dataformats="NCHW",
                )
            # sample validation images
            if valid_len <= img_samples:
                step = 1
            else:
                step = int(np.ceil(valid_len / img_samples))
            valid_smp_set = Subset(valid_set, np.arange(valid_len, step=step))
            for n, [valid_smp_img, valid_smp_lbl] in enumerate(valid_smp_set):
                valid_smp_img = (valid_smp_img.unsqueeze(0)).to(device)
                valid_smp_lbl = (valid_smp_lbl.unsqueeze(0)).to(device)
                valid_smp_out = net(valid_smp_img)
                if QUANT_TENSOR:
                    valid_smp_bbout = YOLOout(
                        valid_smp_out.value, anchors, device, True
                    )
                else:
                    valid_smp_bbout = YOLOout(valid_smp_out, anchors, device, True)
                iou = (IoU_calc(valid_smp_bbout, valid_smp_lbl).squeeze()).cpu().numpy()
                logger.add_image_with_boxes(
                    f"ValidationResults/img_{n}",
                    (valid_smp_img + 1.0) / 2.0,
                    getXYminmax(valid_smp_bbout, valid_smp_lbl, device),
                    labels=["prediction", "true", f"iou: {iou:.8f}"],
                    global_step=epoch,
                    dataformats="NCHW",
                )

    # save network
    os.makedirs("./train_out", exist_ok=True)

    # export network ONNX
    if quantized:
        net.eval()
        onnx_path = f"./train_out/trained_net_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.onnx"
        exportONNX(net, (1, 3, INPUT_SHP[0], INPUT_SHP[1]), onnx_path)

    # save network
    if quantized:
        net_path = f"./train_out/trained_net_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.pth"
        torch.save(net.state_dict(), net_path)
    else:
        net_path = f"./train_out/trained_net_pytorch_a{n_anchors}.pth"
        torch.save(net.state_dict(), net_path)

    # save anchors
    mean = "mean_" if ANCHOR_AVE else ""
    if len_lim == -1:
        anchors_path = f"./train_out/{n_anchors}_{mean}anchors.txt"
    else:
        anchors_path = f"./train_out/{n_anchors}_{mean}anchors_first_{len(dataset)}.txt"
    f = open(anchors_path, "w")
    for anchor in anchors:
        f.write(f"{anchor[0]: .8f}, {anchor[1]: .8f}\n")
    f.close()

    if device=="cuda":
        torch.cuda.empty_cache()

    return [net, anchors]


# ------------------------------------------------------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":

    img_dir = "./Dataset/images"
    lbl_dir = "./Dataset/labels"
    n_anchors = 5
    # anchors = torch.tensor([[0.17775965, 0.12690470],
    #                         [0.11733948, 0.24617620],
    #                         [0.05872642, 0.08477669],
    #                         [0.03005564, 0.04518913],
    #                         [0.06502857, 0.14770794]])
    anchors = torch.tensor(
        [
            [ 0.05757873,  0.08779122],
            [ 0.05757873,  0.08779122],
            [ 0.05757873,  0.08779122],
            [ 0.05757873,  0.08779122],
            [ 0.05757873,  0.08779122],
        ]
    )
    n_epochs = 10
    batch_size = 4
    images_to_log = 3
    dataset_len = 500

    net, anchors = train(
        img_dir,
        lbl_dir,
        len_lim=dataset_len,
        n_epochs=n_epochs,
        n_anchors=n_anchors,
        batch_size=batch_size,
        img_samples=images_to_log,
        quantized=False,
        loss_fnc="yolo",
        export_testset=True,
    )

    for bits in range(3, 9):
        train(
            img_dir,
            lbl_dir,
            len_lim=dataset_len,
            weight_bit_width=int(bits // 2),
            act_bit_width=bits,
            n_epochs=n_epochs,
            anchors=anchors,
            batch_size=batch_size,
            img_samples=images_to_log,
            quantized=True,
            loss_fnc="yolo",
        )
        train(
            img_dir,
            lbl_dir,
            len_lim=dataset_len,
            weight_bit_width=bits,
            act_bit_width=bits,
            n_epochs=n_epochs,
            anchors=anchors,
            batch_size=batch_size,
            img_samples=images_to_log,
            quantized=True,
            loss_fnc="yolo",
        )
