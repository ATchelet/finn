# imports
import numpy as np
import tqdm
import torch
from kmeans_pytorch import kmeans
from torch.nn import Module, MSELoss, BCELoss

# qYOLO imports
from qYOLO.module import *
from qYOLO.dataset import *


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

    def forward(self, pred_, label):
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
            # return loss
            return (
                coor_l_obj * self.l_coor_obj
                + conf_l_obj * self.l_conf_obj
                + conf_l_noobj * self.l_conf_noobj
            )

        else:
            print("User error: Unsupported loss function")
            exit()


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


# TODO Might need to implement it in C/C++ to add it to the board software implementation
def YoloOutput(pred, anchors, w=16, h=9):
    n = anchors.shape[0]
    bb_conf, bb_idx = torch.max(pred[..., -1], -1)
    g_idx = bb_conf.argmax(-1)
    output = pred[..., g_idx, bb_idx[g_idx], :4]
    cx = (g_idx // h) / w
    cy = (g_idx % h) / h
    pw, ph = anchors[bb_idx[g_idx]]
    bx = output[0].sigmoid() + cx
    by = output[1].sigmoid() + cy
    bw = output[2].exp() * pw
    bh = output[3].exp() * ph
    return [bx, by, bw, bh]
