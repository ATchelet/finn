# imports
import os
import numpy as np
import torch
import torch.utils.tensorboard
from torch.nn import Module, Sequential, BatchNorm2d, MSELoss, BCELoss
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage import io
from kmeans_pytorch import kmeans
from brevitas.nn import (
    QuantIdentity,
    QuantConv2d,
    QuantReLU,
    QuantMaxPool2d,
    QuantSigmoid,
)
from brevitas.inject.defaults import *
from brevitas.core.restrict_val import RestrictValueType
from tqdm import tqdm, trange

# size of output layer
O_SIZE = 5  # bb_x, bb_y, bb_w, bb_h, bb_conf

# bounding boxes grid output shape
GRID_SIZE = torch.tensor([9, 16])  # (y, x)


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


class QTinyYOLOv2(Module):
    def __init__(
        self,
        n_anchors,
        weight_bit_width=8,
        act_bit_width=8,
        quant_tensor=True,
        batch_size=1,
    ):
        super(QTinyYOLOv2, self).__init__()
        self.weight_bit_width = int(np.clip(weight_bit_width, 1, 8))
        self.act_bit_width = int(np.clip(act_bit_width, 1, 8))
        self.n_anchors = n_anchors
        self.batch_size = batch_size

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
                3,
                16,
                3,
                1,
                (2, 2),
                bias=False,
                weight_bit_width=8,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(16),
            QuantReLU(bit_width=8, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (1, 1), return_quant_tensor=quant_tensor),
        )
        self.conv2 = Sequential(
            QuantConv2d(
                16,
                32,
                3,
                1,
                (2, 1),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(32),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 1), return_quant_tensor=quant_tensor),
        )
        self.conv3 = Sequential(
            QuantConv2d(
                32,
                64,
                3,
                1,
                (1, 1),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(64),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 1), return_quant_tensor=quant_tensor),
        )
        self.conv4 = Sequential(
            QuantConv2d(
                64,
                128,
                3,
                1,
                (2, 2),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(128),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 0), return_quant_tensor=quant_tensor),
        )
        self.conv5 = Sequential(
            QuantConv2d(
                128,
                256,
                3,
                1,
                (1, 2),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(256),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 0), return_quant_tensor=quant_tensor),
        )
        self.conv6 = Sequential(
            QuantConv2d(
                256,
                512,
                3,
                1,
                (2, 2),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(512),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
            QuantMaxPool2d(2, 2, (0, 0), return_quant_tensor=quant_tensor),
        )
        self.conv7 = Sequential(
            QuantConv2d(
                512,
                512,
                3,
                1,
                (2, 2),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(512),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
        )
        self.conv8 = Sequential(
            QuantConv2d(
                512,
                512,
                3,
                1,
                (1, 2),
                bias=False,
                weight_bit_width=self.weight_bit_width,
                return_quant_tensor=quant_tensor,
            ),
            BatchNorm2d(512),
            QuantReLU(bit_width=self.act_bit_width, return_quant_tensor=quant_tensor),
        )
        self.conv9 = QuantConv2d(
            512,
            self.n_anchors * O_SIZE,
            1,
            1,
            0,
            bias=False,
            weight_bit_width=8,
            return_quant_tensor=quant_tensor,
        )
        self.sig = QuantSigmoid(bit_width=8, return_quant_tensor=quant_tensor)

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
        x = x.flatten(-2, -1)
        x = x.transpose(-2, -1)
        x = x.view(self.batch_size, GRID_SIZE.prod(), self.n_anchors, O_SIZE)

        return x


def YOLOout(output, anchors, device, findBB):
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


def train(
    img_dir,
    lbl_dir,
    weight_bit_width=1,
    act_bit_width=3,
    anchors=False,
    n_anchors=5,
    n_epochs=100,
    batch_size=32,
    len_lim=-1,
    loss_fnc="yolo",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trainig on: {device}")

    # logger
    logger = torch.utils.tensorboard.SummaryWriter()

    # dataset
    transformers = transforms.Compose([ToTensor(), Normalize()])
    dataset = YOLO_dataset(img_dir, lbl_dir, len_lim=len_lim, transform=transformers)
    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    test_len = data_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # get anchors
    if torch.is_tensor(anchors):
        n_anchors = anchors.size(0)
    else:
        print("Calculating Anchors")
        anchors = (getAnchors(train_loader, n_anchors, device)).to(device)

    # network setup
    net = QTinyYOLOv2(n_anchors, weight_bit_width, act_bit_width)
    net = net.to(device)
    loss_func = YOLOLoss(anchors, device, loss_fnc=loss_fnc)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # train network
    print("Training Start")
    for epoch in trange(n_epochs, desc="epoch", unit="epoch"):
        # train + train loss
        train_loss = 0.0
        test_loss = 0.0
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
            loss = loss_func(outputs.value.float(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # test loss
        with torch.no_grad():
            for i, data in tqdm(
                enumerate(test_loader, 0),
                total=len(test_loader),
                desc="test loss",
                unit="batch",
            ):
                test_images, test_labels = data[0].to(device), data[1].to(device)
                test_outputs = net(test_images)
                t_loss = loss_func(test_outputs.value.float(), test_labels.float())
                test_loss += t_loss.item()
        # log loss statistics
        logger.add_scalar("Loss/train", train_loss / train_len, epoch)
        logger.add_scalar("Loss/test", test_loss / test_len, epoch)

        # train accuracy
        with torch.no_grad():
            train_miou = 0.0
            train_AP50 = 0.0
            train_AP75 = 0.0
            train_total = 0
            for data in tqdm(
                train_loader,
                total=len(train_loader),
                desc="train accuracy",
                unit="batch",
            ):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                iou = IoU_calc(YOLOout(outputs.value, anchors, device, True), labels)
                train_total += labels.size(0)
                train_miou += iou.sum()
                train_AP50 += (iou >= 0.5).sum()
                train_AP75 += (iou >= 0.75).sum()
            # log accuracy statistics
            logger.add_scalar("meanIoU/train", train_miou / train_total, epoch)
            logger.add_scalar("meanAP50/train", train_AP50 / train_total, epoch)
            logger.add_scalar("meanAP75/train", train_AP75 / train_total, epoch)
        # test accuracy
        with torch.no_grad():
            test_miou = 0.0
            test_AP50 = 0.0
            test_AP75 = 0.0
            test_total = 0
            for data in tqdm(
                test_loader, total=len(test_loader), desc="test accuracy", unit="batch"
            ):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                iou = IoU_calc(YOLOout(outputs.value, anchors, device, True), labels)
                test_total += labels.size(0)
                test_miou += iou.sum()
                test_AP50 += (iou >= 0.5).sum()
                test_AP75 += (iou >= 0.75).sum()
            # log accuracy statistics
            logger.add_scalar("meanIoU/test", test_miou / test_total, epoch)
            logger.add_scalar("meanAP50/test", test_AP50 / test_total, epoch)
            logger.add_scalar("meanAP75/test", test_AP75 / test_total, epoch)

    # save network
    os.makedirs("./train_out", exist_ok=True)

    net_path = (
        f"./train_out/trained_net_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.pth"
    )
    torch.save(net.state_dict(), net_path)

    # save anchors
    anchors_path = (
        f"./train_out/anchors_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.txt"
    )
    f = open(anchors_path, "a")
    for anchor in anchors:
        f.write(f"{anchor[0]: .8f}, {anchor[1]: .8f}\n")
    f.close()

    return [net, n_anchors]


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


if __name__ == "__main__":
    img_dir = "./Dataset/images"
    lbl_dir = "./Dataset/labels"
    weight_bit_width = 8
    act_bit_width = 8
    n_anchors = 5
    anchors = torch.tensor(
        [
            [0.0324, 0.0795],
            [0.1175, 0.4245],
            [0.0578, 0.1505],
            [0.0634, 0.2507],
            [0.1811, 0.2212],
        ]
    )
    n_epochs = 10
    batch_size = 1

    net, anchors = train(
        img_dir,
        lbl_dir,
        len_lim=500,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        anchors=anchors,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
