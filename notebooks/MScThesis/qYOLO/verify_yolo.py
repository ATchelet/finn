import os
import argparse
import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay
from notebooks.MScThesis.qYOLO.qyolo import IMG_SHP

GRID_SIZE = np.array((20, 20))
IMG_SHAPE = np.array((640, 640, 3))
COEF = np.array((32, 32, 640, 640))
gx = np.arange(GRID_SIZE[1]).repeat_interleave(GRID_SIZE[0])
gy = np.arange(GRID_SIZE[0]).repeat(GRID_SIZE[1])


def load_data(path, bsize=100):
    # load images
    images = np.load(path + "testset_images.npz")
    # images = ((images / 255.0) - 0.5) * 2.0
    # load labels
    labels = np.array([]).reshape(0, 4)
    with open(path + "testset_labels.txt") as f:
        for line in f.readlines():
            lbl_data = [data.strip() for data in line.split("\t")]
            lbl = np.array(lbl_data).astype(float)
            labels = np.vstack([labels, lbl])
        f.close()
    # split to batches
    n_batches = int(images.shape[0] / bsize)
    images = images[n_batches * bsize].reshpae(
        n_batches, bsize, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]
    )
    labels = labels[n_batches * bsize].reshpae(n_batches, bsize, 4)
    return (images, labels)


def YOLOout(output, n_anchors, anchors):
    # reshape output to [batch, grid_boxes, n_anchors, predictions]
    output = output.reshape(-1, GRID_SIZE.prod(), n_anchors, 5)
    output[..., 4] = 1.0 / (1.0 + np.exp(-output[..., 4]))
    output = output.mean(-2)

    output[..., 0] = (1.0 / (1.0 + np.exp(-output[..., 0]))) + gx
    output[..., 1] = (1.0 / (1.0 + np.exp(-output[..., 1]))) + gy
    output[..., 2] = np.exp(output[..., 2]) * anchors[0]
    output[..., 3] = np.exp(output[..., 3]) * anchors[1]

    # find the most probable grid and box
    amax = output[..., 4].argmax(-1)
    output = output[np.arange(output.shape[0]), amax, :4]

    return output


def IoU_calc(pred_, label_):
    pred = pred_ * COEF
    label = label_ * COEF
    # xmin, ymin, xmax, ymax
    pred_bb = np.vstack(
        [
            np.max(pred[..., 0] - (pred[..., 2] / 2), 0.0),
            np.max(pred[..., 1] - (pred[..., 3] / 2), 0.0),
            np.min(pred[..., 0] + (pred[..., 2] / 2), IMG_SHAPE[1]),
            np.min(pred[..., 1] + (pred[..., 3] / 2), IMG_SHAPE[0]),
        ]
    )
    label_bb = np.vstack(
        [
            np.max(label[..., 0] - (label[..., 2] / 2), 0.0),
            np.max(label[..., 1] - (label[..., 3] / 2), 0.0),
            np.min(label[..., 0] + (label[..., 2] / 2), IMG_SHAPE[1]),
            np.min(label[..., 1] + (label[..., 3] / 2), IMG_SHAPE[0]),
        ]
    )
    inter_bb = np.vstack(
        [
            np.max(label_bb[..., 0], pred_bb[..., 0]),
            np.max(label_bb[..., 1], pred_bb[..., 1]),
            np.min(label_bb[..., 2], pred_bb[..., 2]),
            np.min(label_bb[..., 3], pred_bb[..., 3]),
        ]
    )
    # calculate IoU
    label_area = label[..., 2:4].prod(-1)
    pred_area = pred[..., 2:4].prod(-1)
    inter_area = np.max(inter_bb[..., 2] - inter_bb[..., 0], 0.0) * np.max(
        inter_bb[..., 3] - inter_bb[..., 1], 0.0
    )
    # return IoU
    return inter_area / (label_area + pred_area - inter_area)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate mean IoU and AP50 for FINN-generated Quantized YOLO"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=100
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile",
        help='name of bitfile (i.e. "resizer.bit")',
        default="../bitfile/finn-accel.bit",
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir Testset", default="./Testset",
    )
    parser.add_argument(
        "--n_anchors", help="number of anchors", type=int, required=True,
    )
    parser.add_argument(
        "--anchors",
        help="tuple of average width and hight of bounding boxes (anchors)",
        type=float,
        nargs=2,
        required=True,
    )
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    bitfile = args.bitfile
    platform = args.platform
    dataset_root = args.dataset_root
    n_anchors = args.n_anchors
    anchors = args.anchors

    (test_imgs, test_labels) = load_data(bsize, dataset_root)

    miou = 0.0
    ap50 = 0
    n_batches = test_imgs.shape[0]
    total = n_batches * bsize

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
    )

    n_batches = int(total / bsize)

    test_imgs = test_imgs.reshape(n_batches, bsize, -1)
    test_labels = test_labels.reshape(n_batches, bsize)

    for i in range(n_batches):
        inp = test_imgs[i].astype(np.float32)
        exp = test_labels[i]
        out = driver.execute(inp)
        out = YOLOout(out, n_anchors, anchors)
        iou = IoU_calc(out, exp)
        miou += iou.sum()
        ap50 += (iou >= 0.5).sum()
        print(
            "batch %d / %d : IoU>=.50: %d / %d - mean-IoU: %f"
            % (i + 1, n_batches, ap50, bsize, iou.mean())
        )

    miou /= total
    ap50 /= total
    print("Final IoU>=.50: %f - mean-IoU: %f" % (ap50, miou))
