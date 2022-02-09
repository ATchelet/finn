# imports
import torch
import torch.utils.tensorboard
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange

# import qYOLO
from qYOLO.module import *
from qYOLO.dataset import *
from qYOLO.util import *


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
    loss_fnc='yolo',
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Trainig on: {device}")

    # logger
    logger = torch.utils.tensorboard.SummaryWriter()

    # dataset
    transformers = transforms.Compose([ToTensor(), Normalize()])
    dataset = YOLO_dataset(img_dir,
                           lbl_dir,
                           len_lim=len_lim,
                           transform=transformers)
    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    test_len = data_len - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

    # get anchors
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
        for i, data in tqdm(enumerate(train_loader, 0),
                            total=len(train_loader),
                            desc="train loss",
                            unit="batch"):
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
            for i, data in tqdm(enumerate(test_loader, 0),
                                total=len(test_loader),
                                desc="test loss",
                                unit="batch"):
                test_images, test_labels = data[0].to(device), data[1].to(
                    device)
                test_outputs = net(test_images)
                t_loss = loss_func(test_outputs.value.float(),
                                   test_labels.float())
                test_loss += t_loss.item()
        # log loss statistics
        logger.add_scalar('Loss/train', train_loss / train_len, epoch)
        logger.add_scalar('Loss/test', test_loss / test_len, epoch)

        # train accuracy
        with torch.no_grad():
            train_miou = 0.0
            train_AP50 = 0.0
            train_AP75 = 0.0
            train_total = 0
            for data in tqdm(train_loader,
                             total=len(train_loader),
                             desc="train accuracy",
                             unit="batch"):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                iou = IoU_calc(YOLOout(outputs.value, anchors, device, True),
                               labels)
                train_total += labels.size(0)
                train_miou += iou.sum()
                train_AP50 += (iou >= .5).sum()
                train_AP75 += (iou >= .75).sum()
            # log accuracy statistics
            logger.add_scalar('meanIoU/train', train_miou / train_total, epoch)
            logger.add_scalar('meanAP50/train', train_AP50 / train_total,
                              epoch)
            logger.add_scalar('meanAP75/train', train_AP75 / train_total,
                              epoch)
        # test accuracy
        with torch.no_grad():
            test_miou = 0.0
            test_AP50 = 0.0
            test_AP75 = 0.0
            test_total = 0
            for data in tqdm(test_loader,
                             total=len(test_loader),
                             desc="test accuracy",
                             unit="batch"):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                iou = IoU_calc(YOLOout(outputs.value, anchors, device, True),
                               labels)
                test_total += labels.size(0)
                test_miou += iou.sum()
                test_AP50 += (iou >= .5).sum()
                test_AP75 += (iou >= .75).sum()
            # log accuracy statistics
            logger.add_scalar('meanIoU/test', test_miou / test_total, epoch)
            logger.add_scalar('meanAP50/test', test_AP50 / test_total, epoch)
            logger.add_scalar('meanAP75/test', test_AP75 / test_total, epoch)

    # save network
    net_path = f"./train_out/trained_net_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.pth"
    torch.save(net.state_dict(), net_path)

    # save anchors
    anchors_path = f"./train_out/anchors_W{weight_bit_width}A{act_bit_width}_a{n_anchors}.txt"
    f = open(anchors_path, "a")
    for anchor in range(anchors):
        f.write(f"{anchor[0]}, {anchor[1]}\n")
    f.close()

    return [net, n_anchors]
