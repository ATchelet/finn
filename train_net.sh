#!/bin/bash
start_date=$(date)

# set personal environment
source /srv/cdl-eml/User/atchelet/pipenv_at/bin/activate
echo "pip environment activated"

export IMG_DIR=/srv/cdl-eml/User/atchelet/Dataset/images
export LBL_DIR=/srv/cdl-eml/User/atchelet/Dataset/labels

# script
# EDA01
cd /home/atchelet/git/finn/
# EDA02
# cd /home/atchelet/git/finn_at/finn_at/finn_at
echo "Start training"
# train_net.py IMAGES_DIR LABELS_DIR WEIGHT_BIT ACTIVATION_BIT NUM_ANCHORS NUM_EPOCHS BATCH_SIZE
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 0 0 5 40 100
sleep 5
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 3 3 5 80 100
sleep 5
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 4 4 5 80 100
sleep 5
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 8 8 5 80 100
sleep 5
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 1 3 5 80 100
sleep 5
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 2 4 5 80 100
sleep 5
time python3 -m torch.distributed.launch --nproc_per_node 1 ./train_net.py $IMG_DIR $LBL_DIR 4 8 5 80 100
echo "finished!"

end_date=$(date)

echo "Start: $start_date"
echo "End: $end_date"

#Command for executing the task
#ts -L qYOLO_train /home/atchelet/git/finn_at/finn_at/finn_at/train_net.sh
