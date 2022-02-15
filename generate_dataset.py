import os
import xmltodict
import numpy as np
from skimage import io
from tqdm import tqdm

data_dir = "/home/atchelet/Downloads/data_training/"
out_dir = "/home/atchelet/Dataset"
os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

w_px = h_px = 40

for path, subdirs, files in os.walk(data_dir):
    dir = os.path.basename(path)
    i = 0
    for file in tqdm(sorted(files), desc=dir):
        if file.endswith(".jpg"):
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            img[140:500] = io.imread(os.path.join(path, file))
            io.imsave(os.path.join(out_dir, "images", f"{dir}_{i:07d}.jpg"), img)
        if file.endswith(".xml"):
            with open(os.path.join(path, file)) as fd:
                doc = xmltodict.parse(fd.read())
                width = int(doc["annotation"]["size"]["width"])
                height = int(doc["annotation"]["size"]["height"]) + 280
                xmin = int(doc["annotation"]["object"]["bndbox"]["xmin"])
                ymin = int(doc["annotation"]["object"]["bndbox"]["ymin"])
                xmax = int(doc["annotation"]["object"]["bndbox"]["xmax"])
                ymax = int(doc["annotation"]["object"]["bndbox"]["ymax"])
                b_x = ((xmax + xmin) / 2) / width
                b_y = ((ymax + ymin + 280) / 2) / height
                b_w = (xmax - xmin) / width
                b_h = (ymax - ymin) / height
                f = open(os.path.join(out_dir, "labels", f"{dir}_{i:07d}.txt"), "a")
                f.write(f"{dir}_{i:07d}\n{b_x:.8f}\t{b_y:.8f}\t{b_w:.8f}\t{b_h:.8f}\n")
                f.close()
                i += 1
