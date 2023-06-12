import os
from PIL import Image
import numpy as np

def bbox_resizer(x, y, w, h, img_size):
    x_new = 178*x/img_size[0]
    y_new = 218*y/img_size[1]
    w_new =178*w/img_size[0]
    h_new = 218*h/img_size[1]
    return np.float32(x_new),np.float32(y_new), np.float32(w_new),np.float32(h_new)


bb_path = "Anno/list_bbox_celeba.txt"
img_folder = "img_celeba"

with open(bb_path, "r") as f:
    next(f)
    lines = f.readlines()
    for line in lines:
        parts = line.split()
        img_id = parts[0]
        x,y,w,h = map(int, parts[1:])

        img_path = os.path.join(img_folder, img_id)
        img = Image.open(img_path)
        img_size = img.size

        x_s, y_s, w_s, h_s = bbox_resizer(x,y,w,h,img_size)

        with open("Anno/resized_bbox", "a") as f_new:
            f_new.write(f"{img_id} {x_s} {y_s} {w_s} {h_s}\n")

