import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from plotting.CelebADrawer import BboxDrawer

df_landmarks = pd.read_csv("Anno/list_landmarks_resized_celeba.txt", delim_whitespace=True)
print(df_landmarks.columns)
print(df_landmarks[[df_landmarks.columns[0], df_landmarks.columns[1]]].iloc[0])
drawer = BboxDrawer()
model_size = (218, 218)


def format_landmarks(df_landmarks):
    landmark_dict = {}
    row = 0
    print(df_landmarks.columns)
    for i in tqdm(df_landmarks["Name"]):
        data = list(df_landmarks.iloc[row].values)
        landmark_dict[i] = [
            (data[1],data[2]),
            (data[3],data[4]),
            (data[5],data[6]),
            (data[7],data[8]),
            (data[9],data[10])
        ]
        row+=1
    return landmark_dict




def process_image(path, file):
    image = Image.open(path + f"{file}.jpg").convert('RGB')
    return image.size

feat_dicts = format_landmarks(df_landmarks)

new_writer = open("Anno/list_landmarks_standardized_celeba.txt","w")
new_writer.write("image_id lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n")
curr_id = 1
for imgID in tqdm(feat_dicts.keys()):
    #print(feat_dicts[imgID])
    #drawer.process_feats(Image.open(f"img_celeba/{imgID}.jpg").convert("RGB"),feat_dicts[imgID])
    image_size = process_image("img_celeba_bboxed2/",imgID.split(".")[0])
    landmarks_processed = drawer.resize_feats(feat_dicts[imgID],image_size,(218,218))
    new_line = imgID + ".jpg"
    for l_x,l_y in landmarks_processed:
        new_line += f" {l_x} {l_y}"
    new_line += "\n"
    new_writer.write(new_line)
    curr_id += 1
    #drawer.process_feats(image,process_landmarks(bbox_dicts,feat_dicts[imgID]))

new_writer.close()