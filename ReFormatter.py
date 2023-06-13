import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from plotting.CelebADrawer import BboxFeatDrawer

df_landmarks = pd.read_csv("Anno/existing/list_landmarks_celeba.txt", delim_whitespace=True)
print(df_landmarks.columns)
print(df_landmarks[[df_landmarks.columns[0], df_landmarks.columns[1]]].iloc[0])
model = torch.load("BBOX_MODELS/modelEp9.pt")
model.eval()
drawer = BboxFeatDrawer()
model_size = (218, 218)


def format_landmarks(df_landmarks):
    landmark_dict = {}
    helper = lambda i, j, row: tuple(df_landmarks[[df_landmarks.columns[i],
                                                   df_landmarks.columns[j]]].iloc[row])
    for i in tqdm(range(1,len(df_landmarks))):
        landmark_dict[f"{'0' * (6 - len(str(i)))}{i}"] = [
            helper(0, 1, i-1),
            helper(2, 3, i-1),
            helper(4, 5, i-1),
            helper(6, 7, i-1),
            helper(8, 9, i-1)
        ]
    print(landmark_dict.keys())
    return landmark_dict


def process_landmarks(new_dim, landmarks):
    landmarks_new = []
    for l_x, l_y in landmarks:
        if l_x - new_dim["x_1"] <0 or l_x - new_dim["x_1"]>new_dim["width"] or \
            l_y - new_dim["y_1"]<0 or l_y - new_dim["y_1"] > new_dim["height"]:
            print("QUITTING; CRITICAL ERROR")
            return -1
        l_x = round(min(new_dim["width"], max(0, l_x - new_dim["x_1"])))
        l_y = round(min(new_dim["height"], max(0, l_y - new_dim["y_1"])))
        landmarks_new.append((l_x, l_y))
    return landmarks_new


def process_image(path, file):
    image = Image.open(path + f"{file}.jpg").convert('RGB')
    actual_size = image.size
    image_resized = image.resize([218, 218])
    img_transformed = torch.unsqueeze(ToTensor()(image_resized), 0)
    bbox = model(img_transformed.to("cuda")).tolist()[0]
    bbox_dicts = {"x_1": bbox[0], "y_1": bbox[1], "width": bbox[2], "height": bbox[3]}
    bbox_dicts = drawer.resize_bbox(bbox_dicts, model_size, actual_size)
    #drawer.process_img(image=image, bbox=bbox_dicts, base=False)
    image_cropped = image.crop((bbox_dicts["x_1"], bbox_dicts["y_1"],
                                bbox_dicts["width"] + bbox_dicts["x_1"],
                                bbox_dicts["height"] + bbox_dicts["y_1"]))

    return bbox_dicts,image_cropped

feat_dicts = format_landmarks(df_landmarks)

bad_log = []
new_writer = open("Anno/custom/list_landmarks_reshaped_celeba.txt", "w")
new_writer.write("lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y\n")
curr_id = 0
for imgID in tqdm(feat_dicts.keys()):
    #print(feat_dicts[imgID])
    #drawer.process_feats(Image.open(f"img_celeba/{imgID}.jpg").convert("RGB"),feat_dicts[imgID])
    bbox_dicts,image = process_image("img_celeba/",imgID)
    if process_landmarks(bbox_dicts,feat_dicts[imgID]) == -1:
        bad_log.append(imgID)
        continue
    landmarks_processed = process_landmarks(bbox_dicts,feat_dicts[imgID])
    true_id = int(imgID.split(".")[0])
    new_line = imgID+".jpg"
    for l_x,l_y in landmarks_processed:
        new_line += f",{l_x} {l_y}"
    new_line += "\n"
    new_writer.write(new_line)
    image.save(f"img_celeba_bboxed2/{imgID}.jpg")
    curr_id += 1
    #drawer.process_feats(image,process_landmarks(bbox_dicts,feat_dicts[imgID]))

new_writer.close()
print(len(bad_log))