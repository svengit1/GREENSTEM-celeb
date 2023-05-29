import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
from FDA import MatcherDataset
from testing.cnn_class import CNN
from plotting.CelebADrawer import BboxDrawer

drawer = BboxDrawer()
matcher = MatcherDataset("feat_base", path="")
model_one = torch.load("modelEp9.pt", map_location=torch.device('cpu'))
model_one.eval()
print(model_one)
model_two = torch.load("modelEp20.pt", map_location=torch.device('cpu'))
model_two.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 218])])


def process_image(path, file, model,device):
    image = Image.open(path + f"{file}").convert('RGB')
    actual_size = image.size
    image_resized = image.resize([218, 218])
    img_transformed = torch.unsqueeze(ToTensor()(image_resized), 0)
    bbox = model(img_transformed.to(device)).tolist()[0]
    bbox_dicts = {"x_1": bbox[0], "y_1": bbox[1], "width": bbox[2], "height": bbox[3]}
    bbox_dicts = drawer.resize_bbox(bbox_dicts, (218, 218), actual_size)
    # drawer.process_img(image=image, bbox=bbox_dicts, base=False)
    image_cropped = image.crop((bbox_dicts["x_1"], bbox_dicts["y_1"],
                                bbox_dicts["width"] + bbox_dicts["x_1"],
                                bbox_dicts["height"] + bbox_dicts["y_1"]))

    return bbox_dicts, image_cropped


def process(path, file, device=torch.device("cpu")):
    img_size = Image.open(path + file).size
    bbox_dicts, image_new = process_image(path, file, model_one,device)
    img_transformed = torch.unsqueeze(transform(image_new), 0)
    feats = model_two(img_transformed.to(device), train=False).tolist()[0]
    f0 = feats.copy()
    features = [(feats[i], feats[i + 1]) for i in range(0, len(feats), 2)]
    match = matcher(feats=np.array(f0))
    file = str(int(match[0][-1].split(".")[0])-1)
    file = "0"*(6-len(file))+file+".jpg"
    feats_resized = drawer.resize_feats(features, (218, 218), image_new.size)
    feats_reshaped = drawer.reshape_feats(feats_resized,bbox_dicts,img_size)
    return file, image_new,bbox_dicts,feats_resized,feats_reshaped


