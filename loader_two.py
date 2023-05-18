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
model_two = torch.load("modelEp15.pt", map_location=torch.device('cpu'))
model_two.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 218])])


def process_image(path, file, model):
    image = Image.open(path + f"{file}.jpg").convert('RGB')
    actual_size = image.size
    image_resized = image.resize([218, 218])
    img_transformed = torch.unsqueeze(ToTensor()(image_resized), 0)
    bbox = model(img_transformed.to("cuda")).tolist()[0]
    bbox_dicts = {"x_1": bbox[0], "y_1": bbox[1], "width": bbox[2], "height": bbox[3]}
    bbox_dicts = drawer.resize_bbox(bbox_dicts, (218, 218), actual_size)
    # drawer.process_img(image=image, bbox=bbox_dicts, base=False)
    image_cropped = image.crop((bbox_dicts["x_1"], bbox_dicts["y_1"],
                                bbox_dicts["width"] + bbox_dicts["x_1"],
                                bbox_dicts["height"] + bbox_dicts["y_1"]))

    return bbox_dicts, image_cropped


def process(path, file, device=torch.device("cpu")):
    img_size = Image.open(path + file).size()
    bbox_dicts, image_new = process_image(path, file, model_one)
    img_transformed = torch.unsqueeze(transform(image_new), 0)
    feats = model_two(img_transformed.to(device), train=False).tolist()[0]
    f0 = feats.copy()
    features = [(feats[i], feats[i + 1]) for i in range(0, len(feats), 2)]
    match = matcher(feats=np.array(f0))
    file = match[0][-1][:-1] + str(int(match[0][-1][:-1]) - 1)
    return file, image_new, drawer.resize_bbox(bbox_dicts, (218, 218), img_size), \
           drawer.reshape_feats(drawer.resize_feats(features, (218, 218), img_size), bbox_dicts)

