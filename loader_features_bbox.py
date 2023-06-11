import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
from UI.tools.feature_matching_algorithm import MatcherDataset
from plotting.CelebADrawer import BboxDrawer

device = torch.device("cpu")
bbox_model_dir = "modelEp9.pt"
feat_model_dir = "modelEp15.pt"

drawer = BboxDrawer()
matcher = MatcherDataset("feat_base", path="")
bbox_model = torch.load(bbox_model_dir, map_location=device)
bbox_model.eval()
feature_model = torch.load(feat_model_dir, map_location=device)
feature_model.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 218])])


def process_image(path, file, model):
    image = Image.open(path + f"{file}.jpg").convert('RGB')
    actual_size = image.size
    image_resized = image.resize([218, 218])
    img_transformed = torch.unsqueeze(ToTensor()(image_resized), 0)
    bbox = model(img_transformed.to("cuda")).tolist()[0]
    bbox_dicts = {"x_1": bbox[0], "y_1": bbox[1], "width": bbox[2], "height": bbox[3]}
    bbox_dicts = drawer.resize_bbox(bbox_dicts, (218, 218), actual_size)
    image_cropped = image.crop((bbox_dicts["x_1"], bbox_dicts["y_1"],
                                bbox_dicts["width"] + bbox_dicts["x_1"],
                                bbox_dicts["height"] + bbox_dicts["y_1"]))

    return bbox_dicts, image_cropped


def process(path, file, device=torch.device("cpu")):
    img_size = Image.open(path + file).size()
    bbox_dicts, image_new = process_image(path, file, bbox_model)
    img_transformed = torch.unsqueeze(transform(image_new), 0)
    feats = feature_model(img_transformed.to(device), train=False).tolist()[0]
    f0 = feats.copy()
    features = [(feats[i], feats[i + 1]) for i in range(0, len(feats), 2)]
    match = matcher(feats=np.array(f0))
    file = match[0][-1][:-1] + str(int(match[0][-1][:-1]) - 1)
    return file, image_new, drawer.resize_bbox(bbox_dicts, (218, 218), img_size), \
           drawer.reshape_feats(drawer.resize_feats(features, (218, 218), img_size), bbox_dicts,img_size)

