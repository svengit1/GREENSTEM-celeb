import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
from UI.tools.feature_matching_algorithm import MatcherDataset
from plotting.CelebADrawer import BboxDrawer

drawer = BboxDrawer()
device = torch.device("cuda")
matcher = MatcherDataset("feat_base", path="")
model_one = torch.load("BBOX_MODELS/modelEp9.pt", map_location=device)
model_one.eval()
model_two = torch.load("ATTR_MODELS/modelEp15.pt", map_location=device)
model_two.eval()
model_qc = torch.load("INAPPROP_MODELS/modelEp3.pt", map_location=device)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 218])])

def process_image(path, file, model,image=None):
    if not image:
        image = Image.open(path + f"{file}.jpg").convert('RGB')
    actual_size = image.size
    image_resized = image.resize([218, 218])
    img_transformed = torch.unsqueeze(ToTensor()(image_resized), 0)
    return model(img_transformed.to(device)), image, actual_size


def process_image_bbox(path, file, model):
    model.to(device)
    bbox, image, actual_size = process_image(path, file, model)
    bbox = bbox.tolist()[0]
    bbox_dicts = {"x_1": bbox[0], "y_1": bbox[1], "width": bbox[2], "height": bbox[3]}
    bbox_dicts = drawer.resize_bbox(bbox_dicts, (218, 218), actual_size)
    # drawer.process_img(image=image, bbox=bbox_dicts, base=False)
    image_cropped = image.crop((bbox_dicts["x_1"], bbox_dicts["y_1"],
                                bbox_dicts["width"] + bbox_dicts["x_1"],
                                bbox_dicts["height"] + bbox_dicts["y_1"]))
    return bbox_dicts, image_cropped


def process(path, file, device=torch.device("cpu")):
    img_size = Image.open(path + file).size
    bbox_dicts, image_new = process_image_bbox(path, file, model_one)
    img_transformed = torch.unsqueeze(transform(image_new), 0)
    feats = model_two(img_transformed.to(device), train=False).tolist()[0]
    f0 = feats.copy()
    features = [(feats[i], feats[i + 1]) for i in range(0, len(feats), 2)]
    match = matcher(feats=np.array(f0))
    file = match[0][-1][:-1] + str(int(match[0][-1][:-1]) - 1)
    return file, image_new, drawer.resize_bbox(bbox_dicts, (218, 218), img_size), \
        drawer.reshape_feats(drawer.resize_feats(features, (218, 218), img_size), bbox_dicts)

