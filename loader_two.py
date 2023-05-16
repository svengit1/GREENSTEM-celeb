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


def process_image(path, file,model):
    image = Image.open(path + f"{file}.jpg").convert('RGB')
    actual_size = image.size
    image_resized = image.resize([218, 218])
    img_transformed = torch.unsqueeze(ToTensor()(image_resized), 0)
    bbox = model(img_transformed.to("cuda")).tolist()[0]
    bbox_dicts = {"x_1": bbox[0], "y_1": bbox[1], "width": bbox[2], "height": bbox[3]}
    bbox_dicts = drawer.resize_bbox(bbox_dicts, (218,218), actual_size)
    #drawer.process_img(image=image, bbox=bbox_dicts, base=False)
    image_cropped = image.crop((bbox_dicts["x_1"], bbox_dicts["y_1"],
                                bbox_dicts["width"] + bbox_dicts["x_1"],
                                bbox_dicts["height"] + bbox_dicts["y_1"]))

    return bbox_dicts,image_cropped

matcher = MatcherDataset("feat_base",path="")
model_one = torch.load("BBOX_MODELS/modelEp9.pt")
model_one.eval()
model_two = torch.load("ATTR_MODELS/modelEp15.pt")
model_two.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 218])])
path= "img_celeba_bboxed/"
for i in range(119941,119943):
    image = Image.open(path+f"{i}.jpg").convert('RGB')
    image_resized = transforms.Resize([218, 218])(image)
    print(transform(image).shape)
    bbox_new,image_new = process_image(path,f"{i}",model_one)
    img_transformed = torch.unsqueeze(transform(image_new), 0)
    feats = model_two(img_transformed.to("cuda"),train=False).tolist()[0]
    f0 = feats.copy()
    feats = [(feats[i],feats[i+1]) for i in range(0,len(feats),2)]
    print(feats)
    drawer.process_feats(image_new,features=drawer.resize_feats(feats,old_size=(218,218),new_size=image_new.size))
    match = matcher(feats=np.array(f0))
    print(match)
    file = match[0][-1]
    image = Image.open(path +file).convert('RGB')
    image.show()