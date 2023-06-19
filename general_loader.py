import numpy as np
import torch
from PIL import Image
from tools.feature_matching_algorithm import MatcherDataset
from plotting.CelebADrawer import BboxFeatDrawer
from plotting.image_processing_wrappers import generic_tensor_transform, crop_image, result_wrap, argmax

default_model_input_size = (218, 218)

drawer = BboxFeatDrawer()
device = torch.device("cuda")
matcher = MatcherDataset("./feat_base", path="")
models_root = "./trained_models/"
bbox_model_dir = models_root + "bbox_model.pt"
feat_model_dir = models_root + "landmark_model.pt"
bbox_model = torch.load(bbox_model_dir, map_location=device)
bbox_model.eval()
feature_model = torch.load(feat_model_dir, map_location=device)
feature_model.eval()

race_model = torch.load(models_root + "race_model_bbox.pt", device)
age_model = torch.load(models_root + "age_model.pt", map_location=device)
gender_model = torch.load(models_root + "gender_model.pt", map_location=device)

age_dictionary = {1: "young", 0: "old"}
race_dictionary = {1: "black", 0: "white"}
gender_dictionary = {1: "male", 0: "female"}


def process_image(path, file, model, image=None):
    if not image:
        image = Image.open(path + f"{file}").convert('RGB')
    image_tensor = generic_tensor_transform(image, default_model_input_size)
    return result_wrap(model(image_tensor.to(device))), image


def process_image_bbox(path, file, model):
    model.to(device)
    bbox, image = process_image(path, file, model)
    bbox_resized = drawer.resize_bbox(bbox, default_model_input_size, image.size)
    image_cropped = crop_image(image, bbox_resized)
    return bbox_resized, image_cropped


# process vraća match_file, image_new, bbox_resized, feat_resized, sim, inappropriate, age, gender, race

def process(path, file):
    bbox_dicts, image_new = process_image_bbox(path, file, bbox_model)
    feats, img0 = process_image("", "", feature_model, image=image_new)
    img_size = Image.open(path + file).size

    race = argmax(process_image("", "", race_model, image=image_new)[0])
    age = argmax(process_image(path, file, age_model)[0])
    gender = argmax(process_image(path, file, gender_model, image=image_new)[0])

    match = matcher(feats=np.array(feats), race=race, age=age, gender=gender)
    match_file = match[0][-1]
    similarity_index = match[-2]
    inappropriate = match[-1]

    new_bbox = drawer.resize_bbox(bbox_dicts, default_model_input_size, img_size)
    feats_resized = drawer.resize_feats(feats, default_model_input_size, img_size)
    feats_reshaped = drawer.reshape_feats(feats_resized, bbox_dicts, img_size)

    return match_file, image_new, new_bbox, feats_resized, feats_reshaped, similarity_index, inappropriate, \
           age_dictionary[age], gender_dictionary[gender], race_dictionary[race]
