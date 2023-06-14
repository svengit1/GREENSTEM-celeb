import torch
from torchvision import transforms as t


def un_squeeze(image_tensor):
    return torch.unsqueeze(image_tensor, 0)


def generic_tensor_transform(image, new_size):
    return un_squeeze(t.ToTensor()(image.resize(new_size)))


def crop_image(image, bbox):
    return image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

def result_wrap(tensor):
    return tensor.tolist()[0] if tensor.shape != (1,1) else tensor.item()

def argmax(list):
    for l in range(len(list)):
        list[l] = round(list[l])
    return list.index(1)