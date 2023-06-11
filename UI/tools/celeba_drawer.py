import os
import matplotlib.image as image
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


### INIT DAJES DIR_ADDR: TAMO SU SLIKE i CSV_ADDR: ime/adresa samog csv filea
### KADA ZELIS NEKU SLIKU PRIKAZATI SA BBOXOM, ZOVES process_img od inicializiranog objekta,
class BboxDrawer:

    def __init__(self, dir_addr="", csv_addr=""):
        if csv_addr:
            self.bbox_datas = pd.read_csv(csv_addr, delim_whitespace=True)
            self.dir_addr = dir_addr
            self.all_dirs = {int(item.split(".")[0]): item for item in os.listdir(dir_addr)}

    def make_box(self, x_1, y_1, w, h):
        x = [x_1, x_1 + w, x_1 + w, x_1, x_1]
        y = [y_1, y_1, y_1 + h, y_1 + h, y_1]
        return x, y

    def process_img(self, debug=True, base=True, save=False, **kwargs):
        print(kwargs)
        if not base:
            img = kwargs["image"]
            bbox_data = kwargs["bbox"]
        else:
            file = self.all_dirs[kwargs["image_id"]]
            img = image.imread(self.dir_addr + "/" + file)
            bbox_data = self.bbox_datas.iloc[kwargs["image_id"] - 1]
        if debug:
            print(bbox_data)
            print(bbox_data.__class__)
        x, y = self.make_box(bbox_data["x_1"], bbox_data["y_1"], bbox_data["width"], bbox_data["height"])
        draw = ImageDraw.Draw(img)
        draw.rectangle((x[0], y[0], x[2], y[2]), width=10, outline="green")
        if save:
            img.save("processed_bbox.png")
            return
        plt.show()

    def process_feats(self, img, features: list[tuple], size=1.0, color="blue"):
        draw = ImageDraw.Draw(img)
        for l_x, l_y in features:
            draw.polygon((l_x - 10 * size, l_y, l_x, l_y + 10 * size, l_x + 10 * size, l_y,
                          l_x, l_y - 10 * size), fill=color)
        img.save("processed_feats.png")

    def process_resize_img(self, image_id, new_size: tuple = (218, 178), save=False):
        file = self.all_dirs[image_id]
        bbox_data = self.bbox_datas.iloc[image_id - 1]
        image = Image.open(self.dir_addr + "/" + file)
        img = image.resize(new_size)
        image_size = image.size
        bbox_data = self.resize_bbox(bbox_data, image_size, new_size)
        x, y = self.make_box(bbox_data["x_1"],
                             bbox_data["y_1"],
                             bbox_data["width"],
                             bbox_data["height"])
        print(x, y)
        plt.plot(x, y, color="black", linewidth=3)
        if save:
            plt.savefig("processed.jpg")
            return
        plt.imshow(img)
        plt.show()

    def resize_bbox(self, bbox_data: dict, old_size: tuple, new_size: tuple):
        return {"x_1": bbox_data["x_1"] * new_size[0] / old_size[0],
                "y_1": bbox_data["y_1"] * new_size[1] / old_size[1],
                "width": bbox_data["width"] * new_size[0] / old_size[0],
                "height": bbox_data["height"] * new_size[1] / old_size[1]}

    def resize_feats(self, feat_data: list, old_size: tuple, new_size: tuple):
        return [(round(feat_data[i][0] * new_size[0] / old_size[0]),
                 round(feat_data[i][1] * new_size[1] / old_size[1])) for i in range(0, len(feat_data))]

    def reshape_feats(self, feat_data: list, bbox_data, img_size):
        return [(feat_data[i][0] + bbox_data["x_1"],
                 img_size[1] - (feat_data[i][1] + bbox_data["y_1"])) for i in range(0, len(feat_data))]


class PillowManipulator:

    def __init__(self):
        self.drawer_inst = BboxDrawer()

    def make_bbox(self, x_1, y_1, w, h):
        BboxDrawer.make_box(self.drawer_inst, x_1, y_1, w, h)

    def draw_feats(self, image, features: list[tuple], size=1.0, color="blue"):
        BboxDrawer.process_feats(self.drawer_inst, image, features, size, color)

    def __process_datas(self, old_size, new_size, data):
        if data.__class__ == list:
            for i in range(len(data)):
                dat = self.__process_datas(old_size, new_size, data[i])
                if dat:
                    data[i] = dat
        return data[0] / old_size[0] * new_size[0], data[1] / old_size[1] * new_size[1]

    def process_resize_img(self, img, new_size, data: list[tuple]):
        old_size = img.size
        data = self.__process_datas(old_size, new_size, data)
        img = img.resize(new_size)
        return img, data


class NoseManipulator(PillowManipulator):

    def __init__(self):
        super(NoseManipulator, self).__init__()

    def get_nose_area(self, image, feats, crop_size):
        nose_coords = [feats[2][0], feats[2][1]] * 2
        print(nose_coords)
        crop_size = 10 * crop_size
        cropping = [nose_coords[0] - crop_size, nose_coords[1] - crop_size, nose_coords[0] + crop_size,
                    nose_coords[1] + crop_size]
        print(cropping)
        print(image.size)
        return image.crop(cropping)

##PRIMJER
# processor = BboxDrawer(dir_addr="test_images",csv_addr="bbox_mini.csv")
# processor.process_img(image_id=5)
# processor.process_resize_img(image_id=5)