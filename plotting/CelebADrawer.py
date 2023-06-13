from PIL import ImageDraw


# Pretežito gotovo
class BboxFeatDrawer:

    def __process_datas(self, old_size, new_size, data):
        if data.__class__ == list:
            for i in range(len(data)):
                dat = self.__process_datas(old_size, new_size, data[i])
                if dat:
                    data[i] = dat
        elif data.__class__ == tuple:
            return tuple([data[index] / old_size[index] * new_size[index] for index in range(len(data))])
        else:
            raise NotImplementedError(f"Datatype {data.__class__} found!, not supported or invalid.")
        return data

    def paint_bbox(self, img, bbox_data, save=False):
        bbox_data = self.resize_bbox(bbox_data)
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_data, width=10, outline="green")
        if save:
            img.save("processed_bbox.png")
            return

    def paint_features(self, img, features: list[tuple], size=1.0, color="blue"):
        draw = ImageDraw.Draw(img)
        for l_x, l_y in features:
            draw.polygon((l_x - 10 * size, l_y, l_x, l_y + 10 * size, l_x + 10 * size, l_y,
                          l_x, l_y - 10 * size), fill=color)
        img.save("processed_feats.png")

    def __re_tuple(self, list):
        return [(list[i], list[i + 1]) for i in range(0,len(list), 2)] if list[0].__class__ != tuple else list

    def __un_tuple(self,list):
        new_list= []
        for a,b in list:
            new_list.append(a)
            new_list.append(b)
        return new_list

    def resize_bbox(self, bbox_data, old_size: tuple = None, new_size: tuple = None, separate=False):
        if bbox_data.__class__ == dict: self.__standardize_bbox(bbox_data)
        if new_size: bbox_data = self.__un_tuple(self.__process_datas(old_size, new_size, self.__re_tuple(bbox_data)))
        return bbox_data if not separate else self.__separate_tuples(bbox_data)

    def __standardize_bbox(self, bbox):
        bbox = bbox.values()
        return [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])]

    def __separate_tuples(self, array: list[tuple]):
        x = [[] for _ in range(len(array[0]))]
        for element in array:
            for e in range(len(element)):
                x[e].append(element[e])
        return x

    def resize_feats(self, feat_data: list, old_size: tuple, new_size: tuple):
        self.__process_datas(old_size, new_size, self.__re_tuple(feat_data))
        return feat_data

    def reshape_feats(self, feat_data: list, bbox_data, img_size):
        feat_data = self.__re_tuple(feat_data)
        return [(feat_data[i][0] + bbox_data[0],
                 img_size[1] - (feat_data[i][1] + bbox_data[1])) for i in range(0, len(feat_data))]


class PillowManipulator(BboxFeatDrawer):

    def process_resize_img_spec(self, img, new_size, data: list[tuple]):
        old_size = img.size
        data = self.__process_datas(old_size, new_size, data)
        img = img.resize(new_size)
        return img, data
