# -*- coding: utf-8 -* -

import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """MyDataset"""
    def __init__(self, data_path, transform=None, target_transform=None):
        self.classes = os.listdir(data_path)
        self.classes = [class_name for class_name in self.classes if class_name[0] != "."]
        self.class_name_to_id = {class_name: ix for ix, class_name in enumerate(self.classes)}
        self.id_to_class_name = {ix: class_name for ix, class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        print("num classes is:", self.num_classes)
        imgs = []
        for class_name in self.classes:
            image_list = glob.glob(os.path.join(data_path, class_name, "*.jpg"))
            imgs += [(image_file, self.class_name_to_id[class_name]) for image_file in image_list]

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)
