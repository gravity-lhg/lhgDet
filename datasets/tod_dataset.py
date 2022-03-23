# Author: Haoguang Liu
# Data: 2022.3.23 20:09 PM
# Email: 1052979481@qq.com
# Github: https://github.com/gravity-lhg

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class todDataset(Dataset):
    '''for AI-TOD dataset'''
    CLASSES_NAME = (
        '__background__',
        'airplane',
        'bridge',
        'storage-tank',
        'ship',
        'swimming-pool',
        'vehicle',
        'person',
        'wind-mill',
    )
    def __init__(self, root_path, mode, resize_size=None):
        super(todDataset).__init__()
        self.root_path = root_path
        self.mode = mode
        self.resize_szie = resize_size
        self.mean = [0.279, 0.268, 0.216]
        self.std = [0.126, 0.120, 0.114]

        self.ann_path = os.path.join(self.root_path, self.mode, 'labels', '%s.txt')
        self.img_path = os.path.join(self.root_path, self.mode, 'images', '%s.png')

        self.img_list = os.listdir(os.path.join(self.root_path, self.mode, 'images'))
        self.imgID_list = [img_name.replace('.png', '') for img_name in self.img_list]
        self.cat_name2id = dict(zip(todDataset.CLASSES_NAME, range(len(todDataset.CLASSES_NAME))))
        print("INFO ===> AI-TOD dataset init finished !")

    def __getitem__(self, index):
        imgID = self.imgID_list[index]
        img = self._read_img_rgb(self.img_path % imgID)

        boxes = []
        classes = []
        with open(self.ann_path % imgID) as f:
            datas = f.readlines()
            for item in datas:
                itemList = item.split(' ')
                box = [
                    float(itemList[0]), float(itemList[1]), float(itemList[2]), float(itemList[3])
                ]
                box = tuple(box)
                boxes.append(box)
                cat_name = itemList[4].replace('\n', '')
                classes.append(self.cat_name2id[cat_name])

        boxes = np.array(boxes, dtype=np.float32)

        # Convert each data to tensor
        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def __len__(self):
        return len(self.imgID_list)

    def _read_img_rgb(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    # create mini-batch of Tensor-type data
    def collate_fn(self, data):
        imgs_tuple, boxes_tuple, classes_tuple = zip(*data)
        assert len(imgs_tuple) == len(boxes_tuple) == len(classes_tuple), 'There is a problem with the dataset'
        batch_size = len(imgs_tuple)

        # pad image, boxes, and classes
        pad_img_list = []
        pad_boxes_list = []
        pad_classes_list = []

        # image is converted by ToTensor(), (H x W x C) -> (C xH x W)
        h_list = [int(img_Tensor.shape[1]) for img_Tensor in imgs_tuple]
        w_list = [int(img_Tensor.shape[2]) for img_Tensor in imgs_tuple]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img_Tensor = imgs_tuple[i]
            pad_img_list.append(
                # F.pad(img_Tensor, (tuple: start from last dim, (front, back)...), value)
                transforms.Normalize(self.mean, self.std, inplace=True)
                    (F.pad(img_Tensor, (0, int(max_w - img_Tensor.shape[-1]), 0, int(max_h - img_Tensor.shape[-2])), value=0.))
            )

        # Align box and class data
        max_num = 0
        for i in range(batch_size):
            n = boxes_tuple[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(F.pad(boxes_tuple[i], (0, 0, 0, max_num - boxes_tuple[i].shape[0]), value=-1))
            pad_classes_list.append(F.pad(classes_tuple[i], (0, max_num - classes_tuple[i].shape[0]), value=-1))

        batch_imgTensor = torch.stack(pad_img_list)
        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)

        return batch_imgTensor, batch_boxes, batch_classes

if __name__ == '__main__':
    dataSet = todDataset('/Users/lhg/Downloads/AI-TOD', 'train')
    # print(dataSet[0])
    imgs, boxes, classes = dataSet.collate_fn([dataSet[105], dataSet[101], dataSet[200]])
    print(boxes, classes, "\n" ,imgs.shape, boxes.shape, classes.shape, boxes.dtype, classes.dtype, imgs.dtype)