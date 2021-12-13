import os
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class LowContrastReceipts(Dataset):
    def __init__(self, image_dir, width, height, transform=None, color="RGB", training=False, scale=1, ensemble_size=1):
        self.image_dir = image_dir
        self.width = width
        self.height = height
        self.transform = transform
        self.num_ens = ensemble_size
        #Check that order of images are correct
        self.images = sorted(os.listdir(image_dir))
        self.color = color
        self.training = training

        #read all 500 bounding boxes
        self.boxes = []
        self.fixed_imgs = []
        json_file = "data/data_info.json"
        json_dict = json.load(open(json_file))
        data_arr = json_dict["data_info"]
        width = 1224
        height = 1632

        for i in range(500):
            self.fixed_imgs.append(self.createImg(data_arr[i]["image_name"]))

            p1 = data_arr[i]["top_left"]
            p2 = data_arr[i]["top_right"]
            p3 = data_arr[i]["bottom_right"]
            p4 = data_arr[i]["bottom_left"]
            #values are percent of the image to adjust for image size
            if scale == 1:
                parr = [
                    p1[0]/width, p1[1]/height, p2[0]/width, p2[1]/height,
                    p3[0]/width, p3[1]/height, p4[0]/width, p4[1]/height
                ]
            else:
                parr = [
                    100*p1[0]/width, 100*p1[1]/height, 100*p2[0]/width, 100*p2[1]/height,
                    100*p3[0]/width, 100*p3[1]/height, 100*p4[0]/width, 100*p4[1]/height
                ]
            self.boxes.append(np.array(parr, dtype=np.float32))

    def createImg(self, img_path):
        if self.color == "RGB":
            temp_img = Image.open(img_path).resize((self.width,self.height)).convert("RGB")
            image = np.array(temp_img, dtype=np.float32)
        elif self.color == "Gray":
            temp_img = Image.open(img_path).resize((self.width,self.height)).convert("L")
            image = np.array(temp_img, dtype=np.float32)
            image = image[..., np.newaxis]

        image = np.transpose(image, (2, 0, 1))
        return image

    def __len__(self):
        if self.training:
            return len(self.images)//self.num_ens
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.training:
            #get correct bounding box
            imgs = []
            labels = []
            for i in range(self.num_ens):
                imgs.append(self.fixed_imgs[index * self.num_ens + i])
                labels.append(self.boxes[index * self.num_ens + i])
        else:
            #the images are in order, validation is last 100
            image = self.fixed_imgs[index+400]
            label = self.boxes[index+400]
            imgs = []
            labels = []
            for i in range(self.num_ens):
                imgs.append(image)
                labels.append(label)

        # return image, labels
        return np.concatenate(imgs, axis=0), np.stack(labels, axis=0)