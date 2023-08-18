import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from global_config import device_type
from global_config import image_crop_size


class DIV2KDataset(Dataset):
    def __init__(self, train=True, lrtype="LR_bicubic/X2", ycbcr=False):
        # Take images from training or validation dataset if True
        self.train = train
        self.ycbcr = ycbcr
        self.name = "DIV2K"
        self.base_filepath = "./data/DIV2K"
        if train:
            self.base_filepath += "_train_"
        else:
            self.base_filepath += "_valid_"

        self.lr_type = lrtype

        self.wild_level = "x4w1"
        self.resolutions_2017 = ["LR_bicubic/X2", "LR_bicubic/X3",
                                 "LR_bicubic/X4", "LR_unknown/X2", "LR_unknown/X3" "LR_unknown/X4"]
        self.resolutions_2018 = ["LR_x8", "LR_mild", "LR_difficult", "LR_wild"]

        self.res_to_filename_suffix = {"LR_bicubic/X2": "x2", "LR_bicubic/X3": "x3", "LR_bicubic/X4": "x4",
                                       "LR_unknown/X2": "x2", "LR_unknown/X3": "x3", "LR_unknown/X4": "x4",
                                       "LR_x8": "x8", "LR_mild": "x4m", "LR_difficult": "x4d", "LR_wild": self.wild_level}

        # transform function to get tensor from PIL image
        self.transform_img_to_tensor = transforms.ToTensor()

    def image_crop(self, scale_factor):
        result = torch.nn.Sequential(
            transforms.CenterCrop(image_crop_size // scale_factor),
            transforms.Resize((image_crop_size, image_crop_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_crop_size)
        ).to(device_type)
        return result

    def image_crop_bicubic(self, scale_factor):
        result = torch.nn.Sequential(
            transforms.CenterCrop(image_crop_size // scale_factor),
            transforms.Resize((image_crop_size, image_crop_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_crop_size)
        ).to(device_type)
        return result

    def set_LR_type(self, lrtype):
        all_res = self.resolutions_2017 + self.resolutions_2018
        if lrtype not in all_res:
            print("invalid lr-type, lr-type was kept unchanged (" + self.lr_type + ")")
            return
        self.lr_type = lrtype

    def __getitem__(self, idx):
        id = idx+1
        if not self.train:
            id = id + 800

        file_name = "{0:04d}".format(id)

        lr_filename = file_name + \
            self.res_to_filename_suffix[self.lr_type] + ".png"
        hr_filename = file_name + ".png"

        filename_path_lr = "" + self.base_filepath + self.lr_type + "/" + lr_filename
        filename_path_hr = "" + self.base_filepath + "HR" + "/" + hr_filename

        if self.ycbcr:
            lr = Image.open(filename_path_lr).convert("YCbCr")
            hr = Image.open(filename_path_hr).convert("YCbCr")
        else:
            lr = Image.open(filename_path_lr)
            hr = Image.open(filename_path_hr)

        lr_tensor = self.transform_img_to_tensor(lr).to(device_type)
        hr_tensor = self.transform_img_to_tensor(hr).to(device_type)

        final_lr = self.image_crop(2)(lr_tensor).to(device_type)
        final_hr = self.image_crop(1)(hr_tensor).to(device_type)

        if self.ycbcr and self.train:
            final_lr = final_lr[0:1]
            final_hr = final_hr[0:1]

        return final_lr, final_hr

    def get_bicubic(self, index):
        id = index+1
        if not self.train:
            id = id + 800
        file_name = "{0:04d}".format(id)
        lr_filename = file_name + \
            self.res_to_filename_suffix[self.lr_type] + ".png"

        filename_path_lr = "" + self.base_filepath + self.lr_type + "/" + lr_filename
        lr = Image.open(filename_path_lr)
        lr_tensor = self.transform_img_to_tensor(lr).to(device_type)
        final_lr = self.image_crop_bicubic(2)(lr_tensor).to(device_type)
        return final_lr

    def __len__(self):
        if self.train:
            return 50
        else:
            return 100
