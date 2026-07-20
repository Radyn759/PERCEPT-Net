import torch
import random


class PairedAugmentor:
    def __init__(self,
                 use_flip=True,
                 use_rot90=True,
                 use_random_crop=False,
                 crop_size=None):
        self.use_flip = use_flip
        self.use_rot90 = use_rot90
        self.use_random_crop = use_random_crop
        self.crop_size = crop_size

    def __call__(self, input_img, gt_img):
        H, W = input_img.shape[-2:]
        seed = random.randint(0, 100000)

        torch.manual_seed(seed)
        random.seed(seed)

        if self.use_flip and random.random() > 0.5:
            input_img = torch.flip(input_img, dims=[-1])
            gt_img = torch.flip(gt_img, dims=[-1])

        if self.use_flip and random.random() > 0.5:
            input_img = torch.flip(input_img, dims=[-2])
            gt_img = torch.flip(gt_img, dims=[-2])

        if self.use_rot90:
            k = random.randint(0, 3)
            if k > 0:
                input_img = torch.rot90(input_img, k=k, dims=[-2, -1])
                gt_img = torch.rot90(gt_img, k=k, dims=[-2, -1])

        if self.use_random_crop and self.crop_size is not None:
            ch, cw = self.crop_size
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            input_img = input_img[..., top:top + ch, left:left + cw]
            gt_img = gt_img[..., top:top + ch, left:left + cw]

        return input_img, gt_img