# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from PIL import Image
import os.path
from astropy.io import fits
import numpy as np
import code

import datasets.transforms as T

class CocoDetection_dataset(torchvision.datasets.vision.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        noisy_data,
        noisy_datax2,
        transform = None,
        target_transform = None,
        transforms = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
        
    def _load_image_fits(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"].replace('rgb.png', 'emu.fits')
        r = fits.open(os.path.join(self.root, path))[0].data
        r = ((2**32-1)*r).astype(np.uint32)
        path = self.coco.loadImgs(id)[0]["file_name"].replace('emu.fits', 'rgb.png')
        im = np.array(Image.open(os.path.join(self.root, path).replace('fits', '2017')).convert("RGB"))
        if r.shape != im[:,:,0].shape:
            return Image.fromarray(im, "RGB")
        img = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
        img[:,:,2] = im[:,:,2]; img[:,:,1] = r # Radio preprocessed and raw
        #img[:,:,2] = r >> 24; img[:,:,1] = (r >> 16) & 0xFF # Radio raw and raw
        img[:,:,0] = im[:,:,0] # Infrared preprocessed
        return Image.fromarray(img, "RGB")
        
    def _load_image_fitsx2(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"].replace('rgb.png', 'emu.fits')
        r = fits.open(os.path.join(self.root, path))[0].data
        r = ((2**32-1)*r).astype(np.uint32)
        path = self.coco.loadImgs(id)[0]["file_name"].replace('emu.fits', 'rgb.png')
        im = np.array(Image.open(os.path.join(self.root, path).replace('_fits', '2017')).convert("RGB"))
        if r.shape != im[:,:,0].shape:
            return Image.fromarray(im, "RGB")
        img = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
        #img[:,:,2] = im[:,:,2]; img[:,:,1] = r # Radio preprocessed and raw
        img[:,:,2] = r >> 24; img[:,:,1] = (r >> 16) & 0xFF # Radio raw and raw
        img[:,:,0] = im[:,:,0] # Infrared preprocessed
        return Image.fromarray(img, "RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        if self.noisy_data:
            image = self._load_image_fits(id)
        if self.noisy_datax2:
            image = self._load_image_fitsx2(id)
        else:
            image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(CocoDetection_dataset):
    def __init__(self, img_folder, ann_file, noisy_data, noisy_datax2, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file, noisy_data, noisy_datax2)
        self._transforms = transforms
        self.noisy_data = noisy_data
        self.noisy_datax2 = noisy_datax2
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            """if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)"""

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args, crop=True):

    if args.noisy_data or args.noisy_datax2:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.06935824, 0.3000091, 0.29598746], [0.18748689, 0.4532347, 0.4564638]) #Raw radio
            ])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.06935824, 0.016939376, 0.016241161], [0.18748689, 0.10635011, 0.10272085]) # preprocessed radio
            ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #scales = [224, 224+32, 224+2*32, 224+3*32, 224+4*32, 224+5*32, 224+6*32, 224+7*32]
    max_size = 1333 #800 #448

    if image_set == 'train':
    	if crop:
    		return T.Compose([
    			T.RandomHorizontalFlip(),
    			T.RandomRotation(),
    			T.RandomSelect(
    			T.RandomResize(scales, max_size=max_size),
    				T.Compose([
    					T.RandomResize([400, 500, 600]), #([224, 336, 448]),
    					T.RandomSizeCrop(384, 384), #(128, 128), 
    					T.RandomResize(scales, max_size=max_size),
    				])
    			),
    			normalize,
    			])
    	else:
    		return T.Compose([
    			T.RandomHorizontalFlip(),
    			T.RandomRotation(),
    			T.RandomResize(scales, max_size=max_size),
    			normalize,
    		])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=max_size), #256
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = ''  # instances_
    PATHS = {
        "train": (root / "train", root / "annotations" / f'{mode}train.json'),
        "val": (root / "val", root / "annotations" / f'{mode}val.json'),
        "test": (root / "test", root / "annotations" / f'{mode}test.json'),
    }
    if args.noisy_data or args.noisy_datax2:
        PATHS = {
            "train": (root / "train_fits", root / "annotations" / f'{mode}train.json'),
            "val": (root / "val_fits", root / "annotations" / f'{mode}val.json'),
            "test": (root / "test_fits", root / "annotations" / f'{mode}test.json'),
        }
    
    if args.noisy_dataPNG:
        PATHS = {
            "train": (root / "train_noisy", root / "annotations" / f'{mode}train.json'),
            "val": (root / "val_noisy", root / "annotations" / f'{mode}val.json'),
            "test": (root / "test_noisy", root / "annotations" / f'{mode}test.json'),
        }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, args.noisy_data, args.noisy_datax2, transforms=make_coco_transforms(image_set, args), return_masks=True)
    return dataset



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from util.box_ops import box_cxcywh_to_xyxy

    class DefaultArgs():
        def __init__(self):
            self.coco_path = 'RadioGalaxyNET'
            self.masks = None
            self.keypoints = True
            self.noisy_data = False
            self.noisy_datax2 = True
            self.noisy_dataPNG = False

    args = DefaultArgs()
    ds = build(image_set='test', args=args)
    
    for i in range(29,30):
        img, target = ds[i]
        ih, iw = img.shape[-2:]
        target["boxes"] = target["boxes"] * torch.tensor([iw, ih, iw, ih], dtype=torch.float32)
        target["boxes"] = box_cxcywh_to_xyxy(target["boxes"])
        img = img.permute((1,2,0))
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        ax.set_autoscale_on(False)
        print("Total boxes: {}".format(len(target["boxes"])))
        for idx, ann in enumerate(target['boxes']):
            [x, y, w, h] = ann
            p = Rectangle((x, y), (w-x), (h-y), linewidth=2,
                            alpha=0.7, linestyle="solid",
                            edgecolor="red", facecolor='none')
            ax.add_patch(p)
            ax.text(x, y, "{}".format(target['labels'][idx]), color="red", size=14, backgroundcolor="black")
        for idx, ann_keys in enumerate(target['keypoints']):
            x, y, zz = ann_keys * torch.tensor([iw, ih, 1], dtype=torch.float32)
            print(ann_keys * torch.tensor([iw, ih, 1], dtype=torch.float32))
            ax.scatter(x, y, s =50, color='orange')
        plt.show()
