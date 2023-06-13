**Gal-DETR**: Radio Galaxy Detection with Transformers
========

# Usage
Create a Python 3.8 environement with CUDA 11.3.0. 
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools and scipy:
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Install packages in requirements.txt.

## Data preparation

Download and extract RadioGalaxyNET data from the link described in the datasheet.
We expect the directory structure to be the following:
```
RadioGalaxyNET/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
  test/     # test images
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
To ease reproduction of our results we provide model checkpoint [here]().

## Evaluation
To evaluate DETR R50 on test images with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```
# License
The License will be updated after publication. Note that original DETR is released under the Apache 2.0 license. 
