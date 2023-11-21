**Gal-DETR**
========

## Installation
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
```
pip install -r requirements.txt
```

## Data preparation

Download and extract RadioGalaxyNET data from [here](https://data.csiro.au/collection/61068).
We expect the directory structure to be the following:
```
./RadioGalaxyNET/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
  test/     # test images
```

## Training
To train on a single node with 4 gpus run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --output_dir ./outputs_gal/
```
To ease reproduction of our results we provide model checkpoint [here](https://figshare.com/s/4818dcd3abf0a9d425a3). 
Place the model in `./outputs_gal/` directory.

## Evaluation
To evaluate on test images with a single GPU run:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --eval --resume outputs_gal/checkpoint.pth
```
## License
Apache 2.0 license. 
