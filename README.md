# FreeLesion

## Introduction

FreeLesion is designed for generating lesion masks in fundus image synthesis to enhance fundus lesion segmentation. Through curriculum learning, adaptive lesion resampling, and feature loss-guided filtering, FreeLesion has effectively generated a rich and lesion-balanced synthetic dataset with well-aligned masks and images, successfully addressing the issue of limited availability of fundus lesion datasets.


## Updates


## Requirements

Our code is built upon [Stable Diffusion](https://github.com/CompVis/stable-diffusion). Please clone the repository and set up the environment:
```
git clone https://github.com/jianlufu121/FreeLesion.git
cd FreeLesion
conda env create -f environment.yaml
conda activate freelesion
```

You will also need to download the pre-trained Stable Diffusion model (or manually download it from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)):
```
mkdir models/ldm/stable-diffusion
wget -O models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
```

## Data Preparation


**IDRiD**. The dataset can be downloaded [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip). Please download it and generate two files: `idrid_train.txt` and `idrid_val.txt`, then you should get a directory structure as follows:
```
idrid
    annotations/
        training/
            IDRiD_01.png
            ...
        validation/
            IDRiD_55.png
            ...
    images/
        training/
            IDRiD_01.jpg
            ...
        validation/
            IDRiD_55.jpg
            ...
    idrid_train.txt
    idrid_val.txt
```

## Training

To train FreeLesion, run:
```
python main_class.py --base /path/to/config
               -t
               --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt
               -n <exp_name>
               --gpus 0,
               --data_root /path/to/dataset
               --train_txt_file /path/to/dataset/with/train.txt
               --val_txt_file /path/to/dataset/with/val.txt
```

We provide one training scripts: `train_idrid.sh`. Please modify `--data_root`,  `--train_txt_file`, and `--val_txt_file` according to the actual path.The training for the DDR dataset follows the same scripts as the IDRiD training.



## Generation

To generate images under the ALR setting, run:
```
python ALR.py
python scripts/inference.py --batch_size 8
                      --config /path/to/config
                      --ckpt /path/to/trained_model
                      --dataset <dataset name>
                      --outdir /path/to/output
                      --txt_file /path/to/dataset/with/val.txt
                      --data_root /path/to/dataset
                      --plms 
```
We provide one sampling scripts: `sample_idrid.sh`. Please modify `--ckpt`, `--txt_file`, and `--data_root` according to the actual ALR path.

## Filter

To fiter out the noise in the generated images, run:
```
python Filter/FLGF.py <configs>
                      <checkpoint>
                      --real-img-path <> --real-mask-path <>
                      --syn-img-path <> --syn-mask-path <>
                      --filtered-mask-path <> --cluster_viz-path <>
```
We use the [pre-trained SegFormer-B4 model](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer) to calculate the average loss for each category and the average distance to the cluster center on real images, and then filter out noisy synthetic regions.

## Acknowledgments

Our code borrows heavily from [FreestyleNet](https://essunny310.github.io/FreestyleNet/), [FreeMask](https://github.com/LiheYoung/FreeMask)and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
