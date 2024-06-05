# CS166 Final Project: Event Camera Video Reconstruction


# Running with [Anaconda](https://docs.anaconda.com/anaconda/install/)
```
cuda_version=11.8

conda create -y -n firenet python=3.9
conda activate firenet
conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
pip install -r requirements.txt
```

## Inference
Download the pretrained models from [here](https://drive.google.com/file/d/1llzI6hvTwV8dvcXP1nLIkGH5l7Hg41Gn/view?usp=sharing).

You can pick any event videos from [DAVIS](https://rpg.ifi.uzh.ch/davis_data.html) and [HDR](https://rpg.ifi.uzh.ch/E2VID.html). If you download from DAVIS, specify the loader type as Davis. If you download from HDR, specify the loader type as HDR.

To estimate reconstruction:
```
python inference.py --checkpoint_path <path/to/model.pth> --events_file_path </path/to/events> --loader_type <data tpye> --output_folder </path/to/output/dir>
```
For example:
```
python inference.py \
    --checkpoint_path ./checkpoints/checkpoint-epoch970.pth \
    --events_file_path ./Davis/slider_depth \
    --loader_type Davis \
    --output_folder results/inference/slider_depth
```


## Training dataset
You will need to generate the training dataset yourself, using ESIM.
To find out how, please see the [training data generator repo](https://github.com/TimoStoff/esim_config_generator).

Or you can download the dataset [here](https://rpg.ifi.uzh.ch/data/E2VID/datasets/ecoco_depthmaps_test.zip).

## Training
To train a model, you need to create a config file (see `config/config.json` for an example).
In this file, you need to set what model you would like to use, but I only make FireNet work.
You also need to set the training parameters, the training data, the validation data and the output directory.
You can then start the training by invoking

```python train.py --config path/to/config```

If you have a model that would like to keep training from, you can use

```python train.py --config path/to/config --resume /path/to/model.pth```

For example:

```python train.py --config ./config/firenet.json```
