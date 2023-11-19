---
title: YuzuMarker.FontDetection
emoji: 😅
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---

<div align="center">
<h1>✨YuzuMarker.FontDetection✨</h1>
<p>First-ever CJK (Chinese, Japanese, Korean) font recognition model</p>
<p>
    <a href="https://huggingface.co/spaces/gyrojeff/YuzuMarker.FontDetection"><img alt="Click here for Online Demo" src="https://img.shields.io/badge/🤗-Open%20In%20Spaces%20(Online Demo)-blue.svg"/></a>
    <img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/JeffersonQin/YuzuMarker.FontDetection"/>
    <img alt="License" src="https://img.shields.io/github/license/JeffersonQin/YuzuMarker.FontDetection"/>
    <img alt="Contributors" src="https://img.shields.io/github/contributors/JeffersonQin/YuzuMarker.FontDetection"/>
</p>
<a href="https://www.buymeacoffee.com/gyrojeff" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
</div>

## News

- **[Update 2023/05/05] Project recommended on ruanyifeng.com (阮一峰的网络日志 - 科技爱好者周刊): https://www.ruanyifeng.com/blog/2023/05/weekly-issue-253.html**
- **[Update 2023/11/18] Dataset is now opensource! Download on huggingface from here: https://huggingface.co/datasets/gyrojeff/YuzuMarker.FontDetection/tree/master**

## Scene Text Font Dataset Generation

This repository also contains data for automatically generating a dataset of scene text images with different fonts. The dataset is generated using the [CJK font pack by VCB-Studio](https://vcb-s.com/archives/1114) and thousands of background image from [pixiv.net](https://pixiv.net).

The pixiv data will not be shared since it is just randomly scraped. You may prepare your own background dataset that would fit your data distribution as you like.

For the text corpus,

* Chinese is randomly generated from [3500 common Chinese characters](https://gist.github.com/simongfxu/13accd501f6c91e7a423ddc43e674c0f).
* Japanese is randomly generated from a list of lyrics from https://www.uta-net.com.
* Korean is randomly generated from its alphabet.

All text are also mixed with English text to simulate real-world data.

### Data Preparation Walkthrough

1. Download the CJK font pack and extract it to the `dataset/fonts` directory.
2. Prepare the background data and put them in the `dataset/pixivimages` directory.
3. Run following script to clean the file names
   ```bash
   python dataset_filename_preprocess.py
   ```

### Generation Script Walkthrough

Now the preparation is complete. The following command can be used to generate the dataset:

```bash
python font_ds_generate_script.py 1 1
```

Note that the command is followed by two parameters. The second one is to split the task into multiple partitions, and the first one is the index of the partitioned task to run. For example, if you want to run the task in 4 partitions, you can run the following commands in parallel to speed up the process:

```bash
python font_ds_generate_script.py 1 4
python font_ds_generate_script.py 2 4
python font_ds_generate_script.py 3 4
python font_ds_generate_script.py 4 4
```

The generated dataset will be saved in the `dataset/font_img` directory.

Note that `batch_generate_script_cmd_32.bat` and `batch_generate_script_cmd_64.bat` are batch scripts for Windows that can be used to generate the dataset in parallel with 32 partitions and 64 partitions.

### Final Check

Since the task might be terminated unexpectedly or deliberately by user. The script has a caching mechanism to avoid re-generating the same image.

In this case, the script might not be able to detect corruption in cache (might be caused by terminating when writing to files) during this task, thus we also provides a script checking the generated dataset and remove the corrupted images and labels.

```bash
python font_ds_detect_broken.py
```

After running the script, you might want to rerun the generation script to fill up the holes of the removed corrupted files.

### (Optional) Linux Cluster Generation Walkthrough

If you would like to run the generation script on linux clusters, we also provides the environment setup script `linux_venv_setup.sh`.

The prerequisite is that you have a linux cluster with `python3-venv` installed and `python3` is available in the path.

To setup the environment, run the following command:

```bash
./linux_venv_setup.sh
```

The script will create a virtual environment in the `venv` directory and install all the required packages. The script is required in most cases since the script will also install `libraqm` which is required for the text rendering of PIL and is often not installed by default in most linux server distributions.

After the environment is setup, you might compile a task scheduler to deploy generation task in parallel.

The main idea is similar to the direct usage of the script, except that here we accept three parameters,

* `TOTAL_MISSION`: the total number of partitions of the task
* `MIN_MISSION`: the minimum partition index of the task to run
* `MAX_MISSION`: the maximum partition index of the task to run

and the compilation command is as following:

```bash
gcc -D MIN_MISSION=<MIN_MISSION> \
    -D MAX_MISSION=<MAX_MISSION> \
    -D TOTAL_MISSION=<TOTAL_MISSION> \
    batch_generate_script_linux.c \
    -o <object-file-name>.out
```

For example if you want to run the task in 64 partitions, and want to spilit the work on 4 machines, you can compile the following command on each machine:

```bash
# Machine 1
gcc -D MIN_MISSION=1 \
    -D MAX_MISSION=16 \
    -D TOTAL_MISSION=64 \
    batch_generate_script_linux.c \
    -o mission-1-16.out
# Machine 2
gcc -D MIN_MISSION=17 \
    -D MAX_MISSION=32 \
    -D TOTAL_MISSION=64 \
    batch_generate_script_linux.c \
    -o mission-17-32.out
# Machine 3
gcc -D MIN_MISSION=33 \
    -D MAX_MISSION=48 \
    -D TOTAL_MISSION=64 \
    batch_generate_script_linux.c \
    -o mission-33-48.out
# Machine 4
gcc -D MIN_MISSION=49 \
    -D MAX_MISSION=64 \
    -D TOTAL_MISSION=64 \
    batch_generate_script_linux.c \
    -o mission-49-64.out
```

Then you can run the compiled object file on each machine to start the generation task.

```bash
./mission-1-16.out # Machine 1
./mission-17-32.out # Machine 2
./mission-33-48.out # Machine 3
./mission-49-64.out # Machine 4
```

There is also another helper script to check the progress of the generation task. It can be used as following:

```bash
python font_ds_stat.py
```

### MISC Info of the Dataset

The generation is CPU bound, and the generation speed is highly dependent on the CPU performance. Indeed the work itself is an engineering problem.

Some fonts are problematic during the generation process. The script has an manual exclusion list in `config/fonts.yml` and also support unqualified font detection on the fly. The script will automatically skip the problematic fonts and log them for future model training.

## Model Training

Have the dataset ready under the `dataset` directory, you can start training the model. Note that you can have more than one folder of dataset, and the script will automatically merge them as long as you provide the path to the folder by command line arguments.

```bash
$ python train.py -h
usage: train.py [-h] [-d [DEVICES ...]] [-b SINGLE_BATCH_SIZE] [-c CHECKPOINT] [-m {resnet18,resnet34,resnet50,resnet101,deepfont}] [-p] [-i] [-a {v1,v2,v3}]
                [-l LR] [-s [DATASETS ...]] [-n MODEL_NAME] [-f] [-z SIZE] [-t {medium,high,heighest}] [-r]

optional arguments:
  -h, --help            show this help message and exit
  -d [DEVICES ...], --devices [DEVICES ...]
                        GPU devices to use (default: [0])
  -b SINGLE_BATCH_SIZE, --single-batch-size SINGLE_BATCH_SIZE
                        Batch size of single device (default: 64)
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Trainer checkpoint path (default: None)
  -m {resnet18,resnet34,resnet50,resnet101,deepfont}, --model {resnet18,resnet34,resnet50,resnet101,deepfont}
                        Model to use (default: resnet18)
  -p, --pretrained      Use pretrained model for ResNet (default: False)
  -i, --crop-roi-bbox   Crop ROI bounding box (default: False)
  -a {v1,v2,v3}, --augmentation {v1,v2,v3}
                        Augmentation strategy to use (default: None)
  -l LR, --lr LR        Learning rate (default: 0.0001)
  -s [DATASETS ...], --datasets [DATASETS ...]
                        Datasets paths, seperated by space (default: ['./dataset/font_img'])
  -n MODEL_NAME, --model-name MODEL_NAME
                        Model name (default: current tag)
  -f, --font-classification-only
                        Font classification only (default: False)
  -z SIZE, --size SIZE  Model feature image input size (default: 512)
  -t {medium,high,heighest}, --tensor-core {medium,high,heighest}
                        Tensor core precision (default: high)
  -r, --preserve-aspect-ratio-by-random-crop
                        Preserve aspect ratio (default: False)
```

## Font Classification Experiment Results

On our synthesized dataset,

| Backbone | Data Aug | Pretrained | Crop<br>Text<br>BBox | Preserve<br>Aspect<br>Ratio | Output<br>Norm | Input Size | Hyper<br>Param | Accur | Commit | Dataset | Precision |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:| :-: |
| DeepFont | ✔️* | ❌ | ✅ | ❌ | Sigmoid | 105x105 | I<sup>1</sup> | [Can't Converge] | 665559f | I<sup>5</sup> | bfloat16_3x |
| DeepFont | ✔️* | ❌ | ✅ | ❌ | Sigmoid | 105x105 | IV<sup>4</sup> | [Can't Converge] | 665559f | I | bfloat16_3x |
| ResNet-18 | ❌ | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 18.58% | 5c43f60 | I | float32 |
| ResNet-18 | ❌ | ❌ | ❌ | ❌ | Sigmoid | 512x512 | II<sup>2</sup> | 14.39% | 5a85fd3 | I | bfloat16_3x |
| ResNet-18 | ❌ | ❌ | ❌ | ❌ | Tanh | 512x512 | II | 16.24% | ff82fe6 | I | bfloat16_3x |
| ResNet-18 | ✅*<sup>8</sup> | ❌ | ❌ | ❌ | Tanh | 512x512 | II | 27.71% | a976004 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Tanh | 512x512 | I | 29.95% | 8364103 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 29.37% [Early stop] | 8d2e833 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 416x416 | I | [Lower Trend] | d5a3215 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 320x320 | I | [Lower Trend] | afcdd80 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 224x224 | I | [Lower Trend] | 8b9de80 | I | bfloat16_3x |
| ResNet-34 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 32.03% | 912d566 | I | bfloat16_3x |
| ResNet-50 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 34.21% | e980b66 | I | bfloat16_3x |
| ResNet-18 | ✅* | ✅ | ❌ | ❌ | Sigmoid | 512x512 | I | 31.24% | 416c7bb | I | bfloat16_3x |
| ResNet-18 | ✅* | ✅ | ✅ | ❌ | Sigmoid | 512x512 | I | 34.69% | 855e240 | I | bfloat16_3x |
| ResNet-18 | ✔️*<sup>9</sup> | ✅ | ✅ | ❌ | Sigmoid | 512x512 | I | 38.32% | 1750035 | I | bfloat16_3x |
| ResNet-18 | ✔️* | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III<sup>3</sup> | 38.87% | 0693434 | I | bfloat16_3x |
| ResNet-50 | ✔️* | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III | 48.99% | bc0f7fc | II<sup>6</sup> | bfloat16_3x |
| ResNet-50 | ✔️ | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III | 48.45% | 0f071a5 | II | bfloat16_3x |
| ResNet-50 | ✔️ | ✅ | ✅ | ✅<sup>11</sup> | Sigmoid | 512x512 | III | 46.12% | 0f071a5 | II | bfloat16 |
| ResNet-50 | ❕<sup>10</sup> | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III | 43.86% | 0f071a5 | II | bfloat16 |
| ResNet-50 | ❕ | ✅ | ✅ | ✅ | Sigmoid | 512x512 | III | 41.35% | 0f071a5 | II | bfloat16 |

* \* Bug in implementation
* <sup>1</sup> `learning rate = 0.0001, lambda = (2, 0.5, 1)`
* <sup>2</sup> `learning rate = 0.00005, lambda = (4, 0.5, 1)`
* <sup>3</sup> `learning rate = 0.001, lambda = (2, 0.5, 1)`
* <sup>4</sup> `learning rate = 0.01, lambda = (2, 0.5, 1)`
* <sup>5</sup> Initial version of synthesized dataset
* <sup>6</sup> Doubled synthesized dataset (2x)
* <sup>7</sup> Quadruple synthesized dataset (4x)
* <sup>8</sup> Data Augmentation v1: Color Jitter + Random Crop [81%-100%]
* <sup>9</sup> Data Augmentation v2: Color Jitter + Random Crop [30%-130%] + Random Gaussian Blur + Random Gaussian Noise + Random Rotation [-15°, 15°]
* <sup>10</sup> Data Augmentation v3: Color Jitter + Random Crop [30%-130%] + Random Gaussian Blur + Random Gaussian Noise + Random Rotation [-15°, 15°] + Random Horizontal Flip + Random Downsample [1, 2]
* <sup>11</sup> Preserve Aspect Ratio by Random Cropping

## Pretrained Models

Available at: https://huggingface.co/gyrojeff/YuzuMarker.FontDetection/tree/main

Note that since I trained everything on pytorch 2.0 with `torch.compile`, if you want to use the pretrained model you would need to install pytorch 2.0 and compile it with `torch.compile` as in `demo.py`.

## Demo Deployment (Method 1)

To deploy the demo, you would need either the whole font dataset under `./dataset/fonts` or a cache file indicating fonts of model called `font_demo_cache.bin`. This will be later released as resource.

To deploy, first run the following script to generate the demo font image (if you have the fonts dataset):

```bash
python generate_font_sample_image.py
```

then run the following script to start the demo server:

```bash
$ python demo.py -h
usage: demo.py [-h] [-d DEVICE] [-c CHECKPOINT] [-m {resnet18,resnet34,resnet50,resnet101,deepfont}] [-f] [-z SIZE] [-s] [-p PORT] [-a ADDRESS]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        GPU devices to use (default: 0), -1 for CPU
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Trainer checkpoint path (default: None). Use link as huggingface://<user>/<repo>/<file> for huggingface.co models, currently only supports model file in the root
                        directory.
  -m {resnet18,resnet34,resnet50,resnet101,deepfont}, --model {resnet18,resnet34,resnet50,resnet101,deepfont}
                        Model to use (default: resnet18)
  -f, --font-classification-only
                        Font classification only (default: False)
  -z SIZE, --size SIZE  Model feature image input size (default: 512)
  -s, --share           Get public link via Gradio (default: False)
  -p PORT, --port PORT  Port to use for Gradio (default: 7860)
  -a ADDRESS, --address ADDRESS
                        Address to use for Gradio (default: 127.0.0.1)
```

## Demo Deployment (Method 2)

If docker is available on your machine, you can deploy directly by docker as how I did for huggingface space.

You may follow the command line argument provided in the last section to change the last line of the `Dockerfile` to accomodate your needs.

Build the docker image:

```bash
docker build -t yuzumarker.fontdetection .
```

Run the docker image:

```bash
docker run -it -p 7860:7860 yuzumarker.fontdetection
```

## Online Demo

The project is also deployed on Huggingface Space: https://huggingface.co/spaces/gyrojeff/YuzuMarker.FontDetection

## Related works and Resources

* DeepFont: Identify Your Font from An Image: https://arxiv.org/abs/1507.03196
* Font Identification and Recommendations: https://mangahelpers.com/forum/threads/font-identification-and-recommendations.35672/
* Unconstrained Text Detection in Manga: a New Dataset and Baseline: https://arxiv.org/pdf/2009.04042.pdf
* SwordNet: Chinese Character Font Style Recognition Network: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9682683

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JeffersonQin/YuzuMarker.FontDetection&type=Date)](https://star-history.com/#JeffersonQin/YuzuMarker.FontDetection&Date)
