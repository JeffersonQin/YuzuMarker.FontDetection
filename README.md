# YuzuMarker.FontDetection

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

## Font Classification Experiment Results

On our synthesized dataset,

| Backbone | Data Aug | Pretrained | Crop<br>Text<br>BBox | Preserve<br>Aspect<br>Ratio | Output<br>Norm | Input Size | Hyper<br>Param | Accur | Commit | Dataset | Precision |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:| :-: |
| DeepFont | ✔️* | ❌ | ✅ | ❌ | Sigmoid | 105x105 | I<sup>1</sup> | [Can't Converge] | 665559f | I<sup>5</sup> | bfloat16_3x |
| DeepFont | ✔️* | ❌ | ✅ | ❌ | Sigmoid | 105x105 | IV<sup>4</sup> | [Can't Converge] | 665559f | I | bfloat16_3x |
| ResNet-18 | ❌ | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 18.58% | 5c43f60 | I | float32 |
| ResNet-18 | ❌ | ❌ | ❌ | ❌ | Sigmoid | 512x512 | II<sup>2</sup> | 14.39% | 5a85fd3 | I | bfloat16_3x |
| ResNet-18 | ❌ | ❌ | ❌ | ❌ | Tanh | 512x512 | II | 16.24% | ff82fe6 | I | bfloat16_3x |
| ResNet-18 | ✅*<sup>7</sup> | ❌ | ❌ | ❌ | Tanh | 512x512 | II | 27.71% | a976004 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Tanh | 512x512 | I | 29.95% | 8364103 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 29.37% [Early stop] | 8d2e833 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 416x416 | I | [Lower Trend] | d5a3215 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 320x320 | I | [Lower Trend] | afcdd80 | I | bfloat16_3x |
| ResNet-18 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 224x224 | I | [Lower Trend] | 8b9de80 | I | bfloat16_3x |
| ResNet-34 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 32.03% | 912d566 | I | bfloat16_3x |
| ResNet-50 | ✅* | ❌ | ❌ | ❌ | Sigmoid | 512x512 | I | 34.21% | e980b66 | I | bfloat16_3x |
| ResNet-18 | ✅* | ✅ | ❌ | ❌ | Sigmoid | 512x512 | I | 31.24% | 416c7bb | I | bfloat16_3x |
| ResNet-18 | ✅* | ✅ | ✅ | ❌ | Sigmoid | 512x512 | I | 34.69% | 855e240 | I | bfloat16_3x |
| ResNet-18 | ✔️*<sup>8</sup> | ✅ | ✅ | ❌ | Sigmoid | 512x512 | I | 38.32% | 1750035 | I | bfloat16_3x |
| ResNet-18 | ✔️* | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III<sup>3</sup> | 38.87% | 0693434 | I | bfloat16_3x |
| ResNet-50 | ✔️* | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III | 48.99% | bc0f7fc | II<sup>6</sup> | bfloat16_3x |
| ResNet-50 | ✔️ | ✅ | ✅ | ✅<sup>10</sup> | Sigmoid | 512x512 | III | 46.12% | 0f071a5 | II | bfloat16 |
| ResNet-50 | ❕<sup>9</sup> | ✅ | ✅ | ❌ | Sigmoid | 512x512 | III | 43.86% | 0f071a5 | II | bfloat16 |
| ResNet-50 | ❕ | ✅ | ✅ | ✅ | Sigmoid | 512x512 | III | 41.35% | 0f071a5 | II | bfloat16 |

* \* Bug in implementation
* <sup>1</sup> `learning rate = 0.0001, lambda = (2, 0.5, 1)`
* <sup>2</sup> `learning rate = 0.00005, lambda = (4, 0.5, 1)`
* <sup>3</sup> `learning rate = 0.001, lambda = (2, 0.5, 1)`
* <sup>4</sup> `learning rate = 0.01, lambda = (2, 0.5, 1)`
* <sup>5</sup> Initial version of synthesized dataset
* <sup>6</sup> Doubled synthesized dataset
* <sup>7</sup> Data Augmentation v1: Color Jitter + Random Crop [81%-100%]
* <sup>8</sup> Data Augmentation v2: Color Jitter + Random Crop [30%-130%] + Random Gaussian Blur + Random Gaussian Noise + Random Rotation [-15°, 15°]
* <sup>9</sup> Data Augmentation v3: Color Jitter + Random Crop [30%-130%] + Random Gaussian Blur + Random Gaussian Noise + Random Rotation [-15°, 15°] + Random Horizontal Flip + Random Downsample [1, 2]
* <sup>10</sup> Preserve Aspect Ratio by Random Cropping

## Related works and Resources

* Font Identification and Recommendations: https://mangahelpers.com/forum/threads/font-identification-and-recommendations.35672/
* Unconstrained Text Detection in Manga: a New Dataset and Baseline: https://arxiv.org/pdf/2009.04042.pdf
* SwordNet: Chinese Character Font Style Recognition Network: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9682683
