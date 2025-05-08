# FAFormer: Frequency-Analysis-Based Transformer Focusing on Correlation and Specificity for Pansharpening

This repo is the official implementation for FAFormer: Frequency-Analysis-Based Transformer Focusing on Correlation and Specificity for Pansharpening.



## Architecture

![FAFormer](pic/FAFormer.jpg)



## IAB

![IAB](pic/IAB.jpg)



## CFAM

![CFAM](pic/CFAM.jpg)



## SFAM

![SFAM](pic/SFAM.jpg)



## Prerequisites

This environment is mainly based on python=3.6 with CUDA=10.2

```shell
conda create -n faformer python=3.6
conda activate faformer
pip install -r requirements.txt
```



## Test with the pretrained Model

Due to the large size of the dataset, we only provide some samples in './data' to verify the code.

```shell
conda activate faformer
export CUDA_VISIBLE_DEVICES='0';
python main.py -c configs/faformer.py
```

You can modify the config file 'configs/faformer.py' for different purposes.



## Visualize the training process

The training process can be visualized using the TensorBoard library. It is saved under the directory '/runs'.
```shell
tensorboard --logdir=runs/
```



## Useful tools

We have provided several useful tools in the '/tools' folder.



### Preprocessing

The original remote sensing images include large-sized MS images and their corresponding PAN images. We need to generate MS, PAN, and GT images according to the Wald protocol and then cut these images into smaller patches. 
```shell
python tools/handle_raw.py
python tools/clip_patch.py
```



### Visualization

We convert the .tif files to .png files for visualization and merge the multiple cropped patches back into the original large image.
```shell
python tools/visualization.py
```



## Credits

The code of this repository partly relies on [UCGAN](https://github.com/zhysora/UCGAN) and [PanFormer](https://github.com/zhysora/PanFormer). I would like to show my sincere gratitude to authors of them.

