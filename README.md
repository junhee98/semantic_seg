  # Semantic Segmentation for Autonomous Driving

<!-- TOC -->

- [Semantic Segmentation for Autonomous Driving](#semantic-segmentation-for-autonomous-driving)
  - [Requirements](#requirements)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Training](#training)
  - [Inference](#inference)
  - [Code structure](#code-structure)

<!-- /TOC -->

This repo contains a PyTorch an implementation of semantic segmentation models.
We refer to the original code ([link](https://github.com/yassouali/pytorch-segmentation)) which implement basic SegNet.

This repository is developed by [Taesoo Kim](https://github.com/kimtaesu24), and [Junhee Lee](https://github.com/junhee98).
***
## Requirements

We use Ubuntu 20.04. This project requires Python 3.8 and the following Python libraries:
- torch == 1.13.0
- numpy == 1.23.3
- tqdm == 4.32.2
- tensorboard == 1.15.0
- Pillow == 6.2.0
- opencv-python

To install the pacakges used in this repository, type the following command at the project root:
```bash
pip install -r requirements.txt
```
***
## Models 
- (**SegNet**) Basic SegNet 
  - A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [[Paper]](https://arxiv.org/pdf/1511.00561)
- (**SegNet_L_Res**) Basic SegNet + Local residual (intra VGGNet block)
- (**SegNet_G_Res**) Basic SegNet + Local residual (intra VGGNet block) + Global residual (inter Encoder-Decoder)
- (**SegNet_CReLU**) Basic SegNet + Local residual + Global residual + CReLU (replace ReLU of Conv1,3,5 in Encoder)
  - Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units (2016): [[Paper]](https://arxiv.org/pdf/1603.05201v2)
- (**SegNet_Light_CReLU**) Remove some part of SegNet_CReLU's encoder block
- (**SegNet_LRR**) 
  - Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation (2016): [[Paper]](https://arxiv.org/pdf/1605.02264)
***
## Datasets

- **CityScapes:** download from [here](https://www.cityscapes-dataset.com/)
- **CamVid:** download from [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
- **BDD10k:** download from [here](https://www.bdd100k.com/)
- **Mapillary Vistas:** download from [here](https://www.mapillary.com/)
- **Ours (Jeonju):** if you need, please contact us. (without labeling)

<img width="30%" src="https://user-images.githubusercontent.com/80094752/208242173-82a1dc78-152e-4b5d-8708-022a2b0eb77d.jpeg"/><img width="67.1%" src="https://user-images.githubusercontent.com/80094752/208242378-56d43f77-35bf-4213-934e-aebc8a74e7c7.png"/>

### Dataset class

- we modify all classes to the following palette:
<p align="center"><img src="https://user-images.githubusercontent.com/80094752/208243773-48deb9d5-5597-40fb-88b8-f8d553d0d7d7.png" align="center" width="550"></p>


***

## Training
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:

```bash
python train.py --config config.json
```

The training will automatically be run on the GPUs (if more that one is detected and  multipple GPUs were selected in the config file, `torch.nn.DataParalled` is used for multi-gpu training), if not the CPU is used. The log files will be saved in `saved\runs` and the `.pth` chekpoints in `saved\`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```

<p align="center"><img src="https://user-images.githubusercontent.com/80094752/208242693-300faf6e-5b9f-4ed1-b6b0-07a2dd781995.png" align="center" width="900"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/80094752/208242673-a3d511e6-9b45-4b44-a1bf-582ed8b6f81c.png" align="center" width="900"></p>

- Note: if you want to train & test SegNet_Light_CReLU, please replace the comment in SegNet_CReLU.py (from *CreLU* to *Light CReLU*) and use it with *config3-1.json*
***
## Inference

For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --config config.json --model best_model.pth --images images_folder
```

Here are the parameters availble for inference:
```
--output       The folder where the results will be saved (default: outputs).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--config       The config file used for training the model.
```

**Pre-Trained Model:**

- Cityscapes Dataset

| Model              | Backbone |  val mIoU  | inference time |              Pretrained Model              |
|:-------------------|:--------:|:----------:|:--------------:|:------------------------------------------:|
| SegNet (baseline)  |  VGG16   |   82.62%   |    7.15it/s    |        saved/SegNet/best_model.pth         |
| SegNet_L_Res       |  VGG16   |   83.99%   |    6.99it/s    |     saved/SegNet_L_Res/best_model.pth      |
| SegNet_G_Res       |  VGG16   |   85.73%   |    6.68it/s    |     saved/SegNet_G_Res/best_model.pth      |
| SegNet_CReLU       |  VGG16   | **85.99%** |    6.44it/s    | saved/SegNet_CReLU/original/best_model.pth |
| SegNet_Light_CReLU |  VGG16   |   85.37%   |  **7.05it/s**  |  saved/SegNet_CReLU/light/best_model.pth   |
| SegNet_LRR         |  VGG16   |   84.77%   |    6.06it/s    |      saved/SegNet_LRR/best_model.pth       |

- Integrated Datasets (CityScapes, BDD10k, Mapillary Vistas, CamVid) -- in progress!

| Model              | Backbone |  val mIoU  |         Pretrained Model          |
|:-------------------|:--------:|:----------:|:---------------------------------:|
| SegNet_Light_CReLU |  VGG16   |     %      | saved/SegNet_CReLU/best_model.pth |

***
## Demo

- demo.py: Make video by simply running:
```bash
python demo.py --ip images_folder --op save_folder 
```
- demo2.py: Check the realtime inference time by simply running:
```bash
python demo2.py --config config.json --model best_model.pth --images images_folder
```

***
## Code structure

  ```
  autonomous-segmentation/
  │
  ├── train.py - main script to start training
  ├── inference.py - inference using a trained model
  ├── trainer.py - the main trained
  ├── demo.py - make inferenced video with inferenced outputs
  ├── demo2.py - show realtime inference and visualization
  ├── config.json - holds configuration for training
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │   ├── segnet.py
  │   ├── segnet_l_res.py
  │   ├── segnet_g_res.py
  │   ├── segnet_crelu.py
  │   ├── segnet_light_crelu.py 
  │   └── segnet_lrr.py
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```
