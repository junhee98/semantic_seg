import argparse
import scipy
import os
import numpy as np
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict


def save_images(image, mask, output_path, image_file, palette):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    palette = loader.dataset.palette

    # Model
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    model = torch.load(args.model, map_location=device)
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    image_files = sorted(glob(os.path.join(args.images, f'*.{"png"}')))
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            image = Image.open(img_file).convert('RGB')
            # Scale the smaller side to crop size
            w, h = image.size
            if h < w:
                h, w = (480, int(480 * w / h))
            else:
                h, w = (int(480 * h / w), 480)

            image.thumbnail((w, h))
            input = normalize(to_tensor(image)).unsqueeze(0)
            
            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            save_images(image, prediction, args.output, img_file, palette)
            

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
