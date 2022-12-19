import os 
import argparse
import shutil
import scipy
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

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, mask, output_path, image_file, palette):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    #colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    '''output_im = Image.new('RGB', (w*2, h))
    output_im.paste(image, (0,0))
    output_im.paste(colorized_mask, (w,0))
    output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    mask_img = Image.fromarray(mask, 'L')
    mask_img.save(os.path.join(output_path, image_file+'.png'))'''
    return colorized_mask

def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K']
    if dataset_type == 'CityScapes': 
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]        
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    #model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    model = torch.load(args.model, map_location=device)
    '''if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        print('during training')
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)'''
    #for param_tensor in model.state_dict(checkpoint):
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)  
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
            
            '''if args.mode == 'multiscale':
                prediction = multi_scale_predict(model, input, scales, num_classes, device)
            elif args.mode == 'sliding':
                prediction = sliding_predict(model, input, num_classes)'''
            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            colorized_mask = save_images(image, prediction, args.output, img_file, palette)
            
            output_image = [np.array(image),np.array(colorized_mask.convert('RGB'))]
            output_image[0] = cv2.cvtColor(output_image[0],cv2.COLOR_BGR2RGB)
            output_image[1] = cv2.cvtColor(output_image[1],cv2.COLOR_BGR2RGB)
            
            cv2.imshow("Image",np.hstack(output_image))

            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
            

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    '''parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')'''
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    '''parser.add_argument('-ip', '--input_path', default=None,type=str,
                        help='Path to the Image directory')
    parser.add_argument('-op', '--output_path', default=None,type=str,
                        help='Path to the Image directory')'''
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

'''def main(input_path, output_path):

    sub_infer = 'infer'
    sub_output = 'outputs'

    input_list = sorted(os.listdir(input_path))
    output_list = sorted(os.listdir(os.path.join(output_path,sub_infer)))

    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)    

    if not os.path.exists(os.path.join(output_path,sub_output)):
        os.makedirs(os.path.join(output_path,sub_output))


    #Phase1 : input image & infer image concat / visualization 
    print("Phase1")
    for input in tqdm(input_list):
        input_image = cv2.imread(os.path.join(input_path,input),cv2.IMREAD_COLOR)
        infer_image = inference(input)
        output_image = [input_image,infer_image]
        cv2.imshow("Image",np.hstack(output_image))
        #cv2.imwrite(os.path.join(output_path,sub_output,infer),np.hstack(output_image))

        if cv2.waitKey(1) == 27:
            break
    
    h, w, _ = input_image.shape

    cv2.destroyAllWindows()


    #Phase2 : concat image to Video
    print("Phase2")
    frame_size = (int(w*2),h)

    output_list = sorted(os.listdir(os.path.join(output_path,sub_output)))

    out = cv2.VideoWriter(os.path.join(output_path,'result.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 30, frame_size)

    for list in tqdm(output_list):
        output = cv2.imread(os.path.join(output_path,sub_output,list),cv2.IMREAD_COLOR)
        out.write(output)

    out.release()

    #Phase3 : remove dummy directory
    print("Phase3")
    shutil.rmtree(os.path.join(os.path.join(output_path,sub_output)))
    print("Done!")
'''