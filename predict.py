import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import seaborn as sns
from torchvision import datasets, transforms, models
import argparse
import json
from PIL import Image
import time
import argparse
from train import check_gpu


def args():
    parser = argparse.ArgumentParser(description='Image Classifier Application')
    parser.add_argument('--image', type=str, help='define the iamge directory', required = True)
    parser.add_argument('--checkpoint', type=str, help='define the checkpoint file', required = True)
    parser.add_argument('--categories_names', type=str, help='categories to real names')
    parser.add_argument('--top_k', type=int, help='define the top k highest values to show in prediction')
    parser.add_argument('--gpu', type=str, help='Use GPU for training')
    
    results = parser.parse_args()
    return results

def checkpoint_loading(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['architecture'] == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        exec("model = models.{}(pretrained = True)".checkpoint['architecture'])
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['train_indices']
    
    
    return model
def process_image(image_path):
    img = Image.open(image_path)
    width = img.size[0]
    height = img.size[1]
    if width<height:
        resize_measures = [256, 256 * height/width]
    else:
        resize_measures = [256 * width/height, 256]
        
    img.thumbnail(size = resize_measures)
    
    width = img.size[0]
    height = img.size[1]
    
    center = width/2, height/2
    left, top, right, bottom = center[0]-(224/2), center[1]-(224/2), center[0]+(224/2), center[1]+(224/2)
    img = img.crop((left, top, right, bottom))
        
    np_img = np.array(img)/255
    
    img_means = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    
    np_img = (np_img - img_means)/img_std
    
    np_img = np_img.transpose(2, 0, 1)
    
    return np_img

def predict(image_path, model, cat_to_name, top_k, device):
    if type(top_k) == type(None):
        top_k = 5
        print("model uses top 5 highest values in prediction")
    
    model.to(device)
    model.eval()
    image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
    
    output = model.forward(image)
    
    ps = torch.exp(output)
    top_p, top_classes = ps.topk(top_k, dim = 1)
    
    indeces_to_classes = model.class_to_idx
    
    top_p = np.array(top_p.detach())[0]
    top_classes = np.array(top_classes.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_classes]
    flowers = [cat_to_name[lab] for lab in top_classes]
    
    return top_p, top_classes, flowers
def probabilities(top_probs, top_flowers):
    for i, j in enumerate(zip(top_flowers, top_probs)):
        print(f"flower/probability : {i}/{j}")
        
# combining all functions:
def main():
    arg = args()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
        model = checkpoint_loading(arg.checkpoint)
    
        image = process_image(arg.image)
        device = check_gpu(arg.gpu)
        top_p, top_classes, top_flowers = predict(arg.image, model,cat_to_name,arg.top_k, device)
    
        probabilities(top_p, top_flowers)
   
if __name__ == '__main__': main()
    
        

