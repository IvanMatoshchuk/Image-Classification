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


def args():
    parser = argparse.ArgumentParser(description='Image Classifier Application')
    parser.add_argument('--save_dir', type=str, help='define the save directory(string) for checkpoints')
    parser.add_argument('--arch', type=str, help='choose the pre-trained model from torchvision.models')
    parser.add_argument('--learning_rate', type=float, help='define the learning rate as float')
    parser.add_argument('--hidden_units', type=int, help='define the number of hidden units for model.classifier as int')
    parser.add_argument('--epochs', type=int, help='define the number of epochs for training as int')
    parser.add_argument('--gpu', type=str, help='Use GPU for training')
    
    results = parser.parse_args()
    return results
def train_transformer(train_dir):
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])])
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    return train_dataset

def train_loader(train_dataset):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
    return trainloader

def val_transformer(val_dir):
    val_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])])
    val_dataset = datasets.ImageFolder(val_dir, transform = val_transform)
    return val_dataset

def val_loader(val_dataset):
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size = 64)
    return valloader

def test_transformer(test_dir):
    test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])])
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)
    return test_dataset

def test_loader(test_dataset):
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    return testloader

def check_gpu(gpu):
    if gpu == False:
        return torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("no gpu found, using cpu")
    return device



def model_load(architecture = 'densenet121'):
    if type(architecture) == type(None):
        model = models.densenet121(pretrained=True)
        model.name = 'densenet121'
        print('desenet121 is used as network architecture')
    else: 
        exec("model = models.{}(pretrained+True".format(architecture))
        model.name = architecture
    for param in model.parameters():
        param.requires_grad = False
    return model

def model_classifier(model, hidden_units):
    if type(hidden_units) == type(None):
        hidden_units = 512
        print("model uses 512 hidden units")
    input_features = model.classifier.in_features
              
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_units,102),
                                 nn.LogSoftmax(dim=1))
    return classifier

              
def nn_trainer(Model, Trainloader, Validloader, Device, Criterion, Optimizer, Epochs):
    if type(Epochs) == type(None):
        Epochs = 5
        print("5 epochs will be used in training")
    if Epochs != 0:
        for epoch in range(Epochs):
            running_loss = 0
            for images, labels in Trainloader:
                images, labels = images.to(Device), labels.to(Device)
            
                Optimizer.zero_grad()
            
                output = Model.forward(images)
                loss = Criterion(output, labels)
                loss.backward()
                Optimizer.step()
            
                running_loss += loss.item()
            
                #validation part
            else:
                val_loss = 0
                val_accuracy = 0
                Model.eval()
                with torch.no_grad():
                    for images, labels in Validloader:
                        images, labels = images.to(Device), labels.to(Device)
                        outputs = Model.forward(images)
                        v_loss = Criterion(outputs, labels)
                    
                        val_loss += v_loss.item()
                    
                    #validation accuracy
                        ps = torch.exp(outputs)
                        top_class = ps.topk(1, dim=1)
                        
                        equals = (labels.data == ps.max(dim=1)[1])
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
              
                print(f"Epoch {epoch+1}/{Epochs}.. "
                  f"training loss: {running_loss/len(Trainloader):.3f}.. "
                  f"validation loss: {val_loss/len(Validloader):.3f}.. "
                  f"validation accuracy: {val_accuracy/len(Validloader):.3f}..")
    return Model            
    
              
              
def testing(model, testloader, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0
    test_accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
    
        test_loss += loss.item()
    
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = (labels.data == ps.max(dim=1)[1])
        test_accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    print(f"Test loss: {test_loss/len(testloader):.3f}")
    print(f"Test accuracy: {test_accuracy/len(testloader) * 100:.1f}%")
    

def save_checkpoint(Model, Dir, Train_data):
    if type(Dir) == type(None):
        print("no directory, model will not be saved")
    # TODO: Save the checkpoint 
    torch.save({
            'train_indices': Train_data.class_to_idx,
            'model_state_dict': Model.state_dict(),
            'model_cpu': Model.cpu,
            'model_cuda': Model.cuda,
            'classifier': Model.classifier,
            'architecture': Model.name
            }, 'checkpoint.pth')

# combining all functions
def main():
    arg = args()
    #uploading data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transformer(train_dir)
    val_data = val_transformer(val_dir)
    test_data = test_transformer(test_dir)
        
    trainloader = train_loader(train_data)
    valloader = val_loader(val_data)
    testloader = test_loader(test_data)
    
    #loading model
    model = model_load(architecture = arg.arch)
    #classifier
    model.classifier = model_classifier(model, hidden_units = arg.hidden_units)
    #gpu
    device = check_gpu(arg.gpu)
    model.to(device)
    
    #learn_rate
    if type(arg.learning_rate) == type(None):
        learning_rate = 0.003
        print("learning rate selected as 0.003")
    else:
        learning_rate = arg.learning_rate
    
    #criterion and optimizer 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
     
    #training the model
    start = time.time()
    nn_trainer(model, trainloader, valloader, device, criterion, optimizer, arg.epochs)
    print(f"Time taken: {time.time() - start:.3f} sec")
    
    testing(model, testloader, criterion, device)
    save_checkpoint(model,arg.save_dir,train_data)
    
# run
if __name__ == '__main__': main()