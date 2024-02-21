from __future__ import print_function, division
from timm import create_model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torchmetrics.classification import Accuracy, F1Score
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import timm
import PIL
from collections import Counter
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchvision.utils import save_image

# class Dog():
#     def __init__(self, name):
#         self.name = name
        
#     def rename_dog(self, name):
#         print(f"Rename dog {self.name} to {name}")
#         self.name = name
        
# dog1 = Dog("Hund1")
# dog1.rename_dog("pluto")
# dog1.name = "jupiter"
# dog1()

# class Pomoranian(Dog):
#     def __init__(self, name, fellbeschaffenheit):
#         super().__init__(name)
#         self.fellbeschaffenheit = fellbeschaffenheit
    
#     def print_my_pomodings(self):
#         print(f"Name: {self.name} fell: {self.fellbeschaffenheit}")
        
#     def printsomething():
#         print("Hi, I am a pomodingens class inheriting the dog class")
        
#     def __call__(self):
#         print("Hi, my name is ", self.name)
        
# Pomoranian.printsomething()

# class Model():
#     def __init__(self):
#         self.encoder = nn.Linear()
#         self.fc = nn.Linear()
    
#     def forward(self,input):
#         encoder_output = self.encoder(input)
        
#         return self.fc(encoder_output)
    
#     def __call__(self, input):
#         return self.forward(input)

# model = Model()
# model.forward(myimage)
# model(myimage)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(518),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# early_stopper = EarlyStopping(
#     depth=5,
#     ignore=20,
#     method='consistency'
# )


data_dir = r"/home/emylou/four_label_classification"
num_classes= 4

image_datasets = {
    "train": datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"]),
    "test": datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms["test"])
}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #=Nondels.ResNet18_Weights.DEFAULT) #weights=None
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 4)
# model.fc = nn.Sequential(*[nn.Linear(num_ftrs,512),nn.PReLU(),nn.Linear(512, 4)])


# define the LightningModule
class LitEmylouModel(L.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate = 1e-3, false_prediction_save_dir="false_predictions", test_images=None, batch_size=64):    #muss hier noch mehr hin?  def __init__(self, learning_rate, another_parameter, *args, **kwargs):
        super().__init__()
        self.model=model
        self.train_accuracy = Accuracy(task = "multiclass", num_classes = num_classes)
        self.validation_accuracy = Accuracy(task = "multiclass", num_classes = num_classes)
        self.test_accuracy = Accuracy(task = "multiclass", num_classes = num_classes)
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        
        self.false_prediction_save_dir = false_prediction_save_dir
        self.test_images = test_images
        
        os.makedirs(false_prediction_save_dir,exist_ok=True)
        
        #multi_class_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.binary_f1_val = F1Score(num_classes=num_classes, average="macro", task="multiclass")
        #self.binary_f1_val = F1Score(num_classes=num_classes, average="macro", task="multiclass")
        self.micro_f1_val = F1Score(num_classes=num_classes, average="micro", task="multiclass")
        self.classwise_f1_val = F1Score(num_classes=num_classes, average="none", task="multiclass")
        self.weighted_f1_val = F1Score(num_classes=num_classes, average="weighted", task="multiclass") 
        self.binary_f1 = F1Score(num_classes=num_classes, average="macro", task="multiclass")
        self.micro_f1 = F1Score(num_classes=num_classes, average="micro", task="multiclass")
        self.classwise_f1 = F1Score(num_classes=num_classes, average="none", task="multiclass")
        self.weighted_f1 = F1Score(num_classes=num_classes, average="weighted", task="multiclass")      
        
        self.save_hyperparameters(ignore=['model', 'test_images'])

    def training_step(self, batch, batch_idx): 
        images, labels = batch
        output = model(images)
        loss = nn.functional.cross_entropy(output, labels)#, weight =  class_weights.to(output))   
        self.train_accuracy(output, labels)
        self.log("train_loss", loss, prog_bar= True, on_step=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar= True, on_step=True)
        
        return loss  
    
    def validation_step(self, batch, batch_idx): 
        images, labels = batch
        output = model(images)
        loss = nn.functional.cross_entropy(output, labels)   
        
        self.validation_accuracy(output, labels)
        self.binary_f1_val(output, labels)
        self.micro_f1_val(output, labels)
        self.classwise_f1_val(output, labels)
        self.weighted_f1_val(output, labels)
        
        self.log("validation_accuracy", self.validation_accuracy, on_epoch=True, on_step=True, prog_bar=True) #prog_bar= True)        
        self.log("val_macro_f1", self.binary_f1_val, on_epoch=True )
        self.log("val_micro_f1", self.micro_f1_val, on_epoch=True ) 
        for index, val_f1_value in enumerate(self.classwise_f1_val.compute().tolist()):
            self.log(f"val_f1_{index}", val_f1_value, on_epoch=True)
        self.log('val_weighted_f1', self.weighted_f1_val , on_epoch=True )
        
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
    
    
    def test_step(self, batch, batch_idx): 
        images, labels = batch
        output = model(images)
        
        pred_correct = torch.argmax(output, dim=1) == labels
        
        for index, pred in enumerate(pred_correct):
            if pred == 0:
                # 64 is the batch size, please adjust accordingly or make it variable here
                test_image_index = batch_idx*self.batch_size+index
                assert labels[index] == self.test_images[test_image_index][1]
                
                print(f"Image {test_image_index} - {self.test_images[test_image_index][0]} was falsely predicted.")
                save_image(images[index], os.path.join(self.false_prediction_save_dir, f"{str(self.test_images[test_image_index][0]).split('/')[-1]}_pred-{torch.argmax(output[index]).item()}.png"))
                

        
        loss = nn.functional.cross_entropy( output, labels)   
        self.test_accuracy(output, labels)
        self.binary_f1(output, labels)
        self.micro_f1(output, labels)
        self.classwise_f1(output, labels)
        self.weighted_f1(output, labels)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True)     
        self.log("test_macro_f1", self.binary_f1,on_epoch=True)
        self.log("test_micro_f1", self.micro_f1, on_epoch=True ) 
        for index, test_f1_value in enumerate(self.classwise_f1.compute().tolist()):
            self.log(f"test_f1_{index}", test_f1_value, on_epoch=True)
        self.log('weighted_f1', self.weighted_f1 , on_epoch=True )
        
        self.log("test_loss", loss, on_epoch=True)
           
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-6)
        return optimizer

#litmodel = LitEmylouModel(model)

batch_size = 42  #64
num_workers = 8
num_folds = 5
torch.set_float32_matmul_precision('high')

kf = KFold(n_splits=num_folds, shuffle=True)
for fold, (train_index, val_index) in enumerate(kf.split(image_datasets['train'])):
    print(20*"-")
    print(f"Fold {fold}")
    #torch.set_float32_matmul_precision('medium' | 'high')
   
    
    train_subset = Subset(image_datasets['train'], train_index)
    val_subset = Subset(image_datasets['train'], val_index)
    
    label_counter = Counter()
    
    trainval_list = train_subset.dataset.imgs
    
    train_subset_list = [trainval_list[idx] for idx in train_index if idx < len(trainval_list)]

    
    for data_sample in train_subset_list:
        label = data_sample[1]  # Assuming data_sample returns (image, label)
        label_counter[label] += 1
        

    # Calculate weights based on class imbalance
    total_samples = len(train_subset)
    
    
    weights = [1.0 / label_counter[data_sample[1]] for data_sample in train_subset_list]
    
    class_weights = torch.softmax(torch.Tensor([1.0/(label_counter[class_count]/total_samples) for class_count in label_counter]), dim=0)
    # Normalize weights
    sum_weights = sum(weights)
    weights = [weight / sum_weights for weight in weights]
    #breakpoint()

    sampler = torch.utils.data.WeightedRandomSampler( weights, num_samples = 6000, replacement=True, generator=None)
        # Create data loaders for the subsets
    train_loader = DataLoader(train_subset, batch_size=batch_size,shuffle=False, num_workers=num_workers, sampler = sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #, persistent_workers= True)
    
    # print("manual val sanity check")
    
    # for images, labels in tqdm(val_loader):
    #     pass
    
    testloader = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    dataloader = {
        "train": train_loader,
        "val": val_loader,
        "test": testloader
    }
    
    # dino_weights= torch.load(r"/home/emylou/dinov2_vits14_reg4_pretrain.pth" )
    # model = timm.create_model(model_name = 'vit_small_patch14_dinov2.lvd142m', pretrained= True, num_classes = num_classes)
    # model.train()
    
    logger = CSVLogger(save_dir="classifier_results", name= f"Fold_{fold}")
    
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    for transformer_block in model.blocks[-4:]:
        for param in transformer_block.parameters():
            param.requires_grad = True
        
    for param in model.norm.parameters():
        param.requires_grad = True
        
    model.head = nn.Sequential(*[nn.Linear(384,4),nn.Dropout(0.2)])
    
    # breakpoint()
    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) #=Nondels.ResNet18_Weights.DEFAULT) #weights=None
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 4)
    
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #=Nondels.ResNet18_Weights.DEFAULT) #weights=None
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(*[nn.Linear(num_ftrs,512),nn.PReLU(),nn.Linear(512, 4)])
    
    litmodel = LitEmylouModel(model, test_images=testloader.dataset.__dict__['imgs'], false_prediction_save_dir="training_190220241233_false_classified", batch_size=batch_size)
    
    os.makedirs("classifier_results", exist_ok=True)
    
    
    trainer = L.Trainer(max_epochs=500, accelerator="gpu", default_root_dir= 'classifier_results', callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=True)], logger =logger) #, profiler="simple")
    trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders= val_loader)
    # print("Done fitting")

    trainer.test(litmodel, testloader)
    
    ###################################
    #das ist neu
    ###################################
    
  
    
    #model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
    #litmodel_checkpoint =  LitEmylouModel.load_from_checkpoint('/home/emylou/classifier_results/checkpoint.ckpt')
   # litmodel_checkpoint.eval()
    
    # predict with the model
    #y_hat = model(x)
    
    
    
    exit()


