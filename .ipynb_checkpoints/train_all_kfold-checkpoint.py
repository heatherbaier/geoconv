import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time

from utils import *
from dataloader import PlanetData
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name', type = str)
    parser.add_argument('model_name', type = str, help = "can be one of: SpA, DeepAll, GeoConv, FC")
    parser.add_argument('--bs', required = False, default = 64, type = int)
    parser.add_argument('--num_epochs', required = False, default = 200, type = int)
    parser.add_argument('--lr', required = False, type = float, default = 0.00005)
    parser.add_argument('--iters', required = False, default = 64, help = "only needed when model is GeoConv, basically bath size")
    parser.add_argument('--sample', required = False, default = 0, type = int)
    parser.add_argument('--num_kfolds', required = False, default = 3)
    parser.add_argument('--labels_path', required = False, default = "/sciclone/geograd/Heather/c1/data/clean/africa_subset_ys.json")
    parser.add_argument('--transforms_path', required = False, default = "/sciclone/geograd/Heather/c1/transform_stats_sc.json")
    parser.add_argument('--results_dir', required = False, default = "/sciclone/geograd/heather_data/spa_reg/")
    parser.add_argument('--device', required = False, default = "cuda")
    parser.add_argument('--use_scheduler', action='store_true', help='')
    parser.add_argument('--postval', action='store_true', help='')
    parser.add_argument('--ts4', action='store_true', help='')
    parser.add_argument('--use_means', action='store_true', help='')
    parser.add_argument('--norm_coords', action='store_true', help='')
    args = parser.parse_args() 
    
    print(args)
    
    # Add prefix to file paths if running on ts4
    if args.ts4:
        transforms_path = "/rapids/notebooks" + args.transforms_path
        folder_name = "/rapids/notebooks" + args.results_dir + args.folder_name
        labels_path = "/rapids/notebooks" + args.labels_path
    else:
        transforms_path = args.transforms_path
        folder_name = args.results_dir + args.folder_name
        labels_path = args.labels_path        
    
    # Make training folder
    os.mkdir(folder_name)
    
#     Load transformation stats
    with open(args.transforms_path, "r") as f:
        tstats = json.load(f)    
    ts = apply_transforms(tstats)

    # Initialize datasets and transform
    target_dataset = PlanetData(labels_path = labels_path, 
                                transform = ts, 
                                sample = args.sample, 
                                postval = args.postval, 
                                ts4 = args.ts4)

    # Define KFold
    kf = KFold(n_splits = args.num_kfolds, shuffle = True)
    
    num_epochs = args.num_epochs
    device = args.device
    iters = args.iters
    
    # Loop through each fold
    for fold, (train_indices, val_indices) in enumerate(kf.split(target_dataset)):
        
        save_dir = os.path.join(folder_name, f"kfold{fold}")
        
        # Make training folder
        os.mkdir(save_dir)        
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(target_dataset, batch_size = args.bs, sampler = train_sampler, num_workers = 1)
        val_loader = DataLoader(target_dataset, batch_size = args.bs, sampler = val_sampler, num_workers = 1)
        
        dataloaders = {"train": train_loader, "test": val_loader}
        
        # Write validation indices to file
        with open(f"{save_dir}/val_indices_fold{fold}.txt", "w") as val_file:
            val_file.write('\n'.join(map(str, val_indices)))           

        # Instantiate model
        model = construct_model(args.model_name, args.norm_coords, args.use_means)
        model.to(args.device)    

        print(model)

        # Implementation parameters
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)#, weight_decay = 0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)

        best_loss = 100000000
        best_model_wts = model.state_dict()

        gradient_norms = {}
        
        for epoch in range(num_epochs):

            start_time = time.time()  # Start time of the epoch

            with open(f"{save_dir}/records.txt", "a") as f:
                f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))

            with open(f"{save_dir}/records.txt", "a") as f:
                f.write('----------\n')

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                
                norms = []

                data = dataloaders[phase]
                data.dataset.set_stage(phase)

                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for c, (inputs, labels, coords) in enumerate(data):

                    inputs = inputs.to(device)
                    coords = coords.to(device)
                    labels = labels.to(device).view(-1, 1)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):

                        if args.model_name in ["SpA", "GeoConv", "FC"]:
                            outputs = model(inputs, coords)
                        else:
                            outputs = model(inputs)

                        loss = criterion(outputs, labels)

                        # statistics
                        running_loss += loss.item()

                        print(c, len(data), device, end = "\r")

                        # backward + optimize only if in training phase
                        if phase == 'train':

                            loss.backward()

                            if args.model_name == "GeoConv":
                                if c % iters == 0:
                                    optimizer.step()
                                    optimizer.zero_grad()       

                            else:
                                
                                batch_gradient_norms = []
                                for param in model.parameters():
                                    if param.grad is not None:
                                        batch_gradient_norms.append(param.grad.norm().item())

                                norms.append(sum(batch_gradient_norms) / len(batch_gradient_norms))  # Average gradient norm

                                optimizer.step()    
                                optimizer.zero_grad()       



                if phase == "train":

                    gradient_norms[epoch] = norms   

                    with open(f"{save_dir}/grads.json", "w") as f:
                        json.dump(gradient_norms, f)

                epoch_loss = running_loss / c

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

                with open(f"{save_dir}/records.txt", "a") as f:
                    f.write('{} Loss: {:.4f}\n'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'test' and epoch_loss < best_loss:

                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Save each epoch that achieves a higher accuracy than the current best_acc in case the model crashes mid-training
                    save_model(save_dir, epoch, best_model_wts, criterion, optimizer, scheduler, best = True)

                # save just in case...
                save_model(save_dir, epoch, model.state_dict(), criterion, optimizer, scheduler, best = False)

            end_time = time.time()  # End time of the epoch
            epoch_duration = end_time - start_time

            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds.")

            print()            

