import logging
import torch.optim as optim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim

def  evaluate_model(trainloader,testloader, model, device):
    '''
    Evaluates the model on the ingested data
    '''
    #define loss function
    criterion = nn.CrossEntropyLoss()

    #specify
    optimizer = optim.ASGD(model.parameters(), lr=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    epochs = 3
    for epoch in range(epochs):
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        running_loss = 0
        for i,(images,classes) in enumerate(trainloader):pyplot
            if i%20==0: print(f"iteration {i}/{len(trainloader)}")
            #use GPU
            images,classes = images.to(device), classes.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs,classes)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            
            running_loss += loss.item()
        else:
            validation_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                model.eval()
                for images,classes in testloader:
                    #USE GPU
                    images,classes = images.to(device), classes.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, classes)
                    validation_loss += loss.item()
                    
                    ps = torch.exp(outputs)
                    top_p,top_class = ps.topk(1,dim=1)
                    #reshape
                    equals = top_class == classes.view(*top_class.shape)                
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                model.train()
                
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "Valid Loss: {:.3f}.. ".format(validation_loss/len(testloader)),
                "Valid Accuracy: {:.3f}".format(accuracy/len(testloader)))
    #     scheduler.step()
    print("Running loss:", running_loss)
    return running_loss