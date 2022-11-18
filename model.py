from torchvision import models
from torch import nn, optim
import torch
from tqdm import tqdm 
import time
from data_loader import LoadData
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VGG19(nn.Module):
    
    def __init__(self, num_class, pretrained=False):
        super(VGG19, self).__init__()
        if pretrained:
            self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            self.model = models.vgg19()
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_class)
    
    def forward(self, x):
        return self.model(x)

class RunModel():

    def __init__(self, device, train_path, val_path, test_path, batch_size, 
                        lr, weight_decay, momentum,
                        is_scheduler, step_size, gamma,
                        num_class=2, pretrained=False):
        self.device = device
        self.model = VGG19(num_class, pretrained).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=lr, 
                                    weight_decay=weight_decay, 
                                    momentum=momentum)
        self.critetion = nn.CrossEntropyLoss().to(self.device)
        if is_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=step_size,
                                                        gamma=gamma)
        else:
            self.scheduler = None

        data = LoadData(train_path, val_path, test_path, batch_size)
        self.train_data = data.train_loader()
        self.val_data = data.val_loader()
        self.test_data = data.test_loader()
        print("Done load dataset")

    def __save_model(self, save_path, weight_file):
        torch.save({"state_dict":self.model.state_dict()},
                    os.path.join(save_path, weight_file))
                    
    def __train_one_epoch(self, epoch, epochs):
        with torch.set_grad_enabled(True):
            self.model.train()
            total_loss = 0
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(self.train_data),
                        total=len(self.train_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            
            pbar.set_description(f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}][Train]')
            for step, (images, targets) in pbar:
                self.optimizer.zero_grad()
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)
                loss.backward()

                _, predict = torch.max(outputs.data, 1)
                step_acc = (predict == targets).sum().item()
                total_acc = total_acc + step_acc              
                total_loss = total_loss + loss.item()
                total = total + images.size(0)
                self.optimizer.step()

                if step % 250:
                    # print(f'\rEpoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}]: Train loss - {(loss.item()/images.size(0)):.4f} Train acc - {(step_acc/images.size(0)):.4f}', end='\r')
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')
            

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')
            
    def train(self, epochs, save_path, weight_file):
        for epoch in range(1, epochs+1):
            self.__train_one_epoch(epoch, epochs)
            self.val()
            if not self.scheduler:
                 self.scheduler.step()
            self.__save_model(save_path, weight_file)
        
    def val(self):
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_loss = 0
            total_acc = 0
            total = 0

            start = time.time()
            pbar = tqdm(enumerate(self.val_data),
                        total=len(self.val_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for step, (images, targets) in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)
                _, predict = torch.max(outputs.data, 1)
                total_acc = total_acc + (predict == targets).sum()                
                total_loss = total_loss + loss.item()
                total = total + 1

            end = time.time()

            ave_acc = total_acc / (step + 1)
            ave_loss = total_loss / (step + 1)
            print(f'Val loss - {(total_loss/total):.4f} Val acc - {(total_acc/total):.4f} ({(end-start):.4f}s)')
    
    def test(self):
        pass
