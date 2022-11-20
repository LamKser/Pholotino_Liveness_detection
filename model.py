from torchvision import models
from torch import nn, optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
from data_loader import LoadData
import os
from PIL import ImageFile
import numpy as np
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VGG19(nn.Module):

    def __init__(self, num_class, pretrained=False):
        super(VGG19, self).__init__()
        if pretrained:
            self.model = models.vgg19(
                weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            self.model = models.vgg19()
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)


class RunModel():

    def __init__(self, device, 
                        train_path, val_path, test_path, test_video_path, batch_size,
                        lr, weight_decay, momentum,
                        is_scheduler, step_size, gamma,
                        num_class=1, pretrained=False):
        self.device = device
        self.model = VGG19(num_class, pretrained).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   momentum=momentum)
        self.critetion = nn.BCEWithLogitsLoss().to(self.device)
        # self.critetion = nn.CrossEntropyLoss().to(self.device)
        if is_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size=step_size,
                                                       gamma=gamma)
        else:
            self.scheduler = None

        data = LoadData(train_path, val_path, test_path, test_video_path, batch_size)
        self.train_data = data.train_loader()
        self.val_data = data.val_loader()
        self.test_data = data.test_loader()
        self.test_video_data = data.test_video_loader()
        print("Device use:", self.device)
        print("Done load dataset:")


    def __save_model(self, save_path, weight_file):
        torch.save({'state_dict': self.model.state_dict()},
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

            pbar.set_description(
                f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}][Train]')
            for step, (images, targets) in pbar:
                self.optimizer.zero_grad()
                targets = targets.unsqueeze(1).float()
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)
                loss.backward()

                # _, predict = torch.max(outputs.data, 1)
                # predict = 1 if torch.sigmoid(outputs) >= 0.5 else 0
                predict = torch.sigmoid(outputs).round()
                total_acc = total_acc + (predict == targets).sum()
                total_loss = total_loss + loss.item()
                total = total + images.size(0)
                self.optimizer.step()

                if step % 250:
                    # print(f'\rEpoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}]: Train loss - {(loss.item()/images.size(0)):.4f} Train acc - {(step_acc/images.size(0)):.4f}', end='\r')
                    pbar.set_postfix(
                        acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')

        return ave_acc, ave_loss

    def __val(self, epoch, epochs):
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_loss = 0
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(self.val_data),
                        total=len(self.val_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description(
                ' ' * len(f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}]') + '[Valid]')

            for step, (images, targets) in pbar:
                targets = targets.unsqueeze(1).float()
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)

                # _, predict = torch.max(outputs.data, 1)
                predict = torch.sigmoid(outputs).round()
                total_acc = total_acc + (predict == targets).sum()
                total_loss = total_loss + loss.item()
                total = total + images.size(0)

                if step % 200:
                    pbar.set_postfix(
                        acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')

        return ave_acc, ave_loss

    def train(self, epochs, save_path, weight_file):
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []
        for epoch in range(1, epochs+1):
            __train_acc, __train_loss = self.__train_one_epoch(epoch, epochs)
            __val_acc, __val_loss = self.__val(epoch, epochs)
            if not self.scheduler:
                self.scheduler.step()
            train_acc.append(__train_acc)
            train_loss.append(__train_loss)
            val_acc.append(__val_acc)
            val_loss.append(__val_loss)
            self.__save_model(save_path, weight_file)

        return train_acc, train_loss, val_acc, val_loss

    def test(self, file_csv, weight_file):
        df = pd.DataFrame(columns=['file_name', 'liveness_score', 'label'])
        checkpoint = torch.load(weight_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        paths = []
        scores = []
        truths = []
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(self.test_data),
                        total=len(self.test_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description('Testing model')

            for step, (path, images, targets) in pbar:
                # print(path)
                targets = targets.unsqueeze(1).float()
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                # _, predict = torch.max(outputs.data, 1)
                predict = torch.sigmoid(outputs)
                total_acc = total_acc + (predict.round() == targets).sum()
                total = total + images.size(0)

                # Save to csv
                # score = torch.softmax(outputs, 1)
                paths.append(path)
                scores.append(predict.data.cpu().numpy())
                truths.append(targets.data.cpu().numpy())
                if step % 200:
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}')

            ave_acc = total_acc / total
            pbar.set_postfix(acc=f'{ave_acc:.4f}')

        paths = np.array([subpath.split('\\')[-1] for p in paths for subpath in p])
        scores = np.array([subscore for s in scores for subscore in s])
        truths = np.array([subtruth for truth in truths for subtruth in truth])
        df['file_name'] = paths
        df['liveness_score'] = scores
        df['label'] = truths

        df.to_csv(file_csv, index=False)
        print(f'Saved results in {file_csv}')

    def test_video(self):
        video_path, video_files, transform = self.test_video_data
        for video in video_files:
            cap = cv2.VideoCapture(os.path.join(video_path, video_files))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = transform(frame)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                output = self.model(image_tensor)
            print(f"Done {video}")
