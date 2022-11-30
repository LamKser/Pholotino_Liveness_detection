from torchvision import models
from torch import nn, optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
from data_loader import LoadData
import os
from PIL import ImageFile, Image
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Model(nn.Module):

    def __init__(self, name, num_class, pretrained=False):
        super(Model, self).__init__()

        # ResNet 50
        if name == 'resnet50':
            if pretrained:
                self.model = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.model = models.resnet50()

        # ResNet 101
        elif name == 'resnet101':
            if pretrained:
                self.model = models.resnet101(
                    weights=models.ResNet101_Weights.IMAGENET1K_V2)
            else:
                self.model = models.resnet101()

        # Change the number of class
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)


class RunModel():

    def __init__(self, device, name,
                 train_path, val_path, test_path, test_video_path, batch_size,
                 lr, weight_decay, momentum,
                 is_scheduler, step_size, gamma,
                 num_class=1, pretrained=False):
        self.device = device
        self.model = Model(name, num_class, pretrained).to(self.device)
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

        self.data = LoadData(train_path, val_path, test_path,
                             test_video_path, batch_size)

        print("Device use:", self.device)
        print("Done load dataset")

    def __save_model(self, save_path, weight_file):
        # Create path if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(save_path, weight_file))

    def __train_one_epoch(self, epoch, epochs, train_data):
        with torch.set_grad_enabled(True):
            self.model.train()
            total_loss = 0
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(train_data),
                        total=len(train_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description(f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}][Train]')

            for step, (images, targets) in pbar:
                self.optimizer.zero_grad()
                images, targets = images.to(
                    self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)
                loss.backward()

                _, predict = torch.max(outputs.data, 1)
                total_acc = total_acc + (predict == targets).sum().item()
                total_loss = total_loss + loss.item()
                total = total + images.size(0)
                self.optimizer.step()

                if step % 250:
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')

        return ave_acc, ave_loss

    def __val(self, epoch, epochs, val_data):
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_loss = 0
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(val_data),
                        total=len(val_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description(' ' * len(f'Epoch [{epoch}/{epochs}][{self.scheduler.get_last_lr()[0]}]') + '[Valid]')

            for step, (images, targets) in pbar:
                images, targets = images.to(
                    self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.critetion(outputs, targets)

                _, predict = torch.max(outputs.data, 1)
                total_acc = total_acc + (predict == targets).sum().item()
                total_loss = total_loss + loss.item()
                total = total + images.size(0)

                if step % 200:
                    pbar.set_postfix(acc=f'{total_acc/total:.4f}', loss=f'{total_loss/(step + 1):.4f}')

            ave_acc = total_acc / total
            ave_loss = total_loss / (step + 1)
            pbar.set_postfix(acc=f'{ave_acc:.4f}', loss=f'{ave_loss:.4f}')

        return ave_acc, ave_loss

    def train(self, epochs, save_path, weight_file, logger_path, val, continue_train=False):
        train_data = self.data.train_loader()
        if val:
            val_data = self.data.val_loader()

        # Write loss and accuracy to log file
        writer = SummaryWriter(logger_path)

        # Load pretrained weight
        if continue_train:
            checkpoint = torch.load(os.path.join(save_path, weight_file))
            self.model.load_state_dict(checkpoint['state_dict'])

        for epoch in range(1, epochs+1):
            # Train
            __train_acc, __train_loss = self.__train_one_epoch(
                epoch, epochs, train_data)

            # Validation
            if val:
                __val_acc, __val_loss = self.__val(epoch, epochs, val_data)

            if not (self.scheduler is None):
                self.scheduler.step()

            # Write to log file
            if val:
                writer.add_scalars('Loss', {'train': __train_loss,
                                            'val': __val_loss},
                                            epoch)
                writer.add_scalars('Accuracy', {'train': __train_acc,
                                                'val': __val_acc},
                                                epoch)
            else:
                writer.add_scalars('Loss', {'train': __train_loss}, epoch)
                writer.add_scalars('Accuracy', {'train': __train_acc}, epoch)

            self.__save_model(save_path, weight_file)
        writer.close()

    def test(self, file_csv, weight_file):
        df = pd.DataFrame(columns=['file_name', 'liveness_score', 'label'])
        test_data = self.data.test_loader()

        # Load state_dict file
        checkpoint = torch.load(weight_file)
        self.model.load_state_dict(checkpoint['state_dict'])

        paths = []
        scores = []
        truths = []
        with torch.set_grad_enabled(False):
            self.model.eval()
            total_acc = 0
            total = 0

            pbar = tqdm(enumerate(test_data),
                        total=len(test_data),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            pbar.set_description('Testing model')

            for step, (path, images, targets) in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                _, predict = torch.max(outputs.data, 1)
                total_acc = total_acc + (predict == targets).sum().item()
                total = total + images.size(0)

                # Save to csv
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

    def test_video(self, file_csv, weight_file):
        video_path, video_files, transform = self.data.test_video_loader()

        # Load state_dict file
        checkpoint = torch.load(weight_file)
        self.model.load_state_dict(checkpoint['state_dict'])

        df = pd.DataFrame(columns=['fname', 'liveness_score'])
        fname = []
        liveness_score = []

        with torch.set_grad_enabled(False):
            self.model.eval()
            for video in tqdm(video_files, total=len(video_files)):
                cap = cv2.VideoCapture(os.path.join(video_path, video))
                total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                score = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_tensor = transform(Image.fromarray(frame))
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)
                    outputs = self.model(image_tensor)
                    predicted = F.softmax(outputs, 1)
                    score = score + predicted[0][1].item()
                fname.append(video)
                liveness_score.append(score/total_frame)

        df['fname'] = fname
        df['liveness_score'] = liveness_score

        df.to_csv(file_csv, index=False)
        print(f"Saved at {file_csv}")
