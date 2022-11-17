from torchvision.datasets import ImageFolder 
from torchvision import transforms
from torch.utils.data import DataLoader

class LoadData:
    def __init__(self, train_path, val_path, test_path, batch_size=10):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
    
    def train_loader(self):
        transform = transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[1., 1., 1.] )
                        ])
        image_folder = ImageFolder(self.train_path, transform)
        train_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=True)
        return train_data

    def val_loader(self):
        transform = transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[1., 1., 1.] )
                        ])
        image_folder = ImageFolder(self.val_path, transform)
        val_data = DataLoader(image_folder, batch_size=self.batch_size, shuffle=True)
        return val_data
    
    def test_loader(self):
        transform = transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[1., 1., 1.] )
                        ])
        # val_data = DataLoader(image_folder, batch_size=self.batch_size
        pass

# if __name__ == "__main__":
#     dataloader = train_data("D:\\ZaloAI\\Data Process\\Pholotino_Liveness_detection\\train")
#     for step, (x, y) in enumerate(dataloader):
#         # print(step, x.size(), y.size())
#         print(y)