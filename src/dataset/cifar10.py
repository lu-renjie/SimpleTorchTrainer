import torchvision.transforms as T
from torchvision import datasets

from config import CIFAR10_PATH


class CIFAR10:
    def __init__(self, train=True, transform=True):
        path = CIFAR10_PATH
        self.cifar10 = datasets.CIFAR10(path, train=train)
        self.id_to_class = self.cifar10.classes
        self.class_to_id = self.cifar10.class_to_idx
        if transform:
            self.transform = T.Compose([
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
