import torchvision.transforms as T
from torchvision import datasets

from config import IMAGENET_PATH


class ImageNet2012:
    """
    To download imagenet, refer https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html
    to download specified files.

    Then process downloaded data:
        * for train data, unzip it:
            dir=./train 
            for x in `ls $dir/*tar` 
            do     
                filename=`basename $x .tar`     
                mkdir $dir/$filename     
                tar -xvf $x -C $dir/$filename 
            done 
            rm *.tar

        * for val data, refer issue: https://github.com/facebookresearch/dinov2/issues/496
    """

    def __init__(self, train=True, transform=True):
        path = IMAGENET_PATH
        split = "train" if train else "val"
        self.imagenet = datasets.ImageNet(path, split=split)
        self.id_to_class = self.imagenet.classes
        self.class_to_id = self.imagenet.class_to_idx
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
        return len(self.imagenet)

    def __getitem__(self, idx):
        img, label = self.imagenet[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
