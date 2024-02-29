from .AFHQ_dataset import get_afhq_dataset
from .CelebA_HQ_dataset import get_celeba_dataset
from .LSUN_dataset import get_lsun_dataset
from torch.utils.data import DataLoader
from .IMAGENET_dataset import get_imagenet_dataset
from .CelebA_HQ_dataset_dialog import get_celeba_dialog_dataset
from .CelebA_HQ_dataset_with_attr import get_celeba_dataset_attr

from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, test_nums=None, train=True):
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
        self.img_files.sort()
        if test_nums is not None:
            if train:
                self.img_files = self.img_files[:-test_nums]
            else:
                self.img_files = self.img_files[-test_nums:]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.img_files[idx]

def get_dataset(dataset_type, dataset_paths, config, gender=None):
    # if category is CUSTOM, get images from custom arg path
    if config.data.category == "CUSTOM":
        print("inside custom dataset getter")
        test_dataset = CustomImageDataset(dataset_paths['custom_test'], transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return test_dataset

    return test_dataset


def get_dataloader(test_dataset, num_workers=0, shuffle=False):

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {'test': test_loader}


