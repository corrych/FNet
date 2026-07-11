import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from PIL import Image

DATA_DIR = '/mnt/fch/datasets/WHU/'
#DATA_DIR = '/mnt/fch/datasets/Inria-Croped/'
#DATA_DIR = '/mnt/fch/datasets/building/'
if not os.path.exists(DATA_DIR):
    print('no data!')

x_train_dir = os.path.join(DATA_DIR, 'train/image')
y_train_dir = os.path.join(DATA_DIR, 'train/label')

x_valid_dir = os.path.join(DATA_DIR, 'test/image')
y_valid_dir = os.path.join(DATA_DIR, 'test/label')

x_test_dir = os.path.join(DATA_DIR, 'test/image')
y_test_dir = os.path.join(DATA_DIR, 'test/label')

# x_train_dir = os.path.join(DATA_DIR, 'train/image')
# y_train_dir = os.path.join(DATA_DIR, 'train/PNG')

# x_valid_dir = os.path.join(DATA_DIR, 'test/image')
# y_valid_dir = os.path.join(DATA_DIR, 'test/PNG')

# x_test_dir = os.path.join(DATA_DIR, 'test/image')
# y_test_dir = os.path.join(DATA_DIR, 'test/PNG')

#dataset类
class BuildDataSet(Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes,
            augmentation=None,
    ):
       
        self.image_list = os.listdir(images_dir)
        mask_list = os.listdir(masks_dir)
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_list]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in mask_list]

        self.class_values = classes

        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image_pil = Image.open(self.images_fps[i]).convert('RGB')
        image = np.array(image_pil).astype('float32')

        mask_pil = Image.open(self.masks_fps[i]).convert('L')
        mask = np.array(mask_pil) 
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask
    

def train_augmentation():
    train_transform = albu.Compose([
        albu.PadIfNeeded(
            min_height=512,
            min_width=512,   
            position='center'),
        albu.SquareSymmetry(p=1),
        # albu.Normalize( #Inria
        #     mean=(0.42303204, 0.43857941, 0.40342513),
        #     std=(0.18446438, 0.16982971, 0.16293759),
        #     max_pixel_value=255.0,
        #     p=1.0),
        # albu.Normalize( #China
        #     mean=(0.37146923, 0.37817337, 0.38629283),
        #     std=(0.22548223, 0.20468764, 0.19559253),
        #     max_pixel_value=255.0,
        #     p=1.0),
        albu.Normalize( #WHU
            mean=(0.43539026, 0.44513577, 0.41308374), 
            std=(0.21654843, 0.20331743, 0.21737308),
            max_pixel_value=255.0,
            p=1.0),
        albu.ToTensorV2(transpose_mask=True)
    ])
    return train_transform


def val_augmentation():
    val_transform = albu.Compose([
        albu.PadIfNeeded(
            min_height=512,
            min_width=512,  
            position='center'),
        # albu.Normalize( #Inria
        #     mean=(0.42303204, 0.43857941, 0.40342513),
        #     std=(0.18446438, 0.16982971, 0.16293759),
        #     max_pixel_value=255.0,
        #     p=1.0),
        # albu.Normalize( #China
        #     mean=(0.37146923, 0.37817337, 0.38629283),
        #     std=(0.22548223, 0.20468764, 0.19559253),
        #     max_pixel_value=255.0,
        #     p=1.0),
        albu.Normalize( #WHU
            mean=(0.43539026, 0.44513577, 0.41308374), 
            std=(0.21654843, 0.20331743, 0.21737308),
            max_pixel_value=255.0,
            p=1.0),
        albu.ToTensorV2(transpose_mask=True)
    ])
    return val_transform


train_dataset = BuildDataSet(
    x_train_dir,
    y_train_dir,
    augmentation=train_augmentation(),
    classes=[255]
)

test_dataset = BuildDataSet(
    x_test_dir,
    y_test_dir,
    augmentation=val_augmentation(),
    classes=[255]
)

val_dataset = BuildDataSet(
    x_valid_dir,
    y_valid_dir,
    augmentation=val_augmentation(),
    classes=[255]
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)
