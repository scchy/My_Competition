# python3
# Create Date: 2024-07-22
# Author: Scc_hy
# ==================================================================
from pathlib import Path
from torch.utils.data import DataLoader, Dataset 
from typing import List, AnyStr
import torchvision.transforms as transforms
import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 



sp2idx = {
    'RedFox': 0,
    'Hare': 1,
    'LeopardCat': 2,
    'BlackBear': 3,
    'Tiger': 4,
    'Badger': 5,
    'MuskDeer': 6,
    'Cheetah': 7,
    'AmurLeopard': 8,
}

idx2sp = {v:k for k, v in sp2idx.items()}

# 一般情况下，我们不会在验证集和测试集上做数据扩增
# 我们只需要将图片裁剪成同样的大小并装换成Tensor就行
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 当然，我们也可以再测试集中对数据进行扩增（对同样本的不同装换）
#  - 用训练数据的装化方法（train_tfm）去对测试集数据进行转化，产出扩增样本
#  - 对同个照片的不同样本分别进行预测
#  - 最后可以用soft vote / hard vote 等集成方法输出最后的预测
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # TODO:在这部分还可以增加一些图片处理的操作
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    # ToTensor() 放在所有处理的最后
    transforms.ToTensor(),
])


train_add_norm_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    # ToTensor() 放在所有处理的最后
    transforms.ToTensor(),
    transforms.Normalize(                 # 标准化
            mean=[0.485, 0.456, 0.406],      # ImageNet数据集的均值
            std=[0.229, 0.224, 0.225]        # ImageNet数据集的标准差
        )
])

train_add_norm_simple_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor(),
    transforms.Normalize(                 # 标准化
            mean=[0.485, 0.456, 0.406],      # ImageNet数据集的均值
            std=[0.229, 0.224, 0.225]        # ImageNet数据集的标准差
        )
])

test_add_norm_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(                 # 标准化
            mean=[0.485, 0.456, 0.406],      # ImageNet数据集的均值
            std=[0.229, 0.224, 0.225]        # ImageNet数据集的标准差
        )
])



class speciesRecDataSet(Dataset):
    def __init__(self, 
                 data_dir: AnyStr, 
                 tfm, 
                 data_type:AnyStr = 'train', 
                 files: List=None, 
                 tfm_extra=None,
                 **kwargs
                 ):
        super(speciesRecDataSet).__init__()
        self.path = Path(data_dir)
        if data_type != 'test':
            self.files = self.load_train_path()
        else:
            self.files = self.load_test_path()
        if files != None:
            self.files = files
        print(f"One {data_dir} sample", self.files[0])
        self.transform = tfm
        self.transform_extra = tfm_extra

    def load_test_path(self):
        files = list(self.path.iterdir())
        return sorted(files)
    
    def load_train_path(self):
        species_dirs = list(self.path.iterdir())
        files = []
        for sp in species_dirs:
            files.extend(sp.iterdir())
        return sorted(files)
    
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname).convert('RGB')
        tf_now = self.transform 
        if self.transform_extra is not None and np.random.randint(2) == 1:
            tf_now = self.transform_extra 
        im = tf_now(im)
        try:
            label = sp2idx[str(fname).split('/')[-2]]
        except:
            label = -1 # 测试集没有label
        return im, label


def quick_observe(train_dir_root):
    """
    快速观察训练集中的9张照片
    """
    pics_path = [os.path.join(train_dir_root, i) for i in os.listdir(train_dir_root)]
    labels = [i.split('_')[0] for i in os.listdir(train_dir_root)]
    idxs = np.arange(len(labels))
    sample_idx = np.random.choice(idxs, size=9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    for idx_, i in enumerate(sample_idx):
        row = idx_ // 3
        col = idx_ % 3
        img=Image.open(pics_path[i])
        axes[row, col].imshow(img)
        c = labels[i]
        axes[row, col].set_title(f'class_{c}')

    plt.show()