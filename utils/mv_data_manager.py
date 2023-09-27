import torch.utils.data as data
import logging
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from torchvision import transforms
from torch.utils.data import Dataset
from .mv_transforms import *
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[1])



class DataManager(object): # data.dataset
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        
        self.root_path = 'data/modelnet40/'
        self.list_file = None # txt文件
        self.image_tmpl = '_{:03}.png'
        self.transform = None
        self.view_number = 6
        self.total_view = 12
        
        self.use_path = False
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.seed = seed
        self._setup_data()
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    def set_root_path(self, root_path):
        self.root_path = root_path
    
    def set_list_path_file(self, list_file):
        self.list_file = list_file
    
    def set_image_tmpl(self, image_tmpl):
        self.image_tmpl = image_tmpl
    
    def set_transform(self, transform):
        self.transform = transform
    
    def set_view_number(self, view_number):
        self.view_number = view_number
        
    def set_total_view(self, total_view):
        self.total_view = total_view
        
    def set_modality(self, modality):
        self.modality = modality
    
    @property
    def nb_tasks(self):
        return len(self._increments)

    @property
    def use_l2p_trsf(self):
        self._train_trsf = trans_data(True)
        self._test_trsf = trans_data(False)
        self._common_trsf = []

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, self.use_path)
        else:
            return DummyDataset(data, targets, self.use_path)

    def _setup_data(self):
        #assert self.list_file, 'list_file is none'
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875])])
        normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._train_trsf = transforms.Compose([train_augmentation, Stack(roll=False), ToTorchFormatTensor(div=True),normalize])
        self._test_trsf = transforms.Compose([GroupScale(int(256)), GroupCenterCrop(224), Stack(roll=False), ToTorchFormatTensor(div=True),normalize])
        self._common_trsf = []

        self._train_data, self._train_targets = self._get_data_with_views(self.dataset_name,train=True)
        self._test_data, self._test_targets = self._get_data_with_views(self.dataset_name,train=False)
    
        assert len(self._train_data) == len(self._train_targets)
                # Transforms

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if self.shuffle:
            np.random.seed(self.seed)
            order = np.random.permutation(len(order)).tolist()
        
        self._class_order = order
        logging.info(self._class_order)
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)


    def _load_image(self, directory, idx):
        return [Image.open(directory + self.image_tmpl.format(idx)).convert('RGB')]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        offsets = np.arange(self.view_number)
        return offsets + 1

    def _get_val_indices(self, record):
        offsets = np.arange(self.view_number)
        return offsets + 1

    def _get_test_indices(self, record):
        offsets = np.arange(self.view_number)
        return offsets + 1


    def get(self, record, indices, train):
        images = list()
        for view_idx in indices:
            seg_imgs = self._load_image(self.root_path+record.path, view_idx)
            if train:
                seg_imgs = self._train_trsf(seg_imgs)
            else:
                seg_imgs = self._test_trsf(seg_imgs)
            images.extend([np.array(seg_imgs)])
        
        images = np.array(images)
        return images, record.label # view h w c

    def _get_data_with_views(self,dataset_name,train):
        dataset_name = dataset_name[-2:]
        if train:
            list_file = 'v2_'+'trainmodel'+dataset_name+'.txt'
        else:
            list_file = 'v2_'+'testmodel'+dataset_name+'.txt'  
        list_file = 'data/modelnet40/' + list_file     
         
        video_list = [VideoRecord(x.strip().split(' ')) for x in open(list_file)]
        assert video_list, 'video_list is empty'
        
        data_list = []
        label_list = []
        for index in range(len(video_list)):
            record = video_list[index]
            
            # choose views
            view_indices = np.linspace(1, self.total_view, self.view_number, dtype=int)
            
            data_with_mv, data_label = self.get(record, view_indices, index)
            data_list.append(data_with_mv)
            label_list.append(data_label)
        
        return data_list, label_list

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]
    
class DummyDataset(Dataset):
    def __init__(self, images, labels, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        #self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return idx, image, label

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def trans_data(is_train):
        input_size = 224
        resize_im = input_size > 32
        if is_train:
            scale = (0.05, 1.0)
            ratio = (3. / 4., 4. / 3.)
            
            return  [
                transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))
        t.append(transforms.ToTensor())
        
        return t