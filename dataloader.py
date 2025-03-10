import glob
import torch
import nibabel as nib
from torch.utils import data

import pickle
import numpy as np
import os
from itertools import permutations


class torch_Dataset_OASIS(data.Dataset):
    def __init__(self, img_dir, seg_dir, mode, inshape):
        super(torch_Dataset_OASIS, self).__init__()
        self.inshape = inshape
        self.img = glob.glob(img_dir + '*.nii.gz')
        self.img.sort(key=lambda x: int(x[66:-7]))
        self.seg = glob.glob(seg_dir + '*.nii.gz')
        self.seg.sort(key=lambda x: int(x[66:-7]))
        assert len(self.img) == len(self.seg), 'Image number != Segmentation number'
        print('len(self.img) = {}, len(self.seg) = {}'.format(len(self.img), len(self.seg)))
        self.mode = mode
        self.training_img_pair = list(permutations(self.img[0:255], 2))
        self.training_seg_pair = list(permutations(self.seg[0:255], 2))
        self.testing_img_pair = list((moving, atlas) for moving in self.img[256:401] for atlas in self.img[401:405])
        self.testing_seg_pair = list((moving, atlas) for moving in self.seg[256:401] for atlas in self.seg[401:405])
    def __len__(self):
        if self.mode == 'train':
            assert len(self.training_img_pair) == len(self.training_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.training_img_pair)
        elif self.mode == 'test':
            assert len(self.testing_img_pair) == len(self.testing_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.testing_img_pair)
    def __getitem__(self, item):
        if self.inshape == (160, 192, 224):
            if self.mode == 'train':
                mi = torch.from_numpy(nib.load(self.training_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
                fi = torch.from_numpy(nib.load(self.training_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
                ml = torch.from_numpy(nib.load(self.training_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
                fl = torch.from_numpy(nib.load(self.training_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
                pair = (self.training_img_pair[item][0][66:-7], self.training_img_pair[item][1][66:-7])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
            elif self.mode == 'test':
                mi = torch.from_numpy(nib.load(self.testing_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
                fi = torch.from_numpy(nib.load(self.testing_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
                ml = torch.from_numpy(nib.load(self.testing_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
                fl = torch.from_numpy(nib.load(self.testing_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
                pair = (self.testing_img_pair[item][0][66:-7], self.testing_img_pair[item][1][66:-7])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
        elif self.inshape == (160, 160, 192):
            if self.mode == 'train':
                mi = torch.from_numpy(nib.load(self.training_img_pair[item][0]).get_fdata()[48:-48, 47:-49, 19:-45] / 255.0)
                fi = torch.from_numpy(nib.load(self.training_img_pair[item][1]).get_fdata()[48:-48, 47:-49, 19:-45] / 255.0)
                ml = torch.from_numpy(nib.load(self.training_seg_pair[item][0]).get_fdata()[48:-48, 47:-49, 19:-45])
                fl = torch.from_numpy(nib.load(self.training_seg_pair[item][1]).get_fdata()[48:-48, 47:-49, 19:-45])
                pair = (self.training_img_pair[item][0][66:-7], self.training_img_pair[item][1][66:-7])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
            elif self.mode == 'test':
                mi = torch.from_numpy(nib.load(self.testing_img_pair[item][0]).get_fdata()[48:-48, 47:-49, 19:-45] / 255.0)
                fi = torch.from_numpy(nib.load(self.testing_img_pair[item][1]).get_fdata()[48:-48, 47:-49, 19:-45] / 255.0)
                ml = torch.from_numpy(nib.load(self.testing_seg_pair[item][0]).get_fdata()[48:-48, 47:-49, 19:-45])
                fl = torch.from_numpy(nib.load(self.testing_seg_pair[item][1]).get_fdata()[48:-48, 47:-49, 19:-45])
                pair = (self.testing_img_pair[item][0][66:-7], self.testing_img_pair[item][1][66:-7])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()

def torch_Dataloader_OASIS(img_dir, seg_dir, mode, batch_size, inshape, random_seed=None):
    Dataset_OASIS = torch_Dataset_OASIS(img_dir, seg_dir, mode, inshape)

    if random_seed is None:
        loader = data.DataLoader(dataset=Dataset_OASIS, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(random_seed)
        loader = data.DataLoader(dataset=Dataset_OASIS, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False, generator=g)
    return loader

# reference paper: https://github.com/kangmiao15/Dual-Stream-PRNet-Plus/blob/master/dataset/mind101_label_25.txt
GROUP_CONFIG_Mind101 = {
        'parietal_lobe_1': ['postcentral', 'supramarginal', 'superior parietal',  'inferior parietal', ' precuneus'],
        'frontal_lobe_2': ['superior frontal', 'middle frontal', 'inferior frontal', 'lateral orbitofrontal', 'medial orbitofrontal', 'precentral', 'paracentral'],
        'occipital_lobe_3': ['lingual', 'pericalcarine', 'cuneus',  'lateral occipital'],
        'temporal_lobe_4': ['entorhinal', 'parahippocampal', 'fusiform', 'superior temporal', 'middle temporal',  'inferior temporal', 'transverse temporal'],
        'cingulate_lobe_5': ['cingulate', 'insula'],
    }

def group_label_Mind101(label):
    '''
    输入label为为原始的标签数据；
    按照txt表格中的规定，对各标签分成5大类，从而形成新的label数据，该label数据中只有6种[0-5]；
    输出划分好区域的label数据。
    '''
    label_name = '/Extra/yzy/Medical_Image_Registration/Mindboggle-101/mind101_label_25.txt'
    d = {}
    with open(label_name) as f:
        for line in f:
            if line != '\n':
                (value, key) = line.strip().split(',')
                d[key.strip().strip('"')] = int(value)
    label_merged = np.zeros(label.shape, dtype=np.int32)
    region = np.zeros(label.shape, dtype=bool)

    for key in GROUP_CONFIG_Mind101:
        for structure in GROUP_CONFIG_Mind101[key]:
            left_num = d['left ' + structure.strip()]
            right_num = d['right ' + structure.strip()]
            region = np.logical_or(region, np.logical_or(label == left_num, label == right_num))
        label_num = int(key.split('_')[-1])
        label_merged[region] = label_num
        region = np.zeros(label.shape, dtype=bool)
    return label_merged

class torch_Dataset_Mind101(data.Dataset):
    def __init__(self, data_dir, mode):
        super(torch_Dataset_Mind101, self).__init__()
        data_folder = glob.glob(data_dir + '*')
        data_folder.sort()
        self.mode = mode
        self.training_img_pair = list(permutations(data_folder[0:42], 2))
        self.testing_img_pair = list(permutations(data_folder[42:62], 2))
        print('training pairs: {}, testing pairs: {}'.format(len(self.training_img_pair), len(self.testing_img_pair)))

    def __len__(self):
        if self.mode == 'train':
            return len(self.training_img_pair)
        elif self.mode == 'test':
            return len(self.testing_img_pair)

    def __getitem__(self, item):
        if self.mode == 'train':
            m_id = self.training_img_pair[item][0][63:]
            m_data_path = os.path.join(self.training_img_pair[item][0], "t1weighted_brain.MNI152.nii.gz")
            m_label_path = os.path.join(self.training_img_pair[item][0], "labels.DKT31.manual.MNI152.nii.gz")
            f_id = self.training_img_pair[item][1][63:]
            f_data_path = os.path.join(self.training_img_pair[item][1], "t1weighted_brain.MNI152.nii.gz")
            f_label_path = os.path.join(self.training_img_pair[item][1], "labels.DKT31.manual.MNI152.nii.gz")
            mi = torch.from_numpy(nib.load(m_data_path).get_fdata()[11:-11, 13:-13, 11:-11] / 255.0)
            fi = torch.from_numpy(nib.load(f_data_path).get_fdata()[11:-11, 13:-13, 11:-11] / 255.0)
            ml = torch.from_numpy(nib.load(m_label_path).get_fdata()[11:-11, 13:-13, 11:-11])
            fl = torch.from_numpy(nib.load(f_label_path).get_fdata()[11:-11, 13:-13, 11:-11])
            pair = (m_id, f_id)
            return pair, mi.float(), fi.float(), ml.float(), fl.float()
        elif self.mode == 'test':
            m_id = self.testing_img_pair[item][0][63:]
            m_data_path = os.path.join(self.testing_img_pair[item][0], "t1weighted_brain.MNI152.nii.gz")
            m_label_path = os.path.join(self.testing_img_pair[item][0], "labels.DKT31.manual.MNI152.nii.gz")
            f_id = self.testing_img_pair[item][1][63:]
            f_data_path = os.path.join(self.testing_img_pair[item][1], "t1weighted_brain.MNI152.nii.gz")
            f_label_path = os.path.join(self.testing_img_pair[item][1], "labels.DKT31.manual.MNI152.nii.gz")
            mi = torch.from_numpy(nib.load(m_data_path).get_fdata()[11:-11, 13:-13, 11:-11] / 255.0)
            fi = torch.from_numpy(nib.load(f_data_path).get_fdata()[11:-11, 13:-13, 11:-11] / 255.0)
            ml = nib.load(m_label_path).get_fdata()[11:-11, 13:-13, 11:-11]
            fl = nib.load(f_label_path).get_fdata()[11:-11, 13:-13, 11:-11]
            ml = group_label_Mind101(ml)
            fl = group_label_Mind101(fl)
            ml = torch.from_numpy(ml)
            fl = torch.from_numpy(fl)
            pair = (m_id, f_id)
            return pair, mi.float(), fi.float(), ml.float(), fl.float()
        else:
            return None

def torch_Dataloader_Mind101(data_dir, mode, batch_size):
    Dataset_LPBA40 = torch_Dataset_Mind101(data_dir, mode)
    loader = data.DataLoader(dataset=Dataset_LPBA40, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    return loader


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
class torch_Dataset_IXI(data.Dataset):
    def __init__(self, tra_dir, val_dir, mode, inshape=None):
        super(torch_Dataset_IXI, self).__init__()
        self.inshape = inshape
        self.tra = glob.glob(tra_dir + '*.pkl')
        self.tra.sort(key=lambda x: int(x[76:-4]))
        self.val = glob.glob(val_dir + '*.pkl')
        self.val.sort(key=lambda x: int(x[75:-4]))
        print('len(self.tra) = {}, len(self.val) = {}'.format(len(self.tra), len(self.val)))
        self.mode = mode
        self.training_tra_pair = list(permutations(self.tra, 2))
        self.testing_val_pair = list((moving, atlas) for moving in self.val[5:115] for atlas in self.val[0:5])
    def __len__(self):
        if self.mode == 'train':
            return len(self.training_tra_pair)
        elif self.mode == 'test':
            return len(self.testing_val_pair)
    def __getitem__(self, item):
        if self.inshape is None:
            if self.mode == 'train':
                mi, ml = pkload(self.training_tra_pair[item][0])
                fi, fl = pkload(self.training_tra_pair[item][1])
                mi = torch.from_numpy(mi)
                ml = torch.from_numpy(ml)
                fi = torch.from_numpy(fi)
                fl = torch.from_numpy(fl)
                pair = (self.training_tra_pair[item][0][76:-4], self.training_tra_pair[item][1][76:-4])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
            elif self.mode == 'test':
                mi, ml = pkload(self.testing_val_pair[item][0])
                fi, fl = pkload(self.testing_val_pair[item][1])
                mi = torch.from_numpy(mi)
                ml = torch.from_numpy(ml)
                fi = torch.from_numpy(fi)
                fl = torch.from_numpy(fl)
                pair = (self.testing_val_pair[item][0][75:-4], self.testing_val_pair[item][1][75:-4])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
        elif self.inshape == (160, 192, 160):
            if self.mode == 'train':
                mi, ml = pkload(self.training_tra_pair[item][0])
                fi, fl = pkload(self.training_tra_pair[item][1])
                mi = torch.from_numpy(mi[:, :, 32:-32])
                ml = torch.from_numpy(ml[:, :, 32:-32])
                fi = torch.from_numpy(fi[:, :, 32:-32])
                fl = torch.from_numpy(fl[:, :, 32:-32])
                pair = (self.training_tra_pair[item][0][76:-4], self.training_tra_pair[item][1][76:-4])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
            elif self.mode == 'test':
                mi, ml = pkload(self.testing_val_pair[item][0])
                fi, fl = pkload(self.testing_val_pair[item][1])
                mi = torch.from_numpy(mi[:, :, 32:-32])
                ml = torch.from_numpy(ml[:, :, 32:-32])
                fi = torch.from_numpy(fi[:, :, 32:-32])
                fl = torch.from_numpy(fl[:, :, 32:-32])
                pair = (self.testing_val_pair[item][0][75:-4], self.testing_val_pair[item][1][75:-4])
                return pair, mi.float(), fi.float(), ml.float(), fl.float()
        else:
            print("Undefined shape")
            return None


def torch_Dataloader_IXI(tra_dir, val_dir, mode, batch_size, inshape=None, random_seed=None):
    Dataset_IXI = torch_Dataset_IXI(tra_dir, val_dir, mode, inshape)
    if random_seed is None:
        loader = data.DataLoader(dataset=Dataset_IXI, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(random_seed)
        loader = data.DataLoader(dataset=Dataset_IXI, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False, generator=g)
    return loader

