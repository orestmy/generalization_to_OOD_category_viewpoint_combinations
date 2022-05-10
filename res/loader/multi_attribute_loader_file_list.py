import torch.utils.data as data
from torchvision import datasets, models, transforms
IN_SIZE = 224
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import os
import os.path
import sys
import numpy as np
import torch

project_root = os.getcwd()
data_root = "%s/data"%project_root

def make_dataset(list_file, data_dir):
        images = []
        labels = []

        with open(list_file,'r') as F:
            lines = F.readlines()

        for line in lines:
            image = line.rstrip()
            images.append("%s/%s"%(data_dir,image))
            label = image.replace('images/frame','labels/label_frame')
            labels.append(label)


        return images, labels

class FileListFolder(data.Dataset):
    def __init__(self, file_list, attributes_dict, transform, data_dir):
        samples,targets  = make_dataset(file_list, data_dir)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 samples"))

        self.root = file_list

        self.samples = samples
        self.targets = targets

        self.transform = transform

        with open(attributes_dict, 'rb') as F:
            attributes = pickle.load(F)

        self.attributes = attributes


    def __getitem__(self, index):

        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        impath = self.samples[index]
        #if 'om2' in impath:
        #    impath = impath.replace('/om2/user/smadan/car_dataset/','/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/')
        #if 'om5' in impath:
        #    impath = impath.replace('/om5/user/smadan/car_dataset/','/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/')
        sample_label = self.attributes[impath.split('/')[-1]]
        label_path = impath.replace('images/frame','labels/label_frame')

        #impath = impath.replace('/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/','/om5/user/smadan/car_dataset/')
        sample = Image.open(impath)
        
        
        
        floated_labels = []
        for s in sample_label:
            floated_labels.append(float(s))

        if self.transform is not None:
            transformed_sample = self.transform(sample)

        transformed_labels = torch.LongTensor(floated_labels)

        return transformed_sample, transformed_labels, impath

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'

        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def getStats(self):
        dict = {}
        for i in range(5):
            dict['category{}'.format(i)] = {}
        for i in range(len(self.samples)):
            transformed_sample, transformed_labels, impath = self.__getitem__(i)
            category = transformed_labels[1].item()
            rot = transformed_labels[3].item()
            myd = dict['category{}'.format(category)]
            if rot in myd:
                myd[rot] +=1
            else:
                myd[rot] = 1
        return dict

    def getStasMatrix(self):
        occurences = [[0]*5 for _ in range(5)]
        for i in range(len(self.samples)):
            transformed_sample, transformed_labels, impath = self.__getitem__(i)
            category = transformed_labels[1].item()
            rot = transformed_labels[3].item()
            occurences[category][rot] +=1

        print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                         for row in occurences]))
        return occurences

    def getStasMatrix_ImPath(self):
        occurences = [[[] for j in range(5)] for _ in range(5)]
        labels = [[[] for j in range(5)] for _ in range(5)]
        for i in range(len(self.samples)):
            transformed_sample, transformed_labels, impath = self.__getitem__(i)
            category = transformed_labels[1].item()
            rot = transformed_labels[3].item()
            occurences[category][rot].append(impath)
            labels[category][rot].append(transformed_labels)

        return occurences, labels

    def show_images_on_subplot(self):
        from os import listdir
        from os.path import isfile, join
        import matplotlib.image as mpimg

        # mypath = '/home/orest/code/fle-pcb/data/supertoroids_objects81/raw/meshes/img/'
        # imgfiles = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
        occ, lab = self.getStasMatrix_ImPath()
        f, axarr = plt.subplots(5, 5)
        f.set_size_inches(12,12)
        for i in reversed(range(0, 5)):
            for j in range(5):
                if len(occ[i][j]):
                    img = mpimg.imread(occ[i][j][0])
                else:
                    img = np.zeros((224, 224, 3))

                axarr[i, j].imshow(img)
                axarr[i, j].axis('off')
                # axarr[i, j].set_title('Label = {},{}'.format(i,j), fontsize=8)

        f.tight_layout(pad=0.0, h_pad=0)
        plt.axis('off')
        plt.show()


    def show_images_on_subplot_categories(self, category=0, viewpoint=0):
        from os import listdir
        from os.path import isfile, join
        import matplotlib.image as mpimg

        # mypath = '/home/orest/code/fle-pcb/data/supertoroids_objects81/raw/meshes/img/'
        # imgfiles = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
        occ,lab = self.getStasMatrix_ImPath()
        f, axarr = plt.subplots(5, 5)
        f.set_size_inches(12,12)
        if len(occ[category][viewpoint])<25:
            return
        occv = occ[category][viewpoint][:25]
        labv = lab[category][viewpoint][:25]
        k=0
        for i in range(5):
            for j in range(5):
                img = mpimg.imread(occv[k])
                axarr[i, j].imshow(img)
                axarr[i, j].axis('off')
                axarr[i, j].set_title('Label = {}'.format(labv[k].numpy()), fontsize=8)
                k+=1

        f.tight_layout(pad=0.0, h_pad=0)
        plt.axis('off')
        plt.show()