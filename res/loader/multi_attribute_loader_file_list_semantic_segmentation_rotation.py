from torchvision import datasets, models, transforms

from loader.multi_attribute_loader_file_list_semantic_segmentation import FileListFolder as FileListBase


class FileListFolder(FileListBase):

    def __getitem__(self, index):
        '''
        creates multi-class rotation label
        '''
        rotation_label_index = 1
        transformed_sample, formatted_label, impath = super(FileListFolder, self).__getitem__(index)
        rotation_label = self.attributes[impath.split('/')[-1]][rotation_label_index]

        # add to label rotation class where needed
        formatted_label[formatted_label != 0] += rotation_label
        return transformed_sample, formatted_label, impath

    def labels(self):
        segmentation_classes = ['void']
        car_cls = ['rot{}_car'.format(i) for i in range(1, 6)]
        segmentation_classes.extend(car_cls)
        labels_dict = {}
        for i, label in enumerate(segmentation_classes):
            labels_dict[i] = label
        return labels_dict
