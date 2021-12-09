import tarfile
import numpy as np
from io import BytesIO
from torch.utils.data import Dataset

class CorruptedCifar10(Dataset):
    def __init__(
        self, 
        tar_path, 
        corruption_file="CIFAR-10-C/fog.npy", 
        labels_file="CIFAR-10-C/labels.npy", 
        severity=1, 
        transform=None,
        target_transform=None
    ):
  
        self.severity = severity
        self.transform = transform
        self.target_transform = target_transform
        self.tf = tarfile.open(tar_path)

        self.data = self._get_data_from_tar(corruption_file)
        self.labels = self._get_data_from_tar(labels_file)

    
    def _get_data_from_tar(self, file_name):
        """
            10000 images per severity level
        """
        binary_stream = BytesIO()
        binary_stream.write(self.tf.extractfile(file_name).read())
        binary_stream.seek(0)
        data = np.load(binary_stream)
        return data[self.severity*10000:(self.severity*10000)+10000]
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):

        if indx == (self.__len__() - 1):
            self.tf.close()

        image = self.data[indx]
        label = self.labels[indx]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label