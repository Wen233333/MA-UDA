import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib


class CardiacSet(Dataset):
    def __init__(self, dir_path, domain='mr', label=True):
        super(CardiacSet, self).__init__()
        self.dir_path = dir_path
        self.fake_dir_path = self.dir_path + '_fake'
        print(('Real Dataset locate in: {}'.format(self.dir_path)))
        print(('Fake Dataset locate in: {}'.format(self.fake_dir_path)))
        self.domain = domain
        self.images = []
        self.fake_images = []
        self.segmentations = []
        self.label = label
        if domain == 'mr':
            self.fake_domain = 'ct'
        else:
            self.fake_domain = 'mr'

        _allimages = os.listdir(self.dir_path)

        for img_dir in _allimages:
            if img_dir.endswith('img.nii.gz'):
                _image = os.path.join(self.dir_path, img_dir)
                self.images.append(_image)
                _fake_image = os.path.join(self.fake_dir_path, img_dir[:-7]+'_fake.nii.gz')
                self.fake_images.append(_fake_image)
                if self.label:
                    _segmentation = os.path.join(self.dir_path, img_dir[:-10]+'lab.nii.gz')
                    self.segmentations.append(_segmentation)
        if self.label:
            assert (len(self.images) == len(self.fake_images) == len(self.segmentations))
        else:
            assert (len(self.images) == len(self.fake_images))

        print('Number of images in {}: {:d}'.format(self.domain, len(self.images)))

    def __getitem__(self, index):
        return self._make_img_gt_tuple(index)

    def _make_img_gt_tuple(self, index):
        _img = nib.load(self.images[index]).get_fdata().astype(np.float32)
        _fimg = nib.load(self.fake_images[index]).get_fdata().astype(np.float32)

        if self.label:
            _seg = nib.load(self.segmentations[index]).get_fdata().squeeze()
            samples = {'image': _img,
                       'fake_image': _fimg,
                       'segmentation': _seg,
                       'domain': self.domain,
                       'index': index}
        else:
            samples = {'image': _img,
                       'fake_image': _fimg,
                       'domain': self.domain,
                       'index': index}

        return samples

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_source_data = CardiacSet('/data_path', domain='mr')
    train_source_dataloader = torch.utils.data.DataLoader(train_source_data, batch_size=4,
                                                          drop_last=True,
                                                          shuffle=True,
                                                          num_workers=1
                                                          )
    src_iter = iter(train_source_dataloader)
    src_data = src_iter.next()
    print(src_data['image'].shape)
    print(src_data['fake_image'].shape)
    print(src_data['segmentation'].shape)
    print(0)

