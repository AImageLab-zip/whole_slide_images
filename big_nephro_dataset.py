from PIL import Image
import yaml
import os.path
from yaml import CLoader as Loader
# from rotated_rectangle_crop import crop_rotated_rectangle
from torch.nn import ModuleList
from torch import stack
import torch.utils.data as data
import numpy as np
import random
from torch.utils.data.sampler import Sampler
import glob
import glob


# mean values RGB = [0.60608787 0.57173514 0.61699724] | std values RGB = [0.37850211 0.37142419 0.38158805]
class YAMLSegmentationDataset(data.Dataset):

    def __init__(self, dataset=None, transforms=None, split=['training']):
        """
           Initializes a pytorch Dataset object

           :param dataset: A filename (string), to identify the yaml file
              containing the dataset.
           :param transform: Transformation function to be applied to the input
              images (e.g. created with torchvision.transforms.Compose()).
           :param split: A list of strings, one for each dataset split to be
              loaded by the Dataset object.
           """

        self.dataset = dataset
        self.transform = transforms
        self.imgs = []
        self.lbls = []

        data_root = os.path.dirname(dataset)

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                self.imgs.append(os.path.join(data_root, d['images'][i]['location']))
                self.lbls.append(os.path.join(data_root, d['images'][i]['label']))

    def __getitem__(self, index):

        image = np.asarray(Image.open(self.imgs[index]))
        ground = np.load(self.lbls[index])

        if self.transform is not None:
            image, ground = self.transform(image, ground)

        return image, ground, os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.lbls)


class YAML10YDataset(data.Dataset):
    # mean values BGR = [0.81341412 0.76660304 0.83704776] | std values BGR = [0.14812355 0.18829341 0.12363736]
    def __init__(self, dataset, patches_per_bio, transforms=None, split=['training']):
        """
           Initializes a pytorch Dataset object

           :param dataset: A filename (string), to identify the yaml file
              containing the dataset.
           :param transform: Transformation function to be applied to the input
              images (e.g. created with torchvision.transforms.Compose()).
           :param split: A list of strings, one for each dataset split to be
              loaded by the Dataset object.
           """

        self.patches_per_bio = patches_per_bio
        self.dataset = dataset
        self.transform = transforms
        self.bios = {}

        data_root = os.path.dirname(dataset)

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            # build a dictionary to associate each biopsy to the patches and the labels --> d[bio] = {imgs = [list_of_patches_locations]; label = 0...1}
            for i in d['split'][s]:
                img_bio = d['images'][i]['values']['bio']
                img_path = os.path.join(data_root, d['images'][i]['location'])
                if img_bio in self.bios.keys():
                    self.bios[img_bio]['patches'].append(img_path)
                else:
                    img_esrd = d['images'][i]['values']['ESRD']
                    img_fup = float(d['images'][i]['values']['fup'])
                    img_lbl = 0
                    if img_esrd == 'FALSE':
                        img_lbl = 0.5 - min(img_fup, 10) / 20.
                    elif img_fup < 20:
                        img_lbl = 0.5 + (20 - max(img_fup, 10)) / 20.

                    self.bios[img_bio] = {'patches': [img_path], 'label': img_lbl}

    def __getitem__(self, index):
        bio = self.bios[list(self.bios.keys())[index]]
        try:
            patches = random.sample(bio['patches'], self.patches_per_bio)
        except ValueError:
            patches = bio['patches']
            patches += [random.choice(bio['patches']) for _ in range(self.patches_per_bio - len(bio['patches']))]
            random.shuffle(patches)

        ground = bio['label']
        images = []

        for patch in patches:
            # image = np.asarray(Image.open(patch))
            image = Image.open(patch)
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        return stack(images), ground, patches

    def __len__(self):
        return len(self.bios.keys())


class YAML10YBiosDataset(data.Dataset):
    # PATCHES mean values BGR = [0.81341412 0.76660304 0.83704776] | std values BGR = [0.14812355 0.18829341 0.12363736]
    # [0.74629832 0.67295842 0.78365591] | std values RGB = [0.17482606 0.21674619 0.14285819]
    def __init__(self, dataset, crop_type, patches_per_bio, transforms=None, split=['training']):
        """
           Initializes a pytorch Dataset object

           :param dataset: A filename (string), to identify the yaml file
              containing the dataset.
           :param transform: Transformation function to be applied to the input
              images (e.g. created with torchvision.transforms.Compose()).
           :param split: A list of strings, one for each dataset split to be
              loaded by the Dataset object.
           """

        self.patches_per_bio = patches_per_bio
        self.dataset = dataset
        self.transforms = transforms
        self.bios = {}
        self.imgs_root = os.path.join(os.path.dirname(dataset), crop_type + '_images/')

        all_images = glob.glob(self.imgs_root + '*.png')
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                img_bio = d['bios'][i]['bio']
                # imgs_path = glob.glob(self.imgs_root + f'id*_{img_bio}*.png')
                imgs_path = [img for img in all_images if f'_{img_bio}_pas' in img]
                if imgs_path == []:
                    print(f'bio {img_bio} has no images')
                    continue
                img_esrd = d['bios'][i]['ESRD']
                img_fup = float(d['bios'][i]['fup'])
                img_lbl = 0
                if img_esrd == 'FALSE':
                    img_lbl = 0.5 - min(img_fup, 10) / 20.
                elif img_fup < 20:
                    img_lbl = 0.5 + (20 - max(img_fup, 10)) / 20.

                self.bios[img_bio] = {'images': imgs_path, 'label': img_lbl}

        pass

    def __getitem__(self, index):
        bio = self.bios[list(self.bios.keys())[index]]
        try:
            patches = random.sample(bio['images'], self.patches_per_bio)
        except ValueError:
            patches = bio['images']
            patches += [random.choice(bio['images']) for _ in range(self.patches_per_bio - len(bio['images']))]
            random.shuffle(patches)

        ground = bio['label']
        images = []

        for patch in patches:
            # image = np.asarray(Image.open(patch))
            image = Image.open(patch)
            # debug_plot(np.array(image))
            if self.transforms is not None:
                image = self.transforms(image)
            # debug_plot(np.array(image))
            images.append(image)

        return stack(images), ground, patches

    def __len__(self):
        return len(self.bios.keys())


def debug_plot(img, cmap=None):
    from matplotlib import pyplot as plt
    import numpy as np
    img = np.array(img)
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show(block=False)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # preprocess_fn = transforms.Compose([transforms.RandomCrop((1000, 2000), pad_if_needed=True, fill=255)])
    # preprocess_fn = transforms.Compose([transforms.RandomCrop((1000, 2000), pad_if_needed=True, fill=255), transforms.Resize(size=(256, 512))])
    # preprocess_fn = transforms.Resize((256, 256))
    dname = '/nas/softechict-nas-2/fpollastri/data/big_nephro/big_nephro_bios_dataset.yml'
    custom_training_transforms = transforms.Compose([
        # transforms.RandomCrop(512, pad_if_needed=True, fill=255),
        # transforms.Resize((256, 256)),

        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(180, fill=255)]), p=.25),
        # preprocess_fn,
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(contrast=(1.7, 1.7)),
        transforms.ToTensor(),
        # transforms.Normalize((0.813, 0.766, 0.837), (0.148, 0.188, 0.124)),
    ])
    ppb = 1
    dataset = YAML10YBiosDataset(dataset=dname, crop_type='patches', patches_per_bio=ppb, transforms=custom_training_transforms, split=['training', 'test'])

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             # drop_last=True,
                             pin_memory=True)

    rgb = np.zeros((ppb * 450, 3, 256, 256))
    counter = 0
    for i, (b_img, lbl, name) in enumerate(data_loader):
        if i % 10 == 0:
            print(f'doing batch #{i}')
        for img in b_img:
            # for s_img in img:
                # debug_plot(np.moveaxis(np.array(s_img), 0, -1))
                # rgb[counter:counter + ppb] = np.array(s_img)
                # counter += ppb
            pass

    print(f'mean values RGB = {np.mean(rgb, axis=(0, 2, 3))} | std values RGB = {np.std(rgb, axis=(0, 2, 3))}')
