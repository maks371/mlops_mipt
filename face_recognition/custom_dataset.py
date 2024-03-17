import os
from PIL import Image
from torch.utils.data import Dataset


def get_paths(celeb_a_path, dataset_type='train'):

    labels_dict = {
        'train': 0,
        'val': 1,
        'test': 2,
    }

    with open(f'{celeb_a_path}/celebA_train_split.txt', 'r') as f:
        lines = f.readlines()

    lines = [x.strip().split() for x in lines]
    lines = [x[0] for x in lines if int(x[1]) == labels_dict[dataset_type]]

    images_paths = []
    for filename in lines:
        images_paths.append(os.path.join(f'{celeb_a_path}/celebA_imgs/', filename))

    with open(f'{celeb_a_path}/celebA_anno.txt', 'r') as f:
        labels = f.readlines()

    labels = [x.strip().split() for x in labels]
    labels = {x: y for x, y in labels}
    labels = [labels[x.split('/')[-1]] for x in images_paths]

    return images_paths, labels


class CelebADataset(Dataset):
    """Dataset for CelebA"""

    def __init__(self, celeb_a_path, dataset_type, transform, aug=None):

        self.images, self.labels = get_paths(celeb_a_path, dataset_type)

        self.transform = transform
        self.aug = aug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = int(self.labels[idx])

        image = Image.open(img_name)

        if self.aug:
            sample = self.aug(
                image=image,
            )
        else:
            sample = {
                'image': image,
                'label': label,
            }

        sample['image'] = self.transform(sample['image'])

        return sample
