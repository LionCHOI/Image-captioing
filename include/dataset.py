import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset

# image를 data에 사용하도록 변형하는 부분
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# Dataset을 만드는 부분
class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        # 필요한 변수 저장
        self.split_type = split_type
        self.transform = transform

        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))

    def __getitem__(self, index):
        # 이미지 path에서 이미지를 받아들인다.
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        
        # 만약 transform이 있다면 수행
        if self.transform is not None:
            img = self.transform(img)

        # 만약 train이라면
        if self.split_type == 'train':
            return torch.FloatTensor(img), torch.tensor(self.captions[index])

        # 만약 train이 아니라면
        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img), torch.tensor(self.captions[index]), torch.tensor(all_captions)

    def __len__(self):
        return len(self.captions)