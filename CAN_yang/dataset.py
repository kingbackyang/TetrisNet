import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils import load_config

config_file = 'config.yaml'
"""加载config文件"""
param = load_config(config_file)

image_train = param['train_image_path']
image_val = param['eval_image_path']
if image_train.endswith('.pkl'):
    print("load training files")
    with open(image_train, 'rb') as f:
        images_train = pkl.load(f)
elif image_train.endswith('.list'):
    with open(image_train, 'r') as f:
        lines_train = f.readlines()

if image_val.endswith('.pkl'):
    print("load validating files")
    with open(image_val, 'rb') as f:
        images_val = pkl.load(f)
elif image_val.endswith('.list'):
    with open(image_val, 'r') as f:
        lines_val = f.readlines()

with open(param['train_label_path'], 'r', encoding='utf-8', errors='ignore') as f:
    print("load training labels")
    labels_train = f.readlines()
with open(param['eval_label_path'], 'r', encoding='utf-8', errors='ignore') as f:
    print("load validating labels")
    labels_val = f.readlines()


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()

        if image_path == params['train_image_path']:
            self.images = images_train
            self.labels = labels_train
        elif image_path == params['eval_image_path']:
            self.images = images_val
            self.labels = labels_val
        # self.images = {}
        # print(f'data files: {lines}')
        # for line in lines:
        #     name = line.strip()
        #     print(f'loading data file: {name}')
        #     start = time.time()
        #     with open(name, 'rb') as f:
        #         images = pkl.load(f)
        #     self.images.update(images)
        #     print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        # with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        #     self.labels = f.readlines()
        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words, name


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=params['batch_size'], sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def collate_fn(batch_images):

    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path, encoding='gbk') as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}

if __name__ == "__main__":
    from utils import load_config
    config_file = 'config.yaml'
    """加载config文件"""
    params = load_config(config_file)
    get_crohme_dataset(params)