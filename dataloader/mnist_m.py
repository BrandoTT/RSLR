"""DataSet load for MNIST_m"""
import torch
from torchvision import datasets, transforms,utils
import os
import torch.utils.data as data
from PIL import Image
import os
import matplotlib.pyplot as plt

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


def get_loader_mnist_m(train, get_dataset, batch_size, image_size):
    """Get mnist_m"""
    
    target_dataset_name = 'mnist_m' 
    target_image_root = os.path.join('dataset', target_dataset_name)
    
    #image process
    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # train phase
    if train == True:
        #image loader
        train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
        dataset_target = GetLoader(
            data_root=os.path.join(target_image_root, 'mnist_m_train'),
            data_list=train_list,
            transform=img_transform
        )
        dataloader = torch.utils.data.DataLoader(#最终对目标域进行操作的数据是dataloader_target
            dataset=dataset_target,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
    # test phase
    elif train == False:
        pass
    return dataloader 

# if __name__=="__main__":
    
#     dataloader_svhn = get_loader_mnist_m(128, 28)
#     for batch_idx, (inputs, labels) in enumerate(dataloader_svhn):
#         fig = plt.figure()
#         inputs = inputs.detach().cpu()
#         grid = utils.make_grid(inputs)
#         print("Labels", labels)
#         plt.imshow(grid.numpy().transpose((1, 2, 0)))
#         plt.savefig('mnist_m.png')
#         break