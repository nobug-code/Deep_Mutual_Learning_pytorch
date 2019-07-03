import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

def get_data_loader(args):
    dataset = args.dataset

    if(dataset == 'cifar10'):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=trans)
        test_dataset = dset.CIFAR10(root='./data', train=False, download=True, transform=trans)

        assert train_dataset
        assert test_dataset

        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        return train_dataloader, test_dataloader
