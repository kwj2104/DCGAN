import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist(bsize):
    
    train_dataset = datasets.MNIST(root='./data/',
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = datasets.MNIST(root='./data/',
                               train=False, 
                               transform=transforms.ToTensor())
    
    # Treat the greyscale values as probabilities and sample to get binary data
    torch.manual_seed(3435)
#    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
#    train_label = torch.LongTensor([d[1] for d in train_dataset])
#    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
#    test_label = torch.LongTensor([d[1] for d in test_dataset])
    
    train_img = torch.stack([d[0] for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    test_img = torch.stack([d[0] for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])
    
    # Split training and val set
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:10000]
    train_label = train_label[:10000]
    
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)
    train_loader = torch.utils.data.DataLoader(train, batch_size=bsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=bsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=bsize, shuffle=True)

    return train_loader, val_loader, test_loader