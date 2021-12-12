import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from mimo_resnet28_10 import mimo_wide_resnet28



def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    torch.backends.cudnn.benchmark = True

    # set seed
    torch.manual_seed(0)
    # set torch.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Print to see if you have GPU available, if no GPU => change colab runtime
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    cifar10_mean = (0.4914, 0.4822, 0.4465) 
    cifar10_std = (0.2470, 0.2435, 0.2616)

    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)])

    batch_size = 100

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
                                            trainset, 
                                            batch_size=batch_size,
                                            shuffle=True, 
                                            pin_memory=True,
                                            drop_last=True, 
                                            
                                        )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
                                        
    testloader = torch.utils.data.DataLoader(
                                        testset, 
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mimo_wide_resnet28(
        input_shape=(3, 3, 32, 32), # NOTE: PyTorch expects (n_samples, channels, height, width)
        width_multiplier=10,
        num_classes=10,
        ensemble_size=3,
        batch_repitition=4
    ).to(device)

    torch.cuda.empty_cache()
    model.fit(
            trainloader, 
            testloader, 
            epochs=250, 
            trainset_size=len(trainset), 
            batch_size=batch_size
        )

if __name__ == "__main__":
    main()