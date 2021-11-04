import torch
import functools
import torch.nn as nn
from torch import Tensor
import torch.optim as optim


# TODO: HOW TO ADD kernel_initializer='he_normal' in Pytorch?

# Pre-initialized functions
Conv2D = functools.partial(  
        nn.Conv2d,
        bias=False,
    )

BatchNorm = functools.partial(  
    nn.BatchNorm2d,
    eps=1e-5,  
    momentum=0.9)
    

class ResidualBlock(nn.Module):
    """
        Residual block
    """
    def __init__(
                self, 
                input_dim: int, 
                output_dim : int, 
                stride: int = 1
        ) -> None:
        super().__init__()
        self.bn1 = BatchNorm(output_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = Conv2D(input_dim, output_dim, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm(output_dim)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv2D(output_dim, output_dim, kernel_size=3, padding=1, stride=1)
        
        # Case when to perform downsampling
        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != output_dim:
            self.shortcut = nn.Sequential(
                Conv2D(input_dim, output_dim, kernel_size=1, stride=stride),
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
            Reisudual block forward pass
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shortcut(x)
        out = self.relu2(out)
        return out
        
       

class mimo_wide_resnet18(nn.Module):
    """
        Wide resnet18-k
    """    
    
    def __init__(
                self, 
                input_shape: torch.Tensor, # NOTE: Why Tensor?
                num_classes: int, 
                ensemble_size: int, 
                width_multiplier: int = 10
            ) -> None: 
        super().__init__()

        self.input_shape = list(input_shape)
        self.ensemble_size = ensemble_size
        self.num_classes = num_classes

        if ensemble_size != input_shape[0]:
            raise ValueError("The first dimension of input_shape must be ensemble_size")
        
        self.conv1 = Conv2D(3*ensemble_size, 16, kernel_size=3, stride=1, padding=1)
        # add the residual blocks
        group_blocks = list()
        group_blocks.append(ResidualBlock(input_dim=16, output_dim=16*width_multiplier, stride=2))
        group_blocks.append(ResidualBlock(input_dim=16*width_multiplier, output_dim=16*width_multiplier, stride=1))
        self.conv_group1 = nn.Sequential(*group_blocks)

        group_blocks = list()
        group_blocks.append(ResidualBlock(input_dim=16*width_multiplier, output_dim=32*width_multiplier, stride=2))
        group_blocks.append(ResidualBlock(input_dim=32*width_multiplier, output_dim=32*width_multiplier, stride=1))
        self.conv_group2 = nn.Sequential(*group_blocks)

        group_blocks = list()
        group_blocks.append(ResidualBlock(input_dim=32*width_multiplier, output_dim=64*width_multiplier, stride=2))
        group_blocks.append(ResidualBlock(input_dim=64*width_multiplier, output_dim=64*width_multiplier, stride=1))
        self.conv_group3 = nn.Sequential(*group_blocks)
    
        self.bn1 = BatchNorm(64*width_multiplier)
        self.relu1 = nn.ReLU(inplace=True)
        #self.avg_pool = nn.AvgPool2d(kernel_size=8) # NOTE: This didn't work, why?
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # NOTE: Does this do the same thing as above?
        # classification layers
        self.fc1 = nn.Linear(64*width_multiplier, num_classes, bias=True)
        self.outlayer = DenseMultihead(
                        input_size=64*width_multiplier, 
                        nr_units=self.num_classes,
                        ensemble_size=self.ensemble_size
                    )


    def forward(self, input: Tensor) -> Tensor:
        """
            Resnet18-k forward pass
        """
    
        # batchsize needed when reshaping
        batch_size = input.size(dim=0)

        #x = torch.permute(input, dims=(0, 1, 2, 3, 4)) # NOTE: Why permute?
        x = torch.reshape(input, shape=[batch_size, self.input_shape[1] * self.ensemble_size] + self.input_shape[2:])       

        x = self.conv1(x)
        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.outlayer(x) 
        return x

    def fit(self, trainloader, epochs=10, verbose=True):
        """
            Function for training the network
        """

        # TODO: DEFINE TRAINING PARAMETERS HERE

        criterion = nn.CrossEntropyLoss()
        # same paramters as used in the paper
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        # Dataloaders
        #train_loader = data.train_loader
        #val_loader = data.val_loader

        training_losses = []
        training_accuracy = []
      
        for epoch in range(epochs):  
            # training mode
            self.train()
            # Training loss
            training_loss = 0
            # Accuracy variables
            correct = 0
            total = 0
            for x, y in trainloader:   

                BATCHSIZE = 150

                x = torch.cat(self.ensemble_size*[x]) # TODO: SAMPLE self.ensemble_size DATAPOINTS RANDOMLY
                x = x.reshape(BATCHSIZE, self.ensemble_size, 3, 32, 32)

                optimizer.zero_grad()
                # Forward propagation
                outputs = self(x)

                # We want (ensamble, batchsize, class-predictions)
                outputs = outputs.reshape(self.ensemble_size, BATCHSIZE, 10)
                outputs = torch.mean(outputs, dim=0) # NOTE: USE WHEN EVALUATING
              
                loss = criterion(outputs, y)
                # Backward propagation
                loss.backward()
                optimizer.step()
                # Training loss
                training_loss+= loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
    
            # Training accuracy
            train_acc =  100 * (correct / total)
            # Training loss
            training_loss /= len(trainloader)
            
            # Print training loss
            if verbose:
                print(f'Training Loss: {training_loss}')
                print(f'Training Accuracy: {train_acc}')

            # Store training scores
            training_losses.append(training_loss) 
            training_accuracy.append(train_acc)

        return None
        

class DenseMultihead(torch.nn.Linear):
    """ 
        Multiheaded output layer 
    """

    def __init__(
            self, 
            input_size: int,
            nr_units: int,
            ensemble_size:int = 1,
    ) -> None:
        super().__init__(
            in_features=input_size,
            out_features=nr_units*ensemble_size
        )
        self.ensemble_size = ensemble_size

    def forward(self, inputs):
        """ """
        batch_size = inputs.size(dim=0)
        outputs = super().forward(inputs)
        outputs = torch.reshape(
            outputs,
            [batch_size, self.ensemble_size, self.out_features // self.ensemble_size])
        return outputs



    