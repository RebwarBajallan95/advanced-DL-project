import torch
import functools
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import uncertainty_metrics as um
from collections import defaultdict


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
        self.bn1 = BatchNorm(input_dim)
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
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out
        
       
class wide_resnet28(nn.Module):
    """
        Wide resnet28-k
    """    
    
    def __init__(
                self, 
                num_classes: int, 
                width_multiplier: int = 10,
                depth: int = 28
            ) -> None: 
        super().__init__()

        self.num_classes = num_classes
        self.num_blocks = (depth - 4) // 6

        self.conv1 = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
        # add the residual blocks
        self.conv_group1 = self.residual_group_block(16, 16*width_multiplier, stride=1) 
        self.conv_group2 = self.residual_group_block(16*width_multiplier, 32*width_multiplier, stride=2) 
        self.conv_group3 = self.residual_group_block(32*width_multiplier, 64*width_multiplier, stride=2)
    
        self.bn1 = BatchNorm(64*width_multiplier)
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        # classification layers
        self.fc1 = nn.Linear(64*width_multiplier, num_classes, bias=True)

        # initialize weights
        self = self.apply(self.weight_init)
        
        # store loggings
        self.running_stats = dict()
                               
    
    def residual_group_block(self, input_dim, output_dim, stride):
        residual_layers = list()
        residual_layers.append(ResidualBlock(input_dim, output_dim, stride=stride))
        for _ in range(self.num_blocks - 1):
            residual_layers.append(ResidualBlock(output_dim, output_dim, stride=1))
        return nn.Sequential(*residual_layers)
    
    # NOTE: Check if this is correct
    def weight_init(self, layer):
        """
            Layer wight initialization
        """
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
        if isinstance(layer, DenseMultihead):
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
            Resnet28-k forward pass
        """
        x = self.conv1(input)
        x = self.conv_group1(x)
        x = self.conv_group2(x)
        x = self.conv_group3(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x) 
        return x

    def fit(
        self, 
        trainloader,
        testloader, 
        batch_size, 
        trainset_size,
        epochs=10, 
        save_mode_epochs=25,
        verbose=True
        ):
        """
            Function for training the network
        """
        # steps per epoch
        steps_per_epoch = trainset_size // batch_size
        # negative log-likelihood loss
        criterion = nn.NLLLoss()
        # same paramters as used in the paper
        optimizer = optim.SGD(
                        self.parameters(), 
                        lr=0.1, 
                        momentum=0.9,
                        weight_decay=2e-4, 
                        nesterov=True
                    )
        # epochs at which to decay learning rate
        epochs_for_decay = [80, 160, 180]
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # TODO: Is decay rate 0.1 (paper) or 0.2 (code)??
        for _ in range(epochs):
            epoch = list(self.running_stats.keys())[-1] + 1 if len(self.running_stats) > 0 else 0
            print("Epoch: ", epoch)
            # training mode
            self.train()
            # Training loss
            training_loss = 0
            # training step
            for x, y in trainloader:
                
                # map to cuda if GPU available
                x = x.to(next(self.parameters()).device)
                y = y.to(next(self.parameters()).device)
                    
                optimizer.zero_grad()
                # Forward propagation
                logits = self(x)  
                log_probs = F.log_softmax(logits, dim=1)
                
                loss = criterion(log_probs, y)
                # Backward propagation
                loss.backward()
                optimizer.step()
                # Training loss
                training_loss+= loss.item()
                        
            if epoch in epochs_for_decay: scheduler.step()
            # Print training loss
            if verbose:
                print(f'Training Loss: {training_loss/steps_per_epoch}')
        
            # Evaluate network
            test_acc, test_loss, test_ece = self.eval(testloader)

            # loggings
            self.running_stats[epoch] = {}
            self.running_stats[epoch]["Training loss"] = training_loss/steps_per_epoch
            self.running_stats[epoch]["Testing Accuracy"] = test_acc
            self.running_stats[epoch]["Testing loss"] = test_loss
            self.running_stats[epoch]["Testing ECE"] = test_ece
 
            # save model
            torch.save(self, "models/resnet18_10_deter.pt") 

            
    def evaluate(self, testloader):
        """ 
            Evaluate network
        """
        test_iterations = len(testloader)
        testset_size = 0
        # ECE number of bins
        num_bins = 15
        # negative log-likelihood loss
        criterion = nn.NLLLoss()
        correct = 0
        total = 0
        running_ece = 0
        running_loss = 0
        self.eval()
        with torch.no_grad():
            for x_test, y_test in testloader:

                # map to cuda if GPU available
                x_test = x_test.to(next(self.parameters()).device)
                y_test = y_test.to(next(self.parameters()).device)

                logits = self(x_test)
                probs = F.softmax(logits, dim=1)
                log_probs = F.log_softmax(logits, dim=1)

                # testing loss
                loss = criterion(log_probs, y_test)

                ece = um.numpy.ece(labels=y_test.cpu(), probs=probs.cpu(), num_bins=num_bins)
                _, preds = torch.max(probs, 1)
                
                # calculate accuracy
                testset_size += y_test.size(dim=0)
                correct += (preds == y_test).sum().item()
                running_ece += ece
                running_loss += loss.item()

            accuracy = 100 * (correct / testset_size)
            running_loss /= test_iterations
            running_ece /= test_iterations
            print(f"Testing Accuracy: {accuracy}")
            print(f"Testing loss: {running_loss}")
            print(f"Testing ECE: {running_ece}")
     
            return accuracy, running_loss, running_ece

            

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
            [self.ensemble_size, batch_size, self.out_features // self.ensemble_size])
        return outputs
