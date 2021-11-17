import torch
import functools
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
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
        out += self.shortcut(x)
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
                batch_repitition: int,
                ensemble_size: int, 
                width_multiplier: int = 10
            ) -> None: 
        super().__init__()

        self.input_shape = list(input_shape)
        self.ensemble_size = ensemble_size
        self.num_classes = num_classes
        self.batch_repitition = batch_repitition

        if ensemble_size != input_shape[0]:
            raise ValueError("The first dimension of input_shape must be ensemble_size")
        
        self.conv1 = Conv2D(3*ensemble_size, 16, kernel_size=3, stride=1, padding=1)
        # add the residual blocks
        group_blocks = list()
        group_blocks.append(ResidualBlock(input_dim=16, output_dim=16*width_multiplier, stride=1))
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
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        # classification layers
        self.outlayer = DenseMultihead(
                        input_size=64*width_multiplier, 
                        nr_units=self.num_classes,
                        ensemble_size=self.ensemble_size
                    )
        # initialize weights
        self = self.apply(self.weight_init)
        # store loggings
        self.running_stats = dict()
                               
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
            Resnet18-k forward pass
        """
        # batchsize needed when reshaping
        batch_size = input.size(dim=0)
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
                        weight_decay=3e-4, 
                        nesterov=True
                    )
        # TODO: REDO THIS TO REFELECT THE LEARNING RATE DECAY IN THE PAPER
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        training_iterator = iter(trainloader)
        for _ in range(epochs):

            epoch = list(self.running_stats.keys())[-1] + 1 if len(self.running_stats) > 0 else 0
            # loggings
            self.running_stats[epoch] = {
                    "Training loss": [],
                    "Testing loss": [],
                    "Testing Accuracy": [],
                    "Testing ECE": [],
                }
            print("Epoch: ", epoch)
            # training mode
            self.train()
            # Training loss
            training_loss = 0
            for _ in tqdm(range(steps_per_epoch)):
                xs = []
                ys = []
                training_iterator = iter(trainloader)
                for _ in range(self.ensemble_size):
                    # get random sample batch from training set
                    x, y = next(training_iterator)
                    # map to cuda if GPU available
                    x = x.to(next(self.parameters()).device)
                    y = y.to(next(self.parameters()).device)
                    # repeat the batches 'self.batch_repitition' times
                    xs.append(torch.cat(self.batch_repitition * [x]))
                    ys.append(torch.cat(self.batch_repitition * [y]))

                x = torch.cat(xs)
                batch_size = x.size(dim=0) // self.ensemble_size
                x = x.reshape(batch_size, self.ensemble_size, 3, 32, 32) 
                optimizer.zero_grad()
                # Forward propagation
                logits = self(x)  
                log_probs = F.log_softmax(logits, dim=2)
                
                loss = 0
                for i in range(self.ensemble_size):
                    loss += criterion(log_probs[i], ys[i])

                # Backward propagation
                loss.backward()
                optimizer.step()
                # Training loss
                training_loss+= loss.item()
                        
            scheduler.step()
            # Print training loss
            if verbose:
                print(f'Training Loss: {training_loss}')
        
            # Evaluate network
            test_acc, test_loss, test_ece = self.eval(testloader)

            self.running_stats[epoch]["Training loss"].append(training_loss)
            self.running_stats[epoch]["Testing Accuracy"].append(test_acc)
            self.running_stats[epoch]["Testing loss"].append(test_loss)
            self.running_stats[epoch]["Testing ECE"].append(test_ece)

            # save model
            if (epochs % save_mode_epochs == 0) and epochs != 0:
                torch.save(self, "models/resnet18_10.pt") 

            
    def eval(self, testloader):
        """ 
            Evaluate network
        """
        # ECE number of bins
        num_bins = 15
        # negative log-likelihood loss
        criterion = nn.NLLLoss()
        correct = 0
        total = 0
        ece = 0
        # again no gradients needed
        with torch.no_grad():
            for x_test, y_test in testloader:

                # repeat input M times
                x_test = torch.cat(self.ensemble_size * [x_test])
                batch_size = x_test.size(dim=0) // self.ensemble_size
                x_test = x_test.reshape(batch_size, self.ensemble_size, 3, 32, 32) 

                # map to cuda if GPU available
                x_test = x_test.to(next(self.parameters()).device)
                y_test = y_test.to(next(self.parameters()).device)

                logits = self(x_test)
                # calculate mean across ensembles
                logits = torch.mean(logits, dim=0) 
                # testing loss
                loss = criterion(F.log_softmax(logits, dim=1), y_test)

                probs = F.softmax(logits, dim=1)
                ece = um.numpy.ece(labels=y_test.cpu(), probs=probs.cpu(), num_bins=num_bins)
                _, preds = torch.max(probs, 1)
                
                # calculate accuracy
                total += y_test.size(0)
                correct += (preds == y_test).sum().item()
                ece += ece

            accuracy = 100 * (correct / total)
            loss = loss.item()
            print(f"Testing Accuracy: {accuracy}")
            print(f"Testing loss: {loss}")
            print(f"Testing ECE: {ece}")

            return accuracy, loss, ece

            

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



    