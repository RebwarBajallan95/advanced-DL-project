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
from copy import deepcopy


# Pre-initialized functions
Conv2D = functools.partial(  
        nn.Conv2d,
        padding="same",
        bias=False
    )

BatchNorm = functools.partial(  
    nn.BatchNorm2d,
    eps=1e-5,  
    momentum=0.9)
    

       
class mimo_smallcnn(nn.Module):

    def __init__(
                self, 
                input_shape: torch.Tensor, # NOTE: Why Tensor?
                num_classes: int, 
                batch_repitition: int,
                ensemble_size: int, 
            ) -> None: 
        super().__init__()

        self.input_shape = list(input_shape)
        self.ensemble_size = ensemble_size
        self.num_classes = num_classes
        self.batch_repitition = batch_repitition

        if ensemble_size != input_shape[0]:
            raise ValueError("The first dimension of input_shape must be ensemble_size")
        
        self.relu = nn.ReLU(inplace=False)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout(p=0.1, inplace=False)

        self.conv1 = Conv2D(3*ensemble_size, 16, kernel_size=3, padding="same")
        self.conv2 = Conv2D(16, 32, kernel_size=3, padding="same")
        self.conv3 = Conv2D(32, 32, kernel_size=3, padding="same")
        self.conv4 = Conv2D(32, 32, kernel_size=3, padding="same")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.avg_pool = nn.AvgPool2d(kernel_size=8)

        # classification layers
        self.fc1 = nn.Linear(32, num_classes, bias=True)

        # classification layers
        self.outlayer = DenseMultihead(
                        input_size=32, 
                        nr_units=self.num_classes,
                        ensemble_size=self.ensemble_size
                    )
        # initialize weights
        self = self.apply(self.weight_init)
        
        # store loggings
        self.running_stats = dict()
                               
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
            Resnet18-k forward pass
        """
        # batchsize needed when reshaping
        batch_size = input.size(dim=0)
        x = torch.reshape(input, shape=[batch_size, self.input_shape[1] * self.ensemble_size] + self.input_shape[2:])       
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.avg_pool(x)
        x = self.relu(x)
        x = self.drop_out(x)

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
                        lr=1.6*1e-3, 
                        momentum=0.9,
                        weight_decay=3e-4, 
                        nesterov=True
                    )
        # epochs at which to decay learning rate
        epochs_for_decay = [30, 50, 80]
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # TODO: Is decay rate 0.1 (paper) or 0.2 (code)??
        training_iterator = iter(trainloader)
        for _ in range(epochs):
            
            output_layer_start = torch.reshape(self.outlayer.weight, 
                    (self.ensemble_size, self.num_classes, -1)
                ).cpu().detach().numpy()

            print("Learning rate: ", scheduler.get_last_lr())
            epoch = list(self.running_stats.keys())[-1] + 1 if len(self.running_stats) > 0 else 0
            print("Epoch: ", epoch)
            # training mode
            self.train()
            # Training loss
            training_loss = 0
            for _ in (range(steps_per_epoch)):
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
                new_batch_size = x.size(dim=0) // self.ensemble_size
                x = x.reshape(new_batch_size, self.ensemble_size, 3, 32, 32) 
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
                # save output layer weights
                outlayer_weights = torch.reshape(
                            self.outlayer.weight, 
                            (self.ensemble_size, self.num_classes, -1)
                        )
                      
            scheduler.step()

            output_layer_end = torch.reshape(self.outlayer.weight, 
                    (self.ensemble_size, self.num_classes, -1)
                ).cpu().detach().numpy()
                
            weight_change = linearly_interpolate(
                                            output_layer1=output_layer_start, 
                                            output_layer2=output_layer_end,
                                            nr_ensembles=self.ensemble_size
                                        )
      
            weight_delta = {}
            for ens in range(self.ensemble_size):
                weight_delta[ens] = {}
                weight_delta[ens]["w1"] = weight_change[ens][0][0]
                weight_delta[ens]["w2"] = weight_change[ens][0][1]

            # Print training loss
            if verbose:
                print(f'Training Loss: {training_loss/steps_per_epoch}')
        
            # Evaluate network
            test_acc, test_loss, member_accuracies, member_losses, probs = self.eval(testloader)

            # loggings
            self.running_stats[epoch] = {}
            self.running_stats[epoch]["Training loss"] = training_loss/steps_per_epoch
            self.running_stats[epoch]["Training weight delta"] = weight_delta
            self.running_stats[epoch]["Testing Accuracy"] = test_acc
            self.running_stats[epoch]["Testing loss"] = test_loss
            self.running_stats[epoch]["Testing Accuracies"] = member_accuracies
            self.running_stats[epoch]["Testing losses"] = member_losses
            self.running_stats[epoch]["Testing class-predictions"] = probs

            # save model
            torch.save(self, "models/smallcnn.pt") 

            
    def evaluate(self, testloader):
        """ 
            Evaluate network
        """
        test_iterations = len(testloader)
        testset_size = 0
        # negative log-likelihood loss
        criterion = nn.NLLLoss()
        correct = 0
        running_loss = 0
        member_accuracies = [0 for i in range(self.ensemble_size)]
        member_losses = [0 for i in range(self.ensemble_size)]
        # store probabilities for each test sample
        all_probs = []
        self.eval()
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
                probs = F.softmax(logits, dim=2)
                all_probs.append(probs)
                log_probs = F.log_softmax(logits, dim=2)

                # calculate accuracy given each ensemble member
                for i in range(self.ensemble_size):
                    member_probs = probs[i]
                    _, member_preds = torch.max(member_probs, 1)
                    member_correct = (member_preds == y_test).sum().item()
                    member_accuracies[i] += member_correct
                    member_loss = criterion(log_probs[i], y_test).item()
                    member_losses[i] += member_loss

                # calculate mean across ensembles
                log_probs_mean = torch.mean(log_probs, dim=0) 
                probs_mean = torch.mean(probs, dim=0)
                
                # testing loss
                loss = criterion(log_probs_mean, y_test)

                _, preds = torch.max(probs_mean, 1)
                # calculate accuracy
                testset_size += y_test.size(dim=0)
                correct += (preds == y_test).sum().item()
                running_loss += loss.item()

            accuracy = 100 * (correct / testset_size)
            member_accuracies = [100 * (acc/testset_size) for acc in member_accuracies]
            member_losses = [loss/test_iterations for loss in member_losses]
            running_loss /= test_iterations
            print(f"Testing Accuracy: {accuracy}")
            print(f"Testing loss: {running_loss}")
            
            return accuracy, running_loss, member_accuracies, member_losses, all_probs

            

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


def linearly_interpolate(
                output_layer1: np.ndarray, 
                output_layer2: np.ndarray, 
                nr_ensembles: int
            ):
    """
    """
    delta_dir_ens = {}
    for ens in range(nr_ensembles):
        params1 = [w for w in output_layer1[ens]]
        params2 = [w for w in output_layer2[ens]]
        # calculate the direction of which the weights move
        delta_dir_ens[ens] = [(w2 - w1) for w2, w1 in zip(params1, params2)]
        
    return delta_dir_ens