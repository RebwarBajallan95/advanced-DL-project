import torch
import torch.nn as nn
import torch.nn.functional as F
import uncertainty_metrics as um
import numpy as np
import itertools



def eval(model, testloader):
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
        member_accuracies = [0 for i in range(model.ensemble_size)]
        member_losses = [0 for i in range(model.ensemble_size)]
        member_logits = []
        # diversity metrics
        pairwise_disagreement = {
                            "max": [],
                            "min": [],
                            "mean": []
                        }
        pairwise_kl_diversity = {
                            "max": [],
                            "min": [],
                            "mean": []
                        }
        # again no gradients needed
        with torch.no_grad():
            for x_test, y_test in testloader:

                # repeat input M times
                x_test = torch.cat(model.ensemble_size * [x_test])
                batch_size = x_test.size(dim=0) // model.ensemble_size
                x_test = x_test.reshape(batch_size, model.ensemble_size, 3, 32, 32) 

                # map to cuda if GPU available
                x_test = x_test.to(next(model.parameters()).device)
                y_test = y_test.to(next(model.parameters()).device)

                # pre-activations
                logits = model(x_test) 
                member_logits.append(logits)

                probs = F.softmax(logits, dim=2)
                log_probs = F.log_softmax(logits, dim=2)

                batch_disagreements, batch_kl_diversity = batch_diversity(probs)
                
                for key in ["max", "min", "mean"]:
                    pairwise_disagreement[key].append(batch_disagreements[f"disagreement-{key}"])
                    pairwise_kl_diversity[key].append(batch_kl_diversity[f"kl_diversity-{key}"])
                
                # calculate accuracy given each ensemble member
                for i in range(model.ensemble_size):
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

                ece = um.numpy.ece(labels=y_test.cpu(), probs=probs_mean.cpu(), num_bins=num_bins)
                _, preds = torch.max(probs_mean, 1)
                
                # calculate accuracy
                testset_size += y_test.size(dim=0)
                correct += (preds == y_test).sum().item()
                running_ece += ece
                running_loss += loss.item()

            accuracy = 100 * (correct / testset_size)
            member_accuracies = [100 * (acc/testset_size) for acc in member_accuracies]
            member_losses = [loss/test_iterations for loss in member_losses]
            running_loss /= test_iterations
            running_ece /= test_iterations
            print(f"Testing Accuracy: {accuracy}")
            print(f"Testing loss: {running_loss}")
            print(f"Testing ECE: {running_ece}")

            for key in ["max", "min", "mean"]:
                    pairwise_disagreement[key] = sum(pairwise_disagreement[key]) / testset_size
                    pairwise_kl_diversity[key] = sum(pairwise_kl_diversity[key]) / testset_size
   
            return accuracy, running_loss, running_ece, member_accuracies, member_losses, member_logits, diversity, pairwise_disagreement, pairwise_kl_diversity 



def batch_diversity(probs):
    """"
        probs: shape -> (ensemble_size, batch_size, number_classes)
    """

    ensemble_size = probs.size(dim=0)
    batch_disagreements = []
    batch_kl_diversity = []
    for subnet_pair in list(itertools.combinations(range(ensemble_size), 2)):
        probs1 = probs[subnet_pair[0]]
        probs2 = probs[subnet_pair[1]]
        batch_disagreements.append(
            disagreement(probs1, probs2).sum().item()
        )
        batch_kl_diversity.append(
            kl_diveristy(probs1, probs2).sum().item()
        )
    batch_disagreements = np.array(batch_disagreements)
    batch_kl_diversity = np.array(batch_kl_diversity)

    return {
            "disagreement-min": np.min(batch_disagreements), 
            "disagreement-max": np.max(batch_disagreements), 
            "disagreement-mean": np.mean(batch_disagreements)
            }, {
            "kl_diversity-min": np.min(batch_kl_diversity), 
            "kl_diversity-max": np.max(batch_kl_diversity), 
            "kl_diversity-mean": np.mean(batch_kl_diversity)
            }
        

def kl_diveristy(p, q):
    """
    """
    return torch.sum(p * torch.log(p / q), dim=-1)

def disagreement(probs1, probs2):
    """ 
    calculate disagreement given class probabilties
    return: amount of disagrements in predictions
    """
    _, preds1 = torch.max(probs1, 1)
    _, preds2 = torch.max(probs2, 1)
    return (preds1 != preds2)