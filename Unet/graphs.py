import matplotlib.pyplot as plt
import numpy as np
import json

def plot1(X, Y):
    plt.plot(X, Y[0], X, Y[1], X, Y[2], X, Y[3])
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.title('IoU during training for MIMO models\nwith different number of ensamble members')
    plt.xlim([0,72])
    plt.legend(['1 member','2 members','3 members','4 members'], loc = 'lower right')
    plt.savefig('./graph1.png')
    plt.show()
    return

def plot2(X, Y):
    plt.plot(X, Y[0], X, Y[1], X, Y[2])
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.title('IoU during training for the ensemble members and the MIMO model')
    plt.xlim([0,96])
    plt.legend(['MIMO','member 1','member 2'], loc = 'lower right')
    plt.savefig('./graph2.png')
    plt.show()
    return

def getBoxData(filenames):
    curr_iou = [0.5]
    for file in filenames:
        data = json.load(open('./data/train_data/' + file))
        for iou in data['IoUs']:
            curr_iou.append(iou)
    return curr_iou

def getEnsembleData(filenames, numEns):
    curr_ious = [[0.65] for _ in range(numEns)]
    for file in filenames:
        data = json.load(open('./data/train_data/' + file))
        for ious in data['member_ious']:
            for i in range(numEns):
                curr_ious[i].append(ious[i])
    return curr_ious

def readData():
    ious1 = []

    # Ensemble of 1
    filenames = ['1ens_24epochs.json', '1ens_48epochs.json', '1ens_72epochs.json']
    curr_iou = getBoxData(filenames)
    ious1.append(curr_iou)

    # Ensemble of 2
    filenames = ['2ens_24epochs.json', '2ens_48epochs.json', '2ens_72epochs.json']
    curr_iou = getBoxData(filenames)
    ious1.append(curr_iou)

    filenames2 = ['2ens_24epochs.json', '2ens_48epochs.json', '2ens_72epochs.json', '2ens_96epochs.json']
    curr_iou = getBoxData(filenames2)
    member_ious = getEnsembleData(filenames2, 2)
    ious2 = [curr_iou, member_ious[0], member_ious[1]]

    # Ensemble of 3
    filenames = ['3ens_24epochs.json', '3ens_48epochs.json', '3ens_72epochs.json']
    curr_iou = getBoxData(filenames)
    ious1.append(curr_iou)

    # Ensemble of 4
    filenames = ['4ens_24epochs.json', '4ens_48epochs.json', '4ens_72epochs.json']
    curr_iou = getBoxData(filenames)
    ious1.append(curr_iou)

    return ious1, ious2

if __name__ == "__main__":
    ious1, ious2 = readData()
    X1 = list(range(0,73))
    X2 = list(range(0,97))
    plot1(X1, ious1)
    plot2(X2, ious2)



