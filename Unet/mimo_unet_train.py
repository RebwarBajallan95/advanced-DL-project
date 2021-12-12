import torch
import json
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from mimo_unet_model import MimoUNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_IoUs,
    save_BBox_img
)

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 24
NUM_WORKERS = 0
NUM_ENSEMBLES = 1
LOAD_MODEL = False
COLOR = "RGB"
WIDTH = 120
HEIGHT = 160
SCALE = 1

# def train_fn(loader, model, optimizer, loss_fn, scaler):
def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    epoch_losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        epoch_losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return epoch_losses


def main():
    if COLOR == "RGB":
        model = MimoUNET(in_channels=3, out_channels=1, scaling=SCALE, ensemble_size=NUM_ENSEMBLES).to(DEVICE)
    else:
        model = MimoUNET(in_channels=1, out_channels=1, scaling=SCALE, ensemble_size=NUM_ENSEMBLES).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(WIDTH, HEIGHT, BATCH_SIZE, COLOR, SCALE, NUM_ENSEMBLES)

    if LOAD_MODEL:
        # load_checkpoint(torch.load("data/trained_nets/best.pth.tar"), model)
        load_checkpoint(torch.load("data/trained_nets/fully_trained.pth.tar"), model)

    check_IoUs(val_loader, model, device=DEVICE, scale=SCALE, ensemble_size=NUM_ENSEMBLES)
    save_BBox_img(
        val_loader, model, "data/prediction_images/", DEVICE, COLOR, WIDTH, HEIGHT, SCALE, NUM_ENSEMBLES
    )

    out_json = {}
    losses = []
    IoUs = []
    member_ious = []
    best_IoU = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f'==== Epoch {epoch+1} ====')
        losses.append(train_fn(train_loader, model, optimizer, loss_fn))

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # check accuracy
        IoU, curr_member_ious = check_IoUs(val_loader, model, device=DEVICE, scale=SCALE, ensemble_size=NUM_ENSEMBLES)
        if IoU > best_IoU:
            save_checkpoint(checkpoint, filename = "best.pth.tar")
            best_IoU = IoU

        # print some examples to a folder
        save_BBox_img(
            val_loader, model, "data/prediction_images/", DEVICE, COLOR, WIDTH, HEIGHT, SCALE, NUM_ENSEMBLES
        )

        IoUs.append(IoU)
        member_ious.append(curr_member_ious.tolist())

    save_checkpoint(checkpoint, filename = "fully_trained.pth.tar")
    out_json["losses"] = losses
    out_json["IoUs"] = IoUs
    out_json["member_ious"] = member_ious
    json.dump(out_json, open("data/train_data/train_data.json", 'w'))
    print("=================")


if __name__ == "__main__":
    main()