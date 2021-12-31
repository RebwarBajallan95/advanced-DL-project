import json
import copy
import numpy as np
import torch
from PIL import Image, ImageDraw
from mimo_unet_model import MimoUNET
from utils import (
    load_checkpoint,
    get_points_mimo,
    get_points_member
)

def getBox(data):
    width = 1224
    height = 1632

    p1 = data["top_left"]
    p2 = data["top_right"]
    p3 = data["bottom_right"]
    p4 = data["bottom_left"]

    parr = [
        p1[0]/width, p1[1]/height, p2[0]/width, p2[1]/height,
        p3[0]/width, p3[1]/height, p4[0]/width, p4[1]/height
    ]
    return np.array(parr, dtype=np.float32)

def getInputAndImg(name, num_ens):
    width = 120
    height = 160

    full_rgb = Image.open(name).convert("RGB")
    input = full_rgb.resize((width,height))
    input = np.array(input, dtype=np.float32)
    input = input[..., np.newaxis]

    input = np.transpose(input, (3, 2, 0, 1))
    inputs = []
    for i in range(num_ens):
        inputs.append(input)
    return full_rgb, np.concatenate(inputs, axis=1)

def getData(num_ens, num):
    json_file = "data/data_info.json"
    json_dict = json.load(open(json_file))
    data_arr = json_dict["data_info"]
    image, input = getInputAndImg(data_arr[num]["image_name"], num_ens)
    label = getBox(data_arr[num])
    return input, label, image

def drawBox(color, p, img):
    draw = ImageDraw.Draw(img)
    draw.line((p[0][0], p[0][1], p[1][0], p[1][1]), fill=color, width=10)
    draw.line((p[1][0], p[1][1], p[2][0], p[2][1]), fill=color, width=10)
    draw.line((p[2][0], p[2][1], p[3][0], p[3][1]), fill=color, width=10)
    draw.line((p[3][0], p[3][1], p[0][0], p[0][1]), fill=color, width=10)

def genImgs():
    NUM_ENSEMBLES = 2
    net_path = 'data/trained_nets/2ens_96epochs.pth.tar'
    model = MimoUNET(in_channels=3, out_channels=1, scaling=1, ensemble_size=NUM_ENSEMBLES).to('cpu')
    load_checkpoint(torch.load(net_path), model)

    # 412, 420, 421, 425
    x, y, img = getData(NUM_ENSEMBLES, 412)
    true_img = copy.deepcopy(img)
    p = get_points_mimo(1224, 1632, y, 1)
    p = np.round(p)
    drawBox((200, 0, 0), p, true_img)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x)
        x = x.to('cpu')
        preds = model(x)
        mimo_pred = preds[0,0,:,:].mean(0)

        # Draw ensemble 1 prediction
        p1 = get_points_member(0, 0, 1224, 1632, preds, 1)
        p1 = np.round(p1)
        drawBox((0, 0, 200), p1, img)

        # Draw ensemble 2 prediction
        p2 = get_points_member(0, 1, 1224, 1632, preds, 1)
        p2 = np.round(p2)
        drawBox((0, 100, 0), p2, img)

        # Draw mimo prediction
        p = get_points_mimo(1224, 1632, mimo_pred, 1)
        p = np.round(p)
        drawBox((200, 0, 0), p, img)

        # color = (200, 0, 0) #red
        # color = (0, 0, 200) #blue
        # color = (0, 100, 0) #green
    model.train()

    true_img.save('data/example_predictions/out1.jpg')
    img.save('data/example_predictions/out2.jpg')

if __name__ == "__main__":
    genImgs()