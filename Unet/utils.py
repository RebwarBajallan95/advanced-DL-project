import torch
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import torchvision
from receipt_dataset import LowContrastReceipts
from torch.utils.data import DataLoader

TRAIN_DIR = "data/images/training"
VAL_DIR = "data/images/validation"
TRAIN_TRANSFORM = None
VAL_TRANSFORM = None
NUM_WORKERS = 0

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, "data/trained_nets/" + filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loader(batch_size, dir, width, height, transform, color, training, shuffle, scale, ensemble_size):
    ds = LowContrastReceipts(
        image_dir=dir,
        width=width,
        height=height,
        transform=transform,
        color=color,
        training=training,
        scale=scale,
        ensemble_size=ensemble_size
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=shuffle,
    )
    return loader

def get_loaders(width, height, batch_size, color="RGB", scale = 1, ensemble_size = 1):
    train_loader = get_loader(
        batch_size=batch_size,
        dir=TRAIN_DIR,
        width=width,
        height=height,
        transform=TRAIN_TRANSFORM,
        color=color,
        training=True,
        shuffle=True,
        scale=scale,
        ensemble_size=ensemble_size
    )

    val_loader = get_loader(
        batch_size=batch_size,
        dir=VAL_DIR,
        width=width,
        height=height,
        transform=VAL_TRANSFORM,
        color=color,
        training=False,
        shuffle=False,
        scale=scale,
        ensemble_size=ensemble_size
    )
    return train_loader, val_loader

def calc_IoU(pol1_xy, pol2_xy):
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Check that the output are valid polygons, first itteration problem
    if not polygon1_shape.is_valid or not polygon2_shape.is_valid:
        return 0

    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union

def get_points_member(b, e, w, h, y, s):
    y[y < 0] = 0
    y[y > s] = s

    p1 = [y[b,0,e,0].item()/s * w, y[b,0,e,1].item()/s * h]
    p2 = [y[b,0,e,2].item()/s * w, y[b,0,e,3].item()/s * h]
    p3 = [y[b,0,e,4].item()/s * w, y[b,0,e,5].item()/s * h]
    p4 = [y[b,0,e,6].item()/s * w, y[b,0,e,7].item()/s * h]

    return [p1, p2, p3, p4]

def get_points_mimo(w, h, y, s):
    y[y < 0] = 0
    y[y > s] = s

    p1 = [y[0].item()/s * w, y[1].item()/s * h]
    p2 = [y[2].item()/s * w, y[3].item()/s * h]
    p3 = [y[4].item()/s * w, y[5].item()/s * h]
    p4 = [y[6].item()/s * w, y[7].item()/s * h]

    return [p1, p2, p3, p4]

def check_IoUs(loader, model, device="cuda", scale=1, ensemble_size=1):
    model.eval()
    ious = np.zeros(ensemble_size)
    mimo_iou = 0;
    with torch.no_grad():
        num_test = 2
        curr = 0
        for x, y in loader:
            if curr >= num_test:
                break
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            for b in range(preds.shape[0]):
                # All correct are the same
                p1 = get_points_member(b, 0, 1224, 1632, y, scale)

                mean_prediction = preds[b,0,:,:].mean(0)
                p2 = get_points_mimo(1224, 1632, mean_prediction, scale)
                mimo_iou += calc_IoU(p1, p2)/preds.shape[0]

                for e in range(ensemble_size):
                    p2 = get_points_member(b, e, 1224, 1632, preds, scale)
                    curr_iou = calc_IoU(p1, p2)
                    ious[e] += curr_iou/preds.shape[0]
            curr += 1
        ious = ious/num_test
        mimo_iou = mimo_iou/num_test
        for i in range(ensemble_size):
            print(f'IoU for ensemble member {i}: {ious[i]:.4f}')
    print(f'Mimo IoU: {mimo_iou:.4f}')
    model.train()
    return mimo_iou, ious

def get_image(name, color, width, height, ensemble_size):
    full_rgb = Image.open(name).convert("RGB")
    if color == "RGB":
        input = full_rgb.resize((width,height))
        input = np.array(input, dtype=np.float32)
        input = input[..., np.newaxis]
    elif color == "Gray":
        input = full_rgb.resize((width,height)).convert("L")
        input = np.array(input, dtype=np.float32)
        input = input[..., np.newaxis, np.newaxis]

    input = np.transpose(input, (3, 2, 0, 1))
    inputs = []
    for i in range(ensemble_size):
        inputs.append(input)
    return full_rgb, np.concatenate(inputs, axis=1)

def save_BBox_img(loader, model, folder, device, color, width, height, scale, ensemble_size):
    model.eval()
    imgs = [
        "data/images/all/img_011.jpg", "data/images/all/img_023.jpg",
        "data/images/all/img_057.jpg", "data/images/all/img_106.jpg",
        "data/images/all/img_466.jpg", "data/images/all/img_499.jpg",
        "data/images/all/img_498.jpg", "data/images/all/img_467.jpg"
    ]
    for i in range(len(imgs)):
        true_img, x = get_image(imgs[i], color, width, height, ensemble_size)

        with torch.no_grad():
            x = torch.from_numpy(x)
            x = x.to(device)
            preds = model(x)
            mean_prediction = preds[0,0,:,:].mean(0)
            p1 = get_points_mimo(1224, 1632, mean_prediction, scale)
            p1 = np.round(p1)

            #true_img = true_img.resize((120,160))
            draw = ImageDraw.Draw(true_img)
            draw.line((p1[0][0], p1[0][1], p1[1][0], p1[1][1]), fill=128, width=10)
            draw.line((p1[1][0], p1[1][1], p1[2][0], p1[2][1]), fill=128, width=10)
            draw.line((p1[2][0], p1[2][1], p1[3][0], p1[3][1]), fill=128, width=10)
            draw.line((p1[3][0], p1[3][1], p1[0][0], p1[0][1]), fill=128, width=10)
            img_name = f'pred_img_{i}.jpg'
            true_img.save(folder + img_name)

    model.train()