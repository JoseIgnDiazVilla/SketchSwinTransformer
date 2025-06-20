# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Resize,
    Resized,
    EnsureTyped,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
import torch
import einops
import warnings
import glob
import json
import pprint


warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


def main():

    # Set Transforms
    
    num_samples = 4
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image", "label"],
                spatial_size=(340, 340, 340),
                mode=("trilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image", "label"],
                spatial_size=(340, 340, 340),
                mode=("trilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Resized(
                keys=["image"],
                spatial_size=(340, 340, 340),
                mode=("trilinear"),
            ),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
        ]
    )

    # Load Dataset

    dataset_json = {
        "labels": {
            "0": "background",
            "1": "cancer",
        },
        "tensorImageSize": "3D",
        "training": [],
        "validation": []
    }

    masks_paths = sorted(glob.glob('../data/masks/*.nii.gz'))
    for path in masks_paths[:-2]:
        filename = path.rsplit('/', 1)[-1]
        dataset_json["training"].append({
            "image": f'images/{filename}',
            "label": f'masks/{filename}',
        })
    for path in masks_paths[-2:]:
        filename = path.rsplit('/', 1)[-1]
        dataset_json["validation"].append({
            "image": f'images/{filename}',
            "label": f'masks/{filename}',
        })

    datasets = '../data/dataset.json'
    with open(datasets, 'w') as outfile:
        json.dump(dataset_json, outfile)

    pprint.pprint(dataset_json)

    train_files = load_decathlon_datalist(datasets, True, "training")
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=2
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)

    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=2
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    # test_files = load_decathlon_datalist(datasets, True, "test")
    # test_ds = CacheDataset(
    #     data=test_files, transform=test_transforms, cache_num=6, cache_rate=1.0, num_workers=2
    # )
    # test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)

    # Define Model

    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    set_track_meta(False)

    # Load Pretrained Weights

    weight = torch.load("../model/model_swinvit.pt")
    model.load_from(weights=weight)

    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps)" % (global_step, 1.0)
                )
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)"
                % (global_step, max_iterations, loss)
            )
            if (
                global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), "../data/best_metric_model.pth")
                    print(f"\nModel Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}")
                else:
                    print(f"\nModel Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}")
            global_step += 1
        return global_step, dice_val_best, global_step_best
    
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    max_iterations = 30000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=2) # class n
    post_pred = AsDiscrete(argmax=True, to_onehot=2) # class n
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    model.load_state_dict(torch.load("../data/best_metric_model.pth"))

    print(f"train completed, best_metric: {dice_val_best:.4f} at iteration: {global_step_best}")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.savefig("../results/loss.png")

    case_num = 0
    slice_num = 158

    model.load_state_dict(torch.load("../data/best_metric_model.pth"))
    model.eval()
    with torch.no_grad():
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(
            val_inputs, (96, 96, 96), 4, model, overlap=0.8
        )
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num], cmap="gray")
        ax1.set_title('Image')
        ax2.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num])
        ax2.set_title(f'Label')
        ax3.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num])
        ax3.set_title(f'Predict')
        plt.savefig("../results/predict.png")

if __name__ == "__main__":
    main()
