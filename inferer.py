import os
import matplotlib.pyplot as plt
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    EnsureTyped,
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np
import torch
import warnings
import argparse


warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--model_out_channels', type=int, required=True, help="model out channels")
parser.add_argument('-w', '--model_weights_path', type=str, required=True, help="model weights path")
parser.add_argument('-d', '--data_path', type=str, required=True, help="data path")
parser.add_argument('-p', '--prediction_path', type=str, required=True, help="prediction path")
parser.add_argument('-v', '--visualization_path', type=str, required=True, help="visualization path")
args = parser.parse_args()

model_out_channels = args.model_out_channels
model_weights_path = args.model_weights_path
data_path = args.data_path
prediction_path = args.prediction_path
visualization_path = args.visualization_path


test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        EnsureTyped(keys=["image"], device=device, track_meta=True),
    ]
)
data = test_transforms({
    "image": data_path
})

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=model_out_channels,
    feature_size=48,
    use_checkpoint=True,
)
print('Loading model...')
model.load_state_dict(torch.load(model_weights_path))
model.to(device)
model.eval()

with torch.no_grad():
    print('Predicting masks...')
    test_inputs = torch.unsqueeze(data["image"], 1).to(device)
    test_outputs = sliding_window_inference(
        test_inputs, (96, 96, 96), 4, model, overlap=0.8
    )

    test_inputs = test_inputs.cpu().numpy()
    test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()

slice_rate = 0.5
slice_num = int(test_inputs.shape[-1]*slice_rate)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(test_inputs[0, 0, :, :, slice_num], cmap="gray")
ax1.set_title('Image')
ax2.imshow(test_outputs[0, :, :, slice_num])
ax2.set_title(f'Predict')
plt.savefig(visualization_path, bbox_inches='tight')

test_outputs = nib.Nifti1Image(test_outputs, affine=np.eye(4))
nib.save(test_outputs, prediction_path)