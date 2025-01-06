from export import export_model
from pathlib import Path
import torch
import deeplabcut.pose_estimation_pytorch.models as dlc_models


def export_store(cfg_path: str|Path, shuffle: int = 1, snapshotindex: int = -1):
    export_model(cfg_path, shuffle=shuffle, snapshotindex=snapshotindex)

if __name__ == '__main__':
    cfg_path = '/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/config.yaml'
    # export_store(cfg_path)

    exported_model_path = '/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/exported-models-pytorch/DLC_20230424_nn200_tip_resnet_50_iteration-1_shuffle-1/DLC_20230424_nn200_tip_resnet_50_iteration-1_shuffle-1_snapshot-200.pt'
    # Load the exported model
    model_data = torch.load(exported_model_path, map_location="cpu")
    model_cfg = model_data["config"]
    pose_weights = model_data["pose"]
    print(f"model data: {model_data}")
    print(f"model cfg: {model_cfg}")
    print(f"pose weights: {pose_weights}")

    # # Initialize the model
    # model = dlc_models.DLCPoseModel(model_cfg)
    # model.load_state_dict(pose_weights)
    # model.eval()  # Set model to evaluation mode

