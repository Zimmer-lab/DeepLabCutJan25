from pathlib import Path

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import video_inference
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

if __name__ == '__main__':

    train_dir = Path("/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/dlc-models-pytorch/iteration-1/20230424_nn200_tipApr24-trainset95shuffle1/train/")
    pytorch_config_path = train_dir / "pytorch_config.yaml"
    snapshot_path = train_dir / "snapshot-200.pt"

    # video and inference parameters
    # video_path = Path("/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/test-videos/2023_04_22_15_00_tracking_trial_control_fabio_image_stack1.avi")
    video_path = Path("/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/evaluation_custom/200_frames_2023_04_22_15_00_tracking_trial_control_fabio_image_stack5-1.avi")
    max_num_animals = 1
    batch_size = 16

    # read model configuration
    model_cfg = read_config_as_dict(pytorch_config_path)
    bodyparts = model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
    with_identity = model_cfg["metadata"].get("with_identity", False)
    pose_task = Task(model_cfg["method"])
    # print(f"Pose task: {pose_task}")

    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=max_num_animals,
        num_bodyparts=len(bodyparts),
        num_unique_bodyparts=len(unique_bodyparts),
        batch_size=batch_size,
        with_identity=with_identity,
        transform=None,
        detector_transform=None,
    )

    predictions = video_inference(
        video=video_path,
        task=pose_task,
        pose_runner=pose_runner,
        detector_runner=detector_runner,
        # with_identity=False,
    )

    import numpy as np
    import pandas as pd

    print(f"Predictions: {predictions}")

    data = []
    for pred in predictions:
        x, y, likelihood = pred['bodyparts'][0][0]  # Access the x, y, likelihood values
        data.append([x, y, likelihood])

    # Convert to NumPy array
    data_array = np.array(data)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data_array, columns=['x', 'y', 'likelihood'])

    # Output the results
    print("NumPy Array:")
    print(data_array)
    csv_path = '/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/evaluation_custom/DLCJan25_analyze_videos.csv'
    predictions = pd.DataFrame(data_array, columns=['x', 'y', 'likelihood'])
    predictions.to_csv(csv_path, index=False)
