from export import export_model

#         >>> import deeplabcut
#         >>> deeplabcut.export_model(
#         >>>     "/analysis/project/reaching-task/config.yaml",
#         >>>     shuffle=3,
#         >>>     snapshotindex=-1,
#         >>> )

cfg_path = '/Users/fabionaecht/Documents/PhD/dlc/tensorflow_pytorch/refined/20230424_nn200_tip-sam/config.yaml'
shuffle = 1
snapshotindex = -1

export_model(cfg_path, shuffle=shuffle, snapshotindex=snapshotindex)
