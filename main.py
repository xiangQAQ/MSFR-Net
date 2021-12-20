from Trainclass import MTrainer
from pro_p import create_IDanddir
import os



crop_train_dir = "online_dataset/norm_train/"
crop_val_dir = "online_dataset/norm_val/"

output_folder_name = 'output/'
deterministic = True

trainer = MTrainer(output_folder=output_folder_name,
                   train_dir=crop_train_dir,
                   val_dir=crop_val_dir,
                   deterministic=deterministic)


trainer.initialize()
trainer.run_training()