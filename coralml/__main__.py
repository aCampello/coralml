import argparse
import json
import os

from coralml.constants import paths
from coralml.ml.train import train
import coralml.constants.strings as STR

if __name__ == "__main__":
    default_instructions = os.path.join(paths.DATA_FOLDER_PATH, "instructions.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", help="Path to instructions.json file", type=str,
                        default=default_instructions)
    parser.add_argument('--data_folder_path', help='Path to the data directory, where to save the outputs.',
                        type=str, default=paths.DATA_FOLDER_PATH)
    parser.add_argument('--models_folder_path', help='Path to the data directory, where to save the outputs.',
                        type=str, default=paths.MODELS_FOLDER_PATH)
    parser.add_argument('--log_file', help='Path to the log_file',
                        type=str, default=paths.MODELS_FOLDER_PATH)

    args = parser.parse_args()

    # If data_folder_path is provided use this one, otherwise use the pre-configured in paths.

    data_folder_path = args.data_folder_path
    models_folder_path = args.models_folder_path

    instruction_keys = [
        STR.EPOCHS,
        STR.MODEL_NAME,
        STR.NN_INPUT_SIZE,  # TBC
        (STR.STATE_DICT_FILE_PATH, None),  # default = none
        STR.CROPS_PER_IMAGE,
        STR.IMAGES_PER_BATCH,
        STR.BATCH_SIZE,
        (STR.BACKBONE, "resnet"),  # default = resnet
        (STR.DEEPLAB_OUTPUT_STRIDE, 16),  # default = 16
        (STR.LEARNING_RATE, 1e-5),  # default = 1e-5
        (STR.MULTI_GPU, False),  # default = False
        STR.CLASS_STATS_FILE_PATH,
        (STR.USE_LR_SCHEDULER, True)  # default = True
    ]

    # Instructions need the following parameters
    for key in instruction_keys:
        if type(key) == tuple:
            print(f"{key[0]} (default = {key[1]})")
        else:
            print(key)

    with open(args.instructions, "r") as fp:
        instructions = json.load(fp)  # To be replaced

    print("Training with instructions")
    print(json.dumps(instructions, indent=4))
    # with real instruction file
    image_base_dir = "data"

    with open(os.path.join(data_folder_path, "data_train.json"), "r") as fp:
        data_train = json.load(fp)[:3]

    with open(os.path.join(data_folder_path, "data_valid.json"), "r") as fp:
        data_valid = json.load(fp)[:3]

    train(data_train=data_train,
          data_valid=data_valid,
          log_file=args.log_file,
          data_folder_path=data_folder_path,
          image_base_dir=image_base_dir,
          instructions=instructions,
          models_folder_path=models_folder_path)
