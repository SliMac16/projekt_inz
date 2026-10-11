from pathlib import Path

SEED = 42

MANIFEST_PATH = Path("final_dataset_manifest.csv")
RESULTS_DIR = Path("training_results")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0 #4 -multiprocessing DataLoadera powodował segmentation fault

HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]