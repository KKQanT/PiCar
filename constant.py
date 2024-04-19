FOLD = 1

MODEL_NAME = f"corrected_k_fold_custom_loss_concat_vgg16_dave2_{FOLD}"
TRAIN_FILE_PATH = "metadata/corrected_training_norm_k_folds.csv"

TRAIN_IMAGE_PATH = "training_data/training_data/"  # this dataset can be obtained from

DAVE2_PATH = "model/pretrain-dave2.h5"

SAVE_MODEL_PATH = f"{MODEL_NAME}.keras"

TEST_PATH = "test_data/test_data"  # this dataset can be obtained from

BATCH_SIZE = 64
DIM = (66, 200)  # DIM FOR DAVE2
EPOCHS = 600
LR = 0.001