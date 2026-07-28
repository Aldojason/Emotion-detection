import os
import sys
import matplotlib

matplotlib.use("Agg")  # headless backend for servers without a display
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    IMG_SIZE, BATCH_SIZE, EPOCHS,
    DATASET_TRAIN_DIR, DATASET_TEST_DIR, MODELS_DIR, MODEL_PATH,
    detect_dataset_format, load_csv_dataset, build_emotion_model, ensure_dirs
)

def get_folder_generators():
   
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        DATASET_TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    if os.path.isdir(DATASET_TEST_DIR) and len(os.listdir(DATASET_TEST_DIR)) > 0:
        val_generator = val_datagen.flow_from_directory(
            DATASET_TEST_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        train_datagen.validation_split = 0.2
        train_generator = train_datagen.flow_from_directory(
            DATASET_TRAIN_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )
        val_generator = train_datagen.flow_from_directory(
            DATASET_TRAIN_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

    return train_generator, val_generator


def plot_history(history):
    """Saves training curves as PNG images."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "training_history.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Accuracy and Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "accuracy_loss.png"))
    plt.close()


def train_with_folder_data():

    
    train_generator, val_generator = get_folder_generators()

    model = build_emotion_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    return model, history


def train_with_csv_data():
    """Trains the model using the fer2013.csv dataset."""
    X_train, y_train, X_test, y_test = load_csv_dataset()

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen.fit(X_train)

    model = build_emotion_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    return model, history


def main():
    ensure_dirs()

    dataset_format = detect_dataset_format()

    if dataset_format == "none":
        print("=" * 70)
        print("ERROR: No dataset found.")
        print("Please place the FER2013 dataset in one of these formats:")
        print("  1) dataset/train/<emotion>/*.jpg and dataset/test/<emotion>/*.jpg")
        print("  2) dataset/fer2013.csv")
        print("=" * 70)
        sys.exit(1)

    print(f"Detected dataset format: {dataset_format}")

    if dataset_format == "folder":
        model, history = train_with_folder_data()
    else:
        model, history = train_with_csv_data()

    # ModelCheckpoint already saved the BEST model (by val_accuracy) to
    # MODEL_PATH during training. Do not overwrite it with the in-memory
    # model here — if EarlyStopping never triggered (training ran all
    # EPOCHS), the in-memory weights are just the *last* epoch's, which
    # can be worse than the checkpointed best. Only fall back to saving
    # the in-memory model if, for some reason, no checkpoint file exists.
    if os.path.isfile(MODEL_PATH):
        print(f"Best model already saved by ModelCheckpoint to: {MODEL_PATH}")
    else:
        model.save(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

    plot_history(history)
    print("Training curves saved to models/training_history.png and models/accuracy_loss.png")


if __name__ == "__main__":
    main()