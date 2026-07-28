import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    IMG_SIZE, BATCH_SIZE, EMOTION_LABELS,
    DATASET_TEST_DIR, MODELS_DIR, MODEL_PATH,
    detect_dataset_format, load_csv_dataset
)


def evaluate_folder_model(model):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_generator = test_datagen.flow_from_directory(
        DATASET_TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    return y_true, y_pred, class_labels


def evaluate_csv_model(model):
    _, _, X_test, y_test = load_csv_dataset()
    predictions = model.predict(X_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return y_true, y_pred, EMOTION_LABELS


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Trained model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    dataset_format = detect_dataset_format()
    if dataset_format == "none":
        print("ERROR: No dataset found for evaluation.")
        sys.exit(1)

    if dataset_format == "folder":
        y_true, y_pred, class_labels = evaluate_folder_model(model)
    else:
        y_true, y_pred, class_labels = evaluate_csv_model(model)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)

    report_path = os.path.join(MODELS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
