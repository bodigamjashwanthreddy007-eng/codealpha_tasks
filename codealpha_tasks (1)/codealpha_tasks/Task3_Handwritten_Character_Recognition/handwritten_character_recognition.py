import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load MNIST Dataset
# --------------------------------------------------

print("Loading MNIST dataset...")

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add channel dimension for CNN
X_train_full = X_train_full[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Use subsets to make training faster
X_train_full, _, y_train_full, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=20000,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

X_test_small, _, y_test_small, _ = train_test_split(
    X_test,
    y_test,
    train_size=4000,
    random_state=RANDOM_STATE,
    stratify=y_test
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test_small)}")

# --------------------------------------------------
# 3. Build CNN Model
# --------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model:")
model.summary()

# --------------------------------------------------
# 4. Train Model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=128,
    verbose=1
)

# Save trained model
model.save(os.path.join(BASE_DIR, "mnist_cnn.keras"))

# --------------------------------------------------
# 5. Evaluate Model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test_small,
    y_test_small,
    verbose=0
)

y_prob = model.predict(X_test_small, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test_small, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# Classification report
report = classification_report(
    y_test_small,
    y_pred,
    zero_division=0
)

with open(
    os.path.join(RESULTS_DIR, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

# Save metrics
metrics_df = pd.DataFrame({
    "Metric": ["Test Loss", "Test Accuracy"],
    "Value": [test_loss, accuracy]
})

metrics_df.to_csv(
    os.path.join(RESULTS_DIR, "model_metrics.csv"),
    index=False
)

# --------------------------------------------------
# 6. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test_small, y_pred)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("MNIST CNN - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xticks(range(10))
plt.yticks(range(10))
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=150
)

plt.close()

# --------------------------------------------------
# 7. Accuracy Curve
# --------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "accuracy_curve.png"),
    dpi=150
)

plt.close()

# --------------------------------------------------
# 8. Loss Curve
# --------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "loss_curve.png"),
    dpi=150
)

plt.close()

# --------------------------------------------------
# 9. Sample Predictions
# --------------------------------------------------

sample_count = 10

sample_images = X_test_small[:sample_count]
sample_actual = y_test_small[:sample_count]
sample_predicted = y_pred[:sample_count]

plt.figure(figsize=(12, 5))

for i in range(sample_count):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].squeeze(), cmap="gray")
    plt.title(
        f"Actual: {sample_actual[i]}\n"
        f"Predicted: {sample_predicted[i]}"
    )
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "sample_predictions.png"),
    dpi=150
)

plt.close()

with open(
    os.path.join(RESULTS_DIR, "sample_prediction.txt"),
    "w"
) as f:
    for i in range(sample_count):
        f.write(
            f"Sample {i + 1}: "
            f"Actual={sample_actual[i]}, "
            f"Predicted={sample_predicted[i]}\n"
        )

print("\nTask 3 completed successfully!")
print("Results saved inside the results folder.")
