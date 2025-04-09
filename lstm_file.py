import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# =======================
# Load Inertial Signal Data
# =======================

def load_signals(signal_paths):
    signals = []
    for path in signal_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        signals.append(np.loadtxt(path))
    return np.transpose(np.array(signals), (1, 2, 0))  # (samples, 128, 6)

def load_labels(label_path):
    labels = np.loadtxt(label_path).astype(int)
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return to_categorical(encoded), encoder

# =======================
# Dataset Path
# =======================

prefix = "/Users/....../UCI HAR Dataset/"
INPUT_SIGNAL_TYPES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_"
]

train_signals_paths = [prefix + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
test_signals_paths  = [prefix + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]

# =======================
# Load Data
# =======================

X_train = load_signals(train_signals_paths)
X_test = load_signals(test_signals_paths)
y_train, encoder = load_labels(prefix + "train/y_train.txt")
y_test, _ = load_labels(prefix + "test/y_test.txt")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# =======================
# Compute Class Weights
# =======================

y_train_int = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int)
class_weight_dict = dict(enumerate(class_weights))

# =======================
# Build Bidirectional LSTM Model
# =======================

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(128, 6)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# =======================
# Train Model
# =======================

model.fit(X_train, y_train, epochs=40, batch_size=64, validation_split=0.2, class_weight=class_weight_dict)

# =======================
# Evaluate
# =======================

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# =======================
# Predictions & Report
# =======================

y_pred = model.predict(X_test)
y_pred_labels = encoder.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = encoder.inverse_transform(np.argmax(y_test, axis=1))

print("\nClassification Report:\n")
print(classification_report(y_true_labels, y_pred_labels))

# =======================
# Confusion Matrix
# =======================

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
