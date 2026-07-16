import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from tf_keras.applications import EfficientNetB3
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 5
TRAIN_DIR = "dataset/train"
CSV_PATH = "dataset/train.csv"
MODEL_SAVE_PATH = "model/retina_model.h5"

def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

def preprocess_image(image_id):
    path = os.path.join(TRAIN_DIR, image_id + ".png")
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return img

def build_dataset(df):
    images = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        try:
            img = preprocess_image(row["id_code"])
            images.append(img)
            labels.append(row["diagnosis"])
        except Exception as e:
            print(f"Skipping {row['id_code']}: {e}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

class TrainingProgress(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        current   = epoch + 1
        pct       = int((current / self.total_epochs) * 100)
        bar_done  = int(pct / 5)
        bar       = "█" * bar_done + "░" * (20 - bar_done)
        acc       = logs.get("accuracy", 0) * 100
        val_acc   = logs.get("val_accuracy", 0) * 100
        loss      = logs.get("loss", 0)
        val_loss  = logs.get("val_loss", 0)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            best_tag = "  ★ NEW BEST"
        else:
            best_tag = ""

        print(f"\n{'='*60}")
        print(f"  EPOCH  {current:02d} / {self.total_epochs}   [{bar}] {pct}%{best_tag}")
        print(f"{'='*60}")
        print(f"  Train  →  Loss: {loss:.4f}  |  Accuracy: {acc:.2f}%")
        print(f"  Val    →  Loss: {val_loss:.4f}  |  Accuracy: {val_acc:.2f}%")
        print(f"  Best Val Accuracy so far: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")


def build_model():
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

def main():
    os.makedirs("model", exist_ok=True)

    print("Loading dataset...")
    df = load_data()
    print(f"Total samples: {len(df)}")
    print(df["diagnosis"].value_counts().sort_index())

    print("\nPreprocessing images...")
    X, y = build_dataset(df)
    print(f"Dataset shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)

    print("\nBuilding model...")
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        TrainingProgress(total_epochs=EPOCHS),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=0),
        EarlyStopping(patience=5, monitor="val_accuracy", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    main()
