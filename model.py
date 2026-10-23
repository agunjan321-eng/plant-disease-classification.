import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

IMG_SIZE = (96, 96)
BATCH_SIZE = 32

def load_split_arrays(split_name, class_indices, shuffle=True):
    split_dir = os.path.join(DATASET_DIR, split_name)
    images = []
    labels = []
    
    for class_name, idx in class_indices.items():
        class_folder = os.path.join(split_dir, class_name)
        if not os.path.exists(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(class_folder, fname)
                try:
                    img = Image.open(fpath).convert('RGB').resize(IMG_SIZE)
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(idx)
                except Exception as e:
                    pass
                    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    if shuffle and len(X) > 0:
        p = np.random.permutation(len(X))
        X = X[p]
        y = y[p]
        
    return X, y

def build_custom_cnn(num_classes, input_shape=(96, 96, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classifier Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax', name='prediction_output')
    ], name="Custom_Deep_CNN")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_transfer_mobilenet(num_classes, input_shape=(96, 96, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='prediction_output')(x)
    
    model = models.Model(inputs, outputs, name="Transfer_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def train_and_evaluate():
    print("="*60)
    print("PLANT DISEASE CLASSIFICATION: CNN vs TRANSFER LEARNING")
    print("="*60)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    with open(os.path.join(MODELS_DIR, "class_indices.json"), "r") as f:
        class_meta = json.load(f)
    class_indices = class_meta["class_to_idx"]
    classes = class_meta["classes"]
    num_classes = len(classes)
    
    print(f"Loading dataset for {num_classes} classes...")
    X_train, y_train = load_split_arrays("train", class_indices, shuffle=True)
    X_val, y_val = load_split_arrays("val", class_indices, shuffle=False)
    X_test, y_test = load_split_arrays("test", class_indices, shuffle=False)
    
    print(f"Dataset Loaded:")
    print(f" - Train: {X_train.shape[0]} samples, shape: {X_train.shape[1:]}")
    print(f" - Val:   {X_val.shape[0]} samples")
    print(f" - Test:  {X_test.shape[0]} samples")
    
    # MODEL 1: CUSTOM DEEP CNN
    print("\n" + "="*50)
    print("[1/2] Training Model 1: Custom Deep CNN (from scratch)...")
    print("="*50)
    cnn_model = build_custom_cnn(num_classes, input_shape=(96, 96, 3))
    
    t0 = time.time()
    cnn_history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=7,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    cnn_train_time = round(time.time() - t0, 2)
    print(f"CNN Training completed in {cnn_train_time}s")
    
    t0 = time.time()
    cnn_preds_proba = cnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    cnn_latency_ms = round(((time.time() - t0) / len(X_test)) * 1000, 2)
    cnn_preds = np.argmax(cnn_preds_proba, axis=1)
    
    cnn_acc = float(accuracy_score(y_test, cnn_preds))
    cnn_prec, cnn_rec, cnn_f1, _ = precision_recall_fscore_support(y_test, cnn_preds, average='weighted', zero_division=0)
    cnn_cm = confusion_matrix(y_test, cnn_preds).tolist()
    print(f"Custom CNN Test Accuracy: {cnn_acc * 100:.2f}% | Latency: {cnn_latency_ms} ms/leaf")
    
    cnn_path = os.path.join(MODELS_DIR, "custom_cnn_model.keras")
    cnn_model.save(cnn_path)
    cnn_size_mb = round(os.path.getsize(cnn_path) / (1024 * 1024), 2)
    
    # MODEL 2: TRANSFER LEARNING (MobileNetV2)
    print("\n" + "="*50)
    print("[2/2] Training Model 2: Transfer Learning (MobileNetV2)...")
    print("="*50)
    transfer_model, base_mobilenet = build_transfer_mobilenet(num_classes, input_shape=(96, 96, 3))
    
    t0 = time.time()
    # Phase 1: Train top head
    transfer_history_p1 = transfer_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Phase 2: Fine-tuning top layers of base model
    print("Fine-tuning top layers of MobileNetV2...")
    base_mobilenet.trainable = True
    for layer in base_mobilenet.layers[:-20]:
        layer.trainable = False
        
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    transfer_history_p2 = transfer_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=4,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    mobilenet_train_time = round(time.time() - t0, 2)
    print(f"MobileNetV2 Training completed in {mobilenet_train_time}s")
    
    combined_transfer_acc = transfer_history_p1.history['accuracy'] + transfer_history_p2.history['accuracy']
    combined_transfer_val_acc = transfer_history_p1.history['val_accuracy'] + transfer_history_p2.history['val_accuracy']
    combined_transfer_loss = transfer_history_p1.history['loss'] + transfer_history_p2.history['loss']
    combined_transfer_val_loss = transfer_history_p1.history['val_loss'] + transfer_history_p2.history['val_loss']
    
    t0 = time.time()
    mobilenet_preds_proba = transfer_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    mobilenet_latency_ms = round(((time.time() - t0) / len(X_test)) * 1000, 2)
    mobilenet_preds = np.argmax(mobilenet_preds_proba, axis=1)
    
    mob_acc = float(accuracy_score(y_test, mobilenet_preds))
    mob_prec, mob_rec, mob_f1, _ = precision_recall_fscore_support(y_test, mobilenet_preds, average='weighted', zero_division=0)
    mob_cm = confusion_matrix(y_test, mobilenet_preds).tolist()
    print(f"Transfer Learning (MobileNetV2) Test Accuracy: {mob_acc * 100:.2f}% | Latency: {mobilenet_latency_ms} ms/leaf")
    
    mob_path = os.path.join(MODELS_DIR, "mobilenetv2_model.keras")
    transfer_model.save(mob_path)
    mob_size_mb = round(os.path.getsize(mob_path) / (1024 * 1024), 2)
    
    class_report_cnn = classification_report(y_test, cnn_preds, target_names=classes, output_dict=True, zero_division=0)
    class_report_mob = classification_report(y_test, mobilenet_preds, target_names=classes, output_dict=True, zero_division=0)
    
    metrics_data = {
        "dataset_info": {
            "name": "Kaggle - PlantVillage (Multi-Crop Benchmark)",
            "num_classes": num_classes,
            "classes": classes,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "image_resolution": [96, 96, 3]
        },
        "models": {
            "Custom_CNN": {
                "name": "Custom 3-Block Deep CNN",
                "technique": "Supervised Deep Learning from Scratch",
                "accuracy": round(cnn_acc, 4),
                "precision": round(float(cnn_prec), 4),
                "recall": round(float(cnn_rec), 4),
                "f1_score": round(float(cnn_f1), 4),
                "latency_ms": cnn_latency_ms,
                "model_size_mb": cnn_size_mb,
                "train_time_sec": cnn_train_time,
                "epochs": 7,
                "confusion_matrix": cnn_cm,
                "train_history": {
                    "accuracy": [round(float(v), 4) for v in cnn_history.history['accuracy']],
                    "val_accuracy": [round(float(v), 4) for v in cnn_history.history['val_accuracy']],
                    "loss": [round(float(v), 4) for v in cnn_history.history['loss']],
                    "val_loss": [round(float(v), 4) for v in cnn_history.history['val_loss']]
                },
                "per_class_report": class_report_cnn
            },
            "Transfer_MobileNetV2": {
                "name": "Transfer Learning (MobileNetV2)",
                "technique": "Pre-trained ImageNet + Fine-Tuned Head",
                "accuracy": round(mob_acc, 4),
                "precision": round(float(mob_prec), 4),
                "recall": round(float(mob_rec), 4),
                "f1_score": round(float(mob_f1), 4),
                "latency_ms": mobilenet_latency_ms,
                "model_size_mb": mob_size_mb,
                "train_time_sec": mobilenet_train_time,
                "epochs": 9,
                "confusion_matrix": mob_cm,
                "train_history": {
                    "accuracy": [round(float(v), 4) for v in combined_transfer_acc],
                    "val_accuracy": [round(float(v), 4) for v in combined_transfer_val_acc],
                    "loss": [round(float(v), 4) for v in combined_transfer_loss],
                    "val_loss": [round(float(v), 4) for v in combined_transfer_val_loss]
                },
                "per_class_report": class_report_mob
            }
        },
        "champion_model": "Transfer_MobileNetV2" if mob_acc >= cnn_acc else "Custom_CNN"
    }
    
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\nModel metrics successfully exported to: {metrics_path}")
    print(f"Champion Model: {metrics_data['champion_model']}")

if __name__ == "__main__":
    train_and_evaluate()
