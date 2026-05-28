# ============================================================
# train_hybrid.py
# ============================================================

import os
import gc
import time
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Multiply,
    Softmax,
    Reshape
)

from labels import TRAIN_PATIENTS

# ============================================================
# DATASET ROOT
# ============================================================

DATASET_ROOT = (
    r"C:\Users\jagad\Desktop"
    r"\EEG_FINAL_OUTPUT"
    r"\hybrid_npz"
)

# ============================================================
# OUTPUT ROOT
# ============================================================

OUTPUT_ROOT = (
    r"C:\Users\jagad\Desktop"
    r"\EEG_FINAL_OUTPUT"
    r"\hybrid_results"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

# ============================================================
# CONFIG
# ============================================================

WINDOW_SIZE = 1024

TARGET_CHANNELS = 23

SEIZURE_THRESHOLD = 0.25

EPOCHS = 15

BATCH_SIZE = 8

# ============================================================
# TERMINAL BAR
# ============================================================

def print_bar():

    print("=" * 60)

# ============================================================
# START
# ============================================================

print_bar()
print(" FINAL HYBRID TRAINING PIPELINE ")
print_bar()

pipeline_start = time.time()

# ============================================================
# STORE DATASET
# ============================================================

X_all = []
y_all = []

# ============================================================
# LOAD PATIENTS
# ============================================================

for patient in TRAIN_PATIENTS:

    print_bar()

    print(f"PATIENT: {patient}")

    print_bar()

    patient_folder = os.path.join(
        DATASET_ROOT,
        patient
    )

    npz_files = sorted([

        f for f in os.listdir(patient_folder)

        if f.endswith(".npz")

    ])

    print(
        f"\nNPZ Files Found: "
        f"{len(npz_files)}"
    )

    # ========================================================
    # LOAD NPZ
    # ========================================================

    for idx, npz_file in enumerate(npz_files):

        print_bar()

        print(
            f"[{idx+1:03d}/"
            f"{len(npz_files)}] "
            f"{npz_file}"
        )

        print_bar()

        npz_path = os.path.join(
            patient_folder,
            npz_file
        )

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        # ====================================================
        # FLOAT16 MEMORY FIX
        # ====================================================

        X = data["X"].astype(np.float16)

        y = data["y"].astype(np.int8)

        print(f"X Shape: {X.shape}")

        print(
            f"Seizure Windows: "
            f"{np.sum(y == 1)}"
        )

        print(
            f"Normal Windows : "
            f"{np.sum(y == 0)}"
        )

        X_all.append(X)
        y_all.append(y)

        del data
        del X
        del y

        gc.collect()

# ============================================================
# MERGE DATASET
# ============================================================

print_bar()
print(" MERGING DATASET ")
print_bar()

X_all = np.concatenate(
    X_all,
    axis=0
).astype(np.float16)

y_all = np.concatenate(
    y_all,
    axis=0
).astype(np.int8)

gc.collect()

print("\nFinal Dataset Shape:")
print(X_all.shape)

print(
    f"\nTotal Seizure Windows: "
    f"{np.sum(y_all == 1)}"
)

print(
    f"Total Normal Windows : "
    f"{np.sum(y_all == 0)}"
)

# ============================================================
# CHECK SEIZURES
# ============================================================

if np.sum(y_all == 1) == 0:

    raise ValueError(
        "No seizure windows found."
    )

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(

        X_all,
        y_all,

        test_size=0.2,

        stratify=y_all,

        random_state=42

    )
)

# ============================================================
# MEMORY CLEANUP
# ============================================================

del X_all
del y_all

gc.collect()

print("\nTrain Shape:", X_train.shape)

print("Test Shape :", X_test.shape)

# ============================================================
# ONE HOT
# ============================================================

y_train_cat = tf.keras.utils.to_categorical(
    y_train,
    2
)

y_test_cat = tf.keras.utils.to_categorical(
    y_test,
    2
)

# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(alpha=0.85,
               gamma=2.5):

    def loss(y_true,
             y_pred):

        y_true = tf.cast(
            y_true,
            tf.float32
        )

        epsilon = 1e-7

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        cross_entropy = (
            -y_true *
            tf.math.log(y_pred)
        )

        weight = alpha * tf.pow(
            1 - y_pred,
            gamma
        )

        focal = (
            weight *
            cross_entropy
        )

        return tf.reduce_mean(
            tf.reduce_sum(
                focal,
                axis=1
            )
        )

    return loss

# ============================================================
# CHANNEL ATTENTION
# ============================================================

def channel_attention(x):

    attention = GlobalAveragePooling1D()(x)

    attention = Dense(
        x.shape[-1],
        activation='sigmoid'
    )(attention)

    attention = Reshape(
        (1, x.shape[-1])
    )(attention)

    return Multiply()([x, attention])

# ============================================================
# TEMPORAL ATTENTION
# ============================================================

def temporal_attention(x):

    attention = Dense(1)(x)

    attention = Softmax(
        axis=1
    )(attention)

    return Multiply()([x, attention])

# ============================================================
# CLEAR SESSION
# ============================================================

tf.keras.backend.clear_session()

gc.collect()

# ============================================================
# BUILD MODEL
# ============================================================

inputs = Input(
    shape=(
        TARGET_CHANNELS,
        WINDOW_SIZE
    )
)

x = tf.keras.layers.Permute(
    (2,1)
)(inputs)

x = Conv1D(
    64,
    kernel_size=5,
    activation='relu',
    padding='same'
)(x)

x = BatchNormalization()(x)

x = MaxPooling1D(2)(x)

x = Conv1D(
    128,
    kernel_size=3,
    activation='relu',
    padding='same'
)(x)

x = BatchNormalization()(x)

x = MaxPooling1D(2)(x)

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)

x = channel_attention(x)

x = temporal_attention(x)

x = GlobalAveragePooling1D()(x)

x = Dropout(0.5)(x)

outputs = Dense(
    2,
    activation='softmax'
)(x)

model = Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        1e-4
    ),

    loss=focal_loss(
        alpha=0.85,
        gamma=2.5
    ),

    metrics=['accuracy']

)

# ============================================================
# MODEL SUMMARY
# ============================================================

print_bar()
print(" MODEL SUMMARY ")
print_bar()

model.summary()

# ============================================================
# TRAINING
# ============================================================

print_bar()
print(" TRAINING ")
print_bar()

history = model.fit(

    X_train,
    y_train_cat,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=1

)

# ============================================================
# TESTING
# ============================================================

print_bar()
print(" TESTING ")
print_bar()

y_pred_prob = model.predict(
    X_test,
    batch_size=BATCH_SIZE
)

y_pred = (

    y_pred_prob[:, 1] >
    SEIZURE_THRESHOLD

).astype(int)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_pred_prob[:,1]
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print_bar()
print(" FINAL RESULTS ")
print_bar()

print(
    f"\nThreshold   : "
    f"{SEIZURE_THRESHOLD:.2f}"
)

print(
    f"\nAccuracy    : "
    f"{acc:.4f}"
)

print(
    f"Precision   : "
    f"{precision:.4f}"
)

print(
    f"Recall      : "
    f"{recall:.4f}"
)

print(
    f"F1 Score    : "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC     : "
    f"{auc:.4f}"
)

print("\nConfusion Matrix:\n")

print(cm)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print_bar()
print(" PROBABILITY ANALYSIS ")
print_bar()

print(
    f"\nMaximum Seizure Probability : "
    f"{np.max(y_pred_prob[:,1]):.4f}"
)

print(
    f"Mean Seizure Probability    : "
    f"{np.mean(y_pred_prob[:,1]):.4f}"
)

print(
    f"Minimum Seizure Probability : "
    f"{np.min(y_pred_prob[:,1]):.4f}"
)

model_path = os.path.join(
    OUTPUT_ROOT,
    "hybrid_model.keras"
)

model.save(model_path)

print("\nSaved Model:")
print(model_path)

pipeline_time = (
    time.time() - pipeline_start
) / 3600

print_bar()

print(
    f"\nPipeline Time: "
    f"{pipeline_time:.2f} hours"
)

print("\nHYBRID TRAINING COMPLETE.")

print_bar()