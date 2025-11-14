# ======================================================
#   FASHION-MNIST CNN (KERAS / TENSORFLOW)
# ======================================================

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

# ===============================
# 1) Load + Preprocess Data
# ===============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# ===============================
# 2) Data Augmentation
# ===============================
data_augment = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

# ===============================
# 3) Build Advanced CNN
# ===============================
def build_model():
    model = models.Sequential([
        data_augment,

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")
    ])
    return model

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# 4) Callbacks
# ===============================
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

# ===============================
# 5) Train
# ===============================
history = model.fit(
    x_train, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ===============================
# 6) Evaluate
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# ===============================
# 7) Confusion Matrix + Report
# ===============================
y_pred = np.argmax(model.predict(x_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

# ===============================
# 8) Prediction Visualization
# ===============================
def show_preds():
    plt.figure(figsize=(12,6))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        plt.title(f"True={y_test[i]} / Pred={y_pred[i]}")
        plt.axis("off")
show_preds()

# ===============================
# 9) Grad-CAM
# ===============================
def grad_cam(model, img, layer_name):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0]
    heatmap = tf.reduce_mean(tf.multiply(grads, conv_out[0]), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap, (28,28))

    return heatmap

img = x_test[0:1]
heat = grad_cam(model, img, "conv2d_2")

plt.imshow(img.squeeze(), cmap="gray")
plt.imshow(heat, cmap="jet", alpha=0.5)
plt.title("Grad-CAM")
plt.show()

# ===============================
# 10) Save / Load
# ===============================
model.save("keras_fashion_mnist_advanced.h5")
loaded = tf.keras.models.load_model("keras_fashion_mnist_advanced.h5")

print("Loaded Model Accuracy:", loaded.evaluate(x_test, y_test)[1])
