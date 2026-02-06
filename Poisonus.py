import tensorflow as tf
import numpy as np
import os

# 1. Configuration
SIZE = 50
EPOCHS = 200
BATCH_SIZE = 16 
MODEL_NAME = "poisson_long_range_unet"

# 2. Data Loading
def load_and_preprocess_data():
    if not os.path.exists("x_train.bin") or not os.path.exists("y_train.bin"):
        raise FileNotFoundError("Binary files missing. Run the C script first.")

    X = np.fromfile("x_train.bin", dtype=np.float32).reshape(-1, SIZE, SIZE, 1)
    Y = np.fromfile("y_train.bin", dtype=np.float32).reshape(-1, SIZE, SIZE, 1)

    return X / 100.0, Y / 100.0

# 3. Model with Dilated Blocks for Global Propagation
def build_physics_unet(size=50):
    inputs = tf.keras.layers.Input(shape=(size, size, 1))

    # --- Encoder ---
    # We use Dilation here to see further even at high resolution
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) # 25x25

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), dilation_rate=4, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # 12x12

    # --- Bottleneck (Deep Global Understanding) ---
    b1 = tf.keras.layers.Conv2D(256, (3, 3), dilation_rate=8, activation='relu', padding='same')(p2)
    b1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b1)

    # --- Decoder ---
    u1 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(b1)
    u1 = tf.keras.layers.Resizing(25, 25)(u1)
    u1 = tf.keras.layers.Concatenate()([u1, c2])
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c1])
    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    # FINAL RESIDUAL CONNECTION: Add input to help boundary retention
    # This forces the model to learn the "delta" change
    output_delta = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c4)
    
    model = tf.keras.Model(inputs=inputs, outputs=output_delta)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss='mse', metrics=['mae'])
    return model

# 4. Training Loop
def train():
    X, Y = load_and_preprocess_data()
    model = build_physics_unet(SIZE)
    
    # Physics training often benefits from a very slow decay
    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    print("\nTraining to fill the gap (Global Propagation Focus)...")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[lr_decay, early_stop])

    # 5. Export
    model.save(f"{MODEL_NAME}.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("poisson_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("\nModel exported. The 'black hole' in the middle should be gone now.")

if __name__ == "__main__":
    train()