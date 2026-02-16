# Simple MLP classifier compatible with TensorFlow 2.x
import tensorflow as tf
import numpy as np

def build_model(input_dim=561, nclass=6):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(nclass, activation="softmax"),
    ])
    return m

def compile_model(m, lr=0.01):
    m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def get_weights_vector(model):
    # flatten all weights to a single 1D float32 array
    w = []
    for v in model.trainable_variables:
        w.append(tf.reshape(v, [-1]))
    flat = tf.concat(w, axis=0).numpy().astype("float32")
    return flat

def set_weights_vector(model, vec):
    # set model weights from 1D array
    idx = 0
    new_vars = []
    for v in model.trainable_variables:
        shape = v.shape
        size = tf.size(v).numpy()
        part = vec[idx:idx+size]
        part = tf.reshape(tf.convert_to_tensor(part, dtype=v.dtype), shape)
        new_vars.append(part)
        idx += size
    for var, newv in zip(model.trainable_variables, new_vars):
        var.assign(newv)

def weights_zeros_like(model):
    import numpy as np
    z = []
    for v in model.trainable_variables:
        z.append(np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype))
    return z, np.concatenate([x.reshape(-1) for x in z]).astype("float32")
