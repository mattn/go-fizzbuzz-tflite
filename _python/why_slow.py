import time

import numpy as np
import tensorflow as tf


def fizzbuzz(i):
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


def bin(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


trX = np.array([bin(i, 7) for i in range(1, 101)]).astype(np.float32)
trY = np.array([fizzbuzz(i) for i in range(1, 101)]).astype(np.float32)


def make_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=7),
        tf.keras.layers.Activation('tanh'),
        tf.keras.layers.Dense(4, input_dim=64),
        tf.keras.layers.Activation('softmax'),
    ])
    m.compile(loss='categorical_crossentropy', optimizer='adam')
    return m


# 1) model.fit -----------------------------------------------------------
model = make_model()
model.fit(trX, trY, epochs=3, batch_size=64, verbose=0)  # warmup/trace
t = time.perf_counter()
model.fit(trX, trY, epochs=100, batch_size=64, verbose=0)
fit_step = (time.perf_counter() - t) / 200

# 2) train_on_batch ------------------------------------------------------
model = make_model()
model.train_on_batch(trX[:64], trY[:64])  # warmup
t = time.perf_counter()
for _ in range(100):
    model.train_on_batch(trX[:64], trY[:64])
    model.train_on_batch(trX[64:], trY[64:])
tob_step = (time.perf_counter() - t) / 200

# 3) raw tf.function train step ------------------------------------------
model = make_model()
opt = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
opt.build(model.trainable_variables)


@tf.function
def raw_step(x, y):
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        loss = loss_fn(y, p)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


xa = tf.constant(trX[:64]); ya = tf.constant(trY[:64])
xb = tf.constant(trX[64:]); yb = tf.constant(trY[64:])
raw_step(xa, ya); raw_step(xb, yb)  # trace both shapes
t = time.perf_counter()
for _ in range(100):
    raw_step(xa, ya)
    raw_step(xb, yb)
raw_per_step = (time.perf_counter() - t) / 200

# 4) same math in plain numpy, full 3600 epochs --------------------------
rng = np.random.default_rng(1)
lim1 = np.sqrt(6 / 71); lim2 = np.sqrt(6 / 68)
W1 = rng.uniform(-lim1, lim1, (7, 64)).astype(np.float32)
b1 = np.zeros(64, np.float32)
W2 = rng.uniform(-lim2, lim2, (64, 4)).astype(np.float32)
b2 = np.zeros(4, np.float32)
params = [W1, b1, W2, b2]
ms = [np.zeros_like(p) for p in params]
vs = [np.zeros_like(p) for p in params]
lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-7

t = time.perf_counter()
step_no = 0
for epoch in range(3600):
    perm = rng.permutation(100)
    for lo in range(0, 100, 64):
        x = trX[perm[lo:lo + 64]]
        y = trY[perm[lo:lo + 64]]
        h = np.tanh(x @ W1 + b1)
        z = h @ W2 + b2
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        d2 = (p - y) / len(x)
        dh = (d2 @ W2.T) * (1 - h * h)
        grads = [x.T @ dh, dh.sum(0), h.T @ d2, d2.sum(0)]
        step_no += 1
        alpha = lr * np.sqrt(1 - beta2**step_no) / (1 - beta1**step_no)
        for j, g in enumerate(grads):
            ms[j] = beta1 * ms[j] + (1 - beta1) * g
            vs[j] = beta2 * vs[j] + (1 - beta2) * g * g
            params[j] -= alpha * ms[j] / (np.sqrt(vs[j]) + eps)
W1, b1, W2, b2 = params
numpy_total = time.perf_counter() - t
h = np.tanh(trX @ W1 + b1)
z = h @ W2 + b2
acc = (z.argmax(1) == trY.argmax(1)).mean()

print(f'model.fit        : {fit_step*1000:8.2f} ms/step -> '
      f'{fit_step*7200:6.1f}s for 3600 epochs')
print(f'train_on_batch   : {tob_step*1000:8.2f} ms/step -> '
      f'{tob_step*7200:6.1f}s')
print(f'raw tf.function  : {raw_per_step*1000:8.2f} ms/step -> '
      f'{raw_per_step*7200:6.1f}s')
print(f'plain numpy      : {numpy_total/7200*1000:8.2f} ms/step -> '
      f'{numpy_total:6.1f}s (actual full run, acc={acc:.4f})')
