import json
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(__file__)
EPOCHS = 100  # 200 optimizer steps


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

rng = np.random.default_rng(42)
lim1 = np.sqrt(6.0 / (7 + 64))
lim2 = np.sqrt(6.0 / (64 + 4))
W1 = rng.uniform(-lim1, lim1, (7, 64)).astype(np.float32)
b1 = np.zeros(64, np.float32)
W2 = rng.uniform(-lim2, lim2, (64, 4)).astype(np.float32)
b2 = np.zeros(4, np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=7),
    tf.keras.layers.Activation('tanh'),
    tf.keras.layers.Dense(4, input_dim=64),
    tf.keras.layers.Activation('softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.set_weights([W1, b1, W2, b2])

with open(os.path.join(HERE, 'init.json'), 'w') as f:
    json.dump({'w1': W1.tolist(), 'b1': b1.tolist(),
               'w2': W2.tolist(), 'b2': b2.tolist()}, f)


def snap():
    w1, bb1, w2, bb2 = model.get_weights()
    return {'w1': w1.tolist(), 'b1': bb1.tolist(),
            'w2': w2.tolist(), 'b2': bb2.tolist()}


losses = []
after1 = None
for epoch in range(EPOCHS):
    losses.append(float(model.train_on_batch(trX[:64], trY[:64])))
    if after1 is None:
        after1 = snap()
    losses.append(float(model.train_on_batch(trX[64:], trY[64:])))

with open(os.path.join(HERE, 'keras.json'), 'w') as f:
    json.dump({'losses': losses, 'after1': after1, 'final': snap()}, f)
print('keras done, final batch loss:', losses[-1])
