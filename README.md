# go-fizzbuzz-tflite

FizzBuzz powered by a neural network, in Go only.

This repository trains a TensorFlow Lite model in pure Go and runs it with
[go-tflite](https://github.com/mattn/go-tflite). No Python and no TensorFlow
are required for training: the network (7 → Dense(64) → tanh → Dense(4) →
softmax) is trained with a hand-written backprop/Adam implementation, and the
`.tflite` flatbuffer is generated directly from Go.

## Usage

Train the model. This learns FizzBuzz from the numbers 1-100 encoded as 7-bit
binary, and writes `fizzbuzz_model.tflite`.

```
$ go run ./cmd/train
model: dense(7->64) tanh, dense(64->4) softmax, 772 params
epoch  100/3600 - loss: 0.9401 - accuracy: 0.5300
...
epoch 3600/3600 - loss: 0.0431 - accuracy: 1.0000
training took 911.207979ms
wrote fizzbuzz_model.tflite
```

Run FizzBuzz with the trained model.

```
$ go run .
1
2
Fizz
4
Buzz
Fizz
...
FizzBuzz
...
Buzz
```

## Speed

Compared with the equivalent `tf.keras` script (TensorFlow 2.21, Linux
x86-64, 16 cores), training the same model for the same 3600 epochs:

|                        | Python (tf.keras) | Go       |
|------------------------|-------------------|----------|
| whole run              | 3m26s             | 1.05s    |
| training only          | 207.3s            | 0.95s    |
| startup (`import tensorflow`) | 3.0s       | -        |
| write `.tflite`        | 0.53s             | ~1ms     |

### Why is tf.keras 200x slower here?

It is not Python itself — it is per-step fixed overhead in the framework.
Measuring the cost of one optimizer step (one batch) at each layer:

| how the same step runs        | ms/step | 3600 epochs |
|-------------------------------|---------|-------------|
| `model.fit`                   | 28.2    | 203s        |
| `train_on_batch`              | 3.3     | 24s         |
| raw `tf.function` train step  | 2.6     | 18.5s       |
| same math in plain numpy      | 0.25    | 1.8s        |
| this Go loop                  | 0.13    | 0.95s       |

* **~25ms/step is `fit` machinery**: per-epoch data-adapter/iterator setup,
  the callback chain and metric bookkeeping. This dataset is 2 batches per
  epoch, so ~50ms of per-epoch setup lands on 2 steps — many tiny epochs is
  `fit`'s worst case.
* **~2.6ms/step is the TF executor**: a step is a graph of dozens of tiny
  kernels, each paying dispatch and inter-op thread-pool handoffs. With 7x64
  matrices nothing amortizes; TF's fixed costs are designed to disappear
  behind big kernels (large batches, GPU), and here they never do.
* **numpy is within 2x of Go**, which shows the actual math of a
  772-parameter model is negligible.

XNNPACK only accelerates TFLite *inference*; it plays no part in training,
which is why inference is fast in both worlds while training speed differs
by 200x.

## Requirements

* TensorFlow Lite C library (only for inference, see
  [go-tflite](https://github.com/mattn/go-tflite#requirements))

Training has no C dependencies.

## How it works

`cmd/train` implements the same model as `tf.keras`:

* glorot uniform initialization
* Adam optimizer (lr=0.001, β1=0.9, β2=0.999, ε=1e-7)
* categorical crossentropy loss
* 3600 epochs, batch size 64

The training loop is allocation-free (fused forward/backward per sample on
stack arrays) and data-parallel: each batch is split across persistent worker
goroutines with per-worker gradient accumulators. Batches are only ~30µs of
work, so the workers synchronize by spinning on an atomic sequence number
(with a `Gosched` fallback) instead of sleeping — futex wake-ups would cost
more than the work itself.

After training, the weights are serialized into a TensorFlow Lite flatbuffer
(`FULLY_CONNECTED` → `TANH` → `FULLY_CONNECTED` → `SOFTMAX`) using the
flatbuffers Go runtime, following `tensorflow/lite/schema/schema.fbs`. The
resulting file is loadable by any TensorFlow Lite interpreter.

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)
