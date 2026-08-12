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
