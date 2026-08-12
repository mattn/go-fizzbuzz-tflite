# Python comparison scripts

These scripts back the numbers and claims in the main README. They need
TensorFlow:

```
uv venv && uv pip install tensorflow
```

* `make.py` - the original `tf.keras` version of `cmd/train`: trains the same
  model and converts it to `fizzbuzz_model.tflite` with `TFLiteConverter`.
  This is the script the "3m26s vs 1.05s" comparison measures.
* `why_slow.py` - measures the cost of one optimizer step at each layer
  (`model.fit`, `train_on_batch`, raw `tf.function`, plain numpy) to show
  where the time goes.
* `samecheck/` - proves the Go training loop computes the same thing as
  `tf.keras`: trains both from identical initial weights with a fixed batch
  order and compares the loss trajectory and the weights.

```
cd samecheck
python driver.py     # writes init.json + keras.json
go run main.go       # reads init.json, writes go.json
python compare.py    # loss diffs ~1e-7, weight diffs ~1e-7 after 200 steps
```
