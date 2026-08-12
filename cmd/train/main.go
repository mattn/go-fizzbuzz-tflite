package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	flatbuffers "github.com/google/flatbuffers/go"
)

const (
	numDigits = 7
	hidden    = 64
	classes   = 4
	samples   = 100
	epochs    = 3600
	batchSize = 64

	// Adam (Keras defaults)
	lr      = 0.001
	beta1   = 0.9
	beta2   = 0.999
	epsilon = 1e-7
)

func fizzbuzz(i int) []float64 {
	switch {
	case i%15 == 0:
		return []float64{0, 0, 0, 1}
	case i%5 == 0:
		return []float64{0, 0, 1, 0}
	case i%3 == 0:
		return []float64{0, 1, 0, 0}
	default:
		return []float64{1, 0, 0, 0}
	}
}

func bin(i, numDigits int) []float64 {
	v := make([]float64, numDigits)
	for d := 0; d < numDigits; d++ {
		v[d] = float64(i >> d & 1)
	}
	return v
}

// dense is a fully connected layer with Adam optimizer state.
// w is row-major [in][out].
type dense struct {
	in, out        int
	w, b           []float64
	mw, vw, mb, vb []float64
}

func newDense(in, out int, r *rand.Rand) *dense {
	d := &dense{
		in: in, out: out,
		w: make([]float64, in*out), b: make([]float64, out),
		mw: make([]float64, in*out), vw: make([]float64, in*out),
		mb: make([]float64, out), vb: make([]float64, out),
	}
	// glorot_uniform (Keras Dense default)
	limit := math.Sqrt(6.0 / float64(in+out))
	for i := range d.w {
		d.w[i] = (r.Float64()*2 - 1) * limit
	}
	return d
}

func (d *dense) forward(x [][]float64) [][]float64 {
	y := make([][]float64, len(x))
	for n, xi := range x {
		yi := make([]float64, d.out)
		copy(yi, d.b)
		for i, v := range xi {
			if v == 0 {
				continue
			}
			for j, w := range d.w[i*d.out : (i+1)*d.out] {
				yi[j] += v * w
			}
		}
		y[n] = yi
	}
	return y
}

func (d *dense) update(x, dy [][]float64, t int) {
	gw := make([]float64, len(d.w))
	gb := make([]float64, len(d.b))
	for n, dyi := range dy {
		for i, v := range x[n] {
			if v == 0 {
				continue
			}
			for j, g := range dyi {
				gw[i*d.out+j] += v * g
			}
		}
		for j, g := range dyi {
			gb[j] += g
		}
	}
	c1 := 1 - math.Pow(beta1, float64(t))
	c2 := 1 - math.Pow(beta2, float64(t))
	adam := func(w, m, v, g []float64) {
		for i, gi := range g {
			m[i] = beta1*m[i] + (1-beta1)*gi
			v[i] = beta2*v[i] + (1-beta2)*gi*gi
			w[i] -= lr * (m[i] / c1) / (math.Sqrt(v[i]/c2) + epsilon)
		}
	}
	adam(d.w, d.mw, d.vw, gw)
	adam(d.b, d.mb, d.vb, gb)
}

func tanhAll(x [][]float64) [][]float64 {
	y := make([][]float64, len(x))
	for n, xi := range x {
		yi := make([]float64, len(xi))
		for i, v := range xi {
			yi[i] = math.Tanh(v)
		}
		y[n] = yi
	}
	return y
}

func softmaxAll(x [][]float64) [][]float64 {
	y := make([][]float64, len(x))
	for n, xi := range x {
		yi := make([]float64, len(xi))
		max := xi[0]
		for _, v := range xi[1:] {
			if v > max {
				max = v
			}
		}
		var sum float64
		for i, v := range xi {
			yi[i] = math.Exp(v - max)
			sum += yi[i]
		}
		for i := range yi {
			yi[i] /= sum
		}
		y[n] = yi
	}
	return y
}

func argmax(v []float64) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func predict(l1, l2 *dense, x [][]float64) [][]float64 {
	return softmaxAll(l2.forward(tanhAll(l1.forward(x))))
}

func evaluate(l1, l2 *dense, x, y [][]float64) (loss, acc float64) {
	p := predict(l1, l2, x)
	for n, pi := range p {
		loss -= math.Log(math.Max(pi[argmax(y[n])], 1e-12))
		if argmax(pi) == argmax(y[n]) {
			acc++
		}
	}
	return loss / float64(len(x)), acc / float64(len(x))
}

func main() {
	output := flag.String("o", "fizzbuzz_model.tflite", "output model file")
	flag.Parse()

	r := rand.New(rand.NewSource(rand.Int63()))

	trX := make([][]float64, samples)
	trY := make([][]float64, samples)
	for i := 1; i <= samples; i++ {
		trX[i-1] = bin(i, numDigits)
		trY[i-1] = fizzbuzz(i)
	}

	l1 := newDense(numDigits, hidden, r)
	l2 := newDense(hidden, classes, r)
	fmt.Printf("model: dense(%d->%d) tanh, dense(%d->%d) softmax, %d params\n",
		numDigits, hidden, hidden, classes,
		len(l1.w)+len(l1.b)+len(l2.w)+len(l2.b))

	t := 0
	for epoch := 1; epoch <= epochs; epoch++ {
		perm := r.Perm(samples)
		for start := 0; start < samples; start += batchSize {
			end := min(start+batchSize, samples)
			x := make([][]float64, 0, end-start)
			y := make([][]float64, 0, end-start)
			for _, i := range perm[start:end] {
				x = append(x, trX[i])
				y = append(y, trY[i])
			}

			h := tanhAll(l1.forward(x))
			p := softmaxAll(l2.forward(h))

			// categorical crossentropy + softmax: dz2 = (p - y) / n
			n := float64(len(x))
			dz2 := make([][]float64, len(x))
			for i, pi := range p {
				di := make([]float64, classes)
				for j := range di {
					di[j] = (pi[j] - y[i][j]) / n
				}
				dz2[i] = di
			}
			// dz1 = (dz2 . W2^T) * (1 - h^2)
			dz1 := make([][]float64, len(x))
			for i, di := range dz2 {
				dh := make([]float64, hidden)
				for j := range dh {
					for k, g := range di {
						dh[j] += g * l2.w[j*classes+k]
					}
					dh[j] *= 1 - h[i][j]*h[i][j]
				}
				dz1[i] = dh
			}

			t++
			l2.update(h, dz2, t)
			l1.update(x, dz1, t)
		}
		if epoch%100 == 0 || epoch == epochs {
			loss, acc := evaluate(l1, l2, trX, trY)
			fmt.Printf("epoch %4d/%d - loss: %.4f - accuracy: %.4f\n", epoch, epochs, loss, acc)
		}
	}

	if err := os.WriteFile(*output, buildTFLite(l1, l2), 0644); err != nil {
		log.Fatal(err)
	}
	fmt.Println("wrote " + *output)
}

// TFLite flatbuffer schema constants (tensorflow/lite/schema/schema.fbs).
const (
	opFullyConnected = 9
	opSoftmax        = 25
	opTanh           = 28

	optionsFullyConnected = 8 // BuiltinOptions union member index
	optionsSoftmax        = 9
)

func floatBytes(v []float64) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(x)))
	}
	return b
}

// transposed returns d.w as a TFLite fully-connected filter: [out][in] row-major.
func (d *dense) transposed() []float64 {
	t := make([]float64, len(d.w))
	for i := 0; i < d.in; i++ {
		for j := 0; j < d.out; j++ {
			t[j*d.in+i] = d.w[i*d.out+j]
		}
	}
	return t
}

func fbIntVector(b *flatbuffers.Builder, vals []int32) flatbuffers.UOffsetT {
	b.StartVector(4, len(vals), 4)
	for i := len(vals) - 1; i >= 0; i-- {
		b.PrependInt32(vals[i])
	}
	return b.EndVector(len(vals))
}

func fbOffsetVector(b *flatbuffers.Builder, offs []flatbuffers.UOffsetT) flatbuffers.UOffsetT {
	b.StartVector(4, len(offs), 4)
	for i := len(offs) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offs[i])
	}
	return b.EndVector(len(offs))
}

// fbBuffer builds a Buffer table. TFLite wants tensor data 16-byte aligned.
func fbBuffer(b *flatbuffers.Builder, data []byte) flatbuffers.UOffsetT {
	var off flatbuffers.UOffsetT
	if len(data) > 0 {
		b.Prep(16, len(data))
		off = b.CreateByteVector(data)
	}
	b.StartObject(1)
	if len(data) > 0 {
		b.PrependUOffsetTSlot(0, off, 0)
	}
	return b.EndObject()
}

// fbTensor builds a float32 Tensor table.
func fbTensor(b *flatbuffers.Builder, name string, shape []int32, buffer int32) flatbuffers.UOffsetT {
	nameOff := b.CreateString(name)
	shapeOff := fbIntVector(b, shape)
	b.StartObject(4)
	b.PrependUOffsetTSlot(0, shapeOff, 0)
	// slot 1 (type) omitted: FLOAT32 = 0 is the default
	b.PrependUint32Slot(2, uint32(buffer), 0)
	b.PrependUOffsetTSlot(3, nameOff, 0)
	return b.EndObject()
}

func fbOperatorCode(b *flatbuffers.Builder, code int32) flatbuffers.UOffsetT {
	b.StartObject(4)
	b.PrependInt8Slot(0, int8(code), 0) // deprecated_builtin_code
	b.PrependInt32Slot(3, code, 0)      // builtin_code
	return b.EndObject()
}

func fbOperator(b *flatbuffers.Builder, opcodeIndex uint32, inputs, outputs []int32, optionsType byte, options flatbuffers.UOffsetT) flatbuffers.UOffsetT {
	inOff := fbIntVector(b, inputs)
	outOff := fbIntVector(b, outputs)
	b.StartObject(5)
	b.PrependUint32Slot(0, opcodeIndex, 0)
	b.PrependUOffsetTSlot(1, inOff, 0)
	b.PrependUOffsetTSlot(2, outOff, 0)
	if optionsType != 0 {
		b.PrependByteSlot(3, optionsType, 0)
		b.PrependUOffsetTSlot(4, options, 0)
	}
	return b.EndObject()
}

func buildTFLite(l1, l2 *dense) []byte {
	b := flatbuffers.NewBuilder(16 * 1024)

	buffers := []flatbuffers.UOffsetT{
		fbBuffer(b, nil), // buffer 0 is the empty sentinel
		fbBuffer(b, floatBytes(l1.transposed())),
		fbBuffer(b, floatBytes(l1.b)),
		fbBuffer(b, floatBytes(l2.transposed())),
		fbBuffer(b, floatBytes(l2.b)),
	}

	tensors := []flatbuffers.UOffsetT{
		fbTensor(b, "input", []int32{1, numDigits}, 0),
		fbTensor(b, "dense/kernel", []int32{hidden, numDigits}, 1),
		fbTensor(b, "dense/bias", []int32{hidden}, 2),
		fbTensor(b, "dense/BiasAdd", []int32{1, hidden}, 0),
		fbTensor(b, "tanh", []int32{1, hidden}, 0),
		fbTensor(b, "dense_1/kernel", []int32{classes, hidden}, 3),
		fbTensor(b, "dense_1/bias", []int32{classes}, 4),
		fbTensor(b, "dense_1/BiasAdd", []int32{1, classes}, 0),
		fbTensor(b, "output", []int32{1, classes}, 0),
	}

	opcodes := []flatbuffers.UOffsetT{
		fbOperatorCode(b, opFullyConnected),
		fbOperatorCode(b, opTanh),
		fbOperatorCode(b, opSoftmax),
	}

	// FullyConnectedOptions: all fields default
	b.StartObject(4)
	fcOpts1 := b.EndObject()
	b.StartObject(4)
	fcOpts2 := b.EndObject()
	// SoftmaxOptions: beta = 1.0
	b.StartObject(1)
	b.PrependFloat32Slot(0, 1.0, 0.0)
	smOpts := b.EndObject()

	operators := []flatbuffers.UOffsetT{
		fbOperator(b, 0, []int32{0, 1, 2}, []int32{3}, optionsFullyConnected, fcOpts1),
		fbOperator(b, 1, []int32{3}, []int32{4}, 0, 0),
		fbOperator(b, 0, []int32{4, 5, 6}, []int32{7}, optionsFullyConnected, fcOpts2),
		fbOperator(b, 2, []int32{7}, []int32{8}, optionsSoftmax, smOpts),
	}

	subgraphName := b.CreateString("main")
	tensorsOff := fbOffsetVector(b, tensors)
	inputsOff := fbIntVector(b, []int32{0})
	outputsOff := fbIntVector(b, []int32{8})
	operatorsOff := fbOffsetVector(b, operators)
	b.StartObject(5)
	b.PrependUOffsetTSlot(0, tensorsOff, 0)
	b.PrependUOffsetTSlot(1, inputsOff, 0)
	b.PrependUOffsetTSlot(2, outputsOff, 0)
	b.PrependUOffsetTSlot(3, operatorsOff, 0)
	b.PrependUOffsetTSlot(4, subgraphName, 0)
	subgraph := b.EndObject()

	description := b.CreateString("FizzBuzz model converted from Go.")
	opcodesOff := fbOffsetVector(b, opcodes)
	subgraphsOff := fbOffsetVector(b, []flatbuffers.UOffsetT{subgraph})
	buffersOff := fbOffsetVector(b, buffers)
	b.StartObject(5)
	b.PrependUint32Slot(0, 3, 0) // schema version
	b.PrependUOffsetTSlot(1, opcodesOff, 0)
	b.PrependUOffsetTSlot(2, subgraphsOff, 0)
	b.PrependUOffsetTSlot(3, description, 0)
	b.PrependUOffsetTSlot(4, buffersOff, 0)
	model := b.EndObject()

	b.FinishWithFileIdentifier(model, []byte("TFL3"))
	return b.FinishedBytes()
}
