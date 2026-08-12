package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync/atomic"
	"time"

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

// apply performs one Adam update from accumulated gradients.
func (d *dense) apply(gw, gb []float64, t int) {
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

// grad holds per-batch gradient accumulators, allocated once.
type grad struct {
	w1, b1, w2, b2 []float64
}

func newGrad() *grad {
	return &grad{
		w1: make([]float64, numDigits*hidden),
		b1: make([]float64, hidden),
		w2: make([]float64, hidden*classes),
		// padded to a full cache line: workers write b2 on every sample, and
		// 32-byte allocations from different workers can share a line
		b2: make([]float64, 8)[:classes],
	}
}

func (g *grad) reset() {
	clear(g.w1)
	clear(g.b1)
	clear(g.w2)
	clear(g.b2)
}

func (g *grad) add(o *grad) {
	for i, v := range o.w1 {
		g.w1[i] += v
	}
	for i, v := range o.b1 {
		g.b1[i] += v
	}
	for i, v := range o.w2 {
		g.w2[i] += v
	}
	for i, v := range o.b2 {
		g.b2[i] += v
	}
}

// forward computes h = tanh(x.W1 + b1) and p = softmax(h.W2 + b2).
func forward(l1, l2 *dense, x []float64, h *[hidden]float64, p *[classes]float64) {
	copy(h[:], l1.b)
	for i, v := range x {
		if v == 0 {
			continue
		}
		w := l1.w[i*hidden : (i+1)*hidden]
		for j := range h {
			h[j] += v * w[j]
		}
	}
	for j := range h {
		h[j] = math.Tanh(h[j])
	}

	copy(p[:], l2.b)
	for j := range h {
		w := l2.w[j*classes : (j+1)*classes]
		for k := range p {
			p[k] += h[j] * w[k]
		}
	}
	max := p[0]
	for _, v := range p[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for k := range p {
		p[k] = math.Exp(p[k] - max)
		sum += p[k]
	}
	for k := range p {
		p[k] /= sum
	}
}

// step runs forward + backward for one sample and accumulates gradients.
func step(l1, l2 *dense, x, y []float64, invN float64, g *grad) {
	var h [hidden]float64
	var p [classes]float64
	forward(l1, l2, x, &h, &p)

	// categorical crossentropy + softmax: dz2 = (p - y) / n
	var d2 [classes]float64
	for k := range d2 {
		d2[k] = (p[k] - y[k]) * invN
		g.b2[k] += d2[k]
	}

	// gw2 += h x d2, dz1 = (dz2 . W2^T) * (1 - h^2)
	var d1 [hidden]float64
	for j := range h {
		w := l2.w[j*classes : (j+1)*classes]
		gw := g.w2[j*classes : (j+1)*classes]
		var dh float64
		for k, d := range d2 {
			gw[k] += h[j] * d
			dh += w[k] * d
		}
		d1[j] = dh * (1 - h[j]*h[j])
		g.b1[j] += d1[j]
	}
	for i, v := range x {
		if v == 0 {
			continue
		}
		gw := g.w1[i*hidden : (i+1)*hidden]
		for j := range d1 {
			gw[j] += v * d1[j]
		}
	}
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

func evaluate(l1, l2 *dense, x, y [][]float64) (loss, acc float64) {
	var h [hidden]float64
	var p [classes]float64
	for n := range x {
		forward(l1, l2, x[n], &h, &p)
		loss -= math.Log(math.Max(p[argmax(y[n])], 1e-12))
		if argmax(p[:]) == argmax(y[n]) {
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
	idx := make([]int, samples)
	for i := 1; i <= samples; i++ {
		trX[i-1] = bin(i, numDigits)
		trY[i-1] = fizzbuzz(i)
		idx[i-1] = i - 1
	}

	l1 := newDense(numDigits, hidden, r)
	l2 := newDense(hidden, classes, r)
	fmt.Printf("model: dense(%d->%d) tanh, dense(%d->%d) softmax, %d params\n",
		numDigits, hidden, hidden, classes,
		len(l1.w)+len(l1.b)+len(l2.w)+len(l2.b))

	// Data-parallel workers, each with its own gradient accumulator. A batch
	// is only ~30µs of work, so sleeping between batches costs more in futex
	// wake-ups than it saves: persistent workers spin on an atomic sequence
	// number instead, and the main goroutine processes chunk 0 itself.
	nw := min(runtime.GOMAXPROCS(0), 4)
	grads := make([]*grad, nw)
	for i := range grads {
		grads[i] = newGrad()
	}
	var (
		ids       []int
		invN      float64
		chunk     int
		seq, done atomic.Uint32
		quit      atomic.Bool
	)
	for w := 1; w < nw; w++ {
		go func(w int, g *grad) {
			for last := uint32(0); ; last++ {
				for spins := 0; seq.Load() == last; spins++ {
					if quit.Load() {
						return
					}
					if spins > 1000 {
						// yield so an oversubscribed scheduler can run others
						runtime.Gosched()
						spins = 0
					}
				}
				g.reset()
				lo := w * chunk
				for _, i := range ids[min(lo, len(ids)):min(lo+chunk, len(ids))] {
					step(l1, l2, trX[i], trY[i], invN, g)
				}
				done.Add(1)
			}
		}(w, grads[w])
	}

	started := time.Now()
	t := 0
	for epoch := 1; epoch <= epochs; epoch++ {
		r.Shuffle(samples, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
		for lo := 0; lo < samples; lo += batchSize {
			hi := min(lo+batchSize, samples)
			ids = idx[lo:hi]
			invN = 1 / float64(hi-lo)
			chunk = (hi - lo + nw - 1) / nw
			done.Store(0)
			seq.Add(1)
			grads[0].reset()
			for _, i := range ids[:min(chunk, len(ids))] {
				step(l1, l2, trX[i], trY[i], invN, grads[0])
			}
			for spins := 0; done.Load() < uint32(nw-1); spins++ {
				if spins > 1000 {
					runtime.Gosched()
					spins = 0
				}
			}
			for w := 1; w < nw; w++ {
				grads[0].add(grads[w])
			}
			t++
			l2.apply(grads[0].w2, grads[0].b2, t)
			l1.apply(grads[0].w1, grads[0].b1, t)
		}
		if epoch%100 == 0 || epoch == epochs {
			loss, acc := evaluate(l1, l2, trX, trY)
			fmt.Printf("epoch %4d/%d - loss: %.4f - accuracy: %.4f\n", epoch, epochs, loss, acc)
		}
	}
	quit.Store(true)
	fmt.Printf("training took %v\n", time.Since(started))

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
