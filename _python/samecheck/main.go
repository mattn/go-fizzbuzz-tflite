// Same-training check: runs the exact training-loop math from
// go-fizzbuzz-tflite/cmd/train with fixed initial weights and a fixed batch
// order (no shuffle), so the trajectory can be compared against Keras.
package main

import (
	"encoding/json"
	"log"
	"math"
	"os"
	"path/filepath"
)

const (
	numDigits = 7
	hidden    = 64
	classes   = 4
	samples   = 100
	epochs    = 100

	lr      = 0.001
	beta1   = 0.9
	beta2   = 0.999
	epsilon = 1e-7
)

type dense struct {
	in, out        int
	w, b           []float64
	mw, vw, mb, vb []float64
}

func (d *dense) apply(gw, gb []float64, t int) {
	alpha := lr * math.Sqrt(1-math.Pow(beta2, float64(t))) / (1 - math.Pow(beta1, float64(t)))
	adam := func(w, m, v, g []float64) {
		for i, gi := range g {
			m[i] = beta1*m[i] + (1-beta1)*gi
			v[i] = beta2*v[i] + (1-beta2)*gi*gi
			w[i] -= alpha * m[i] / (math.Sqrt(v[i]) + epsilon)
		}
	}
	adam(d.w, d.mw, d.vw, gw)
	adam(d.b, d.mb, d.vb, gb)
}

type grad struct {
	w1, b1, w2, b2 []float64
}

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

// step is cmd/train's step plus per-sample loss reporting.
func step(l1, l2 *dense, x, y []float64, invN float64, g *grad) float64 {
	var h [hidden]float64
	var p [classes]float64
	forward(l1, l2, x, &h, &p)

	var loss float64
	var d2 [classes]float64
	for k := range d2 {
		if y[k] == 1 {
			loss = -math.Log(math.Max(p[k], 1e-12))
		}
		d2[k] = (p[k] - y[k]) * invN
		g.b2[k] += d2[k]
	}
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
	return loss
}

type weights struct {
	W1 [][]float64 `json:"w1"`
	B1 []float64   `json:"b1"`
	W2 [][]float64 `json:"w2"`
	B2 []float64   `json:"b2"`
}

func flatten(m [][]float64) []float64 {
	out := make([]float64, 0, len(m)*len(m[0]))
	for _, row := range m {
		out = append(out, row...)
	}
	return out
}

func nest(v []float64, rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = v[i*cols : (i+1)*cols]
	}
	return out
}

func snap(l1, l2 *dense) weights {
	c := func(v []float64) []float64 { return append([]float64(nil), v...) }
	return weights{
		W1: nest(c(l1.w), numDigits, hidden), B1: c(l1.b),
		W2: nest(c(l2.w), hidden, classes), B2: c(l2.b),
	}
}

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

func main() {
	here := "."
	if len(os.Args) > 1 {
		here = os.Args[1]
	}
	raw, err := os.ReadFile(filepath.Join(here, "init.json"))
	if err != nil {
		log.Fatal(err)
	}
	var init weights
	if err := json.Unmarshal(raw, &init); err != nil {
		log.Fatal(err)
	}
	l1 := &dense{in: numDigits, out: hidden,
		w: flatten(init.W1), b: append([]float64(nil), init.B1...),
		mw: make([]float64, numDigits*hidden), vw: make([]float64, numDigits*hidden),
		mb: make([]float64, hidden), vb: make([]float64, hidden)}
	l2 := &dense{in: hidden, out: classes,
		w: flatten(init.W2), b: append([]float64(nil), init.B2...),
		mw: make([]float64, hidden*classes), vw: make([]float64, hidden*classes),
		mb: make([]float64, classes), vb: make([]float64, classes)}

	trX := make([][]float64, samples)
	trY := make([][]float64, samples)
	for i := 1; i <= samples; i++ {
		trX[i-1] = bin(i, numDigits)
		trY[i-1] = fizzbuzz(i)
	}

	g := &grad{
		w1: make([]float64, numDigits*hidden), b1: make([]float64, hidden),
		w2: make([]float64, hidden*classes), b2: make([]float64, classes),
	}
	var losses []float64
	var after1 *weights
	t := 0
	for epoch := 0; epoch < epochs; epoch++ {
		for _, r := range [][2]int{{0, 64}, {64, 100}} {
			clear(g.w1)
			clear(g.b1)
			clear(g.w2)
			clear(g.b2)
			invN := 1 / float64(r[1]-r[0])
			var loss float64
			for i := r[0]; i < r[1]; i++ {
				loss += step(l1, l2, trX[i], trY[i], invN, g)
			}
			losses = append(losses, loss*invN)
			t++
			l2.apply(g.w2, g.b2, t)
			l1.apply(g.w1, g.b1, t)
			if after1 == nil {
				s := snap(l1, l2)
				after1 = &s
			}
		}
	}
	out, _ := json.Marshal(map[string]any{
		"losses": losses, "after1": after1, "final": snap(l1, l2),
	})
	if err := os.WriteFile(filepath.Join(here, "go.json"), out, 0644); err != nil {
		log.Fatal(err)
	}
}
