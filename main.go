package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/mattn/go-tflite"
)

func bin(n int, numDigits int) []float32 {
	f := make([]float32, numDigits)
	for i := 0; i < numDigits; i++ {
		f[i] = float32(n >> i & 1)
	}
	return f
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func display(v []float32, i int) {
	switch argmax(v) {
	case 0:
		fmt.Println(i)
	case 1:
		fmt.Println("Fizz")
	case 2:
		fmt.Println("Buzz")
	case 3:
		fmt.Println("FizzBuzz")
	}
}

func main() {
	modelPath := flag.String("model", "fizzbuzz_model.tflite", "path to model file")
	flag.Parse()

	model := tflite.NewModelFromFile(*modelPath)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	interpreter := tflite.NewInterpreter(model, nil)
	if interpreter == nil {
		log.Fatal("cannot create interpreter")
	}
	defer interpreter.Delete()

	if interpreter.AllocateTensors() != tflite.OK {
		log.Fatal("cannot allocate tensors")
	}

	for i := 1; i <= 100; i++ {
		buf := bin(i, 7)
		interpreter.GetInputTensor(0).CopyFromBuffer(&buf[0])
		if interpreter.Invoke() != tflite.OK {
			log.Fatal("cannot invoke interpreter")
		}
		display(interpreter.GetOutputTensor(0).Float32s(), i)
	}
}
