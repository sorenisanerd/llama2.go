package llama2

import (
	"fmt"
	"runtime"
	"testing"
)

func BenchmarkMatmul(b *testing.B) {
	// Test with different sizes
	sizes := []int{32, 64, 128, 256, 512}

	for _, size := range sizes {
		x := make([]float32, size)
		w := make([]float32, size*size)
		xout := make([]float32, size)

		// Fill with test data
		for i := range x {
			x[i] = 1.0
			for j := range x {
				w[i*size+j] = 1.0
			}
		}

		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			runtime.GC()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matmul(xout, x, w)
			}
		})
	}
}
