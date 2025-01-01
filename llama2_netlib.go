//go:build (cgo && darwin && netlib) || (cgo && !darwin)

package llama2

// #cgo LDFLAGS: -lopenblas
import "C"

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/netlib/blas/netlib"
)

var blas32 netlib.Implementation

func matmul(xout, x, w []float32) {
	blas32.Sgemv(
		blas.NoTrans,
		len(xout),
		len(x),
		1,
		w,
		len(x),
		x,
		1,
		0, xout, 1)
}
