//go:build ((cgo && darwin && netlib) || (cgo && !darwin)) && !nonetlib

package llama2

// #cgo LDFLAGS: -lopenblas
import "C"

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/netlib/blas/netlib"
)

func init() {
	fmt.Fprintln(os.Stderr, "Using netlib for matrix multiplication")
}

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
