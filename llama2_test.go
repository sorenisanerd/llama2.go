package llama2

import "testing"

func Test_matmul(t *testing.T) {
	type args struct {
		xout []float32
		x    []float32
		w    []float32
	}
	tests := []struct {
		name string
		args args
		want []float32
	}{
		{
			name: "2x2_matrix_multiplication",
			args: args{
				xout: make([]float32, 2),
				x:    []float32{1, 2},
				w:    []float32{1, 2, 3, 4},
			},
			want: []float32{5, 11},
		},
		{
			name: "3x3_matrix_multiplication",
			args: args{
				xout: make([]float32, 3),
				x:    []float32{1, 2, 3},
				w:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			},
			want: []float32{14, 32, 50},
		},
		{
			name: "xout_not_part_of_input",
			args: args{
				xout: []float32{1, 1},
				x:    []float32{1, 2},
				w:    []float32{1, 2, 3, 4},
			},
			want: []float32{5, 11},
		},
		{
			name: "2x3_matrix_multiplication",
			args: args{
				xout: make([]float32, 3),
				x:    []float32{1, 2},
				w:    []float32{1, 2, 3, 4, 5, 6},
			},
			want: []float32{5, 11, 17},
		},
		{
			name: "1x1_matrix_multiplication",
			args: args{
				xout: make([]float32, 1),
				x:    []float32{2},
				w:    []float32{3},
			},
			want: []float32{6},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matmul(tt.args.xout, tt.args.x, tt.args.w)
			for i := range tt.args.xout {
				if tt.args.xout[i] != tt.want[i] {
					t.Errorf("matmul() = %v, want %v", tt.args.xout, tt.want)
				}
			}
		})
	}
}
