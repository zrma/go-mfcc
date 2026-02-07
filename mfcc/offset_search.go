package mfcc

import (
	"math"
)

const (
	// offsetSearchFFTThresholdOps는 오프셋 탐색에서 FFT 기반 컨볼루션이
	// 이득이 될 가능성이 높은 대략적인 연산량 기준이다.
	// (naive: O((N-M+1)*M*D), FFT: O(D*n*log2(n)))
	offsetSearchFFTThresholdOps = 5_000_000

	// offsetSearchMaxFFTSize는 너무 큰 FFT로 인해 메모리 사용이 폭증하는 것을 방지한다.
	// n=2^22이면 complex128 2개 버퍼(입력/출력)만으로도 수백 MB가 될 수 있다.
	offsetSearchMaxFFTSize = 1 << 22
)

func findOffsetFrameIndex(mfccWhole, mfccChunk [][]float64) int {
	if len(mfccChunk) == 0 || len(mfccWhole) < len(mfccChunk) {
		return 0
	}

	coeffCount := coeffCountForMFCCs(mfccChunk, mfccWhole)
	startCoeff := distanceStartCoeff
	if coeffCount <= startCoeff {
		return 0
	}

	dim := coeffCount - startCoeff
	naiveOps := int64(len(mfccWhole)-len(mfccChunk)+1) * int64(len(mfccChunk)) * int64(dim)

	useFFT := len(mfccWhole) >= 2048 && len(mfccChunk) >= 128 && naiveOps >= offsetSearchFFTThresholdOps
	if useFFT {
		if offset, ok := findOffsetFrameIndexFFT(mfccWhole, mfccChunk, coeffCount, startCoeff); ok {
			return offset
		}
	}

	return findOffsetFrameIndexNaive(mfccWhole, mfccChunk, coeffCount, startCoeff)
}

func findOffsetFrameIndexNaive(mfccWhole, mfccChunk [][]float64, coeffCount, startCoeff int) int {
	if len(mfccChunk) == 0 || len(mfccWhole) < len(mfccChunk) {
		return 0
	}
	if coeffCount <= startCoeff {
		return 0
	}

	minDistance := math.MaxFloat64
	offset := 0

	limit := len(mfccWhole) - len(mfccChunk) + 1
	for i := range limit {
		distance := 0.0
	loop:
		for j := range len(mfccChunk) {
			chunkFrame := mfccChunk[j]
			wholeFrame := mfccWhole[i+j]

			for k := startCoeff; k < coeffCount; k++ {
				diff := wholeFrame[k] - chunkFrame[k]
				distance += diff * diff
				if distance >= minDistance {
					break loop
				}
			}
		}
		if distance < minDistance {
			minDistance = distance
			offset = i
		}
	}

	return offset
}

func findOffsetFrameIndexFFT(mfccWhole, mfccChunk [][]float64, coeffCount, startCoeff int) (int, bool) {
	if len(mfccChunk) == 0 || len(mfccWhole) < len(mfccChunk) {
		return 0, false
	}
	if coeffCount <= startCoeff {
		return 0, false
	}

	n := len(mfccWhole)
	m := len(mfccChunk)
	nConv := n + m - 1
	nFFT := nextPow2(nConv)
	if nFFT <= 0 || nFFT&(nFFT-1) != 0 {
		return 0, false
	}
	if nFFT > offsetSearchMaxFFTSize {
		return 0, false
	}

	outLen := n - m + 1

	// 전체/청크 에너지(제곱합)
	wholeFrameEnergy := make([]float64, n)
	for i := range mfccWhole {
		frame := mfccWhole[i]
		sum := 0.0
		for k := startCoeff; k < coeffCount; k++ {
			v := frame[k]
			sum += v * v
		}
		wholeFrameEnergy[i] = sum
	}
	prefix := make([]float64, n+1)
	for i := range wholeFrameEnergy {
		prefix[i+1] = prefix[i] + wholeFrameEnergy[i]
	}

	chunkEnergy := 0.0
	for i := range mfccChunk {
		frame := mfccChunk[i]
		for k := startCoeff; k < coeffCount; k++ {
			v := frame[k]
			chunkEnergy += v * v
		}
	}

	// corrSum[offset] = sum_k sum_j whole[offset+j,k]*chunk[j,k]
	corrSum := make([]float64, outLen)

	fa := make([]complex128, nFFT)
	fb := make([]complex128, nFFT)

	for coeff := startCoeff; coeff < coeffCount; coeff++ {
		clear(fa)
		clear(fb)

		for i := 0; i < n; i++ {
			fa[i] = complex(mfccWhole[i][coeff], 0)
		}
		for j := 0; j < m; j++ {
			fb[j] = complex(mfccChunk[m-1-j][coeff], 0)
		}

		fftInPlace(fa, false)
		fftInPlace(fb, false)
		for i := range fa {
			fa[i] *= fb[i]
		}
		fftInPlace(fa, true)

		base := m - 1
		for offset := 0; offset < outLen; offset++ {
			corrSum[offset] += real(fa[offset+base])
		}
	}

	best := 0
	minDistance := math.Inf(1)
	for offset := 0; offset < outLen; offset++ {
		wholeEnergyWindow := prefix[offset+m] - prefix[offset]
		distance := wholeEnergyWindow + chunkEnergy - 2*corrSum[offset]
		if distance < minDistance {
			minDistance = distance
			best = offset
		}
	}

	return best, true
}

// fftInPlace는 길이가 2의 거듭제곱인 입력에 대해 in-place FFT/IFFT를 수행한다.
// inverse=true이면 역변환을 수행하며, 결과는 1/N으로 스케일된다.
func fftInPlace(x []complex128, inverse bool) {
	n := len(x)
	if n <= 1 {
		return
	}

	// bit-reversal permutation (in-place)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}

	// iterative Cooley-Tukey
	for length := 2; length <= n; length <<= 1 {
		half := length / 2
		angle := 2 * math.Pi / float64(length)
		if !inverse {
			angle = -angle
		}
		sin, cos := math.Sincos(angle)
		wlen := complex(cos, sin)

		for i := 0; i < n; i += length {
			w := complex(1.0, 0.0)
			for j := 0; j < half; j++ {
				u := x[i+j]
				v := x[i+j+half] * w
				x[i+j] = u + v
				x[i+j+half] = u - v
				w *= wlen
			}
		}
	}

	if inverse {
		scale := complex(1.0/float64(n), 0)
		for i := range x {
			x[i] *= scale
		}
	}
}
