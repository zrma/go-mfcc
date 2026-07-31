package mfcc

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
)

func TestFindOffsetFrameIndexFFT_MatchesNaiveOnExactMatch(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	const (
		coeffCount = defaultNumCoefficients
		nFrames    = 4096
		mFrames    = 512
		wantOffset = 1234
	)

	r := rand.New(rand.NewSource(1))

	whole := make([][]float64, nFrames)
	for i := range whole {
		frame := make([]float64, coeffCount)
		for k := range frame {
			frame[k] = r.NormFloat64()
		}
		whole[i] = frame
	}

	chunk := make([][]float64, mFrames)
	for i := range chunk {
		src := whole[wantOffset+i]
		dst := make([]float64, len(src))
		copy(dst, src)
		chunk[i] = dst
	}

	startCoeff := distanceStartCoeff
	gotNaive, naiveDistance := findOffsetFrameIndexNaiveWithDistance(whole, chunk, coeffCount, startCoeff)
	gotFFT, fftDistance, ok := findOffsetFrameIndexFFTWithDistance(whole, chunk, coeffCount, startCoeff)
	require.True(t, ok)

	assert.Equal(t, wantOffset, gotNaive)
	assert.Equal(t, wantOffset, gotFFT)
	assert.InDelta(t, 0.0, naiveDistance, 1e-12)
	assert.InDelta(t, 0.0, fftDistance, 1e-7)
}

func TestFindOffsetFrameIndexFFT_TiedMatchesPreferEarliest(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	const (
		coeffCount = defaultNumCoefficients
		nFrames    = 4096
		mFrames    = 512
	)

	whole := make([][]float64, nFrames)
	for i := range whole {
		whole[i] = make([]float64, coeffCount)
		whole[i][1] = 1
	}
	chunk := make([][]float64, mFrames)
	for i := range chunk {
		chunk[i] = make([]float64, coeffCount)
		chunk[i][1] = 1
	}

	offset, distance, ok := findOffsetFrameIndexFFTWithDistance(whole, chunk, coeffCount, distanceStartCoeff)
	require.True(t, ok)
	assert.Zero(t, offset)
	assert.InDelta(t, 0.0, distance, 1e-7)
}
