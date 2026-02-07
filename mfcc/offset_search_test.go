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
	gotNaive := findOffsetFrameIndexNaive(whole, chunk, coeffCount, startCoeff)
	gotFFT, ok := findOffsetFrameIndexFFT(whole, chunk, coeffCount, startCoeff)
	require.True(t, ok)

	assert.Equal(t, wantOffset, gotNaive)
	assert.Equal(t, wantOffset, gotFFT)
}
