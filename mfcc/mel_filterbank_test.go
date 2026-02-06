package mfcc

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
)

func TestMelScale_Invertible(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	freqs := []float64{0, 1, 50, 100, 300, 1_000, 4_000, 8_000}
	for _, freq := range freqs {
		mel := hzToMel(freq)
		back := melToHz(mel)
		tol := 1e-9 * math.Max(1, freq)
		assert.InDelta(t, freq, back, tol)
	}

	// HTK mel 정의에서는 1000Hz 부근이 대략 1000 mel 근처에 온다.
	assert.InDelta(t, 1_000.0, hzToMel(1_000.0), 0.1)
}

func TestCreateFilterBank_SlaneyNormalization_ApproxUnitArea(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	const (
		sampleRate = 16_000
		nfft       = 512
	)

	fb := createFilterBank(nfft, sampleRate, defaultNumFilters)
	require.Len(t, fb, defaultNumFilters)

	df := float64(sampleRate) / float64(nfft) // FFT bin width in Hz
	binCount := nfft/2 + 1

	for i, filter := range fb {
		require.Len(t, filter, binCount)

		sum := 0.0
		for _, w := range filter {
			assert.GreaterOrEqual(t, w, 0.0)
			sum += w
		}
		require.Greater(t, sum, 0.0, "filter %d is empty", i)

		areaHz := sum * df
		assert.InDeltaf(t, 1.0, areaHz, 0.15, "filter %d area=%f", i, areaHz)
	}
}
