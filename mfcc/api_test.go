package mfcc

import (
	"math"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
)

func TestComputeMFCCFromWav(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "mono.wav")

	windowSize, hopSize := frameParamsFromDurations(16_000, defaultWindowDuration, defaultHopDuration)
	writeTestWav(t, path, 16_000, 16, 1, make([]int, windowSize))

	mfccs, sampleRate, gotHop, err := ComputeMFCCFromWav(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)
	assert.Equal(t, hopSize, gotHop)

	require.Len(t, mfccs, 1)
	assert.Len(t, mfccs[0], defaultNumCoefficients)
}

func TestNewExtractor_CustomConfig(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	cfg := DefaultConfig()
	cfg.NumFilters = 10
	cfg.NumCoefficients = 5
	cfg.DisablePreEmphasis = true

	extractor, err := NewExtractor(16_000, cfg)
	require.NoError(t, err)
	assert.Equal(t, 10, extractor.NumFilters())
	assert.Equal(t, 5, extractor.NumCoefficients())

	samples := make([]float64, extractor.WindowSize())
	mfccs, err := extractor.Calculate(samples)
	require.NoError(t, err)
	require.Len(t, mfccs, 1)
	assert.Len(t, mfccs[0], 5)
}

func TestNewExtractor_InvalidDurations(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	cfg := DefaultConfig()
	cfg.WindowDuration = -5 * time.Millisecond
	_, err := NewExtractor(16_000, cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid window duration")

	cfg = DefaultConfig()
	cfg.HopDuration = -5 * time.Millisecond
	_, err = NewExtractor(16_000, cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid hop duration")
}

func TestComputeASRFeatures_Stacked(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	sampleRate := 16_000
	windowSize, hopSize := frameParamsFromDurations(sampleRate, defaultWindowDuration, defaultHopDuration)
	samples := make([]float64, windowSize+2*hopSize)

	mfccs, hop, err := ComputeMFCC(samples, sampleRate)
	require.NoError(t, err)

	features, gotHop, err := ComputeASRFeatures(samples, sampleRate)
	require.NoError(t, err)
	assert.Equal(t, hop, gotHop)
	assert.Len(t, features, len(mfccs))

	if len(mfccs) > 0 {
		assert.Len(t, features[0], len(mfccs[0])*3)
	}

	for _, frame := range features {
		for _, v := range frame {
			assert.False(t, math.IsNaN(v) || math.IsInf(v, 0))
		}
	}
}

func TestComputeDelta_Linear(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	features := [][]float64{
		{0},
		{1},
		{2},
		{3},
		{4},
	}

	delta, err := ComputeDelta(features, 2)
	require.NoError(t, err)
	require.Len(t, delta, len(features))
	assert.InDelta(t, 1.0, delta[2][0], 1e-12)
}

func TestComputeDelta_RejectsInvalidFeatures(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	tests := []struct {
		name     string
		features [][]float64
	}{
		{"ragged", [][]float64{{1, 2}, {3}}},
		{"nan", [][]float64{{1, math.NaN()}}},
		{"inf", [][]float64{{1, math.Inf(1)}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ComputeDelta(tt.features, 2)
			require.Error(t, err)
		})
	}
}

func TestAppendDeltas_OrderZeroIgnoresWindow(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	features := [][]float64{
		{1, 2},
		{3, 4},
	}

	out, err := AppendDeltas(features, 0, 0)
	require.NoError(t, err)
	require.Len(t, out, len(features))

	for i := range features {
		assert.InDeltaSlice(t, features[i], out[i], 0)
		if len(features[i]) > 0 {
			assert.False(t, &features[i][0] == &out[i][0])
		}
	}
}

func TestAppendDeltas_RejectsNegativeOrder(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	_, err := AppendDeltas([][]float64{{1, 2}}, 2, -1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid delta order")
}

func TestComputeASRFeaturesWithConfig_RejectsNegativeDeltaWindow(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	_, _, err := ComputeASRFeaturesWithConfig(nil, 16_000, DefaultConfig(), -1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid delta window")
}

func TestExtractorCalculate_RejectsInvalidSamples(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := NewExtractor(16_000, DefaultConfig())
	require.NoError(t, err)

	tests := []struct {
		name  string
		value float64
	}{
		{"nan", math.NaN()},
		{"inf", math.Inf(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			samples := make([]float64, extractor.WindowSize())
			samples[0] = tt.value

			_, err := extractor.Calculate(samples)
			require.Error(t, err)
			assert.Contains(t, err.Error(), "invalid sample")
		})
	}
}

func TestComputeCMVN_RespectsStdFloorOnZeroVariance(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{
		{1, 2},
		{1, 2},
	}
	stdFloor := 0.5

	mean, std := ComputeCMVN(mfcc, stdFloor)

	assert.InDeltaSlice(t, []float64{1, 2}, mean, 1e-12)
	assert.InDeltaSlice(t, []float64{stdFloor, stdFloor}, std, 1e-12)
}

func TestComputeCMVNChecked_RejectsInvalidMatrix(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mean, std, err := ComputeCMVNChecked([][]float64{{1, 2}, {3}}, 0)
	require.Error(t, err)
	assert.Nil(t, mean)
	assert.Nil(t, std)
	assert.Contains(t, err.Error(), "inconsistent coefficient count")
}

func TestApplyCMVN_FloorsProvidedStdWithoutMutation(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{
		{1, 2},
		{3, 4},
	}
	mean := []float64{0, 0}
	std := []float64{0, 0.1}
	stdCopy := slices.Clone(std)
	stdFloor := 0.5

	gotMean, gotStd := ApplyCMVN(mfcc, mean, std, stdFloor)

	assert.Equal(t, mean, gotMean)
	assert.Equal(t, stdCopy, std)
	assert.InDeltaSlice(t, []float64{stdFloor, stdFloor}, gotStd, 1e-12)

	expected := [][]float64{
		{2, 4},
		{6, 8},
	}
	for i := range expected {
		assert.InDeltaSlice(t, expected[i], mfcc[i], 1e-12)
	}
}

func TestApplyCMVNChecked_RejectsInvalidStatsBeforeMutation(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{{1, 2}, {3, 4}}
	want := [][]float64{{1, 2}, {3, 4}}

	_, _, err := ApplyCMVNChecked(mfcc, []float64{0}, nil, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mean coefficient count mismatch")
	assert.Equal(t, want, mfcc)

	gotMean, gotStd := ApplyCMVN(mfcc, nil, []float64{1, math.NaN()}, 0)
	assert.Nil(t, gotMean)
	assert.Nil(t, gotStd)
	assert.Equal(t, want, mfcc)
}

func TestExtractorCalculate_ValidatesMFCCOutput(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := NewExtractor(16_000, DefaultConfig())
	require.NoError(t, err)

	dct := make([][]float64, len(extractor.dctMatrix))
	for i := range dct {
		dct[i] = slices.Clone(extractor.dctMatrix[i])
	}
	dct[0][0] = math.NaN()
	extractor.dctMatrix = dct

	samples := make([]float64, extractor.WindowSize())
	_, err = extractor.Calculate(samples)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid MFCC")
}
