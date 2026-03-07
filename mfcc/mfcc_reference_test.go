package mfcc

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
)

func TestExtractorCalculate_MatchesIndependentReference(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	cfg := DefaultConfig()
	cfg.DisablePreEmphasis = true

	extractor, err := NewExtractor(16_000, cfg)
	require.NoError(t, err)

	samples := make([]float64, extractor.WindowSize())
	for i := range samples {
		phase := 2 * math.Pi * float64(i) / float64(len(samples))
		samples[i] = 0.6*math.Sin(7*phase) + 0.25*math.Cos(13*phase)
	}

	got, err := extractor.Calculate(samples)
	require.NoError(t, err)
	require.Len(t, got, 1)

	want := independentReferenceMFCC(
		samples,
		extractor.sampleRate,
		extractor.windowSize,
		extractor.nfft,
		extractor.numFilters,
		extractor.numCoefficients,
		extractor.energyFloor,
	)

	assert.InDeltaSlice(t, want, got[0], 1e-9)
}

func TestExtractorCalculate_MatchesIndependentReferenceWithPreEmphasisAndFraming(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	cfg := DefaultConfig()

	extractor, err := NewExtractor(16_000, cfg)
	require.NoError(t, err)

	frameCount := 4
	samples := make([]float64, extractor.WindowSize()+(frameCount-1)*extractor.HopSize())
	for i := range samples {
		phase := 2 * math.Pi * float64(i) / float64(len(samples))
		samples[i] = 0.4*math.Sin(11*phase) + 0.15*math.Cos(23*phase) + 0.1*math.Sin(37*phase)
	}

	got, err := extractor.Calculate(samples)
	require.NoError(t, err)
	require.Len(t, got, frameCount)

	want := independentReferenceMFCCs(
		samples,
		extractor.sampleRate,
		extractor.windowSize,
		extractor.hopSize,
		extractor.nfft,
		extractor.numFilters,
		extractor.numCoefficients,
		extractor.preEmphasis,
		extractor.energyFloor,
	)

	require.Len(t, want, len(got))
	for i := range got {
		assert.InDeltaSlice(t, want[i], got[i], 1e-9)
	}
}

func independentReferenceMFCC(samples []float64, sampleRate, windowSize, nfft, numFilters, numCoefficients int, energyFloor float64) []float64 {
	return independentReferenceMFCCs(
		samples,
		sampleRate,
		windowSize,
		windowSize,
		nfft,
		numFilters,
		numCoefficients,
		0,
		energyFloor,
	)[0]
}

func independentReferenceMFCCs(samples []float64, sampleRate, windowSize, hopSize, nfft, numFilters, numCoefficients int, preEmphasis, energyFloor float64) [][]float64 {
	preprocessed := independentPreprocess(samples, preEmphasis)
	frameCount := 1 + (len(preprocessed)-windowSize)/hopSize
	filterBank := independentCreateFilterBank(nfft, sampleRate, numFilters)
	dct := independentDCTMatrix(numCoefficients, numFilters)

	out := make([][]float64, frameCount)
	for frame := 0; frame < frameCount; frame++ {
		start := frame * hopSize
		windowed := independentApplyHamming(preprocessed[start : start+windowSize])
		spectrum := independentDFTReal(windowed, nfft)
		power := independentSingleSidedPowerSpectrum(spectrum, nfft)
		logEnergies := independentApplyFilterBank(power, filterBank, energyFloor)
		out[frame] = independentApplyDCT(logEnergies, dct)
	}
	return out
}

func independentMeanCenter(samples []float64) []float64 {
	mean := 0.0
	for _, v := range samples {
		mean += v
	}
	mean /= float64(len(samples))

	out := make([]float64, len(samples))
	for i, v := range samples {
		out[i] = v - mean
	}
	return out
}

func independentPreprocess(samples []float64, preEmphasis float64) []float64 {
	centered := independentMeanCenter(samples)
	if len(centered) == 0 {
		return centered
	}
	out := make([]float64, len(centered))
	out[0] = centered[0]
	for i := 1; i < len(centered); i++ {
		out[i] = centered[i] - preEmphasis*centered[i-1]
	}
	return out
}

func independentApplyHamming(samples []float64) []float64 {
	out := make([]float64, len(samples))
	if len(samples) == 1 {
		out[0] = samples[0]
		return out
	}
	for i, v := range samples {
		w := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(len(samples)-1))
		out[i] = v * w
	}
	return out
}

func independentApplyDCT(logEnergies []float64, dct [][]float64) []float64 {
	out := make([]float64, len(dct))
	for i := range out {
		sum := 0.0
		for j := range logEnergies {
			sum += dct[i][j] * logEnergies[j]
		}
		out[i] = sum
	}
	return out
}

func independentDFTReal(samples []float64, nfft int) []complex128 {
	out := make([]complex128, nfft)
	for k := 0; k < nfft; k++ {
		sum := complex(0, 0)
		for n := 0; n < len(samples); n++ {
			angle := -2 * math.Pi * float64(k*n) / float64(nfft)
			sum += complex(samples[n], 0) * cmplxExp(angle)
		}
		out[k] = sum
	}
	return out
}

func independentSingleSidedPowerSpectrum(spectrum []complex128, nfft int) []float64 {
	binCount := nfft/2 + 1
	out := make([]float64, binCount)
	for i := 0; i < binCount; i++ {
		v := spectrum[i]
		power := (real(v)*real(v) + imag(v)*imag(v)) / float64(nfft)
		if i != 0 && i != binCount-1 {
			power *= 2
		}
		out[i] = power
	}
	return out
}

func independentCreateFilterBank(nfft, sampleRate, numFilters int) [][]float64 {
	lowerMel := 0.0
	upperMel := 2595.0 * math.Log10(1+(float64(sampleRate)/2)/700.0)

	melPoints := make([]float64, numFilters+2)
	step := (upperMel - lowerMel) / float64(len(melPoints)-1)
	for i := range melPoints {
		melPoints[i] = lowerMel + float64(i)*step
	}

	hzPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		hzPoints[i] = 700.0 * (math.Pow(10, mel/2595.0) - 1)
	}

	binCount := nfft/2 + 1
	binPoints := make([]int, len(hzPoints))
	for i, hz := range hzPoints {
		bin := int(math.Floor((float64(nfft) + 1) * hz / float64(sampleRate)))
		if bin < 0 {
			bin = 0
		}
		if bin >= binCount {
			bin = binCount - 1
		}
		binPoints[i] = bin
	}

	filterBank := make([][]float64, numFilters)
	for i := range filterBank {
		filter := make([]float64, binCount)
		start := binPoints[i]
		mid := binPoints[i+1]
		end := binPoints[i+2]

		if start != mid {
			denom := float64(mid - start)
			for j := start; j < mid; j++ {
				filter[j] = (float64(j) - float64(start)) / denom
			}
		}
		if mid != end {
			denom := float64(end - mid)
			for j := mid; j < end; j++ {
				filter[j] = (float64(end) - float64(j)) / denom
			}
		}

		bandWidth := hzPoints[i+2] - hzPoints[i]
		if bandWidth > 0 {
			scale := 2.0 / bandWidth
			for j := range filter {
				filter[j] *= scale
			}
		}
		filterBank[i] = filter
	}

	return filterBank
}

func independentApplyFilterBank(power []float64, filterBank [][]float64, energyFloor float64) []float64 {
	out := make([]float64, len(filterBank))
	logFloor := math.Log(energyFloor)
	for i, filter := range filterBank {
		sum := 0.0
		for j, weight := range filter {
			sum += power[j] * weight
		}
		if sum < energyFloor {
			out[i] = logFloor
			continue
		}
		out[i] = math.Log(sum)
	}
	return out
}

func independentDCTMatrix(numCoefficients, numFilters int) [][]float64 {
	out := make([][]float64, numCoefficients)
	for i := range out {
		scale := math.Sqrt(2.0 / float64(numFilters))
		if i == 0 {
			scale = math.Sqrt(1.0 / float64(numFilters))
		}
		row := make([]float64, numFilters)
		for j := range row {
			row[j] = scale * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(numFilters))
		}
		out[i] = row
	}
	return out
}

func cmplxExp(angle float64) complex128 {
	sin, cos := math.Sincos(angle)
	return complex(cos, sin)
}
