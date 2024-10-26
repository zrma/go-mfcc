package mfcc

import (
	"math"
	"math/cmplx"
	"os"

	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
	"github.com/pkg/errors"
	"go.uber.org/multierr"
)

func FindOffset(wholeWavPath, chunkWavPath string) (float64, error) {
	wholeSamples, sampleRate, err := readWavFile(wholeWavPath)
	if err != nil {
		return 0, errors.Wrap(err, "read whole wav file failed")
	}

	chunkSamples, _, err := readWavFile(chunkWavPath)
	if err != nil {
		return 0, errors.Wrap(err, "read chunk wav file failed")
	}

	if len(wholeSamples) < len(chunkSamples) {
		return 0, errors.New("whole wav file is shorter than chunk wav file")
	}

	mfccWhole, hopSize := calculateMFCC(wholeSamples, sampleRate)
	mfccChunk, _ := calculateMFCC(chunkSamples, sampleRate)

	return findOffset(mfccWhole, mfccChunk, sampleRate, hopSize), nil
}

func readWavFile(filepath string) (data []float64, sampleRate int, err error) {
	file0, err := os.Open(filepath)
	if err != nil {
		return nil, 0, errors.Wrap(err, "open wav file failed")
	}
	defer func() {
		if err0 := file0.Close(); err0 != nil {
			err = multierr.Append(err, err0)
		}
	}()

	decoder := wav.NewDecoder(file0)
	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, 0, errors.Wrap(err, "decode wav file failed")
	}

	data = make([]float64, len(buf.Data))
	for i := range data {
		data[i] = float64(buf.Data[i]) / 32_768.0
	}

	sampleRate = int(decoder.SampleRate)
	return data, sampleRate, nil
}

func calculateMFCC(samples []float64, sampleRate int) ([][]float64, int) {
	windowSize := int(float64(sampleRate) * 0.025) // 25ms
	hopSize := int(float64(sampleRate) * 0.010)    // 10ms

	res := make([][]float64, 0)

	hannWindow := window.Hann(windowSize)
	for i := 0; i < len(samples)-windowSize; i += hopSize {
		windowedSamples := make([]float64, windowSize)
		for j := 0; j < windowSize; j++ {
			windowedSamples[j] = samples[i+j] * hannWindow[j]
		}
		spectrum := fft.FFTReal(windowedSamples)
		mfcc := computeMFCC(spectrum, sampleRate)
		res = append(res, mfcc)
	}

	return res, hopSize
}

const (
	numFilters      = 26
	numCoefficients = 13
)

// Compute the MFCCs from the given spectrum
func computeMFCC(spectrum []complex128, sampleRate int) []float64 {

	// Convert complex values to magnitudes
	magnitudes := make([]float64, len(spectrum)/2)
	for i := range magnitudes {
		magnitudes[i] = cmplx.Abs(spectrum[i])
	}

	// Compute the filterBank
	filterBank := createFilterBank(len(magnitudes)*2, sampleRate)

	// Apply the filterBank to the magnitudes
	filteredMagnitudes := applyFilterBank(magnitudes, filterBank)

	// Compute the MFCCs
	logEnergies := make([]float64, numFilters)
	for i := 0; i < numFilters; i++ {
		if filteredMagnitudes[i] == 0 {
			filteredMagnitudes[i] = math.SmallestNonzeroFloat64
		}
		logEnergies[i] = math.Log(filteredMagnitudes[i])
	}

	// Compute the DCT
	mfcc := make([]float64, numCoefficients)
	for i := 0; i < numCoefficients; i++ {
		sum := 0.0
		for j := 0; j < numFilters; j++ {
			sum += logEnergies[j] * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(numFilters))
		}
		mfcc[i] = sum
	}

	return mfcc
}

func createFilterBank(nfft, sampleRate int) [][]float64 {
	filterBank := make([][]float64, numFilters)
	lower := hzToMel(0)
	upper := hzToMel(float64(sampleRate) / 2)

	// Calculate the points evenly spaced in mels
	melPoints := linspace(lower, upper, numFilters+2)

	// Convert mel points back to frequency
	hzPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		hzPoints[i] = melToHz(mel)
	}

	// Convert hz points to fft bins
	binPoints := make([]int, len(hzPoints))
	for i, hz := range hzPoints {
		binPoints[i] = int(math.Round(hz * float64(nfft) / float64(sampleRate)))
	}

	for i := 0; i < numFilters; i++ {
		filterBank[i] = make([]float64, nfft/2)
		start := binPoints[i]
		mid := binPoints[i+1]
		end := binPoints[i+2]

		if start < 0 {
			start = 0
		}
		if mid > nfft/2 {
			mid = nfft / 2
		}
		if end > nfft/2 {
			end = nfft / 2
		}

		for j := start; j < mid; j++ {
			filterBank[i][j] = (float64(j) - float64(start)) / (float64(mid) - float64(start))
		}
		for j := mid; j < end; j++ {
			filterBank[i][j] = (float64(end) - float64(j)) / (float64(end) - float64(mid))
		}
	}

	return filterBank
}

func linspace(start, end float64, numPoints int) []float64 {
	if numPoints <= 1 {
		return []float64{start}
	}

	step := (end - start) / float64(numPoints-1)
	points := make([]float64, numPoints)

	for i := range points {
		points[i] = start + float64(i)*step
	}

	return points
}

func applyFilterBank(magnitudes []float64, filterBank [][]float64) []float64 {
	filtered := make([]float64, len(filterBank))

	for i, filter := range filterBank {
		sum := 0.0
		minLen := min(len(magnitudes), len(filter))
		for j := 0; j < minLen; j++ {
			sum += magnitudes[j] * filter[j]
		}
		filtered[i] = sum
	}

	return filtered
}

// https://en.wikipedia.org/wiki/Mel_scale
func hzToMel(freq float64) float64 {
	return 2_595 * math.Log10(1+freq/700)
}

func melToHz(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2_595) - 1)
}

func findOffset(mfccWhole, mfccChunk [][]float64, sampleRate, hopSize int) float64 {
	minDistance := math.MaxFloat64
	offset := 0

	for i := 0; i <= len(mfccWhole)-len(mfccChunk); i++ {
		distance := 0.0
		for j := 0; j < len(mfccChunk); j++ {
			for k := 0; k < len(mfccChunk[j]); k++ {
				diff := mfccWhole[i+j][k] - mfccChunk[j][k]
				distance += diff * diff
				if distance >= minDistance {
					break
				}
			}
		}
		if distance < minDistance {
			minDistance = distance
			offset = i
		}
	}

	return float64(offset) * float64(hopSize) / float64(sampleRate)
}
