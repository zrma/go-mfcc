package mfcc

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

const defaultDeltaWindow = 2

// ComputeDelta는 주어진 특징에 대한 1차 델타를 계산한다.
func ComputeDelta(features [][]float64, window int) ([][]float64, error) {
	if len(features) == 0 {
		return nil, errors.New("no features to compute delta")
	}
	if window <= 0 {
		return nil, fmt.Errorf("invalid delta window: %d", window)
	}
	coeffCount, err := validateFeatureMatrix(features)
	if err != nil {
		return nil, fmt.Errorf("invalid features for delta: %w", err)
	}

	denom := 0.0
	for n := 1; n <= window; n++ {
		nf := float64(n)
		denom += nf * nf
	}
	denom *= 2

	frameCount := len(features)
	delta := make([][]float64, frameCount)
	buf := make([]float64, frameCount*coeffCount)
	for i := range delta {
		delta[i] = buf[i*coeffCount : (i+1)*coeffCount]
	}

	for t := range delta {
		dst := delta[t]
		for k := range dst {
			sum := 0.0
			for n := 1; n <= window; n++ {
				prev := t - n
				if prev < 0 {
					prev = 0
				}
				next := t + n
				if next >= frameCount {
					next = frameCount - 1
				}
				sum += float64(n) * (features[next][k] - features[prev][k])
			}
			dst[k] = sum / denom
		}
	}

	return delta, nil
}

// AppendDeltas는 원본 + delta(+ delta-delta)를 결합한 특징을 반환한다.
func AppendDeltas(features [][]float64, window, order int) ([][]float64, error) {
	if len(features) == 0 {
		return nil, errors.New("no features to append deltas")
	}
	if order < 0 {
		return nil, fmt.Errorf("invalid delta order: %d", order)
	}
	if order > 2 {
		return nil, fmt.Errorf("unsupported delta order: %d", order)
	}
	coeffCount, err := validateFeatureMatrix(features)
	if err != nil {
		return nil, fmt.Errorf("invalid features for delta append: %w", err)
	}
	if order == 0 {
		return cloneFeatures(features), nil
	}
	if window <= 0 {
		return nil, fmt.Errorf("invalid delta window: %d", window)
	}

	delta1, err := ComputeDelta(features, window)
	if err != nil {
		return nil, err
	}

	var delta2 [][]float64
	if order >= 2 {
		delta2, err = ComputeDelta(delta1, window)
		if err != nil {
			return nil, err
		}
	}

	frameCount := len(features)
	totalCoeff := coeffCount * (1 + order)
	out := make([][]float64, frameCount)
	buf := make([]float64, frameCount*totalCoeff)
	for i := range out {
		out[i] = buf[i*totalCoeff : (i+1)*totalCoeff]
	}

	for i := range out {
		offset := 0
		copy(out[i][offset:offset+coeffCount], features[i][:coeffCount])
		offset += coeffCount
		copy(out[i][offset:offset+coeffCount], delta1[i])
		offset += coeffCount
		if order >= 2 {
			copy(out[i][offset:offset+coeffCount], delta2[i])
		}
	}

	return out, nil
}

// ComputeASRFeatures는 MFCC + CMVN + delta/delta-delta를 결합한 특징을 계산한다.
func ComputeASRFeatures(samples []float64, sampleRate int) ([][]float64, int, error) {
	return ComputeASRFeaturesWithConfig(samples, sampleRate, DefaultConfig(), defaultDeltaWindow)
}

// ComputeASRFeaturesWithConfig는 설정과 델타 윈도우를 지정해 ASR용 특징을 계산한다.
func ComputeASRFeaturesWithConfig(samples []float64, sampleRate int, cfg Config, deltaWindow int) ([][]float64, int, error) {
	if deltaWindow < 0 {
		return nil, 0, fmt.Errorf("invalid delta window: %d", deltaWindow)
	}
	if deltaWindow == 0 {
		deltaWindow = defaultDeltaWindow
	}

	extractor, err := NewExtractor(sampleRate, cfg)
	if err != nil {
		return nil, 0, err
	}
	mfcc, err := extractor.Calculate(samples)
	if err != nil {
		return nil, 0, err
	}
	applyCMVNWithFloor(mfcc, nil, nil, extractor.cmvnStdFloor)

	features, err := AppendDeltas(mfcc, deltaWindow, 2)
	if err != nil {
		return nil, 0, err
	}
	return features, extractor.HopSize(), nil
}

// ComputeASRFeaturesFromWav는 WAV 파일에서 ASR용 특징을 계산한다.
func ComputeASRFeaturesFromWav(path string) ([][]float64, int, int, error) {
	return ComputeASRFeaturesFromWavWithConfig(path, DefaultConfig(), defaultDeltaWindow)
}

// ComputeASRFeaturesFromWavWithConfig는 설정을 적용해 WAV 파일에서 ASR용 특징을 계산한다.
func ComputeASRFeaturesFromWavWithConfig(path string, cfg Config, deltaWindow int) ([][]float64, int, int, error) {
	samples, sampleRate, err := ReadWavMono(path)
	if err != nil {
		return nil, 0, 0, err
	}
	features, hopSize, err := ComputeASRFeaturesWithConfig(samples, sampleRate, cfg, deltaWindow)
	if err != nil {
		return nil, 0, 0, err
	}
	return features, sampleRate, hopSize, nil
}

func validateFeatureMatrix(features [][]float64) (int, error) {
	if len(features) == 0 {
		return 0, errors.New("no feature frames")
	}
	coeffCount := len(features[0])
	if coeffCount == 0 {
		return 0, errors.New("no feature coefficients")
	}
	for frameIndex, frame := range features {
		if len(frame) != coeffCount {
			return 0, fmt.Errorf(
				"inconsistent coefficient count at frame %d: got %d, want %d",
				frameIndex,
				len(frame),
				coeffCount,
			)
		}
		for coeffIndex, value := range frame {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return 0, fmt.Errorf("non-finite feature at frame %d, coeff %d", frameIndex, coeffIndex)
			}
		}
	}
	return coeffCount, nil
}

func cloneFeatures(src [][]float64) [][]float64 {
	if len(src) == 0 {
		return nil
	}
	out := make([][]float64, len(src))
	for i, frame := range src {
		out[i] = slices.Clone(frame)
	}
	return out
}
