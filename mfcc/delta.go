package mfcc

import (
	"slices"

	"github.com/pkg/errors"
)

const defaultDeltaWindow = 2

// ComputeDelta는 주어진 특징에 대한 1차 델타를 계산한다.
func ComputeDelta(features [][]float64, window int) ([][]float64, error) {
	if len(features) == 0 {
		return nil, errors.New("no features to compute delta")
	}
	if window <= 0 {
		return nil, errors.Errorf("invalid delta window: %d", window)
	}
	coeffCount := minFeatureLength(features)
	if coeffCount == 0 {
		return nil, errors.New("no coefficients to compute delta")
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
	if order <= 0 {
		return cloneFeatures(features), nil
	}
	if order > 2 {
		return nil, errors.Errorf("unsupported delta order: %d", order)
	}
	if window <= 0 {
		return nil, errors.Errorf("invalid delta window: %d", window)
	}

	coeffCount := minFeatureLength(features)
	if coeffCount == 0 {
		return nil, errors.New("no coefficients to append deltas")
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
	if deltaWindow <= 0 {
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

func minFeatureLength(features [][]float64) int {
	if len(features) == 0 {
		return 0
	}
	minLen := len(features[0])
	for _, frame := range features[1:] {
		if len(frame) < minLen {
			minLen = len(frame)
		}
	}
	return minLen
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
