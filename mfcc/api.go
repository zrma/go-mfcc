package mfcc

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// Config는 MFCC 계산에 사용할 기본 설정값을 담는다.
// 명시하지 않은 값은 DefaultConfig의 기본값으로 채워진다.
type Config struct {
	WindowDuration     time.Duration
	HopDuration        time.Duration
	NumFilters         int
	NumCoefficients    int
	PreEmphasis        float64
	DisablePreEmphasis bool
	EnergyFloor        float64
	CMVNStdFloor       float64
}

// DefaultConfig는 MFCC 계산에 사용되는 기본 설정을 반환한다.
func DefaultConfig() Config {
	return Config{
		WindowDuration:  defaultWindowDuration,
		HopDuration:     defaultHopDuration,
		NumFilters:      defaultNumFilters,
		NumCoefficients: defaultNumCoefficients,
		PreEmphasis:     defaultPreEmphasis,
		EnergyFloor:     defaultEnergyFloor,
		CMVNStdFloor:    defaultCMVNStdFloor,
	}
}

// NewExtractor는 재사용 가능한 MFCC 추출기를 생성한다.
func NewExtractor(sampleRate int, cfg Config) (*Extractor, error) {
	resolved, err := getMFCCConfig(sampleRate, cfg)
	if err != nil {
		return nil, err
	}

	powerScale := 0.0
	if resolved.nfft > 0 {
		powerScale = 1.0 / float64(resolved.nfft)
	}

	return &Extractor{
		sampleRate:      resolved.sampleRate,
		windowSize:      resolved.windowSize,
		hopSize:         resolved.hopSize,
		nfft:            resolved.nfft,
		binCount:        resolved.binCount,
		powerScale:      powerScale,
		numFilters:      resolved.numFilters,
		numCoefficients: resolved.numCoefficients,
		preEmphasis:     resolved.preEmphasis,
		energyFloor:     resolved.energyFloor,
		logEnergyFloor:  math.Log(resolved.energyFloor),
		cmvnStdFloor:    resolved.cmvnStdFloor,
		dctMatrix:       resolved.dctMatrix,
		hamming:         resolved.hamming,
		filterBank:      resolved.filterBank,
		filterStarts:    resolved.filterStarts,
		filterEnds:      resolved.filterEnds,
		windowed:        make([]float64, resolved.nfft),
		powerSpectrum:   make([]float64, resolved.binCount),
		filtered:        make([]float64, resolved.numFilters),
	}, nil
}

// SampleRate는 Extractor가 참조하는 샘플레이트를 반환한다.
func (e *Extractor) SampleRate() int {
	if e == nil {
		return 0
	}
	return e.sampleRate
}

// WindowSize는 분석 창의 길이를 샘플 단위로 반환한다.
func (e *Extractor) WindowSize() int {
	if e == nil {
		return 0
	}
	return e.windowSize
}

// HopSize는 프레임 홉 길이를 샘플 단위로 반환한다.
func (e *Extractor) HopSize() int {
	if e == nil {
		return 0
	}
	return e.hopSize
}

// NumFilters는 멜 필터 개수를 반환한다.
func (e *Extractor) NumFilters() int {
	if e == nil {
		return 0
	}
	return e.numFilters
}

// NumCoefficients는 MFCC 계수 개수를 반환한다.
func (e *Extractor) NumCoefficients() int {
	if e == nil {
		return 0
	}
	return e.numCoefficients
}

// Calculate는 입력 샘플에서 MFCC를 계산한다.
func (e *Extractor) Calculate(samples []float64) ([][]float64, error) {
	if e == nil {
		return nil, errors.New("extractor is nil")
	}
	if len(samples) < e.windowSize {
		return nil, fmt.Errorf("input too short to compute MFCCs: need at least %d samples", e.windowSize)
	}
	mean, err := meanAndValidateSamples(samples)
	if err != nil {
		return nil, err
	}
	mfcc := e.calculateWithMean(samples, mean)
	if len(mfcc) == 0 {
		return nil, errors.New("failed to compute MFCCs")
	}
	if err := validateMFCC(mfcc); err != nil {
		return nil, fmt.Errorf("invalid MFCC output: %w", err)
	}
	return mfcc, nil
}

// ReadWavMono는 WAV 파일을 읽어 모노로 변환한 샘플과 샘플레이트를 반환한다.
func ReadWavMono(path string) ([]float64, int, error) {
	return readWavFile(path)
}

// ComputeMFCC는 기본 설정으로 MFCC를 계산한다.
func ComputeMFCC(samples []float64, sampleRate int) ([][]float64, int, error) {
	return ComputeMFCCWithConfig(samples, sampleRate, DefaultConfig())
}

// ComputeMFCCWithConfig는 지정한 설정으로 MFCC를 계산한다.
func ComputeMFCCWithConfig(samples []float64, sampleRate int, cfg Config) ([][]float64, int, error) {
	extractor, err := NewExtractor(sampleRate, cfg)
	if err != nil {
		return nil, 0, err
	}
	mfcc, err := extractor.Calculate(samples)
	if err != nil {
		return nil, 0, err
	}
	return mfcc, extractor.HopSize(), nil
}

// ComputeMFCCFromWav는 WAV 파일에서 MFCC를 계산한다.
func ComputeMFCCFromWav(path string) ([][]float64, int, int, error) {
	return ComputeMFCCFromWavWithConfig(path, DefaultConfig())
}

// ComputeMFCCFromWavWithConfig는 설정을 적용해 WAV 파일에서 MFCC를 계산한다.
func ComputeMFCCFromWavWithConfig(path string, cfg Config) ([][]float64, int, int, error) {
	samples, sampleRate, err := ReadWavMono(path)
	if err != nil {
		return nil, 0, 0, err
	}
	mfcc, hopSize, err := ComputeMFCCWithConfig(samples, sampleRate, cfg)
	if err != nil {
		return nil, 0, 0, err
	}
	return mfcc, sampleRate, hopSize, nil
}

// ComputeCMVN는 CMVN 통계를 계산한다. stdFloor가 0 이하이면 기본값을 사용한다.
func ComputeCMVN(mfcc [][]float64, stdFloor float64) ([]float64, []float64) {
	if stdFloor <= 0 {
		stdFloor = defaultCMVNStdFloor
	}
	return cmnMeanStdWithFloor(mfcc, stdFloor)
}

// ApplyCMVN은 CMVN을 적용한다. stdFloor가 0 이하이면 기본값을 사용한다.
func ApplyCMVN(mfcc [][]float64, mean, std []float64, stdFloor float64) ([]float64, []float64) {
	if stdFloor <= 0 {
		stdFloor = defaultCMVNStdFloor
	}
	return applyCMVNWithFloor(mfcc, mean, std, stdFloor)
}
