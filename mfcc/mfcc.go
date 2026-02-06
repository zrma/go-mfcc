package mfcc

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"sync"
	"time"

	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
)

const (
	defaultPreEmphasis     = 0.97
	defaultEnergyFloor     = 1e-12
	defaultCMVNStdFloor    = 1e-3
	defaultNumFilters      = 26
	defaultNumCoefficients = 13
	defaultWindowDuration  = 25 * time.Millisecond
	defaultHopDuration     = 10 * time.Millisecond
	distanceStartCoeff     = 1
)

// FindOffset는 전체 WAV와 그 일부를 나타내는 WAV를 받아,
// chunk가 whole에서 시작하는 시간을 초 단위로 반환한다.
func FindOffset(wholeWavPath, chunkWavPath string) (float64, error) {
	return FindOffsetWithConfig(wholeWavPath, chunkWavPath, DefaultConfig())
}

// FindOffsetWithConfig는 지정한 설정으로 MFCC를 계산해 offset을 추정한다.
func FindOffsetWithConfig(wholeWavPath, chunkWavPath string, cfg Config) (float64, error) {
	wholeSamples, sampleRate, err := readWavFile(wholeWavPath)
	if err != nil {
		return 0, fmt.Errorf("read whole wav file failed: %w", err)
	}

	chunkSamples, chunkSampleRate, err := readWavFile(chunkWavPath)
	if err != nil {
		return 0, fmt.Errorf("read chunk wav file failed: %w", err)
	}

	if sampleRate != chunkSampleRate {
		return 0, fmt.Errorf("sample rate mismatch: whole %dHz, chunk %dHz", sampleRate, chunkSampleRate)
	}

	if len(wholeSamples) < len(chunkSamples) {
		return 0, errors.New("whole wav file is shorter than chunk wav file")
	}

	extractor, err := NewExtractor(sampleRate, cfg)
	if err != nil {
		return 0, fmt.Errorf("init MFCC extractor failed: %w", err)
	}
	if extractor.numCoefficients <= distanceStartCoeff {
		return 0, fmt.Errorf(
			"insufficient MFCC coefficients for offset search: got %d, need > %d",
			extractor.numCoefficients,
			distanceStartCoeff,
		)
	}

	mfccWhole, err := calculateMFCCForOffset("whole", extractor, wholeSamples)
	if err != nil {
		return 0, err
	}
	mfccChunk, err := calculateMFCCForOffset("chunk", extractor, chunkSamples)
	if err != nil {
		return 0, err
	}
	if coeffCount := coeffCountForMFCCs(mfccWhole, mfccChunk); coeffCount <= distanceStartCoeff {
		return 0, fmt.Errorf(
			"insufficient MFCC coefficients after extraction: got %d, need > %d",
			coeffCount,
			distanceStartCoeff,
		)
	}

	if _, _, err := normalizeForMatch(mfccWhole, mfccChunk, extractor.cmvnStdFloor); err != nil {
		return 0, fmt.Errorf("normalize MFCCs failed: %w", err)
	}

	return findOffset(mfccWhole, mfccChunk, sampleRate, extractor.hopSize), nil
}

func shortInputError(label string, windowSize, sampleRate int) error {
	windowMs := float64(windowSize) * 1000 / float64(sampleRate)
	return fmt.Errorf("%s wav data too short to compute MFCCs: need at least %d samples (about %.2fms) per frame", label, windowSize, windowMs)
}

func calculateMFCCForOffset(label string, extractor *Extractor, samples []float64) ([][]float64, error) {
	if extractor == nil {
		return nil, errors.New("MFCC extractor is nil")
	}
	if len(samples) < extractor.windowSize {
		return nil, shortInputError(label, extractor.windowSize, extractor.sampleRate)
	}
	mean, err := meanAndValidateSamples(samples)
	if err != nil {
		return nil, fmt.Errorf("invalid %s samples: %w", label, err)
	}

	mfcc := extractor.calculateWithMean(samples, mean)
	if len(mfcc) == 0 {
		return nil, fmt.Errorf("failed to compute MFCCs from %s wav", label)
	}
	if err := validateMFCC(mfcc); err != nil {
		return nil, fmt.Errorf("invalid MFCC from %s wav: %w", label, err)
	}
	return mfcc, nil
}

func frameParamsFromDurations(sampleRate int, windowDuration, hopDuration time.Duration) (windowSize, hopSize int) {
	if sampleRate <= 0 {
		return 0, 0
	}
	if windowDuration <= 0 || hopDuration <= 0 {
		return 0, 0
	}
	windowSize = int(math.Round(float64(sampleRate) * (float64(windowDuration) / float64(time.Second))))
	hopSize = int(math.Round(float64(sampleRate) * (float64(hopDuration) / float64(time.Second))))
	if windowSize <= 0 || hopSize <= 0 {
		return 0, 0
	}
	return windowSize, hopSize
}

func resolveDuration(value, fallback time.Duration, label string) (time.Duration, error) {
	if value == 0 {
		return fallback, nil
	}
	if value < 0 {
		return 0, fmt.Errorf("invalid %s duration: %s", label, value)
	}
	return value, nil
}

func resolvePositiveFloat(value, fallback float64, label string) (float64, error) {
	if value == 0 {
		value = fallback
	}
	if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 {
		return 0, fmt.Errorf("invalid %s: %f", label, value)
	}
	return value, nil
}

// Extractor는 MFCC 계산에 필요한 창/필터 구성을 재사용하는 추출기다.
// 내부 버퍼를 재사용하므로 동시 접근에는 안전하지 않다.
type Extractor struct {
	sampleRate      int
	windowSize      int
	hopSize         int
	nfft            int
	binCount        int
	powerScale      float64
	numFilters      int
	numCoefficients int
	preEmphasis     float64
	energyFloor     float64
	logEnergyFloor  float64
	cmvnStdFloor    float64
	dctMatrix       [][]float64

	hamming       []float64
	filterBank    [][]float64
	filterStarts  []int
	filterEnds    []int
	windowed      []float64
	powerSpectrum []float64
	filtered      []float64
	preprocessed  []float64
}

type mfccStaticConfig struct {
	sampleRate      int
	windowSize      int
	nfft            int
	binCount        int
	numFilters      int
	numCoefficients int
	hamming         []float64
	filterBank      [][]float64
	filterStarts    []int
	filterEnds      []int
	dctMatrix       [][]float64
}

type mfccConfig struct {
	*mfccStaticConfig
	hopSize      int
	preEmphasis  float64
	energyFloor  float64
	cmvnStdFloor float64
}

// sample rate 및 프레임/필터 구성이 같은 경우 정적 윈도우/필터 뱅크를 캐싱해 재사용한다.
var mfccStaticConfigCache sync.Map

type resolvedConfig struct {
	sampleRate      int
	windowSize      int
	hopSize         int
	numFilters      int
	numCoefficients int
	preEmphasis     float64
	energyFloor     float64
	cmvnStdFloor    float64
}

type staticConfigKey struct {
	sampleRate      int
	windowSize      int
	numFilters      int
	numCoefficients int
}

func normalizeConfig(sampleRate int, cfg Config) (resolvedConfig, error) {
	if sampleRate <= 0 {
		return resolvedConfig{}, fmt.Errorf("invalid sample rate: %dHz", sampleRate)
	}

	windowDuration, err := resolveDuration(cfg.WindowDuration, defaultWindowDuration, "window")
	if err != nil {
		return resolvedConfig{}, err
	}
	hopDuration, err := resolveDuration(cfg.HopDuration, defaultHopDuration, "hop")
	if err != nil {
		return resolvedConfig{}, err
	}

	windowSize, hopSize := frameParamsFromDurations(sampleRate, windowDuration, hopDuration)
	if windowSize <= 0 || hopSize <= 0 {
		return resolvedConfig{}, errors.New("invalid window or hop size")
	}

	numFilters := cfg.NumFilters
	if numFilters == 0 {
		numFilters = defaultNumFilters
	}
	numCoefficients := cfg.NumCoefficients
	if numCoefficients == 0 {
		numCoefficients = defaultNumCoefficients
	}
	if numFilters <= 0 {
		return resolvedConfig{}, fmt.Errorf("invalid num filters: %d", numFilters)
	}
	if numCoefficients <= 0 {
		return resolvedConfig{}, fmt.Errorf("invalid num coefficients: %d", numCoefficients)
	}
	if numCoefficients > numFilters {
		return resolvedConfig{}, fmt.Errorf("num coefficients (%d) exceed num filters (%d)", numCoefficients, numFilters)
	}

	preEmphasis := cfg.PreEmphasis
	if cfg.DisablePreEmphasis {
		preEmphasis = 0
	} else if preEmphasis == 0 {
		preEmphasis = defaultPreEmphasis
	}
	if math.IsNaN(preEmphasis) || math.IsInf(preEmphasis, 0) {
		return resolvedConfig{}, errors.New("invalid pre-emphasis")
	}
	if preEmphasis < 0 || preEmphasis > 1 {
		return resolvedConfig{}, fmt.Errorf("pre-emphasis out of range: %f", preEmphasis)
	}

	energyFloor, err := resolvePositiveFloat(cfg.EnergyFloor, defaultEnergyFloor, "energy floor")
	if err != nil {
		return resolvedConfig{}, err
	}

	cmvnStdFloor, err := resolvePositiveFloat(cfg.CMVNStdFloor, defaultCMVNStdFloor, "CMVN std floor")
	if err != nil {
		return resolvedConfig{}, err
	}

	return resolvedConfig{
		sampleRate:      sampleRate,
		windowSize:      windowSize,
		hopSize:         hopSize,
		numFilters:      numFilters,
		numCoefficients: numCoefficients,
		preEmphasis:     preEmphasis,
		energyFloor:     energyFloor,
		cmvnStdFloor:    cmvnStdFloor,
	}, nil
}

func getMFCCConfig(sampleRate int, cfg Config) (*mfccConfig, error) {
	resolved, err := normalizeConfig(sampleRate, cfg)
	if err != nil {
		return nil, err
	}

	cacheKey := staticConfigKey{
		sampleRate:      resolved.sampleRate,
		windowSize:      resolved.windowSize,
		numFilters:      resolved.numFilters,
		numCoefficients: resolved.numCoefficients,
	}
	var staticCfg *mfccStaticConfig
	if cached, ok := mfccStaticConfigCache.Load(cacheKey); ok {
		staticCfg = cached.(*mfccStaticConfig)
	} else {
		nfft := nextPow2(resolved.windowSize)
		binCount := nfft/2 + 1
		if binCount < resolved.numFilters+2 {
			// mel 필터 뱅크를 모두 배치할 만큼 주파수 해상도가 부족하다.
			return nil, fmt.Errorf("insufficient mel resolution: sample rate %dHz, bin count %d", sampleRate, binCount)
		}
		filterBank := createFilterBank(nfft, sampleRate, resolved.numFilters)
		filterStarts, filterEnds := filterRanges(filterBank)
		activeFilters := 0
		for i := range filterStarts {
			if filterEnds[i] > filterStarts[i] {
				activeFilters++
			}
		}
		if activeFilters < resolved.numFilters {
			// 멜 필터가 겹쳐서 일부가 사라지는 경우는 표준 MFCC 특성을 유지할 수 없다.
			return nil, fmt.Errorf("invalid mel filter bank: active %d < requested %d", activeFilters, resolved.numFilters)
		}

		staticCfg = &mfccStaticConfig{
			sampleRate:      sampleRate,
			windowSize:      resolved.windowSize,
			nfft:            nfft,
			binCount:        binCount,
			numFilters:      resolved.numFilters,
			numCoefficients: resolved.numCoefficients,
			hamming:         window.Hamming(resolved.windowSize),
			filterBank:      filterBank,
			filterStarts:    filterStarts,
			filterEnds:      filterEnds,
			dctMatrix:       makeDCTMatrix(resolved.numCoefficients, resolved.numFilters),
		}

		if cached, loaded := mfccStaticConfigCache.LoadOrStore(cacheKey, staticCfg); loaded {
			staticCfg = cached.(*mfccStaticConfig)
		}
	}

	return &mfccConfig{
		mfccStaticConfig: staticCfg,
		hopSize:          resolved.hopSize,
		preEmphasis:      resolved.preEmphasis,
		energyFloor:      resolved.energyFloor,
		cmvnStdFloor:     resolved.cmvnStdFloor,
	}, nil
}

func newMFCCExtractor(sampleRate int) (*Extractor, error) {
	return NewExtractor(sampleRate, DefaultConfig())
}

func (e *Extractor) calculateWithMean(samples []float64, dcMean float64) [][]float64 {
	if e == nil {
		return nil
	}

	if len(samples) < e.windowSize {
		return nil
	}

	preprocessed := e.preparePreprocessed(samples, dcMean)
	frameCount := e.frameCount(len(preprocessed))
	if frameCount == 0 {
		return nil
	}

	if e.windowSize < e.nfft {
		clear(e.windowed[e.windowSize:])
	}

	return e.computeFrames(preprocessed, frameCount)
}

func (e *Extractor) computeMFCC(spectrum []complex128, dst []float64) []float64 {
	if e == nil || len(spectrum) == 0 {
		return nil
	}

	if len(dst) < e.numCoefficients {
		dst = make([]float64, e.numCoefficients)
	} else {
		dst = dst[:e.numCoefficients]
	}

	e.computePowerSpectrum(spectrum)

	applyFilterBank(e.powerSpectrum, e.filterBank, e.filtered, e.filterStarts, e.filterEnds)

	for i := range e.filtered {
		energy := e.filtered[i]
		if energy < e.energyFloor {
			e.filtered[i] = e.logEnergyFloor
			continue
		}
		e.filtered[i] = math.Log(energy)
	}

	for i := range dst {
		sum := 0.0
		for j := range e.filtered {
			sum += e.filtered[j] * e.dctMatrix[i][j]
		}
		dst[i] = sum
	}

	return dst
}

func (e *Extractor) computePowerSpectrum(spectrum []complex128) {
	if e == nil || len(e.powerSpectrum) == 0 {
		return
	}
	if e.nfft <= 0 || len(spectrum) == 0 {
		clear(e.powerSpectrum)
		return
	}

	binCount := len(e.powerSpectrum)
	limit := min(binCount, len(spectrum))
	scale := e.powerScale
	lastBin := binCount - 1

	for i := range limit {
		v := spectrum[i]
		power := (real(v)*real(v) + imag(v)*imag(v)) * scale
		if lastBin > 0 && i != 0 && i != lastBin {
			// 실수 입력 FFT에서 절반만 쓰므로 켤레 대칭 구간을 보상한다.
			power *= 2
		}
		e.powerSpectrum[i] = power
	}

	if limit < binCount {
		clear(e.powerSpectrum[limit:])
	}
}

func (e *Extractor) preparePreprocessed(samples []float64, dcMean float64) []float64 {
	e.preprocessed = preprocessSamplesWithMeanInto(e.preprocessed, samples, e.preEmphasis, dcMean)
	return e.preprocessed
}

func (e *Extractor) frameCount(sampleCount int) int {
	if sampleCount < e.windowSize || e.hopSize <= 0 {
		return 0
	}
	return 1 + (sampleCount-e.windowSize)/e.hopSize
}

func (e *Extractor) computeFrames(preprocessed []float64, frameCount int) [][]float64 {
	res := make([][]float64, frameCount)
	coeffs := make([]float64, frameCount*e.numCoefficients)

	for frame := range frameCount {
		start := frame * e.hopSize
		e.applyWindow(preprocessed[start : start+e.windowSize])
		spectrum := fft.FFTReal(e.windowed)
		dst := coeffs[frame*e.numCoefficients : (frame+1)*e.numCoefficients]
		res[frame] = e.computeMFCC(spectrum, dst)
	}

	return res
}

func (e *Extractor) applyWindow(samples []float64) {
	for i, sample := range samples {
		e.windowed[i] = sample * e.hamming[i]
	}
}

func makeDCTMatrix(numCoefficients, numFilters int) [][]float64 {
	if numCoefficients <= 0 || numFilters <= 0 {
		return nil
	}
	m := make([][]float64, numCoefficients)
	for i := range m {
		scale := math.Sqrt(2.0 / float64(numFilters))
		if i == 0 {
			scale = math.Sqrt(1.0 / float64(numFilters))
		}
		row := make([]float64, numFilters)
		for j := range row {
			row[j] = scale * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(numFilters))
		}
		m[i] = row
	}
	return m
}

func createFilterBank(nfft, sampleRate, numFilters int) [][]float64 {
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

	binCount := nfft/2 + 1

	// Convert hz points to fft bins
	binPoints := make([]int, len(hzPoints))
	for i, hz := range hzPoints {
		bin := max(int(math.Floor((float64(nfft)+1)*hz/float64(sampleRate))), 0)
		if bin >= binCount {
			bin = binCount - 1
		}
		binPoints[i] = bin
	}

	for i := range filterBank {
		filterBank[i] = make([]float64, binCount)
		start := binPoints[i]
		mid := binPoints[i+1]
		end := binPoints[i+2]

		if start < 0 {
			start = 0
		}
		if mid >= binCount {
			mid = binCount - 1
		}
		if end >= binCount {
			end = binCount - 1
		}
		if start == mid || mid == end {
			continue
		}

		riseDenom := float64(mid - start)
		if riseDenom > 0 {
			for j := start; j < mid; j++ {
				filterBank[i][j] = (float64(j) - float64(start)) / riseDenom
			}
		}
		fallDenom := float64(end - mid)
		if fallDenom > 0 {
			for j := mid; j < end; j++ {
				filterBank[i][j] = (float64(end) - float64(j)) / fallDenom
			}
		}

		// Slaney-style area normalization
		bandWidth := hzPoints[i+2] - hzPoints[i]
		if bandWidth > 0 {
			scale := 2.0 / bandWidth
			for j := range filterBank[i] {
				filterBank[i][j] *= scale
			}
		}
	}

	return filterBank
}

// 각 필터의 유효 구간만 따로 잡아서 불필요한 곱셈을 줄인다.
func filterRanges(filterBank [][]float64) ([]int, []int) {
	starts := make([]int, len(filterBank))
	ends := make([]int, len(filterBank))

	for i, filter := range filterBank {
		start := len(filter)
		end := 0
		for j, v := range filter {
			if v != 0 {
				if j < start {
					start = j
				}
				end = j + 1
			}
		}
		if end == 0 {
			start = 0
		}
		starts[i] = start
		ends[i] = end
	}

	return starts, ends
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

func applyFilterBank(powers []float64, filterBank [][]float64, filtered []float64, starts, ends []int) {
	useRanges := len(starts) == len(filterBank) && len(ends) == len(filterBank)

	for i, filter := range filterBank {
		start := 0
		end := len(filter)
		if useRanges {
			start = starts[i]
			end = ends[i]
		}
		limit := min(len(powers), end)
		if start >= limit {
			filtered[i] = 0
			continue
		}
		sum := 0.0
		for j := start; j < limit; j++ {
			sum += powers[j] * filter[j]
		}
		filtered[i] = sum
	}
}

func cmnMean(mfcc [][]float64) []float64 {
	if len(mfcc) == 0 {
		return nil
	}
	coeffCount := coeffCountForMFCCs(mfcc)
	if coeffCount == 0 {
		return []float64{}
	}

	means := make([]float64, coeffCount)
	for _, frame := range mfcc {
		for i := range means {
			means[i] += frame[i]
		}
	}
	for i := range means {
		means[i] /= float64(len(mfcc))
	}

	return means
}

func cmnMeanStd(mfcc [][]float64) (mean []float64, std []float64) {
	return cmnMeanStdWithFloor(mfcc, defaultCMVNStdFloor)
}

func cmnMeanStdWithFloor(mfcc [][]float64, stdFloor float64) (mean []float64, std []float64) {
	return cmnMeanStdWithFloorForAll(stdFloor, mfcc)
}

func cmnStdWithMean(mfcc [][]float64, mean []float64, stdFloor float64) []float64 {
	if len(mfcc) == 0 || len(mean) == 0 {
		return nil
	}

	coeffCount := min(len(mean), coeffCountForMFCCs(mfcc))
	if coeffCount == 0 {
		return nil
	}

	std := make([]float64, coeffCount)
	count := 0.0

	for _, frame := range mfcc {
		limit := min(coeffCount, len(frame))
		count++
		for i := range std[:limit] {
			diff := frame[i] - mean[i]
			std[i] += diff * diff
		}
	}
	if count == 0 {
		return std
	}

	for i := range std {
		variance := std[i] / count
		if variance < 0 {
			variance = 0
		}
		stdVal := math.Sqrt(variance)
		if stdVal < stdFloor {
			stdVal = stdFloor
		}
		std[i] = stdVal
	}

	return std
}

func cmnMeanStdWithFloorForAll(stdFloor float64, all ...[][]float64) (mean []float64, std []float64) {
	coeffCount := coeffCountForMFCCs(all...)
	if coeffCount == 0 {
		return nil, nil
	}

	mean = make([]float64, coeffCount)
	m2 := make([]float64, coeffCount)
	count := 0.0

	for _, mfcc := range all {
		for _, frame := range mfcc {
			limit := min(coeffCount, len(frame))
			count++
			for i := range mean[:limit] {
				v := frame[i]
				delta := v - mean[i]
				mean[i] += delta / count
				m2[i] += delta * (v - mean[i])
			}
		}
	}
	if count == 0 {
		return nil, nil
	}

	std = make([]float64, coeffCount)
	for i := range std {
		variance := m2[i] / count
		if variance < 0 {
			variance = 0
		}
		stdVal := math.Sqrt(variance)
		if stdVal < stdFloor {
			stdVal = stdFloor
		}
		std[i] = stdVal
	}

	return mean, std
}

func coeffCountForMFCCs(all ...[][]float64) int {
	coeffCount := 0
	for _, mfcc := range all {
		if len(mfcc) == 0 {
			continue
		}
		if coeffCount == 0 || len(mfcc[0]) < coeffCount {
			coeffCount = len(mfcc[0])
		}
		for _, frame := range mfcc[1:] {
			if len(frame) < coeffCount {
				coeffCount = len(frame)
			}
		}
	}
	return coeffCount
}

func applyCMVN(mfcc [][]float64, mean, std []float64) ([]float64, []float64) {
	return applyCMVNWithFloor(mfcc, mean, std, defaultCMVNStdFloor)
}

func applyCMVNWithFloor(mfcc [][]float64, mean, std []float64, stdFloor float64) ([]float64, []float64) {
	if len(mfcc) == 0 {
		return mean, std
	}
	if len(mean) == 0 && len(std) == 0 {
		mean, std = cmnMeanStdWithFloor(mfcc, stdFloor)
	} else {
		if len(mean) == 0 {
			mean = cmnMean(mfcc)
		}
		if len(std) == 0 {
			std = cmnStdWithMean(mfcc, mean, stdFloor)
		}
	}
	if len(mean) == 0 || len(std) == 0 {
		return mean, std
	}

	std = applyCMVNStats(mfcc, mean, std, stdFloor)
	return mean, std
}

func applyCMVNStats(mfcc [][]float64, mean, std []float64, stdFloor float64) []float64 {
	if len(mfcc) == 0 || len(mean) == 0 || len(std) == 0 {
		return std
	}
	std = applyStdFloor(std, stdFloor)
	coeffCount := min(len(mean), len(std))
	for _, frame := range mfcc {
		limit := min(coeffCount, len(frame))
		for i := range frame[:limit] {
			frame[i] -= mean[i]
			frame[i] /= std[i]
		}
	}
	return std
}

func applyStdFloor(std []float64, stdFloor float64) []float64 {
	if len(std) == 0 || stdFloor <= 0 {
		return std
	}
	for _, v := range std {
		if v < stdFloor {
			out := slices.Clone(std)
			for i, s := range out {
				if s < stdFloor {
					out[i] = stdFloor
				}
			}
			return out
		}
	}
	return std
}

// normalizeForMatch는 whole에서 계산한 CMVN 통계만을 사용해 두 입력을 정규화한다.
func normalizeForMatch(mfccWhole, mfccChunk [][]float64, stdFloor float64) ([]float64, []float64, error) {
	if len(mfccWhole) == 0 || len(mfccChunk) == 0 {
		return nil, nil, errors.New("no MFCC frames to normalize")
	}
	if stdFloor <= 0 {
		stdFloor = defaultCMVNStdFloor
	}
	mean, std := cmnMeanStdWithFloor(mfccWhole, stdFloor)
	if len(mean) == 0 || len(std) == 0 {
		return nil, nil, errors.New("failed to compute CMVN stats")
	}
	std = applyCMVNStats(mfccWhole, mean, std, stdFloor)
	applyCMVNStats(mfccChunk, mean, std, stdFloor)
	coeffCount := coeffCountForMFCCs(mfccWhole, mfccChunk)
	l2NormalizeFramesWithLimit(mfccWhole, distanceStartCoeff, coeffCount)
	l2NormalizeFramesWithLimit(mfccChunk, distanceStartCoeff, coeffCount)
	return mean, std, nil
}

func validateMFCC(mfcc [][]float64) error {
	for i, frame := range mfcc {
		for j, v := range frame {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("invalid MFCC value at frame %d, coeff %d", i, j)
			}
		}
	}
	return nil
}

func meanAndValidateSamples(samples []float64) (float64, error) {
	if len(samples) == 0 {
		return 0, nil
	}
	sum := 0.0
	for i, v := range samples {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("invalid sample value at index %d", i)
		}
		sum += v
	}
	return sum / float64(len(samples)), nil
}

func l2NormalizeFrames(mfcc [][]float64, startCoeff int) {
	l2NormalizeFramesWithLimit(mfcc, startCoeff, 0)
}

func l2NormalizeFramesWithLimit(mfcc [][]float64, startCoeff, coeffLimit int) {
	if coeffLimit <= 0 {
		coeffLimit = coeffCountForMFCCs(mfcc)
	}
	for _, frame := range mfcc {
		if len(frame) <= startCoeff {
			continue
		}
		limit := len(frame)
		if coeffLimit > 0 && coeffLimit < limit {
			limit = coeffLimit
		}
		if limit <= startCoeff {
			continue
		}
		sum := 0.0
		for i := startCoeff; i < limit; i++ {
			sum += frame[i] * frame[i]
		}
		if sum <= defaultEnergyFloor {
			continue
		}
		scale := 1.0 / math.Sqrt(sum)
		for i := startCoeff; i < limit; i++ {
			frame[i] *= scale
		}
	}
}

const (
	htkMelScale = 2595.0
	htkMelHzRef = 700.0
)

// https://en.wikipedia.org/wiki/Mel_scale
func hzToMel(freq float64) float64 {
	return htkMelScale * math.Log10(1+freq/htkMelHzRef)
}

func melToHz(mel float64) float64 {
	return htkMelHzRef * (math.Pow(10, mel/htkMelScale) - 1)
}

func meanValue(samples []float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	sum := 0.0
	for _, s := range samples {
		sum += s
	}
	return sum / float64(len(samples))
}

func preprocessSamplesWithMeanInto(dst, samples []float64, preEmphasis, mean float64) []float64 {
	if len(samples) == 0 {
		return dst[:0]
	}

	if math.IsNaN(mean) || math.IsInf(mean, 0) {
		mean = meanValue(samples)
	}

	if cap(dst) < len(samples) {
		dst = make([]float64, len(samples))
	}
	dst = dst[:len(samples)]

	prev := samples[0] - mean
	dst[0] = prev

	if preEmphasis <= 0 {
		for i := 1; i < len(samples); i++ {
			dst[i] = samples[i] - mean
		}
		return dst
	}

	for i := 1; i < len(samples); i++ {
		cur := samples[i] - mean
		dst[i] = cur - preEmphasis*prev
		prev = cur
	}
	return dst
}

func nextPow2(n int) int {
	if n <= 0 {
		return 0
	}
	if n&(n-1) == 0 {
		return n
	}
	return 1 << bits.Len(uint(n))
}

func findOffset(mfccWhole, mfccChunk [][]float64, sampleRate, hopSize int) float64 {
	if len(mfccChunk) == 0 || len(mfccWhole) < len(mfccChunk) {
		return 0
	}

	coeffCount := coeffCountForMFCCs(mfccChunk, mfccWhole)
	if coeffCount == 0 {
		return 0
	}

	minDistance := math.MaxFloat64
	offset := 0

	// C0(0번째 켑스트럼)은 전체 에너지 변화(녹음 게인 등)에 가장 민감하므로
	// 오프셋 검색에서는 제외해 gain 차이에 더 강인하게 만든다.
	startCoeff := distanceStartCoeff
	if coeffCount <= startCoeff {
		return 0
	}

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

	return float64(offset) * float64(hopSize) / float64(sampleRate)
}
