package mfcc

import (
	"encoding/binary"
	"io"
	"math"
	"math/bits"
	"os"
	"sync"

	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
	"github.com/pkg/errors"
	"go.uber.org/multierr"
)

const (
	preEmphasis        = 0.97
	energyFloor        = 1e-12
	cmvnStdFloor       = 1e-3
	distanceStartCoeff = 1

	wavFormatPCM        = 1
	wavFormatIEEEFloat  = 3
	wavFormatExtensible = 0xFFFE
)

var (
	wavSubFormatPCM       = [16]byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}
	wavSubFormatIEEEFloat = [16]byte{0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}
)

// FindOffset는 전체 WAV와 그 일부를 나타내는 WAV를 받아,
// chunk가 whole에서 시작하는 시간을 초 단위로 반환한다.
func FindOffset(wholeWavPath, chunkWavPath string) (float64, error) {
	wholeSamples, sampleRate, err := readWavFile(wholeWavPath)
	if err != nil {
		return 0, errors.Wrap(err, "read whole wav file failed")
	}
	if sampleRate <= 0 {
		return 0, errors.Errorf("invalid sample rate: %dHz", sampleRate)
	}

	chunkSamples, chunkSampleRate, err := readWavFile(chunkWavPath)
	if err != nil {
		return 0, errors.Wrap(err, "read chunk wav file failed")
	}
	if chunkSampleRate <= 0 {
		return 0, errors.Errorf("invalid sample rate: %dHz", chunkSampleRate)
	}

	if sampleRate != chunkSampleRate {
		return 0, errors.Errorf("sample rate mismatch: whole %dHz, chunk %dHz", sampleRate, chunkSampleRate)
	}

	if len(wholeSamples) < len(chunkSamples) {
		return 0, errors.New("whole wav file is shorter than chunk wav file")
	}

	extractor := newMFCCExtractor(sampleRate)
	if extractor == nil {
		return 0, errors.Errorf("sample rate too low or insufficient mel resolution to compute MFCCs: %dHz", sampleRate)
	}

	dcMean := meanValue(wholeSamples)
	mfccWhole := extractor.calculateWithMean(wholeSamples, dcMean)
	mfccChunk := extractor.calculateWithMean(chunkSamples, dcMean)
	if len(mfccWhole) == 0 || len(mfccChunk) == 0 {
		return 0, errors.Errorf("wav data too short to compute MFCCs: need at least %d samples (about 25ms) per frame", extractor.windowSize)
	}

	mean, std := normalizeForMatch(mfccWhole, mfccChunk)
	if len(mean) == 0 || len(std) == 0 {
		return 0, errors.New("failed to normalize MFCCs")
	}
	if err := validateMFCC(mfccWhole); err != nil {
		return 0, errors.Wrap(err, "invalid MFCC from whole wav")
	}
	if err := validateMFCC(mfccChunk); err != nil {
		return 0, errors.Wrap(err, "invalid MFCC from chunk wav")
	}

	return findOffset(mfccWhole, mfccChunk, sampleRate, extractor.hopSize), nil
}

type wavFormatInfo struct {
	audioFormat        uint16
	validBitsPerSample uint16
	subFormat          [16]byte
}

func (i wavFormatInfo) isIEEEFloat() bool {
	if i.audioFormat == wavFormatIEEEFloat {
		return true
	}
	return i.audioFormat == wavFormatExtensible && i.subFormat == wavSubFormatIEEEFloat
}

func (i wavFormatInfo) isPCM() bool {
	if i.audioFormat == wavFormatPCM {
		return true
	}
	return i.audioFormat == wavFormatExtensible && i.subFormat == wavSubFormatPCM
}

func probeWavFormat(r io.ReadSeeker) (wavFormatInfo, error) {
	var info wavFormatInfo
	if r == nil {
		return info, errors.New("wav reader is nil")
	}

	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return info, errors.Wrap(err, "rewind wav reader failed")
	}

	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return info, errors.Wrap(err, "read RIFF header failed")
	}
	if string(header[:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return info, errors.New("not a RIFF/WAVE file")
	}

	for {
		var chunkID [4]byte
		if _, err := io.ReadFull(r, chunkID[:]); err != nil {
			return info, errors.Wrap(err, "read chunk ID failed")
		}

		var chunkSize uint32
		if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
			return info, errors.Wrap(err, "read chunk size failed")
		}
		if chunkSize == 0 {
			return info, errors.New("invalid chunk with zero size before fmt chunk")
		}

		if string(chunkID[:]) == "fmt " {
			if chunkSize > 1<<10 {
				return info, errors.Errorf("fmt chunk too large: %d bytes", chunkSize)
			}

			buf := make([]byte, chunkSize)
			if _, err := io.ReadFull(r, buf); err != nil {
				return info, errors.Wrap(err, "read fmt chunk failed")
			}
			if len(buf) < 16 {
				return info, errors.New("fmt chunk too short")
			}

			info.audioFormat = binary.LittleEndian.Uint16(buf[0:2])
			if info.audioFormat == wavFormatExtensible {
				extraSize := binary.LittleEndian.Uint16(buf[16:18])
				// WAVE_FORMAT_EXTENSIBLE stores valid bits (2 bytes), channel mask (4 bytes), and sub-format GUID (16 bytes)
				if extraSize >= 2 && len(buf) >= 20 {
					info.validBitsPerSample = binary.LittleEndian.Uint16(buf[18:20])
				}
				if extraSize >= 22 && len(buf) >= 40 {
					copy(info.subFormat[:], buf[24:40])
				}
			}

			return info, nil
		}

		skip := int64(chunkSize)
		if chunkSize%2 == 1 {
			skip++
		}

		if _, err := r.Seek(skip, io.SeekCurrent); err != nil {
			return info, errors.Wrap(err, "skip non-fmt chunk failed")
		}
	}
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

	formatInfo, err := probeWavFormat(file0)
	if err != nil {
		return nil, 0, errors.Wrap(err, "parse wav format failed")
	}
	if _, err := file0.Seek(0, io.SeekStart); err != nil {
		return nil, 0, errors.Wrap(err, "rewind wav file failed")
	}

	decoder := wav.NewDecoder(file0)
	if err := decoder.FwdToPCM(); err != nil {
		return nil, 0, errors.Wrap(err, "decode wav header failed")
	}

	channels := int(decoder.NumChans)
	if channels <= 0 {
		channels = 1
	}
	sampleRate = int(decoder.SampleRate)

	audioFormat := formatInfo.audioFormat
	if audioFormat == 0 {
		audioFormat = decoder.WavAudioFormat
	}
	isFloat := formatInfo.isIEEEFloat() || audioFormat == wavFormatIEEEFloat
	isPCM := formatInfo.isPCM() || audioFormat == wavFormatPCM

	switch {
	case isFloat:
		data, err = decodeFloatPCM(decoder, channels)
		if err != nil {
			return nil, 0, errors.Wrap(err, "decode float wav data failed")
		}
		return data, sampleRate, nil
	case isPCM:
	default:
		return nil, 0, errors.Errorf("unsupported wav format: code=%d subformat=%x", audioFormat, formatInfo.subFormat)
	}

	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, 0, errors.Wrap(err, "decode wav file failed")
	}

	if buf == nil || buf.Format == nil {
		return nil, 0, errors.New("invalid PCM buffer or format")
	}

	if rem := len(buf.Data) % channels; rem != 0 {
		return nil, 0, errors.Errorf("wav data length (%d samples) is not divisible by channel count (%d)", len(buf.Data), channels)
	}
	frames := len(buf.Data) / channels

	bitDepth := buf.SourceBitDepth
	if bitDepth <= 0 {
		bitDepth = int(decoder.BitDepth)
	}
	if bitDepth <= 0 {
		return nil, 0, errors.New("unknown source bit depth")
	}
	if vb := int(formatInfo.validBitsPerSample); vb > bitDepth {
		return nil, 0, errors.Errorf("valid bits per sample (%d) exceed bit depth (%d)", vb, bitDepth)
	}
	// WAVE_FORMAT_EXTENSIBLE PCM은 유효 비트를 상위 비트에 맞춰 저장하므로 컨테이너 비트수를 기준으로 정규화한다.
	normalizer := math.Pow(2, float64(bitDepth)-1)
	offset := 0.0
	if bitDepth == 8 {
		// 8비트 PCM은 unsigned 데이터이므로 중앙값(0x80)을 제거해 0 기준으로 맞춘다.
		offset = normalizer
	}

	data = make([]float64, frames)
	for i := range frames {
		sum := 0.0
		for c := 0; c < channels; c++ {
			sample := float64(buf.Data[i*channels+c]) - offset
			sum += sample / normalizer
		}
		data[i] = sum / float64(channels)
	}

	return data, sampleRate, nil
}

func decodeFloatPCM(decoder *wav.Decoder, channels int) ([]float64, error) {
	if decoder == nil || decoder.PCMChunk == nil {
		return nil, errors.New("PCM chunk not found")
	}

	bitDepth := int(decoder.BitDepth)
	if bitDepth <= 0 {
		return nil, errors.New("unknown source bit depth")
	}
	bytesPerSample := (bitDepth-1)/8 + 1
	bytesPerFrame := bytesPerSample * channels
	if bytesPerFrame <= 0 {
		return nil, errors.New("invalid frame size")
	}

	pcmBytes := make([]byte, decoder.PCMSize)
	if _, err := io.ReadFull(decoder.PCMChunk, pcmBytes); err != nil {
		return nil, errors.Wrap(err, "read PCM chunk failed")
	}

	if rem := len(pcmBytes) % bytesPerFrame; rem != 0 {
		return nil, errors.Errorf("wav data length (%d bytes) is not divisible by frame size (%d)", len(pcmBytes), bytesPerFrame)
	}
	frames := len(pcmBytes) / bytesPerFrame

	data := make([]float64, frames)
	for i := 0; i < frames; i++ {
		sum := 0.0
		frameOffset := i * bytesPerFrame
		for c := 0; c < channels; c++ {
			start := frameOffset + c*bytesPerSample
			sampleBytes := pcmBytes[start : start+bytesPerSample]

			var sample float64
			switch bytesPerSample {
			case 4:
				sample = float64(math.Float32frombits(binary.LittleEndian.Uint32(sampleBytes)))
			case 8:
				sample = math.Float64frombits(binary.LittleEndian.Uint64(sampleBytes))
			default:
				return nil, errors.Errorf("unsupported float bit depth: %d", bitDepth)
			}
			if math.IsNaN(sample) || math.IsInf(sample, 0) {
				return nil, errors.Errorf("invalid float PCM sample at frame %d, channel %d", i, c)
			}
			sum += sample
		}
		data[i] = sum / float64(channels)
	}

	return data, nil
}

func frameParams(sampleRate int) (windowSize, hopSize int) {
	if sampleRate <= 0 {
		return 0, 0
	}
	windowSize = int(math.Round(float64(sampleRate) * 0.025)) // 25ms
	hopSize = int(math.Round(float64(sampleRate) * 0.010))    // 10ms
	if windowSize <= 0 || hopSize <= 0 {
		return 0, 0
	}
	return windowSize, hopSize
}

// 두 wav 모두에서 동일한 창/필터 구성과 스크래치 버퍼를 재사용한다.
type mfccExtractor struct {
	windowSize int
	hopSize    int
	nfft       int
	binCount   int

	hamming       []float64
	filterBank    [][]float64
	filterStarts  []int
	filterEnds    []int
	windowed      []float64
	powerSpectrum []float64
	filtered      []float64
	preprocessed  []float64
}

type mfccConfig struct {
	windowSize   int
	hopSize      int
	nfft         int
	binCount     int
	hamming      []float64
	filterBank   [][]float64
	filterStarts []int
	filterEnds   []int
}

// sample rate마다 반복 계산이 발생하는 윈도우/필터 뱅크 구성을 캐싱해 재사용한다.
var mfccConfigCache sync.Map

func getMFCCConfig(sampleRate int) *mfccConfig {
	windowSize, hopSize := frameParams(sampleRate)
	if windowSize <= 0 || hopSize <= 0 {
		return nil
	}

	if cached, ok := mfccConfigCache.Load(sampleRate); ok {
		return cached.(*mfccConfig)
	}

	nfft := nextPow2(windowSize)
	binCount := nfft/2 + 1
	if binCount < numFilters+2 {
		// mel 필터 뱅크를 26개 모두 배치할 만큼 주파수 해상도가 부족하다.
		return nil
	}
	filterBank := createFilterBank(nfft, sampleRate)
	filterStarts, filterEnds := filterRanges(filterBank)
	activeFilters := 0
	for i := range filterStarts {
		if filterEnds[i] > filterStarts[i] {
			activeFilters++
		}
	}
	if activeFilters < numFilters {
		// 멜 필터가 겹쳐서 일부가 사라지는 경우는 표준 MFCC 특성을 유지할 수 없다.
		return nil
	}

	cfg := &mfccConfig{
		windowSize:   windowSize,
		hopSize:      hopSize,
		nfft:         nfft,
		binCount:     binCount,
		hamming:      window.Hamming(windowSize),
		filterBank:   filterBank,
		filterStarts: filterStarts,
		filterEnds:   filterEnds,
	}

	if cached, loaded := mfccConfigCache.LoadOrStore(sampleRate, cfg); loaded {
		return cached.(*mfccConfig)
	}

	return cfg
}

func newMFCCExtractor(sampleRate int) *mfccExtractor {
	cfg := getMFCCConfig(sampleRate)
	if cfg == nil {
		return nil
	}

	return &mfccExtractor{
		windowSize:    cfg.windowSize,
		hopSize:       cfg.hopSize,
		nfft:          cfg.nfft,
		binCount:      cfg.binCount,
		hamming:       cfg.hamming,
		filterBank:    cfg.filterBank,
		filterStarts:  cfg.filterStarts,
		filterEnds:    cfg.filterEnds,
		windowed:      make([]float64, cfg.nfft),
		powerSpectrum: make([]float64, cfg.binCount),
		filtered:      make([]float64, numFilters),
	}
}

func calculateMFCC(samples []float64, sampleRate int) ([][]float64, int) {
	extractor := newMFCCExtractor(sampleRate)
	if extractor == nil {
		return nil, 0
	}

	return extractor.calculate(samples), extractor.hopSize
}

const (
	numFilters      = 26
	numCoefficients = 13
)

func (e *mfccExtractor) calculate(samples []float64) [][]float64 {
	return e.calculateWithMean(samples, math.NaN())
}

func (e *mfccExtractor) calculateWithMean(samples []float64, dcMean float64) [][]float64 {
	if e == nil {
		return nil
	}

	if len(samples) < e.windowSize {
		return nil
	}

	preprocessed := preprocessSamplesWithMeanInto(e.preprocessed, samples, preEmphasis, dcMean)
	e.preprocessed = preprocessed
	frameCount := 1 + (len(preprocessed)-e.windowSize)/e.hopSize
	res := make([][]float64, frameCount)
	coeffs := make([]float64, frameCount*numCoefficients)

	if e.windowSize < e.nfft {
		clear(e.windowed[e.windowSize:])
	}

	for frame := range frameCount {
		start := frame * e.hopSize
		for j := 0; j < e.windowSize; j++ {
			e.windowed[j] = preprocessed[start+j] * e.hamming[j]
		}
		spectrum := fft.FFTReal(e.windowed)
		dst := coeffs[frame*numCoefficients : (frame+1)*numCoefficients]
		res[frame] = e.computeMFCC(spectrum, dst)
	}

	return res
}

func (e *mfccExtractor) computeMFCC(spectrum []complex128, dst []float64) []float64 {
	if e == nil || len(spectrum) == 0 {
		return nil
	}

	if len(dst) < numCoefficients {
		dst = make([]float64, numCoefficients)
	} else {
		dst = dst[:numCoefficients]
	}

	scale := 1.0 / float64(e.nfft)
	lastBin := e.binCount - 1
	for i := 0; i < e.binCount && i < len(spectrum); i++ {
		v := spectrum[i]
		power := (real(v)*real(v) + imag(v)*imag(v)) * scale
		if lastBin > 0 && i != 0 && i != lastBin {
			// 실수 입력 FFT에서 절반만 쓰므로 켤레 대칭 구간을 보상한다.
			power *= 2
		}
		e.powerSpectrum[i] = power
	}

	applyFilterBank(e.powerSpectrum, e.filterBank, e.filtered, e.filterStarts, e.filterEnds)

	for i := range e.filtered {
		energy := e.filtered[i]
		if energy < energyFloor {
			energy = energyFloor
		}
		e.filtered[i] = math.Log(energy)
	}

	for i := range numCoefficients {
		sum := 0.0
		for j := range numFilters {
			sum += e.filtered[j] * dctMatrix[i][j]
		}
		dst[i] = sum
	}

	return dst
}

var dctMatrix = func() [numCoefficients][numFilters]float64 {
	var m [numCoefficients][numFilters]float64
	for i := range numCoefficients {
		scale := math.Sqrt(2.0 / float64(numFilters))
		if i == 0 {
			scale = math.Sqrt(1.0 / float64(numFilters))
		}
		for j := range numFilters {
			m[i][j] = scale * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(numFilters))
		}
	}
	return m
}()

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

	for i := range numFilters {
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

		for j := start; j < mid; j++ {
			denom := float64(mid - start)
			if denom == 0 {
				continue
			}
			filterBank[i][j] = (float64(j) - float64(start)) / denom
		}
		for j := mid; j < end; j++ {
			denom := float64(end - mid)
			if denom == 0 {
				continue
			}
			filterBank[i][j] = (float64(end) - float64(j)) / denom
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

func applyFilterBank(magnitudes []float64, filterBank [][]float64, filtered []float64, starts, ends []int) {
	for i, filter := range filterBank {
		start := 0
		end := len(filter)
		if i < len(starts) {
			start = starts[i]
		}
		if i < len(ends) {
			end = ends[i]
		}
		limit := min(len(magnitudes), end)
		if start >= limit {
			filtered[i] = 0
			continue
		}
		sum := 0.0
		for j := start; j < limit; j++ {
			sum += magnitudes[j] * filter[j]
		}
		filtered[i] = sum
	}
}

func cmnMean(mfcc [][]float64) []float64 {
	if len(mfcc) == 0 {
		return nil
	}

	coeffCount := len(mfcc[0])
	for _, frame := range mfcc[1:] {
		if len(frame) < coeffCount {
			coeffCount = len(frame)
		}
	}
	means := make([]float64, coeffCount)

	for _, frame := range mfcc {
		for i := 0; i < coeffCount; i++ {
			means[i] += frame[i]
		}
	}
	for i := range means {
		means[i] /= float64(len(mfcc))
	}

	return means
}

func cmnMeanStd(mfcc [][]float64) (mean []float64, std []float64) {
	if len(mfcc) == 0 {
		return nil, nil
	}

	coeffCount := len(mfcc[0])
	for _, frame := range mfcc[1:] {
		if len(frame) < coeffCount {
			coeffCount = len(frame)
		}
	}
	if coeffCount == 0 {
		return nil, nil
	}

	mean = make([]float64, coeffCount)
	m2 := make([]float64, coeffCount)
	count := 0.0

	for _, frame := range mfcc {
		limit := min(coeffCount, len(frame))
		count++
		for i := 0; i < limit; i++ {
			v := frame[i]
			delta := v - mean[i]
			mean[i] += delta / count
			m2[i] += delta * (v - mean[i])
		}
	}

	std = make([]float64, coeffCount)
	if count == 0 {
		return mean, std
	}
	for i := range std {
		variance := m2[i] / count
		if variance < 0 {
			variance = 0
		}
		stdVal := math.Sqrt(variance)
		switch {
		case stdVal == 0:
			stdVal = 1
		case stdVal < cmvnStdFloor:
			stdVal = cmvnStdFloor
		}
		std[i] = stdVal
	}

	return mean, std
}

func cmnMeanStdMulti(all ...[][]float64) (mean []float64, std []float64) {
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
			for i := 0; i < limit; i++ {
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
		switch {
		case stdVal == 0:
			stdVal = 1
		case stdVal < cmvnStdFloor:
			stdVal = cmvnStdFloor
		}
		std[i] = stdVal
	}

	return mean, std
}

func applyCMN(mfcc [][]float64, mean []float64) []float64 {
	if len(mfcc) == 0 {
		return mean
	}
	if len(mean) == 0 {
		mean = cmnMean(mfcc)
	}
	if len(mean) == 0 {
		return mean
	}

	coeffCount := len(mean)
	for _, frame := range mfcc {
		limit := min(coeffCount, len(frame))
		for i := range limit {
			frame[i] -= mean[i]
		}
	}
	return mean
}

func applyCMVN(mfcc [][]float64, mean, std []float64) ([]float64, []float64) {
	if len(mfcc) == 0 {
		return mean, std
	}
	if len(mean) == 0 || len(std) == 0 {
		mean, std = cmnMeanStd(mfcc)
	}
	if len(mean) == 0 || len(std) == 0 {
		return mean, std
	}

	coeffCount := min(len(mean), len(std))
	for _, frame := range mfcc {
		limit := min(coeffCount, len(frame))
		for i := 0; i < limit; i++ {
			frame[i] -= mean[i]
			stdVal := std[i]
			if stdVal < cmvnStdFloor {
				stdVal = cmvnStdFloor
			}
			frame[i] /= stdVal
		}
	}
	return mean, std
}

// normalizeForMatch는 whole에서 계산한 CMVN 통계만을 사용해 두 입력을 정규화한다.
func normalizeForMatch(mfccWhole, mfccChunk [][]float64) ([]float64, []float64) {
	mean, std := cmnMeanStd(mfccWhole)
	applyCMVN(mfccWhole, mean, std)
	applyCMVN(mfccChunk, mean, std)
	l2NormalizeFrames(mfccWhole, distanceStartCoeff)
	l2NormalizeFrames(mfccChunk, distanceStartCoeff)
	return mean, std
}

func validateMFCC(mfcc [][]float64) error {
	for i, frame := range mfcc {
		for j, v := range frame {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return errors.Errorf("invalid MFCC value at frame %d, coeff %d", i, j)
			}
		}
	}
	return nil
}

func l2NormalizeFrames(mfcc [][]float64, startCoeff int) {
	for _, frame := range mfcc {
		if len(frame) <= startCoeff {
			continue
		}
		sum := 0.0
		for i := startCoeff; i < len(frame); i++ {
			sum += frame[i] * frame[i]
		}
		if sum <= energyFloor {
			continue
		}
		scale := 1.0 / math.Sqrt(sum)
		for i := startCoeff; i < len(frame); i++ {
			frame[i] *= scale
		}
	}
}

// https://en.wikipedia.org/wiki/Mel_scale
func hzToMel(freq float64) float64 {
	return 2_595 * math.Log10(1+freq/700)
}

func melToHz(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2_595) - 1)
}

func preprocessSamples(samples []float64, preEmphasis float64) []float64 {
	return preprocessSamplesWithMean(samples, preEmphasis, math.NaN())
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

func preprocessSamplesWithMean(samples []float64, preEmphasis, mean float64) []float64 {
	return preprocessSamplesWithMeanInto(nil, samples, preEmphasis, mean)
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

	coeffCount := min(len(mfccChunk[0]), len(mfccWhole[0]))
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

	for i := 0; i <= len(mfccWhole)-len(mfccChunk); i++ {
		distance := 0.0
	loop:
		for j := range mfccChunk {
			chunkFrame := mfccChunk[j]
			wholeFrame := mfccWhole[i+j]

			limit := coeffCount
			if len(chunkFrame) < limit {
				limit = len(chunkFrame)
			}
			if len(wholeFrame) < limit {
				limit = len(wholeFrame)
			}

			for k := startCoeff; k < limit; k++ {
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
