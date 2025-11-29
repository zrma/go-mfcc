package mfcc

import (
	"encoding/binary"
	"io"
	"math"
	"os"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/pkg/errors"
	"go.uber.org/multierr"
)

const (
	wavFormatPCM        = 1
	wavFormatIEEEFloat  = 3
	wavFormatExtensible = 0xFFFE
)

var (
	wavSubFormatPCM       = [16]byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}
	wavSubFormatIEEEFloat = [16]byte{0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}
)

type wavFormatInfo struct {
	audioFormat        uint16
	validBitsPerSample uint16
	subFormat          [16]byte
	dataChunkSize      uint32
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

	foundFmt := false
	foundData := false

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
			switch string(chunkID[:]) {
			case "fmt ":
				return info, errors.New("fmt chunk too short")
			case "data":
				info.dataChunkSize = chunkSize
				foundData = true
				if foundFmt {
					return info, nil
				}
			default:
				// 크기가 0인 부가 chunk는 그대로 넘긴다.
			}
			continue
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
				if len(buf) < 18 {
					return info, errors.New("fmt chunk too short for extensible format")
				}
				extraSize := binary.LittleEndian.Uint16(buf[16:18])
				if extraSize < 22 {
					return info, errors.Errorf("invalid extensible fmt chunk extra size: %d", extraSize)
				}
				if len(buf) < 18+int(extraSize) {
					return info, errors.Errorf("fmt chunk too short for extensible format: need %d bytes, got %d", 18+int(extraSize), len(buf))
				}
				// WAVE_FORMAT_EXTENSIBLE stores valid bits (2 bytes), channel mask (4 bytes), and sub-format GUID (16 bytes)
				info.validBitsPerSample = binary.LittleEndian.Uint16(buf[18:20])
				copy(info.subFormat[:], buf[24:40])
			}

			foundFmt = true
			if chunkSize%2 == 1 {
				if _, err := r.Seek(1, io.SeekCurrent); err != nil {
					return info, errors.Wrap(err, "skip fmt padding failed")
				}
			}
			if foundData {
				return info, nil
			}
			continue
		}

		if string(chunkID[:]) == "data" {
			info.dataChunkSize = chunkSize
			foundData = true
			if foundFmt {
				return info, nil
			}
			skip := int64(chunkSize)
			if chunkSize%2 == 1 {
				skip++
			}
			if _, err := r.Seek(skip, io.SeekCurrent); err != nil {
				return info, errors.Wrap(err, "skip data chunk failed")
			}
			continue
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
		return nil, 0, errors.Errorf("invalid channel count: %d", decoder.NumChans)
	}
	sampleRate = int(decoder.SampleRate)
	if sampleRate <= 0 {
		return nil, 0, errors.Errorf("invalid sample rate: %dHz", sampleRate)
	}

	audioFormat := formatInfo.audioFormat
	if audioFormat == 0 {
		audioFormat = decoder.WavAudioFormat
	}
	isFloat := formatInfo.isIEEEFloat() || audioFormat == wavFormatIEEEFloat
	isPCM := formatInfo.isPCM() || audioFormat == wavFormatPCM

	switch {
	case isFloat:
		data, err = decodeFloatPCM(decoder, channels, int(formatInfo.dataChunkSize))
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

	data, err = decodeIntPCMToMono(buf, formatInfo, channels, int(decoder.BitDepth))
	if err != nil {
		return nil, 0, err
	}
	return data, sampleRate, nil
}

func decodeIntPCMToMono(buf *audio.IntBuffer, formatInfo wavFormatInfo, channels int, decoderBitDepth int) ([]float64, error) {
	if buf == nil || buf.Format == nil {
		return nil, errors.New("invalid PCM buffer or format")
	}

	bitDepth := buf.SourceBitDepth
	if bitDepth <= 0 {
		bitDepth = decoderBitDepth
	}
	if bitDepth <= 0 {
		return nil, errors.New("unknown source bit depth")
	}
	bytesPerSample := (bitDepth-1)/8 + 1
	sampleCount := len(buf.Data)
	if formatInfo.dataChunkSize > 0 {
		dataSize := int(formatInfo.dataChunkSize)
		if dataSize%bytesPerSample != 0 {
			return nil, errors.Errorf("wav data size (%d bytes) is not aligned to sample size (%d bytes)", dataSize, bytesPerSample)
		}
		expectedSamples := dataSize / bytesPerSample
		if expectedSamples > sampleCount {
			return nil, errors.Errorf("wav data truncated: expected %d samples, got %d", expectedSamples, sampleCount)
		}
		if expectedSamples < sampleCount {
			buf.Data = buf.Data[:expectedSamples]
			sampleCount = expectedSamples
		}
	}
	if rem := sampleCount % channels; rem != 0 {
		return nil, errors.Errorf("wav data length (%d samples) is not divisible by channel count (%d)", sampleCount, channels)
	}
	frames := sampleCount / channels

	validBits := int(formatInfo.validBitsPerSample)
	if validBits > bitDepth {
		return nil, errors.Errorf("valid bits per sample (%d) exceed bit depth (%d)", validBits, bitDepth)
	}
	normalizerBits := bitDepth
	unsignedPCM := bitDepth == 8
	if formatInfo.audioFormat == wavFormatExtensible && validBits > 0 && validBits < bitDepth {
		if isRightAlignedPCM(buf.Data, validBits, bitDepth) {
			// WAVE_FORMAT_EXTENSIBLE PCM은 유효 비트를 LSB에 두는 경우가 있어, 해당 경우엔 validBits 기준으로 정규화한다.
			normalizerBits = validBits
			if validBits == 8 {
				unsignedPCM = true
			}
		}
	}
	normalizer := math.Pow(2, float64(normalizerBits)-1)
	offset := 0.0
	if unsignedPCM {
		// 8비트 PCM은 unsigned 데이터이므로 중앙값(0x80)을 제거해 0 기준으로 맞춘다.
		offset = normalizer
	}

	data := make([]float64, frames)
	for i := range frames {
		sum := 0.0
		base := i * channels
		for c := range channels {
			sample := float64(buf.Data[base+c]) - offset
			sum += sample / normalizer
		}
		data[i] = sum / float64(channels)
	}

	return data, nil
}

func decodeFloatPCM(decoder *wav.Decoder, channels int, dataChunkSize int) ([]float64, error) {
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

	byteCount := decoder.PCMSize
	if dataChunkSize > 0 {
		if dataChunkSize > decoder.PCMSize {
			return nil, errors.Errorf("wav data size (%d bytes) exceeds available PCM size (%d bytes)", dataChunkSize, decoder.PCMSize)
		}
		byteCount = dataChunkSize
	}

	pcmBytes := make([]byte, byteCount)
	if _, err := io.ReadFull(decoder.PCMChunk, pcmBytes); err != nil {
		return nil, errors.Wrap(err, "read PCM chunk failed")
	}

	if rem := len(pcmBytes) % bytesPerFrame; rem != 0 {
		return nil, errors.Errorf("wav data length (%d bytes) is not divisible by frame size (%d)", len(pcmBytes), bytesPerFrame)
	}
	frames := len(pcmBytes) / bytesPerFrame

	data := make([]float64, frames)
	for i := range data {
		sum := 0.0
		frameOffset := i * bytesPerFrame
		for c := range channels {
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

// WAVE_FORMAT_EXTENSIBLE PCM 샘플이 LSB 정렬인지 판별한다.
// validBits는 정보용이므로, 하위 비트가 실제로 사용될 때만 LSB 정렬로 본다.
func isRightAlignedPCM(samples []int, validBits, bitDepth int) bool {
	if validBits <= 0 || validBits >= bitDepth {
		return false
	}
	shift := bitDepth - validBits
	if shift <= 0 || shift >= 63 {
		return false
	}

	maxValid := int64(1<<(validBits-1)) - 1
	mask := uint64(1<<uint(shift)) - 1

	hasLowBits := false
	maxAbs := int64(0)
	for _, sample := range samples {
		if uint64(sample)&mask != 0 {
			hasLowBits = true
		}
		abs := int64(sample)
		if abs < 0 {
			abs = -abs
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	if maxAbs > maxValid {
		return false
	}
	return hasLowBits
}
