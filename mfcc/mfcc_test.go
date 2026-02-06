package mfcc

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
)

func writeTestWav(t *testing.T, path string, sampleRate, bitDepth, numChannels int, data []int) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	enc := wav.NewEncoder(f, sampleRate, bitDepth, numChannels, 1)
	buf := &audio.IntBuffer{
		Data: data,
		Format: &audio.Format{
			NumChannels: numChannels,
			SampleRate:  sampleRate,
		},
		SourceBitDepth: bitDepth,
	}

	require.NoError(t, enc.Write(buf))
	require.NoError(t, enc.Close())
}

func writeFloatWav(t *testing.T, path string, sampleRate, bitDepth, numChannels int, data []float32) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	bytesPerSample := bitDepth / 8
	require.NotZero(t, bytesPerSample)

	if len(data)%numChannels != 0 {
		t.Fatalf("data length (%d) is not divisible by channel count (%d)", len(data), numChannels)
	}

	blockAlign := numChannels * bytesPerSample
	dataSize := len(data) * bytesPerSample
	byteRate := sampleRate * blockAlign
	const fmtChunkSize = 16
	riffSize := 4 + (8 + fmtChunkSize) + (8 + dataSize)

	write := func(v any) {
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	}

	_, err = f.Write([]byte("RIFF"))
	require.NoError(t, err)
	write(uint32(riffSize))
	_, err = f.Write([]byte("WAVE"))
	require.NoError(t, err)

	_, err = f.Write([]byte("fmt "))
	require.NoError(t, err)
	write(uint32(fmtChunkSize))
	write(uint16(wavFormatIEEEFloat))
	write(uint16(numChannels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitDepth))

	_, err = f.Write([]byte("data"))
	require.NoError(t, err)
	write(uint32(dataSize))
	for _, sample := range data {
		write(sample)
	}
}

func writeExtensibleFloatWav(t *testing.T, path string, sampleRate, bitDepth, numChannels int, data []float32) {
	writeExtensibleFloatWavWithSubFormat(t, path, sampleRate, bitDepth, numChannels, data, wavSubFormatIEEEFloat)
}

func writeExtensibleFloatWavWithSubFormat(t *testing.T, path string, sampleRate, bitDepth, numChannels int, data []float32, subFormat [16]byte) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	bytesPerSample := bitDepth / 8
	require.NotZero(t, bytesPerSample)

	if len(data)%numChannels != 0 {
		t.Fatalf("data length (%d) is not divisible by channel count (%d)", len(data), numChannels)
	}

	blockAlign := numChannels * bytesPerSample
	dataSize := len(data) * bytesPerSample
	byteRate := sampleRate * blockAlign
	const (
		fmtChunkBaseSize = 16
		fmtCbSize        = 22
	)
	fmtChunkSize := fmtChunkBaseSize + 2 + fmtCbSize
	riffSize := 4 + (8 + fmtChunkSize) + (8 + dataSize)

	write := func(v any) {
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	}

	_, err = f.Write([]byte("RIFF"))
	require.NoError(t, err)
	write(uint32(riffSize))
	_, err = f.Write([]byte("WAVE"))
	require.NoError(t, err)

	_, err = f.Write([]byte("fmt "))
	require.NoError(t, err)
	write(uint32(fmtChunkSize))
	write(uint16(wavFormatExtensible))
	write(uint16(numChannels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitDepth))
	write(uint16(fmtCbSize))
	write(uint16(bitDepth)) // valid bits per sample
	write(uint32(0))        // channel mask
	write(subFormat)

	_, err = f.Write([]byte("data"))
	require.NoError(t, err)
	write(uint32(dataSize))
	for _, sample := range data {
		write(sample)
	}
}

func writeExtensiblePCMWav(t *testing.T, path string, sampleRate, bitDepth, validBits, numChannels int, data []int) {
	writeExtensiblePCMWavWithShift(t, path, sampleRate, bitDepth, validBits, numChannels, data, bitDepth-validBits)
}

func writeExtensiblePCMWavRightAligned(t *testing.T, path string, sampleRate, bitDepth, validBits, numChannels int, data []int) {
	writeExtensiblePCMWavWithShift(t, path, sampleRate, bitDepth, validBits, numChannels, data, 0)
}

func writeExtensiblePCMWavWithShift(t *testing.T, path string, sampleRate, bitDepth, validBits, numChannels int, data []int, shift int) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	bytesPerSample := bitDepth / 8
	require.NotZero(t, bytesPerSample)
	require.Zero(t, bitDepth%8)
	require.GreaterOrEqual(t, bitDepth, validBits)
	require.Positive(t, validBits)
	require.GreaterOrEqual(t, shift, 0)
	require.LessOrEqual(t, shift, bitDepth-validBits)

	if len(data)%numChannels != 0 {
		t.Fatalf("data length (%d) is not divisible by channel count (%d)", len(data), numChannels)
	}

	blockAlign := numChannels * bytesPerSample
	dataSize := len(data) * bytesPerSample
	byteRate := sampleRate * blockAlign
	const (
		fmtChunkBaseSize = 16
		fmtCbSize        = 22
	)
	fmtChunkSize := fmtChunkBaseSize + 2 + fmtCbSize
	riffSize := 4 + (8 + fmtChunkSize) + (8 + dataSize)

	write := func(v any) {
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	}

	_, err = f.Write([]byte("RIFF"))
	require.NoError(t, err)
	write(uint32(riffSize))
	_, err = f.Write([]byte("WAVE"))
	require.NoError(t, err)

	_, err = f.Write([]byte("fmt "))
	require.NoError(t, err)
	write(uint32(fmtChunkSize))
	write(uint16(wavFormatExtensible))
	write(uint16(numChannels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitDepth))
	write(uint16(fmtCbSize))
	write(uint16(validBits)) // valid bits per sample
	write(uint32(0))         // channel mask
	write(wavSubFormatPCM)

	_, err = f.Write([]byte("data"))
	require.NoError(t, err)
	write(uint32(dataSize))
	maxMagnitude := 1 << (validBits - 1)
	for _, sample := range data {
		require.Less(t, sample, maxMagnitude)
		require.GreaterOrEqual(t, sample, -maxMagnitude)
		shifted := sample << shift
		for i := range bytesPerSample {
			b := byte(shifted >> (8 * i))
			write(b)
		}
	}
}

func writeBrokenExtensibleWav(t *testing.T, path string, sampleRate, bitDepth, numChannels int, data []int) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	bytesPerSample := bitDepth / 8
	require.NotZero(t, bytesPerSample)

	if len(data)%numChannels != 0 {
		t.Fatalf("data length (%d) is not divisible by channel count (%d)", len(data), numChannels)
	}

	blockAlign := numChannels * bytesPerSample
	dataSize := len(data) * bytesPerSample
	byteRate := sampleRate * blockAlign
	const fmtChunkSize = 16 // 확장 포맷인데 추가 바이트가 없는 비정상 헤더를 생성한다.
	riffSize := 4 + (8 + fmtChunkSize) + (8 + dataSize)

	write := func(v any) {
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	}

	_, err = f.Write([]byte("RIFF"))
	require.NoError(t, err)
	write(uint32(riffSize))
	_, err = f.Write([]byte("WAVE"))
	require.NoError(t, err)

	_, err = f.Write([]byte("fmt "))
	require.NoError(t, err)
	write(uint32(fmtChunkSize))
	write(uint16(wavFormatExtensible))
	write(uint16(numChannels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitDepth))

	_, err = f.Write([]byte("data"))
	require.NoError(t, err)
	write(uint32(dataSize))
	for _, sample := range data {
		for i := range bytesPerSample {
			b := byte(sample >> (8 * i))
			write(b)
		}
	}
}

func writeUnsigned8BitWavWithPadding(t *testing.T, path string, sampleRate, numChannels int, data []byte) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	if len(data)%numChannels != 0 {
		t.Fatalf("data length (%d) is not divisible by channel count (%d)", len(data), numChannels)
	}

	const (
		bitDepth     = 8
		fmtChunkSize = 16
	)
	bytesPerSample := bitDepth / 8
	blockAlign := numChannels * bytesPerSample
	dataSize := len(data) * bytesPerSample
	padding := 0
	if dataSize%2 == 1 {
		padding = 1
	}
	byteRate := sampleRate * blockAlign
	riffSize := 4 + (8 + fmtChunkSize) + (8 + dataSize + padding)

	write := func(v any) {
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	}

	_, err = f.Write([]byte("RIFF"))
	require.NoError(t, err)
	write(uint32(riffSize))
	_, err = f.Write([]byte("WAVE"))
	require.NoError(t, err)

	_, err = f.Write([]byte("fmt "))
	require.NoError(t, err)
	write(uint32(fmtChunkSize))
	write(uint16(wavFormatPCM))
	write(uint16(numChannels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitDepth))

	_, err = f.Write([]byte("data"))
	require.NoError(t, err)
	write(uint32(dataSize))
	_, err = f.Write(data)
	require.NoError(t, err)
	if padding == 1 {
		write(uint8(0))
	}
}

func writeWavWithZeroSizeChunk(t *testing.T, path string, sampleRate, bitDepth, numChannels int, data []int16) {
	t.Helper()

	f, err := os.Create(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })

	bytesPerSample := bitDepth / 8
	require.NotZero(t, bytesPerSample)
	require.Zero(t, bitDepth%8)

	if len(data)%numChannels != 0 {
		t.Fatalf("data length (%d) is not divisible by channel count (%d)", len(data), numChannels)
	}

	blockAlign := numChannels * bytesPerSample
	dataSize := len(data) * bytesPerSample
	byteRate := sampleRate * blockAlign
	const (
		fmtChunkSize  = 16
		junkChunkSize = 0
	)
	riffSize := 4 + (8 + fmtChunkSize) + (8 + junkChunkSize) + (8 + dataSize)

	write := func(v any) {
		require.NoError(t, binary.Write(f, binary.LittleEndian, v))
	}

	_, err = f.Write([]byte("RIFF"))
	require.NoError(t, err)
	write(uint32(riffSize))
	_, err = f.Write([]byte("WAVE"))
	require.NoError(t, err)

	_, err = f.Write([]byte("fmt "))
	require.NoError(t, err)
	write(uint32(fmtChunkSize))
	write(uint16(wavFormatPCM))
	write(uint16(numChannels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitDepth))

	_, err = f.Write([]byte("JUNK"))
	require.NoError(t, err)
	write(uint32(junkChunkSize))

	_, err = f.Write([]byte("data"))
	require.NoError(t, err)
	write(uint32(dataSize))
	for _, sample := range data {
		write(sample)
	}
}

func TestFindOffset(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	const testPathPrefix = "testdata"

	for i, tt := range []struct {
		wholeWavPath string
		chunkWavPath string
		want         float64
	}{
		{
			filepath.Join(testPathPrefix, "sample.wav"),
			filepath.Join(testPathPrefix, "sample_3_to_7.wav"),
			3.00,
		},
		{
			filepath.Join(testPathPrefix, "sample.wav"),
			filepath.Join(testPathPrefix, "sample_11.11_to_12.34.wav"),
			11.11,
		},
		{
			filepath.Join(testPathPrefix, "sample.wav"),
			filepath.Join(testPathPrefix, "sample_15.75_to_18.84.wav"),
			15.75,
		},
		{
			filepath.Join(testPathPrefix, "sample.wav"),
			filepath.Join(testPathPrefix, "sample_15.75_to_20.wav"),
			15.75,
		},
	} {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			got, err := FindOffset(tt.wholeWavPath, tt.chunkWavPath)
			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFindOffsetWithConfig_ZeroConfigUsesDefaults(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	got, err := FindOffsetWithConfig(
		filepath.Join("testdata", "sample.wav"),
		filepath.Join("testdata", "sample_3_to_7.wav"),
		Config{},
	)
	require.NoError(t, err)
	assert.Equal(t, 3.00, got)
}

func TestFindOffsetWithConfig_InvalidConfig(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	cfg := DefaultConfig()
	cfg.NumFilters = 10
	cfg.NumCoefficients = 20

	_, err := FindOffsetWithConfig(
		filepath.Join("testdata", "sample.wav"),
		filepath.Join("testdata", "sample_3_to_7.wav"),
		cfg,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "num coefficients")
}

func TestFindOffsetWithConfig_InsufficientCoefficients(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	cfg := DefaultConfig()
	cfg.NumCoefficients = 1

	_, err := FindOffsetWithConfig(
		filepath.Join("testdata", "sample.wav"),
		filepath.Join("testdata", "sample_3_to_7.wav"),
		cfg,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient MFCC coefficients")
}

func TestReadWavFile_NormalizesMultiChannel(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "stereo.wav")
	writeTestWav(t, path, 16_000, 16, 2, []int{1_000, 3_000, -2_000, -1_000})

	data, sampleRate, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)

	expected := []float64{
		2_000.0 / 32_768.0,
		-1_500.0 / 32_768.0,
	}
	assert.InDeltaSlice(t, expected, data, 1e-9)
}

func TestReadWavFile_FloatPCM(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "float.wav")
	writeFloatWav(t, path, 16_000, 32, 2, []float32{0.25, -0.25, 0.5, 0.75})

	data, sampleRate, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)

	expected := []float64{
		(0.25 - 0.25) / 2,
		(0.5 + 0.75) / 2,
	}
	assert.InDeltaSlice(t, expected, data, 1e-6)
}

func TestReadWavFile_ExtensibleFloatPCM(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "float-extensible.wav")
	writeExtensibleFloatWav(t, path, 16_000, 32, 2, []float32{0.25, -0.25, 0.5, 0.75})

	data, sampleRate, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)

	expected := []float64{
		(0.25 - 0.25) / 2,
		(0.5 + 0.75) / 2,
	}
	assert.InDeltaSlice(t, expected, data, 1e-6)
}

func TestReadWavFile_FloatPCMRejectsNaN(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "float-nan.wav")
	writeFloatWav(t, path, 16_000, 32, 1, []float32{float32(math.NaN()), float32(math.Inf(1))})

	_, _, err := readWavFile(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid float PCM sample")
}

func TestReadWavFile_Unsigned8BitPCM(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "u8.wav")
	writeTestWav(t, path, 16_000, 8, 1, []int{0, 128, 255})

	data, sampleRate, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)

	expected := []float64{-1.0, 0, 127.0 / 128.0}
	assert.InDeltaSlice(t, expected, data, 1e-12)
}

func TestReadWavFile_Unsigned8BitPCMWithPadding(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "u8-padding.wav")
	writeUnsigned8BitWavWithPadding(t, path, 16_000, 1, []byte{0, 128, 255, 64, 192})

	data, sampleRate, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)

	expected := []float64{-1.0, 0, 127.0 / 128.0, -0.5, 0.5}
	assert.Len(t, data, len(expected))
	assert.InDeltaSlice(t, expected, data, 1e-12)
}

func TestReadWavFile_ZeroSizeChunk(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "zero-chunk.wav")
	writeWavWithZeroSizeChunk(t, path, 16_000, 16, 1, []int16{0, 1_000})

	data, sampleRate, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, 16_000, sampleRate)

	expected := []float64{0, 1_000.0 / 32_768.0}
	assert.InDeltaSlice(t, expected, data, 1e-12)
}

func TestReadWavFile_UnsupportedExtensibleSubFormat(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "float-extensible-unknown.wav")

	var subFormat [16]byte
	copy(subFormat[:], []byte{0x13, 0x37, 0x00, 0x00})

	writeExtensibleFloatWavWithSubFormat(t, path, 16_000, 32, 1, []float32{0.25, -0.5}, subFormat)

	_, _, err := readWavFile(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported wav format")
}

func TestReadWavFile_ExtensibleFmtChunkTooShort(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "broken-extensible.wav")
	writeBrokenExtensibleWav(t, path, 16_000, 16, 1, []int{0, 1})

	_, _, err := readWavFile(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "extensible")
}

func TestReadWavFile_ExtensiblePCMValidBits(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "pcm-extensible-validbits.wav")

	const (
		sampleRate  = 16_000
		bitDepth    = 24
		validBits   = 20
		numChannels = 1
	)

	data := []int{
		524_287,  // max positive in 20 bits
		-262_144, // -0.5 in 20-bit range
	}

	writeExtensiblePCMWav(t, path, sampleRate, bitDepth, validBits, numChannels, data)

	out, sr, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, sampleRate, sr)

	normalizer := math.Pow(2, float64(validBits)-1)
	expected := []float64{
		float64(data[0]) / normalizer,
		float64(data[1]) / normalizer,
	}
	assert.InDeltaSlice(t, expected, out, 1e-12)
}

func TestReadWavFile_ExtensiblePCMValidBitsRightAligned(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "pcm-extensible-right.wav")

	const (
		sampleRate  = 16_000
		bitDepth    = 24
		validBits   = 20
		numChannels = 1
	)

	data := []int{
		12_345,
		-12_345,
	}

	writeExtensiblePCMWavRightAligned(t, path, sampleRate, bitDepth, validBits, numChannels, data)

	out, sr, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, sampleRate, sr)

	normalizer := math.Pow(2, float64(validBits)-1)
	expected := []float64{
		float64(data[0]) / normalizer,
		float64(data[1]) / normalizer,
	}
	assert.InDeltaSlice(t, expected, out, 1e-12)
}

func TestReadWavFile_ExtensiblePCMZeroLowBits_AssumesMSBAligned(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "pcm-extensible-right-zeros.wav")

	const (
		sampleRate  = 16_000
		bitDepth    = 24
		validBits   = 20
		numChannels = 1
	)

	data := []int{
		16,
		-32,
		48,
	}

	writeExtensiblePCMWavRightAligned(t, path, sampleRate, bitDepth, validBits, numChannels, data)

	out, sr, err := readWavFile(path)
	require.NoError(t, err)
	assert.Equal(t, sampleRate, sr)

	normalizer := math.Pow(2, float64(bitDepth)-1)
	expected := []float64{
		float64(data[0]) / normalizer,
		float64(data[1]) / normalizer,
		float64(data[2]) / normalizer,
	}
	assert.InDeltaSlice(t, expected, out, 1e-12)
}

func TestReadWavFile_ChannelMismatch(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	path := filepath.Join(dir, "truncated.wav")
	writeTestWav(t, path, 16_000, 16, 2, []int{1_000, -1_000, 500, -500})

	info, err := os.Stat(path)
	require.NoError(t, err)
	require.NoError(t, os.Truncate(path, info.Size()-2)) // drop one 16-bit sample

	_, _, err = readWavFile(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "truncated")
}

func TestCalculateMFCC_UsesLastWindow(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	sampleRate := 16_000
	windowSize, hopSize := frameParamsFromDurations(sampleRate, defaultWindowDuration, defaultHopDuration)
	samples := make([]float64, windowSize)
	for i := range samples {
		samples[i] = 1.0
	}

	mfcc, hop, err := ComputeMFCC(samples, sampleRate)
	require.NoError(t, err)
	require.Len(t, mfcc, 1)
	assert.Equal(t, hopSize, hop)
}

func TestFrameParamsFromDurations_StandardDurations(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	for _, tt := range []struct {
		sampleRate int
		window     int
		hop        int
	}{
		{8_000, 200, 80},
		{16_000, 400, 160},
		{44_100, 1_103, 441},
	} {
		tt := tt
		t.Run(fmt.Sprintf("%dHz", tt.sampleRate), func(t *testing.T) {
			gotWindow, gotHop := frameParamsFromDurations(tt.sampleRate, defaultWindowDuration, defaultHopDuration)
			assert.Equal(t, tt.window, gotWindow)
			assert.Equal(t, tt.hop, gotHop)
		})
	}
}

func TestFindOffset_SampleRateMismatch(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	writeTestWav(t, whole, 16_000, 16, 1, []int{0, 0, 0, 0})
	writeTestWav(t, chunk, 8_000, 16, 1, []int{0, 0})

	_, err := FindOffset(whole, chunk)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sample rate mismatch")
}

func TestFindOffset_SampleRateTooLow(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	writeTestWav(t, whole, 1_000, 16, 1, []int{0, 0, 0, 0})
	writeTestWav(t, chunk, 1_000, 16, 1, []int{0, 0})

	_, err := FindOffset(whole, chunk)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient mel resolution")
}

func TestNewMFCCExtractor_CommonSampleRates(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	for _, sr := range []int{8_000, 16_000, 44_100} {
		sr := sr
		t.Run(fmt.Sprintf("%dHz", sr), func(t *testing.T) {
			extractor, err := newMFCCExtractor(sr)
			require.NoError(t, err)
			require.NotNil(t, extractor)
			assert.Len(t, extractor.filterBank, extractor.numFilters)
			for i := range extractor.filterBank {
				assert.Greater(t, extractor.filterEnds[i], extractor.filterStarts[i])
			}
		})
	}
}

func TestNewMFCCExtractor_CachesImmutableSetup(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor1, err := newMFCCExtractor(16_000)
	require.NoError(t, err)
	require.NotNil(t, extractor1)
	extractor2, err := newMFCCExtractor(16_000)
	require.NoError(t, err)
	require.NotNil(t, extractor2)

	require.NotEmpty(t, extractor1.hamming)
	require.NotEmpty(t, extractor1.filterBank)
	require.NotEmpty(t, extractor1.filterBank[0])
	require.NotEmpty(t, extractor1.filterStarts)
	require.NotEmpty(t, extractor1.filterEnds)

	assert.Equal(t, extractor1.windowSize, extractor2.windowSize)
	assert.Equal(t, extractor1.hopSize, extractor2.hopSize)
	assert.Equal(t, extractor1.nfft, extractor2.nfft)
	assert.Equal(t, extractor1.binCount, extractor2.binCount)

	assert.Same(t, &extractor1.hamming[0], &extractor2.hamming[0])
	assert.Same(t, &extractor1.filterBank[0][0], &extractor2.filterBank[0][0])
	assert.Same(t, &extractor1.filterStarts[0], &extractor2.filterStarts[0])
	assert.Same(t, &extractor1.filterEnds[0], &extractor2.filterEnds[0])

	assert.NotSame(t, &extractor1.windowed[0], &extractor2.windowed[0])
	assert.NotSame(t, &extractor1.powerSpectrum[0], &extractor2.powerSpectrum[0])
	assert.NotSame(t, &extractor1.filtered[0], &extractor2.filtered[0])
}

func TestFindOffset_MelFiltersRequireResolution(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	const (
		sampleRate  = 2_000
		bitDepth    = 16
		numChannels = 1
	)

	writeTestWav(t, whole, sampleRate, bitDepth, numChannels, make([]int, sampleRate))
	writeTestWav(t, chunk, sampleRate, bitDepth, numChannels, make([]int, sampleRate/2))

	_, err := FindOffset(whole, chunk)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mel")
}

func TestFindOffset_ChunkTooShort(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	writeTestWav(t, whole, 44_100, 16, 1, make([]int, 2_500))
	writeTestWav(t, chunk, 44_100, 16, 1, make([]int, 10))

	_, err := FindOffset(whole, chunk)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too short")
}

func TestCalculateMFCCForOffset_InvalidSamples(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := newMFCCExtractor(16_000)
	require.NoError(t, err)

	samples := make([]float64, extractor.windowSize)
	samples[0] = math.NaN()

	_, err = calculateMFCCForOffset("whole", extractor, samples)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid whole samples")
}

func TestPreprocessSamples_RemovesDCAndPreEmphasis(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	out := preprocessSamplesWithMeanInto(nil, []float64{1, 1, 1}, 0.97, math.NaN())
	require.Len(t, out, 3)
	assert.InDeltaSlice(t, []float64{0, 0, 0}, out, 1e-12)
}

func TestCreateFilterBank_NoNaN(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	fb := createFilterBank(32, 100, defaultNumFilters)
	for _, filter := range fb {
		for _, v := range filter {
			assert.False(t, math.IsNaN(v) || math.IsInf(v, 0))
		}
	}
}

func TestComputeMFCC_SingleSidedSpectrum(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor := &Extractor{
		nfft:            8,
		binCount:        5,
		powerScale:      1.0 / 8.0,
		numFilters:      defaultNumFilters,
		numCoefficients: defaultNumCoefficients,
		energyFloor:     defaultEnergyFloor,
		logEnergyFloor:  math.Log(defaultEnergyFloor),
		dctMatrix:       makeDCTMatrix(defaultNumCoefficients, defaultNumFilters),
		filterBank:      make([][]float64, defaultNumFilters),
		filterStarts:    make([]int, defaultNumFilters),
		filterEnds:      make([]int, defaultNumFilters),
		filtered:        make([]float64, defaultNumFilters),
		powerSpectrum:   make([]float64, 5),
	}

	spectrum := []complex128{
		1 + 0i,
		2 + 0i,
		3 + 0i,
		4 + 0i,
		5 + 0i,
	}
	dst := make([]float64, defaultNumCoefficients)

	extractor.computeMFCC(spectrum, dst)

	expected := []float64{
		0.125, // DC (1^2 / 8)
		1.0,   // doubled non-DC (2^2 / 8 * 2)
		2.25,  // doubled non-DC (3^2 / 8 * 2)
		4.0,   // doubled non-DC (4^2 / 8 * 2)
		3.125, // Nyquist (5^2 / 8)
	}
	assert.InDeltaSlice(t, expected, extractor.powerSpectrum, 1e-12)
}

func TestComputeMFCC_ZeroSignal_OnlyC0HasEnergy(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := NewExtractor(16_000, DefaultConfig())
	require.NoError(t, err)

	samples := make([]float64, extractor.WindowSize())
	mfccs, err := extractor.Calculate(samples)
	require.NoError(t, err)
	require.Len(t, mfccs, 1)
	require.Len(t, mfccs[0], extractor.NumCoefficients())

	for c := 1; c < extractor.NumCoefficients(); c++ {
		assert.InDelta(t, 0.0, mfccs[0][c], 1e-9)
	}
}

func TestComputeMFCC_ScalingAffectsOnlyC0(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := NewExtractor(16_000, DefaultConfig())
	require.NoError(t, err)

	samples := make([]float64, extractor.WindowSize())
	r := rand.New(rand.NewSource(1))
	for i := range samples {
		samples[i] = r.Float64()*2 - 1
	}

	mfcc, err := extractor.Calculate(samples)
	require.NoError(t, err)
	require.Len(t, mfcc, 1)
	require.Len(t, mfcc[0], extractor.NumCoefficients())

	const scale = 0.25
	scaled := make([]float64, len(samples))
	for i := range scaled {
		scaled[i] = samples[i] * scale
	}

	scaledMFCC, err := extractor.Calculate(scaled)
	require.NoError(t, err)
	require.Len(t, scaledMFCC, 1)
	require.Len(t, scaledMFCC[0], extractor.NumCoefficients())

	for c := 1; c < extractor.NumCoefficients(); c++ {
		assert.InDelta(t, mfcc[0][c], scaledMFCC[0][c], 1e-9)
	}
	assert.Greater(t, math.Abs(mfcc[0][0]-scaledMFCC[0][0]), 1e-9)
}

func TestValidateMFCC_DetectsInvalidValue(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	valid := [][]float64{{0, 1}, {2, 3}}
	require.NoError(t, validateMFCC(valid))

	invalid := [][]float64{{0, math.NaN()}}
	err := validateMFCC(invalid)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid MFCC value")
}

func TestCmnMeanStd_ComputesPopulationValues(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{
		{1, 2, 3},
		{3, 4, 5},
	}

	mean, std := cmnMeanStd(mfcc)
	assert.InDeltaSlice(t, []float64{2, 3, 4}, mean, 1e-12)
	assert.InDeltaSlice(t, []float64{1, 1, 1}, std, 1e-12)
}

func TestCmnMeanStdMulti_AggregatesAllInputs(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	b := [][]float64{
		{5, 6},
	}

	mean, std := cmnMeanStdWithFloorForAll(defaultCMVNStdFloor, a, b)

	assert.InDeltaSlice(t, []float64{3, 4}, mean, 1e-12)
	expectedStd := []float64{
		math.Sqrt(8.0 / 3.0),
		math.Sqrt(8.0 / 3.0),
	}
	assert.InDeltaSlice(t, expectedStd, std, 1e-12)
}

func TestCalculateMFCC_ApplyCMN(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	sampleRate := 16_000
	windowSize, _ := frameParamsFromDurations(sampleRate, defaultWindowDuration, defaultHopDuration)
	samples := make([]float64, windowSize) // zero signal

	mfccs, _, err := ComputeMFCC(samples, sampleRate)
	require.NoError(t, err)
	require.NotEmpty(t, mfccs)
	mean := cmnMean(mfccs)
	require.NotEmpty(t, mean)
	for _, frame := range mfccs {
		limit := min(len(mean), len(frame))
		for i := range frame[:limit] {
			frame[i] -= mean[i]
		}
	}

	for c := range len(mfccs[0]) {
		sum := 0.0
		for _, frame := range mfccs {
			sum += frame[c]
		}
		mean := sum / float64(len(mfccs))
		assert.InDelta(t, 0, mean, 1e-12)
	}
}

func TestNormalizeForMatch_UsesWholeStats(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfccWhole := [][]float64{
		{0, 1, 2},
		{10, 11, 12},
	}
	mfccChunk := [][]float64{
		{10, 11, 12},
	}

	wholeCopy := copyMFCC(mfccWhole)
	chunkCopy := copyMFCC(mfccChunk)

	mean, std, err := normalizeForMatch(mfccWhole, mfccChunk, defaultCMVNStdFloor)
	require.NoError(t, err)
	expectedMean, expectedStd := cmnMeanStdWithFloor(wholeCopy, defaultCMVNStdFloor)

	assert.InDeltaSlice(t, expectedMean, mean, 1e-12)
	assert.InDeltaSlice(t, expectedStd, std, 1e-12)

	normalizedChunk := copyMFCC(chunkCopy)
	applyCMVN(normalizedChunk, expectedMean, expectedStd)
	l2NormalizeFrames(normalizedChunk, distanceStartCoeff)

	require.Len(t, mfccChunk, 1)
	assert.InDeltaSlice(t, normalizedChunk[0], mfccChunk[0], 1e-12)
}

func TestApplyCMVN_FloorsTinyStdDev(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{
		{1e-9, 2e-9, 3e-9},
		{1.1e-9, 2.1e-9, 3.1e-9},
	}

	mean, std := cmnMeanStd(mfcc)
	require.NotEmpty(t, std)
	for _, s := range std {
		assert.GreaterOrEqual(t, s, defaultCMVNStdFloor)
	}

	applyCMVN(mfcc, mean, std)

	for _, frame := range mfcc {
		for _, coeff := range frame {
			assert.False(t, math.IsNaN(coeff) || math.IsInf(coeff, 0))
			assert.InDelta(t, 0, coeff, 1e-4)
		}
	}
}

func TestApplyCMVN_UsesProvidedMeanWhenStdMissing(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{
		{1, 2},
		{3, 4},
	}
	mean := []float64{0, 0}

	gotMean, gotStd := applyCMVNWithFloor(mfcc, mean, nil, defaultCMVNStdFloor)

	assert.Equal(t, mean, gotMean)

	expectedStd := []float64{
		math.Sqrt(5),
		math.Sqrt(10),
	}
	assert.InDeltaSlice(t, expectedStd, gotStd, 1e-12)

	expected := [][]float64{
		{1 / expectedStd[0], 2 / expectedStd[1]},
		{3 / expectedStd[0], 4 / expectedStd[1]},
	}
	for i := range expected {
		assert.InDeltaSlice(t, expected[i], mfcc[i], 1e-12)
	}
}

func TestL2NormalizeFrames_SkipTinyEnergy(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	mfcc := [][]float64{
		{0, 1e-8, -1e-8},
	}
	expected := copyMFCC(mfcc)

	l2NormalizeFrames(mfcc, distanceStartCoeff)

	require.Len(t, mfcc, len(expected))
	assert.InDeltaSlice(t, expected[0], mfcc[0], 0)
}

func TestDCTMatrix_Orthonormal(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	matrix := makeDCTMatrix(defaultNumCoefficients, defaultNumFilters)
	for i := range matrix {
		for j := i; j < len(matrix); j++ {
			dot := 0.0
			for k := range defaultNumFilters {
				dot += matrix[i][k] * matrix[j][k]
			}
			if i == j {
				assert.InDelta(t, 1.0, dot, 1e-12)
			} else {
				assert.InDelta(t, 0.0, dot, 1e-12)
			}
		}
	}
}

func TestCalculateWithMean_ReusesPreemphasisBuffer(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := newMFCCExtractor(16_000)
	require.NoError(t, err)
	require.NotNil(t, extractor)

	long := make([]float64, extractor.windowSize*2)
	for i := range long {
		long[i] = float64(i)
	}

	mfccLong := extractor.calculateWithMean(long, math.NaN())
	require.NotEmpty(t, mfccLong)
	require.NotEmpty(t, extractor.preprocessed)

	firstPtr := &extractor.preprocessed[0]
	firstCap := cap(extractor.preprocessed)

	short := make([]float64, extractor.windowSize)
	for i := range short {
		short[i] = float64(i)
	}

	mfccShort := extractor.calculateWithMean(short, math.NaN())
	require.NotEmpty(t, mfccShort)
	assert.Equal(t, firstPtr, &extractor.preprocessed[0])
	assert.Equal(t, firstCap, cap(extractor.preprocessed))
}

func TestCalculateWithMean_ZeroPadsWindow(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	extractor, err := newMFCCExtractor(16_000)
	require.NoError(t, err)
	require.NotNil(t, extractor)
	require.Greater(t, extractor.nfft, extractor.windowSize)

	samples := make([]float64, extractor.windowSize)
	samples[0] = 1.0

	baseline := extractor.calculateWithMean(samples, 0)
	require.Len(t, baseline, 1)
	baseFrame := slices.Clone(baseline[0])

	for i := extractor.windowSize; i < extractor.nfft; i++ {
		extractor.windowed[i] = 5.0
	}

	polluted := extractor.calculateWithMean(samples, 0)
	require.Len(t, polluted, 1)

	assert.InDeltaSlice(t, baseFrame, polluted[0], 1e-12)
}

func TestFindOffset_UsesWholeMeanForCMN(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	const (
		sampleRate  = 16_000
		bitDepth    = 16
		numChannels = 1
		amplitude   = 12_000
	)

	wholeData := make([]int, sampleRate*2)
	for n := range sampleRate {
		wholeData[n] = int(float64(amplitude) * math.Sin(2*math.Pi*440*float64(n)/float64(sampleRate)))
	}
	for n := range sampleRate {
		wholeData[sampleRate+n] = int(float64(amplitude) * math.Sin(2*math.Pi*880*float64(n)/float64(sampleRate)))
	}

	writeTestWav(t, whole, sampleRate, bitDepth, numChannels, wholeData)
	writeTestWav(t, chunk, sampleRate, bitDepth, numChannels, wholeData[sampleRate:])

	offset, err := FindOffset(whole, chunk)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, offset, 1e-2)
}

func TestFindOffset_HandlesDCOffset(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	const (
		sampleRate  = 16_000
		bitDepth    = 16
		numChannels = 1
		amplitude   = 8_000
		dcOffset    = 6_000
	)

	wholeData := make([]int, sampleRate*2)
	for n := sampleRate; n < len(wholeData); n++ {
		tone := int(float64(amplitude) * math.Sin(2*math.Pi*440*float64(n-sampleRate)/float64(sampleRate)))
		wholeData[n] = dcOffset + tone
	}

	writeTestWav(t, whole, sampleRate, bitDepth, numChannels, wholeData)
	writeTestWav(t, chunk, sampleRate, bitDepth, numChannels, wholeData[sampleRate:])

	offset, err := FindOffset(whole, chunk)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, offset, 1e-2)
}

func TestFindOffset_FloatFormats(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()

	const (
		sampleRate  = 16_000
		bitDepth    = 32
		numChannels = 2
		durationSec = 2
	)

	frameCount := sampleRate * durationSec
	interleaved := make([]float32, frameCount*numChannels)

	for n := range frameCount {
		base := 0.5 * math.Sin(2*math.Pi*440*float64(n)/float64(sampleRate))
		if n >= sampleRate {
			base = 0.6*math.Sin(2*math.Pi*880*float64(n-sampleRate)/float64(sampleRate)) + 0.1
		}
		left := base
		right := base * 0.7
		interleaved[2*n] = float32(left)
		interleaved[2*n+1] = float32(right)
	}

	chunkFrames := sampleRate / 2 // 0.5초 분량을 잘라낸다.
	start := sampleRate * numChannels
	end := start + chunkFrames*numChannels
	chunkData := interleaved[start:end]

	for name, writer := range map[string]func(string, []float32){
		"ieee-float": func(path string, data []float32) {
			writeFloatWav(t, path, sampleRate, bitDepth, numChannels, data)
		},
		"extensible-float": func(path string, data []float32) {
			writeExtensibleFloatWav(t, path, sampleRate, bitDepth, numChannels, data)
		},
	} {
		name := name
		writer := writer

		t.Run(name, func(t *testing.T) {
			whole := filepath.Join(dir, fmt.Sprintf("%s-whole.wav", name))
			chunk := filepath.Join(dir, fmt.Sprintf("%s-chunk.wav", name))

			writer(whole, interleaved)
			writer(chunk, chunkData)

			offset, err := FindOffset(whole, chunk)
			require.NoError(t, err)
			assert.InDelta(t, 1.0, offset, 1e-2)
		})
	}
}

func TestFindOffset_RobustToGainAndNoise(t *testing.T) {
	t.Cleanup(func() { goleak.VerifyNone(t) })

	dir := t.TempDir()
	whole := filepath.Join(dir, "whole.wav")
	chunk := filepath.Join(dir, "chunk.wav")

	const (
		sampleRate  = 16_000
		bitDepth    = 16
		numChannels = 1
		baseAmp     = 8_000
	)

	wholeData := make([]int, sampleRate*2)
	for n := range sampleRate {
		wholeData[n] = int(float64(baseAmp) * math.Sin(2*math.Pi*440*float64(n)/float64(sampleRate)))
	}
	for n := sampleRate; n < len(wholeData); n++ {
		wholeData[n] = int(float64(baseAmp) * math.Sin(2*math.Pi*880*float64(n-sampleRate)/float64(sampleRate)))
	}

	writeTestWav(t, whole, sampleRate, bitDepth, numChannels, wholeData)

	chunkData := make([]int, sampleRate/2)
	rng := rand.New(rand.NewSource(42))
	for i := range chunkData {
		tone := float64(wholeData[sampleRate+i])
		noise := rng.NormFloat64() * 600
		chunkData[i] = int(1.4*tone + noise)
	}
	writeTestWav(t, chunk, sampleRate, bitDepth, numChannels, chunkData)

	offset, err := FindOffset(whole, chunk)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, offset, 1e-2)
}

func copyMFCC(src [][]float64) [][]float64 {
	dst := make([][]float64, len(src))
	for i, frame := range src {
		dst[i] = slices.Clone(frame)
	}
	return dst
}
