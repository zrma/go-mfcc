# go-mfcc

Go library that reads two WAV files, extracts MFCCs, and estimates the time offset between them.

## Install

```bash
go get github.com/zrma/go-mfcc/mfcc
```

## Usage

```go
package main

import (
	"fmt"

	"github.com/zrma/go-mfcc/mfcc"
)

func main() {
	offset, err := mfcc.FindOffset("whole.wav", "chunk.wav")
	if err != nil {
		panic(err)
	}
	fmt.Printf("chunk starts at %.2f seconds\n", offset)
}
```

### 설정을 적용한 offset 계산

```go
cfg := mfcc.DefaultConfig()
cfg.NumFilters = 20
cfg.NumCoefficients = 13
cfg.DisablePreEmphasis = true

offset, err := mfcc.FindOffsetWithConfig("whole.wav", "chunk.wav", cfg)
if err != nil {
	panic(err)
}
_ = offset
```

### MFCC 계산

```go
samples, sampleRate, err := mfcc.ReadWavMono("audio.wav")
if err != nil {
	panic(err)
}

mfccs, hopSize, err := mfcc.ComputeMFCC(samples, sampleRate)
if err != nil {
	panic(err)
}
_ = hopSize // 샘플 단위의 프레임 홉 길이
_ = mfccs
```

### Extractor 재사용

```go
cfg := mfcc.DefaultConfig()
cfg.NumFilters = 20
cfg.NumCoefficients = 13

extractor, err := mfcc.NewExtractor(16_000, cfg)
if err != nil {
	panic(err)
}
mfccs, err := extractor.Calculate(samples)
if err != nil {
	panic(err)
}
_ = mfccs
```

### ASR용 특징 (CMVN + delta/delta-delta)

```go
features, hopSize, err := mfcc.ComputeASRFeatures(samples, sampleRate)
if err != nil {
	panic(err)
}
_ = features // 1프레임당 3 * NumCoefficients
_ = hopSize
```

Notes:
- Both WAV files must share the same sample rate (positive, non-zero).
- The sample rate must be high enough to build 26 mel filters with a 25 ms window. Anything below ~2.6 kHz leaves some filters empty and is rejected; common telephony/music rates (8 kHz/16 kHz/44.1 kHz) are fine.
- Linear PCM (format 1) and IEEE float (format 3) WAV files are supported; other formats return an error.
- Each file must be long enough for at least one 25 ms window (`round(0.025 * sampleRate)` samples); otherwise `FindOffset` returns an error.
- The whole WAV must be at least as long as the chunk; shorter inputs return an error.
- Analysis uses 25 ms Hamming windows with 10 ms hop, 26 mel filters, log filterbank energies (natural log), orthonormal DCT-II (13 coeffs), shared cepstral mean/variance normalization using stats from the whole file (also applied to the chunk), per-frame L2 normalization (coeffs 1..12), and offset search ignores the 0th cepstral coefficient to stay gain-robust.
- Mel filters are spaced using the HTK mel scale (2595 * log10(1 + f/700)) with Slaney-style area normalization, and the power spectrum is treated as single-sided (non-DC/non-Nyquist bins are doubled).
- Samples are mean-centered before optional pre-emphasis (default 0.97) and FFT analysis.
- Multi-channel audio is averaged to mono before analysis.
- MFCC/ASR 계산은 DefaultConfig를 사용하며, 필요하면 Config로 필터/계수/프레임 설정을 변경할 수 있다.

## Development

```bash
go test ./...
```
