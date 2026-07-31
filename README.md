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

### offset 후보의 해상도와 품질 확인

```go
match, err := mfcc.FindOffsetMatch("whole.wav", "chunk.wav")
if err != nil {
	panic(err)
}

fmt.Printf("offset: %.2fs (resolution: %.3fs)\n", match.OffsetSeconds, match.ResolutionSeconds)
fmt.Printf("mean frame distance: %.6f\n", match.MeanFrameDistance)
```

`MeanFrameDistance`는 정규화된 MFCC frame 간 제곱거리의 평균이며 낮을수록
가깝습니다. 입력 도메인마다 분포가 다르므로 match 판정 threshold는 호출자가
검증한 값으로 결정해야 합니다.

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

### 검증된 CMVN 적용

```go
mean, std, err := mfcc.ComputeCMVNChecked(mfccs, 0)
if err != nil {
	panic(err)
}
_, _, err = mfcc.ApplyCMVNChecked(mfccs, mean, std, 0) // mfccs를 in-place로 변경
if err != nil {
	panic(err)
}
```

Notes:
- Both WAV files must share the same sample rate (positive, non-zero).
- The sample rate must be high enough to build 26 mel filters with a 25 ms window. Anything below ~2.6 kHz leaves some filters empty and is rejected; common telephony/music rates (8 kHz/16 kHz/44.1 kHz) are fine.
- Linear PCM (format 1) and IEEE float (format 3) WAV files are supported; other formats return an error.
- A declared WAV data chunk must fit inside the actual file; truncated declarations are rejected before PCM allocation.
- Each file must be long enough for at least one 25 ms window (`round(0.025 * sampleRate)` samples); otherwise `FindOffset` returns an error.
- The whole WAV must be at least as long as the chunk; shorter inputs return an error.
- `FindOffset` returns the best candidate without applying a universal match threshold. The result is quantized to the configured hop duration (10 ms by default); use `FindOffsetMatch` to inspect its resolution and mean frame distance.
- Analysis uses 25 ms Hamming windows with 10 ms hop, 26 mel filters, log filterbank energies (natural log), orthonormal DCT-II (13 coeffs), shared cepstral mean/variance normalization using stats from the whole file (also applied to the chunk), per-frame L2 normalization (coeffs 1..12), and offset search ignores the 0th cepstral coefficient to stay gain-robust.
- Mel filters are spaced using the HTK mel scale (2595 * log10(1 + f/700)) with Slaney-style area normalization, and the power spectrum is treated as single-sided (non-DC/non-Nyquist bins are doubled).
- Samples are mean-centered before optional pre-emphasis (default 0.97) and FFT analysis.
- Multi-channel audio is averaged to mono before analysis.
- MFCC/ASR 계산은 DefaultConfig를 사용하며, 필요하면 Config로 필터/계수/프레임 설정을 변경할 수 있다.
- `ComputeDelta`, `AppendDeltas`와 checked CMVN APIs reject ragged or non-finite feature matrices instead of silently truncating coefficients.

## Development

```bash
scripts/check.sh
```

에이전트 작업은 [AGENTS.md](AGENTS.md)에서 시작하고 현재 baseline과 완료 조건은
[docs/HANDOFF.md](docs/HANDOFF.md)를 따른다.
