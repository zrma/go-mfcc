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

Notes:
- Both WAV files must share the same sample rate (positive, non-zero).
- The sample rate must be high enough to build 26 mel filters with a 25 ms window. Anything below ~2.6 kHz leaves some filters empty and is rejected; common telephony/music rates (8 kHz/16 kHz/44.1 kHz) are fine.
- Linear PCM (format 1) and IEEE float (format 3) WAV files are supported; other formats return an error.
- Each file must be long enough for at least one 25 ms window (`round(0.025 * sampleRate)` samples); otherwise `FindOffset` returns an error.
- The whole WAV must be at least as long as the chunk; shorter inputs return an error.
- Analysis uses 25 ms Hamming windows with 10 ms hop, 26 mel filters, log energy, DCT-II (13 coeffs), shared cepstral mean/variance normalization using stats from the whole file (also applied to the chunk), per-frame L2 normalization (coeffs 1..12), and offset search ignores the 0th cepstral coefficient to stay gain-robust.
- Multi-channel audio is averaged to mono before analysis.

## Development

```bash
go test ./...
```
