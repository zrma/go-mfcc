package mfcc

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/goleak"
)

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
