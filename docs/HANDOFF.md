# go-mfcc Handoff

## Start Here

1. `AGENTS.md`와 `docs/agent-harness.md`를 읽는다.
2. `jj status`와 `jj diff`로 기존 change를 확인한다.
3. public API 변경이면 `README.md`와 `mfcc/api.go`를 함께 읽는다.
4. task-relevant implementation/test에서 focused verification을 수행한다.
5. `scripts/check.sh`로 전체 local gate를 닫는다.

## Current Baseline

- Linear PCM과 IEEE float WAV를 mono sample로 읽는다.
- 선언된 WAV data chunk가 실제 파일 경계를 넘으면 PCM allocation 전에 거부한다.
- configurable MFCC extraction과 기본 MFCC helper를 제공한다.
- checked CMVN과 고정 차원/finite 입력을 검증하는 delta/ASR feature를 제공한다.
- 정적 MFCC 설정은 최대 32개만 cache하며 Extractor별 작업 buffer는 공유하지 않는다.
- 두 WAV의 sample rate와 length contract를 검증하고 MFCC 기반 offset을 추정하며,
  hop 해상도와 mean frame distance를 함께 조회할 수 있다.
- tracked WAV와 synthetic input을 사용한 API, filterbank, reference, offset test가
  있다.

## Architecture Map

- `mfcc/api.go`: exported configuration, extractor와 convenience API.
- `mfcc/wav.go`: WAV decoding과 mono conversion.
- `mfcc/mfcc.go`: frame, spectrum, mel filterbank와 coefficient extraction.
- `mfcc/delta.go`: ASR delta와 delta-delta feature.
- `mfcc/offset_search.go`: normalized feature 기반 offset search.
- `mfcc/*_test.go`, `mfcc/testdata/`: behavioral evidence.

## Completion Rule

patch나 focused test만으로 완료하지 않는다. public API/example consistency,
`scripts/check.sh`, generated drift와 publication boundary를 확인한다. push, tag와
release는 별도 권한 경계다.
