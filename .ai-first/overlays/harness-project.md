## Project Overlay

- `README.md`의 public examples와 `mfcc/api.go`의 exported API를 함께 유지한다.
- MFCC extraction은 configuration validation, mono conversion, frame construction,
  mel filterbank, log energy와 DCT/normalization 단계를 명시적으로 보존한다.
- offset search는 두 입력의 sample rate·length precondition과 gain-robust comparison
  contract를 유지한다.
- ASR feature는 base coefficient, delta와 delta-delta의 frame alignment를 test로
  증명한다.
- 변경은 focused package test에서 시작해 format, vet, 전체 test와 publication
  boundary까지 넓힌다.

## Related Documents

- Navigation and baseline: `docs/HANDOFF.md`.
- Public usage and constraints: `README.md`.
- Public API: `mfcc/api.go`.
- Canonical local gate: `scripts/check.sh`.
