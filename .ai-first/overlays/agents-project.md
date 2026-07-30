## Repository Overlay

- public package contract는 `mfcc/api.go`, 입력·signal-processing semantics는
  `mfcc/*.go`와 대응 test를 source of truth로 사용한다.
- WAV parsing, MFCC normalization, delta feature와 offset search를 변경할 때는
  focused test와 전체 `scripts/check.sh`를 모두 실행한다.
- sample rate, frame/window size, filter/coefficient 수와 input length validation을
  조용히 완화하지 않는다.
- numerical behavior를 바꿀 때는 synthetic fixture뿐 아니라 tracked WAV fixture의
  offset/reference test를 함께 검증한다.
- 생성된 `AGENTS.md`, `docs/agent-harness.md`, `.ai-first/check.py`는 직접 수정하지
  않고 `.ai-first.toml` 또는 `.ai-first/overlays/`를 수정한 뒤 render한다.
- public artifact에는 machine-local path, private inventory, raw diagnostic output,
  credential과 다른 저장소의 진행 상태를 기록하지 않는다.
- local VCS는 `jj`를 사용한다. 전체 local gate와 publication gate가 통과해도
  push, tag와 release는 별도 권한 경계다.
