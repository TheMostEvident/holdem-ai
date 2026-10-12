# holdem-ai — Phase 5: Agent Interface & RL Compatibility

> 상태: **additive integration kit**. 특정 GitHub 커밋의 파일을 직접 조회·수정하거나 기존 테스트를 실행한 결과가 아니다. 작성 시 제공된 기준은 `a75b632`이며 실제 API 연결은 별도 확인해야 한다.
>
> 최소 Python: **3.11**. 실행 환경에서는 **Python 3.13**으로 테스트했다. 임포트 시 NumPy 또는 RLCard를 필수 요구하지 않는다.

## 1. 목표와 책임 분리

| 모듈 | 구현 | 범위 |
|---|---|---|
| `schema.py` | `InformationState`, `Turn`, `Terminal`, `TurnBasedEnvironment` | 에이전트 경계 |
| `policy.py` | `AgentPolicy`, Random, tabular CFR 정책 샘플러, ε-greedy | 불법 행동 제외 |
| `betting.py` | 7개 No-Limit 베팅 추상화, 정수 칩 금액 및 mask | **엔진에서 계산한 법적 범위**가 반드시 필요 |
| `rewards.py` | 칩 변화량 및 토너먼트 승자 보상 | 종료 시점 보상 |
| `runner.py` | 2인 이상 멀티 에이전트 turn-based self-play와 trajectory | terminal-only rewards |
| `rlcard_adapter.py` | RLCard `reset` / `step` / `is_over` / `get_payoffs` | RLCard API wrapper |
| `snapshot.py` | `deepcopy` 기반 실험적 snapshot | 엔진별 RNG 검증 별도 필요 |

기존 코드는 **수정·삭제하지 않는다**. 엔진 규칙을 새로 구현하는 패치가 아니며 2~6인/tournament 기존 클래스에 연결할 수 있는 경계 모듈을 제공한다.

## 2. 수학적 정의

실제 세계 상태와 관측 사이를 엄격히 구별한다.

$$
 s_t \in \mathcal S,\qquad i_t=P(s_t),\qquad
 I_{i,t}=O_i(s_t,h_t^{\mathrm{obs}})
$$

- `s_t`: 전체 카드·덱·게임 정보가 들어 있는 엔진 상태. **Agent에 전달하지 않는다.**
- `I_{i,t}`: 행동 플레이어 `i`의 프라이빗 카드 + 공개 정보 + 관측 가능한 행동 이력.
- `P(s_t)`: 현재 행동할 플레이어.
- `h_t^{obs}`: 공개 행동 이력과 본인이 관찰한 과거 정보. **CFR의 perfect recall 요건**을 만족하려면 거리낌 없이 요약·삭제하면 안 된다.

법적 행동 마스크:

$$
 m_i(I_t,a)=\mathbf 1[a \in \mathcal A(I_t)],\qquad
 \pi^m(a\mid I_t)=\frac{m_i(I_t,a)\,\pi(a\mid I_t)}{\sum_{a'}m_i(I_t,a')\,\pi(a'\mid I_t)}
$$

합이 0인 정책 가중치는 현재 합법 행동에 균등 확률을 할당한다. 음수/NaN/inf 정책 가중치는 오류로 처리한다.

한 핸드에서의 칩 보상 (레이크 없음):

$$
 r_i^{\text{hand}}=\frac{S_{i,\mathrm{end}}-S_{i,\mathrm{start}}}{BB},\qquad \sum_i r_i=0
$$

주의: `scale=BB`는 **같은 핸드 시작 시점의 BB**를 사용하라. 토너먼트에서는 BB가 시간이 지나며 바뀌므로 BB-normalized reward를 핸드 간 비교할 때 기준을 기록해야 한다.

토너먼트 전체 종료 시점의 승자 독식 보상 (제로섬 중심화 버전):

$$
 r_i^{\text{tournament}}=\mathbf1[i=w]-\frac{1}{N},\qquad \sum_i r_i=0
$$

이것은 한 핸드의 칩 증감과 **다른 목적함수**다. 동일한 transition에서 뜻 없이 두 값을 합하지 않는다. 유료 상금 구조에서는 대신 순위별 상금/ICM 기반 utility 함수를 만든다.

자기 행동에 대한 RL 전이:

$$
 \tau_{i,t}=(I_{i,t},a_{i,t},r_{i,t},I_{i,t+1},d_{i,t})
$$

여기서 `I_{i,t+1}`은 다른 플레이어가 행동한 직후가 아니라 **같은 플레이어의 다음 의사결정 시점** 상태이다. 마지막 행동에서만 terminal reward를 부여하고 그 이전 행동들의 immediate reward는 0으로 보관한다. `EpisodeResult.rewards`는 행동하지 않은 좌석도 포함한 전원의 종료 utility를 제공한다.

## 3. RLCard 실행 예시

기존 RLCard를 사용하는 엔진을 별도로 교정한 프로젝트라면, 아래 예시는 **RLCard 순정 2인 HUNL**을 대상으로만 직접 적용된다.

```python
import rlcard
from holdem_agent_phase5 import (
    RLCardTurnDriver, RandomLegalPolicy, run_self_play,
)

env = rlcard.make("no-limit-holdem", config={"seed": 19})
# ID를 추측하지 않고 num_actions에 따라 id별 라벨만 부여한다.
labels = tuple(f"action_{i}" for i in range(env.num_actions))
driver = RLCardTurnDriver(env, action_labels=labels)
policies = {i: RandomLegalPolicy() for i in range(env.num_players)}
result = run_self_play(driver, policies, seed=19)
print("payoffs", result.rewards)
print("decisions", result.num_decisions)
print("seat_0_trajectory", len(result.for_seat(0)))
```

**주의:** RLCard의 `get_payoffs()` 단위는 이용 중인 게임 구현에 좌우된다. `RLCardTurnDriver`는 반환 값을 **그대로** reward 벡터로 전달하며 자동으로 big blind로 나누지 않는다. 코드 수준에서 확인하고 `terminal_rewards_fn`을 직접 설정한다.

### 3.1 6인/토너먼트 커스텀 엔진 연결

기존 Engine의 클래스명/메서드 시그니처가 확인되지 않았으므로 코드를 바꾸지 않고 다음 Protocol을 구현하는 **추가 어댑터**를 작성한다.

```python
from holdem_agent_phase5 import Turn, Terminal, InformationState

class ExistingEngineDriver:
    def __init__(self, existing_engine):
        self.engine = existing_engine

    def reset(self, seed: int | None = None) -> Turn | Terminal:
        # 1) 기존 엔진 reset 호출
        # 2) current_player의 관측 정보만 읽기
        # 3) 엔진 legal_actions를 mask로 변환
        # 4) InformationState(player_id=..., features=..., legal_action_mask=...)
        raise NotImplementedError("Fill using the actual Phase 1-4 engine API")

    def step(self, action_index: int) -> Turn | Terminal:
        # 1) 이전 InformationState.assert_legal(action_index)
        # 2) action_index를 기존 엔진 action 포맷으로 변환
        # 3) 기존 엔진 진행 / 종료 여부 확인
        # 4) 종료 시에는 좌석 전원의 utility로 Terminal(...)
        raise NotImplementedError("Fill using the actual Phase 1-4 engine API")
```

정확한 engine action 변환 없이 `step`을 묵시적으로 호출해서는 안 된다. Seat ID는 탈락해도 재할당하지 않는다.

## 4. No-Limit 행동 추상화

정책 출력 head를 다음 7개로 고정할 수 있다.

```python
BET_ACTION_LABELS = (
    "fold", "check", "call", "raise_half_pot", "raise_pot", "raise_double_pot", "all_in"
)
```

현재 스트리트에 이미 낸 금액을 `c`, 이번에 콜해야 할 추가 금액을 `k`, 행동 직전 pot을 `P`, 포트 배수를 `f`라 하면 후보 총 베팅 목표값은

$$
 B_{\mathrm{to}}^{(f)}=c+k+\mathrm{round}_{\mathrm{chip}}\bigl(f(P+k)\bigr),\qquad f\in\{0.5,1,2\}.
$$

이 수식은 **일관적인 raise 크기 후보**를 생성한다. 포커의 최소 레이즈/올인/재레이즈 개방 여부를 판정하는 수식으로 대체하지 않는다.

```python
from holdem_agent_phase5 import NoLimitBetContext, abstract_bet_actions

context = NoLimitBetContext(
    pot=100, committed=10, to_call=20, stack=200,
    min_raise_to=50, max_raise_to=210,
    can_fold=True, can_check=False, can_call=True, can_raise=True,
)
choices = abstract_bet_actions(context)
print(choices.mask)       # (True, False, True, True, True, False, True)
print(choices.resolve(3))  # EngineBetAction(kind='raise_to', raise_to=90)
```

`min_raise_to`, `max_raise_to`, `can_raise`는 **기존 게임 엔진이 책임지는 값**이다. 이미 최소 레이즈보다 적게 올인한 short all-in, 리오픈이 불가능한 상황, 토너먼트 유효 스택, side pot은 엔진 검증을 유지해야 한다. 이 abstraction은 중복 action 크기를 제거한다. `EngineBetAction.raise_to`는 **이번 스트리트 총 커밋 금액**(absolute total)이지 이번 액션에서 추가 지출한 칩(`raise_by`)이 아니다. 연결 시 해당 변환을 확인한다.

## 5. 정보 유출 방지와 CFR 가정

- `get_perfect_information()` / 엔진의 전체 덱 / 상대 홀카드는 정책에 넘기지 않는다.
- RLCard `state['obs']`는 플레이어 시점 observation으로 사용하고, `raw_obs`는 own `hand`, `public_cards`, `stage`와 명시된 public numeric 필드만 읽는다.
- 현재 릴리스의 RLCard 기본 adapter는 `history=()`이다. **이 상태 그대로 학습한 Tabular CFR은 perfect recall을 보장하지 않는다.** `history_fn`에 유효한 공개 액션 및 거리별(street) 분리 정보를 명시적으로 전달해야 한다.
- `infoset_key()`는 stable JSON serialization일 뿐, 별도의 게임 추상화·정보 유출 검증이 끝난 GTO information set이라는 뜻은 아니다.
- `AgentPolicy`는 학습된 tabular/Deep CFR 행동 분포, DQN/PPO action head 등과 교체할 수 있다. 학습 알고리즘 자체(CFR regret update / advantage training / PPO optimizer)는 Phase 5 인터페이스 범위 **밖**이다.

## 6. Snapshot vs Step Back

`copy.deepcopy(env)`로 생성한 스냅샷은 일부 엔진에서 기능할 수 있지만, 아래 조건이 실제 게임 엔진에서 테스트되어야 한다.

1. 카드/Dealer RNG 및 행동 추첨 RNG가 독립적으로 복제되는가?
2. 한 분기를 실행해도 다른 분기의 카드 순서와 stack이 변하지 않는가?
3. Action history / 블라인드 시점 / 토너먼트 탈락 좌석이 독립적으로 보존되는가?
4. RNG가 포함된 chance node를 포함하여 같은 action path 재생 시 동일한 결과가 나오는가?

RLCard의 `step_back()`을 사용할 때에는 `allow_step_back=True`가 필요하고, **액션 기록과 RNG의 처리까지** 별도로 검증해야 한다. `deepcopy`만으로 CFR에 적합한 chance-sampled traversal을 보장하지 않는다.

## 7. 적용 및 테스트

압축 해제 위치: **repository root (pyproject.toml 있는 위치)**. 동일한 이름의 파일이 이미 있을 경우에는 덮어쓰기 전에 충돌을 수동 점검한다.

Bash / macOS / Linux:

```bash
cd path/to/holdem-ai
unzip -n /path/to/holdem-ai-phase5-agent-kit.zip
uv run python -m pytest -q tests/test_phase5_*.py
uv run python -m pytest -q
```

PowerShell:

```powershell
Set-Location path\to\holdem-ai
Expand-Archive -Path path\to\holdem-ai-phase5-agent-kit.zip -DestinationPath .
uv run python -m pytest -q tests
```

첫 번째 명령은 새 기능 테스트, 두 번째는 기존 1~4 Phase 포함 전체 회귀 테스트다. 적용 전 `git status`에서 다른 작업 변경 사항을 확인하고, 완료 후 `git diff --stat; git status`를 확인한 뒤 커밋한다.

```bash
git status
uv run python -m pytest -q
# tests pass 후
git add holdem_agent_phase5 tests/test_phase5_*.py docs/PHASE5_AGENT_INTERFACE.md scripts/phase5_rlcard_smoke.py
git commit -m "feat(agent): add phase 5 observation policy reward and self-play interfaces"
git push
```

**중요:** 이 키트만 테스트하여 프로젝트 전체 `113 passed, 1 xfailed`의 회귀 통과를 단정해서는 안 된다. RLCard 실물 통합 테스트는 현재 환경에 라이브러리가 없어 skip 되었으며 실제 프로젝트 `uv run`에서 재확인해야 한다.

## 8. Phase 5 종료 조건 (검증 체크리스트)

- [x] Pure-python Agent / Observation / Reward 인터페이스 및 isolated tests
- [x] Legality mask, RNG deterministic random baseline, 6-seat trajectory fixture
- [x] 플레이어 private data whitelist 테스트
- [x] `deepcopy` smoke / synthetic RNG cloning test
- [ ] 프로젝트 기존 엔진 API와 직접 연결한 actual adapter
- [ ] 실제 RLCard 라이브러리를 이용한 통합 테스트 (현재 skip)
- [ ] 6-max 토너먼트 **실제** rollouts + action mapping 검증
- [ ] 원본 113개 테스트 전체 회귀 실행
- [ ] perfect recall history 공급 및 게임 정보 유출 audit
- [ ] real-engine deepcopy / step_back RNG 정확성 검증
- [ ] CFR / Deep CFR / RL 학습 파이프라인 구현 (후속 Phase)
