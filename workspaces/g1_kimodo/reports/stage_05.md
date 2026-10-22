# Stage 05 — MuJoCo sim-to-sim 프레임워크와 전이 baseline

- 날짜: 2026-10-21 (Asia/Seoul)
- 판정: **프레임워크 연결 완료 / 전체 성능 gate 부분 완료**.
- 사용자 지시: 걷기 성능 개선에 앞서 MuJoCo를 연결해 하나의 프레임워크를 먼저 완성한다. stage4 walking 실패를 유지한 채 stage5를 실행했다.
- [비교 영상](stage_05_gallery.html) · [정량 결과](stage_05_summary.json) · [재현 절차](../docs/SIM2SIM.md) · [baseline package](../exports/sim2sim_v11_baseline/README.md)

## 목표와 변경

Kimodo canonical reference → Isaac Lab PPO/평가/export → **동일 TorchScript 정책의 실제 MuJoCo 자유 베이스 물리 평가** → 같은 초기 상태의 정량·영상 비교까지 연결했다. MuJoCo에서 재학습하지 않았다. 앞으로 reference/정책 후보를 이 프레임워크에서 동일 조건으로 비교한다.

새 `sim2sim`, `assess-stage5`, `review-sim2sim` 명령과 실행 가이드를 추가했다. 새 Isaac evaluation export에도 checksum manifest가 자동 생성되어 bridge 입력으로 바로 사용할 수 있다. 원본 모션·canonical14개·기존 v11 가중치·평가 임계값을 수정하지 않았다.

run:

- Isaac: `runs/20261021_130836_931130_stage04_multimotion_eval_v11/`
- MuJoCo nominal: `runs/stage05_mujoco_v11_box_compiled/`
- MuJoCo timestep refinement: `runs/stage05_mujoco_v11_refined/`
- 같은 가중치: `exports/stage04_candidate_v11/policy.pt`
- 정책 SHA256: `bfc2ba749660c2c7111d89fca02c34d7fbe3af2ca52efbe4faf313bf7457fd7e`
- source checkpoint SHA256: `b33509dfa7191927591db1c5cfaccc2f0fa045c2f3699e1989ef83f3793508c5`
- dataset: `data/processed/20261021_115621_761537_stage03_curated/final/dataset.json`
- frozen evaluation SHA256: `f5df6cd6be95b6254b4771c497a8bba0d615d9c80544ebbad34c636122f82c31`

세부 파일 hash·joint maps·버전·solver/contact 값은 각 run의 `resolved.json`/`provenance`에 보존했다. 공유 Isaac 환경과 다른 workspace는 변경하지 않았다. 현재 작업은 모두 종료됐다.

## 측정 결과

validation6clips × reset10seeds = **60episodes/backend**. seed 숫자를 다시 생성하는 대신 Isaac에 저장된 root pose, world link twist,29joint pose/velocity, previous action을 그대로 사용했다. 전체 clip와1초 final hold가 성공 조건이다.

| 동작 | Isaac | MuJoCo 2ms | MuJoCo 1ms | 성공률 하락 |
|---|---:|---:|---:|---:|
| 서기 | 20/20 | 20/20 | 20/20 | 0pp |
| 손 흔들기 | 20/20 | 20/20 | 20/20 | 0pp |
| 느린 걷기 | 0/20 | 0/20 | 0/20 | 0pp |

서기361steps=7.22s, wave411steps=8.22s를 전부 추종했다. 걷기는 MuJoCo100~121steps=2.00~2.42s에 옆으로 넘어졌다. Isaac의133~143steps=2.66~2.86s보다 이르다. 실패한 걷기도 전체 분모에 남겼다. 성공률 하락0pp는 걷기 전이가 성공했다는 뜻이 아니다.

nominal MuJoCo의 **episode별 RMSE 최대값**:

| 동작 | joint rad | root-relative body m | world root m | hand m | torque 포화 최대 비율 |
|---|---:|---:|---:|---:|---:|
| 서기 | 0.00749 | 0.00459 | 0.00551 | 0.00720 | 0 |
| wave | 0.04223 | 0.01785 | 0.00843 | 0.04074 | 0 |
| 걷기 | 0.08098 | 0.15098 | 0.32404 | 0.35573 | 0.001415 |

걷기20개 모두 full-clip, no-fall, body/root/hand 오차 조건에 실패했다. 모든60episode의 finite·joint·torque saturation gate는 통과했다. 걷기 근본 실패는 양쪽에서 재현되며, 엔진 차이만으로 해결된 결과는 없다.

| 검증 | 측정 | 판정 |
|---|---|---|
| 같은 export 가중치, 데이터, 평가 계약, 명시 초기 상태 | 비교·독립 검사13/13 | PASS |
| 기록된 Isaac reset 관측과 MuJoCo 관측 | 대표6개의194차원 최대 오차0 | PASS |
| GPU actor vs CPU TorchScript | 저장2466관측, 최대1.90735e-6 | PASS |
| 비대칭 quaternion/world twist/history 관측 수식 | 무작위256상태, NumPy vs Torch 최대2.38419e-7 | PASS |
| PD clipping/target/속도 guard | 무작위256상태, NumPy vs 공유 Isaac Torch torque 차이0Nm | PASS, 명령 계산 범위 |
| 저장 trace로 독립 RMSE·포화 재계산 | Isaac 최대RMSE 차이2.32e-7, MuJoCo5.55e-17 | PASS |
| MuJoCo timestep2ms→1ms | 60개 성공 판정 변경0, 종료 시각 차이 최대0.02s | 측정 완료 |
| 모든 동작 nominal≥90% | 걷기0/20 | FAIL |
| Isaac 대비 성공률 하락≤10pp | 모든 동작0pp | PASS |

MuJoCo nominal 약38.1초, refinement 약70.2초의 CPU wall time이었다. 이 시간은 현재 머신의 단일 실행 측정이며 범용 benchmark가 아니다. 영상3개 및 각10-frame sheet를 확인했다. 이는60개 영상 전체의 수동 검토를 뜻하지 않는다.

## 제어·접촉과 수치 검증 범위

- 관측194차원/행동29차원, root-link body frame, wxyz, history1, reset history0, 고정 scale/clamp, 같은 lookahead0/5/15frames.
- target=`clip(q_ref + 0.25*clip(action,-1,1), joint_limits)`를50Hz로 갱신하고10개의500Hz PD tick 동안 고정한다. torque clipping 및 rated-speed 초과 시 outward torque 억제를 재현했다.
- physics만 nominal2ms/refinement1ms로 바꾸고 정책50Hz·PD500Hz는 유지했다. MuJoCo qpos/qvel은 reset 외에 쓰지 않았다.
- MuJoCo ground는 compile 전 box200×200×0.2m/topZ0, sliding friction0.8로 구성했다. canonical robot asset hash는 유지했다. contact soft constraints/solver와 PhysX TGS는 차이가 있으므로 동등하다고 주장하지 않는다. 성능을 보고 마찰을 조절하지 않았다.
- MuJoCo refinement에서 episode RMSE 차이 최대: joint0.00118rad, body0.00373m, root0.01120m, hand0.00950m. 낙상 걷기에서는 궤적의 정확한 시점이 수치 설정에 민감할 수 있다. 성공/실패 판정은 유지됐다.
- PhysX 실제 applied-effort 계측과 Isaac timestep refinement는 **미측정**이다. 현재 PD fixture는 공유 Isaac 계산 코드와 별도 NumPy 계산을 비교했으며, 실제 PhysX 적용 torque를 측정했다고 부르지 않는다.

초기 구현 진단 `runs/stage05_mujoco_v11_nominal/`은 compile 후 plane→box 형상을 바꿔 collision metadata가 맞지 않았으므로 유효 평가에서 제외했다. `INVALID_DIAGNOSTIC.json`에 원인과 대체 run을 기록했다. compile 전에 box를 구성한 canonical nominal run만 위 결과에 사용했다. 진단 실패를 지우거나 성능 결과로 섞지 않았다.

## 남은 조건과 다음 작업

**프레임워크 연결은 완료**했고 기존 정책의 서기·wave가 MuJoCo에서도 유지된다. 걷기 실패 때문에 stage4/5 전체 성능 gate와 배포 가능 상태는 통과하지 않았다. 다음 “개선”은 지지발 전환·COM/ZMP 및 시간 연속성을 반영한 reference 개선을 우선하고, QA→Isaac 학습/export→동일 초기 상태 MuJoCo 비교를 반복한다. 기존 서기·wave20/20은 회귀 기준이다.

실제 PhysX torque fixture, Isaac timestep refinement, robustness sweep, 미사용 final 생성 seeds303/404 평가는 별도 남아 있다. final을 튜닝에 사용하지 않았다. v12/v13 walking conditioning은 QA 실패로 계속 미채택 상태다.
