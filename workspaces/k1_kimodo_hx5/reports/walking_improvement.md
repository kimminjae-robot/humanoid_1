# 걷기 개선 — 보수적인 보행 전신 제어

- 날짜: 2026-10-21
- 판정: **선택한 짧은 보폭·느린 걷기의 reference QA 및 MuJoCo 물리 평가 통과**.
- [비교 영상](gallery.html) · [추가 걷기 영상](gallery_walk_validation202.html) · [정량 JSON](walking_improvement_summary.json) · [이전 실패 보고서](baseline.md)

## 결과와 적용 범위

| 항목 | 이전 | 개선 후 |
|---|---|---|
| 걷기 성공 | 0/3, 생성 clip1개 | **6/6, 생성 clip2개 × reset3개** |
| 완료 시간 | 약2.66초에 낙상 | clip101:35.78초 / clip202:41.78초, 각각 마지막1초 hold 포함 |
| 실제 root 전진 | 전체 모션 미완료 | clip101 약0.302m / clip202 약0.368m |
| 실제 발 들기 | 낙상으로 중단 | clip101 좌3/우2회, clip202 좌3/우3회 |
| 서기 회귀 | 3/3 | 3/3 유지 |
| wave 회귀 | 3/3 | 3/3 유지 |

원본 Kimodo 걷기를 같은 보폭·속도로 재현한 결과는 아니다. **앞 방향 발 이동을30%로 축소**, 발 교대 전 COM 이동3초·발 swing3초로 재구성했다. root는 upright/heading0, 상체는 중립 자세로 두었다. 손가락은 기존 일정 목표를 유지한다. 원래의 빠른 걷기나 상체 스타일 보존은 아직 검증되지 않았다. 여러 프롬프트 일반화·하드웨어 성능을 의미하지 않는다.

## 원인과 수정

기존 G1→K1 변환은 K1 발 형상·관절 구조에 비해 보폭이 크고, 지지발 교대 전에 COM 이동이 부족했다. 이전 v2 reference는23.6mm ground penetration과82.8mm 발 IK 오차가 남았다. 별도로 reference 속도를0으로 가정하는 제어와 양쪽 발목에 동일 피드백을 주는 방식도 보행의 관성을 처리하지 못했다.

새 `improve_walk.py`는 보존된 Kimodo footstep 순서와 도착 위치를 사용한다. 짧은 contact flicker를 하나의 swing으로 병합하고, K1 발 구체에 맞춘 평평한 sole pose·55mm swing clearance·quintic 시간 보간·이륙 전 COM 이동을 구성한다. 발 pose와 COM을 함께 푸는 K1 IK로 reference를 만든다. 원본 G1 파일을 수정하지 않았다.

reference만 개선한 v3는 모든 QA를 통과했지만 기존 balance 제어에서는3.24~3.30초에 넘어졌다. 더 느린 v4도 기존 제어에서는 약4.54초에 낙상했다. 초기 hard-contact inverse-dynamics QP는 MuJoCo의 soft contact와 결합했을 때 발 자세/속도가 불안정하고 infeasible 상태가 발생해 채택하지 않았다. 이 후보의 코드·실패 로그를 보존했다.

최종 제어는 **centroidal 지면반력 QP + 몸체 torque PD**다. root 위치·속도·자세와 reference 가속도에서 원하는 base wrench를 계산하고, 계획된 접촉 발에 양의 normal force, 보수적 마찰계수0.6의 마찰 제약, sole 안의 COP 범위, yaw moment 제약을 적용해 분배한다. Jacobian으로 실제 몸체 joint torque에 반영하고 reference joint velocity를 포함한 PD를 더한다. 최종 모터 명령은 기존 effort/속도 제한을 적용한다. QP는 목표 wrench 분배이고 실제 soft-contact force와의 정확한 일치를 주장하지 않는다. 모터 토크는 마지막 clipping으로 제한한다.

물리 timestep2ms, reference50Hz, QP/몸체 PD500Hz다. 손40관절은 공식 position actuator를 유지하고 손의 실제 질량·관성·충돌을 포함했다. 축소 dynamics는 손가락 가속도를 무시하는 근사를 사용한다. 로봇 MJCF hash·물리 마찰·모터 제한·수치 평가 임계값은 바꾸지 않았다. root는 자유 베이스이며 실제 rollout에서 qpos/qvel·외력을 덮어쓰지 않는다. MuJoCo `mj_step`과 joint motor 명령으로 움직였다. Isaac 학습 정책 전이는 아니다.

## 검증

현재 canonical `data/dataset.json`은 v1 stand/wave와 새 walk101/walk202, 총4clips다. 이전 dataset은 `data/dataset_baseline.json`에 남겼다. 걷기는 동작별 자세를 reference에 맞춰 reset하고, 동일한 seeds1000~1002의0.002rad 관절 perturbation을 사용했다. **reference 초기 자세가 변경됐으므로 이전 baseline과 절대 qpos가 같다는 뜻은 아니다.** 각 episode의 실제 초기 qpos/qvel은 저장했다.

| 동작 | QA | 실제 MuJoCo 성공 |
|---|---|---:|
| stand101 | PASS | 3/3 |
| wave101 | PASS | 3/3 |
| walk101 | PASS | 3/3 |
| walk202 | PASS | 3/3 |

두 walking reference 모두 ground/self penetration·관절 limit 위반0, stance slip fraction0이었다. 모든 수치 gate는 초기 `configs/evaluation.json` 그대로다. 6개 walking episode의 최대 RMSE는 joint0.00439rad, root0.00245m, wrist0.00248m, 몸체 torque saturation fraction0이었다. 원본보다 짧고 느린 새 reference에서 계산한 값이므로 이전 빠른 reference의 오차와 같은 과제 난이도라고 해석하지 않는다.

clip101에서 선택한 제어기/변환 파라미터를 바꾸지 않고 clip202를 추가 평가했다. 이는 K1 개선 과정에서 추가한 검증 clip이며, 기존 G1 프로젝트 전체에서 전혀 사용하지 않았던 final motion이라는 뜻은 아니다. final 생성 seeds303/404는 미사용 상태다.

clip101의 같은3개 절대 초기 상태로 physics timestep을2ms→1ms로 줄이고 reference50Hz·QP/PD500Hz를 유지했다. **3/3 통과, 성공 판정 변경0**이었다. paired RMSE 차이는 최대joint3.85e-6rad/root9.37e-6m/wrist8.74e-6m였다. 반복 refinement3회를 nominal6회에 합산하지 않았다.

저장된 실제 qpos로 독립 FK/RMSE 재계산, 실제 root displacement, 실제 foot lift 횟수를 확인했다. RMSE 재계산 차이는 최대1.39e-17이었다. 500Hz 토크 포화 event 전체를 저장된50Hz qpos만으로 독립 재구성한 것은 아니다. clip101과202의10-frame 비교 sheet도 검토했다. camera가 로봇을 따라가므로 전진 거리는 world root 좌표로 계산했다.

## 실행·산출물

- 기본 회귀: `runs/walking_improved_v4` (stand/wave/walk101,9episodes).
- 추가 clip: `runs/walk_validation202` (3episodes).
- 적분 refinement: `runs/walking_v4_refined` (walk101,3episodes).
- reference: `data/walk_v4`, `data/walk_v4_validation202`.
- 독립 감사: 각 run의 `readiness.json`.
- 실제 초기 상태·reference·rollout: 각 episode NPZ와 `result.json`.
- 후보 이력: `walk_v3_*`, `walk_v4_balance`, `walk_v4_motion_balance` 및 관련 logs. 중단된 QP 후보는 성공률 분모로 섞지 않고 실행 실패로 남겼다.

QP solver OSQP1.0.4 및 작은 의존성은 이 workspace의 `.deps`에만 설치했다. 기존 가상환경과 공유 Isaac 환경을 변경하지 않았다. 핀은 `requirements/walking_solver.txt`, 채택된 evaluator/제어기/자산·평가 설정은 각 run의 `provenance`와 checksum에 보존했다. 기존 Kimodo UI는 유지했다.

다음 개선은 보폭 또는 속도를 한 번에 한 축씩 늘리거나 Kimodo 상체 스타일을 복원하는 것이다. 두 걷기 clip과 stand/wave 회귀를 유지하면서 평가한다. 현재 성공은 느린 보수적 보행 범위이며, 원래 Kimodo 속도로의 복원·강건성·실물 검증은 남아 있다.
