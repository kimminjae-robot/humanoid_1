# 경례 물리 추종 개선

선택한 경례를 **MuJoCo 물리 실행3/3 성공**으로 개선했다. 기존6초 모션과 마지막1초 유지까지 각6.96초를 완료했다. reference QA 전체 통과, physics timestep2ms→1ms 재검증도3/3 통과했다. [최종 비교 영상](gallery_salute_improved.html) · [정량 결과](salute_improvement_summary.json) · [이전 실패](salute.md).

## 왜 자세 재생은 되고 물리 실행은 넘어졌나

참조 재생은 프레임마다 관절과 root를 지정하므로 중력·토크·접촉에 의해 자세가 무너지지 않는다. 기존 참조는 보기에는 경례였지만 오른발을 옆으로 이동시키고 root가 좌우로 약17cm 움직였다. 오른발의 접촉 추정도 전체의 약80%뿐이었다. K1 발 형상과 접촉에 맞지 않는 바닥 침투, 손가락과 골반의44.24mm 자기 침투가 함께 있었다. 원래 v1 IK는 위치 오차를 줄이는 경로이며 이런 자세의 동역학적 실현성을 보장하지 않는다.

원본 참조에서 제어만 지면반력 QP로 바꿔도0/3이었다. 반면 발을 고정하고 COM을 지지 중심에 맞추자 기존 balance로도 전체 길이3/3을 완료했다. 따라서 이번 실험에서는 지지 자세 보정의 효과가 컸다. 손 충돌을 줄인 후에는 팔 복귀 중의 추종 오차와 균형 제어가 추가 개선 대상이었다. 모든 실패 원인이 하나라고 단정한 결과는 아니다.

## 채택한 보정과 제어

- 양발을 K1 중립 위치에 고정하고 발바닥을 평평하게 맞췄다. sole 초기 여유1mm, root 높이 목표0.77m, COM 수평 위치는 양발 중간이다. root 방향은 기존 K1 참조를 유지했다.
- 실제 MuJoCo 충돌 형상의 거리를 사용해 팔 IK를 조정했다. 손을 내리는 구간에는 손목이 몸 옆에서 충분히 떨어지도록 부드러운 여유를 추가했다. 손가락은 일정 목표를 유지하며 실제 관절·질량·충돌을 그대로 포함한다.
- 기존 걷기용 centroidal 지면반력 QP와 reference 속도를 포함한 torque PD를 경례에도 명시적으로 적용했다. `--controller centroidal`은 동작 이름에 의존하지 않고 입력의 접촉 계획을 사용한다. 모터 토크/속도 제한, 마찰, 중력, 충돌, 평가 기준은 바꾸지 않았다.

**보정 범위에는 작은 수치 조정 이상의 변화도 있다.** 원본 K1 참조 대비 root 이동 수정 최대15.67cm, 팔을 내리는 구간의 오른손 torso-local 경로 수정 최대15.97cm, 팔 관절 수정 최대0.55rad(31.5°)다. 원래의 발·골반 이동과 손을 몸 가까이 내리는 경로를 그대로 재현한 결과는 아니다. 경례 유지 구간1.4~4.4초의 오른손 torso-local 위치 변화는 최대0.0082mm였다. 이 수치는 이전 K1 리타게팅 대비이며 G1 원본과의 절대 자세 일치를 뜻하지 않는다. 정확한 손바닥·손가락 경례 형태는 별도 검증하지 않았다.

## 검증 결과

| 항목 | 이전 | 최종 |
|---|---|---|
| reference QA | 자기 충돌 FAIL | 전체 PASS |
| reference 바닥/자기 침투 | 19.24/44.24mm | 0/0mm |
| 실제 물리 성공 | 0/3,2.00~2.04초 낙상 | **3/3,각6.96초 완료** |
| 최대 episode joint RMSE | — | 0.00930rad |
| 최대 episode root RMSE | — | 3.10mm |
| 최대 episode 양손목 RMSE | — | 5.71mm |
| 실제 자기 침투 최대 | 이전500Hz 미기록 | 약0.76mm |
| 몸체 토크 포화 비율 최대 | — | 약0.050% |
| timestep1ms 재검증 | — | 3/3,자기 침투 최대0.86mm |

실제 자기 침투는 각 물리 step 전 contact에서 기록한 최대값이다. 50Hz 저장 상태를 이용한 별도 FK/접촉 감사도 수행했으며 이 별도 감사만으로500Hz peak를 독립 검증했다고 주장하지 않는다. root는 자유 베이스이고 물리 실행 중 root/qpos/qvel을 지정하지 않았다. 외부 지지대·root actuator 없이 실제 joint torque와 `mj_step`으로 움직였다. 이것은 MuJoCo 제어 검증이며 Isaac 정책의 sim-to-sim 결과는 아니다.

nominal은 생성 clip1개×초기 perturbation3개(seeds1000/1001/1002)다. 새 모션을 여러 개 생성해 선별하지 않았다. timestep 재검증은 nominal과 동일한 절대 초기 상태를 사용했고 성공 판정 변경0, paired wrist RMSE 차이 최대0.233mm였다. 재검증3회를 nominal3회에 합쳐 일반화 성공률로 표시하지 않는다. 기존 stand/wave는 각각3/3을 유지했고 이전 채택 run과 초기 상태·지표·성공 판정이 정확히 같았다.

원본·최종 참조/물리10-frame sheet를 검토했다. 오른손을 이마 부근으로 올려 유지한 뒤 내리고, 끝까지 서 있는 흐름을 확인했다. 영상은 seed1000 대표 사례다.

## 조정 이력

| 참조/제어 | QA | 실제 성공 | 해석 |
|---|---|---:|---|
| raw / balance | FAIL | 0/3 | 초기 실패 |
| raw / centroidal | FAIL | 0/3 | 제어만 바꿔도 실패 |
| fixed feet v1 / balance | FAIL | 3/3 | 지지 보정 효과, 손 충돌 잔존 |
| v2 / balance,centroidal | PASS | 각3/3 | 실제 손·골반 접촉 잔존 |
| v3 / centroidal | FAIL | 3/3 | 충돌 여유 확대 시 속도 peak 실패, 미채택 |
| v4 / balance,centroidal | PASS | 각3/3 | 실제 손·골반 접촉 잔존 |
| v5 / balance | PASS | 3/3 | 여유만 확대해도 실제 접촉 해결 안 됨 |
| v6 / balance | PASS | 1/3 | 손목 경로 변경 후 팔 복귀에서2회 낙상 |
| **v6 / centroidal** | **PASS** | **3/3** | **최종 채택, 실제 침투0.76mm** |

각 행의3회는 같은 생성 clip의 reset 반복이다. 모든 후보와 초기 실패는 보존했다. 단순 어깨 벌림 offset sweep도 확인했지만 이것만으로는 충분하지 않아 채택하지 않았다. 더 큰 여유를 주는 것이 항상 더 나은 결과를 만들지는 않았다.

## 실행과 다음 범위

최종 참조는 `data/conditioned/salute_standing_v6/reference.npz`, index는 `reports/dataset_salute_standing_v6.json`, 설정·초기화 이력은 `configs/salute_stabilization.json`이다. 초기 상태·실행은 `runs/salute_standing_v6_centroidal`, refinement는 `runs/salute_standing_v6_refined`에 있다. 독립 감사는 각 run의 `readiness.json`, 자세 변화·실제 접촉 감사는 최종 run의 `fidelity.json`이다.

```bash
# 기존 label을 덮어쓰지 않도록 새 label 사용
bash workspaces/k1_kimodo_hx5/run.sh evaluate \
  --controller centroidal \
  --references "$PWD/workspaces/k1_kimodo_hx5/reports/dataset_salute_standing_v6.json" \
  --label salute_repeat_001
bash workspaces/k1_kimodo_hx5/run.sh review --run salute_repeat_001 \
  --gallery-name gallery_salute_repeat_001.html
```

기존 canonical4clips와 Kimodo UI는 유지했다. 개선 경례는 별도의 채택된 참조/실행 설정이다. 기본 generic `process`의 v1/balance가 모든 새 모션에 이번 보정을 자동 적용하는 상태는 아니다. 이번 결과는 한 경례 clip을 조정한 성공이며 범용 생성·리타게팅 완성을 의미하지 않는다. 다음에는 접촉 분석과 지지·충돌 보정을 공통 경로에 통합하고 새로운 모션으로 검증해야 한다.
