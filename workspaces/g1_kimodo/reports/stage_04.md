# Stage 04 — Isaac Lab 추종 학습·평가·export

- 날짜: 2026-10-21 (Asia/Seoul)
- 판정: **부분 완료 — 서기·오른손 흔들기 통과, 걷기 실패. 4단계 전체 미통과.**
- 같은 최종 후보 정책의 nominal 평가: 서기 **20/20**, wave **20/20**, walk **0/20**. 5단계로 진입하지 않습니다.
- 학습 run: `runs/20261021_125809_049195_stage04_multimotion_residual_train_v11/`; 평가 run: `runs/20261021_130836_931130_stage04_multimotion_eval_v11/`.
- [정량 요약](stage_04_summary.json) · [전체 독립 검사](stage_04_readiness.json) · [실제 rollout 비교 영상](stage_04_gallery.html).

## 목표와 현재 산출물

workspace 전용 G1 자유 베이스 reference tracking task, RSL-RL PPO 학습·재개, 명시적 reset 평가, CPU/GPU TorchScript export, 독립 수치 검증과 실제 rollout 영상까지 구현했습니다. 목표 자세를 매 프레임 상태에 덮어쓰지 않고, 500Hz 물리 stepping과 토크로 로봇을 움직입니다. 학습은 canonical reference의 **train8개(stand3/wave3/walk2)**만 사용했습니다. 원본15개·정제14개·탈락walking train33을 보존하며 final 생성 seeds303/404는 사용하지 않았습니다.

[후보 export](../exports/stage04_candidate_v11/README.md)는 두 MLP 분기와 완결된 관측/제어 계약·fixture·checksum을 포함합니다. 전체 동작용 `deployable`은 false입니다. 이 후보를 걷기 또는 MuJoCo sim-to-sim 통과 정책으로 표시하지 않습니다.

## 측정 결과

각 동작 validation2 clips × reset seeds1000~1009 =20회, 총60회입니다. frame0에서 시작해 전체 reference+추가1초 hold 동안 평가하고, 조기 종료한 episode도 분모에 포함합니다. reset의 실제 root pose/twist·29 joint state·previous action을 모든 case에 저장했습니다.

| 지표 | 서기 | wave | 걷기 | 기준 |
|---|---:|---:|---:|---|
| 성공 | 20/20 | 20/20 | 0/20 | 동작별 ≥90% |
| 실제 평가 길이 | 7.22s | 8.22s | 2.66~2.86s 후 낙상 | 전체 clip+1s |
| 최대 episode joint RMSE | 0.00725rad | 0.04228rad | 0.08437rad | ≤0.35rad |
| 최대 body relative RMSE | 0.00381m | 0.01780m | 0.12322m | ≤0.10m |
| 최대 world root RMSE | 0.00504m | 0.00787m | 0.30515m | ≤0.20m |
| 최대 world hand RMSE | 0.00628m | 0.04084m | 0.30401m | ≤0.12m |
| 최대 torque saturation 비율 | 0% | 0% | 0.226% | ≤1% |

걷기의 오차는 실패 prefix에 대한 값이며, 종료 이후를 성공으로 보간하지 않았습니다. 낙상·full-clip·body/root/hand gate가 실패했습니다. 큰 모터 토크 포화가 주원인이라고 판단하지 않습니다.

- 관측 frame/PD 독립 수학 검사 **8/8 PASS**: SciPy 회전 대조, quaternion 부호, target/effort clip, rated 속도 이상 outward torque 차단 및 braking 유지.
- 별도 수평 낙하 fixture **3/3 PASS**: 실제 ground-filtered torso 힘 최대 약1877N, physics substep 동안 target 유지, finite 상태.
- 최종 단계 검사 **20/21 PASS**. 유일한 미통과 항목은 동작별90% 성공 조건입니다. 60개 episode의 source qpos에서 NumPy joint/root RMSE, MuJoCo FK body/hand RMSE와 포화 횟수를 별도 재계산해 저장 지표와 대조했습니다. FK 차이는1mm, joint/root1e-5 단위 tolerance 안입니다.
- 보존한 balance actor의 모든 parameter를 원 checkpoint와 CPU에서 비교해 **최대 차이0**입니다.
- 실제 관측2466개에서 TorchScript GPU action 오차0, CPU 대 학습 GPU 최대 오차 **1.90735e-6**로1e-5 목표 통과입니다.

## task·정책·환경

실행 코드는 `tracking/env.py`, `tracking/core.py`, `tracking/initialize.py`, `tracking/policy.py`, `scripts/track.py`입니다. 구체 계약은 [stiff profile](../configs/tracking_contract_stiff.json)과 평가 run의 `export/contract.json`입니다. 기존 `motion_contract.json`은1단계 provenance를 보존하고, 새 계약이 base SHA256을 참조합니다. export 계약에 null 필드는 없습니다.

194차원 관측은 body-frame root 선속도/각속도/중력, reference 대비 joint 오차, joint 속도, 이전 clipped action, body-frame root 위치/방향 오차, 0/0.1/0.3초 joint lookahead, reference 선속도와 phase입니다. 고정 field scale 후 [-20,20] clip이며 empirical normalization은 없습니다. history1, reset previous action0, source29관절 순서를 실제 Isaac 이름 permutation으로 연결합니다.

`q_target = clip(q_reference + .25 * clip(action,-1,1), joint_limits)`를50Hz에서 갱신하고10 physics steps 동안 유지합니다. 외부 `kp*(q_target-q)-kd*qd`를500Hz로 계산해 effort clip과 rated 속도 이상 outward torque suppression을 적용합니다. PD calibration은 hip pitch/roll160/6, knee200/8, ankle120/5, waist pitch/roll120/5(kp/kd)이며 다른 gain과 물리 effort·velocity·joint limit은 유지합니다. source G1 자산·질량·관성·collision hash는 변경하지 않았습니다.

actor는 검증된 balance MLP256/128/128을 고정한 분기와 동일 구조의 trainable residual MLP를 합칩니다. critic은256/128/128, 모두ELU입니다. 최초 balance actor는 train clip의 quasistatic 하중과 synthetic 작은 각도 feedback으로 초기화한 뒤 PPO로 보정했습니다. stand PPO에서는 balance 입력 열만 학습해 자세 입력 shortcut을 막았습니다. 이 analytical teacher는 추론/export 때 호출하지 않습니다. 전체 후보의 residual MLP는 zero-output에서 시작해 세 동작을 학습합니다. 순수 from-scratch PPO나 공식 Kimodo tracking 정책을 재현했다고 주장하지 않습니다.

최종 residual PPO: seed42,128env,80 추가 updates(누적 iteration84), horizon32,5epochs,4minibatches, adaptive KL0.01, gamma.99, GAE.95, entropy.002, 시작 std.10. walking/stand/wave sampling weights3/1/1, frame0 reset입니다. tracking reward는 joint/body/root/hand 지수형 가중치2/2/1/1과 action 크기·변화 penalty, 종료 penalty입니다. 학습만 큰 root/body 오차로 조기 recovery reset하며 nominal 평가는 이를 사용하지 않습니다. 학습은 약555.6초였고 saved actor parameter 최대 변화0.07169, finite 검사를 통과했습니다.

환경: 기존 Isaac Lab release2.3.2(source package0.54.2), Isaac Sim5.1.0.0, torch2.7.0+cu128, RSL-RL3.1.2, NumPy1.26.0. QA 환경은 workspace MuJoCo3.5.0 / NumPy1.26.4 / SciPy1.17.1입니다. 공유 Isaac 환경에 패키지를 설치하지 않았습니다. 다른 workspace의 `rbq10_wmp/train_paper.py`가 GPU 약13.5GiB를 사용 중이며 이 작업은 변경하지 않았습니다. 소규모 task의 PhysX buffer를 줄여128env를 실행했습니다. torch allocator peak 약43.5MB는 전체 PhysX VRAM과 구분합니다.

## 실패 원인과 개선 평가

초기 reference PD만으로는 서기0/20이었습니다. cold-start PPO checkpoint50도0/20, support bias 초기화는 생존 시간을2.12~3.22초로 늘렸지만 실패했습니다. initial teacher 학습이 previous action·절대 자세 입력의 상관관계에 의존하는 문제를 발견해 independent/zero history 샘플과 입력 열 제약으로 수정했습니다. analytical controller 자체는 별도 PhysX 서기20/20이었으나 learned-policy 성공 분모에 포함하지 않았습니다. 입력 제약을 PPO 동안에도 유지한 v9 MLP가 서기·wave 각각20/20을 통과했습니다.

초기 contact filter는 floor 부모 Xform을 가리켜 검출하지 못했습니다. 상면Z=0의200m static box와 실제 collision prim `/World/ground/geometry/mesh`를 사용하고 낙하 fixture로 수정 확인했습니다. 바닥 friction.8·restitution0, contact offset.002/rest0, TGS16/4입니다. contact-blind 초기 실행은 nominal 통과 근거로 사용하지 않았습니다. OOM·dtype·초기 checkpoint logger 오류와 Kit cleanup도 실패 run에 남겼고, 실행 supervisor가 report/traceback 뒤 cleanup을 제한합니다.

걷기는 지지발 전환 중 좌우로 무너집니다. 기존 balance actor를 고정하고 residual PPO를80 updates 학습했지만 같은60-case 평가에서0/20이 유지됐습니다. 기존 서기·wave 회귀는 없었습니다. [v9 baseline](stage_04_v9_baseline.json)과 각 run/checkpoint·TensorBoard·로그를 보존합니다.

[reference wrench 진단](stage_04_reference_wrench_diagnostic.json)에서는50Hz 차분 가속도로 정확한 reference를 실행하는 데 필요한 ground wrench/COP를 계산했습니다. walking clip별 약8.8~25.4% frame에서 요구 COP가 추정 지지 영역 밖5mm 이상이며, 일부 clip은 순간 요구 마찰 비율2.54~4.38과 관절 가속도약313~541rad/s²가 나옵니다. **차분 민감도가 있고 허용 추종 오차 안의 다른 실행 궤적까지 불가능하다는 증명은 아닙니다.** 다만 단순 kinematic gate만으로 동적 적합성이 보장되지 않는 원인 근거입니다. 성공 기준을 완화하지 않았습니다.

단순 joint smoothing은 침투/slip 등 기존 core QA를 만족하는 경우에도 foot target을최대수cm 바꿔5mm IK budget을 넘었습니다. 낮은 pelvis·무릎 최소 flexion·acceleration regularization을 적용한 v12와 lateral foot scale을 추가한 v13은 각각 walking5개 모두 전체 QA에서 탈락했습니다. [v12 로그](stage_04_conditioning_v12.log) · [v13 로그](stage_04_conditioning_v13.log) · [smoothing 측정](stage_04_smoothing_probe.json). **이 안들은 채택하지 않았고 canonical14개와 evaluation 임계값은 그대로입니다.**

## 재현·검토

```bash
bash workspaces/g1_kimodo/run.sh validate-tracking-math
bash workspaces/g1_kimodo/run.sh verify-frozen-balance workspaces/g1_kimodo/runs/20261021_125809_049195_stage04_multimotion_residual_train_v11/model_84.pt
bash workspaces/g1_kimodo/run.sh assess-stage4 workspaces/g1_kimodo/runs/20261021_130836_931130_stage04_multimotion_eval_v11
# 마지막 명령은 walking 성공 조건 때문에 exit2를 반환하는 것이 현재 기대 결과입니다.
# 같은 policy·gain·reset 목록 정의로 새 평가 run 생성:
bash workspaces/g1_kimodo/run.sh track --label reevaluation --mode evaluate --motions stand wave walk_slow --checkpoint workspaces/g1_kimodo/runs/20261021_125809_049195_stage04_multimotion_residual_train_v11/model_84.pt --timeout-s 600
```

평가 run에는60개 initial state·episode metric, 전체 episode qpos/포화 trace, CPU/GPU action fixture, source/config snapshot과 SHA256이 있습니다. 여섯 seed1000 case의10개 균등 시점 frame을 실제 검토했습니다. 서기와 오른손 wave 의미·추종을 확인했고, 걷기 lateral fall도 영상으로 확인했습니다. 연속 전 구간을 시각 검토했다고 주장하지 않습니다. 수치 검사는60회 전체 측정 구간에 수행했습니다. 영상은 **Isaac dynamics에서 저장한 상태를 MuJoCo FK로 렌더링한 재구성**이며 MuJoCo 정책 dynamics/sim-to-sim 결과가 아닙니다.

## 다음 작업

4단계를 계속합니다. 지지발 전환 전에 무게중심을 옮기는 COM/ZMP trajectory와 foot/COM 제약을 함께 만족하는 시간 전체 IK를 우선 검토합니다. 현재의 frame별 IK+root support-center smoothing을 단순히 더 학습하는 접근은 우선순위를 낮춥니다. 새 reference는 원본/현 canonical과 별도 version으로 만들고 같은 QA·split·후보 분모·영상 검토를 거친 뒤에만 채택합니다. 이후 걷기 PPO와 같은 정책의60-case 회귀를 다시 수행해 동작별≥90%를 만족해야4단계 완료입니다. 5단계는 그 정책·가중치를 바꾸지 않고 MuJoCo에서 검증합니다.
