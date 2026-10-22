# Kimodo 조준 모션 → K1/HX5 MuJoCo 확인

새 Kimodo 모션 생성, K1 변환, MuJoCo 참조 재생·물리 평가를 완료했다. **참조 자세 확인은 가능하지만 QA 및 물리 추종은 실패**했다. 범용 리타게팅 완성이나 실물 동작 성공으로 판단하지 않는다.

[원본→K1→물리 비교 화면](gallery_pistol_aim_002.html) · [3분할 MP4](videos/pistol_aim_002/source_reference_physics.mp4) · [정량 결과](pistol_aim_summary.json)

## 생성 및 관찰

공식 Kimodo-G1-RP-v1 + 고정 revision의 공식 Llama3/LLM2Vec 인코더를 사용했다. seed73111, 6초/30Hz/180frames, denoising100, post-processing 활성화, CFG separated[2,2]. 새 프롬프트는 제자리에서 양팔을 들어 양손으로 정면 조준 후 유지하는 동작이다. 원본의 생성 설정·NPZ·CSV·hash는 `data/generated/pistol_aim_002`에 보존했다. 첫 시도001은 converter 반환값을 Tensor로만 가정한 export 오류였고, 같은 프롬프트·seed로 재실행했다. 실패 로그를 보존했으며 품질에 따른 seed 선별은 하지 않았다.

원본과 K1의 10-frame sheet를 검토했다. 양팔을 앞으로 들어 유지하는 형태는 확인했다. 정확한 양손 지지, 손목 조준 방향, 권총 파지는 검증하지 않았다. 이번에는 권총 물체를 장착하지 않았고 손가락40관절은 기존 일정 목표를 유지했다. 프롬프트에만 권총이 등장하며 생성 모델이 소품을 생성하는 것은 아니다.

## 공통 경로 결과

모션 이름은 `custom`, 기존 공통 v1 task-space IK와 balance 제어기를 사용했다. 조준 전용 관절 궤적·보정 규칙을 추가하지 않았다. 원본 시간축·root 방향을 보존해50Hz/299frames로 변환했다. 걷기 전용 compiler는 적용하지 않았다.

| 항목 | 결과 |
|---|---|
| 생성 파일 형식 | PASS |
| K1 reference QA | FAIL |
| 최대 바닥 침투 | 30.69mm (허용20mm 초과) |
| 최대 자기 충돌 깊이 | 26.23mm (허용10mm 초과) |
| 최대 sampled foot IK 위치 오차 | 0.623mm |
| 최대 sampled arm IK 위치 오차 | 101.54mm |
| 관절 범위·속도·지지발 slip·foot IK gate | PASS |
| MuJoCo 실제 물리 성공 | **0/3** |
| 물리 종료 | 3회 모두1.56초, 낙상 |
| 계획된 모션+마지막 hold | 6.96초 |

3회는 독립 생성3개가 아니라 생성 clip1개에 reset perturbation seeds1000/1001/1002를 적용한 결과다. 실제 root/wrist RMSE와 모든 초기 상태·rollout은 `runs/pistol_aim_002`에 있다. 독립 FK/RMSE 감사는 통과했으나 이는 기록 정합성 검사이며 물리 성공이 아니다. motor saturation은0이었고 root/wrist 추종 및 전체 길이·낙상 gate를 통과하지 못했다. 영상에서 조기 종료 이후 실제 상태는 마지막 프레임으로 유지하고 종료 문구를 표시했다.

발 IK가 정확하더라도 G1과 K1의 발 충돌 형상 차이를 공통 v1 경로가 충분히 보정하지 못했다. arm IK 오차도 커서 원본 자세의 정밀 보존을 주장할 수 없다. 바닥 침투·자기 충돌과 제어 실패의 개별 인과 기여는 이번 실험으로 분리하지 않았다. 이 결과를 성공시키기 위해 원본 자세를 임의로 바꾸거나 임계값을 완화하지 않았다.

## 재현과 다음 목표

생성 설정은 `configs/pistol_aim_002.json`. 기존 label은 덮어쓰지 않으므로 반복 시 새 label을 사용한다.

```bash
bash workspaces/k1_kimodo_hx5/run.sh generate --config <새_label의_설정.json>
bash workspaces/k1_kimodo_hx5/run.sh process \
  --source-csv <생성한_qpos.csv> --source-manifest <생성_manifest.json> \
  --fps 30 --label <새_label>
```

새 `process` 명령은 CSV snapshot→리타게팅→QA/물리 평가→독립 감사→영상까지 연결한다. QA/물리 실패도 진단 영상을 생성하고 최종 exit2 및 `performance_passed:false`로 남긴다. UI의 Generate 버튼에 자동 연결된 상태는 아니며 범용 추종 제어기를 완성한 것도 아니다.

최종 목표는 범용 생성·리타게팅이다. 다음은 이름에 의존하지 않는 접촉 분석, foot geometry·전신 충돌·팔/손목 자세 보존을 함께 푸는 공통 리타게터와 생성 UI 자동 연동이다. 이 조준 clip을 새로운 실패 회귀 사례로 보존한다. 기존 성공4clips 및 평가 기준·로봇 모델·실행 중 Kimodo UI는 유지했다.
