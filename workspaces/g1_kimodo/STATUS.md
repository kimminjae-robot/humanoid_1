# 현재 상태

- 기준일: 2026-10-21 (Asia/Seoul)
- **생성 → Isaac 학습·export → MuJoCo sim-to-sim 평가 프레임워크 연결 완료.**
- 0~3단계 완료. 4·5단계 성능은 부분 완료이며, 기본3종 전체 성공 기준은 아직 미통과입니다.

| 동작 | Isaac | MuJoCo |
|---|---:|---:|
| 서기 | 20/20 | 20/20 |
| 손 흔들기 | 20/20 | 20/20 |
| 느린 걷기 | 0/20 | 0/20 |

같은 v11 TorchScript 가중치·canonical reference·명시적으로 저장된60개 초기 상태를 사용했습니다. qpos 재생 없이 MuJoCo 자유 베이스·중력·접촉에서 실제 torque 제어를 수행했습니다. 정책50Hz·PD500Hz, nominal 적분2ms 및 refinement1ms 모두 평가했고60개 성공·실패 판정이 동일했습니다. 성공률 하락0pp는 걷기 성공을 뜻하지 않습니다.

[비교 영상](reports/stage_05_gallery.html) · [5단계 보고서](reports/stage_05.md) · [정량 결과](reports/stage_05_summary.json) · [실행 가이드](docs/SIM2SIM.md) · [기준 패키지](exports/sim2sim_v11_baseline/README.md).

다음은 양쪽 걷기의 지지발 전환·COM/ZMP reference 개선 → QA → Isaac 학습/회귀 → 동일 정책 MuJoCo 재평가입니다. 이제 “개선” 요청은 이 전체 프레임워크로 전후 비교합니다. MuJoCo는 걷기2.00~2.42초에, Isaac은2.66~2.86초에 넘어집니다. 기존 서기·wave 성능을 함께 보존해야 합니다. robustness/final 평가, 실제 PhysX applied-torque fixture, Isaac timestep refinement는 미측정입니다.

이번 평가는 CPU 작업으로 종료됐으며 GPU 영상 생성도 종료됐습니다. 기존 사용자 학습/공유 Isaac 환경은 변경하지 않았습니다. 원본15개·canonical14개(train8/validation6)와 final 생성 seeds303/404 미사용 상태를 유지합니다. v12/v13 reference는 QA 실패로 미채택입니다.
