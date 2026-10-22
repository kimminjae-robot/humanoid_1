# 현재 상태

- **경례 개선 완료:** 원래 clip의 지지 자세·팔 복귀 경로 보정과 centroidal QP/torque PD로 QA PASS, 실제 물리 **3/3**,각6.96초 완료. timestep1ms 재검증3/3. 실제 자기 침투 최대0.76mm(nominal). [최종 경례 영상](reports/gallery_salute_improved.html) · [개선 보고서](reports/salute_improvement.md).
- 선택 경례는 `reports/dataset_salute_standing_v6.json`, controller는 명시적으로 `centroidal`. 기존 canonical4clips와 기본 generic process는 유지했다. 기존 실패 경례 및 아래 실패 이력은 보존했다.

- 새 경례 모션(seed73112,6초) 생성·K1 참조 재생 확인 완료. 자기 충돌44.24mm로 QA 실패, 실제 물리0/3(2.00~2.04초 낙상). [경례 비교 화면](reports/gallery_salute_001.html) · [보고서](reports/salute.md). canonical 성공 세트에는 채택하지 않았다.

- 최종 목표: 범용 생성·리타게팅 및 Kimodo→K1 MuJoCo 자동 처리. 개별 모션의 성공만으로 범용 완료를 선언하지 않는다.
- 새 조준 모션(seed73111,6초) 생성·공통 v1 변환·MuJoCo 확인 완료. QA 실패(바닥30.69mm/자기 충돌26.23mm), 실제 물리0/3(1.56초 낙상). [비교 화면](reports/gallery_pistol_aim_002.html) · [보고서](reports/pistol_aim.md). 기존 canonical4clips에는 채택하지 않았다.
- 새 `generate`와 `process` 명령 추가. process는 CSV부터 QA/물리/감사/영상까지 연결하며 실패도 기록한다. UI 자동 연결·범용 추종은 아직 미완성이다.

- 2026-10-21: **짧은 보폭·느린 걷기 MuJoCo 물리 평가 통과.**
- 걷기 **0/3 → 6/6** (Kimodo clip2개 × reset3개). stand3/3·wave3/3 유지.
- walk101:35.78초 완료/약0.302m 전진/좌3·우2회 발 들기.
- walk202:41.78초 완료/약0.368m 전진/좌3·우3회 발 들기.
- 두 걷기 reference 모두 QA PASS. clip101 physics timestep 절반 검사도3/3 통과.
- **원본 보폭/속도 재현은 아님:** 앞 방향 발 이동30%, COM transfer3초/swing3초, upright root·상체 중립으로 조건화했다.

[최신 비교 영상](reports/gallery.html) · [추가 걷기 영상](reports/gallery_walk_validation202.html) · [걷기 개선 보고서](reports/walking_improvement.md) · [정량 결과](reports/walking_improvement_summary.json).

현재 canonical `data/dataset.json`: stand/wave v1 + 보수적인 walk101/walk202, 총4clips. 이전 걷기 실패와 baseline은 `data/dataset_baseline.json`, `runs/baseline`, `reports/baseline.md`에 보존한다. 제어기는 걷기에 centroidal 지면반력 QP+torque PD, stand/wave에는 기존 balance를 사용한다. 물리/모터 제한과 평가 임계값은 유지했다. 손63관절 중 손40관절은 일정 목표를 유지하고 손 조작은 범위 밖이다.

기본 회귀 run `walking_improved_v4`, 추가 `walk_validation202`, refinement `walking_v4_refined`. 기존 Kimodo UI/공유 Isaac 환경/G1 실험은 유지했다. 다음 우선순위는 이름에 의존하지 않는 접촉 분석·전신 충돌/팔 자세 보존을 포함한 공통 리타게터와 UI 자동 연결이다. 임의 프롬프트 일반화/원래 속도/실물 장착·보행은 미검증이다.
