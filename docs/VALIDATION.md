# 배포본 재검증 — 2026-10-22

원래 실험 폴더와 분리된 `/tmp/kimodo_portability_check`에 배포 파일을 복사하고, 새로 만든 전용 Python 3.11 가상환경에 PyPI 의존성을 설치해 검증했습니다. 별도 물리 PC 검증은 아니며 같은 호스트의 독립 경로·환경 검사입니다.

| 검사 | 결과 |
|---|---|
| 고정 MuJoCo 의존성의 새 가상환경 설치 | PASS, MuJoCo 3.5.0 / NumPy 1.26.4 / SciPy 1.17.1 / OSQP 1.0.4 |
| 두 모델의 상대 asset 경로와 nq/nv/nu | PASS |
| K1 기존 모델과 배포 모델의 compiled 숫자 배열 | 경로 문자열 메타데이터 `_sizes`, `mesh_pathadr`를 제외한 452개 배열 동일 |
| K1 원본 대비 11개 비대칭 FK·질량 검사 | PASS, 위치 오차 0, 회전 오차 약 6.01e-16rad |
| K1 원본 자산에서 prepare 재구성 + 검증 | PASS |
| K1 canonical 4 clips × 3 resets | QA PASS, 서기 3/3·wave 3/3·걷기 6/6 |
| K1 최종 경례 | QA PASS, 3/3, 각 6.96초 완료 |
| K1 저장 trace 독립 지표 감사 | 두 run 모두 integration/performance PASS |
| G1 CPU Torch 2.7.0 새 설치·관측/정책/PD fixtures | PASS |
| G1 MuJoCo 60 episodes | 서기 20/20·wave 20/20·걷기 0/20, 기존 판정 재현 |
| G1 저장 trace·동일 초기 상태 독립 감사 | integration PASS, performance FAIL 유지 |
| Python 구문·셸 구문 | PASS |
| 배포 텍스트의 HF/GitHub/AWS 키·개인 키 형식 검색 | 발견 없음 |

집계 결과는 [reports/portability_summary.json](../reports/portability_summary.json)에 있습니다. 큰 trace와 전체 실험 이력은 로컬 재현 패키지에 보존했습니다. 공개 배포본의 코드·자산·reference는 검사한 동일 파일의 부분집합입니다.

이번 포장 작업에서는 생성 모델을 다시 다운로드하거나 Kimodo UI를 변경하지 않았습니다. 다른 PC의 GPU 생성·Isaac 재학습·렌더링은 추가 설치 뒤 확인해야 합니다. 원래 1ms refinement 결과는 배포본에 보존했고 이 포장 검사에서는 nominal 물리 평가를 재실행했습니다.
