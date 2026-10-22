# Kimodo humanoid motion — G1 / K1 + HX5

텍스트 모션 생성, 로봇 리타게팅, 실제 물리 추종을 실험한 프로젝트입니다. **G1 학습·sim-to-sim 프레임워크와 K1+HX5 제어 실험을 다른 PC에서 재실행할 수 있도록 추렸습니다.**

| 프로젝트 | 구현 | 확인한 결과 |
|---|---|---|
| Unitree G1 | Kimodo → reference 정제 → Isaac Lab PPO → 동일 정책 MuJoCo 평가 | 양쪽 서기 20/20, wave 20/20, 걷기 0/20 |
| AI Sapiens K1 + HX5-D20 | G1 모션 → K1 전신 IK → 접촉·충돌 보정 → MuJoCo QP/PD 제어 | 서기 3/3, wave 3/3, 보수적인 걷기 6/6 |
| K1 경례 개선 | 고정 발·COM 보정 + 팔 복귀 경로 수정 + centroidal 제어 | QA PASS, 물리 3/3, 1ms timestep 재검사 3/3 |

각 분모는 제한된 생성 clip과 reset 반복입니다. K1 걷기는 원본 발 전진의 30%, 체중 이동·swing 각각 3초로 바꿨습니다. 경례는 팔을 내리는 경로도 수정했습니다. 임의 프롬프트 일반화, 원래 속도·스타일 전체 재현, 실물 로봇 검증은 완료되지 않았습니다. 새 조준 모션의 실패도 보존했습니다. K1 결과는 QP/PD 제어이며 Isaac 학습 정책 전이가 아닙니다.

## 빠른 실행: K1 경례

Linux / Python 3.11 기준입니다. 저장된 모션의 물리 평가는 CPU에서 실행하며 Kimodo 가중치나 Isaac Sim이 필요하지 않습니다.

```bash
git clone https://github.com/kimminjae-robot/humanoid_1.git
cd humanoid_1
bash setup.sh mujoco
bash workspaces/k1_kimodo_hx5/run.sh validate-robot
bash workspaces/k1_kimodo_hx5/run.sh evaluate \
  --controller centroidal \
  --references "$PWD/workspaces/k1_kimodo_hx5/reports/dataset_salute_standing_v6.json" \
  --label salute_pc_001
bash workspaces/k1_kimodo_hx5/run.sh assess --run salute_pc_001
```

같은 label로 재실행하면 기존 결과를 덮어쓰지 않습니다. 새 label을 사용하세요. `evaluate` 종료 코드 0은 실행 완료이고, 성능은 `runs/<label>/result.json:all_motions_passed`와 `kinematic_qa.json:passed`를 함께 확인합니다.

- [프로젝트 정리](docs/PROJECT_SUMMARY.ko.md)
- [다른 PC 설치·G1 재평가·새 모션 생성·Isaac 학습](docs/REPRODUCE.md)
- [포함·제외 파일과 출처](docs/PACKAGE_SCOPE.md)
- [별도 경로·독립 환경 재실행 결과](docs/VALIDATION.md)
- [K1 경례 비교 영상](workspaces/k1_kimodo_hx5/reports/gallery_salute_improved.html)
- [경례 개선 근거](workspaces/k1_kimodo_hx5/reports/salute_improvement.md)
- [G1 양쪽 엔진 비교](workspaces/g1_kimodo/reports/stage_05.md)

HTML 영상은 clone 후 브라우저로 열거나 저장소에서 `python3 -m http.server 8000`으로 볼 수 있습니다. GitHub 파일 보기에서는 HTML이 실행되지 않습니다.

필수 로봇 메시, 선택 reference, G1 정책·체크포인트와 결과 근거를 포함합니다. 가상환경, HF 캐시·토큰, 대형 생성 모델과 중복 실험은 제외했습니다. 모델은 고정 revision으로 내려받습니다. `sources.lock.json`과 `PACKAGE_MANIFEST.json`에 출처와 배포 파일 해시를 기록합니다.
