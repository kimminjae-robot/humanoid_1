# 다른 PC에서 재현하기

저장소 루트에서 실행합니다. 기준 환경은 Linux와 Python 3.11입니다. Windows/macOS 네이티브 실행은 검증하지 않았습니다. 아래 단계는 필요한 기능에 따라 선택합니다.

## A. 저장된 K1 모션의 물리 평가

GPU, HF 로그인, Isaac Sim 없이 실행할 수 있습니다.

```bash
bash setup.sh mujoco
bash workspaces/k1_kimodo_hx5/run.sh validate-robot
bash workspaces/k1_kimodo_hx5/run.sh evaluate --label canonical_pc_001
bash workspaces/k1_kimodo_hx5/run.sh assess --run canonical_pc_001
```

서기·wave와 조건화된 걷기 2개를 총 12회 평가합니다. 경례는 README의 별도 reference와 명시적 `centroidal`을 사용합니다. `--physics-dt 0.001`로 timestep 검사를 추가할 수 있습니다. 각 실행에는 새 label이 필요합니다.

영상 재생성에는 시스템 `ffmpeg`와 렌더링 환경이 필요합니다. 기본 EGL 대신 GPU 없는 환경에서 OSMesa를 사용할 수 있지만 시스템 OSMesa 라이브러리는 별도로 필요합니다.

```bash
bash workspaces/k1_kimodo_hx5/run.sh review --run salute_pc_001 \
  --gallery-name gallery_salute_pc_001.html
```

새 G1 CSV의 리타게팅·검사는 다음과 같습니다. CSV는 xyz+wxyz+29관절의 36열이며 실제 입력 fps를 전달합니다.

```bash
bash workspaces/k1_kimodo_hx5/run.sh retarget \
  --source-csv /absolute/path/motion.csv --fps 30 \
  --clip-id new_motion_001 --motion-id custom
bash workspaces/k1_kimodo_hx5/run.sh evaluate \
  --references "$PWD/workspaces/k1_kimodo_hx5/reports/retarget_new_motion_001.json" \
  --label new_motion_001
```

`process`는 retarget→평가→감사→영상을 연결하며 성능 실패 시 exit 2입니다. 기본 변환이 경례의 수동 보정을 모든 새 모션에 자동 적용하지는 않습니다.

## B. G1의 저장된 동일 정책을 MuJoCo에서 평가

```bash
bash setup.sh policy
bash workspaces/g1_kimodo/run.sh sim2sim --fixtures-only
bash workspaces/g1_kimodo/run.sh sim2sim
```

CPU Torch 2.7.0을 추가 설치합니다. 원래 결과는 Torch 2.7.0+cu128의 CPU actor로 측정했습니다. 가중치·관측·PD fixture와 실제 결과를 새 PC에서 확인하세요. 위 동적 평가는 저장된 Isaac 초기 상태 60개를 사용합니다. 기대 기준은 서기 20/20, wave 20/20, 걷기 0/20이며 실행 완료와 전체 성능 통과를 구분합니다.

## C. 새 텍스트로 Kimodo 모션 생성

이 단계는 NVIDIA CUDA 환경, 공식 모델 다운로드, Llama 접근 승인이 필요합니다. 원래 생성은 16GB GPU에서 실행했고 공식 텍스트 인코더는 CPU에 올렸습니다. 기반 Llama는 약 16.1GB, 어댑터는 약 0.34GB이며 생성 가중치·가상환경을 더한 저장 공간이 필요합니다. 원래 프로젝트는 40GiB 여유 공간을 임시 예산으로 사용했으며 보장된 최소 사양은 아닙니다.

```bash
bash setup.sh generator
# 터미널에서 직접 로그인합니다. 토큰을 저장소에 넣지 마세요.
HF_HOME="$PWD/workspaces/g1_kimodo/cache/huggingface" \
  workspaces/g1_kimodo/.venvs/kimodo/bin/hf auth login
workspaces/g1_kimodo/.venvs/kimodo/bin/python -I tools/download_generator.py
bash workspaces/g1_kimodo/run.sh download-official
bash workspaces/g1_kimodo/run.sh cache-official
bash workspaces/g1_kimodo/run.sh smoke-official
```

`sources.lock.json`과 `configs/official_encoder.json`의 revision을 사용합니다. `setup.sh generator`는 CUDA 12.8용 Torch를 선택하므로 드라이버·GPU 호환성을 새 PC에서 확인합니다. 모델 가중치와 승인된 인코더는 배포본에 포함하지 않습니다. [HF 준비](../workspaces/g1_kimodo/docs/HF_SETUP.md)를 참고하세요.

새 config의 `label`을 바꾸고 실행합니다. 보존된 `salute_001` label은 이미 존재하므로 그대로 쓰지 않습니다.

```bash
cp workspaces/k1_kimodo_hx5/configs/salute_001.json /tmp/salute_new.json
# /tmp/salute_new.json의 label을 salute_new_001로 수정
bash workspaces/k1_kimodo_hx5/run.sh generate --config /tmp/salute_new.json
```

생성·변환 성공은 QA 또는 물리 성공을 뜻하지 않습니다.

## D. Isaac Lab 학습·정책 재평가

Isaac Lab 2.3.2와 해당 Isaac Sim/RSL-RL 환경을 별도로 준비합니다. 기준 IsaacLab commit은 `37ddf626871758333d6ed89cf64ad702aef127d0`입니다. Isaac Lab 전체 소스와 런타임은 이 저장소에 포함하지 않습니다. 설치된 환경의 Python 경로를 지정하면 이 저장소는 외부 디렉터리에 있어도 실행할 수 있습니다.

```bash
export G1_ISAACLAB_PYTHON=/absolute/path/to/isaaclab/env/bin/python
bash workspaces/g1_kimodo/run.sh validate-g1-isaac
bash workspaces/g1_kimodo/run.sh track \
  --mode train --motions stand --num-envs 32 --iterations 5 \
  --contract "$PWD/workspaces/g1_kimodo/configs/tracking_contract_stiff.json" \
  --label pc_stand_training
```

공유 환경에 자동 설치하지 않습니다. 별도 PC의 Isaac 설치와 GPU 평가는 이번 포장 작업에서 새로 검증하지 않았습니다. 설정·실행 정책·USD를 보존했고 Python 경로를 이식 가능하게 처리했습니다. 위 명령은 새 학습을 시작하는 예시입니다. 학습 재개용 체크포인트는 공개 배포 범위에서 제외했습니다. 기존 v11의 MuJoCo 재평가는 B 절차를 사용합니다.

## 실행 환경을 직접 지정하기

K1은 `K1_KIMODO_PYTHON`, G1 MuJoCo/정책은 `G1_MUJOCO_PYTHON`, 생성은 `KIMODO_GENERATION_PYTHON`, Isaac은 `G1_ISAACLAB_PYTHON`으로 Python을 지정할 수 있습니다. 기본 CPU 환경은 저장소 루트 `.venv`입니다.
