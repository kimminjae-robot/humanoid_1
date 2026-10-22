# 공개 배포 범위

실행에 필요한 코드·설정, K1 결합 모델이 읽는 메시와 원본 FK 검증용 메시, G1 canonical reference, K1 기준 4개와 경례 reference, G1 실행용 TorchScript 정책·관측 fixture, 동일 초기 상태 비교용 Isaac result/rollout 1개를 포함합니다. K1의 경례 보정 초기화 chain은 작은 NPZ reference로 보존합니다.

학습 재개용 checkpoint, 생성 원본 전체, 중복 실험, MuJoCo 평가 rollout, 영상·이미지, 가상환경, HF 모델 가중치·캐시·토큰은 제외했습니다. 전체 로컬 재현 패키지는 별도로 보존했습니다. 역사 보고서에 나오는 모든 trial 폴더는 공개 배포본에 포함하지 않습니다.

로봇 자산은 공개 저장소의 고정 revision이며 원래 라이선스를 licenses/에 보존했습니다. `sources.lock.json`에 Kimodo·K1·HX5·Unitree 출처를 기록했습니다. G1/K1 reference는 이 프로젝트에서 생성·변환한 로봇 관절과 자세의 수치 배열입니다. 공개 배포본에는 사람 영상이나 실측 개인정보를 포함하지 않습니다.

K1 MJCF의 메시 경로를 상대 경로로 바꾸고 배포용 robot_manifest의 해시를 갱신했습니다. 로봇의 수치 파라미터와 reference 배열은 원래 검증 값입니다. 과거 측정 JSON에는 당시 파일 경로와 원래 XML 해시가 남을 수 있습니다. 배포 파일의 실제 해시는 PACKAGE_MANIFEST.json에 있습니다.
