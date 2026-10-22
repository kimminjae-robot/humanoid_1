# 공개용 파일 범위 확인

대상은 사용자가 직접 지정한 `https://github.com/kimminjae-robot/humanoid_1.git`입니다. GitHub 연결에서 인증 사용자 login은 kimminjae-robot이고, 저장소 owner도 kimminjae-robot, 저장소 ID는 1429232313이며 admin/push 권한을 확인했습니다.

처음의 전체 재현 패키지 약 134MB에서 공개 범위를 줄였습니다. 모든 학습 재개용 checkpoint, 영상·이미지, 생성 원본 전체, 중복 평가 실행과 실험 trace를 제외했습니다. 원래 전체 자료는 로컬 `workspaces/kimodo_portable`에 보존했습니다.

공개용에는 아래 실행 필수 자료만 남깁니다.

- 코드·설정·문서와 고정 버전 설치 도구.
- 공개 K1/HX5/G1 로봇 자산 및 라이선스. K1/HX5 메시와 Unitree URDF는 sources.lock.json의 공개 저장소 revision으로 출처를 확인했습니다.
- 26개 NPZ: canonical 로봇 reference, 경례 조건화 chain, 정책 관측 fixture, 독립 sim-to-sim 비교에 필요한 Isaac rollout 1개. 배열 스키마를 검사했고 object/pickle dtype은 없습니다. 내용은 프로젝트에서 생성·변환한 로봇 관절·위치·접촉 등 수치와 관절명·provenance 문자열입니다.
- 실행용 TorchScript 정책 1개: 206138개 유한한 파라미터의 로봇 관측→행동 MLP입니다. 학습 재개용 checkpoint가 아닙니다. 같은 정책 CPU 실행과 관측·PD fixture를 다시 통과했습니다.
- 같은 초기 상태 재평가를 위한 Isaac result/contract와 집계된 물리 평가 결과.

공개 파일의 텍스트와 NPZ 문자열에서 HF/GitHub/AWS 키 및 개인 키 형식을 검사했고 발견하지 않았습니다. cache, token, .env, 가상환경은 포함하지 않습니다. 큰 생성 가중치는 각 공식 제공처에서 사용자가 별도로 내려받습니다. 저장소 전체 파일의 해시는 PACKAGE_MANIFEST.json으로 검증할 수 있습니다.
