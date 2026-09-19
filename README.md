


## Implementation plan

### Research Goal

- Research scope:

    - Image to image retrieval

    - 유사 상품 검색 품질 개선 (직접 정의된 유사도-평가 기준에 따라)

- 평가 기준

    - 유사도를 얼마나 논리적으로 정의하는가

    - 어떻게 해당 유사도를 높일 수 있도록 모델을 sft 할 것인가

    - 논리적으로 정량, 정성적 평가 진행.

- Ultimate goal:

    - SigLIP2 의 검색 품질은 개선되었는가

    - Tianmu-MERE 와 비교했을 떄 어떤 상황에서 더 잘하고, 어떤 상황에서 잘 못하는 가?

    - 결과의 한계와 다음으로 검증할 가설 제시.

    - 단순히 점수의 높낮이보다 그 결과가 왜 유사상품 검색이라는 목표를 더 잘 또는 덜 달성했다고 볼 수 있는 지에 대한 근거를 설명.

### 확정된 유사도 정의

- 척도: **실루엣** = 원피스 마스크의 외곽선 전체를 정규화해서 겹쳐본 것 (2026-09-19 문구 정정). 절대 크기·위치는 canonical 정규화로 제거되고, 소매 모양과 기장 비율은 외곽선의 일부이므로 자연스럽게 포함된다. 패턴/색상/넥라인 디테일은 마스크에 담기지 않으므로 제외된다. 별도 가공(소매 제거, 기장 정규화 등) 없이 마스크 IoU를 그대로 쓴다 — 정의와 측정 방법을 일치시켜 단순하게 유지하기 위함.

- 평가 대상 모델(SigLIP2, Tianmu-MERE) 임베딩으로는 유사도를 정의하지 않음 (순환논리 방지).

- 사람 라벨링 골드셋은 없음. threshold 등은 소규모 샘플을 육안으로 스팟체크해서 결정 (아래 마이닝 파이프라인 5번 참고).

- **positive pair 생성: shape-descriptor 기반 실제 쌍 마이닝** (2026-09-19 확정, baseline 성격 — 정교한 corner case 처리보다 이해하기 쉽고 납득 가능한 방법 우선)

  - 서로 다른 item인데 실루엣이 실제로 유사한 이미지 쌍을 데이터에서 직접 찾아서 사용.

  - color 축은 이번 범위에서 계속 제외 (기존 결정 유지) — silhouette 축 1종만 사용.

#### Positive pair 마이닝 파이프라인 (canonical-resize + IoU, 단순화된 baseline)

핵심 아이디어: **두 원피스의 실루엣을 같은 크기의 정사각형 틀에 맞춰 놓고, 겹치는 비율(IoU)을 직접 재는 것.** 별도의 통계적 형태 기술자(Hu moment 등)나 근사 최근접 탐색 라이브러리(faiss) 없이, 마스크를 정규화된 크기로 리사이즈한 뒤 곧바로 픽셀 단위로 비교한다 — 구현이 단순하고 결과를 눈으로 바로 검증할 수 있다.

1. **마스크 추출**: Grounded-SAM(Grounding DINO + SAM)으로 사용 대상 이미지(train 4,000장 규모 + test 100장)에 대해 원피스 영역 binary mask + bbox 추출 → `data/masks/`. 검출 실패분은 `failed.csv`에 기록하고 건너뛴다

   - **2단계 처리 (2026-09-19 확정)**: Grounding DINO로 박스를 검출한 뒤, 그 박스를 **원본 해상도 좌표로 환산해 원본에서 크롭한 다음 SAM에 입력**하고, 나온 마스크를 원본 전체 좌표계에 되붙인다. test 이미지가 Pexels 원본(최대 3648x5472)이고 인물이 프레임에서 작게 찍힌 거리 샷이 많아, 통째로 축소해 넣으면 마스크 품질이 떨어지기 때문. train(800x800)에서는 크롭 단계가 사실상 항등이라 train/test가 같은 코드 경로를 쓴다.

2. **Canonical mask 생성**: 마스크를 bbox로 크롭 → 정사각형에 가운데 정렬로 padding(letterbox) → 64x64로 resize한 binary mask 생성. 이 한 번의 리사이즈로 크기·위치 차이가 자동으로 제거됨 (별도 descriptor 계산 불필요). 회전은 정규화하지 않음 — 제품 사진이 대체로 정면·직립으로 촬영된다는 전제 (한계로 기록, 회전된/뒤집힌 착장은 이번 baseline에서 다루지 않음).

3. **전체 쌍 IoU 계산**: 64x64 canonical mask를 4096차원 이진 벡터로 펼친 뒤, `intersection = masks @ masks.T`(행렬곱), `union = area_i + area_j - intersection`으로 전체 쌍의 IoU를 계산 (행 청크 단위로 나눠 청크마다 top-K만 남김, L40S 1장으로 충분).

   - **near-duplicate 제외 (2026-09-19 수정)**: 원안은 "같은 item_id 제외"였으나 실측 결과 item_id는 이미지당 정확히 1개(item_id 29,350 : unique image 26,326, 같은 이미지가 geo별 리스팅으로 여러 item_id에 중복 등장)라 **제외해도 아무것도 걸러지지 않는 no-op**이었음. 실제로 막아야 할 대상(동일 상품의 색상 변형)은 서로 다른 image_id로 존재하므로 아래 두 규칙을 대신 적용:

     - `name` 문자열이 완전히 일치하는 이미지 그룹끼리 후보에서 제외 (1,407그룹 / 4,153장). 언어별 텍스트 휴리스틱 없이 완전일치만 사용해 별도 가정을 만들지 않음.

     - IoU가 상한을 넘는 쌍 제외 (이름이 달라도 실루엣이 사실상 동일한 near-duplicate 차단). 상한값은 5번 스팟체크에서 확정.

   - 목적은 원안과 동일 — cross-instance 신호 보장 (같은 상품의 다른 사진이 섞이면 학습이 지나치게 쉬워짐).

4. **top-K 선택**: anchor별로 IoU가 가장 높은 상위 K개(K=50)를 후보로 남김.

5. **Threshold 결정 (스팟체크, 2026-09-19 확정, 이후 재조정)**: 골드셋이 없으므로 IoU 점수대별로 샘플을 뽑아 육안 비교. 1차로 하한 0.70 / 상한 0.97을 확정했으나, **최종적으로 하한 0.90 / 상한 0.95로 재조정** — 0.9~0.95 구간이 실루엣 일치도가 가장 높고 명백히 다른 상품인 "가장 좋은 품질의 positive"였기 때문. 대신 후보 anchor 커버리지가 크게 줄어(2,000장 기준 1,906→538개) train 이미지 규모를 2,000 → 4,000장으로 늘려 anchor 수를 보충(1,128개). 근거와 샘플은 `discussion_summary.md` 참고.

6. **출력**: `data/pairs/mined_pairs.csv` (anchor_image_id, positive_image_id, iou_score)

### 확정된 fine-tuning 전략: prototype-matching self-distillation

- SigLIP2 vision encoder(so400m) 위에 **새 prototype head를 구현**해서 얹음. (공개 체크포인트/transformers 구현에 self-distillation head가 없음을 확인함 — 원조 DINO처럼 backbone과 head가 end-to-end로 같이 학습된 게 아니라는 점은 limitation으로 기록)

- Teacher: EMA update, momentum cosine 0.996 → 1.0

- Collapse 방지: DINO 방식 centering + sharpening (temperature 비대칭) 세트로 사용

- Loss: 표준 DINO cross-entropy (teacher softmax(logit/teacher_temp, stop-gradient) vs student softmax(logit/student_temp), teacher-view × student-view 조합 합산, 동일 view 쌍 제외)

- Prototype head 스펙: MLP 1152 → 2048 → 2048 → 256(bottleneck, L2-normalize) → weight-normalized linear → K=1024, student temp 0.1(고정), teacher temp 0.04→0.07 warm-up, centering momentum 0.9

- Backbone: **LoRA**로 fine-tune (전체 fine-tune + LLRD 대신 채택, 2026-09-19 확정 — baseline 성격에 맞게 단순화)

  - Backbone은 전부 freeze, attention의 Q/V projection에 LoRA adapter 삽입 (rank r=16, alpha=32, dropout 0.1 — 통상적으로 쓰이는 값, 튜닝 없음)

  - 학습 대상은 LoRA adapter + prototype head 뿐. **레이어별로 다른 학습률을 주는 LLRD 없이, LoRA·head 모두 동일한 flat LR(1e-4) 하나만 사용** — "어떤 파라미터를 얼마나 다르게 학습시킬지"에 대한 별도 설계/설명이 필요 없어 baseline으로서 납득하기 쉬움

  - L40S(48GB) 단일 GPU 기준: backbone이 대부분 freeze되어 있어 메모리 부담이 크게 줄어듦. bf16 사용, gradient checkpointing은 batch size 확보를 위해 필요 시 적용

- **View 구성** (mined pair 기반으로 재설계, 확정):

  - Teacher: anchor 이미지의 masked global crop **1장만** 처리 (target 분포 생성)

  - Student: 아래 5장 처리 (teacher와 합쳐 총 6 forward pass, 2026-09-19 수정)

    - mined positive(다른 item, 유사 실루엣)의 masked global crop 1장 (cross-instance 신호의 핵심)

    - **boundary-anchored local crop** 4장 — anchor 2장 + positive 2장 (아래 상세)

  - **anchor의 masked global crop + photometric augmentation view는 제외** (2026-09-19 확정) — teacher가 보는 anchor global crop과 색조 차이만 있어 사실상 거의 같은 입력. student가 이 둘을 구분 없이 맞히는 게 너무 쉬워 collapse로 가는 지름길이 될 위험이 큼. same-instance consistency 신호 없이 cross-instance(positive) + local crop만으로 학습

  - 마스크 추출(1회성 전처리) 이후로는 순수 crop/augmentation 연산만 필요 — GPU 부담이 크지 않음

- View 전처리(global crop): **dilated mask의 bounding box로 크롭 후, 마스크 밖 영역은 0으로 채움** (배경 포함 학습 시 "배경이 같아서 유사하다"는 shortcut을 막기 위함)

- **Boundary-anchored local crop 샘플링** (확정, 단순화된 방식):

  - 핵심 아이디어: **실루엣 경계에 있는 픽셀 위에 crop 중심을 두면, 그 crop은 안쪽(옷)과 바깥쪽(배경)을 동시에 담게 된다.** 별도의 contour 추적 알고리즘 없이 마스크 연산 두 번으로 구현 가능.

  - 경계 픽셀 집합 = `mask - erode(mask, 3x3 kernel)` (마스크에서 살짝 깎아낸(erode) 버전을 뺀 나머지가 곧 가장자리 픽셀)

  - 이 경계 픽셀 집합에서 무작위로 한 점을 골라 crop 중심으로 사용 (arc-length 균등 샘플링 등 별도 파라미터화 불필요)

  - crop window 크기: bbox 대각선의 25~35% (랜덤), 중심점 기준 소량 jitter

  - 안전장치: 내부(mask=1)/외부(mask=0) 픽셀 비율이 최소 15%~최대 85% 범위를 벗어나면(이미지 경계에 걸리는 등 예외 상황) 재샘플링

  - 학습 시점에 on-the-fly 샘플링 (사전 생성 없음)

### 확정된 평가/모니터링 방식

- **정식 train/val split을 두지 않음.** 대신 학습 데이터 내 고정 쿼리 샘플 5~10개를 골라 매 checkpoint/epoch마다 top-10 검색 결과(썸네일+유사도)를 정성적으로 트래킹.

- Collapse 모니터링 로깅(복잡한 가정 없는 단순 지표만): 개별 샘플 예측 entropy 평균, 배치 평균 예측 entropy, dead prototype 비율.

- eval.sh: test 데이터(100장) 확보 후 동일한 방식으로 검색 결과 확인 예정.

### File structure

1) 모델 학습 및 관리는 pytorch-lightning 사용.

2) RunPod **Network Volume**은 /workspace에 마운트하고, 아래 data, experiments, ckpts 경로는 /workspace/emb/ 아래에서 사용. (pod 재생성/중지에도 유지됨.)

```

- env

    - requirements.txt (또는 pyproject.toml) : RunPod 기본 PyTorch 이미지 위에 추가로 필요한 Python dependency를 고정. PyTorch/Transformers 계열로 통일하고 big_vision(JAX/TF)은 설치하지 않음.

    - pod.sh : 필요 시 추후 runpodctl 기반으로 L40S pod 생성/시작/중지/삭제를 자동화. 현재는 RunPod UI로 pod를 관리.

- notebooks

    - 00_data_preparation.ipynb : GLAMI-1M dresses 서브셋 원본 확인 (item_id-image_id 다대일 관계, 결측치, geo 분포, 이미지 육안 점검)

    - discussion_summary.md : 유사도 정의/학습 방향 결정사항 기록

    - descriptor_poc.ipynb : canonical mask 리사이즈/IoU 계산 PoC + IoU 점수대별 샘플 육안 스팟체크로 threshold 결정 (기존 generate_poc_pairs.py 대체)

    - view_sanity_check.ipynb : positive pair(anchor-positive) 및 local/global view 시각화 샘플 생성 — train, test(100장) 양쪽 모두 확인. test는 원피스 mask 추출 결과를 전수 조사(직접 확인) 대상으로 별도 정리

    - emb_analysis/ : SigLIP2(SFT 전)·Tianmu-MERE vision embedding 진단 (SFT 체크포인트 나오면 SFT 후 결과와 나란히 비교하도록 확장 예정)

      - baseline shape 진단 (2026-09-19 결과 반영, "완전히 shape-blind"가 아니라 "약하게만 잡음 → SFT로 강화" 프레임으로 수정):

        1. **질적 비교 그리드**: 실루엣이 뚜렷이 다른 카테고리(A라인/bodycon/맥시 등)에서 anchor 몇 개를 골라, 각 anchor에 대해 (a) mined_pairs.csv 기준 shape-IoU top-5, (b) SigLIP2 임베딩 cosine sim top-5, (c) Tianmu-MERE 임베딩 cosine sim top-5를 나란히 배치. shape-IoU top-5와 겹치는 검색 결과가 이미 종종 나온다 (완전한 우연은 아님).

        2. **정량 지표 (shape recall)**: anchor별 mined_pairs.csv의 positive 전체를 proxy positive로 두고, SigLIP2/Tianmu-MERE 임베딩 검색 결과 top-N(N=20) 안에 이 positive가 몇 개 들어오는지 Recall@N 계산 (`src/tools/shape_recall.py`). 무작위 N장 추출 시 기대 recall(`N / (전체 - 1)`)을 baseline으로 같이 표기 — anchor 1,128개 평균으로 모델별 막대그래프 1장. **결과: SigLIP2 0.134 / Tianmu-MERE 0.115, random baseline 0.005 대비 각각 23~27배** — 완전히 못 잡는 건 아니고 약한 신호가 이미 있음. SFT 후 동일 지표로 재측정해 baseline과 나란히 비교하는 것이 eval.py의 핵심 정량 지표가 됨.

        - 임베딩 2D 투영(PCA/UMAP) 비교는 **범위에서 제외** (2026-09-19 확정) — 위 두 가지로 이미 결론이 나와 추가 정보가 크지 않다고 판단.

        - retrieval.py, shape_descriptors.py, mined_pairs.csv를 그대로 재사용해서 만들 수 있어 별도 구현 최소화.

- data

    - raw

        - GLAMI-1M-dataset/ : GLAMI-1M 원본 csv (train/test), LICENSE 등

        - GLAMI-1M-dresses-train.csv : category_name=dresses 필터링 결과 (29,350 rows, unique image_id 26,326)

        - images-800px/{image_id}.jpg : 위 unique image_id에 해당하는 800x800 원본 이미지 (train dresses 26,326장만, test와 분리)

        - images-800px-test/ : test 100장 원본(`NNN_{pexels_id}.jpg`) + manifest.json + ATTRIBUTION.md (train 폴더와 물리적으로 분리)

    - masks/{image_id}.png : Grounded-SAM 추출 binary mask, 원본 좌표계 (+ bbox/det_score/mask_area_ratio는 index.csv에 기록). 검출 실패분은 failed.csv에 따로 기록

    - masks-test/ : test 100장의 마스크 (train과 분리, 전수 조사 대상)

    - descriptors/canonical_masks.npy : image_id별 64x64 canonical(letterbox+resize) binary mask 모음

    - pairs/mined_pairs.csv : 마이닝된 positive pair (anchor_image_id, positive_image_id, iou_score)

    - GLAMI-1M 공식 human-labeled test split은 사용하지 않음 (train.csv만 사용)

    - test data (100 장 원피스 사진, data/raw/images-800px-test/ 에 저장함, train과 폴더 분리됨 — 2026-09-19) — 제공 완료. 학습 및 validation 에는 절대 사용 금지, mask 추출 결과는 전수 조사 대상.

- ckpts

    - Tianmu-MERE/, siglip2-so400m-patch14-384/ : 비교 대상 모델

    - grounding-dino/, sam/ : 마스크 추출 파이프라인용. 둘 다 HF transformers 포맷 (`IDEA-Research/grounding-dino-base`, `facebook/sam-vit-large`) — SAM 원본 `.pth`는 transformers로 바로 로드할 수 없어 HF 스냅샷으로 대체 (2026-09-19)

- configs

    - self_distill_base.yaml : 기본 템플릿 (batch size/precision은 L40S 실측 후 재산정)

    - self_distill_v1.yaml : 실제 실행용

- experiemnts

    - <run_name>/{config.yaml, checkpoints/, logs/, retrieval_samples/epoch_XX/query_<id>.jpg} : 학습 산출물. retrieval_samples는 고정 쿼리의 top-10 검색 결과 트래킹

- src

    - func

        - obj_func.py : DINO cross-entropy (centering + sharpening)

    - modules

        - backbone.py : SigLIP2 vision encoder (freeze) + LoRA adapter (Q/V projection, r=16)

        - prototype_head.py : MLP(1152→2048→2048→256) + weight-normalized linear(256→1024)

    - model

        - base.py : LightningModule 추상 클래스

        - self_distill.py : student/teacher EMA, 6-view 학습 스텝(teacher anchor global + student: positive global + local 4), collapse 로깅

    - tools

        - retrieval.py : 고정 쿼리 top-k 검색 + 썸네일 그리드 저장 (shape-blindness 질적 비교 그리드에도 재사용)

        - collapse_metrics.py : entropy 2종 + dead prototype 비율

        - shape_descriptors.py : mask → canonical 64x64 리사이즈, 전체 쌍 IoU 행렬곱 함수

    - dataset

        - extract_masks.py : Grounded-SAM 마스크 일괄 추출

        - mine_pairs.py : canonical mask 생성 → 전체 쌍 IoU 행렬곱 → item_id 중복 제외 → top-K → mined_pairs.csv

        - view_dataset.py : global view(mask bbox crop + zero-fill) 로딩, boundary-anchored local crop 샘플링

        - transforms.py : crop / zero-fill / boundary-anchored local-crop 연산

    train.py; 모델 학습, collapse 모니터링 로깅, 고정 쿼리 top-10 트래킹 — mined pair 기반 6-view 구조로 재구현 필요

    eval.py ;

        - fine-tuned SigLIP2, SigLIP2, Tianmu-MERE 비교

        - 주요 쿼리의 모델 별 Top-10 검색 결과 (train subset, test) 저장.

        - 개선된 사례와 악화된 사례 분석

        - 유사도를 정량적으로 평가가 가능하다면 정량화된 지표를 train / test 에 대해 report. **핵심 지표는 shape Recall@20** (`shape_recall.py` 재사용) — baseline(SigLIP2 0.134 / Tianmu-MERE 0.115, random 0.005)과 SFT 후 값을 나란히 비교 (2026-09-19 확정)

- scripts

    - download_glami_dresses.py : 800px archive를 하나씩 받아 dresses 이미지만 추출

    - download_checkpoints.py : SigLIP2, Tianmu-MERE, Grounding DINO, SAM checkpoint 다운로드

    - download_assets.sh : RunPod Network Volume에서 두 downloader를 순서대로 실행

    - extract_masks.sh ; data/masks 생성

    - mine_pairs.sh ; data/descriptors, data/pairs 생성

    - train.sh ; train data 사용

    - eval.sh ; test data 사용 — 미착수

```

### RunPod (L40S) 환경

연구 코드 작성/수정은 로컬에서 진행하고 git으로 관리한다. GPU가 필요한 작업만 RunPod pod(L40S 1장)에서 실행하며, RunPod에서는 git clone/pull로 최신 코드를 가져와 실행한다. Python 환경은 PyTorch/Transformers로 통일하며 JAX/TensorFlow 기반 `big_vision`은 설치하지 않는다.

- image: RunPod 기본 `Runpod Pytorch 2.8.0` 이미지 사용 (현재 확인된 container image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`). custom Docker image는 사용하지 않음.

- GPU: L40S 1장 (48GB VRAM, Ada Lovelace). 실제 pod에서 PyTorch 2.8.0+cu128, CUDA 사용 가능, NVIDIA L40S 인식 확인 완료. 마스크 추출(Grounded-SAM, 1회성 전처리), pair 마이닝(IoU 행렬곱), 학습(LoRA)은 단일 L40S 기준으로 진행.

- Persistent storage: RunPod **Network Volume**을 `/workspace`에 마운트. 실제 mount 확인 완료. 프로젝트 persistent 경로는 `/workspace/emb/{data,ckpts,experiments,cache,repo}` 사용. pod를 삭제/재생성해도 Network Volume의 데이터는 유지.

- Source workflow: 로컬에서 Codex로 코드 작성/수정 → git commit/push → GPU 실행이 필요할 때만 RunPod pod 시작 → `/workspace/emb/repo`에서 git clone/pull → GPU 작업 실행. RunPod에서 상시 코드 개발하지 않음.

- Python dependency: 기본 이미지에 없는 패키지는 repository의 `requirements.txt` 또는 `pyproject.toml`로 버전을 관리하고 pod에서 설치. 환경이 충분히 안정화되기 전에는 custom Docker image를 별도로 만들지 않음.

- Pod 관리: 현재는 RunPod UI로 생성/시작/중지/삭제. 반복 작업이 많아질 경우에만 `runpodctl`/`env/pod.sh` 자동화를 추가.

- Jupyter/포트: 기본적으로 사용하지 않음. Web terminal/SSH 중심으로 사용하고, 필요한 경우에만 포트를 명시적으로 연다.

- L40S(48GB) 기준: backbone을 LoRA로 freeze하여 full fine-tune 대비 메모리 부담을 줄임. bf16 사용, batch size는 실측 후 필요 시 gradient accumulation으로 조정.

- GLAMI-1M과 model checkpoint 다운로드는 환경 구축에 포함하지 않는다.

유의점:

1) fall-back 사용 금지, 차라리 에러가 떠서 문제를 파악하고 근본 원인을 고칠 수 있어야 함.

2) 코드의 유지 보수가 쉽게 작성하는 것을 선호하나, 지나치게 verbose 하게 코드 작성 금지.

3) pod 실행은 기본적으로 background 에서 실행할 수 있도록.

4) 코드 작성에 주석을 난잡하게 달지 말 것. 구현한 모듈/함수의 필수 설명만 달 것.

5) 통계치와 아이디어는 항상 직관적이고 이해하기 쉬운 것 사용. 절대 난잡한 가정 넣지 말 것.

네 역할:

- Senior deeplearning engineer.

- Highly professional in code optimization, deep learning

### Requirements

- Model

    - fine-tuned SigLIP2 (성능 평가)

    - Tianmu-MERE (baseline) : https://huggingface.co/TianmuLab/Tianmu-MERE

    - finetuning 안된 SigLIP2 (baseline) : google/siglip2-so400m-patch14-384

        - 확인됨: Tianmu-MERE의 base_model이 정확히 이 체크포인트(vision) + BAAI/bge-base-zh-v1.5(text) — 동일 backbone 전제 검증 완료

    - Grounded-SAM(grounding-dino-base + sam-vit-large) : 마스크 추출용 (positive pair 마이닝의 입력)

- Data

    - train dataset:

        - GLAMI-1M 800x800 버전; category_name = dresses

        - unique image_id 기준 dedup으로 26,326장 확보. **단, 이번 baseline에서는 그 중 무작위 4,000장 규모만 사용** (2026-09-19 확정 — 시간 제약. threshold를 0.90~0.95로 좁히면서 anchor 커버리지 보충을 위해 2,000 → 4,000으로 증량. 마스크 추출은 재실행 시 기존 결과를 건너뛰므로 필요해지면 그때 추가 생성)

        - item_id 레벨 텍스트 메타데이터는 image-to-image 과제라 불필요. 단 near-duplicate 제외용으로 `name` 컬럼만 사용

        - GLAMI-1M 공식 human-labeled test split은 사용하지 않음

        - 정식 train/val split 없음 (평가/모니터링 방식 참고)

    - test dataset:

        - 100장 원피스 사진, data/raw/images-800px-test/ 에 저장 (train 폴더와 분리) — 제공 완료.

        - 학습 및 validation 에는 절대 사용 금지

        - 모델 평가 및 분석 용도. mask 추출 결과는 전수 조사 대상.

### 작업 순서

0) 환경 구축 — RunPod L40S 1장 + 기본 PyTorch 2.8.0 image + Network Volume(/workspace) 구성 및 GPU/mount 검증 완료

1) 데이터 준비 — **원본 확보 완료 (2026-09-19)**: train 이미지 26,326장(`images-800px/`), test 100장(`images-800px-test/`, 분리 저장), `GLAMI-1M-dresses-train.csv`(29,350 rows), checkpoint 4종. 마스크 추출·descriptor 계산·pair 마이닝은 새 파이프라인으로 수행 필요 (2번의 선행 조건)

2) 유사도 정의 — 속성 기반 실루엣 축 확정, positive pair 생성 방식(shape-descriptor 마이닝) 확정, boundary-anchored local crop 확정

   - Positive pair 와 local crop (local-, global- view) 는 시각화 샘플 만들고 confirm 받을 것 (`view_sanity_check.ipynb`).

     - train 과 test data 모두에 대해서 확인할 것.

     - 특히 test에서 원피스가 잘 추출되는지 확인할 예정이며, 100장이므로 추출된 원피스는 직접 전수 조사함.

   - pretrained model(SigLIP2, Tianmu-MERE)이 우리가 정의한 shape 유사성을 얼마나 못 잡는지 보이는 자료 생성 (`notebooks/emb_analysis`, 상세는 File structure 참고). **완료 (2026-09-19)** — 결과는 "완전히 못 잡음"이 아니라 "약하게만 잡음, SFT로 강화 필요"로 프레임 확정.

     - (1) anchor별 shape-IoU top-5 vs 두 모델 임베딩 top-5 검색 결과를 나란히 비교하는 질적 그리드

     - (2) mined_pairs.csv의 positive를 proxy positive로 둔 Recall@20 vs random baseline 막대그래프 (정량). SigLIP2 0.134 / Tianmu-MERE 0.115 vs random 0.005. **SFT 후에도 동일 방식으로 재측정해 baseline과 나란히 비교 — eval.py의 핵심 지표로 채택**

     - (3) 임베딩 2D 투영 비교는 제외 (2026-09-19 확정, (1)(2)로 충분하다고 판단)

3) File structure 구체화 — **완료.** 확정된 마이닝/crop 방식 반영(본 문서), `src/dataset`, `src/tools`, `src/modules`, `src/func`, `src/model`, `train.py` 전부 구현 완료

4) 실험 결과 도출 — **학습 파이프라인 구현 + smoke run 완료 (2026-09-19), 본 학습은 대기**

   - `train.py` + `src/model/self_distill.py`: LoRA backbone + prototype head + DINO self-distillation, mined pair 기반 6-view 구조로 구현 완료

   - `configs/self_distill_v1.yaml`로 300-step smoke run 실행 — crash/NaN 없음, entropy가 collapse 없이 유지됨, 체크포인트·fixed-query 검색 결과 저장 확인 (`experiments/smoke_v1/`). **품질 확보용 본 학습(에폭 수, 배치 크기 등)은 별도 논의 후 진행 예정** — 이 smoke run 자체는 실제 체크포인트가 아님

   - eval.py/eval.sh - test 데이터(100장), 본 학습 완료 후 착수

   - notebooks/emb_analysis: SFT 전 SigLIP2 vs Tianmu-MERE shape recall 비교 완료 (M5, 위 참고). SFT 후 체크포인트는 본 학습 완료 후 추가 예정

유의 사항: 절대 미루어 짐작해서 코드를 작성하지 말 것. 명확하지 않은 부분은 항상 논의 후 결정 및 구현.
