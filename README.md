# LightEnhanceGAN — 저조도 이미지 향상용 경량 GAN

저조도(low-light) 이미지를 정상조도로 복원하는 **경량 U-Net Generator + PatchGAN Discriminator**.
임베디드 타깃(Jetson Orin Nano, 15 W, 30 FPS) 배포를 목표로 설계한 200 K 파라미터급 모델.

| 지표 | 값 |
|---|---|
| 파라미터 | **205,093** (≈ 205 K) |
| FLOPs (256×256) | **1.889 G** |
| LOL eval15 PSNR | **19.72 dB** |
| LOL eval15 SSIM | **0.823** |
| ONNX FP32 파일 크기 | **0.78 MB** |
| ONNX FP16 파일 크기 | **0.40 MB** |
| GPU FPS (PyTorch) | ≈ 160 (RTX 4060) |

학습 데이터셋: **LOL** ([Wei et al., BMVC'18](https://daooshee.github.io/BMVC2018website/)) — our485 (학습) + eval15 (평가) RGB 페어.

---

## 1. 빠른 시작 (Quick start)

### 1.1 의존성 설치

```bash
# 핵심 학습 의존성
pip install torch torchvision tqdm numpy pillow matplotlib

# (옵션) 평가 지표
pip install lpips

# (옵션) ONNX 변환 / 양자화
pip install onnx onnxruntime onnxconverter-common
```

### 1.2 데이터셋 배치

```
DataSet/LOLdataset/
├── our485/{low,high}/*.png       # 학습 페어 485장
└── eval15/{low,high}/*.png       # 평가 페어 15장
```

### 1.3 최종 모델 학습 (한 번에 Stage 1 + Stage 2 + 자동 평가)

```bash
python train_hybrid_v1_final.py \
    --data_root "C:/대학교/Projects/SmallSizePM_GAN_model/DataSet/LOLdataset"
```

중단되어도 같은 명령으로 이어서 재개됨 (매 epoch `*_last.pth` 저장).
결과 산출물:
- `checkpoints/hybrid_v1_stage1_best.pth`, `checkpoints/hybrid_v1_stage2_best.pth`
- `results/hybrid_v1_final/` — 비교 PNG, `comparison_table.csv`, `metrics_summary.txt`

### 1.4 ONNX 변환 + 양자화 + 벤치마크

```bash
python deploy/export_model.py \
    --checkpoint checkpoints/hybrid_v1_stage2_best.pth \
    --data_root "C:/대학교/Projects/SmallSizePM_GAN_model/DataSet/LOLdataset"
```

`deploy/models/` 에 `light_enhance_gan_{fp32,fp16,int8}.onnx` 생성 후
자동으로 PSNR/SSIM/FPS/크기 비교표 출력.

---

## 2. 최종 모델 — `hybrid_v1` 사양

4-stage U-Net, base_filters=32, attention(CA+SA) 유지.
**`input_conv` 만 표준 Conv2d**, 나머지 8 블록은 모두 **DSConv (Depthwise Separable Convolution)**.

```
input (3,256,256)
  └─ input_conv  [Standard Conv]   → 32 ×256  ─┐
       └─ enc1   [DSConv s=2]      → 64 ×128 ─┐│
            └─ enc2 [DSConv s=2]   → 128×64  ┐││
                 └─ enc3 [DSConv s=2] → 256×32││
                      └─ bottleneck [DSConv]  ││
                           └─ Attention(CA+SA)││
                                └─ dec3 [DSConv] +skip ─┘│
                                     └─ dec2 [DSConv] +skip ─┘
                                          └─ dec1 [DSConv] +skip
                                               └─ output_conv 1×1 → Tanh
                                                    → output (3,256,256)
```

- **입출력 범위**: `[-1, 1]` (Tanh)
- **conv_config**: `{input_conv: standard, enc1~3: dsconv, bottleneck: dsconv, dec1~3: dsconv}`
- **변경 방법**: [models/generator.py](models/generator.py) 의 `LightEnhanceGenerator(conv_config=...)` 인자로
  블록별 `"standard"` / `"dsconv"` 자유 지정 가능. 5가지 hybrid 변형(v1~v5) 의
  ablation 결과는 [experiments/hybrid_ablation.py](experiments/hybrid_ablation.py) 참고.

---

## 3. 2-Stage 학습 전략

1-stage GAN 학습 시 Discriminator 가 Generator 를 압도하여 PSNR 이 14-16 dB 부근에서 정체되는
현상이 LOL 같은 입출력 통계가 크게 다른 데이터셋에서 흔하다. ESRGAN, pix2pixHD 도 동일 패턴.
본 프로젝트는 **PSNR-oriented pre-training + GAN fine-tuning** 의 2-stage 전략을 사용.

| Stage | 목적 | Loss | lr | 에폭 |
|---|---|---|---|---|
| **1** | Generator 단독 supervised pre-training | `L1(1.0) + VGG(0.5) + SSIM(1.0)` | `1e-3`, cosine | 100 |
| **2** | GAN fine-tuning (텍스처 미세 보강) | `+ λ_adv(0.01)·BCE` | `1e-5` (G & D) | 50 |

### GAN 안정화 (Stage 2 에 모두 적용)

| 기법 | 설정 | 출처 |
|---|---|---|
| `d_update_freq=2` (G 2 step 당 D 1 step) | — | DCGAN community |
| One-sided label smoothing (real=0.9) | — | Salimans 2016 |
| Instance noise (σ=0.1, 20 epoch 선형 감쇠) | — | Sønderby 2017 |
| Spectral normalization on D | — | Miyato 2018 |
| D forward FP32 강제 (SN + AMP NaN 회피) | — | 본 프로젝트 |

### AMP 자동 비활성 정책

- Stage 1: `lr=1e-3 + AMP` → GradScaler overflow 로 step skip 누적 → **AMP 자동 off**
- Stage 2: `spectral_norm + AMP` → D weight NaN → **AMP 자동 off**

### Resume

매 epoch 끝에 `hybrid_v1_stage{1,2}_last.pth` 저장. 같은 명령 재실행 시 자동으로 이어서.
완료된 stage 는 `*_complete.flag` 파일 존재 → 건너뜀.
강제 재학습은 `--force_stage1` / `--force_stage2`.

---

## 4. 프로젝트 구조

```
CODES/
├── config.py                    TrainConfig 데이터클래스 (1-stage 학습용)
├── train.py                     1-stage GAN 학습 스크립트 (legacy)
├── train_two_stage.py           2-stage 학습 (legacy, base 모델용)
├── train_hybrid_v1_final.py     ★ 최종 모델 학습 (Stage 1 + 2 자동 + 평가)
├── evaluate.py                  단독 평가 스크립트
├── test_architecture.py         모델 구조 단위 테스트
├── test_dataloader.py           데이터 로더 단위 테스트
│
├── data/                        ── 데이터 패키지 (코드만, 데이터셋 자체는 외부)
│   ├── dataset.py                  LOLDataset (페어 로드, 정규화, augmentation 호출)
│   ├── dataloader.py               get_train_loader / get_eval_loader 팩토리
│   └── augmentation.py             paired geometric + low-only photometric aug
│
├── models/                      ── 신경망 정의
│   ├── generator.py                LightEnhanceGenerator + DSConv + ConvBlock + (CA, SA, LightAttention)
│   ├── discriminator.py            PatchGANDiscriminator (16×16 logits, spectral_norm 옵션)
│   └── losses.py                   CombinedLoss / SupervisedLoss / DiscriminatorLoss / PerceptualLoss / SSIMLoss
│
├── utils/                       ── 평가 / 분석 / 로깅
│   ├── metrics.py                  PSNR, SSIM, LPIPS, evaluate(), benchmark_model_full()
│   ├── model_analysis.py           파라미터 카운트, MAC/FLOPs hook 카운터, FPS 벤치마크
│   └── logger.py                   TrainLogger (CSV + 콘솔 + 샘플 PNG)
│
├── experiments/                 ── ablation / 비교 실험
│   ├── ablation.py                 5변형 비교 (full / no_attention / no_dsconv / small_ch / no_ssim)
│   ├── hybrid_ablation.py          DSConv↔Standard 블록 hybrid v1~v5
│   ├── comparison.py               우리 모델 vs Zero-DCE/EnlightenGAN/FUnIE-GAN 등
│   └── generate_tables.py          논문용 LaTeX 표 생성
│
└── deploy/                      ── 임베디드 배포
    ├── export_model.py             PyTorch → ONNX FP32 → FP16 → INT8 + 검증 + 자동 벤치마크
    └── benchmark.py                PyTorch + 3개 ONNX 모델 종합 벤치마크 (단독 실행 가능)
```

**산출물 (.gitignore 됨)**: `checkpoints*/`, `logs*/`, `results*/`, `experiments/{checkpoints,results}/`, `deploy/models/`

---

## 5. 손실 함수 (Loss)

### Stage 1 — `SupervisedLoss` ([models/losses.py](models/losses.py))

```
L_S1 = λ_L1 · L1(fake, real)
     + λ_VGG · L1(VGG16_relu3_3(fake), VGG16_relu3_3(real))   # perceptual
     + λ_SSIM · (1 - SSIM(fake, real))
```

기본 가중치: `λ_L1=1.0, λ_VGG=0.5, λ_SSIM=1.0`.

### Stage 2 — `CombinedLoss` ([models/losses.py](models/losses.py))

```
L_S2 = λ_adv · BCE(D(low ‖ fake), 1)   # cGAN, λ_adv=0.01 (작게)
     + λ_L1 · L1 + λ_VGG · VGG + λ_SSIM · SSIM
```

`λ_adv=0.01` 로 작게 유지하여 GAN 신호가 supervised loss 를 압도하지 않게.

### Discriminator — `DiscriminatorLoss`

```
L_D = 0.5 · ( BCE(D(real_pair), 0.9) + BCE(D(fake_pair), 0.0) )   # one-sided label smoothing
```

---

## 6. 배포 (Deploy) 결과

`deploy/export_model.py` 의 최근 실측 (LOL eval15, 256×256, batch=1, CPU=Intel Win11 노트북):

| Model | Size | PSNR | SSIM | FPS (GPU) | FPS (CPU) | 비고 |
|---|---|---|---|---|---|---|
| PyTorch FP32 (.pth) | 10.4 MB | 19.72 | 0.823 | ≈ 160 | 20 | optimizer 상태 포함 |
| **ONNX FP32** | **0.78 MB** | **19.72** | **0.823** | — | **109** | dynamic batch, opset 17 |
| **ONNX FP16** ★ | **0.40 MB** | **19.72** | **0.823** | — | 75 | 손실 0.003 dB, **deploy 권장** |
| ONNX INT8 (dynamic) | 0.24 MB | 18.04 | 0.744 | — | 5.5 | ⚠️ PSNR -1.68 dB |

ONNX 검증: PyTorch ↔ ONNX FP32 max abs diff = **9.08 × 10⁻⁶** (수치적으로 일치).

**INT8 가 느린 이유**: Conv-heavy 모델에 dynamic quantization 적용 시 매 inference 마다
dequantize → conv → quantize 가 일어남. 정적(static) quantization + calibration
data 가 필요한 케이스. 임베디드 실배포에는 **FP16 권장**.

**GPU FPS = "—"**: 위 결과는 `onnxruntime` (CPU only) 기준. GPU 벤치마크가 필요하면:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
python deploy/benchmark.py \
    --checkpoint checkpoints/hybrid_v1_stage2_best.pth \
    --data_root "C:/대학교/Projects/SmallSizePM_GAN_model/DataSet/LOLdataset"
```

---

## 7. 학술적 근거 (Design rationale)

| 컴포넌트 | 출처 | 채택 이유 |
|---|---|---|
| U-Net 4-stage + skip | FUnIE-GAN [Islam et al., RA-L 2020] | 저시점 카메라의 fine-grained texture 보존 |
| Depthwise Separable Conv | Zero-DCE++ [Li et al., TPAMI 2021] | 표준 conv 대비 MAC 8-9× 절감 |
| CA + SA Attention | CBAM [Woo et al., ECCV 2018] | 저조도 영역에 큰 가중치를 학습으로 유도 |
| Conditional PatchGAN | pix2pix [Isola et al., CVPR 2017] | 국소 사실성 강조, 차선/표지 보존 |
| Spectral Norm | Miyato et al., ICLR 2018 | D 의 Lipschitz ≤ 1 제약, mode collapse 완화 |
| Instance Noise | Sønderby et al., 2017 | D 입력에 noise 주입으로 G 학습 신호 유지 |
| 2-stage (PSNR pre-train + GAN finetune) | ESRGAN, pix2pixHD | LOL 처럼 통계 차이가 큰 데이터셋에서 PSNR 정체 회피 |

---

## 8. 알려진 이슈 / 제약

1. **AMP off**: Stage 1 (lr 큼), Stage 2 (SN) 모두 AMP 비활성. RTX 4060 기준 epoch 당 ~30s.
2. **INT8 동적 양자화**: 위 §6 참고. Static quant 로 전환 필요.
3. **ONNX 배치 축**: FP32/FP16 은 dynamic, INT8 만 static batch=1
   (symbolic shape inference 한계 회피).
4. **Windows cp949 콘솔**: 모든 학습 스크립트가 `sys.stdout.reconfigure("utf-8")` 호출하여
   한글 출력 안전.
5. **LPIPS**: `lpips` 라이브러리 미설치 시 NaN 으로 출력 (학습/평가 자체는 영향 없음).

---

## 9. 라이센스 / 인용

학술/연구 목적 코드. LOL 데이터셋의 원본 라이센스는 별도 확인 필요
([daooshee.github.io/BMVC2018website](https://daooshee.github.io/BMVC2018website/)).

본 코드를 활용 시 LightEnhanceGAN (hybrid_v1) 항목으로
`experiments/comparison.py` 가 출력하는 LaTeX 표를 그대로 사용 가능.

---

## 10. 자주 쓰는 명령 모음

```bash
# 최종 모델 풀 학습 + 평가
python train_hybrid_v1_final.py --data_root <LOL_path>

# Stage 2 만 강제 재시작 (Stage 1 best 그대로 활용)
python train_hybrid_v1_final.py --data_root <LOL_path> --force_stage2

# 학습 건너뛰고 best 체크포인트로 평가만
python train_hybrid_v1_final.py --data_root <LOL_path> --skip_train

# ONNX 변환 + 양자화 + 벤치마크
python deploy/export_model.py \
    --checkpoint checkpoints/hybrid_v1_stage2_best.pth \
    --data_root <LOL_path>

# 변환 없이 벤치마크만 (이미 .onnx 가 있을 때)
python deploy/benchmark.py \
    --checkpoint checkpoints/hybrid_v1_stage2_best.pth \
    --data_root <LOL_path>

# Ablation (5 변형 비교)
python experiments/ablation.py --data_root <LOL_path>

# Hybrid 변형 비교 (v1~v5)
python experiments/hybrid_ablation.py --data_root <LOL_path>

# 우리 모델 vs 기존 방법 비교표
python experiments/comparison.py \
    --checkpoint checkpoints/hybrid_v1_stage2_best.pth \
    --data_root <LOL_path>
```
