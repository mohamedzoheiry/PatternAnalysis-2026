# Diagnosis of Alzheimer's Disease using GFNet on ADNI Brain MRI Dataset
## Introduction

### Problem Description

Alzheimer’s disease is a common form of dementia that affects memory, thinking skills, and the ability to complete daily tasks. Alzheimer’s causes progressive brain atrophy, which becomes visible as shrinkage of the hippocampus and cortex, and enlargement of the ventricles. As such, deep learning models can be employed to learn to detect such changes in the structure of the brain, supporting early screening so that patients can receive early treatment and better outcomes. The purpose of this project is to train and use a Global Filter Network (GFNet) Model to classify brain MRI slices as either having Alzheimer’s Disease (AD) or being Cognitively Normal (CN). The GFNet Model will be compared against a simple baseline model to evaluate whether its added complexity and computational cost provide a meaningful improvement in classification performance. The model will be trained on the ADNI (Alzheimer’s Disease Neuroimaging Initiative) dataset, which contains T1-weighted structural brain MRI scans from AD and CN subjects.

### Algorithm Description

GFNet [1] is a vision-transformer style architecture that replaces the self-attention sub-layer with a global filter layer. Each 224x224 input image is split into 16x16 patches, giving a 14x14 grid of 196 tokens. Each patch is linearly embedded into a 256-dimensional token, and a learnable positional embedding is added. The tokens then pass through a stack of 12 GFNet Blocks. Each of these blocks applies layer normalisation, the global filter layer, a second layer normalisation, and a feed-forward MLP, with a single residual connection around the whole block, as per the official implementation [2].

The global filter layer consists of three main operations:

1. A 2D discrete Fourier Transform to convert the input features to the frequency domain.
2. An element-wise multiplication between the frequency domain features and the global filters.
3. A 2D inverse Fourier transform to convert the features back to the spatial domain.

This is equivalent to a circular convolution with a kernel as large as the whole token grid, so every region of the brain can influence every other region in a single layer. It does this at O(L log L) cost in the number of tokens L, compared with O(L²) for self-attention.

After the final block, a layer normalisation is applied, the tokens are averaged (global average pooling), and a linear classification head outputs the AD/CN prediction. The network is trained with cross-entropy loss.

![GFNet architecture](images/gfnet_architecture.png)

## Feasibility Review

**Intended user:** Radiologists and other medical professionals who review T1-weighted brain MRIs for Alzheimer’s disease. The prototype is an assistant for diagnosis, and not a replacement. It handles high-confidence AD/CN cases and forwards ambiguous, low-confidence cases to a specialist for review.

**Scope:** Binary AD vs CN classification from the 20 pre-processed 2D slices per patient in the ADNI dataset. GFNet is compared against a simple CNN baseline on identical training/validation/test splits. Performance is reported at both the slice and patient level.

**Acceptance criteria:**

- **Accuracy:** GFNet must achieve >=80% slice-level accuracy on the test set.
- **Quantitative metrics:** Per-class precision and recall, macro-F1 and AUROC must be reported. AD recall has the highest priority because missed AD cases have the most severe consequences.
- **Confidence:** A confidence threshold will be chosen on the validation set such that cases above it are automated, and cases below are forwarded to a specialist for review.
- **Resources:** Peak VRAM <= 8GB, parameter count <= 25M. Inference latency per patient is reported for both models.

**Model Choice and Course Concepts:**

- **Model:** GFNet (Hard Difficulty)
- **Baseline:** A simple CNN with 4 blocks (Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2), followed by global average pooling and a linear head.
- **Dataset:** ADNI brain MRI dataset

**Preliminary results:**

A data audit was conducted to understand the properties of the dataset, and to verify that the train, validation, and test splits are separated at the patient level with no leakage.

```text
== Counts ==
split  class   images  patients
train  AD       10400       520
train  NC       11120       556
test   AD        4460       223
test   NC        4540       227

== Image properties (sample of 200) ==
sizes: {(256, 240): 200}
modes: {'L': 200}
intensity range: 0..255

== Leakage checks ==
patients in both train and test: 0
patients in both AD and NC (train): 0
patients in both AD and NC (test): 0
PASS: no patient-level leakage

== Slices per patient ==
train  AD    min=20 max=20 most common=[(20, 520)]
train  NC    min=20 max=20 most common=[(20, 556)]
test   AD    min=20 max=20 most common=[(20, 223)]
test   NC    min=20 max=20 most common=[(20, 227)]

== Validation split (patients) ==
train AD=442 NC=473 total=915 AD share=0.48
val   AD=78 NC=83 total=161 AD share=0.48
test  AD=223 NC=227 total=450 AD share=0.50

Saved patient lists to splits/val_split.json
```

The dataset is roughly balanced, with AD patients making up 48-50% of total patients across all splits. Every patient has exactly 20 slices. Leakage checks confirm that no patient ID appears in more than one split or in both classes. The image have uniform size and intensity range, so preprocessing only requires resizing (cropping) and normalisation.

**Risks:**

- **Overfitting:** GFNet can memorise the training set instead of learning patterns. This can be mitigated using data augmentation, and a stronger model.
- **Asymmetric error cost:** A false negative is far more costly than a false positive. To mitigate this, AD recall must be prioritised when selecting an operating threshold.
- **Hidden leakage:** The leakage check above only checks patient IDs. If one patient has more than one ID, their scans can appear in both training and test sets, which cannot be detected.

**Budget:** A single A100 GPU on Rangpur, with training limited to 45 minutes per run. The main constraint is queue wait times and GPU availability.

**Next planned experiment:** Train both models with data augmentation, and measure performance on the test set.

**Fallbacks:**

- **Overfitting:** Can reduce depth or embedding dimension, increase stochastic depth, strengthen augmentation.
- **Underfitting:** Can increase depth or embedding dimension, or train for more epochs.
- **Patient-level aggregation:** Can average predictions across each patient’s 20 slices to smooth out noisy individual slices.

## Dependencies + Reproducibility of Results

This project has the following dependencies:

- python - v3.11.16
- torch - v2.13.0
- torchvision - v0.28.0
- numpy - v2.4.6
- pillow - v12.3.0
- scikit-learn - v1.9.0
- matplotlib - v3.11.1

**Reproducibility of Results:**


## Inputs, Outputs, Plots

## Data Preprocessing and Training

## Investigation of the Open Research Dilemma

### Quantitative Benchmarking

### Resource Profiling

### Qualitative Error Autopsies

### Recommendation

## AI Disclosure

## References