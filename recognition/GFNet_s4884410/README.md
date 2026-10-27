# Diagnosis of Alzheimer's Disease using GFNet on ADNI Brain MRI Dataset

## Introduction
### Algorithm Description

### Problem Description

## Feasibility Review
1.	User need, Scope & Acceptance Criteria

**Intended User**: Radiologists / Medical Professionals who review T1 Brain MRIs for Alzeimer’s disease. Can be used to screen for confident AD/CN cases, and to forward ambiguous cases to a specialist for review.

**Scope**: AD vs CN classification from the 20 pre-processed slices per patient. GFNet is compared against a simple CNN baseline on identical splits.

**Acceptance Criteria**: 
a.	GFNet must achieve >= 80% slice accuracy on test slices.

b.	Resources: peak VRAM <= X GB, training time <= 30 mins on A100 GPU, parameter count <= X

c.	Precision, macro-F1, AUROC.

d.	???

2.	Model Choice and Course Concepts

**Model**: GFNet (Hard difficulty)

**Baseline**: Simple CNN with 4 blocks (Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2)

**Dataset**: ADNI Brain MRI Dataset.

3.	Preliminary Feasibility Evidence
```
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
4.	Risks, Budget & Fallback
Risks:

a.	Overfitting: Memorising training set can mean that the model fails to recognise patterns and cannot perform accurately on test data (for example, first GFNet training run). Can address this my modifying size of model (changing parameter count), as well as data augmentation.

b.	False negative vs false positive: Risk and cost of false negative is significantly higher than that of a false positive. 

c.	Cannot guarantee that there is no leakage between the data splits. A single patient may have more than one ID, causing them to appear in both the test and training splits.

d.	GFNet may not outperform the baseline model.

Budget:

a.	Must run on A100 GPU

b.	Training time + resource accessibility


Fallbacks:

If GFNet cannot exceed 80% accuracy, or beat the baseline model, we can attempt to fix this by:

1.	Modifying size of GFNet model (Increasing depth / parameter count)
2.	Increasing stochastic depth (0.1 -> 0.2-0.3)

## Dependencies

## Data Splitting and Preprocessing

## Training and Validation of Models

## Investigation of the Open Research Dilemma
### Quantitative Benchmarking of GFNet against Baseline CNN Model

### Resource Profiling

### Qualitative Error Autopsies

### Engineering Recommendation and Trade-off Synthesis

## Artificial Intelligence Usage Disclosure

## References
Must include: GFNet paper, GFNet github repo, timm github repo. CITE IN SCRIPTS AND HERE.