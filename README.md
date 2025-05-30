# AUEB NLP Group at ImageCLEFmedical Caption 2025

This repository contains the **Concept Detection**, **Caption Prediction**, and **Explainability** models developed by the **AUEB NLP Group** for the **9th edition of the ImageCLEFmedical Caption evaluation campaign (2025)**.

---

## 🔍 Concept Detection

### 📌 Task Overview

The **Concept Detection** task is a **multi-label classification** problem involving a set of **2,479 biomedical concepts**, each represented by a **UMLS Concept Unique Identifier (CUI)**. The goal is to assign all relevant medical concepts to a given radiology image—covering both high-level modalities (e.g., *CT*, *MRI*) and fine-grained anatomical or pathological entities.

---

### 🧪 Our Approach

Our submitted systems were based on **deep learning architectures**, combining both feature-based and retrieval-based strategies:

- **CNN + Feed-Forward Classifier**  
  We used various CNN backbones (e.g., **ResNet**, **EfficientNet**) pre-trained on **ImageNet** to extract image embeddings, followed by fully connected layers for multi-label prediction.

- **Modality-Specific Enhancements**  
  To address lower performance on **ultrasonography** images, we experimented with **targeted fine-tuning** and **label masking** strategies that specialized model behavior for this modality (see our paper for full details).

- **Ensemble Models**  
  Our final submission included **ensembles** combining outputs from all models above using **union-based** and **intersection-based aggregation** strategies. This boosted robustness and helped capture complementary signals across systems.

---

### 🏆 Performance

Our **Concept Detection system ranked 1st** among all participating teams in the 2025 challenge. It demonstrated strong generalization across imaging modalities and achieved a well-balanced trade-off between precision and recall.

---

## 📂 Repository Structure

