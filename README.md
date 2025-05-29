AUEB NLP Group at ImageCLEFmedical Caption 2025

This repository contains the Concept Detection, Caption Prediction and Explainability models developed by the AUEB NLP Group as part of our participation in the 9th edition of the ImageCLEFmedical Caption evaluation campaign (2025).

Concept Detection

📌 Task Overview
The Concept Detection task involves multi-label classification over a set of 2,479 biomedical concepts, each corresponding to a UMLS Concept Unique Identifier (CUI). The goal is to assign all relevant medical concepts to a given radiology image, capturing both high-level modalities (e.g., CT, MRI) and fine-grained anatomical or pathological terms.

🧪 Our Approach
Our submitted systems were based on deep learning architectures, integrating both feature-based and retrieval-based strategies:

CNN + Feed-Forward Classifier: We used various CNN backbones (e.g., ResNet, EfficientNet) pre-trained on ImageNet to extract image embeddings, followed by fully connected layers to perform multi-label prediction.
Image Retrieval + k-NN Tag Propagation: We developed a retrieval-based method using dense image embeddings. Given a query image, the system retrieves its most similar neighbors from the training set and predicts concepts based on tag frequency among the top-k.
Modality-Specific Enhancements: Motivated by lower performance on ultrasonography images, we explored targeted fine-tuning and label masking techniques to specialize predictions on this modality (see paper for details).
Ensemble Models: Our final submission included ensembles of the above systems. These combined predictions using union- and confidence-weighted aggregation strategies to boost robustness and generalization.
🏆 Performance
Our Concept Detection system ranked 1st among all participating teams in the 2025 challenge, demonstrating strong generalization across imaging modalities and a competitive balance between precision and recall.
