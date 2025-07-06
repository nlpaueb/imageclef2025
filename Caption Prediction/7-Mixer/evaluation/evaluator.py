import pandas as pd
import numpy as np
import re
import string
import torch
from tqdm import tqdm
import os
import base64

from bert_score import BERTScorer
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import evaluate
from evaluation.alignscore.src.alignscore import AlignScore
from evaluation.medcat_scorer import MedCatScorer
from evaluation.medimageinsight.medimageinsightmodel import MedImageInsight

# ========= SETUP PATHS =========
image_path = "./all_images/"
medcat_model_path = "./evaluation/umls_self_train_model_pt2ch_3760d588371755d0.zip"
medimage_model_path = "./evaluation/medimageinsight/2024.09.27/"

# ========= PREPROCESSING =========
def preprocess_caption(caption, case_sensitive=False):
    translator = str.maketrans("", "", string.punctuation)
    number_regex = re.compile(r"\d+")
    if not case_sensitive:
        caption = caption.lower()
    caption = number_regex.sub("number", caption)
    caption = caption.translate(translator)
    return caption

# ========= LOAD MODELS ONCE =========
def load_evaluation_models(device="cuda:0"):
    models = {}

    models["bertscorer"] = BERTScorer(
        model_type="microsoft/deberta-xlarge-mnli",
        batch_size=64,
        device=device
    )

    models["bleurt_model"] = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12").eval().to(device)
    models["bleurt_tokenizer"] = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")

    models["alignscore"] = AlignScore(
        model="roberta-large",
        batch_size=32,
        device=device,
        ckpt_path="./evaluation/alignscore/models/AlignScore/AlignScore-large.ckpt",
        evaluation_mode="nli_sp",
        verbose=False,
    )

    models["medcat"] = MedCatScorer(model_path=medcat_model_path)

    models["medimage"] = MedImageInsight(
        model_dir=medimage_model_path,
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model/clip_tokenizer_4.16.2"
    )
    models["medimage"].load_model()

    return models

# ========= METRIC FUNCTIONS =========
def compute_bertscore(candidates, references, scorer):
    scores = []
    for cand, ref in zip(candidates, references):
        if len(cand) == 0 and len(ref) == 0:
            scores.append(1)
        else:
            _, _, F1 = scorer.score(
                [preprocess_caption(cand)], [preprocess_caption(ref)]
            )
            scores.append(F1.item())
    return np.mean(scores)

def compute_rouge(candidates, references):
    rouge = evaluate.load("rouge")
    scores = []
    for cand, ref in zip(candidates, references):
        if len(cand) == 0 and len(ref) == 0:
            scores.append(1)
        else:
            result = rouge.compute(
                predictions=[preprocess_caption(cand)],
                references=[preprocess_caption(ref)],
                use_aggregator=False,
                use_stemmer=False,
            )
            scores.append(result["rouge1"])
    return np.mean(scores)

def compute_bleurt(candidates, references, model, tokenizer):
    scores = []
    device = next(model.parameters()).device 
    with torch.no_grad():
        for ref, cand in zip(references, candidates):
            inputs = tokenizer(
                [preprocess_caption(ref)],
                [preprocess_caption(cand)],
                padding='longest',
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()} 
            output = model(**inputs).logits.flatten().item()
            scores.append(output)
    return np.mean(scores)

def compute_alignscore(candidates, references, scorer):
    scores = []
    for cand, ref in zip(candidates, references):
        if len(cand) == 0 and len(ref) == 0:
            scores.append(1)
        else:
            score = scorer.score(contexts=[preprocess_caption(ref)], claims=[preprocess_caption(cand)])[0]
            scores.append(score)
    return np.mean(scores)

def compute_medcat(candidates, references, scorer):
    scores = []
    for cand, ref in zip(candidates, references):
        if len(cand) == 0 and len(ref) == 0:
            scores.append(1)
        else:
            score = scorer.score(preprocess_caption(ref), preprocess_caption(cand))
            scores.append(score)
    return np.mean(scores)

def compute_similarity(ids, candidates, references, scorer):
    scores = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for id_, cand, ref in zip(ids, candidates, references):
        image_file = os.path.join(image_path, id_ + ".jpg")
        if not os.path.exists(image_file):
            scores.append(1)
            continue
        if len(cand) == 0 and len(ref) == 0:
            scores.append(1)
            continue

        with open(image_file, "rb") as f:
            img_b64 = base64.encodebytes(f.read()).decode("utf-8")

        try:
            embeddings = scorer.encode(images=[img_b64], texts=[preprocess_caption(cand)])
            v = torch.tensor(embeddings["image_embeddings"][0]).to(device)  
            c = torch.tensor(embeddings["text_embeddings"][0]).to(device)   

            cos = torch.dot(c, v) / (torch.norm(c) * torch.norm(v))
            score = 2.5 * max(cos.item(), 0)
        except Exception as e:
            print("Error:", e)
            score = 1

        scores.append(score)
    return np.mean(scores)

# ========= COMPUTE WEIGHTED AVERAGE =========
def compute_weighted_average(ids, candidates, references, models):
    weights = {
        "BERTScore": 1/6,
        "ROUGE": 1/6,
        "BLEURT": 1/6,
        "AlignScore": 1/6,
        "MedCAT": 1/6,
        "Image-Caption Similarity": 1/6
    }

    scores = {
        "BERTScore": compute_bertscore(candidates, references, models["bertscorer"]),
        "ROUGE": compute_rouge(candidates, references),
        "BLEURT": compute_bleurt(candidates, references, models["bleurt_model"], models["bleurt_tokenizer"]),
        "AlignScore": compute_alignscore(candidates, references, models["alignscore"]),
        "MedCAT": compute_medcat(candidates, references, models["medcat"]),
        "Image-Caption Similarity": compute_similarity(ids, candidates, references, models["medimage"]),
    }

    weighted_average = sum(weights[k] * scores[k] for k in weights)
    
    return weighted_average, scores
