# Multi-modal-Sentiment-Analysis-Dashboard
A late-fusion NLP and Computer Vision model built with DistilBERT and ViT.

# 🚀 Multi-Modal Sentiment Analysis Engine

Bridging NLP and Computer Vision for Context-Aware AI

# 📌 Project Overview

Traditional sentiment analysis relies solely on text, often missing crucial emotional context from visual cues (e.g., sarcasm or environmental atmosphere). This project implements a Multi-Modal Late-Fusion Architecture that processes both textual captions and associated images to predict sentiment with higher reliability.

# 🛠️ Tech Stack

1. Language: Python 3.12
2. Deep Learning: PyTorch
3. NLP Encoder: DistilBERT (via Hugging Face)
4. Vision Encoder: ViT - Vision Transformer (via Hugging Face)
5. Interface: Gradio (for real-time inference)

# Environment: 

1. Google Colab (T4 GPU)

# Architecture & Logic

This model utilizes a Late-Fusion strategy:

1. Text Branch: Uses distilbert-base-uncased to extract a 768-dimensional feature vector from the text.
2. Image Branch: Uses google/vit-base-patch16-224 to extract a 768-dimensional feature vector from the image pixels.
3. Fusion Head: The vectors are concatenated (1536-dim) and passed through a custom fully-connected MLP with Dropout layers to prevent overfitting.

# The "Correction" Phase (Alignment Training)

1. The Problem: Upon initial integration, the model yielded a near-random distribution (approx. 33% accuracy across 3 classes). This occurred because while the encoders were pre-trained, the fusion weights were randomly initialized.

2. The Solution: I implemented a Supervised Alignment Loop. By fine-tuning the classification head on a specific dataset (e.g., the "Burger" example), I "corrected" the model’s internal weights. This taught the model to:

  a) Identify positive visual signals (vibrant colors, high-quality food).
  b) Correctly weigh textual adjectives (e.g., "delicious") against visual evidence.
  c) Move the probability distribution from a confused state to a high-confidence prediction.

# 📊 Results

1. Improved Accuracy: Post-correction accuracy improved significantly over the base random initialization.
2. Robustness: Handles cases of "Modality Conflict" (e.g., negative text with positive images) by weighing the fusion head.
