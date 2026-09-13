# NLP Alert Generation Pipeline

End-to-end NLP system that converts raw English text into a short, natural-language alert by combining three complementary components:

1. **Named Entity Recognition** — a custom BiLSTM model with Word2Vec embeddings
2. **Sentiment Analysis** — a Naive Bayes classifier implemented from scratch
3. **Alert Generation** — Google **FLAN-T5-Large** (Hugging Face Transformers)

Given a title and a text sample, the pipeline extracts entities, predicts sentiment, and conditions a sequence-to-sequence model on both signals to generate a concise alert grounded in the source text.

> **Team project** developed as part of NLP coursework at Universidad Pontificia Comillas (ICAI). I contributed across the full pipeline — preprocessing, model development, training, integration, inference, testing, and debugging.

## Pipeline

```
Title + Text → Preprocessing → BiLSTM NER ──┐
                              → Naive Bayes ─┼→ Prompt → FLAN-T5-Large → Alert
                                Sentiment ────┘
```

## Example

**Input**
```
Title: The Batman
Text: Matt Reeves delivers a dark and visually stunning film,
      with Robert Pattinson giving an outstanding lead performance.
```

**Output**
```
Entities:  Matt Reeves [PER], Robert Pattinson [PER]
Sentiment: Positive
Alert:     A short, generated summary grounded in the input text.
```

## Tech Stack

Python · PyTorch · Gensim (Word2Vec) · Hugging Face Transformers (FLAN-T5-Large) · spaCy · NumPy · Pandas

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/nlp-alert-generation.git
cd nlp-alert-generation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

Trained checkpoints for NER and sentiment are included; FLAN-T5-Large loads from Hugging Face on first run.

## Repository Structure

```
src/        pipeline, NER, sentiment, generation modules
models/     trained checkpoints (LSTM, Naive Bayes)
data/       dataset instructions (raw datasets not redistributed)
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for full technical details, model configuration, training setup, and evaluation notes.

## Author

Team project developed at Universidad Pontificia Comillas (ICAI).
Portfolio version maintained by **Íñigo Serrano**.
