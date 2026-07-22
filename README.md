# Named Entity Recognition using DistilBERT

Production-ready Streamlit deployment for a saved DistilBERT Named Entity Recognition model. The application performs inference only using the resources already stored in `Best_DistilBERT_Model/`.

## Project Overview

This project loads a trained Hugging Face DistilBERT token classification model from local files and predicts named entities for user-provided text. It does not retrain, fine-tune, download, recreate, or modify the model or tokenizer.

## Features

- Local-only loading from `Best_DistilBERT_Model/`
- Cached DistilBERT model, tokenizer, and label mapping
- Single-sentence and multi-sentence inference
- Softmax confidence scores for every token
- Prediction table with token, label, and confidence
- Colored entity highlighting
- Extracted entity groups with duplicate removal
- Statistics dashboard
- Token confidence chart
- Entity distribution chart
- CSV and JSON downloads
- Professional Streamlit UI with sidebar resource status
- Graceful handling for empty input, missing files, and prediction errors

## Installation

Create and activate a Python environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Folder Structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
└── Best_DistilBERT_Model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── idx2tag.pkl
    ├── training_args.bin
    └── ner_transformer_distilbert.ipynb
```

## Run Instructions

Run the application from the project root:

```bash
streamlit run app.py
```

The app automatically loads all required resources from:

```text
Best_DistilBERT_Model/
```

No manual file selection is required.

## Example Usage

1. Start the Streamlit app.
2. Type or paste text into the main text area.
3. Click **Predict**.
4. Review the prediction table, highlighted text, extracted entities, statistics, and charts.
5. Download results as CSV or JSON when needed.

Example sentence:

```text
John works at Google in London. Ahmed joined Microsoft in Cairo for the FIFA technology summit.
```

## Screenshots

Add screenshots of the running application here after deployment.

## Inference Policy

This deployment is inference-only:

- No retraining
- No fine-tuning
- No model downloads
- No tokenizer creation
- No model modification
- No replacement of saved artifacts

All predictions are generated from the existing saved DistilBERT model and tokenizer files.
