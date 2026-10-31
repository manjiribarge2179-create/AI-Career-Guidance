# Career Guidance AI — 10th & 12th Students

A custom-trained LLM chatbot for career guidance. Trained **only on our own dataset** — no Gemini, OpenAI, or any external API at the backend.

## Project Structure

```
career-guidance-llm/
├── data/
│   ├── career_dataset.json     # 40+ Q&A pairs for training
│   ├── career_guidance.db      # SQLite database (auto-created)
│   ├── career_model.pth        # Trained model weights (after training)
│   └── vocab.pkl               # Vocabulary (after training)
├── model.py                    # Transformer model definition
├── train.py                    # Training script
├── chatbot.py                  # Inference engine (greedy decode + TF-IDF fallback)
├── database.py                 # SQLite DB: careers, exams, colleges
├── app.py                      # Flask web app
├── templates/
│   └── index.html              # Web UI
└── requirements.txt
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This trains a Transformer model on `data/career_dataset.json` and saves:
- `data/career_model.pth` — model weights
- `data/vocab.pkl` — vocabulary

Training runs for 100 epochs. On CPU it takes ~2-5 minutes.

### 3. Start the web app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

## How It Works

- **Model**: A small Transformer (encoder-decoder) with ~3M parameters trained from scratch on our dataset.
- **Inference**: Greedy decoding. Falls back to TF-IDF cosine similarity over the dataset when model output is too short.
- **Database**: SQLite stores careers, streams, entrance exams, and top colleges.
- **No external APIs**: Zero dependency on Gemini, OpenAI, or any cloud LLM.

## Adding More Training Data

Edit `data/career_dataset.json` and add more `{"input": "...", "response": "..."}` pairs, then re-run `python train.py`.
