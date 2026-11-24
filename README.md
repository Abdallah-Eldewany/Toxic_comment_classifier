# Toxic Comment Classifier

Multi-label toxic comment detection with a Bidirectional GRU, trained on the Jigsaw toxic comments dataset and deployed as a live demo.

**Live demo:** https://huggingface.co/spaces/Abdallah-Eldewany/toxic-comment-classifier

## What it does
Given a comment, the model scores six labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. The demo flags every label whose score passes an adjustable threshold.

## Pipeline (end to end)
1. Text cleaning (`preprocess.py`): lowercasing, removing links and mentions, expanding contractions, undoing symbol disguises, keeping letters only, shortening repeated letters.
2. Tokenizing and padding (30,000 words, max length 200).
3. Model training and comparison in the notebook: Bidirectional GRU vs Bidirectional LSTM.
4. Export of the model (`model.keras`) and tokenizer (`tokenizer.pkl`).
5. Deployment as a Gradio app on Hugging Face Spaces (`app.py`). Inference runs in plain NumPy, so no TensorFlow is needed at serving time.

## Results (validation AUC)
| Model | Best validation AUC |
|---|---|
| Bidirectional GRU | 0.9704 (epoch 4) |
| Bidirectional LSTM | 0.9649 (epoch 3) |

The GRU was chosen for deployment. These are validation scores on the held-out split, not production accuracy.

## Use the API
```python
from gradio_client import Client

client = Client("Abdallah-Eldewany/toxic-comment-classifier")
result = client.predict(
    comment="You are so stupid and nobody likes you.",
    threshold=0.5,
    api_name="/predict",
)
print(result)
```

## Files
- `Toxic_Comment_Classification_GRU_LSTM.ipynb`: data, training, comparison, experiments log
- `preprocess.py`: text cleaning
- `app.py`: Gradio app
- `model.keras`, `tokenizer.pkl`: trained model and tokenizer
- `requirements.txt`: serving dependencies

## Limitations
The model was trained on English text with correct spelling. Slang, heavy misspellings and sarcasm can fool it.
