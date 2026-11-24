import io
import os
import pickle
import zipfile

import gradio as gr
import h5py
import numpy as np
import spaces  # needed on ZeroGPU Spaces

from preprocess import clean_text

MAX_LEN = 200  # same as training
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HERE = os.path.dirname(os.path.abspath(__file__))


# ---- Load the tokenizer without TensorFlow or Keras ----
class _Stub:
    """Stand-in for the Keras Tokenizer class: just keeps its saved data."""
    def __init__(self, *args, **kwargs):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("keras", "tensorflow", "tf_keras")):
            return _Stub
        return super().find_class(module, name)


with open(os.path.join(HERE, "tokenizer.pkl"), "rb") as f:
    tok = _Unpickler(f).load()
word_index = tok.word_index
num_words = tok.num_words
oov_index = word_index.get(tok.oov_token) if tok.oov_token else None


def to_ids(text):
    ids = []
    for w in text.split():
        i = word_index.get(w)
        if i is not None:
            if num_words and i >= num_words:
                if oov_index is not None:
                    ids.append(oov_index)
            else:
                ids.append(i)
        elif oov_index is not None:
            ids.append(oov_index)
    ids = ids[:MAX_LEN]
    return np.array(ids + [0] * (MAX_LEN - len(ids)))  # pad with 0 at the end


# ---- Load the model weights (plain numpy, no TensorFlow) ----
with zipfile.ZipFile(os.path.join(HERE, "model.keras")) as z:
    h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
    def var(path, i):
        return np.array(h5["layers/" + path + "/vars/" + str(i)])
    emb = var("embedding", 0)
    gru = {d: [var(f"bidirectional/{d}_layer/cell", i) for i in range(3)] for d in ("forward", "backward")}
    d1 = [var("dense", 0), var("dense", 1)]
    d2 = [var("dense_1", 0), var("dense_1", 1)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_gru(x, kernel, rec, bias, backwards=False):
    units = rec.shape[0]
    xs = x @ kernel + bias[0]          # all input projections at once
    h = np.zeros(units)
    out = np.zeros((len(x), units))
    steps = range(len(x) - 1, -1, -1) if backwards else range(len(x))
    for t in steps:
        hr = h @ rec + bias[1]
        z = sigmoid(xs[t, :units] + hr[:units])
        r = sigmoid(xs[t, units:2 * units] + hr[units:2 * units])
        hh = np.tanh(xs[t, 2 * units:] + r * hr[2 * units:])
        h = z * h + (1 - z) * hh
        out[t] = h
    return out


def score(text):
    x = emb[to_ids(text)]                                   # (200, 128)
    fwd = run_gru(x, *gru["forward"])
    bwd = run_gru(x, *gru["backward"], backwards=True)
    pooled = np.concatenate([fwd, bwd], axis=1).max(axis=0)  # (128,)
    hidden = np.maximum(pooled @ d1[0] + d1[1], 0)
    return sigmoid(hidden @ d2[0] + d2[1])


@spaces.GPU(duration=5)
def predict(comment, threshold):
    if not comment or not comment.strip():
        return "Please write a comment first.", {}
    probs = score(clean_text(comment))
    scores = {label: float(p) for label, p in zip(LABELS, probs)}
    flagged = [label for label, p in scores.items() if p >= threshold]
    verdict = ("Toxic content detected: " + ", ".join(flagged)) if flagged else "Looks fine."
    return verdict, scores


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Comment", lines=4, value="You are so stupid and nobody likes you."),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Alert threshold"),
    ],
    outputs=[gr.Textbox(label="Result"), gr.Label(label="Scores", num_top_classes=6)],
    title="Toxic Comment Detector",
    description="Bidirectional GRU trained on the Jigsaw toxic comments dataset.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
