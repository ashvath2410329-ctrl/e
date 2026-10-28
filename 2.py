import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Embedding, LSTM, Add, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

class_names = ["t-shirt", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]
templates = ["a photo of", "this picture shows", "an image containing",
             "a small picture of", "here is"]

def sentence(template, label):
    name = class_names[label]
    article = "an" if name[0] in "aeiou" else "a"
    return f"{template} {article} {name}"

def caption(label):
    return f"startseq {sentence(random.choice(templates), label)} endseq"

n_train, n_test = 4000, 500
train_idx = np.random.choice(len(x_train), n_train, replace=False)
test_idx = np.random.choice(len(x_test), n_test, replace=False)
train_images = x_train[train_idx]
test_images = x_test[test_idx]
train_captions = [caption(l) for l in y_train[train_idx]]
test_captions = [caption(l) for l in y_test[test_idx]]
print("Example captions:", train_captions[:3])

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(train_captions)
vocab_size = len(tokenizer.word_index) + 1
train_seqs = tokenizer.texts_to_sequences(train_captions)
test_seqs = tokenizer.texts_to_sequences(test_captions)
max_len = max(len(s) for s in train_seqs)
print(f"Vocab size: {vocab_size}, Max caption length: {max_len}")

def build_pairs(images, seqs):
    imgs, words, nxt = [], [], []
    for img, seq in zip(images, seqs):
        for t in range(1, len(seq)):
            imgs.append(img)
            words.append(seq[:t])
            nxt.append(seq[t])
    return np.array(imgs), pad_sequences(words, maxlen=max_len), np.array(nxt)

imgs, words, nxt = build_pairs(train_images, train_seqs)
imgs_te, words_te, nxt_te = build_pairs(test_images, test_seqs)
print("Training pairs:", imgs.shape[0], "| Test pairs:", imgs_te.shape[0])

image_in = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(image_in)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
eye = Dense(128, activation="relu")(x)
eye = Dropout(0.3)(eye)

words_in = Input(shape=(max_len,))
y = Embedding(vocab_size, 128)(words_in)
y = Dropout(0.3)(y)
reader = LSTM(128)(y)

z = Add()([eye, reader])
z = Dense(128, activation="relu")(z)
out = Dense(vocab_size, activation="softmax")(z)

caption_model = Model([image_in, words_in], out)
caption_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
caption_model.summary()

history = caption_model.fit([imgs, words], nxt, validation_data=([imgs_te, words_te], nxt_te),
                            epochs=20, batch_size=64, verbose=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="test")
ax[0].set_title("Loss")
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="test")
ax[1].set_title("Next-word accuracy")
for a in ax:
    a.set_xlabel("Epoch")
    a.legend()
plt.tight_layout()
plt.show()

def generate_batch(images):
    texts = ["startseq"] * len(images)
    done = [False] * len(images)
    for _ in range(max_len):
        ids = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)
        idx = caption_model.predict([images, ids], verbose=0).argmax(axis=1)
        for i, k in enumerate(idx):
            if not done[i]:
                w = tokenizer.index_word.get(k, "endseq")
                if w == "endseq":
                    done[i] = True
                else:
                    texts[i] += " " + w
        if all(done):
            break
    return [t.replace("startseq", "").strip() for t in texts]

names = [n.replace("-", " ") for n in class_names]
order = sorted(range(10), key=lambda i: -len(names[i]))

def caption_class(c):
    for i in order:
        if names[i] in c:
            return i
    return -1

pred_caps = generate_batch(test_images)
true = y_test[test_idx]
pred = np.array([caption_class(c) for c in pred_caps])

print("Caption class accuracy:", round(accuracy_score(true, pred), 4))
print(classification_report(true, pred, labels=range(10), target_names=class_names, zero_division=0))

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [[sentence(t, l).replace("-", " ").split() for t in templates] for l in true]
    hyps = [c.split() for c in pred_caps]
    sm = SmoothingFunction().method1
    print("BLEU-1:", round(corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=sm), 4))
    print("BLEU-4:", round(corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sm), 4))
except ImportError:
    print("nltk not installed, BLEU skipped")

cm = confusion_matrix(true, pred, labels=range(10))
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(10), class_names, rotation=45)
plt.yticks(range(10), class_names)
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Class in generated caption")
plt.ylabel("True class")
plt.title("Caption confusion matrix")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, i in zip(axes.ravel(), np.random.choice(len(test_images), 8, replace=False)):
    ax.imshow(test_images[i].squeeze(), cmap="gray")
    ax.set_title(f"True: {class_names[true[i]]}\n{pred_caps[i]}", fontsize=9,
                 color="green" if pred[i] == true[i] else "red")
    ax.axis("off")
plt.tight_layout()
plt.show()
