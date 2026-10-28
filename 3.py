import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

V = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=V)

def multi_hot(seqs):
    out = np.zeros((len(seqs), V), dtype="float32")
    for i, s in enumerate(seqs):
        out[i, s] = 1
    return out

x_train, x_test = multi_hot(x_train), multi_hot(x_test)
print(x_train.shape, x_test.shape)

inp = Input(shape=(V,))
h = Dense(128, activation="relu")(inp)
h = Dense(64, activation="relu")(h)
latent = Dense(32, activation="relu")(h)
h = Dense(64, activation="relu")(latent)
h = Dense(128, activation="relu")(h)
out = Dense(V, activation="sigmoid")(h)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
ae_hist = autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.2, verbose=2)

plt.plot(ae_hist.history["loss"], label="train")
plt.plot(ae_hist.history["val_loss"], label="validation")
plt.title("Autoencoder reconstruction loss")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.legend()
plt.show()

sub = x_test[:5000]
recon = autoencoder.predict(sub, verbose=0)
print("Test BCE loss:", round(autoencoder.evaluate(x_test, x_test, verbose=0), 4))
print("Reconstruction MSE (autoencoder):", round(float(np.mean((sub - recon) ** 2)), 5))
print("Reconstruction MSE (baseline, mean word usage):", round(float(np.mean((sub - x_train.mean(axis=0)) ** 2)), 5))

encoder = Model(inp, latent)
f_train = encoder.predict(x_train, verbose=0)
f_test = encoder.predict(x_test, verbose=0)

classifier = Sequential([
    Input(shape=(32,)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
clf_hist = classifier.fit(f_train, y_train, epochs=15, batch_size=128, validation_split=0.2, verbose=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(clf_hist.history["loss"], label="train")
ax[0].plot(clf_hist.history["val_loss"], label="validation")
ax[0].set_title("Classifier loss")
ax[1].plot(clf_hist.history["accuracy"], label="train")
ax[1].plot(clf_hist.history["val_accuracy"], label="validation")
ax[1].set_title("Classifier accuracy")
for a in ax:
    a.set_xlabel("Epoch")
    a.legend()
plt.tight_layout()
plt.show()

y_prob = classifier.predict(f_test, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].imshow(cm, cmap="Blues")
ax[0].set_xticks([0, 1], ["Negative", "Positive"])
ax[0].set_yticks([0, 1], ["Negative", "Positive"])
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, cm[i, j], ha="center", va="center")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("True")
ax[0].set_title("Confusion matrix")
ax[1].plot(fpr, tpr, label="AE features")
ax[1].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
ax[1].set_xlabel("False positive rate")
ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC curve")
ax[1].legend()
pts = PCA(n_components=2).fit_transform(f_test[:3000])
ax[2].scatter(pts[:, 0], pts[:, 1], c=y_test[:3000], cmap="coolwarm", s=4)
ax[2].set_title("Latent features, 2-d PCA (red = positive)")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.mean = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 256)
        self.fc5 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mean(h), self.logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


def vae_loss(recon, x, mean, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return bce + kl, bce, kl


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)

def run_epoch(data_loader, train):
    model.train() if train else model.eval()
    total_recon, total_kl, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for x, _ in data_loader:
            x = x.view(-1, 784).to(device)
            recon, mean, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mean, logvar)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n += x.size(0)
    return total_recon / n, total_kl / n


epochs = 15
hist = {"tr_recon": [], "tr_kl": [], "te_recon": [], "te_kl": []}
for epoch in range(1, epochs + 1):
    tr_r, tr_k = run_epoch(loader, True)
    te_r, te_k = run_epoch(test_loader, False)
    hist["tr_recon"].append(tr_r)
    hist["tr_kl"].append(tr_k)
    hist["te_recon"].append(te_r)
    hist["te_kl"].append(te_k)
    print(f"Epoch {epoch:2d}/{epochs} | train loss: {tr_r + tr_k:.2f} (recon {tr_r:.2f}, kl {tr_k:.2f}) | "
          f"test loss: {te_r + te_k:.2f}")


ep = range(1, epochs + 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ep, np.array(hist["tr_recon"]) + np.array(hist["tr_kl"]), label="train")
ax[0].plot(ep, np.array(hist["te_recon"]) + np.array(hist["te_kl"]), label="test")
ax[0].set_title("Total loss per sample")
ax[1].plot(ep, hist["tr_recon"], label="train")
ax[1].plot(ep, hist["te_recon"], label="test")
ax[1].set_title("Reconstruction loss (BCE)")
ax[2].plot(ep, hist["tr_kl"], label="train")
ax[2].plot(ep, hist["te_kl"], label="test")
ax[2].set_title("KL divergence")
for a in ax:
    a.set_xlabel("Epoch")
    a.legend()
plt.tight_layout()
plt.show()

model.eval()
x, _ = next(iter(test_loader))
with torch.no_grad():
    recon, _, _ = model(x.view(-1, 784).to(device))
print("Test reconstruction MSE (one batch):", round(F.mse_loss(recon.cpu(), x.view(-1, 784)).item(), 5))

fig, ax = plt.subplots(2, 10, figsize=(15, 3.2))
for i in range(10):
    ax[0, i].imshow(x[i].squeeze(), cmap="gray")
    ax[1, i].imshow(recon[i].cpu().view(28, 28), cmap="gray")
    ax[0, i].axis("off")
    ax[1, i].axis("off")
ax[0, 0].set_title("Original", loc="left")
ax[1, 0].set_title("Reconstructed", loc="left")
plt.show()

all_mean, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        mean, _ = model.encode(x.view(-1, 784).to(device))
        all_mean.append(mean.cpu().numpy())
        all_labels.append(y.numpy())
all_mean = np.concatenate(all_mean, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_mean[:, 0], all_mean[:, 1], c=all_labels, cmap="tab10", s=3)
plt.colorbar(scatter, label="Digit label")
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.title("Test digits in the 2D latent space")
plt.show()

n = 15
size = 28
figure = np.zeros((size * n, size * n))
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]

with torch.no_grad():
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            digit = model.decode(z).cpu().numpy()[0].reshape(size, size)
            figure[i * size:(i + 1) * size, j * size:(j + 1) * size] = digit

plt.figure(figsize=(8, 8))
plt.imshow(figure, cmap="Greys_r")
plt.axis("off")
plt.title("Digits generated by sampling the VAE latent space")
plt.show()

with torch.no_grad():
    samples = model.decode(torch.randn(16, 2).to(device)).cpu().view(-1, 28, 28)
fig, ax = plt.subplots(2, 8, figsize=(12, 3.2))
for a, s in zip(ax.ravel(), samples):
    a.imshow(s, cmap="gray")
    a.axis("off")
fig.suptitle("16 random samples from N(0, 1)")
plt.show()
