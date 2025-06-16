import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

train_labels = train_df.iloc[:, 0].values
train_pixels = train_df.iloc[:, 1:].values / 255.0

test_labels = test_df.iloc[:, 0].values
test_pixels = test_df.iloc[:, 1:].values / 255.0

# 1. Label distribution comparison
def plot_label_dist(label_A, label_B):
    df = pd.DataFrame({
        'Label': list(label_A) + list(label_B),
        'Set': ['Train']*len(label_A) + ['Test']*len(label_B)
    })

    plt.figure(figsize=(7,4))
    sns.histplot(
        data=df,
        x="Label",
        hue="Set",
        bins=np.arange(11)-0.5,
        multiple="dodge",
        stat="density",
        shrink=0.8
    )
    plt.title("Label Distribution: Train vs Test")
    plt.xticks(range(10))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

plot_label_dist(train_labels, test_labels)

# 2. Pixel intensity mean shift
def plot_pixel_means(A_pixels, B_pixels):
    mean_A = A_pixels.mean(axis=0)
    mean_B = B_pixels.mean(axis=0)
    diff = mean_B - mean_A

    plt.figure(figsize=(10,4))
    plt.plot(diff[:100], label='First 100 pixels')
    plt.title("Pixel Mean Shift (Test - Train)")
    plt.xlabel("Pixel Index")
    plt.ylabel("Mean Difference")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_pixel_means(train_pixels, test_pixels)

# 3. PCA visualization (using full datasets)
def plot_pca(A, B):
    pca = PCA(n_components=2)
    X = np.vstack((A, B))
    y = np.array(['Train']*len(A) + ['Test']*len(B))
    X_pca = pca.fit_transform(X)
    
    df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_plot["Domain"] = y

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Domain", alpha=0.3, s=10)
    plt.title("PCA: Train vs Test Domains")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

plot_pca(train_pixels, test_pixels)

# 4. Domain classifier (Train vs Test)
def domain_classifier(A, B):
    X = np.vstack((A, B))
    y = np.array([0]*len(A) + [1]*len(B))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Domain classification accuracy (Train vs Test):", accuracy_score(y_test, preds))

domain_classifier(train_pixels, test_pixels)
