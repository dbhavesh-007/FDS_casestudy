import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
SHADES = ["black", "dimgray", "gray", "darkgray", "lightgray", "silver", "white"]
GRAY_MAP = LinearSegmentedColormap.from_list("gray_map", ["#d0d0d0", "#ffffff", "#8a8a8a"])

def save(name):
    plt.savefig(f"figures/{name}.png")
    plt.show()

airline_raw = pd.read_excel("data/EastWestAirlines.xlsx", sheet_name="data")
print(airline_raw.shape)
airline_raw.head()

airline_raw.info()
airline_raw.describe().T

print(airline_raw.isnull().sum())
print("Duplicate rows:", airline_raw.duplicated().sum())
print("Duplicate IDs:", airline_raw["ID#"].duplicated().sum())

airline = airline_raw.drop(columns="ID#")
features = [col for col in airline.columns if col != "Award?"]
print(features)

outlier_cols = ["Balance", "Bonus_miles", "Bonus_trans", "Flight_miles_12mo", "Flight_trans_12"]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, col in zip(axes.ravel(), outlier_cols):
    ax.boxplot(airline[col], patch_artist=True,
               boxprops=dict(facecolor="lightgray", color="black"),
               medianprops=dict(color="black"),
               whiskerprops=dict(color="black"),
               capprops=dict(color="black"),
               flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=3))
    ax.set_title(col)
    ax.set_ylabel("Value")
axes.ravel()[-1].axis("off")
fig.suptitle("Box Plots Before Outlier Removal", fontsize=15)
plt.tight_layout()
save("cs2_boxplots_before")

z_scores = (airline[outlier_cols] - airline[outlier_cols].mean()) / airline[outlier_cols].std()
extreme_rows = (z_scores.abs() > 5).any(axis=1)
print("Rows before removal:", len(airline))
print("Extreme rows found:", extreme_rows.sum())
airline_clean = airline[~extreme_rows].reset_index(drop=True)
print("Rows after removal:", len(airline_clean))
airline_clean[outlier_cols].describe().T

scaler = StandardScaler()
X = scaler.fit_transform(airline_clean[features])
X_df = pd.DataFrame(X, columns=features)
X_df.describe().T[["mean", "std"]].round(3)

numeric_cols = ["Balance", "Qual_miles", "Bonus_miles", "Bonus_trans", "Flight_miles_12mo", "Flight_trans_12", "Days_since_enroll"]
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, col in zip(axes.ravel(), numeric_cols):
    ax.hist(airline_clean[col], bins=30, color="gray", edgecolor="black")
    ax.set_title(f"{col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of Customers")
axes.ravel()[-1].axis("off")
axes.ravel()[-2].axis("off")
plt.tight_layout()
save("cs2_histograms")
print(airline_clean[numeric_cols].skew().round(2))

card_cols = ["cc1_miles", "cc2_miles", "cc3_miles"]
card_titles = ["Frequent Flyer Card", "Rewards Card", "Small Business Card"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for ax, col, title in zip(axes, card_cols, card_titles):
    counts = airline_clean[col].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="gray", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Miles Category (1 to 5)")
    ax.set_ylabel("Number of Customers")
plt.tight_layout()
save("cs2_card_categories")
print(airline_clean[card_cols].apply(lambda s: s.value_counts(normalize=True).round(3)))

award_counts = airline_clean["Award?"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["No Award", "Award"], award_counts.values, color="gray", edgecolor="black")
ax.set_title("Customers by Award Status")
ax.set_xlabel("Award Status")
ax.set_ylabel("Number of Customers")
save("cs2_award_status")
print(award_counts)

corr_airline = airline_clean[features + ["Award?"]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_airline, cmap=GRAY_MAP, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_airline)))
ax.set_yticks(range(len(corr_airline)))
ax.set_xticklabels(corr_airline.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_airline.columns)
for i in range(len(corr_airline)):
    for j in range(len(corr_airline)):
        ax.text(j, i, f"{corr_airline.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Heatmap of Airline Customer Features")
fig.colorbar(im, ax=ax, label="Correlation Coefficient")
save("cs2_heatmap")

pairs = [("Balance", "Bonus_miles"), ("Bonus_miles", "Bonus_trans"), ("Flight_miles_12mo", "Flight_trans_12"), ("Balance", "Days_since_enroll")]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (a, b) in zip(axes.ravel(), pairs):
    ax.scatter(airline_clean[a], airline_clean[b], s=8, color="black", alpha=0.35)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{b} against {a}")
plt.tight_layout()
save("cs2_scatter_pairs")

k_values = range(2, 11)
inertia_list = []
silhouette_list = []
for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia_list.append(model.inertia_)
    silhouette_list.append(silhouette_score(X, model.labels_))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(list(k_values), inertia_list, marker="o", color="black")
axes[0].set_title("Elbow Curve for K-Means")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(k_values), silhouette_list, marker="s", color="black")
axes[1].set_title("Silhouette Score for Different K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
save("cs2_kmeans_elbow")
print(pd.DataFrame({"K": list(k_values), "Inertia": np.round(inertia_list, 1), "Silhouette": np.round(silhouette_list, 3)}).to_string(index=False))

K_FINAL = 3
kmeans = KMeans(n_clusters=K_FINAL, n_init=10, random_state=42).fit(X)
airline_clean["KMeans"] = kmeans.labels_
kmeans_silhouette = silhouette_score(X, kmeans.labels_)
print("Silhouette score:", round(kmeans_silhouette, 3))
print(airline_clean["KMeans"].value_counts().sort_index())

Z = linkage(X, method="ward")
fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(Z, truncate_mode="lastp", p=30, color_threshold=0, above_threshold_color="black", ax=ax)
ax.set_title("Dendrogram Using Ward Linkage (Last 30 Merged Clusters)")
ax.set_xlabel("Cluster Size (Number of Customers)")
ax.set_ylabel("Merge Distance")
save("cs2_dendrogram")

linkage_rows = []
for method in ["ward", "complete", "average", "single"]:
    for k in [3, 4, 5]:
        labels = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
        sizes = np.bincount(labels)
        linkage_rows.append({
            "Linkage": method,
            "Clusters": k,
            "Silhouette": round(silhouette_score(X, labels), 3),
            "Largest Cluster Share": round(sizes.max() / len(labels), 3),
        })
linkage_table = pd.DataFrame(linkage_rows)
print(linkage_table.to_string(index=False))

hier = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward")
airline_clean["Hierarchical"] = hier.fit_predict(X)
hier_silhouette = silhouette_score(X, airline_clean["Hierarchical"])
print("Silhouette score:", round(hier_silhouette, 3))
print(airline_clean["Hierarchical"].value_counts().sort_index())

MIN_PTS = 10
neighbors = NearestNeighbors(n_neighbors=MIN_PTS).fit(X)
distances, _ = neighbors.kneighbors(X)
k_distances = np.sort(distances[:, -1])
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_distances, color="black")
ax.axhline(2.0, color="black", linestyle="--")
ax.set_title("K-Distance Plot for Choosing Epsilon (minPts = 10)")
ax.set_xlabel("Customers Sorted by Distance")
ax.set_ylabel("Distance to 10th Nearest Neighbour")
save("cs2_kdistance")

dbscan_rows = []
for eps in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for min_pts in [10, 20]:
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_share = (labels == -1).mean()
        if n_clusters > 1:
            score = silhouette_score(X[labels != -1], labels[labels != -1])
        else:
            score = np.nan
        dbscan_rows.append({
            "Epsilon": eps,
            "minPts": min_pts,
            "Clusters": n_clusters,
            "Noise Share": round(noise_share, 3),
            "Silhouette": round(score, 3) if not np.isnan(score) else np.nan,
        })
dbscan_table = pd.DataFrame(dbscan_rows)
print(dbscan_table.to_string(index=False))

dbscan = DBSCAN(eps=2.0, min_samples=10).fit(X)
airline_clean["DBSCAN"] = dbscan.labels_
mask = dbscan.labels_ != -1
dbscan_silhouette = silhouette_score(X[mask], dbscan.labels_[mask])
print("Silhouette score (without noise):", round(dbscan_silhouette, 3))
print(airline_clean["DBSCAN"].value_counts().sort_index())

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print("Explained variance:", np.round(pca.explained_variance_ratio_, 3))
print("Total:", round(pca.explained_variance_ratio_.sum(), 3))

def plot_clusters(ax, labels, title):
    for idx, lab in enumerate(sorted(set(labels))):
        pts = X_pca[labels == lab]
        name = "Noise" if lab == -1 else f"Cluster {lab}"
        marker = "x" if lab == -1 else MARKERS[idx % len(MARKERS)]
        shade = "black" if lab == -1 else SHADES[idx % 5]
        if marker == "x":
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c="black", marker="x", label=name)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=shade, marker=marker, edgecolors="black", linewidths=0.4, label=name)
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(fontsize=10)

for labels, title, name in [
    (airline_clean["KMeans"].values, "K-Means Clusters (K = 3)", "cs2_plot_kmeans"),
    (airline_clean["Hierarchical"].values, "Hierarchical Clusters (Ward, K = 3)", "cs2_plot_hierarchical"),
    (airline_clean["DBSCAN"].values, "DBSCAN Clusters (eps = 2.0, minPts = 10)", "cs2_plot_dbscan"),
]:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_clusters(ax, labels, title)
    save(name)

profile_cols = ["Balance", "Qual_miles", "cc1_miles", "Bonus_miles", "Bonus_trans", "Flight_miles_12mo", "Flight_trans_12", "Days_since_enroll", "Award?"]
for algo in ["KMeans", "Hierarchical", "DBSCAN"]:
    print("\n" + algo)
    prof = airline_clean.groupby(algo)[profile_cols].mean().round(2)
    prof.insert(0, "Customers", airline_clean[algo].value_counts().sort_index())
    print(prof.to_string())

kmeans_profile = airline_clean.groupby("KMeans")[["Balance", "Bonus_miles", "Flight_miles_12mo", "Days_since_enroll"]].mean()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), kmeans_profile.columns):
    ax.bar(kmeans_profile.index.astype(str), kmeans_profile[col], color="gray", edgecolor="black")
    ax.set_title(f"Average {col}")
    ax.set_xlabel("K-Means Cluster")
    ax.set_ylabel("Average Value")
plt.tight_layout()
save("cs2_kmeans_profile")

summary = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical (Ward)", "DBSCAN"],
    "Clusters": [K_FINAL, K_FINAL, 3],
    "Silhouette": [round(kmeans_silhouette, 3), round(hier_silhouette, 3), round(dbscan_silhouette, 3)],
    "Noise Points": [0, 0, int((dbscan.labels_ == -1).sum())],
})
print(summary.to_string(index=False))

red = pd.read_csv("data/winequality-red.csv", sep=";")
white = pd.read_csv("data/winequality-white.csv", sep=";")
red["type"] = "red"
white["type"] = "white"
wine = pd.concat([red, white], ignore_index=True)
print(red.shape, white.shape, wine.shape)
wine.head()

wine.info()
print(wine.isnull().sum().sum())
print("Duplicate rows:", wine.duplicated().sum())
wine.describe().T

attributes = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
              "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

selected = ["alcohol", "volatile acidity", "residual sugar"]
stats_rows = []
for col in selected:
    stats_rows.append({
        "Attribute": col,
        "Mean": wine[col].mean(),
        "Median": wine[col].median(),
        "Mode": wine[col].mode()[0],
        "Std Dev": wine[col].std(),
        "Skewness": wine[col].skew(),
    })
stats_table = pd.DataFrame(stats_rows).set_index("Attribute").round(3)
print(stats_table.to_string())

cv_table = pd.DataFrame({
    "Mean": wine[selected].mean(),
    "Std Dev": wine[selected].std(),
})
cv_table["CV (%)"] = cv_table["Std Dev"] / cv_table["Mean"] * 100
print(cv_table.round(3).to_string())
print("Highest relative variability:", cv_table["CV (%)"].idxmax())

r_value, p_value = stats.pearsonr(wine["alcohol"], wine["quality"])
print("Pearson r:", round(r_value, 4))
print("p-value:", p_value)

quality_counts = wine["quality"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(quality_counts.index, quality_counts.values, color="gray", edgecolor="black", width=0.8)
for q, n in quality_counts.items():
    ax.text(q, n + 30, str(n), ha="center", fontsize=9)
ax.set_title("Histogram of Wine Quality Scores")
ax.set_xlabel("Quality Score")
ax.set_ylabel("Number of Wines")
ax.set_xticks(quality_counts.index)
save("cs5_quality_histogram")
print(quality_counts)
print((quality_counts / len(wine) * 100).round(2))
print("Skewness:", round(wine["quality"].skew(), 3))

quality_levels = sorted(wine["quality"].unique())
groups = [wine.loc[wine["quality"] == q, "alcohol"] for q in quality_levels]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.boxplot(groups, labels=quality_levels, patch_artist=True,
           boxprops=dict(facecolor="lightgray", color="black"),
           medianprops=dict(color="black", linewidth=1.8),
           whiskerprops=dict(color="black"),
           capprops=dict(color="black"),
           flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=4))
ax.set_title("Alcohol Content by Wine Quality Score")
ax.set_xlabel("Quality Score")
ax.set_ylabel("Alcohol (% by volume)")
save("cs5_alcohol_boxplot")
print(wine.groupby("quality")["alcohol"].agg(["count", "median", "mean"]).round(2))
for q, g in zip(quality_levels, groups):
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = q3 - q1
    n_out = ((g < q1 - 1.5 * iqr) | (g > q3 + 1.5 * iqr)).sum()
    print("Quality", q, "outliers:", n_out)

corr_wine = wine[attributes + ["quality"]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_wine, cmap=GRAY_MAP, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_wine)))
ax.set_yticks(range(len(corr_wine)))
ax.set_xticklabels(corr_wine.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_wine.columns)
for i in range(len(corr_wine)):
    for j in range(len(corr_wine)):
        ax.text(j, i, f"{corr_wine.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Heatmap of Wine Attributes and Quality")
fig.colorbar(im, ax=ax, label="Correlation Coefficient")
save("cs5_heatmap")

quality_corr = corr_wine["quality"].drop("quality")
ranked = quality_corr.reindex(quality_corr.abs().sort_values(ascending=False).index)
print(ranked.round(4).to_string())

X_wine = wine[attributes]
y_wine = wine["quality"]
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
lin_model = LinearRegression().fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))
print("Training R2:", round(lin_model.score(X_train, y_train), 4))
print("Within 0.5 of actual:", round((np.abs(y_test - y_pred) <= 0.5).mean() * 100, 1), "%")
print("Within 1.0 of actual:", round((np.abs(y_test - y_pred) <= 1.0).mean() * 100, 1), "%")

std_coef = pd.Series(lin_model.coef_ * X_train.std().values / y_train.std(), index=attributes)
coef_table = pd.DataFrame({
    "Raw Coefficient": lin_model.coef_,
    "Standardized Coefficient": std_coef,
})
coef_table = coef_table.reindex(std_coef.abs().sort_values(ascending=False).index)
print("Intercept:", round(lin_model.intercept_, 4))
print(coef_table.round(4).to_string())

vif_rows = []
for col in attributes:
    others = [a for a in attributes if a != col]
    aux = LinearRegression().fit(X_wine[others], X_wine[col])
    r2_aux = aux.score(X_wine[others], X_wine[col])
    vif_rows.append({"Attribute": col, "VIF": 1 / (1 - r2_aux)})
vif_table = pd.DataFrame(vif_rows).set_index("Attribute").sort_values("VIF", ascending=False)
print(vif_table.round(2).to_string())

print(corr_wine.loc[["density", "alcohol", "residual sugar", "total sulfur dioxide", "free sulfur dioxide"],
                    ["density", "alcohol", "residual sugar", "total sulfur dioxide", "free sulfur dioxide"]].round(3))

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.scatter(y_test, y_pred, s=14, color="black", alpha=0.35)
ax.plot([3, 9], [3, 9], color="black", linestyle="--", label="Perfect prediction")
ax.set_title("Predicted Quality Against Actual Quality (Test Set)")
ax.set_xlabel("Actual Quality Score")
ax.set_ylabel("Predicted Quality Score")
ax.legend()
save("cs5_pred_vs_actual")
print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).groupby("Actual")["Predicted"].agg(["count", "mean"]).round(2))

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuals, bins=30, color="gray", edgecolor="black")
ax.set_title("Distribution of Prediction Errors (Test Set)")
ax.set_xlabel("Actual Minus Predicted Quality")
ax.set_ylabel("Number of Wines")
save("cs5_residuals")
print("Mean error:", round(residuals.mean(), 4))
print("Largest under-prediction:", round(residuals.max(), 2))
print("Largest over-prediction:", round(residuals.min(), 2))

baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
print("Baseline RMSE (always predict the mean):", round(baseline_rmse, 4))
print("Model RMSE:", round(rmse, 4))
print("Improvement:", round((1 - rmse / baseline_rmse) * 100, 1), "%")
