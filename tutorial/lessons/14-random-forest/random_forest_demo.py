"""
第 14 課：Random Forest
資料集：Breast Cancer Wisconsin (Diagnostic)（sklearn 內建）

訓練 Random Forest，印出 feature importance（模擬 radiomics 特徵篩選的
使用情境），並和單棵 Decision Tree、第 11 課的 Logistic Regression 比較。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    data = load_breast_cancer()
    X, y, feature_names = data.data, data.target, data.feature_names

    print("== 三種模型的 5-fold CV accuracy 比較（穩定性 vs 單棵樹）==")
    models = {
        "Decision Tree (單棵樹)": DecisionTreeClassifier(random_state=42),
        "Random Forest (200 棵樹)": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    }
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"  {name:28s} mean={scores.mean():.4f}  std={scores.std():.4f}")
    print("=> 注意 Random Forest 的平均準確率通常比單棵 Decision Tree 高，")
    print("   這是因為單棵樹很容易對訓練資料的雜訊過擬合，而 bagging（多棵樹")
    print("   分別在不同的 bootstrap 子集上訓練、再投票平均）能有效降低")
    print("   這種 variance，讓模型在『沒看過的資料』上更穩定。\n")

    print("== Random Forest 的 Feature Importance（模擬 radiomics 特徵篩選）==")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    for i in order[:10]:
        print(f"  {feature_names[i]:28s} importance={importances[i]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    top_n = 10
    ax.barh([feature_names[i] for i in order[:top_n]][::-1], importances[order[:top_n]][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest: Top 10 重要特徵")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "14_random_forest_importance.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")
    print("=> 在放射組學 (radiomics) 分析中，可以先用這種 feature importance")
    print("   排序，只保留前面幾個最重要的特徵，再拿去做後續分析或建模，")
    print("   兼顧可解釋性與降維。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 14 課）：
# 1) 解釋 bagging（自助抽樣聚合）如何幫助降低模型 variance？可以對照
#    上面 Decision Tree 和 Random Forest 的平均準確率差異來說明；也可以
#    試著把 DecisionTreeClassifier(max_depth=3) 加上深度限制，觀察它
#    的表現是否會比較接近 Random Forest。
# 2) 只取 feature importance 排名前 5 的特徵重新訓練 Random Forest，
#    比較準確率跟用全部 30 個特徵時差多少？這說明了什麼？
# ------------------------------------------------------------------
