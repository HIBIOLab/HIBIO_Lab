"""
第 11 課：Logistic Regression
資料集：Breast Cancer Wisconsin (Diagnostic)（sklearn 內建）

訓練一個 logistic regression 來預測腫瘤是良性還是惡性，並把每個特徵的
係數換算成 odds ratio，示範這個模型『可以被解釋』的特性 —— 這是它常被
當作臨床風險預測 baseline 的原因。
"""

import sys

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    data = load_breast_cancer()
    X, y, feature_names = data.data, data.target, data.feature_names
    print(f"資料筆數: {len(y)}，特徵數: {X.shape[1]}，惡性比例: {y.mean():.1%}")
    print("(標籤定義：0=malignant 惡性, 1=benign 良性，這是 sklearn 內建的編碼方式)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    print(f"Test accuracy = {acc:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted, labels=[0=malignant,1=benign]):")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))

    print("\n== 係數解讀：標準化後係數 + Odds Ratio ==")
    print("（因為特徵已標準化，係數大小可以互相比較；odds ratio > 1 表示")
    print(" 該特徵越大，模型越傾向預測『良性(1)』；< 1 則傾向預測『惡性(0)』）\n")

    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    order = np.argsort(np.abs(coefs))[::-1]  # 依影响力排序

    print(f"{'特徵名稱':32s} {'標準化係數':>12s} {'Odds Ratio':>12s}")
    for i in order[:10]:
        print(f"{feature_names[i]:32s} {coefs[i]:12.3f} {odds_ratios[i]:12.3f}")

    print("\n=> 影響力最大的前幾個特徵，就是模型認為對『良性/惡性』判斷")
    print("   最關鍵的腫瘤量測值，這種可解釋性是很多黑盒模型做不到的。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 11 課）：
# 1) 寫出 logistic regression 的假設函數 sigma(w.x+b) 與 cross-entropy
#    loss 的公式（紙筆或註解皆可），並在程式中用 model.coef_、
#    model.intercept_ 印出實際學到的 w 和 b。
# 2) 把 StandardScaler 拿掉直接用原始特徵訓練，觀察係數的大小是否還能
#    直接拿來比較特徵重要性？為什麼標準化對「解釋係數」很重要？
# ------------------------------------------------------------------
