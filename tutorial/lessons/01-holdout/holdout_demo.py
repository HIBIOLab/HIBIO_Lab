"""
第 1 課：Hold-out 驗證法
資料集：Breast Cancer Wisconsin (Diagnostic) —— sklearn 內建，569 位病人、30 項腫瘤特徵。
        ⚠️ sklearn 的實際標籤編碼是「惡性 malignant = 0（212 筆）、良性 benign = 1（357 筆）」，
           所以 data.target 的平均值其實是「良性」比例，不是惡性比例（原版註解寫反了，這裡已修正）。

這支程式想讓你「親眼看到」單次 hold-out 切分的結果有多不穩定：
我們用 10 個不同的隨機種子各切一次 train/test，同一個模型、同一份資料，
準確率卻會忽高忽低 —— 這就是只做一次 hold-out 驗證的風險。

後半段還多做一個「不平衡資料」的對照：原始資料良性/惡性大約 63/37，其實不算太不平衡，
所以 stratify 的效果不明顯。我們把惡性 downsample 成稀有陽性（約 10%），
就能清楚看到 stratify 的作用：保證每次切分的類別比例都一致。
"""
import sys
if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEEDS = [1, 36, 87, 96, 111, 666, 1111, 1234, 2026, 6666]


def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names


def run_single_holdout(X, y, random_state, stratify=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred)


def make_imbalanced(X, y, minority_class=0, minority_ratio=0.10, random_state=42):
    """把 minority_class 這一類 downsample，讓它只佔 minority_ratio，
    模擬真實臨床篩檢中「陽性（惡性）樣本很稀少」的情境。
    （惡性在 sklearn 是 class 0，所以預設 minority_class=0。）"""
    rng = np.random.default_rng(random_state)
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]
    n_major = len(majority_idx)
    # 由 minority_ratio = n_keep / (n_keep + n_major) 反推要留幾個少數類樣本
    n_keep = int(round(minority_ratio * n_major / (1 - minority_ratio)))
    minority_keep = rng.choice(minority_idx, size=n_keep, replace=False)
    keep = np.concatenate([majority_idx, minority_keep])
    rng.shuffle(keep)
    return X[keep], y[keep]


def run_holdout_imbalanced(X, y, random_state, stratify, minority_class=0):
    """跟 run_single_holdout 一樣做一次 hold-out，但額外回傳
    (accuracy, 少數類 recall, test 集裡少數類的比例)，方便觀察不平衡下的行為。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=5000)
    model.fit(scaler.transform(X_train), y_train)
    pred = model.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, pred)
    # 惡性(minority_class)當陽性，看模型有沒有把惡性抓出來
    rec = recall_score(y_test, pred, pos_label=minority_class, zero_division=0)
    minority_prop = (y_test == minority_class).mean()
    return acc, rec, minority_prop


def main():
    X, y, _ = load_data()
    mal_ratio = (y == 0).mean()   # 0 = malignant（惡性）
    print(f"資料筆數: {len(y)}，惡性(malignant, class 0)比例: {mal_ratio:.1%}，"
          f"良性(benign, class 1)比例: {1 - mal_ratio:.1%}\n")

    print("== 用 10 個不同的隨機種子各做一次 hold-out（80/20 切分）==")
    accs = []
    for seed in RANDOM_SEEDS:
        acc = run_single_holdout(X, y, random_state=seed, stratify=True)
        accs.append(acc)
        print(f"  random_state={seed:4d}  test accuracy = {acc:.4f}")
    accs = np.array(accs)
    print(f"\n10 次 hold-out 的準確率：mean={accs.mean():.4f}, "
          f"std={accs.std():.4f}, min={accs.min():.4f}, max={accs.max():.4f}")
    print("=> 光是換一個隨機種子，準確率就可能差好幾個百分點。")
    print("   如果論文只報告『其中一次』的結果，讀者根本不知道這個數字有多穩定。")

    print("\n== 對照：不做 stratify（不保持良性/惡性比例）的 hold-out ==")
    accs_no_strat = np.array(
        [run_single_holdout(X, y, random_state=s, stratify=False) for s in RANDOM_SEEDS]
    )
    print(f"  不 stratify: mean={accs_no_strat.mean():.4f}, std={accs_no_strat.std():.4f}")
    print(f"  有 stratify: mean={accs.mean():.4f}, std={accs.std():.4f}")
    print("=> 這份資料還算平衡，所以兩者差異不大，stratify 的好處看不太出來。")

    # ==================================================================
    # 進階：把資料弄成「不平衡」，stratify 的效果才會明顯
    # ==================================================================
    print("\n== 進階：不平衡資料上的 stratify（把惡性 downsample 到 ~10%）==")
    X_imb, y_imb = make_imbalanced(X, y, minority_class=0, minority_ratio=0.10, random_state=42)
    print(f"不平衡後 筆數={len(y_imb)}，惡性(陽性)比例={(y_imb == 0).mean():.1%}"
          f"（原本 {mal_ratio:.1%}）")

    for stratify in (False, True):
        accs_i, recs_i, props_i = [], [], []
        for seed in RANDOM_SEEDS:
            acc, rec, prop = run_holdout_imbalanced(X_imb, y_imb, seed, stratify)
            accs_i.append(acc)
            recs_i.append(rec)
            props_i.append(prop)
        accs_i, recs_i, props_i = np.array(accs_i), np.array(recs_i), np.array(props_i)
        tag = "有 stratify" if stratify else "不 stratify"
        print(f"\n  --- {tag} ---")
        print(f"    每次 test 集的惡性比例：min={props_i.min():.1%}, max={props_i.max():.1%}")
        print(f"    accuracy    : mean={accs_i.mean():.4f}, std={accs_i.std():.4f}")
        print(f"    惡性 recall : mean={recs_i.mean():.4f}, std={recs_i.std():.4f}")

    print("\n=> stratify 最直接、保證成立的效果：每個 test 集的類別比例都鎖在母體比例")
    print("   （上面『有 stratify』那組 min=max）；不 stratify 時比例會亂飄，有的切分惡性")
    print("   樣本特別少，測出來的 recall 就忽高忽低。")
    print("   另外注意：accuracy 一直很高（全猜良性就有 ~90%），在不平衡資料上會騙人，")
    print("   真正該看的是少數類別（惡性）的 recall。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 1 課）：
# 1) 把 test_size 從 0.2 改成 0.5 再跑一次，觀察 std 是變大還是變小？為什麼？
# 2) 修改 run_single_holdout，改用 sklearn.tree.DecisionTreeClassifier 取代
#    LogisticRegression，比較兩個模型對「切分方式」的敏感程度是否不同。
# 3) 在不平衡那段，把 minority_ratio 從 0.10 改成 0.05，觀察「不 stratify」時
#    test 集惡性比例的 min/max 落差是不是更大、惡性 recall 是不是更不穩。
#    （提示：少數類樣本越少，單次 hold-out 越不可靠 —— 這也是為什麼不平衡
#     資料通常要用 stratified k-fold cross-validation，而不是只切一次。）
# ------------------------------------------------------------------
