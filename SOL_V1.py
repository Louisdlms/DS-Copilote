# ===============================================================
# Pipeline corrigé : exclusion des tXX (timestamps) des actions
# et extraction de features temporelles par fenêtre
# Auteur : DEG (corrigé)
# Date : 08/10/2025
# ===============================================================

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------- Chemins (modifie si besoin) ----------
TRAIN_PATH = "Data/train.csv"         # chemin par défaut
TEST_PATH = "Data/test.csv"
SAMPLE_PATH = "Data/sample_submission.csv"
OUTPUT_PATH = "Data/submission.csv"
RANDOM_STATE = 42

# ===============================================================
# 1) Lecture robuste des fichiers (séparateur = ',')
# ===============================================================
def read_variable_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # split simple car chaque champ est séparé par une virgule
            parts = line.strip().split(",")
            rows.append(parts)
    max_len = max(len(r) for r in rows)
    df = pd.DataFrame(rows, columns=[f"col_{i}" for i in range(max_len)])
    return df

train_raw = read_variable_csv(TRAIN_PATH)
test_raw = read_variable_csv(TEST_PATH)
print(f"✅ Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

# ===============================================================
# 2) Parsing : util (label), navigateur, liste des tokens/actions
# ===============================================================
def parse_traces(df, is_train=True):
    df = df.copy()
    if is_train:
        df = df.rename(columns={"col_0": "util", "col_1": "navigateur"})
        df["actions"] = df.loc[:, "col_2":].apply(lambda x: [a for a in x.dropna().values if a != ""], axis=1)
        return df[["util", "navigateur", "actions"]]
    else:
        df = df.rename(columns={"col_0": "navigateur"})
        df["actions"] = df.loc[:, "col_1":].apply(lambda x: [a for a in x.dropna().values if a != ""], axis=1)
        return df[["navigateur", "actions"]]

train_df = parse_traces(train_raw, is_train=True)
test_df = parse_traces(test_raw, is_train=False)


print("Exemples (train) :")
print(train_df.head(3))

# ===============================================================
# 3) Traitement des tokens : séparer timestamps tXX et "vraies" actions
# ===============================================================
t_re = re.compile(r"^t(\d+)$")  # capture des tXX

def analyze_tokens(tokens):
    """
    tokens : liste de tokens tels qu'extraient depuis la ligne
    Retourne dict with:
       - actions_no_t : liste d'actions (sans tXX)
       - t_values     : liste d'entiers (valeurs des tXX) dans l'ordre d'apparition
       - windows      : liste counts d'actions par fenêtre temporelle
    """
    t_values = []
    windows = [0]  # commence par une fenêtre avant le premier tXX
    actions_no_t = []

    for tok in tokens:
        if tok is None:
            continue
        tok = tok.strip()
        if tok == "":
            continue
        m = t_re.match(tok)
        if m:
            # token temporel -> nouvelle fenêtre
            t_values.append(int(m.group(1)))
            # commence nouvelle fenêtre
            windows.append(0)
        else:
            # token action réel
            actions_no_t.append(tok)
            windows[-1] += 1

    # enlever la dernière fenêtre vide si le dernier token était un tXX
    if len(windows) > 0 and windows[-1] == 0:
        windows.pop()

    return {
        "actions_no_t": actions_no_t,
        "t_values": t_values,
        "windows": windows
    }

# ===============================================================
# 4) Extraction de features par session (corrigée)
# ===============================================================
from scipy.stats import entropy
from collections import Counter

def extract_features(df, is_train=True):
    feats = []
    it = tqdm(df.itertuples(index=False), total=len(df), desc="extract_features")

    for row in it:
        nav = getattr(row, "navigateur", "")
        tokens = getattr(row, "actions", [])
        tokens = [t for t in tokens if isinstance(t, str) and t.strip() != ""]

        # Analyse (sépare tXX)
        analysis = analyze_tokens(tokens)
        actions = analysis["actions_no_t"]
        t_values = analysis["t_values"]
        windows = analysis["windows"]

        n_actions = len(actions)
        n_unique_actions = len(set(actions))
        n_modifs = sum(1 for a in actions if a.endswith("1"))
        n_modules = len(re.findall(r"\([^)]+\)", " ".join(actions)))
        n_configs = len(re.findall(r"<[^>]+>", " ".join(actions)))
        n_chaines = len(re.findall(r"\$[^$]+\$", " ".join(actions)))

        max_time = max(t_values) if t_values else 0
        n_time_tokens = len(t_values)
        n_windows = len(windows) if windows else (1 if n_actions > 0 else 0)

        mean_actions_per_window = float(np.mean(windows)) if windows else float(n_actions)
        max_actions_per_window = int(np.max(windows)) if windows else int(n_actions)
        std_actions_per_window = float(np.std(windows)) if windows else 0.0
        action_rate = float(n_actions) / max_time if max_time > 0 else float(n_actions)

        # --- nouvelles features ---

        ratio_unique_actions = n_unique_actions / n_actions if n_actions > 0 else 0
        ratio_modifs = n_modifs / n_actions if n_actions > 0 else 0
        ratio_modules = n_modules / n_actions if n_actions > 0 else 0
        ratio_configs = n_configs / n_actions if n_actions > 0 else 0

        # Diversité
        if n_actions > 0:
            counts = np.array(list(Counter(actions).values()))
            entropy_actions = float(entropy(counts))
            top_action_freq = float(np.max(counts) / n_actions)
        else:
            entropy_actions = 0.0
            top_action_freq = 0.0

        # Temporalité
        window_density = n_actions / n_windows if n_windows > 0 else n_actions
        avg_window_change = std_actions_per_window / mean_actions_per_window if mean_actions_per_window > 0 else 0

        # Ordre des actions
        starts_with_config = int(len(actions) > 0 and "<" in actions[0])
        ends_with_modif = int(len(actions) > 0 and actions[-1].endswith("1"))

        feats.append({
            "n_actions": n_actions,
            "n_unique_actions": n_unique_actions,
            "n_modifs": n_modifs,
            "n_modules": n_modules,
            "n_configs": n_configs,
            "n_chaines": n_chaines,
            "max_time": max_time,
            "n_time_tokens": n_time_tokens,
            "n_windows": n_windows,
            "mean_actions_per_window": mean_actions_per_window,
            "max_actions_per_window": max_actions_per_window,
            "std_actions_per_window": std_actions_per_window,
            "action_rate": action_rate,
            # nouvelles :
            "ratio_unique_actions": ratio_unique_actions,
            "ratio_modifs": ratio_modifs,
            "ratio_modules": ratio_modules,
            "ratio_configs": ratio_configs,
            "entropy_actions": entropy_actions,
            "top_action_freq": top_action_freq,
            "window_density": window_density,
            "avg_window_change": avg_window_change,
            "starts_with_config": starts_with_config,
            "ends_with_modif": ends_with_modif,
            "navigateur": nav
        })

    feats_df = pd.DataFrame(feats)
    if is_train:
        feats_df["util"] = df["util"].values
    return feats_df

# Extraire features
train_feats = extract_features(train_df, is_train=True)
test_feats = extract_features(test_df, is_train=False)



# ===============================================================
# 5) Encodage & préparation des matrices
# ===============================================================
le_user = LabelEncoder()
y = le_user.fit_transform(train_feats["util"])
X = train_feats.drop(columns=["util"])

# one-hot navigateur (s'il y a des valeurs manquantes on crée 'navigateur_')
X = pd.get_dummies(X, columns=["navigateur"], dummy_na=True)
test_feats = pd.get_dummies(test_feats, columns=["navigateur"], dummy_na=True)
# align columns (ajoute colonnes manquantes à test ou train)
test_feats = test_feats.reindex(columns=X.columns, fill_value=0)

from sklearn.preprocessing import StandardScaler

# ===============================================================
# Normalisation des variables numériques
# ===============================================================
# Sélection des colonnes numériques à normaliser
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Initialisation du scaler
scaler = StandardScaler()

# Ajustement sur le train, puis transformation
X[num_cols] = scaler.fit_transform(X[num_cols])
test_feats[num_cols] = scaler.transform(test_feats[num_cols])

print(f"✅ Normalisation appliquée sur {len(num_cols)} variables numériques.")


# ===============================================================
# 6) Entraînement (RandomForest) & évaluation
# ===============================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=RANDOM_STATE, stratify=y)
model = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy de validation : {acc:.4f}")

# ===============================================================
# 7) Prédiction sur le test + génération du fichier de soumission
# ===============================================================
X_test = test_feats.copy()
test_pred = model.predict(X_test)
test_pred_labels = le_user.inverse_transform(test_pred)



# Lire le sample_submission (si présent) et remplacer la colonne label
sample_sub = pd.read_csv(SAMPLE_PATH)
# suppose format sample_sub : [id, target]
sample_sub.iloc[:, 1] = test_pred_labels
sample_sub.to_csv(OUTPUT_PATH, index=False)
print(f"Fichier de soumission écrit -> {OUTPUT_PATH}")
