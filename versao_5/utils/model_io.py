"""utils/model_io.py — Model persistence helper (CORRECTION, added).

Root-cause diagnostic report, Secções 6/7/8/11 (recomendação #3): nenhuma
função de explicabilidade (contrafactual, ACI, SHAP, permutação) tinha
acesso ao StandardScaler usado em tempo de treino, porque apenas o objecto
`model` era gravado em pickle — nunca o scaler nem a lista de colunas usada.

Esta correcção substitui `pickle.dump(model, f)` / `pickle.load(f)` por um
"bundle" {"model", "scaler", "feat_cols"} persistido e recuperado por estas
duas funções, usadas consistentemente em:
  - validation/walk_forward.py   (grava, ao fim de cada spec×modelo)
  - pipeline.py _train_and_evaluate  (grava o modelo final)
  - explainability/ablation.py   (lê; mantém compatibilidade com pickles antigos)
  - explainability/innovations.py (lê; usa scaler antes de qualquer predict())
  - pipeline.py _run_explainability (lê; escalona X_all/X_test antes de SHAP/permutação)

Mantém compatibilidade retroativa: se um ficheiro .pkl antigo (gravado antes
desta correcção) for lido, ele contém apenas o objecto do modelo bruto — a
função devolve scaler=None e feat_cols=None nesse caso, e quem chama deve
assumir os dados já estão na escala em que o modelo foi treinado (como
acontecia antes) ou tratar isso como "sem garantia de escala".
"""
import pickle


def save_model_bundle(path: str, model, scaler=None, feat_cols=None) -> None:
    bundle = {"model": model, "scaler": scaler, "feat_cols": list(feat_cols) if feat_cols is not None else None}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: str):
    """Returns (model, scaler_or_None, feat_cols_or_None)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("scaler"), obj.get("feat_cols")
    # Backward compatibility with pre-correction pickles (bare model object)
    return obj, None, None
