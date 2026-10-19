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
