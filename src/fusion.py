def fuse_predictions(preds: list[dict], weights: dict | None = None) -> dict:
    if weights is None:
        weights = {"text": 1.0, "image": 1.2, "audio": 1.2, "video": 1.2}

    agg = {}
    for p in preds:
        mod = p.get("modality", "unknown")
        emo = p.get("emotion", "unknown")
        score = float(p.get("score", 0.0))
        w = float(weights.get(mod, 1.0))
        agg[emo] = agg.get(emo, 0.0) + score * w

    final_emotion, final_score = max(agg.items(), key=lambda x: x[1])
    return {"final_emotion": final_emotion, "fusion_scores": agg, "final_score": final_score}
