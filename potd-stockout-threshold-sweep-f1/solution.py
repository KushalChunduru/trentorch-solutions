def best_f1_threshold(scores: list[float], y: list[int]) -> tuple[float, float]:
    if not scores or len(scores) != len(y):
        raise ValueError("scores and y must have the same non-zero length.")

    total_positive = sum(y)
    if total_positive == 0:
        raise ValueError("Recall is undefined for every threshold: no positive labels.")

    samples = sorted(zip(scores, y), reverse=True)
    best_threshold = samples[0][0]
    best_num, best_den = 0, 1
    tp = fp = 0
    i = 0

    while i < len(samples):
        threshold = samples[i][0]

        # Include all samples with this score before evaluating.
        while i < len(samples) and samples[i][0] == threshold:
            label = samples[i][1]
            tp += label
            fp += 1 - label
            i += 1

        # F1 = 2*TP / (2*TP + FP + FN), where FN = total_positive - TP.
        numerator = 2 * tp
        denominator = total_positive + tp + fp

        # Exact comparison; strict improvement preserves the larger threshold.
        if numerator * best_den > best_num * denominator:
            best_threshold = threshold
            best_num, best_den = numerator, denominator

    return best_threshold, best_num / best_den
