def ctc_greedy_decode(log_probs, blank=0):
    best_path = log_probs.argmax(dim=-1)
    decoded = []
    prev = blank
    for idx in best_path:
        if idx != blank and idx != prev:
            decoded.append(idx.item())
        prev = idx
    return decoded


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Edit (Levenshtein) Distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]


from khmernormalizer import normalize as khmer_normalize

def calculate_cer(preds: list, tgts: list) -> float:
    """Compute Character Error Rate (CER) across lists of predictions and ground truths."""
    total_dist = 0
    total_len = 0
    for pred, tgt in zip(preds, tgts):
        # Normalize to canonical form using khmernormalizer
        n_pred = khmer_normalize(pred)
        n_tgt = khmer_normalize(tgt)
        total_dist += levenshtein_distance(n_pred, n_tgt)
        total_len += len(n_tgt)
    return total_dist / total_len if total_len > 0 else 0.0


