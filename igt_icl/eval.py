"""
Contains the evaluation scripts for comparing predicted and gold IGT
Adapted from SIGMORPHON 2023 Shared Task code
"""

import pandas as pd
import re
from jiwer import wer
from typing import List
import evaluate
from torchtext.data.metrics import bleu_score


def strip_gloss_punctuation(glosses: str):
    """Strips any punctuation from gloss string (assuming it is surrounded by spaces)"""
    return re.sub(r"(\s|^)[^\w\s](\s|$)", " ", glosses).strip()


def eval_accuracy(pred: List[List[str]], gold: List[List[str]]) -> dict:
    """Computes the average and overall accuracy, where predicted labels must be
    in the correct position in the list."""
    total_correct_predictions = 0
    total_tokens = 0
    summed_accuracies = 0

    for entry_pred, entry_gold, i in zip(pred, gold, range(len(gold))):
        entry_correct_predictions = 0

        for token_index in range(len(entry_gold)):
            # For each token, check if it matches
            if (
                token_index < len(entry_pred)
                and entry_pred[token_index] == entry_gold[token_index]
                and entry_pred[token_index] != "[UNK]"
            ):
                entry_correct_predictions += 1

        entry_accuracy = entry_correct_predictions / len(entry_gold)
        summed_accuracies += entry_accuracy

        total_correct_predictions += entry_correct_predictions
        total_tokens += len(entry_gold)

    total_entries = len(gold)
    average_accuracy = summed_accuracies / total_entries
    overall_accuracy = total_correct_predictions / total_tokens
    return {"average_accuracy": average_accuracy, "accuracy": overall_accuracy}


def eval_stems_grams(pred: List[List[str]], gold: List[List[str]]) -> dict:
    perf = {
        "stem": {"correct": 0, "pred": 0, "gold": 0},
        "gram": {"correct": 0, "pred": 0, "gold": 0},
    }

    for entry_pred, entry_gold in zip(pred, gold):
        for token_index in range(len(entry_gold)):
            # We can determine if a token is a stem or gram by checking if it is
            # all uppercase
            token_type = "gram" if entry_gold[token_index].isupper() else "stem"
            perf[token_type]["gold"] += 1

            if token_index < len(entry_pred):
                pred_token_type = (
                    "gram" if entry_pred[token_index].isupper() else "stem"
                )
                perf[pred_token_type]["pred"] += 1

                if entry_pred[token_index] == entry_gold[token_index]:
                    # Correct prediction
                    perf[token_type]["correct"] += 1

    stem_perf = {
        "prec": 0
        if perf["stem"]["pred"] == 0
        else perf["stem"]["correct"] / perf["stem"]["pred"],
        "rec": perf["stem"]["correct"] / perf["stem"]["gold"],
    }
    if (stem_perf["prec"] + stem_perf["rec"]) == 0:
        stem_perf["f1"] = 0
    else:
        stem_perf["f1"] = (
            2
            * (stem_perf["prec"] * stem_perf["rec"])
            / (stem_perf["prec"] + stem_perf["rec"])
        )

    gram_perf = {
        "prec": 0
        if perf["gram"]["pred"] == 0
        else perf["gram"]["correct"] / perf["gram"]["pred"],
        "rec": perf["gram"]["correct"] / perf["gram"]["gold"],
    }
    if (gram_perf["prec"] + gram_perf["rec"]) == 0:
        gram_perf["f1"] = 0
    else:
        gram_perf["f1"] = (
            2
            * (gram_perf["prec"] * gram_perf["rec"])
            / (gram_perf["prec"] + gram_perf["rec"])
        )
    return {"stem": stem_perf, "gram": gram_perf}


def eval_error_rate(pred: List[str], gold: List[str]) -> float:
    prediction = ' '.join(pred)
    reference = ' '.join(gold)
    return wer(reference, prediction)


def eval_avg_error_rate(pred: List[List[str]], gold: List[List[str]]) -> float:
    total_error_rate = 0
    for (sample_pred, sample_gold) in zip(pred, gold):
        total_error_rate += eval_error_rate(sample_pred, sample_gold)
    avg_error_rate = total_error_rate / len(pred)
    return avg_error_rate


def eval_morpheme_glosses(
    pred_morphemes: List[List[str]], gold_morphemes: List[List[str]]
):
    """Evaluates the performance at the morpheme level"""
    morpheme_eval = eval_accuracy(pred_morphemes, gold_morphemes)
    class_eval = eval_stems_grams(pred_morphemes, gold_morphemes)
    bleu = bleu_score(pred_morphemes, [[line] for line in gold_morphemes])
    mer = eval_avg_error_rate(pred_morphemes, gold_morphemes)
    return {"morpheme_level": morpheme_eval, "classes": class_eval, "bleu": bleu, "MER": mer}


def eval_word_glosses(pred_words: List[List[str]], gold_words: List[List[str]]):
    """Evaluates the performance at the morpheme level"""
    word_eval = eval_accuracy(pred_words, gold_words)
    bleu = bleu_score(pred_words, [[line] for line in gold_words])
    wer = eval_avg_error_rate(pred_words, gold_words)
    return {"word_level": word_eval, "bleu": bleu, "WER": wer}


def evaluate_igt(
    predictions: List[str],
    references: List[str],
):
    """Computes metrics over IGT predictions

    Args:
        predictions (List[str]): List of predicted gloss lines.
        references (List[str]): List of reference gloss lines.

    Returns:
        (Dict): A dictionary of several metrics
    """
    predictions = [strip_gloss_punctuation(pred) for pred in predictions]
    references = [strip_gloss_punctuation(g) for g in references]

    pred_words = [pred.split() for pred in predictions]
    gold_words = [gloss.split() for gloss in references]
    pred_morphemes = [re.split(r"\s|-", pred) for pred in predictions]
    gold_morphemes = [re.split(r"\s|-", gloss) for gloss in references]

    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(
        predictions=references, references=references, word_order=2
    )

    return {
        **eval_word_glosses(
            pred_words=pred_words, gold_words=gold_words
        ),
        **eval_morpheme_glosses(
            pred_morphemes=pred_morphemes, gold_morphemes=gold_morphemes
        ),
        'chrf': chrf_score['score'],
    }
