from lingua import Language, LanguageDetectorBuilder
import stanza
import collections
import pathlib
import json
import pandas as pd
import argparse
import re


# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(
    description='Evaluate language mixing + Translate-and-Think behaviors in model outputs'
)
parser.add_argument(
    '--data_dir',
    type=str,
    default="experiments/crosslingual_mgsm/artifacts/s1",
    help='Directory containing model outputs',
)
parser.add_argument(
    '--output_dir',
    type=str,
    default="experiments/language_mixing/artifacts_protim/",
    help='Directory to save analysis results',
)
parser.add_argument(
    '--artifact_names',
    type=str,
    default="s1.1-32B-*8000",
    help='Glob pattern to match artifact folders',
)

args = parser.parse_args()
data_dir = pathlib.Path(args.data_dir)
output_dir = pathlib.Path(args.output_dir)
artifact_names = args.artifact_names
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def analyze_cm(lid_detector, text):
    """
    return sents -> annotations (substring, language, confidence)
    """
    lid_pred_confid = lid_detector.compute_language_confidence_values(text)
    if lid_pred_confid and lid_pred_confid[0].value <= 0.95:
        print(
            f"Warning: dominant language prediction has only confidence {lid_pred_confid[0].value:.2f}%."
        )

    # segment document into sentences
    nlp = stanza.Pipeline(
        lang=lid_pred_confid[0].language.iso_code_639_1.name.lower(),
        processors='tokenize',
    )
    doc = nlp(text)
    segmented_sents = [sentence.text for sentence in doc.sentences]

    result = collections.defaultdict(list)
    for sent_i, sent in enumerate(segmented_sents):
        multiple_langs_annos = lid_detector.detect_multiple_languages_of(sent)
        for anno in multiple_langs_annos:
            substring = sent[anno.start_index:anno.end_index]
            lid_pred_confid = lid_detector.compute_language_confidence_values(substring)
            lang, conf = lid_pred_confid[0].language.name, lid_pred_confid[0].value
            result[sent].append((sent_i, substring, lang, conf))
    return result


def detect_translate_and_think(question, think, res):
    """
    Heuristics to detect Translate-and-Think behaviors:
    1. Explicit: mentions translation or "in English/Chinese this means..."
    2. Silent: dominant shift to English mid-CoT but final answer in non-English
    """
    explicit_markers = re.search(
        r"(translate|in English|means that|in Chinese)",
        think,
        re.IGNORECASE,
    )

    # language distribution per sentence
    lang_counts = collections.Counter()
    for sent, spans in res.items():
        langs = [e[2] for e in spans]
        for l in langs:
            lang_counts[l] += 1

    # Silent TaT heuristic: lots of English spans + final answer language ≠ English
    silent_detected = False
    if lang_counts.get("ENGLISH", 0) > sum(lang_counts.values()) * 0.3:  # >30% English
        if not question.strip().startswith("In English"):  # input wasn't English
            silent_detected = True

    return explicit_markers is not None, silent_detected, lang_counts


# -----------------------------
# Self-correction detection
# -----------------------------
_SELF_CORR_MARKERS = re.compile(
    r"\b("
    r"sorry|actually|wait|hold on|on second thought|to correct|correction"
    r"|i (?:was|am|were) wrong|i (?:made|make) (?:a )?mistake"
    r"|let me (?:fix|correct)|to be precise|more accurately"
    r"|should be|the correct (?:value|answer|result) is"
    r")\b",
    re.IGNORECASE,
)

_CONTRASTIVES = re.compile(
    r"\b(but|however|though|yet|nevertheless|nonetheless|instead|rather)\b",
    re.IGNORECASE,
)

_EDIT_PATTERNS = re.compile(
    r"(\S.*?\S)\s*(?:->|→|⇒|becomes|change(?:s|d)? to)\s*(\S.*?\S)",
    re.IGNORECASE,
)

_NUM_REGEX = re.compile(r"\d+(?:\.\d+)?")


def _numbers(s):
    return set(_NUM_REGEX.findall(s))


def detect_self_correction(think, res):
    """
    Detect self-corrections in `think` using sentence-level analysis + heuristics.
    Returns:
      has_correction (bool),
      details (list of dicts),
      score (float)
    """
    # Build ordered sentences from `res` (sentence text -> spans with sent index)
    ordered = []
    for sent_text, spans in res.items():
        if not spans:
            continue
        sent_idx = spans[0][0]
        ordered.append((sent_idx, sent_text))
    ordered.sort(key=lambda x: x[0])

    details = []
    seen_nums = set()
    score = 0.0

    for sent_idx, sent in ordered:
        reasons = []
        markers = [m.group(0) for m in _SELF_CORR_MARKERS.finditer(sent)]
        if markers:
            reasons.append("explicit_markers")

        if _CONTRASTIVES.search(sent):
            reasons.append("contrastive_revision")

        if _EDIT_PATTERNS.search(sent):
            reasons.append("edit_pattern")

        # Numeric revision: new numbers appear alongside correction cues
        cur_nums = _numbers(sent)
        numeric_new = bool(cur_nums - seen_nums)
        has_numeric_cue = bool(markers) or re.search(
            r"\b(recompute|recalculate|fix|correct|should be|not|instead)\b",
            sent,
            re.IGNORECASE,
        )
        if numeric_new and has_numeric_cue:
            reasons.append("numeric_revision")

        if reasons:
            details.append(
                {
                    "sent_idx": sent_idx,
                    "sentence": sent.strip(),
                    "reasons": reasons,
                    "markers": markers,
                    "nums_before": sorted(seen_nums),
                    "nums_after": sorted(cur_nums),
                }
            )
            # Scoring: explicit markers/edit patterns weigh more; contrast adds a bit
            sc = 0.0
            if "explicit_markers" in reasons:
                sc += 1.0
            if "edit_pattern" in reasons:
                sc += 0.8
            if "numeric_revision" in reasons:
                sc += 0.7
            if "contrastive_revision" in reasons:
                sc += 0.3
            score += sc

        # Update seen numbers after analyzing sentence
        seen_nums |= cur_nums

    has_correction = len(details) > 0
    return has_correction, details, score


# -----------------------------
# Language map
# -----------------------------
LANGMAP = {
    "ru": Language.RUSSIAN,
    "ja": Language.JAPANESE,
    "th": Language.THAI,
    "zh": Language.CHINESE,
}


# -----------------------------
# Main loop
# -----------------------------
result = collections.defaultdict(dict)

for folder in data_dir.glob(artifact_names):
    langcode = folder.name.rsplit("_", 1)[0][-2:]
    if langcode not in LANGMAP:
        continue

    print(folder)

    languages = [Language.ENGLISH, Language.CHINESE, LANGMAP[langcode]]
    lid_detector = LanguageDetectorBuilder.from_languages(*languages).build()

    cm_instances = total_instances = 0
    cm_intra_sents = no_cm_only_en = no_cm_native = total_sents = 0
    tat_explicit = tat_silent = 0

    # Self-correction accumulators
    selfcorr_instances = 0
    selfcorr_rows = []
    selfcorr_score_sum = 0.0

    sent_sus = []
    sent_extract = []

    for MODEL in folder.glob("*"):
        for fp in MODEL.glob("samples_*"):
            with open(fp, encoding='utf-8') as rf:
                for i, line in enumerate(rf):
                    if i >= 250:
                        break

                    line = json.loads(line)
                    question = line['doc']['question']
                    response = line["resps"][0][0]

                    try:
                        think, answer = response.rsplit("<|im_start|>answer\n", 1)
                        think = think.removeprefix("<|im_start|>think\n")

                        res = analyze_cm(lid_detector, think)

                        # Detect self-correction
                        has_corr, corr_details, corr_score = detect_self_correction(think, res)
                        if has_corr:
                            selfcorr_instances += 1
                            selfcorr_score_sum += corr_score
                            for d in corr_details:
                                selfcorr_rows.append(
                                    [
                                        question,
                                        d["sent_idx"],
                                        d["sentence"],
                                        ",".join(d["reasons"]),
                                        ",".join(d["markers"]) if d["markers"] else "",
                                        " ".join(d["nums_before"]),
                                        " ".join(d["nums_after"]),
                                    ]
                                )

                        # --- Detect Translate-and-Think behaviors ---
                        explicit, silent, lang_counts = detect_translate_and_think(
                            question, think, res
                        )
                        if explicit:
                            tat_explicit += 1
                        if silent:
                            tat_silent += 1

                        has_cm = False
                        for k, v in res.items():
                            total_sents += 1
                            if len(v) > 1:
                                sent_i = v[0][0]
                                languages = set([e[2] for e in v])

                                if len(languages) == 1:
                                    if list(languages)[0] == "ENGLISH":
                                        no_cm_only_en += 1
                                    else:
                                        no_cm_native += 1
                                    continue

                                # Quote-and-Think check
                                has_cm = True
                                has_quote_extract = False
                                is_quote_extract_precise = False
                                for e in v:
                                    if e[2] != "ENGLISH" and (
                                        ("\"" in e[1]) or ("\'" in e[1])
                                    ):
                                        has_quote_extract = True
                                        extracted_phrase = (
                                            e[1].replace("\"", "")
                                            .replace("\'", "")
                                            .strip()
                                        )
                                        if extracted_phrase in question:
                                            is_quote_extract_precise = True

                                if has_quote_extract:
                                    sent_extract.append(
                                        (question, v, is_quote_extract_precise)
                                    )
                                else:
                                    sent_sus.append((question, v))
                                cm_intra_sents += 1
                            else:
                                languages = set([e[2] for e in v])
                                if len(languages) == 1:
                                    if list(languages)[0] == "ENGLISH":
                                        no_cm_only_en += 1
                                    else:
                                        no_cm_native += 1

                        if has_cm:
                            cm_instances += 1
                        total_instances += 1
                    except:
                        raise

    count_quote_think_precise = sum(
        [is_quote_extract_precise for _, _, is_quote_extract_precise in sent_extract]
    )

    with open(output_dir / f"{folder.name}.txt", "w+", encoding="utf-8") as wf:
        wf.write(f"{cm_instances=}\n")
        wf.write(f"{total_instances=}\n\n")

        wf.write(f"{total_sents=}\n")
        wf.write(f"{no_cm_only_en=}\n")
        wf.write(f"{no_cm_native=}\n")
        wf.write(f"{cm_intra_sents=}\n")

        wf.write(f"- {len(sent_extract)=}\n")
        wf.write(f"- {count_quote_think_precise=}\n")
        wf.write(f"- {len(sent_sus)=}\n")

        wf.write("======\n")
        wf.write(f"Chances of an instance having CM: {cm_instances/total_instances:.3f}\n" if total_instances else "Chances of an instance having CM: n/a\n")
        wf.write(f"Chances of a sentence having CM: {cm_intra_sents/total_sents:.3f}\n" if total_sents else "Chances of a sentence having CM: n/a\n")
        wf.write(f"Absolute count of quote-and-think: {len(sent_extract)}\n")
        wf.write(f"Absolute count of intersentential: {no_cm_native}\n")
        wf.write(f"Absolute count of flagged: {len(sent_sus)}\n")

        # Self-correction summary
        avg_corr_score = (selfcorr_score_sum / selfcorr_instances) if selfcorr_instances else 0.0
        wf.write("======\n")
        wf.write(f"Self-correction instances: {selfcorr_instances}\n")
        wf.write(f"Avg self-correction score: {avg_corr_score:.3f}\n")

        wf.write("======\n")
        wf.write(f"Translate-and-Think explicit: {tat_explicit}\n")
        wf.write(f"Translate-and-Think silent: {tat_silent}\n")
        wf.write(f"Total TaT: {tat_explicit + tat_silent}\n")

    # Save flagged sentences
    sus_data = []
    for question, e in sent_sus:
        sent = ''.join(x[1] for x in e)
        sus_data.append([question, sent])

    if sus_data:
        pd.DataFrame(sus_data, columns=['question', 'sentence']).to_csv(
            output_dir / f"flagged_{folder.name}.tsv", sep='\t', index=False
        )

    # Save self-corrections
    if selfcorr_rows:
        pd.DataFrame(
            selfcorr_rows,
            columns=[
                "question",
                "sent_idx",
                "sentence",
                "reasons",
                "markers",
                "nums_before",
                "nums_after",
            ],
        ).to_csv(output_dir / f"selfcorr_{folder.name}.tsv", sep='\t', index=False)