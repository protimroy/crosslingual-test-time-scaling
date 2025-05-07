from lingua import Language, LanguageDetectorBuilder
import stanza
import fasttext
from huggingface_hub import hf_hub_download
import collections
from pprint import pp
from tqdm import tqdm
import pathlib
import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluate dominant language in model outputs')
parser.add_argument('--data_dir', type=str, default="experiments/crosslingual_mgsm/artifacts/s1",
                    help='Directory containing model outputs')
parser.add_argument('--artifact_names', type=str, default="s1.1-14B-mgsm_direct*8000",
                    help='Glob pattern to match artifact folders')

args = parser.parse_args()
data_dir = pathlib.Path(args.data_dir)
artifact_names = args.artifact_names

def analyze_cm_dominant(lid_detector, text):
    """
    return sents -> annotations (substring, language, confidence)
    """

    # 1. get the overall language prediction (dominant language) for the document.
    lid_pred_confid = lid_detector.compute_language_confidence_values(text)
    if lid_pred_confid[0].value <= 0.95:
        print(f"Warning: dominant language prediction has only confidence {lid_pred_confid[0].value:.2f}%.")
    lang = lid_pred_confid[0].language.iso_code_639_1.name.lower()

    return lang

def skip_equation(substring):
    if "\\times" in substring or "\\text" in substring or "\\frac" in substring \
        or "\\right" in substring or "\\left" in substring or "\\begin" in substring or "\\end" in substring:
        return True

    return False
    
########################################################################
# ru, ja, th, zh
LANGMAP = {
    "ru": Language.RUSSIAN,
    "ja": Language.JAPANESE,
    "th": Language.THAI,
    "zh": Language.CHINESE,
}

result = collections.defaultdict(dict) # query_lang -> dominant_lang -> counter

for folder in data_dir.glob(artifact_names):
    langcode = folder.name.rsplit("_", 1)[0][-2:]
    if langcode not in LANGMAP:
        continue

    languages = [Language.ENGLISH, 
                Language.CHINESE,
                LANGMAP[langcode]] # specific language (prompt language)
    lid_detector = LanguageDetectorBuilder.from_languages(*languages).build()

    for MODEL in folder.glob("*"):
        for fp in MODEL.glob("samples_*"):
            print(fp)
            with open(fp) as rf:
                for i, line in tqdm(enumerate(rf)):
                    if i >= 250:
                        break 
                    line = json.loads(line)
                    response = line["resps"][0][0]
                    response = response.replace("Step-by-Step Answer:", "")
                    dominant_lang = analyze_cm_dominant(lid_detector, response)
                    result[langcode][dominant_lang] = result[langcode].get(dominant_lang, 0) + 1
                    
pp(result)