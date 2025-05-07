from lingua import Language, LanguageDetectorBuilder
import stanza
import fasttext
from huggingface_hub import hf_hub_download
import collections
from pprint import pp
import pathlib
import json
import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluate language mixing patterns in model outputs')
parser.add_argument('--data_dir', type=str, default="experiments/crosslingual_mgsm/artifacts/s1",
                    help='Directory containing model outputs')
parser.add_argument('--output_dir', type=str, default="experiments/language_mixing/artifacts/",
                    help='Directory to save analysis results')
parser.add_argument('--artifact_names', type=str, default="s1.1-32B-*8000",
                    help='Glob pattern to match artifact folders')

args = parser.parse_args()
data_dir = pathlib.Path(args.data_dir)
output_dir = pathlib.Path(args.output_dir)
artifact_names = args.artifact_names

def analyze_cm(lid_detector, text):
    """
    return sents -> annotations (substring, language, confidence)
    """

    # 1. get the overall language prediction (dominant language) for the document.
    lid_pred_confid = lid_detector.compute_language_confidence_values(text)
    if lid_pred_confid[0].value <= 0.95:
        print(f"Warning: dominant language prediction has only confidence {lid_pred_confid[0].value:.2f}%.")
    
    # 2. segment the document into sentences
    nlp = stanza.Pipeline(lang=lid_pred_confid[0].language.iso_code_639_1.name.lower(), processors='tokenize')
    doc = nlp(text)
    segmented_sents = [sentence.text for sentence in doc.sentences]
    
    # 3. extract language code for potentially code-mixed sentences
    result = collections.defaultdict(list) # sents to annotations (substring, language, confidence)
    for sent_i, sent in enumerate(segmented_sents):
        multiple_langs_annos = lid_detector.detect_multiple_languages_of(sent)
        for anno in multiple_langs_annos:
            substring = sent[anno.start_index:anno.end_index] # anno doesn't come with confidence
            lid_pred_confid = lid_detector.compute_language_confidence_values(substring)
            lang, conf = lid_pred_confid[0].language.name, lid_pred_confid[0].value
            # print(substring, lang, conf)
            result[sent].append((sent_i, substring, lang, conf))
    return result
###########

########################################################################
# ru, ja, th, zh
LANGMAP = {
    "ru": Language.RUSSIAN,
    "ja": Language.JAPANESE,
    "th": Language.THAI,
    "zh": Language.CHINESE,
}

result = collections.defaultdict(dict) # lang -> lang -> cm_cases

for folder in data_dir.glob(artifact_names):
    langcode = folder.name.rsplit("_", 1)[0][-2:]
    if langcode not in LANGMAP:
        continue
    print(folder)
    languages = [Language.ENGLISH, 
                Language.CHINESE,
                LANGMAP[langcode]] # specific language (prompt language)
    lid_detector = LanguageDetectorBuilder.from_languages(*languages).build()

    cm_instances = total_instances = 0
    cm_intra_sents = no_cm_only_en = no_cm_native = total_sents = 0
    for MODEL in folder.glob("*"):
        for fp in MODEL.glob("samples_*"):
            texts = list()
            sent_is = list()

            sent_sus = list()
            sent_extract = list()
            with open(fp) as rf:
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
                        
                        has_cm = False
                        for k, v in res.items():
                            total_sents += 1
                            if len(v) > 1: 
                                # sentence has code-mixing
                                sent_i = v[0][0]
                                languages = set([e[2] for e in v])
                                if len(languages) == 1:
                                    if list(languages)[0] == "ENGLISH":
                                        no_cm_only_en += 1
                                    else:
                                        no_cm_native += 1
                                    continue
                                
                                # check for quote-and-think
                                has_cm = True
                                has_quote_extract = is_quote_extract_precise = False
                                for e in v:
                                    if e[2] != "ENGLISH" and ("\"" in e[1] or "\'" in e[1]):
                                        has_quote_extract=True
                                        extracted_phrase = e[1].replace("\"", "").replace("\'", "").strip()
                                        if extracted_phrase in question:
                                            is_quote_extract_precise = True
                                        else:
                                            # print(e[1])
                                            # print(extracted_phrase)
                                            # print(question)
                                            # assert False
                                            ...
                                
                                if has_quote_extract:
                                    # quote-and-think pattern
                                    sent_extract.append((question, v, is_quote_extract_precise))
                                else:
                                    # flagged sentences for manual annotations
                                    sent_sus.append((question, v))
                                
                                # total intrasentential code-mixing sentences (including quote-and-think)
                                cm_intra_sents += 1
                            else:
                                languages = set([e[2] for e in v])
                                if len(languages) == 1:
                                    if list(languages)[0] == "ENGLISH":
                                        no_cm_only_en += 1
                                    else:
                                        no_cm_native += 1

                        ### instance having code-mixing
                        if has_cm:
                            cm_instances += 1
                        total_instances += 1
                    except:
                        raise

    count_quote_think_precise = sum([is_quote_extract_precise for question, e, is_quote_extract_precise in sent_extract])
    wf = open(output_dir / f"{folder.name}.txt", "w+")
    wf.write(f"{cm_instances=}\n")
    wf.write(f"{total_instances=}\n")
    wf.write(f"\n")
    wf.write(f"{total_sents=}\n")
    wf.write(f"{no_cm_only_en=}\n")
    wf.write(f"{no_cm_native=}\n")
    wf.write(f"{cm_intra_sents=}\n")
    wf.write(f"- {len(sent_extract)=}\n")
    wf.write(f"- {count_quote_think_precise=}\n")
    wf.write(f"- {len(sent_sus)=}\n")
    wf.write(f"======\n")
    wf.write(f"Chances of an instance having CM: {cm_instances/total_instances}\n")
    wf.write(f"Chances of a sentence having CM: {cm_intra_sents/total_sents}\n")
    wf.write(f"Absolute count of quote-and-think: {len(sent_extract)}\n")
    wf.write(f"Absolute count of intersentential: {no_cm_native}\n")
    wf.write(f"Absolute count of flagged: {len(sent_sus)}\n")
    wf.write(f"======\n")
    
    # print(sent_sus[0])
    
    sus_data = []
    for question, e in sent_sus:
        sent = ''.join(x[1] for x in e)
        sus_data.append([question, sent])
    
    sus_df = pd.DataFrame(sus_data, columns=['question', 'sentence'])
    sus_df.to_csv(output_dir / f"flagged_{folder.name}.tsv", sep='\t', index=False)