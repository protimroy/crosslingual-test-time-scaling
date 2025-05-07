import datasets

def _process_docs(dataset: datasets.Dataset, prompt_style: str = None) -> datasets.Dataset:
    def _process_doc(doc):
        assert len(doc['turns']) == 1 # all are single-turn
        out_doc = {
            "prompt": doc["turns"][0],
            "prompt_style": doc["prompt_style"],
            "question_id": doc["question_id"],
            "category": doc["category"],
        } 
        return out_doc

    docs = dataset.map(_process_doc)
    if not prompt_style:
        return docs
    return docs.filter(lambda ex: ex["prompt_style"] == prompt_style)

def process_docs_translate_fr(dataset):
    return _process_docs(dataset, prompt_style="translate-fr")

def process_docs_translate_ml(dataset):
    return _process_docs(dataset, prompt_style="translate-ml")

def process_docs_translate_mr(dataset):
    return _process_docs(dataset, prompt_style="translate-mr")

def process_docs_translate_ta(dataset):
    return _process_docs(dataset, prompt_style="translate-ta")

def process_docs_translate_zh(dataset):
    return _process_docs(dataset, prompt_style="translate-zh-cn")

def process_docs_base(dataset):
    return _process_docs(dataset, prompt_style="base")