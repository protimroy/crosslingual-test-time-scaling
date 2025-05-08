import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from pathlib import Path
OPENAI_TOKEN = "<TOKEN>"
client = OpenAI(api_key=OPENAI_TOKEN)

def parse_and_read_mmlu_file(filename):
    # Regular expression to extract language ID and domain
    pattern = r'samples_global_mmlu_([a-z]{2})_([a-z_]+)_\d{4}-\d{2}-\d{2}T'
    
    match = re.search(pattern, filename)
    if not match:
        return None, None, None
    
    language_id = match.group(1)
    domain = match.group(2)

    # Read and parse the JSONL file
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        return language_id, domain, None
    
    return language_id, domain, data


def extract_answer(text):
    q = text['doc']['question']
    options = [text['doc'][f'option_{x}'] for x in "abcd"]
    response = text['resps'][0][0].split("<|im_start|>answer")[-1].strip()
    answer_options = f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}"
    template = """Your task is to extract answer (A, B, C, or D) from the generated response based on the question and the option choices.\n\nQuestion: {q}\nAnswer choices:\n{answer_options}\n\nResponse: \'{response}\'"""
    user_query = template.format(q=q, answer_options=answer_options, response=response)

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": "Your task is to extract the answer choice from the Response field. Do not attempt answer the question in the Question field yourself."
        },
        {
        "role": "user",
        "content": user_query
        }
    ],
    temperature=0,
    max_tokens=2,
    )
    res = response.choices[0].message.content
    return res

def compute_exact_match(data, wf=None):
    total = len(data)
    correct = 0
    
    # Initialize counters for CS and CA categories
    cs_total = 0
    cs_correct = 0
    
    ca_total = 0
    ca_correct = 0
    
    for item in data:
        answer = extract_answer(item)[0]
        
        # save the extracted answer to the item
        item['extracted_answer'] = answer
        wf.write(json.dumps(item) + "\n")
        
        cultural_sensitivity = item['doc'].get('cultural_sensitivity_label', '')
        
        # Update overall counts
        if answer: 
            if answer == item['target']:
                correct += 1
        
        # Update category-specific counts
        if cultural_sensitivity == 'CS':
            cs_total += 1
            if answer:
                if answer == item['target']:
                    cs_correct += 1
        elif cultural_sensitivity == 'CA':
            ca_total += 1
            if answer:
                if answer == item['target']:
                    ca_correct += 1
                    
    
    # Calculate overall accuracy metrics
    overall_accuracy = correct / total if total > 0 else 0
    
    # Calculate CS accuracy metrics
    cs_accuracy = cs_correct / cs_total if cs_total > 0 else 0

    # Calculate CA accuracy metrics
    ca_accuracy = ca_correct / ca_total if ca_total > 0 else 0

    return {
        'overall': (overall_accuracy),
        'CS': (cs_accuracy, cs_total),
        'CA': (ca_accuracy, ca_total)
    }



if __name__ == "__main__":
    model_size = '32'
    budget = '8000'
    model_name = "s1.1"
    dataset_name = "global_mmlu"
    folder_path = f"{model_name}-{model_size}B-{dataset_name}_{budget}/"
    folder = Path(folder_path)

    # Walk through the directory and its subdirectories
    output_results = []
    for path in tqdm(folder.rglob('*.jsonl')):
        print(f"Processing: {path.name}")
        wf = open(folder_path+path.name.replace(".jsonl", "_extracted.jsonl"), "w+", encoding='utf-8')
        language_id, domain, data = parse_and_read_mmlu_file(f"{folder_path}/{str(path.name)}")
        
        if data is None:
            print(f"Skipping {path} - not an MMLU file or error reading file")
            continue
            
        evaluation_results = compute_exact_match(data, wf=wf)
        accuracy= evaluation_results['overall']
        cs_accuracy, cs_total = evaluation_results['CS']
        ca_accuracy, ca_total = evaluation_results['CA']
        print(f"Language ID: {language_id}, Domain: {domain}, Accuracy: {accuracy:.4f}")
        output_results.append({
            'language_id': language_id,
            'domain': domain,
            'accuracy': accuracy,
            'cs_accuracy': cs_accuracy,
            'ca_accuracy': ca_accuracy,
        })

    # Save the results to a CSV file
    df = pd.DataFrame(output_results)
    df.to_csv(f"{folder_path}/results_CS_CA_api.csv", index=False)

  




