import json
from openai import OpenAI
from tqdm import tqdm
import os
import argparse

client = OpenAI()

# Set up argument parser
parser = argparse.ArgumentParser(description='Extract answers from model responses using OpenAI API')
parser.add_argument('--input_file', type=str, 
                    default="experiments/crossdomain/artifacts/copal_id/s1.1-32B-copal_id_500/simplescaling__s1.1-32B/samples_s1_copal_id_standard_2025-04-06T01-01-07.875581.jsonl",
                    help='Path to the input JSONL file containing model responses')
args = parser.parse_args()

##############
fp = args.input_file
write_fp = fp.replace("/samples_", "/extracted_samples_")
print(f"write to {write_fp}")
##############

def get_response(q, response):
	template = """Your task is to extract answer (A or B) from the Response field based on the question and the option choices. Output 'X' if no answer can be extracted.\n\nQuestion: {q}\n\nResponse: \'{response}\'"""
	user_query = template.format(q=q, response=response)
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

# Check if write_fp exists and compare line counts
def should_process_file(input_fp, output_fp):
	# If output file doesn't exist, we should process
	if not os.path.exists(output_fp):
		return True
	
	# Count lines in input file
	input_line_count = 0
	with open(input_fp) as f:
		for _ in f:
			input_line_count += 1

	# Count lines in output file
	output_line_count = 0
	with open(output_fp) as f:
		for _ in f:
			output_line_count += 1
	
	# Process if output has fewer lines than input
	return output_line_count < input_line_count

def process_fp(fp, write_fp):
	# Check if write_fp exists and get the number of processed lines
	existing_lines = 0
	if os.path.exists(write_fp):
		with open(write_fp) as f:
			for _ in f:
				existing_lines += 1
		wf = open(write_fp, "a")  # Append mode if file exists
	else:
		wf = open(write_fp, "w+")  # Create new file if it doesn't exist
		
	with open(fp) as rf:
		for i, line in tqdm(enumerate(rf)):
			# Skip lines that have already been processed
			if i < existing_lines:
				continue
				
			line = json.loads(line)
			question = line['arguments']['gen_args_0']['arg_0'].split("|im_start|>user")[-1].split("<|im_end|>")[0]
			response = line['resps'][0][0].split("<|im_start|>answer")[-1].strip()
			extracted_answer = get_response(question, response)
			line['extracted_answer'] = extracted_answer

			wf.write(json.dumps(line) + "\n")

############################################
### extract answers
if should_process_file(fp, write_fp):
	process_fp(fp, write_fp)

### get accuracy
with open(write_fp) as rf:
	for line in rf:
		line = json.loads(line)

		correct += int(line["extracted_answer"] == line["target"])
		total += 1

print(f"accuracy: {correct/total*100:.2f}")