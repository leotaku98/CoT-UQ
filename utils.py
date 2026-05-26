import json
import os
import logging
import re

Dataset_Folder = 'dataset'

def print_exp(args, return_flag=0):
    info = ''
    for k, v in vars(args).items():
        info += '{}:{}\n'.format(k, v)
    print('---------------experiment args---------------')
    print(info)
    print('---------------------------------------------')
    if return_flag == 0:
        return
    elif return_flag == 1:
        return info
    else:
        pass

def load_data(args):
    decoder = json.JSONDecoder()
    questions = []
    answers = []
    ids = []
    types = []
    datapath = args.datapath if args.datapath else '{}/{}/{}.json'.format(Dataset_Folder, args.dataset, args.dataset)
    # read dataset file
    if args.dataset.lower() in ['2wikimhqa', 'gsm8k', 'hotpotqa', 'svamp', 'asdiv']:
        with open(datapath) as f:
            if args.dataset.lower() in ['hotpotqa','2wikimhqa']:
                json_data = []
                for line in f.readlines():
                    dic = json.loads(line)
                    json_data.append(dic)
            else:
                json_data = json.load(f)

            for idx, line in enumerate(json_data):
                if args.dataset.lower() in ['svamp', 'asdiv']:
                    if line['Body'][-1] != '.':
                        q = line['Body'].strip() + ". " + line["Question"].strip()
                    else:
                        q = line['Body'].strip() + " " + line["Question"].strip()
                    a = float(line["Answer"])
                    id = line["ID"]
                elif args.dataset in ['hotpotQA']:
                    q = line['question']
                    a = line['answer']
                    id = line['id']
                    t = line['type']
                    types.append(t)
                elif args.dataset in ['2WikimhQA']:
                    if line['type'] == "inference":
                        q = line['question']
                        a = line['answer']
                        id = line['_id']
                        t = line['type']
                        types.append(t)
                    else: continue
                elif args.dataset.lower() in ['gsm8k']:
                    q = line['question']
                    a = float(line['answer'])
                    id = 'temp_{}'.format(idx)
                else:
                    raise ValueError('not support dataset: {}'.format(args.dataset))
                questions.append(q)
                answers.append(a)
                ids.append(id)
            print("The Number of Different Questions: ", len(questions))
    else:
        raise ValueError('not support dataset: {}'.format(args.dataset))

    if args.test_end == 'full':
        if args.dataset.lower() in ['hotpotqa', '2wikimhqa']:
            return questions[int(args.test_start):], answers[int(args.test_start):], ids[int(args.test_start):], types[int(args.test_start):]
        else:
            return questions[int(args.test_start):], answers[int(args.test_start):], ids[int(args.test_start):]
    else:
        s, e = int(args.test_start), int(args.test_end)
        if args.dataset.lower() in ['hotpotqa', '2wikimhqa']:
            return questions[s:e], answers[s:e], ids[s:e], types[s:e]
        else:
            return questions[s:e], answers[s:e], ids[s:e]

def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()

#############################################
########### Inference Refining ###########
############################################# 

def is_effectively_empty(obj):
    
    if obj is None:
        return True

    if isinstance(obj, (int, float)) and obj == 0:
        return True

    if obj == "":
        return True

    if isinstance(obj, list):
        return all(is_effectively_empty(item) for item in obj)
    
    if isinstance(obj, dict):
        if len(obj) == 0: 
            return True
        return all(is_effectively_empty(value) for value in obj.values())
    return False

def step_exacts_2_list(response):
    # Split response into lines and filter out empty lines
    lines = response.splitlines()
    lines = [line for line in lines if line.strip()]

    keywords_by_step = []
    contributions_by_step = []
    valid_response_text = []

    for line in lines:
        # Match lines starting with "Step X:"
        match = re.search(r"Step \d+: (.+)", line)
        if match:
            if "(/" not in line or "/)" not in line:
                continue  # Skip invalid lines

            # Extract keywords with contributions
            keywords_w_contribution = match.group(1).split("; ")

            # Check for valid format and skip invalid lines
            if any("(/" not in key_w_c or "/)" not in key_w_c for key_w_c in keywords_w_contribution):
                continue

            try:
                # Extract keywords and contributions
                keywords = [key_w_c.split("(/")[0].strip() for key_w_c in keywords_w_contribution]
                contributions = [int(key_w_c.split("(/")[1].split("/)")[0].strip()) for key_w_c in keywords_w_contribution]
            except ValueError:
                return False  # Return False if contributions cannot be converted to int

            for i in contributions:
                if i > 10:
                    return False

            keywords_by_step.append(keywords)
            contributions_by_step.append(contributions)
            valid_response_text.append(line)  # Add valid lines from the original response

    # If no valid lines are found, return False
    if not valid_response_text:
        return False

    return "\n".join(valid_response_text), keywords_by_step, contributions_by_step

def parse_response_to_dict(response):
    steps = {}  
    final_answer = None

    # Match Final Answer
    match = re.search(r"Final Answer:\s*(.+?)\s*(?=(\n|$))", response, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        response_before_final_answer = response[:match.end()].strip()
    else:
        return None, None, None

    # Match Steps
    matches = list(re.finditer(r'(Step \d+):', response_before_final_answer))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response_before_final_answer)
        segment = response[start:end].strip()
        steps[match.group(1)] = segment

    return_response = response_before_final_answer
    return final_answer, steps, return_response

def setup_log(args):    
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(args.output_path, "output_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 
    # log.debug(f"#########{args.name}############")
    return log



def extract_keystep(llm_response, contribution_scores=None):
    """Return the text of the reasoning step with the highest average contribution score."""
    if contribution_scores is None:
        return ""
    step_avg_cons = []
    for step, inner_dict in contribution_scores.items():
        con_list = [v for v in inner_dict.values() if v is not None]
        avg_con = sum(con_list) / len(con_list) if con_list else 0
        step_avg_cons.append(avg_con)
    max_con = max(step_avg_cons)
    step_idx = len(step_avg_cons) - 1 - step_avg_cons[::-1].index(max_con)
    return_text = llm_response.split("\n")[step_idx].strip()
    return_text = return_text.split(f"Step {str(step_idx + 1)}: ")[-1]
    return return_text.strip()


def extract_probing_confidence(response):
    """Parse a numeric confidence value from a model's self-probing response."""
    match = re.search(r"(\d+(\.\d+)?)%", response)
    if match:
        return float(match.group(1)) / 100

    match = re.search(r"confidence[:\s]*`*\s*(\d+(\.\d+)?)`*", response, re.IGNORECASE)
    if match:
        confidence = float(match.group(1))
        if confidence > 1:
            confidence /= 100
        return confidence

    first_line = response.strip().split("\n")[0]
    try:
        confidence = float(first_line)
        if confidence > 1:
            confidence /= 100
        return confidence
    except ValueError:
        pass

    match = re.search(r"(\d+(\.\d+)?)", first_line)
    if match:
        confidence = float(match.group(1))
        if confidence > 1:
            confidence /= 100
        return confidence

    return None
