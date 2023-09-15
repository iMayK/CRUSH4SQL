import re
import pandas as pd

import torch

from utils.openai_main import get_openai_embedding, get_hallucinated_segments

from utils.score import get_scored_docs
from utils.greedy_selection import greedy_select

from utils.sql_utils import generate_sql

def prepare_correct_txt_sql_pairs(correct_txt_sql_pairs):
    sheet_id = '1PTiGJcXDntJNPVjkFRdgSersW4HqkICQtDc7zezr6w8'

    df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")

    for index, row in df.iterrows():
        question = row['question']
        if question not in correct_txt_sql_pairs:
            if row['final_remarks'] == "all_good":
                sql = row['additional_SQL_feedback']
            elif row['isNonExecutable'] == "yes":
                sql = row['additional_SQL_feedback']
            elif row['isSqlWrongUnsure'] == "yes" and row['additional_SQL_feedback'] != "NA":
                sql = row['additional_SQL_feedback']
            elif row['isIncompleteRetrieval'] == "yes" and row['additional_SQL_feedback'] != "NA":
                sql = row['additional_SQL_feedback']
            else:
                continue
            correct_txt_sql_pairs[question] = {
                'sql': sql
            }
    return correct_txt_sql_pairs

def extract_items(segment):
    # Check if the string matches the pattern word1(word2, word3)
    pattern = r'(\w+)\(([\w\s,]+)\)'
    match = re.match(pattern, segment)

    if match:
        word1 = match.group(1).replace(' ', '_')
        words = match.group(2).split(', ')
        return [f"{word1}.{word.replace(' ', '_')}" for word in words]
    else:
        return None

def get_relevant_fewshot_examples(question, correct_txt_sql_pairs, api_type, api_key, endpoint, topk=5):
    for item in correct_txt_sql_pairs:
        if 'embedding' not in correct_txt_sql_pairs[item]:
            correct_txt_sql_pairs[item]['embedding'] = get_openai_embedding(item, api_type, api_key, endpoint)

    question_embedding = get_openai_embedding(question, api_type, api_key, endpoint)
    
    list_items = list(correct_txt_sql_pairs.keys())
    similarity_scores = torch.nn.functional.cosine_similarity(
        question_embedding.unsqueeze(0),
        torch.stack([correct_txt_sql_pairs[item]['embedding'] for item in list_items]),
        dim=-1
    )

    _, sorted_indices = torch.sort(similarity_scores, descending=True)
    topk_pairs = []
    for idx in sorted_indices[:topk]:
        topk_pairs.append(
            {
                'question': list_items[idx],
                'sql': correct_txt_sql_pairs[list_items[idx]]['sql']
            }
        )
    return topk_pairs

def clean_segments(segments):
    segments = [segment.replace('/', ' ').replace('-', ' ') for segment in segments]
    segments = [schema_item for segment in segments for schema_item in extract_items(segment)]
    segments = [segment for segment in segments if "." in segment]
    return segments

def ndap_pipeline(question, api_type, api_key, endpoint, api_version, correct_txt_sql_pairs):
    decomposition_prompt_used = 'hallucinate_schema_ndap' 

    hallucinated_schema = get_hallucinated_segments(decomposition_prompt_used, question, api_type, api_key, endpoint, api_version)

    segments = clean_segments(hallucinated_schema)

    scored_docs = get_scored_docs(question, segments, api_type, api_key, endpoint, api_version)

    greedy_docs = greedy_select(segments, scored_docs, BUDGET=20)

    sql_input = {
        'question': question,
        'docs': greedy_docs,
    }

    prepare_correct_txt_sql_pairs(correct_txt_sql_pairs)

    if len(correct_txt_sql_pairs) > 0:
        prompting_type = 'fewshot'
        fewshot_examples = get_relevant_fewshot_examples(question, correct_txt_sql_pairs, api_type, api_key, endpoint, api_version, topk=3)
    else:
        prompting_type = 'base'
        fewshot_examples = []

    input_prompt, pred_sql, pred_schema = generate_sql(
        sql_input,
        api_type,
        api_key,
        endpoint,
        api_version,
        prompting_type=prompting_type,
        fewshot_examples=fewshot_examples,
    )

    return hallucinated_schema, pred_schema, pred_sql


