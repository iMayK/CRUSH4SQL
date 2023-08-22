import re

import torch

from utils.openai_main import get_openai_embedding

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

def get_relevant_fewshot_examples(question, correct_txt_sql_pairs, api_key, endpoint, topk=5):
    for item in correct_txt_sql_pairs:
        if 'embedding' not in item:
            item['embedding'] = get_openai_embedding(item['question'], api_key, endpoint)

    question_embedding = get_openai_embedding(question, api_key, endpoint)
    
    similarity_scores = torch.nn.functional.cosine_similarity(
        question_embedding.unsqueeze(0),
        torch.stack([item['embedding'] for item in correct_txt_sql_pairs]),
        dim=-1
    )
    _, sorted_indices = torch.sort(similarity_scores, descending=True)
    topk_pairs = []
    for idx in sorted_indices[:topk]:
        topk_pairs.append(
            {
                'question': correct_txt_sql_pairs[idx]['question'],
                'sql': correct_txt_sql_pairs[idx]['sql']
            }
        )
    return topk_pairs