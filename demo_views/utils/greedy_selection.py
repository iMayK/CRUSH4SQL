# %%
import os
import json
import math
from collections import defaultdict

# %%
TOPK = 100
REWARDS = {
    'same_table': 0,
    'same_db': 0,
    'diff_db': 0,
}
USE_eij_MODIFIED = True
APPLY_MEAN_ENTROPY = True
NORMALIZE_cos = True

file_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(file_dir, 'relation_map_for_unclean.json')
RELATION_MAP = json.load(open(json_file_path))

# %%
def get_entropy(segment, cos):
    q_distri = cos[segment]
    q_distri = [(score+1)/2 for score in q_distri] 
    sum_q_distri = sum(q_distri)
    q_distri = [score/sum_q_distri for score in q_distri]

    entropy = -sum([p*math.log(p) for p in q_distri])  

    return entropy

def get_snm(segment, n, n_idx, cos, selected, mean_entropy):
    if n in selected:
        return 0
    entropy = get_entropy(segment, cos)

    if APPLY_MEAN_ENTROPY:
        inv_entropy = mean_entropy - entropy
    else:
        inv_entropy = -entropy

    weight = 1/(1+math.exp(-inv_entropy))               

    if NORMALIZE_cos:
        return weight * (cos[segment][n_idx] + 1)/2
    else:
        return weight * cos[segment][n_idx]

def get_eij(n, selected):
    if n in selected:
        return 0
    adj_score = 0
    for item in selected:
        if RELATION_MAP[item]['code'] == RELATION_MAP[n]['code']:
            adj_score += REWARDS['same_table']
        elif RELATION_MAP[item]['source'] == RELATION_MAP[n]['source']:
            adj_score += REWARDS['same_db'] 
        else:
            adj_score += REWARDS['diff_db']
    return adj_score

def get_eij_modified(n, selected):
    if n in selected:
        return 0
    adj_score = 0
    same_table_count, same_db_count = 0, 0
    for item in selected:
        if RELATION_MAP[item]['code'] == RELATION_MAP[n]['code']:
            same_table_count += 1
        elif RELATION_MAP[item]['source'] == RELATION_MAP[n]['source']:
            same_db_count += 1
        else:
            pass
    adj_score = REWARDS['same_table']*(math.log(same_table_count+1)) + REWARDS['same_db']*(math.log(same_db_count+1))
    return adj_score

def get_complete_score(segment, n, n_idx, cos, selected, mean_entropy):
    snm = get_snm(segment, n, n_idx, cos, selected, mean_entropy)
    if USE_eij_MODIFIED:
        eij = get_eij_modified(n, selected)
    else:
        eij = get_eij(n, selected)
    return (snm + eij, n)

#%%
def get_data(docs, segments):
    cos = defaultdict(list)
    schema_items = []
    for doc in docs[:TOPK]:
        schema_items.append(doc['doc_name'])
        for segment in segments:
            cos[segment].append(doc[segment])
    return cos, schema_items

# %%
def greedy_select(segments, docs, BUDGET):
    '''
        segments: list of segments
        docs: sorted list of dicts, where each dict is of the form 
              {
                'name': ..., 
                'score': ...,           # final score
                'segment1': ...,        # score for segment1
                'segment2': ...,        # score for segment2
                ...
                }
    '''
    cos, schema_items = get_data(docs, segments)

    list_entropies = []
    for segment in segments:
        list_entropies.append(get_entropy(segment, cos))
    mean_entropy = sum(list_entropies)/len(list_entropies)

    M = len(segments)
    covered = [0 for _ in range(M)]
    selected = set()
    while len(selected) < BUDGET:
        if sum(covered) == M:                         # reset covered if BUDGET is not reached and all segments are covered
            covered = [0 for _ in range(M)]

        lst = []
        for i, segment in enumerate(segments):
            if covered[i] == 1:
                continue

            lst_score = []              # list of (score, n) tuples
            for n_idx, n in enumerate(schema_items):
                lst_score.append(
                    get_complete_score(segment, n, n_idx, cos, selected, mean_entropy)
                )

            s_dash, n_dash = max(lst_score, key=lambda x: x[0])
            best_n = (s_dash, n_dash)

            lst.append((i, best_n))
        i_dash, (s_dash, n_dash) = max(lst, key=lambda x: x[1][0])
        covered[i_dash] = 1
        selected.add(n_dash)
    return list(selected)

def extract_predicted_tables_uncleaned(selected_lst, gold_tables, gold_codes):
    table_codes = set() 
    for item in selected_lst:
        table_codes.add(RELATION_MAP[item]['code'])
    return list(table_codes)

def extract_predicted_tables(selected_lst, filename, gold_tables, gold_codes):
    if "unclean" in filename:
        return extract_predicted_tables_uncleaned(selected_lst, gold_tables, gold_codes)
    table_codes = set()
    for item in selected_lst:
        table_code = item.split('.')[1][-4:]
        table_codes.add(table_code)
    return list(table_codes)



