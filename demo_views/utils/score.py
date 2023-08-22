import os
import re
import pickle

from utils.openai_main import get_openai_embedding
#from utils.sgpt_utils import tokenize_with_specb, get_token_embedding, get_segment_embedding 

import torch

#device = 'cuda'

def get_contextual_emb(question, segments):
    all_text = []
    temp = "question: {} segment: {}"
    for segment in segments:
        all_text.append(temp.format(question, segment))

    batch_tokens = tokenize_with_specb(
        all_text,
        is_query=True
    ).to(device)

    embeddings = get_token_embedding(batch_tokens)

    seg_embs = []
    for i in range(len(batch_tokens['input_ids'])):
        segment_emb = get_segment_embedding(
            i,
            embeddings,
            batch_tokens['input_ids'][i].tolist(),
            batch_tokens['attention_mask'][i].tolist()
        )
        seg_embs.append(segment_emb)

    seg_embs = torch.stack(seg_embs)

    return seg_embs.to(device)

def ranking(A, B, list_schema_elements, segments, aggr_type="max"):
    Z = torch.nn.functional.cosine_similarity(
        A.unsqueeze(1),
        B.unsqueeze(0),
        dim=-1
    )                                               # no_of_segments x m

    if aggr_type == "max":
        Z_aggr, _ = torch.max(Z, dim=0)                  # [m]
    else:
        Z_aggr = torch.sum(Z, dim=0)                     # [m]
        
    docs_score, docs_idx = torch.sort(Z_aggr, descending=True)
    docs_score = docs_score.tolist()
    docs_idx = docs_idx.tolist()

    docs = []
    for idx, score in zip(docs_idx, docs_score):
        temp = {}
        temp['doc_name'] = list_schema_elements[idx]
        temp['score'] = score
        segment_scores = Z[:, idx].tolist()
        for segment, segment_score in zip(segments, segment_scores):
            temp[segment] = segment_score
        docs.append(temp)
    
    return docs

def get_scored_docs(
    question,
    segments,
    api_key,
    endpoint,
    aggr_type='max'
):
    A = []
    for segment in segments:
        A.append(get_openai_embedding(segment, api_key, endpoint))
    #A = torch.stack(A).to(device)                                         # num_of_segments x d                                                      
    A = torch.stack(A)                                                     # num_of_segments x d                                                      
        
    #A = get_openai_embedding(segments, api_key, endpoint)                 # num_of_segments x d -> WRONG
    #A = get_contextual_emb(question, segments)                            # num_of_segments x d

    file_dir = os.path.dirname(os.path.realpath(__file__))

    doc_file_path = os.path.join(file_dir, 'ndap_super_flat_unclean.txt')
    with open(doc_file_path, 'r') as fp:
        list_schema_elements = fp.readlines()
    list_schema_elements = [item.strip() for item in list_schema_elements]
    #print(len(list_schema_elements), list_schema_elements[:4])

    embedding_file_path = os.path.join(file_dir, 'openai_docs_unclean_embedding.pickle')
    with open(embedding_file_path, 'rb') as doc_pkl:
        #docs_embedding = pickle.load(doc_pkl).to(device)
        docs_embedding = pickle.load(doc_pkl)
        #print(docs_embedding.shape)

    docs = ranking(A, docs_embedding, list_schema_elements, segments, aggr_type=aggr_type)

    return docs[:100]




