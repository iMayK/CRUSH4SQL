from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

device = 'cuda'

# For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit")
model = AutoModel.from_pretrained("Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit")
## Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
model.eval()
model.to(device)

SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]


def tokenize_with_specb(texts, is_query):
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True)   
    # Add special brackets & pay attention to them
    for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
        if is_query:
            seq.insert(0, SPECB_QUE_BOS)
            seq.append(SPECB_QUE_EOS)
        else:
            seq.insert(0, SPECB_DOC_BOS)
            seq.append(SPECB_DOC_EOS)
        att.insert(0, 1)
        att.append(1)
    # Add padding
    batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens

def find_last_idx_of_keyword(lst, keyword):
    idx = len(lst) - 1
    while idx >= 0 and lst[idx] != keyword:
        idx -= 1
    if idx >= 0:
        return idx + 1
    else:
        return -1

def get_segment_embedding(i, embeddings, batch_i_token_ids, batch_i_token_masks):
    start_idx = find_last_idx_of_keyword(batch_i_token_ids, 25)
    end_idx = find_last_idx_of_keyword(batch_i_token_ids, 60) - 1  # fix for the special bractet attached in ASYM CASE

    if end_idx == -2:
        end_idx = len(batch_i_token_ids)
    if end_idx - start_idx == 1:
        segment_embedding = embeddings[i][start_idx]
    else:
        sum_embedding = torch.sum(
            embeddings[i][start_idx:end_idx],
            dim=0
        )
        sum_mask = torch.sum(
            torch.arange(
                start=start_idx+1,
                end=end_idx+1
            ).view(-1, 1).repeat(1, sum_embedding.shape[0]),
            dim=0
        ).to(sum_embedding.device)
        segment_embedding = sum_embedding / sum_mask
    return segment_embedding

def get_embedding(batch_tokens):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings

def get_token_embedding(batch_tokens):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(
            **batch_tokens,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    return last_hidden_state*weights

def main():
    text = "question: {} segment: {}"
    texts = [
        text.format("What is the name of the person who is the author of the book The C Programming Language", "author of the book The C Programming Language"),
        text.format("Explain the difference between a stack and a queue.", "difference between a stack and a queue")
    ]

    batch_tokens = tokenize_with_specb(
        texts,
        is_query=True
    ).to(device)

    print(batch_tokens)


if __name__ == "__main__":
    main()