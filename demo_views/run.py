import re

from utils.openai_main import get_hallucinated_segments, get_openai_embedding

from utils.score import get_scored_docs
from utils.greedy_selection import greedy_select

from utils.prompts import PROMPTS
from utils.sql_utils import generate_sql

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

print('prompting type: ')
prompting_types = list(PROMPTS.keys())
for idx, prompting_type in enumerate(prompting_types):
    print(f"{idx}: {prompting_type}")
prompting_type = prompting_types[int(input('your input: '))]

api_key = input('AZURE_OPENAI_API_KEY: ')
endpoint = input('AZURE_OPENAI_ENDPOINT: ')

question = input('\n\nquestion: ')
decomposition_prompt_used = 'hallucinate_schema_ndap' 
segments = get_hallucinated_segments(decomposition_prompt_used, question, api_key, endpoint)
segments = [segment.replace('/', ' ').replace('-', ' ') for segment in segments]
segments = [schema_item for segment in segments for schema_item in extract_items(segment)]
segments = [segment for segment in segments if "." in segment]
print('segments: ', segments)

scored_docs = get_scored_docs(question, segments, api_key, endpoint)

greedy_docs = greedy_select(segments, scored_docs, BUDGET=20)

sql_input = {
    'question': question,
    'docs': greedy_docs,
}

prompt, response, schema = generate_sql(sql_input, prompting_type=prompting_type)

print(f'\nSchema:\n {schema}\n\n')
print(response)


