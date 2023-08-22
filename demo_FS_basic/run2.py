from utils.openai_main import get_hallucinated_segments

from utils.score import get_scored_docs
from utils.greedy_selection import greedy_select

from utils.sql_utils import generate_sql

from utils.utils import extract_items, get_relevant_fewshot_examples

api_key = input('AZURE_OPENAI_API_KEY: ')
endpoint = input('AZURE_OPENAI_ENDPOINT: ')
correct_txt_sql_pairs = []

while 1:
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

    if len(correct_txt_sql_pairs) > 0:
        prompting_type = 'fewshot'
        fewshot_examples = get_relevant_fewshot_examples(question, correct_txt_sql_pairs, api_key, endpoint, topk=3)
    else:
        prompting_type = 'base'
        fewshot_examples = []

    prompt, response, schema = generate_sql(sql_input, prompting_type=prompting_type, fewshot_examples=fewshot_examples)

    print(f'\nprompt:\n {prompt}\n\n')
    print(response)

    check = input('\n\nGenerated SQL maybe incorrect. Want to register correct SQL? Press "y" to `proceed`, "n" to `skip`, `q` to `quit`: ')
    if check == 'y':
        correct_sql = input('Enter correct SQL: ')
        correct_txt_sql_pairs.append(
            {
                'question': question,
                'sql': correct_sql,
            }
        )
    elif check == 'q':
        exit()

###
'''
    here, we will collect few shot examples on the go and use them in a fewshot prompt style for future questions
'''