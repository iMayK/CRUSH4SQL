from utils.utils import ndap_pipeline

api_key = input('AZURE_OPENAI_API_KEY: ')
endpoint = input('AZURE_OPENAI_ENDPOINT: ')

correct_txt_sql_pairs = []

question = input('\nEnter question: ')
ndap_pipeline(question, api_key, endpoint, correct_txt_sql_pairs)

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
   more abstract code for easy use
'''