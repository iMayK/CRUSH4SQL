from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

feedback_options = [
    'Retrieval is wrong',
    'Retrieval is fine, but some other relevant table/columns are missing',
    'Retrieval is correct, but generated SQL is wrong',
    'Generated SQL seems correct, execution error',
    'Everything looks fine!'
]

columns = [
    'question',
    'isWrongRetrieval',
    'wrong_retrieval_feedback',
    'isIncompleteRetrieval',
    'additional_table_feedback',
    'isSqlWrongUnsure',
    'additional_SQL_feedback',
    'isNonExecutable',
    'final_remarks'
]

def get_feedback(question, predicted_sql):
    new_row = [question]
    print('Feedback options:')
    for idx, option in enumerate(feedback_options):
        print(f'{idx+1}: {option}')
    choosen_option = int(input(f'\nYour input: (press any number from 1 .. {len(feedback_options)}): '))

    if choosen_option == 1:
        response = input('\nCan you provide correct table(s)? Press `y` for yes, `n` for no: ')
        correct_tables = "NA"
        if response == 'y':
            correct_tables = input('\nPlease enter correct table names/codes:\n')
        new_row.append(["yes", correct_tables, "NA", 'NA', "NA", 'NA', "NA", 'NA'])
        print('\nThank you for the feedback!')

    elif choosen_option == 2:
        new_row.extend(["no", "NA"]) # skipping 1st option details

        response = input('\nCan you provide additional correct table(s)? Press `y` for yes, `n` for no: ')
        additional_tables = "NA"
        if response == 'y':
            additional_tables = input('\nPlease enter additional relevant table names/codes (1 in each line):\n')
        new_row.extend(["yes", additional_tables])

        response = int(input('\nIs the SQL correct based on the retrieval? (press `1` for yes, `2` for no, `3` for not sure): '))
        correct_sql = "NA"
        if response in [2, 3]:
            response = input('\nWant to register correct SQL? Press `y` for yes, `n` for no: ')
            if response == 'y':
                correct_sql = input('\nPlease enter correct SQL:\n')
            new_row.extend(["yes", correct_sql])
        else:
            new_row.extend(["no", predicted_sql])

        new_row.extend(["NA", "NA"])
        print('\nThank you for the feedback!')

    elif choosen_option == 3:
        new_row.extend(["no", "NA", "no", "NA"]) # skipping initial 2 options

        response = input('\nWant to register correct SQL? Press `y` for yes, `n` for no: ')
        correct_sql = "NA"
        if response == 'y':
            correct_sql = input('\nPlease enter correct SQL:\n')
        new_row.extend(["yes", correct_sql])

        new_row.extend(["yes", "NA"])
        print('\nThank you for the feedback!')
    
    elif choosen_option == 4:
        new_row.extend(["no", "NA", "no", "NA", "no", predicted_sql, "yes", "NA"])
        print('\nThank you for the feedback!')

    elif choosen_option == 5:
        new_row.extend(["no", "NA", "no", "NA", "no", predicted_sql, "no", "all_good"])
        print('\nThank you for the feedback!')

    return new_row

def submit_response(new_row):
    worksheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1PTiGJcXDntJNPVjkFRdgSersW4HqkICQtDc7zezr6w8/edit#gid=0').sheet1

    # get_all_values gives a list of rows.
    rows = worksheet.get_all_values()

    # Find the last row with data
    last_row = len(rows)

    # Append the feedbacks to the first column
    for i, feedback in enumerate(new_row):
        worksheet.update_cell(last_row + 1, i + 1, feedback)

def feedback_pipeline(question, predicted_sql):
    feedback_data = get_feedback(question, predicted_sql)
    submit_response(feedback_data)
