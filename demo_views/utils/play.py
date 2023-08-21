#%%
import sys

import re
import json
from tqdm import tqdm

import torch
import pickle

sys.path.append('/raid/infolab/mayankk/RA/june')

from utils.openai_main import demand_segments # from root of june workdir
from utils.openai_main import demand_embedding

#%%
def extract_items(segment):
    # Check if the string matches the pattern word1(word2, word3)
    pattern = r'(\w+)\(([\w\s,]+)\)'
    match = re.match(pattern, segment)

    if match:
        word1 = match.group(1).replace(' ', '_')
        words = match.group(2).split(', ')
        return [word1] + [f"{word1}.{word.replace(' ', '_')}" for word in words]
    else:
        return None

#%%
filepath = "/raid/infolab/mayankk/RA/june/data/ndap/dump/"
filename = "finance_queries_v2.json"
data = json.load(open(filepath + filename))

lst_questions = []
lst_embeddings = []
for item in tqdm(data):
    question = item['question']
    print('question: ', question)

    decomposition_prompt_used = 'hallucinate_schema_ndap' 
    segments = demand_segments(decomposition_prompt_used, question)
    segments = [segment.replace('/', ' ').replace('-', ' ') for segment in segments]
    segments = [schema_item for segment in segments for schema_item in extract_items(segment)]
    segments = [segment for segment in segments if "." in segment]
    item['question_decomposition_hallucinate_schema_ndap'] = segments

    lst_questions.append(question)
    lst_embeddings.append(demand_embedding(question))

print(lst_questions[:2])
print(data[:2])

#%%
embeddings = torch.stack(lst_embeddings)
print(embeddings.shape)

#%%
with open('finance_queries_v2_flat_text.txt', 'w') as f:
    f.write('\n'.join(lst_questions))

with open('finance_queries_v2_flat_text_embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f)

with open(filepath + filename, 'w') as f:
    json.dump(data, f, indent=4)


############################################################################################################
#%%
import openai

import sys
sys.path.append('/raid/infolab/mayankk/RA/june')

import utils.openai_main as openai_main 

#%%
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Your task is to create a syntactically correct SQL query based on an input question. Make sure to use only the column names specified in the schema description and avoid querying for columns that do not exist. Also, pay attention to which column belongs to which table."
    },
    {
        "role": "assistant",
        "content": "Sure, I'd be happy to!"
    },
    {
        "role": "user",
        "content": "Please follow these instructions carefully:\n First, I will provide you with an example that includes a database schema, a question based on it, and a detailed step-by-step approach on how to construct the SQL query for the question. Pay close attention to the example.\n After the example, I will give you a new question along with its corresponding schema. You will need to write an SQL query for it using the strategy described in the example."
    },
    {
        "role": "assistant",
        "content": "Sure. Please provide the example with the schema information, question, and detailed solution, and I'll assist you accordingly."
    },
    {
        "role": "user",
        "content": "Again, be attentive to the instructions mentioned previously.\n I am providing you the Schema information for the example followed by the corresponding question and the detailed solution.\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Country' | 'Imports or exports' | 'SourceYear' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'YearCode' | 'Year' | 'Commodity type'\n 'India' | 'Exports' | '2009-10' | '2973.19' | '2881.22' | '2009' | 'Financial Year (Apr - Mar), 2009' | 'Other cereals'\n 'India' | 'Exports' | '2017-18' | '4913.28' | '714.0' | '2017' | 'Financial Year (Apr - Mar), 2017' | 'Fresh fruits'\n 'India' | 'Imports' | '2012-13' | '3.59' | 'nan' | '2012' | 'Financial Year (Apr - Mar), 2012' | 'Percentage Share of Agricultural Imports in National Imports'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Exports in rupees' | 'Country' | 'Commodity name' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n '165' | 'India' | 'Carpet' | 'Italy' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '120' | 'India' | 'Tea' | 'Iraq' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '66' | 'India' | 'Cashew' | 'France' | '2019' | 'Financial Year (Apr - Mar), 2019'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Exports in crores' | 'Exports in US millions' | 'Country' | 'YearCode' | 'Year'\n '203571' | '44560' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '2219854' | '313361' | 'India' | '2019' | 'Financial Year (Apr - Mar), 2019'\n '2159043' | '291808' | 'India' | '2020' | 'Financial Year (Apr - Mar), 2020'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Sourceyear' | 'Exports in usd' | 'Exports in rupees' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Tobacco' | '1996-97' | '4.2' | '30' | 'India' | 'U.S.A' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Carpets' | '2012-13' | '91.0' | '700' | 'India' | 'U.K.' | '2012' | 'Financial Year (Apr - Mar), 2012'\n 'Rice' | '1996-97' | '33.6' | '220' | 'India' | 'Kuwait' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n\n Question: 'Year wise list of India's export destination countries.'\n\n Solution:\n Step 1: Firstly, let's analyze the tables in the schema.\n\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n This table contains information about India\'s imports and exports of principal agricultural commodities. However, the question does not specifically ask for agricultural commodities. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. Further, columns such as 'Imports or exports', 'Quantity of imports or exports of principal agricultural commodities', 'Value of imports or exports of principal agricultural commodities', etc. do not mention any information about the destination country. Therefore, for the given question, which is about India\'s export destination countries, this table is not directly useful.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n This table focuses on the export of select commodities from India to different countries and is potentially relevant to the question. Let\'s look into the table content. The column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in rupees. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. The column 'Commodity name' contains 'Carpet', 'Tea', etc., so this column indicates the type of commodity exported. Column 'Export of selected new item commodities to principal countries.' contains values like 'Italy', 'Iraq', etc. which are basically different countries, so this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. This table can provide insights into India's export destination countries and can be utilized to answer the question effectively.\n\n Table: 'Selected Economic and Social Indicator 6159'\n This table contains selected economic and social indicators for India. It is difficult to determine just from the name of the table whether it is relevant or not. Let's analyze the content of the table to get better understanding. The column 'Exports in crores' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in crores. Similarly, column 'Exports in US millions' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US millions. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. There are no other columns in the table which have information about other countries. Thus, this table may not be directly useful for answering the question.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n This table contains information about exports of select commodities from India to principal countries based on an old commodity classification. This table could be relevant to the question. Let's look into the table content. The 'Commodity name' column indicates the type of commodity being exported, such as 'Tobacco,' 'Carpets,' and 'Rice.' Based on the values in the column 'SourceYear', this column contains information about the year for which the data is available. Column 'Exports in usd' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US dollars. Similarly, the column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in Indian rupees. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. Column 'Export to country' contains values like 'U.S.A', 'Kuwait', etc. indicating different countries. So this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. Based on the above analysis, I think this table is relevant. Let's move to the next table.\n\n\n Step 2: Selecting relevant tables:\n\n Based on the analysis, it appears that the second table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (New commodity classification as per 2009-10) 7455') and the fourth table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (Old commodity classification) 7454') are relevant for determining India's export destination countries.\n\n\n Step 3: Determining relevant schema:\n\n Based on the question and information from the relevant tables obtained in step 2, this is the relevant schema:\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'('Exports in rupees' , 'Commodity name' , 'Export of selected new item commodities to principal countries.' , 'Yearcode' )\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'('Commodity name' , 'Exports in usd' , 'Exports in rupees' , 'Export to country' , 'Yearcode' ).\n Ensure that each table in the relevant schema refer only to columns that were present in its original description at the start.\n\n\n Step 4: Finally, based on the question and the relevant schema, we can write the SQL as:\n\n WITH export_data AS ( SELECT 'Yearcode', 'Commodity name', 'Export of selected new item commodities to principal countries.' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455' GROUP BY 'Year', 'Commodity name', destination_country UNION ALL SELECT 'Yearcode', 'Commodity name', 'Export to country' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454' GROUP BY 'Year', 'Commodity name', destination_country ) SELECT 'Year', destination_country, sum(total_exports) FROM export_data GROUP BY 'Year', 'destination_country'"
    },
    {
        "role": "assistant",
        "content": "Thank you for providing the detailed solution. Based on the example, I understand the strategy to be used for writing the SQL query. Please go ahead and provide the next question along with the corresponding schema, and I'll assist you in writing the SQL query using the same strategy."
    },
    {
        "role": "user",
        "content": "Now, write SQL for the question mentioned below based on the corresponding schema information using the same strategy mentioned in the example previously.\n\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Imports or exports' | 'Commodity type' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'Country' | 'YearCode' | 'Year'\n 'Exports' | 'Oil meals' | '7831.79' | '4671.14' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n 'Imports' | 'Fresh vegetables' | '11.12' | '8.55' | 'India' | '2016' | 'Financial Year (Apr - Mar), 2016'\n 'Imports' | 'Tobacco unmanufactured' | '69.47' | '1.54' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Exports in usd' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Rice' | '31.0' | 'India' | 'U.S.A' | '2002' | 'Financial Year (Apr - Mar), 2002'\n 'Tea' | '75.3' | 'India' | 'Others' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Iron Ore' | '0.0' | 'India' | 'Chile' | '2008' | 'Financial Year (Apr - Mar), 2008'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( New commodity classification as per 2009-10 ) 7453'\n 'Imports in usd' | 'Imports of principal commodities' | 'Imports in inr' | 'Country' | 'Yearcode' | 'Year'\n '14977.5' | 'Iron & Steel' | '97960' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '241.3' | 'Sulphur & Unroasted Iron Pyrts' | '1099' | 'India' | '2010' | 'Financial Year (Apr - Mar), 2010'\n '22732.5' | 'Transport equipment' | '146509' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452'\n 'Imports of principal commodities' | 'Sourceyear' | 'Imports in rupees' | 'Country' | 'Year' | 'Imports in usd' | 'Yearcode'\n 'Crude Rubber, including Synthetic and Reclaimed' | '1991-92 ' | '182.0' | 'India' | 'Financial Year (Apr - Mar), 1991' | '73.8' | '1991'\n 'Organic and Inorganic Chemicals' | '2012-13 ' | '105141.0' | 'India' | 'Financial Year (Apr - Mar), 2012' | '19323.9' | '2012'\n 'Capital Goods' | '1988-89 ' | '6956.0' | 'India' | 'Financial Year (Apr - Mar), 1988' | '4803.0' | '1988'\n\n Table: 'Imports and Exports of Principal Agricultural Commodities of India 6129'\n 'Imports of agricultural commodities' | 'Percentage of agricultural imports to total national imports' | 'Agricultural exports to total national imports' | 'Country' | 'Yearcode' | 'Year'\n '54365.29' | '3.99' | '9.99' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n '5890.1' | '4.8' | '19.18' | 'India' | '1995' | 'Financial Year (Apr - Mar), 1995'\n '6612.6' | '4.76' | '20.33' | 'India' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Imports in US millions' | 'Country' | 'YearCode' | 'Year'\n '381008' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '50536' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '15869' | 'India' | '1980' | 'Financial Year (Apr - Mar), 1980'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Commodity name' | 'Country' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n 'Rice' | 'India' | 'France' | '2015' | 'Financial Year (Apr - Mar), 2015'\n 'Cashew' | 'India' | 'U.S.A' | '2013' | 'Financial Year (Apr - Mar), 2013'\n 'Cotton Yarn/Fabs./made-ups, Handloom Products etc.' | 'India' | 'Mauritius' | '2018' | 'Financial Year (Apr - Mar), 2018'\n\n Question: Which commodity had the highest imports in Old Commodity Classification?\n SQL Query:{} \n\n\nKeep note of all the instructions mentioned previously and use the same strategy as mentioned in the example for this question as well."
    },
]

for item in messages:
    print(item['role'])
    print(item['content'])
    print()


# %%
def generate():
    deployment_name = 'chatgpt-35-16k'
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Your task is to create a syntactically correct SQL query based on an input question. Make sure to use only the column names specified in the schema description and avoid querying for columns that do not exist. Also, pay attention to which column belongs to which table."
        },
        {
            "role": "assistant",
            "content": "Sure, I'd be happy to!"
        },
        {
            "role": "user",
            "content": "Please follow these instructions carefully:\n First, I will provide you with an example that includes a database schema, a question based on it, and a detailed step-by-step approach on how to construct the SQL query for the question. Pay close attention to the example.\n After the example, I will give you a new question along with its corresponding schema. You will need to write an SQL query for it using the strategy described in the example."
        },
        {
            "role": "assistant",
            "content": "Sure. Please provide the example with the schema information, question, and detailed solution, and I'll assist you accordingly."
        },
        {
            "role": "user",
            "content": "Again, be attentive to the instructions mentioned previously.\n I am providing you the Schema information for the example followed by the corresponding question and the detailed solution.\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Country' | 'Imports or exports' | 'SourceYear' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'YearCode' | 'Year' | 'Commodity type'\n 'India' | 'Exports' | '2009-10' | '2973.19' | '2881.22' | '2009' | 'Financial Year (Apr - Mar), 2009' | 'Other cereals'\n 'India' | 'Exports' | '2017-18' | '4913.28' | '714.0' | '2017' | 'Financial Year (Apr - Mar), 2017' | 'Fresh fruits'\n 'India' | 'Imports' | '2012-13' | '3.59' | 'nan' | '2012' | 'Financial Year (Apr - Mar), 2012' | 'Percentage Share of Agricultural Imports in National Imports'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Exports in rupees' | 'Country' | 'Commodity name' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n '165' | 'India' | 'Carpet' | 'Italy' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '120' | 'India' | 'Tea' | 'Iraq' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '66' | 'India' | 'Cashew' | 'France' | '2019' | 'Financial Year (Apr - Mar), 2019'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Exports in crores' | 'Exports in US millions' | 'Country' | 'YearCode' | 'Year'\n '203571' | '44560' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '2219854' | '313361' | 'India' | '2019' | 'Financial Year (Apr - Mar), 2019'\n '2159043' | '291808' | 'India' | '2020' | 'Financial Year (Apr - Mar), 2020'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Sourceyear' | 'Exports in usd' | 'Exports in rupees' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Tobacco' | '1996-97' | '4.2' | '30' | 'India' | 'U.S.A' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Carpets' | '2012-13' | '91.0' | '700' | 'India' | 'U.K.' | '2012' | 'Financial Year (Apr - Mar), 2012'\n 'Rice' | '1996-97' | '33.6' | '220' | 'India' | 'Kuwait' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n\n Question: 'Year wise list of India's export destination countries.'\n\n Solution:\n Step 1: Firstly, let's analyze the tables in the schema.\n\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n This table contains information about India\'s imports and exports of principal agricultural commodities. However, the question does not specifically ask for agricultural commodities. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. Further, columns such as 'Imports or exports', 'Quantity of imports or exports of principal agricultural commodities', 'Value of imports or exports of principal agricultural commodities', etc. do not mention any information about the destination country. Therefore, for the given question, which is about India\'s export destination countries, this table is not directly useful.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n This table focuses on the export of select commodities from India to different countries and is potentially relevant to the question. Let\'s look into the table content. The column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in rupees. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. The column 'Commodity name' contains 'Carpet', 'Tea', etc., so this column indicates the type of commodity exported. Column 'Export of selected new item commodities to principal countries.' contains values like 'Italy', 'Iraq', etc. which are basically different countries, so this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. This table can provide insights into India's export destination countries and can be utilized to answer the question effectively.\n\n Table: 'Selected Economic and Social Indicator 6159'\n This table contains selected economic and social indicators for India. It is difficult to determine just from the name of the table whether it is relevant or not. Let's analyze the content of the table to get better understanding. The column 'Exports in crores' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in crores. Similarly, column 'Exports in US millions' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US millions. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. There are no other columns in the table which have information about other countries. Thus, this table may not be directly useful for answering the question.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n This table contains information about exports of select commodities from India to principal countries based on an old commodity classification. This table could be relevant to the question. Let's look into the table content. The 'Commodity name' column indicates the type of commodity being exported, such as 'Tobacco,' 'Carpets,' and 'Rice.' Based on the values in the column 'SourceYear', this column contains information about the year for which the data is available. Column 'Exports in usd' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US dollars. Similarly, the column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in Indian rupees. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. Column 'Export to country' contains values like 'U.S.A', 'Kuwait', etc. indicating different countries. So this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. Based on the above analysis, I think this table is relevant. Let's move to the next table.\n\n\n Step 2: Selecting relevant tables:\n\n Based on the analysis, it appears that the second table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (New commodity classification as per 2009-10) 7455') and the fourth table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (Old commodity classification) 7454') are relevant for determining India's export destination countries.\n\n\n Step 3: Determining relevant schema:\n\n Based on the question and information from the relevant tables obtained in step 2, this is the relevant schema:\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'('Exports in rupees' , 'Commodity name' , 'Export of selected new item commodities to principal countries.' , 'Yearcode' )\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'('Commodity name' , 'Exports in usd' , 'Exports in rupees' , 'Export to country' , 'Yearcode' ).\n Ensure that each table in the relevant schema refer only to columns that were present in its original description at the start.\n\n\n Step 4: Finally, based on the question and the relevant schema, we can write the SQL as:\n\n WITH export_data AS ( SELECT 'Yearcode', 'Commodity name', 'Export of selected new item commodities to principal countries.' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455' GROUP BY 'Year', 'Commodity name', destination_country UNION ALL SELECT 'Yearcode', 'Commodity name', 'Export to country' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454' GROUP BY 'Year', 'Commodity name', destination_country ) SELECT 'Year', destination_country, sum(total_exports) FROM export_data GROUP BY 'Year', 'destination_country'"
        },
        {
            "role": "assistant",
            "content": "Thank you for providing the detailed solution. Based on the example, I understand the strategy to be used for writing the SQL query. Please go ahead and provide the next question along with the corresponding schema, and I'll assist you in writing the SQL query using the same strategy."
        },
        {
            "role": "user",
            "content": "Now, write SQL for the question mentioned below based on the corresponding schema information using the same strategy mentioned in the example previously.\n\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Imports or exports' | 'Commodity type' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'Country' | 'YearCode' | 'Year'\n 'Exports' | 'Oil meals' | '7831.79' | '4671.14' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n 'Imports' | 'Fresh vegetables' | '11.12' | '8.55' | 'India' | '2016' | 'Financial Year (Apr - Mar), 2016'\n 'Imports' | 'Tobacco unmanufactured' | '69.47' | '1.54' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Exports in usd' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Rice' | '31.0' | 'India' | 'U.S.A' | '2002' | 'Financial Year (Apr - Mar), 2002'\n 'Tea' | '75.3' | 'India' | 'Others' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Iron Ore' | '0.0' | 'India' | 'Chile' | '2008' | 'Financial Year (Apr - Mar), 2008'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( New commodity classification as per 2009-10 ) 7453'\n 'Imports in usd' | 'Imports of principal commodities' | 'Imports in inr' | 'Country' | 'Yearcode' | 'Year'\n '14977.5' | 'Iron & Steel' | '97960' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '241.3' | 'Sulphur & Unroasted Iron Pyrts' | '1099' | 'India' | '2010' | 'Financial Year (Apr - Mar), 2010'\n '22732.5' | 'Transport equipment' | '146509' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452'\n 'Imports of principal commodities' | 'Sourceyear' | 'Imports in rupees' | 'Country' | 'Year' | 'Imports in usd' | 'Yearcode'\n 'Crude Rubber, including Synthetic and Reclaimed' | '1991-92 ' | '182.0' | 'India' | 'Financial Year (Apr - Mar), 1991' | '73.8' | '1991'\n 'Organic and Inorganic Chemicals' | '2012-13 ' | '105141.0' | 'India' | 'Financial Year (Apr - Mar), 2012' | '19323.9' | '2012'\n 'Capital Goods' | '1988-89 ' | '6956.0' | 'India' | 'Financial Year (Apr - Mar), 1988' | '4803.0' | '1988'\n\n Table: 'Imports and Exports of Principal Agricultural Commodities of India 6129'\n 'Imports of agricultural commodities' | 'Percentage of agricultural imports to total national imports' | 'Agricultural exports to total national imports' | 'Country' | 'Yearcode' | 'Year'\n '54365.29' | '3.99' | '9.99' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n '5890.1' | '4.8' | '19.18' | 'India' | '1995' | 'Financial Year (Apr - Mar), 1995'\n '6612.6' | '4.76' | '20.33' | 'India' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Imports in US millions' | 'Country' | 'YearCode' | 'Year'\n '381008' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '50536' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '15869' | 'India' | '1980' | 'Financial Year (Apr - Mar), 1980'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Commodity name' | 'Country' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n 'Rice' | 'India' | 'France' | '2015' | 'Financial Year (Apr - Mar), 2015'\n 'Cashew' | 'India' | 'U.S.A' | '2013' | 'Financial Year (Apr - Mar), 2013'\n 'Cotton Yarn/Fabs./made-ups, Handloom Products etc.' | 'India' | 'Mauritius' | '2018' | 'Financial Year (Apr - Mar), 2018'\n\n Question: Which commodity had the highest imports in Old Commodity Classification?\n SQL Query:{} \n\n\nKeep note of all the instructions mentioned previously and use the same strategy as mentioned in the example for this question as well."
        },
    ]

    response = openai.ChatCompletion.create(
        engine = deployment_name,
        messages=messages,
        temperature=0
    )

    return response['choices'][0]['message']['content']

# %%
output = generate()
print(output)

#%%
output

#%%
#{
#    "role": "assistant",
#    "content": "Based on the provided schema and question, we can follow the same strategy as in the example to write the SQL query. Here's the SQL query to find the commodity with the highest imports in the old commodity classification:\n\n```sql\nSELECT 'Commodity name', MAX('Imports in rupees') AS highest_imports\nFROM 'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452'\nGROUP BY 'Commodity name'\nORDER BY highest_imports DESC\nLIMIT 1;\n```\n\nPlease note that you should replace the single quotes around the column names and table names with backticks (`) if your database requires it." 
#},
#{
#    "role": "user",
#    "content": "please specify whether or not the strategy used for writing the SQL query is the same as the one used in the example.\n\nAlso, what was the strategy, show me the steps you took to write the SQL query."
#}


# %%
### 
##note: different prompt at the end of the example which resulted in slightly different output (like it actually answered the question using the strategy)

#{
#    "role": "user",
#    "content": "Now, write SQL for the question mentioned below based on the corresponding schema information using the same strategy mentioned in the example previously.\n\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Imports or exports' | 'Commodity type' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'Country' | 'YearCode' | 'Year'\n 'Exports' | 'Oil meals' | '7831.79' | '4671.14' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n 'Imports' | 'Fresh vegetables' | '11.12' | '8.55' | 'India' | '2016' | 'Financial Year (Apr - Mar), 2016'\n 'Imports' | 'Tobacco unmanufactured' | '69.47' | '1.54' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Exports in usd' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Rice' | '31.0' | 'India' | 'U.S.A' | '2002' | 'Financial Year (Apr - Mar), 2002'\n 'Tea' | '75.3' | 'India' | 'Others' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Iron Ore' | '0.0' | 'India' | 'Chile' | '2008' | 'Financial Year (Apr - Mar), 2008'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( New commodity classification as per 2009-10 ) 7453'\n 'Imports in usd' | 'Imports of principal commodities' | 'Imports in inr' | 'Country' | 'Yearcode' | 'Year'\n '14977.5' | 'Iron & Steel' | '97960' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '241.3' | 'Sulphur & Unroasted Iron Pyrts' | '1099' | 'India' | '2010' | 'Financial Year (Apr - Mar), 2010'\n '22732.5' | 'Transport equipment' | '146509' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452'\n 'Imports of principal commodities' | 'Sourceyear' | 'Imports in rupees' | 'Country' | 'Year' | 'Imports in usd' | 'Yearcode'\n 'Crude Rubber, including Synthetic and Reclaimed' | '1991-92 ' | '182.0' | 'India' | 'Financial Year (Apr - Mar), 1991' | '73.8' | '1991'\n 'Organic and Inorganic Chemicals' | '2012-13 ' | '105141.0' | 'India' | 'Financial Year (Apr - Mar), 2012' | '19323.9' | '2012'\n 'Capital Goods' | '1988-89 ' | '6956.0' | 'India' | 'Financial Year (Apr - Mar), 1988' | '4803.0' | '1988'\n\n Table: 'Imports and Exports of Principal Agricultural Commodities of India 6129'\n 'Imports of agricultural commodities' | 'Percentage of agricultural imports to total national imports' | 'Agricultural exports to total national imports' | 'Country' | 'Yearcode' | 'Year'\n '54365.29' | '3.99' | '9.99' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n '5890.1' | '4.8' | '19.18' | 'India' | '1995' | 'Financial Year (Apr - Mar), 1995'\n '6612.6' | '4.76' | '20.33' | 'India' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Imports in US millions' | 'Country' | 'YearCode' | 'Year'\n '381008' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '50536' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '15869' | 'India' | '1980' | 'Financial Year (Apr - Mar), 1980'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Commodity name' | 'Country' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n 'Rice' | 'India' | 'France' | '2015' | 'Financial Year (Apr - Mar), 2015'\n 'Cashew' | 'India' | 'U.S.A' | '2013' | 'Financial Year (Apr - Mar), 2013'\n 'Cotton Yarn/Fabs./made-ups, Handloom Products etc.' | 'India' | 'Mauritius' | '2018' | 'Financial Year (Apr - Mar), 2018'\n\n Question: Which commodity had the highest imports in Old Commodity Classification?\n SQL Query:{} \n\n\n Also, please specify whether or not the strategy used for writing the SQL query is the same as the one used in the example."
#}

## actual output:
#'Based on the provided schema and the strategy used in the example, here is the SQL query for the question "Which commodity had the highest imports in Old Commodity Classification?":\n\nWITH import_data AS (\n  SELECT \'Commodity name\', SUM(\'Imports in rupees\') AS total_imports\n  FROM \'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452\'\n  GROUP BY \'Commodity name\'\n)\nSELECT \'Commodity name\', total_imports\nFROM import_data\nWHERE total_imports = (\n  SELECT MAX(total_imports)\n  FROM import_data\n);\n\nThe strategy used for writing the SQL query is the same as the one used in the example.'


#%%
def generate():
    deployment_name = 'chatgpt-35-16k'
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Your task is to create a syntactically correct SQL query based on an input question. Make sure to use only the column names specified in the schema description and avoid querying for columns that do not exist. Also, pay attention to which column belongs to which table."
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Sure, I'd be happy to!"
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Please follow these instructions carefully:\n First, I will provide you with an example that includes a database schema, a question based on it, and a detailed step-by-step approach on how to construct the SQL query for the question. Pay close attention to the example.\n After the example, I will give you a new question along with its corresponding schema. You will need to write an SQL query for it using the strategy described in the example."
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Sure. Please provide the example with the schema information, question, and detailed solution, and I'll assist you accordingly."
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Again, be attentive to the instructions mentioned previously.\n I am providing you the Schema information for the example followed by the corresponding question and the detailed solution.\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Country' | 'Imports or exports' | 'SourceYear' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'YearCode' | 'Year' | 'Commodity type'\n 'India' | 'Exports' | '2009-10' | '2973.19' | '2881.22' | '2009' | 'Financial Year (Apr - Mar), 2009' | 'Other cereals'\n 'India' | 'Exports' | '2017-18' | '4913.28' | '714.0' | '2017' | 'Financial Year (Apr - Mar), 2017' | 'Fresh fruits'\n 'India' | 'Imports' | '2012-13' | '3.59' | 'nan' | '2012' | 'Financial Year (Apr - Mar), 2012' | 'Percentage Share of Agricultural Imports in National Imports'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Exports in rupees' | 'Country' | 'Commodity name' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n '165' | 'India' | 'Carpet' | 'Italy' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '120' | 'India' | 'Tea' | 'Iraq' | '2020' | 'Financial Year (Apr - Mar), 2020'\n '66' | 'India' | 'Cashew' | 'France' | '2019' | 'Financial Year (Apr - Mar), 2019'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Exports in crores' | 'Exports in US millions' | 'Country' | 'YearCode' | 'Year'\n '203571' | '44560' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '2219854' | '313361' | 'India' | '2019' | 'Financial Year (Apr - Mar), 2019'\n '2159043' | '291808' | 'India' | '2020' | 'Financial Year (Apr - Mar), 2020'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Sourceyear' | 'Exports in usd' | 'Exports in rupees' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Tobacco' | '1996-97' | '4.2' | '30' | 'India' | 'U.S.A' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Carpets' | '2012-13' | '91.0' | '700' | 'India' | 'U.K.' | '2012' | 'Financial Year (Apr - Mar), 2012'\n 'Rice' | '1996-97' | '33.6' | '220' | 'India' | 'Kuwait' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n\n Question: 'Year wise list of India's export destination countries.'\n\n Solution:\n Step 1: Firstly, let's analyze the tables in the schema.\n\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n This table contains information about India\'s imports and exports of principal agricultural commodities. However, the question does not specifically ask for agricultural commodities. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. Further, columns such as 'Imports or exports', 'Quantity of imports or exports of principal agricultural commodities', 'Value of imports or exports of principal agricultural commodities', etc. do not mention any information about the destination country. Therefore, for the given question, which is about India\'s export destination countries, this table is not directly useful.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n This table focuses on the export of select commodities from India to different countries and is potentially relevant to the question. Let\'s look into the table content. The column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in rupees. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. The column 'Commodity name' contains 'Carpet', 'Tea', etc., so this column indicates the type of commodity exported. Column 'Export of selected new item commodities to principal countries.' contains values like 'Italy', 'Iraq', etc. which are basically different countries, so this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. This table can provide insights into India's export destination countries and can be utilized to answer the question effectively.\n\n Table: 'Selected Economic and Social Indicator 6159'\n This table contains selected economic and social indicators for India. It is difficult to determine just from the name of the table whether it is relevant or not. Let's analyze the content of the table to get better understanding. The column 'Exports in crores' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in crores. Similarly, column 'Exports in US millions' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US millions. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. There are no other columns in the table which have information about other countries. Thus, this table may not be directly useful for answering the question.\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n This table contains information about exports of select commodities from India to principal countries based on an old commodity classification. This table could be relevant to the question. Let's look into the table content. The 'Commodity name' column indicates the type of commodity being exported, such as 'Tobacco,' 'Carpets,' and 'Rice.' Based on the values in the column 'SourceYear', this column contains information about the year for which the data is available. Column 'Exports in usd' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US dollars. Similarly, the column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in Indian rupees. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. Column 'Export to country' contains values like 'U.S.A', 'Kuwait', etc. indicating different countries. So this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. Based on the above analysis, I think this table is relevant. Let's move to the next table.\n\n\n Step 2: Selecting relevant tables:\n\n Based on the analysis, it appears that the second table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (New commodity classification as per 2009-10) 7455') and the fourth table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (Old commodity classification) 7454') are relevant for determining India's export destination countries.\n\n\n Step 3: Determining relevant schema:\n\n Based on the question and information from the relevant tables obtained in step 2, this is the relevant schema:\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'('Exports in rupees' , 'Commodity name' , 'Export of selected new item commodities to principal countries.' , 'Yearcode' )\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'('Commodity name' , 'Exports in usd' , 'Exports in rupees' , 'Export to country' , 'Yearcode' ).\n Ensure that each table in the relevant schema refer only to columns that were present in its original description at the start.\n\n\n Step 4: Finally, based on the question and the relevant schema, we can write the SQL as:\n\n WITH export_data AS ( SELECT 'Yearcode', 'Commodity name', 'Export of selected new item commodities to principal countries.' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455' GROUP BY 'Year', 'Commodity name', destination_country UNION ALL SELECT 'Yearcode', 'Commodity name', 'Export to country' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454' GROUP BY 'Year', 'Commodity name', destination_country ) SELECT 'Year', destination_country, sum(total_exports) FROM export_data GROUP BY 'Year', 'destination_country'"
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Thank you for providing the detailed solution. Based on the example, I understand the strategy to be used for writing the SQL query. Please go ahead and provide the next question along with the corresponding schema, and I'll assist you in writing the SQL query using the same strategy."
        },
        {
            "role": "user",
            "content": "Now, write SQL for the question mentioned below based on the corresponding schema information using the same strategy mentioned in the example previously.\n\n Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n Table: 'India\’s Imports or Exports of Principal Agricultural Commodities 6165'\n 'Imports or exports' | 'Commodity type' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'Country' | 'YearCode' | 'Year'\n 'Exports' | 'Oil meals' | '7831.79' | '4671.14' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n 'Imports' | 'Fresh vegetables' | '11.12' | '8.55' | 'India' | '2016' | 'Financial Year (Apr - Mar), 2016'\n 'Imports' | 'Tobacco unmanufactured' | '69.47' | '1.54' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n 'Commodity name' | 'Exports in usd' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n 'Rice' | '31.0' | 'India' | 'U.S.A' | '2002' | 'Financial Year (Apr - Mar), 2002'\n 'Tea' | '75.3' | 'India' | 'Others' | '1996' | 'Financial Year (Apr - Mar), 1996'\n 'Iron Ore' | '0.0' | 'India' | 'Chile' | '2008' | 'Financial Year (Apr - Mar), 2008'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( New commodity classification as per 2009-10 ) 7453'\n 'Imports in usd' | 'Imports of principal commodities' | 'Imports in inr' | 'Country' | 'Yearcode' | 'Year'\n '14977.5' | 'Iron & Steel' | '97960' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '241.3' | 'Sulphur & Unroasted Iron Pyrts' | '1099' | 'India' | '2010' | 'Financial Year (Apr - Mar), 2010'\n '22732.5' | 'Transport equipment' | '146509' | 'India' | '2017' | 'Financial Year (Apr - Mar), 2017'\n\n Table: 'Trade and Balance of Payments - Imports of Principal Commodities ( Old commodity classification ) 7452'\n 'Imports of principal commodities' | 'Sourceyear' | 'Imports in rupees' | 'Country' | 'Year' | 'Imports in usd' | 'Yearcode'\n 'Crude Rubber, including Synthetic and Reclaimed' | '1991-92 ' | '182.0' | 'India' | 'Financial Year (Apr - Mar), 1991' | '73.8' | '1991'\n 'Organic and Inorganic Chemicals' | '2012-13 ' | '105141.0' | 'India' | 'Financial Year (Apr - Mar), 2012' | '19323.9' | '2012'\n 'Capital Goods' | '1988-89 ' | '6956.0' | 'India' | 'Financial Year (Apr - Mar), 1988' | '4803.0' | '1988'\n\n Table: 'Imports and Exports of Principal Agricultural Commodities of India 6129'\n 'Imports of agricultural commodities' | 'Percentage of agricultural imports to total national imports' | 'Agricultural exports to total national imports' | 'Country' | 'Yearcode' | 'Year'\n '54365.29' | '3.99' | '9.99' | 'India' | '2009' | 'Financial Year (Apr - Mar), 2009'\n '5890.1' | '4.8' | '19.18' | 'India' | '1995' | 'Financial Year (Apr - Mar), 1995'\n '6612.6' | '4.76' | '20.33' | 'India' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n Table: 'Selected Economic and Social Indicator 6159'\n 'Imports in US millions' | 'Country' | 'YearCode' | 'Year'\n '381008' | 'India' | '2015' | 'Financial Year (Apr - Mar), 2015'\n '50536' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n '15869' | 'India' | '1980' | 'Financial Year (Apr - Mar), 1980'\n\n Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n 'Commodity name' | 'Country' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n 'Rice' | 'India' | 'France' | '2015' | 'Financial Year (Apr - Mar), 2015'\n 'Cashew' | 'India' | 'U.S.A' | '2013' | 'Financial Year (Apr - Mar), 2013'\n 'Cotton Yarn/Fabs./made-ups, Handloom Products etc.' | 'India' | 'Mauritius' | '2018' | 'Financial Year (Apr - Mar), 2018'\n\n Question: Which commodity had the highest imports in Old Commodity Classification?\n SQL Query:{} \n\n\nKeep note of all the instructions mentioned previously and use the same strategy as mentioned in the example for this question as well."
        },
    ]

    response = openai.ChatCompletion.create(
        engine = deployment_name,
        messages=messages,
        temperature=0
    )

    return response['choices'][0]['message']['content']

#%%
output = generate()
print(output)

# %%
output

# %%
STEPS_B = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Your task is to create a syntactically correct SQL query based on an input question."
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Sure, I'd be happy to!"
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "First, I'll present an example that includes a database schema, a related question, and a detailed, step-by-step guide on how to construct the SQL query to answer it. Please pay close attention to this example and follow the steps carefully. Afterward, I'll provide you with a new question and its corresponding schema. Your task will be to write an SQL query for it, using the same strategy outlined in the example. Make sure to follow the steps in the same order and apply the same techniques to construct your query."
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Sure. Please provide the example with the schema information, question, and detailed solution, and I'll assist you accordingly."
    },
    {
        "role": "system",
        "name": "example_user",
        "content": (
            "Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n "
            "Table: 'India’s Imports or Exports of Principal Agricultural Commodities 6165'\n "
            "'Country' | 'Imports or exports' | 'SourceYear' | 'Value of imports or exports of principal agricultural commodities' | 'Quantity of imports or exports of principal agricultural commodities' | 'YearCode' | 'Year' | 'Commodity type'\n "
            "'India' | 'Exports' | '2009-10' | '2973.19' | '2881.22' | '2009' | 'Financial Year (Apr - Mar), 2009' | 'Other cereals'\n "
            "'India' | 'Exports' | '2017-18' | '4913.28' | '714.0' | '2017' | 'Financial Year (Apr - Mar), 2017' | 'Fresh fruits'\n "
            "'India' | 'Imports' | '2012-13' | '18.29' | 'nan' | '2012' | 'Financial Year (Apr - Mar), 2012' | 'Poultry products'\n\n "
            "Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n "
            "'Exports in rupees' | 'Country' | 'Commodity name' | 'Export of selected new item commodities to principal countries.' | 'Yearcode' | 'Year'\n "
            "'165' | 'India' | 'Carpet' | 'Italy' | '2020' | 'Financial Year (Apr - Mar), 2020'\n "
            "'120' | 'India' | 'Tea' | 'Iraq' | '2020' | 'Financial Year (Apr - Mar), 2020'\n "
            "'66' | 'India' | 'Cashew' | 'France' | '2019' | 'Financial Year (Apr - Mar), 2019'\n\n "
            "Table: 'Selected Economic and Social Indicator 6159'\n "
            "'Exports in crores' | 'Exports in US millions' | 'Country' | 'YearCode' | 'Year'\n "
            "'203571' | '44560' | 'India' | '2000' | 'Financial Year (Apr - Mar), 2000'\n "
            "'2219854' | '313361' | 'India' | '2019' | 'Financial Year (Apr - Mar), 2019'\n "
            "'2159043' | '291808' | 'India' | '2020' | 'Financial Year (Apr - Mar), 2020'\n\n "
            "Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n "
            "'Commodity name' | 'Sourceyear' | 'Exports in usd' | 'Exports in rupees' | 'Country' | 'Export to country' | 'Yearcode' | 'Year'\n "
            "'Tobacco' | '1996-97' | '4.2' | '30' | 'India' | 'U.S.A' | '1996' | 'Financial Year (Apr - Mar), 1996'\n "
            "'Carpets' | '2012-13' | '91.0' | '700' | 'India' | 'U.K.' | '2012' | 'Financial Year (Apr - Mar), 2012'\n "
            "'Rice' | '1996-97' | '33.6' | '220' | 'India' | 'Kuwait' | '1996' | 'Financial Year (Apr - Mar), 1996'\n\n\n "
            "Question: 'Year wise list of India's export destination countries.'\n\n "
            "Solution:\n "
            "Step 1: Firstly, let's analyze the tables in the schema.\n\n "
            "Table: 'India’s Imports or Exports of Principal Agricultural Commodities 6165'\n "
            "This table contains information about India's imports and exports of principal agricultural commodities. However, the question does not specifically ask for agricultural commodities. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. Further, columns such as 'Imports or exports', 'Quantity of imports or exports of principal agricultural commodities', 'Value of imports or exports of principal agricultural commodities', etc. do not mention any information about the destination country. Therefore, for the given question, which is about India's export destination countries, this table is not directly useful.\n\n "
            "Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'\n "
            "This table focuses on the export of select commodities from India to different countries and is potentially relevant to the question. Let's look into the table content. The column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in rupees. The 'Country' column contains 'India' as the value, which is not a destination country, but rather the source country. The column 'Commodity name' contains 'Carpet', 'Tea', etc., so this column indicates the type of commodity exported. Column 'Export of selected new item commodities to principal countries.' contains values like 'Italy', 'Iraq', etc. which are basically different countries, so this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. This table can provide insights into India's export destination countries and can be utilized to answer the question effectively.\n\n "
            "Table: 'Selected Economic and Social Indicator 6159'\n "
            "This table contains selected economic and social indicators for India. It is difficult to determine just from the name of the table whether it is relevant or not. Let's analyze the content of the table to get better understanding. The column 'Exports in crores' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in crores. Similarly, column 'Exports in US millions' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US millions. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. There are no other columns in the table which have information about other countries. Thus, this table may not be directly useful for answering the question.\n\n "
            "Table: 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'\n "
            "This table contains information about exports of select commodities from India to principal countries based on an old commodity classification. This table could be relevant to the question. Let's look into the table content. The 'Commodity name' column indicates the type of commodity being exported, such as 'Tobacco,' 'Carpets,' and 'Rice.' Based on the values in the column 'SourceYear', this column contains information about the year for which the data is available. Column 'Exports in usd' contains numerical values, and based on the name and values in it, I think it is about the amount of the commodity exported in US dollars. Similarly, the column 'Exports in rupees' contains numerical data. Based on the name of the column and values in it, I think the column is about the total amount of the commodity exported in Indian rupees. Column 'Country' contains 'India' as the value, which is not a destination country, but rather the source country. Column 'Export to country' contains values like 'U.S.A', 'Kuwait', etc. indicating different countries. So this column provides information about the destination country to which items are exported from India. The 'Year' and 'Yearcode' columns represent the financial year and the corresponding year of the export. Based on the above analysis, I think this table is relevant.\n\n\n "
            "Step 2: Selecting relevant tables:\n\n "
            "Based on the analysis in step 1, we can conclude that second table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (New commodity classification as per 2009-10) 7455') and the fourth table ('Trade and Balance of Payments - Exports of Select Commodities to Principal Countries (Old commodity classification) 7454') are relevant for determining India's export destination countries.\n\n\n "
            "Step 3: Determining relevant schema:\n\n While determining relevant schema, ensure that each table in the relevant schema refer only to columns that were present in its original description at the start.\n Based on the question and information from the relevant tables obtained in step 2, this is the relevant schema:\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455'('Exports in rupees' , 'Commodity name' , 'Export of selected new item commodities to principal countries.' , 'Yearcode' )\n 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454'('Commodity name' , 'Exports in usd' , 'Exports in rupees' , 'Export to country' , 'Yearcode' ).\n\n\n "
            "Step 4: Finally, based on the question and the relevant schema, we can write the SQL as:\n\n "
            "WITH export_data AS ( SELECT 'Yearcode', 'Commodity name', 'Export of selected new item commodities to principal countries.' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( New commodity classification as per 2009-10 ) 7455' GROUP BY 'Year', 'Commodity name', destination_country UNION ALL SELECT 'Yearcode', 'Commodity name', 'Export to country' AS destination_country, SUM('Exports in rupees') AS total_exports FROM 'Trade and Balance of Payments - Exports of Select Commodities to Principal Countries ( Old commodity classification ) 7454' GROUP BY 'Year', 'Commodity name', destination_country ) SELECT 'Year', destination_country, sum(total_exports) FROM export_data GROUP BY 'Year', 'destination_country'"
        )
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Thank you for providing the detailed solution. Based on the example, I understand the strategy to be used for writing the SQL query. Please go ahead and provide the next question along with the corresponding schema, and I'll assist you in writing the SQL query using the same strategy."
    },
    {
        "role": "user",
        "content": "Schema information:\n Here are a few rows from each table in the schema. Keep in mind that the tables in the schema may have more rows than what is shown here.\n {}\n\n Question: {}\n\n\nRemember, your task is to write syntactically correct SQL query.\nPay attention to use only the column names that you can see in the schema description of the respective tables. Be careful to not query for columns that do not exist.\nAlso, pay very careful attention to which column is in which table.\nAnd yeah, do not forget to use the same strategy as mentioned in the above example for this question as well."
    }
]

for item in STEPS_B:
    print(item['role'])
    print(item['name'] if 'name' in item else '')
    print(item['content'])
    print()





# %%
############################################################################################################
import sys
sys.path.append('/raid/infolab/mayankk/RA/june')
import utils.openai_main as openai_main 

import time
import openai

def generate_U(uni_prompt):
    deployment_name = 'chatgpt-35-16k'
    response = '' 
    try:
        response = openai.ChatCompletion.create(
            engine = deployment_name,
            messages=[
                    {"role": "user", "content": uni_prompt}
                ],
            temperature=0
        )
    except Exception as e:
        time.sleep(10)
        try:
            response = openai.ChatCompletion.create(
                engine = deployment_name,
                messages=[
                        {"role": "user", "content": uni_prompt}
                    ],
                temperature=0
            )
        except Exception as e:
            return f"An error occured: {str(e)}"
    return response['choices'][0]['message']['content'] 

test_prompt = (
    "Consider the following schema:\n\n"
    "commodity_types(commodity_type_code, commodity_type_name, classification_system)\n"
    "commodity_export_data(origin_country, destination_country, export_year, commodity_type_code, export_amount)\n"
    "commodity_import_data(origin_country, destination_country, import_year, commodity_type_code, import_amount)\n\n"
    "Based on the above schema, please write SQL for the following question: \"{}\""
)

test_prompt_2 = (
    "Given an input question, create a syntactically correct SQL query to run. "
    "Never query for all the columns from a specific table. You must query only the columns that are needed to answer the question. " 
    "Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. " 
    "Also, pay attention to which column is in which table.\n"
    "Consider the following schema:\n\n"
    "commodity_types(commodity_type_code (Primary Key), commodity_type_name, classification_system)\n"
    "commodity_export_data(origin_country, destination_country, export_year, commodity_type_code (Foreign Key referencing commodity_types.commodity_type_code), export_amount)\n"
    "commodity_import_data(origin_country, destination_country, import_year, commodity_type_code (Foreign Key referencing commodity_types.commodity_type_code), import_amount)\n\n"
    "Based on the above schema, please write SQL for the following question: \"{}\"\n\n"
    "Make use of subqueries whenever possible. Properly examine the need for using aggregate functions."
)

test_prompt_3 = (
    "Consider the following schema:\n\n"
	"Table: commodity_types\nColumns: \n- commodity_type_code (data type: VARCHAR)\n- commodity_type_name (data type: VARCHAR)\n- classification_system (data type: VARCHAR)\n\n"
	"Table: commodity_export_data\nColumns:\n- origin_country (data type: VARCHAR)\n- destination_country (data type: VARCHAR)\n- export_year (data type: INT)\n- commodity_type_code (data type: VARCHAR)\n- export_amount (data type: DECIMAL)\n\n"
	"Table: commodity_import_data\nColumns:\n- origin_country (data type: VARCHAR)\n- destination_country (data type: VARCHAR)\n- import_year (data type: INT)\n- commodity_type_code (data type: VARCHAR)\n- import_amount (data type:\n\n"
	"Based on the above schema and instructions, please write SQL for the following question: \"{}\"\n\n"
	"Make use of subqueries whenever possible. Properly examine the need for using aggregate functions."
)

test_prompt_4 = (
    "Consider the following schema:\n\n"
	"commodity_types(\n    commodity_type_code (Primary Key),\n    commodity_type_name,\n    classification_system\n)\n\n"
	"countries(\n    country_code (Primary Key),\n    country_name\n)\n\n"
	"commodity_data(\n    data_id (Primary Key),\n    origin_country_code (Foreign Key referencing countries.country_code),\n    destination_country_code (Foreign Key referencing countries.country_code),\n    year,\n    commodity_type_code (Foreign Key referencing commodity_types.commodity_type_code),\n    export_amount,\n    import_amount\n)\n\n"
	"total_trade(\n    origin_country_code (Foreign Key referencing countries.country_code),\n    destination_country_code (Foreign Key referencing countries.country_code),\n    total_export_amount,\n    total_import_amount\n)\n\n"
	"\nBased on the above schema and instructions, please write SQL for the following question: \"{}\""
)

test_prompt_5 = (
    "Consider the following schema:\n\n"
    "commodity_types(commodity_type_code, commodity_type_name, classification_system)\n"
    "commodity_export_data(origin_country, destination_country, export_year, commodity_type_code, export_amount)\n"
    "commodity_import_data(origin_country, destination_country, import_year, commodity_type_code, import_amount)\n\n"
    "Based on the schema above, please write SQL for \"{}\""
)

print("\nprompt: ", test_prompt.format("What is the total amount of exports from India to the United States in the year 2019?"))
print("\nprompt: ", test_prompt_2.format("What is the total amount of exports from India to the United States in the year 2019?"))
print("\nprompt: ", test_prompt_3.format("What is the total amount of exports from India to the United States in the year 2019?"))
print("\nprompt: ", test_prompt_4.format("What is the total amount of exports from India to the United States in the year 2019?"))
print("\nprompt: ", test_prompt_5.format("What is the total amount of exports from India to the United States in the year 2019?"))

# %%
questions = [
    "Which commodity had the highest imports in Old Commodity Classification?",
    "Which commodity had the lowest imports in Old Commodity Classification?",
    "Which commodity had the highest imports in New Commodity Classification (as per 2009-10)?",
    "Which commodity had the lowest imports in New Commodity Classification (as per 2009-10)?",
    "Maximum exports from India was to which country?",
    "Which country received the minimum exports from India?",
    "What are the main countries that India exports its goods to?",
    "How has the list of India's biggest export destination countries changed over the past decade?",
    "Which countries have consistently been the top export destinations for India?",
    "What are the major goods or sectors that India exports to these countries?",
    "How do these export destinations contribute to India's overall export revenue?"
]
questions[:]

# %%
from tqdm import tqdm

trial_number = 2
with open(f'../results/trial_number_{trial_number}.txt', 'w') as f:
    for question in tqdm(questions):
        pred = generate_U(test_prompt_2.format(question))
        print('Prompt: ', test_prompt_2.format(question), file=f)
        print('\nResponse: ', pred, file=f)
        print('\n-------------------\n\n', file=f)

#################
# %%
questions_2 = [
    "Which country received the minimum exports from India?",
    "How has the list of India's biggest export destination countries changed over the past decade?",
    "Which countries have consistently been the top export destinations for India?",
    "What are the major goods or sectors that India exports to these countries?",
    "Which commodity had the highest imports in Old Commodity Classification?",
    "Which commodity had the lowest imports in New Commodity Classification (as per 2009-10)?",
]
questions_2[:]

# %%
for question in questions_2:
    pred = generate_U(test_prompt_5.format(question))
    print('Prompt: ', test_prompt_5.format(question))
    print('\nResponse: ', pred)
    print('\n-------------------\n\n')

# %%
