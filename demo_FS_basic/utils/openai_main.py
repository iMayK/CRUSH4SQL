import sys

import torch
import openai
from openai.embeddings_utils import get_embedding

PROMPT = {
    'base': (
        "Find most important variables from the question.\n\n"
        "Example 1:\n\nQuestion: What are the names of all stations with a latitude smaller than 37.5?\n\nVariables:\n1:stations\n2:latitude\n\n"
        "Example 2:\n\nQuestion: Count the number of members in club Bootup Balitmore whose age is above 18.\n\nVariables:\n1:age\n2:club\n\n"
        "Example 3:\n\nQuestion: Show the season, the player, and the name of the team that players belong to.\n\nVariables:\n1:player\n2:season\n3:match\n\n"
        "Example 4:\n\nQuestion: Find the first name and age of the students who are playing both Football and Lacrosse.\n\nVariables:\n1:student\n2:age\n3:game\n4:football\n\n"
        "Example 5:\n\nQuestion: What are the names of tourist attractions that can be reached by bus or is at address 254 Ottilie Junction?\n\nVariables:\n1:transport\n2:tourist\n3:tourist attraction\n\n"
        "Example 6:\n\nQuestion: Give the name of the highest paid instructor.\n\nVariables:\n1:instructor\n2:amount\n3:salary\n\n"
        "Example 7:\n\nQuestion: {0}\n\nVariables:"
    ),
    'relation_info': (
        "Find most important variables from the question.\n\n"
        "Example 1:\n\nQuestion: What are the names of all stations with a latitude smaller than 37.5?\n\nVariables:\n1:latitude of the station\n2:name of the station\n\n"
        "Example 2:\n\nQuestion: Count the number of members in club Bootup Balitmore whose age is above 18.\n\nVariables:\n1:age of club members\n2:name of the club\n\n"
        "Example 3:\n\nQuestion: Show the season, the player, and the name of the team that players belong to.\n\nVariables:\n1:name of the team the player belongs to\n2:season(s) played by player\n3:name of the player\n\n"
        "Example 4:\n\nQuestion: Find the first name and age of the students who are playing both Football and Lacrosse.\n\nVariables:\n1:sports played by student\n2:age of student\n3:name of student\n\n"
        "Example 5:\n\nQuestion: What are the names of tourist attractions that can be reached by bus or is at address 254 Ottilie Junction?\n\nVariables:\n1:name of tourist attraction\n2:tourist attraction reachable by bus\n3:address of tourist attraction\n\n"
        "Example 6:\n\nQuestion: Give the name of the highest paid instructor.\n\nVariables:\n1:salary of instructor\n2:name of instructor\n\n"
        "Example 7:\n\nQuestion: {0}\n\nVariables:"
    ),
    'from_sql_info_original': (
        "Find most important variables from the question.\n\n"
        "Example 1:\n\nQuestion: What are the names of all stations with a latitude smaller than 37.5?\n\nVariables:\n1:lat of station\n2:name of station\n\n"
        "Example 2:\n\nQuestion: Count the number of members in club Bootup Balitmore whose age is above 18.\n\nVariables:\n1:clubdesc of club\n2:clubid of club\n3:clublocation of club\n4:clubname of club\n5:clubid of member_of_club\n6:stuid of member_of_club\n7:age of student\n8:stuid of student\n\n"
        "Example 3:\n\nQuestion: Show the season, the player, and the name of the team that players belong to.\n\nVariables:\n1:player of match_season\n2:season of match_season\n3:team of match_season\n4:name of team\n5:team_id of team\n\n"
        "Example 4:\n\nQuestion: Find the first name and age of the students who are playing both Football and Lacrosse.\n\nVariables:\n1:sportname of sportsinfo\n2:stuid of sportsinfo\n3:age of student\n4:fname of student\n5:stuid of student\n\n"
        "Example 5:\n\nQuestion: What are the names of tourist attractions that can be reached by bus or is at address 254 Ottilie Junction?\n\nVariables:\n1:address of locations\n2:location_id of locations\n3:how_to_get_there of tourist_attractions\n4:location_id of tourist_attractions\n5:name of tourist_attractions\n\n"
        "Example 6:\n\nQuestion: Give the name of the highest paid instructor.\n\nVariables:\n1:name of instructor\n2:salary of instructor\n\n"
        "Example 7:\n\nQuestion: {0}\n\nVariables:"
    ),
    'from_sql_info':(
        "Find most important variables from the question.\n\n"
        "Example 1:\n\nQuestion: What are the names of all stations with a latitude smaller than 37.5?\n\nVariables:\n1:latitude of station\n2:name of station\n\n"
        "Example 2:\n\nQuestion: Count the number of members in club Bootup Balitmore whose age is above 18.\n\nVariables:\n1:club description of club\n2:club id of club\n3:club location of club\n4:club name of club\n5:club id of member of club\n6:student id of member of club\n7:age of student\n8:student id of student\n\n"
        "Example 3:\n\nQuestion: Show the season, the player, and the name of the team that players belong to.\n\nVariables:\n1:player of match season\n2:season of match season\n3:team of match season\n4:name of team\n5:team id of team\n\n"
        "Example 4:\n\nQuestion: Find the first name and age of the students who are playing both Football and Lacrosse.\n\nVariables:\n1:sport name of sports info\n2:student id of sports info\n3:age of student\n4:first name of student\n5:student id of student\n\n"
        "Example 5:\n\nQuestion: What are the names of tourist attractions that can be reached by bus or is at address 254 Ottilie Junction?\n\nVariables:\n1:address of locations\n2:location id of locations\n3:how to get there of tourist attractions\n4:location id of tourist attractions\n5:name of tourist attractions\n\n"
        "Example 6:\n\nQuestion: Give the name of the highest paid instructor.\n\nVariables:\n1:name of instructor\n2:salary of instructor\n\n"
        "Example 7:\n\nQuestion: {0}\n\nVariables:"
    ),
    'hallucinate_schema':(
        "Hallucinate the minimal schema of a relational database that can be used to answer the natural language question. Here are some examples:\n\n"
        "Example 1:\n\nQuestion: What are the names of all stations with a latitude smaller than 37.5?\n\nTables:\n1:station(name, latitude)\n\n"
        "Example 2:\n\nQuestion: Count the number of members in club Bootup Balitmore whose age is above 18.\n\nTables:\n1:club(name, club id, club description, location)\n2:member_of_club(club id, student id)\n3:student(student id, age)\n\n"
        "Example 3:\n\nQuestion: Show the season, the player, and the name of the team that players belong to.\n\nTables:\n1:match_season(season, team, player)\n2:team(name, team identifier)\n\n"
        "Example 4:\n\nQuestion: Find the first name and age of the students who are playing both Football and Lacrosse.\n\nTables:\n1:student(first name, age, student id)\n2:sportsinfo(student id, sportname)\n\n"
        "Example 5:\n\nQuestion: What are the names of tourist attractions that can be reached by bus or is at address 254 Ottilie Junction?\n\nTables:\n1:locations(address, location id)\n2:tourist_attractions(location id, name, how to get there)\n\n"
        "Example 6:\n\nQuestion: Give the name of the highest paid instructor.\n\nTables:\n1:instructor(name, salary)\n\n"
        "Example 7:\n\nQuestion: {0}\n\nTables:"
    ),
    'hallucinate_schema_ndap':(
        "Hallucinate the minimal schema of a relational database that can be used to answer the natural language question. Here are some examples:\n\n"
        "Example 1:\n\nQuestion: what is the correlation between child nourishment and parental education in the state of Madhya Pradesh?\n\nTables:\n1:family_health_survey(child age, child nourishment)\n2:population_census(age, state, male literate population, female literate population)\n\n"
        "Example 2:\n\nQuestion: health center per population ratio at the village level or district level from the year 2015?\n\nTables:\n1:health_care_infrastructure(village, district, health care facilities)\n2:population_census(district, male population, female population)\n\n"
        "Example 3:\n\nQuestion: distribution of medical professionals by type across regions from 2011 onwards from the state of Kerala?\n\nTables:\n1:village_amenities_census(state, medical professional)\n2:health_statistics_statewise(medical professional)\n\n"
        "Example 4:\n\nQuestion: Correlation between road connectivity/length of road and Mother Mortality Rate (MMR) during 2011 from the state UK?\n\nTables:\n1:health_and_well_being(state, year, maternal mortality ratio)\n2:health_information_indicators_district(maternal deaths)\n3:health_facilities_under_scheme(scheme, maternal death)\n4:Road_statistics(state, road type)\n\n"
        "Example 5:\n\nQuestion: What is the trend for CPI of goods excluding food and fuel?\n\nTables:\n1:inflation_money_and_credit(year, Categories of Consumer Expenditure)\n\n"
        "Example 6:\n\nQuestion: Correlation between number of bank branches and district growth?\n\nTables:\n1:town_amenities_census(health institutions, academic institutions, public works department)\n2:bank_details(number of branches, bank type)\n\n"
        "Example 7:\n\nQuestion: {0}\n\nTables:"
    )
}

def get_openai_embedding(query, api_type, api_key, endpoint, api_version):
    """
    Get text embedding using the Azure OpenAI API.

    Args:
        query (str): The input text query to be embedded.
        api_key (str): Your Azure OpenAI API key.
        endpoint (str): The Azure OpenAI API endpoint.
        api_type (str): whether azure or python 
        api_version (str): The Azure OpenAI API version.

    Returns:
        torch.Tensor: A tensor representing the text embedding.
    """
    if api_type == "azure":
        openai.api_type = "azure"
        openai.api_key = api_key
        openai.api_base = endpoint
        openai.api_version = api_version 

        deployment_name='text-embedding-ada-002'
        embedding = get_embedding(
            query,
            engine=deployment_name # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
        )
    else:
        openai.api_key = api_key
        model_id = "text-embedding-ada-002"
        embedding = openai.Embedding.create(
            input=query,
            model=model_id
        )['data'][0]['embedding']

    # embedding will be a list of len dimension of vector
    embedding_tensor = torch.Tensor(embedding)
    return embedding_tensor

def get_hallucinated_segments(prompt_type, query, api_type, api_key, endpoint, api_version, max_tokens=1000, temperature=0):
    prompt = PROMPT[prompt_type].format(query)
    if api_type == "azure":
        openai.api_type = "azure"
        openai.api_key = api_key
        openai.api_base = endpoint
        openai.api_version = api_version 

        deployment_name = 'prefix-text-davinci-003' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
        response = openai.Completion.create(
            engine=deployment_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        text = response['choices'][0]['text']
    else:
        openai.api_key = api_key
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        text = response.choices[0].text

    try:
        segments = [l for l in text.splitlines() if l != ""]
        segments = [segment.split(':')[1].strip() for segment in segments]
        return segments
    except:
        return [text]

def main():

    query = "What is the debt-GDP ratio of India?"

    api_type = 'azure'
    api_key = input('AZURE_OPENAI_API_KEY: ')
    endpoint = input('AZURE_OPENAI_ENDPOINT: ')

    ### test generating query embedding
    #embedding = get_openai_embedding(query, api_type, api_key, endpoint)
    #print(query)
    #print(embedding.shape)

    ### test generating segments
    
    #for prompt_type, prompt in PROMPT.items():
        #print(prompt_type)
        #print(prompt)
        #print()

    print('query: ', query)
    segments = get_hallucinated_segments('hallucinate_schema_ndap', query, api_type, api_key, endpoint)
    print(segments)


if __name__ == "__main__":
    main()

