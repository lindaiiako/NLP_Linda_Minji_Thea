IT_SYS_PROMPT = '''
You are given a dialog between a HUMAN and an ASSISTANT.

An ASSISTANT response may consist of the following entity names:
address: address of the restaurant, hotel, or attraction
area: general location of the restaurant, hotel, or attraction (centre, east, west, north, south)
entrance_fee: entrance fee information for the attraction
food: type of food (e.g., italian, modern european, etc.)
name: name of the restaurant, hotel, or attraction
phone: phone number of the restaurant, hotel, or attraction
postcode: post code of the restaurant, hotel, or attraction
pricerange: general price range of the restaurant, hotel, or attraction (cheap, moderate, expensive)
type: type of hotel (guest house, hotel) or type of attraction (e.g., multiple sports, architecture, entertainment, etc.)
stars: hotel star rating (e.g., 3, 4, etc.)
ref: hotel or restaurant booking reference number
choice: number of available options

Given the input dialog, generate the list of entity names in the next ASSISTANT response using `|` delimiter without repetitions (example: area | ref | name). If there is no entity in the response, just write "[no entity]".
Always use the following output format:
```
ASSISTANT:
```
'''


IT_SYS_PROMPT_OLD = '''
You are given an incomplete dialog between a HUMAN and an ASSISTANT.
The ASSISTANT party helps the HUMAN in booking and/or finding information about a restaurant, hotel, or attraction in Cambridge.

Given the input dialog, identify the last HUMAN utterance and predict the list of entity names in the NEXT ASSISTANT response.
List of valid entity names and their correponding descriptions:
"address": address of the restaurant, hotel, or attraction
"area": general location of the restaurant, hotel, or attraction
"entrance_fee": entrance fee information for the attraction
"food": type of food
"name": name of the restaurant, hotel, or attraction
"phone": phone number of the restaurant, hotel, or attraction
"postcode": post code of the restaurant, hotel, or attraction
"pricerange": general price range of the restaurant, hotel, or attraction
"type": type of hotel or type of attraction
"stars": hotel star rating
"ref": hotel or restaurant booking reference number
"choice": number of available options


Always use list format for your output without repetitions, using `|` as delimiter (example: area | stars | pricerange). If the list is empty, just write "[no entity]".
Always select only from the valid entity type names. Do not change spelling or add new entities.
Output only one list and nothing else.
'''

ANTROPHIC_PROMPT = '''You are an AI assistant tasked with predicting entity types from incomplete task-oriented dialogs. You will be given a dialog, a list of entity types, and additional information to help guide your decision.

Following is the incomplete task-oriented dialog:
{DIALOG}

This is a list of entity types that should be included in the ASSISTANT's next response: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice
This is an additional information, including another model's prediction and confidence score: {ADDITIONAL_INFO}
You have to predict which entity types from the given list should be included in the ASSISTANT's next response. Use the additional information to guide your decision.

Consider the dialog context, the given entity types, and the additional information, provide your final answer within [Answer] tags. Remember to use the `|` delimiter without repetitions. If there is no entity needed, just write "[no entity]". '''.strip()
