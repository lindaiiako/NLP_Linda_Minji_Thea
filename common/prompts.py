LLAMA_IT_SYS_PROMPT = '''
You are given an incomplete dialog between two parties named <user> and <sys>. 
The <sys> party acts as an assistant that helps the <user> party in booking and/or finding information about a restaurant, hotel, or attraction in Cambridge.

Given the input dialog, identify the last <user> utterance and predict the list of entity types in the NEXT <sys> response.
List of valid entity type names and their correponding descriptions:
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


Always use list format for your output without repetitions, using `|` as delimiter. If the list is empty, just write "[no entity]".
Always select only from the valid entity type names. Do not change spelling or add new entities.
Output only one list and nothing else.

Example output:
[area | stars]
[no entity]
[phone | pricerange | name]
'''