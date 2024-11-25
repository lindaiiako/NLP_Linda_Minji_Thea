LLAMA_IT_SYS_PROMPT = '''
You are given an incomplete dialog between two parties named <user> and <sys>. 
The <sys> party acts as an assistant that helps the <user> party in booking and/or finding information about a restaurant, hotel, or attraction in Cambridge.
Your task is to identify the entity types in the next possible <sys> response. The possible entity types are:
1. "address": address of the restaurant, hotel, or attraction
2. "area": general location of the restaurant, hotel, or attraction (centre, east, west, north, south)
3. "entrance_fee": entrance fee information for the attraction
4. "food": type of food (e.g., italian, modern european, etc.)
5. "name": name of the restaurant, hotel, or attraction
6. "phone": phone number of the restaurant, hotel, or attraction
7. "postcode": post code of the restaurant, hotel, or attraction
8. "pricerange": general price range of the restaurant, hotel, or attraction (cheap, moderate, expensive)
9. "type": type of hotel (guest house, hotel) or type of attraction (e.g., multiple sports, architecture, entertainment, etc.)
10. "stars": hotel star rating (e.g., 3, 4, etc.)
11. "ref": hotel or restaurant booking reference number 
12: "choice": number of available options after <sys> searches the DB 

Let's think step-by-step.
What could be the entity types in the <sys> party's next response?  
Output your answer in the following format:
```
[your answer here]
```
Use `|` as delimeter for the list. If there is no entity type in the response, just write "no entity".
'''