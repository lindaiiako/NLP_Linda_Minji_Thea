EP_FEW_SHOT_CONTEXT_SIMILARITY = """
### Instruction
You are an AI assistant responsible for predicting entity types required in the ASSISTANT's next response based on an incomplete task-oriented dialog.

- Input: A dialog and a predefined list of possible entity types.
- Output: A list of entity types that should appear in the ASSISTANT's next response, separated by `|` (e.g., area | ref | name). If no entity is needed, respond with "[no entity]".
- Ensure there are no repetitions in the list.

**Predefined Entity Types:** address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice.

### Example
"""

EP_ZERO_SHOT_CONTEXT_SIMILARITY = """
### Instruction
You are an AI assistant responsible for predicting entity types required in the ASSISTANT's next response based on an incomplete task-oriented dialog.

- Input: A dialog and a predefined list of possible entity types.
- Output: A list of entity types that should appear in the ASSISTANT's next response, separated by `|` (e.g., area | ref | name). If no entity is needed, respond with "[no entity]".
- Ensure there are no repetitions in the list.

**Predefined Entity Types:** address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice.

""".strip()

SUPER_ICL_PROMPT = """### Instruction: 
You are an AI assistant tasked with predicting a list of entity types that should be included in the ASSISTANT's next response.

- Input:
  - A dialog between a USER and an ASSISTANT.
  - Predictions and confidence scores from another model.
- Output: A list of entity types required in the ASSISTANT's next response using the predefined entity types below. Your output should be separated by `|` (e.g., area | ref | name) without any additional comments. If no entity is required, respond with "[no entity]".

**Predefined Entity Types**: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice.

### Example: 
[Dialog] USER: I kind of need some help finding a nice hotel in the north part of town.
[Model Prediction] name | stars | pricerange | area
[Confidence] 0.5392633
[Entity Types] area | name | pricerange | type

"""

SUPER_ICL_PROMPT_WO_CONFIDENCE = """ ### Instruction:
You are an AI assistant tasked with predicting a list of entity types that should be included in the ASSISTANT's next response.

- Input: A dialog between a USER and an ASSISTANT, and predictions from another model.
- Output: A list of entity types required in the ASSISTANT's next response using the predefined entity types below. Your output should be separated by `|` (e.g., area | ref | name) without any additional comments. If no entity is required, respond with "[no entity]".

**Predefined Entity Types**: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice.

### Example: 
[Dialog] USER: I kind of need some help finding a nice hotel in the north part of town.
[Model Prediction] name | stars | pricerange | area
[Entity Types] area | name | pricerange | type

"""
