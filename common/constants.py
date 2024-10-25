
ET_LABELS_KEY_TO_VAL = {0: 'address', 1: 'area', 2: 'food', 3: 'name', 4: 'phone', 5: 'postcode', 6: 'pricerange', 7: 'type', 
                                           8: 'ref', 9: 'choice', 10: 'stars', 11: 'entrance_fee', 12: 'none'}
ET_LABELS_VAL_TO_KEY = {v: k for k, v in ET_LABELS_KEY_TO_VAL.items()}
NUM_CLASSES = len(ET_LABELS_KEY_TO_VAL)
