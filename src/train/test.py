new_dict = {
    "a": 1,
    "b": 2,
    "c": 3,
}

other_dict = {
    "d": 4,
    "e": 5,
    "f": 6, 
    **new_dict
}

# for key, item in new_dict.items(): 
#     print(key, item)
for i, (key, item) in enumerate(other_dict.items()): 
    print(i, key, item)
