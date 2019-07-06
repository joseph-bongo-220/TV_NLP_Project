import sys

def second(elem):
    """takes second item of list"""
    return elem[1]

# flatten a list of lists
def flatten(x):
    """Flattens a list of lists into a single list"""
    flattened = [val for sublist in x for val in sublist]
    return flattened

def sort_dict(dict_unsorted, value=True, descending=True):
    if value:
        dict_sorted = sorted(dict_unsorted.items(), key=lambda x: x[1], reverse=descending)
    else:
        dict_sorted = sorted(dict_unsorted.items(), key=lambda x: x[0], reverse=descending)
    return dict_sorted

def smallest_distance(list1, list2):
    a = 0
    b = 0
    distance = sys.maxsize 
    while (a < len(list1) and b < len(list2)): 
        if (abs(list1[a] - list2[b]) < distance): 
            distance = abs(list1[a] - list2[b]) 
        if (list1[a] < list2[b]): 
            a += 1
        else: 
            b += 1 
    return distance