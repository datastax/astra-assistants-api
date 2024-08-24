import re


def camel_to_snake(name):
    """Convert a camelCase name to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_keys_to_snake_case(data):
    """Recursively convert all keys in a dictionary (or list of dictionaries) from camelCase to snake_case."""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = camel_to_snake(k)
            new_dict[new_key] = convert_keys_to_snake_case(v)  # Recursively apply to nested dictionaries/lists
        return new_dict
    elif isinstance(data, list):
        return [convert_keys_to_snake_case(item) for item in data]
    else:
        return data


import re

def snake_to_camel(name):
    """Convert a snake_case name to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def convert_keys_to_camel_case(data):
    """Recursively convert all keys in a dictionary (or list of dictionaries) from snake_case to camelCase."""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = snake_to_camel(k)
            new_dict[new_key] = convert_keys_to_camel_case(v)  # Recursively apply to nested dictionaries/lists
        return new_dict
    elif isinstance(data, list):
        return [convert_keys_to_camel_case(item) for item in data]
    else:
        return data
