import json
from typing import Type, Dict, Any, List, get_origin, Annotated, get_args

from pydantic import BaseModel


def map_model(source_instance: BaseModel, target_model_class: Type[BaseModel],
              extra_fields: Dict[str, Any] = {}) -> BaseModel:
    combined_fields = combine_fields(extra_fields, source_instance, target_model_class)

    # Create an instance of the target model class using the extracted and adjusted fields
    return target_model_class(**combined_fields)


def combine_fields(extra_fields, source_instance, target_model_class):
    field_values = {}
    # Iterate over the fields in the target model class
    for field_name, field_type in target_model_class.__fields__.items():
        value = None
        if field_name in source_instance.__fields__:
            value = getattr(source_instance, field_name)
        # extra_fields can override source_instance values
        if field_name in extra_fields:
            value = extra_fields[field_name]

        # Handle Annotated type by extracting the base type
        origin_type = get_origin(field_type.annotation)

        # if origin_type is Annotated:
        #    base_type = get_args(field_type)[0]
        #    origin_type = get_origin(base_type)
        #    if origin_type is None:
        #        origin_type = base_type

        # Check if the field type is a List and the value is None
        if origin_type is list and value is None:
            field_values[field_name] = []
        else:
            field_values[field_name] = value
    # Merge field_values with extra_fields, where extra_fields take precedence
    combined_fields = {**field_values, **extra_fields}
    return combined_fields
