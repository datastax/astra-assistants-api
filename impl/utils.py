import base64
import datetime
import hashlib
import json
import logging
import secrets
import traceback
from typing import Type, Dict, Any, List, get_origin, Annotated, get_args, Union

import pydantic
from fastapi import HTTPException
from pydantic import BaseModel

from impl.astra_vector import CassandraClient

logger = logging.getLogger(__name__)

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


async def store_object(astradb: CassandraClient, obj: BaseModel, target_class: Type[BaseModel], table_name: str,
                       extra_fields: Dict[str, Any]):
    try:
        combined_fields = combine_fields(extra_fields, obj, target_class)
        combined_obj: target_class = target_class.construct(**combined_fields)

        # TODO is there a better way to do this
        # flatten nested objects into json
        obj_dict = combined_obj.to_dict()
        for key, value in obj_dict.items():
            if isinstance(value, list):
                if len(value) == 0:
                    obj_dict[key] = None
                else:
                    for i in range(len(value)):
                        if isinstance(value[i], list):
                            obj_dict[key][i] = json.dumps(value[i])
                        if isinstance(value[i], dict):
                            obj_dict[key][i] = json.dumps(value[i])
            # special handling for metadata column which is actually stored as a map in cassandra
            # (other objects are json strings)
            if key != "metadata" and isinstance(value, dict):
                obj_dict[key] = json.dumps(value)

        astradb.upsert_table_from_dict(table_name=table_name, obj=obj_dict)
        return combined_obj
    except Exception as e:
        logger.error(f"store_object failed {e} for table {table_name} and object {obj}, dbid: {astradb.dbid}")
        raise HTTPException(status_code=500, detail=f"Error reading {table_name}: {e}")


def read_object(astradb: CassandraClient, target_class: Type[BaseModel], table_name: str, partition_keys: List[str],
                 args: Dict[str, Any]):
    try:
        objs = read_objects(astradb, target_class, table_name, partition_keys, args)
    except Exception as e:
        logger.error(f"read_object failed {e} for table {table_name}, dbid: {astradb.dbid}")
        logger.error(f"trace: {traceback.format_exc()}")
        raise HTTPException(status_code=404, detail=f"{target_class.__name__} not found.")
    if len(objs) == 0:
        # Maybe pass down name
        logger.warn(f"did not find partition_keys {partition_keys} and args {args} for {target_class.__name__} in table {table_name} for dbid: {astradb.dbid}")
        raise HTTPException(status_code=404, detail=f"{target_class.__name__} not found.")
    return objs[0]



def read_objects(astradb: CassandraClient, target_class: Type[BaseModel], table_name: str, partition_keys: List[str],
                args: Dict[str, Any]):
    obj = None
    try:
        json_objs = astradb.select_from_table_by_pk(table=table_name, partition_keys=partition_keys, args=args)
        if len(json_objs) == 0:
            raise HTTPException(status_code=404, detail=f"{args} not found in table {table_name}.")

        obj_list = []
        for json_obj in json_objs:
            for field_name, field_type in target_class.__fields__.items():
                annotation = field_type.annotation
                if (
                        annotation is not None
                        and json_obj[field_name] is not None
                        and hasattr(annotation, 'from_json')
                ):
                    if 'actual_instance' in annotation.__fields__:
                        try:
                            json_obj[field_name] = annotation(actual_instance=json_obj[field_name])
                        except Exception as e:
                            try:
                                json_obj[field_name] = annotation.from_json(json_obj[field_name])
                            except Exception as e:
                                raise e
                    else:
                        json_obj[field_name] = annotation.from_json(json_obj[field_name])
                elif get_origin(annotation) is list:
                    if json_obj[field_name] is None:
                        json_obj[field_name] = []
                    else:
                        for i in range(len(json_obj[field_name])):
                            if isinstance(json_obj[field_name][i], str):
                                json_obj[field_name][i] = annotation.__args__[0].from_json(json_obj[field_name][i])
                            else:
                                if 'actual_instance' in annotation.__args__[0].__fields__:
                                     json_obj[field_name][i] = annotation.__args__[0](actual_instance=json_obj[field_name][i])
                                else:
                                    logger.error(f"error reading object from {table_name} - {field_name} is an object: {json_obj[field_name][i]}  but {annotation} does not take objects.")
                                    raise HTTPException(status_code=500, detail=f"Error reading {table_name}: {field_name}.")
                elif get_origin(annotation) is Union:
                    if hasattr(get_args(annotation)[0], 'from_json'):
                        if json_obj[field_name] is not None and isinstance(json_obj[field_name], str):
                            if 'actual_instance' in get_args(annotation)[0].__fields__:
                                json_obj[field_name] = get_args(annotation)[0](actual_instance=json_obj[field_name])
                            else:
                                json_obj[field_name] = get_args(annotation)[0].from_json(json_obj[field_name])
                    if get_origin(get_args(annotation)[0]) is Annotated:
                        if get_args(get_args(annotation)[0])[0] is int and isinstance(json_obj[field_name], datetime.datetime):
                            json_obj[field_name] = int(json_obj[field_name].timestamp()*1000)
                elif annotation is int and isinstance(json_obj[field_name], datetime.datetime):
                    json_obj[field_name] = int(json_obj[field_name].timestamp()*1000)
            try:
                obj = target_class(**json_obj)
                obj_list.append(obj)
            except Exception as e:
                if isinstance(e, pydantic.ValidationError):
                    logger.warn(f"ignoring bad object from {table_name} - {e} json_obj: {json_obj}")
                else:
                    raise HTTPException(status_code=500, detail=f"Error reading {table_name}: {e}")
        return obj_list
    except Exception as e:
        if hasattr(e, 'status_code') and e.status_code== 404:
            raise e
        msg = f"read_objects failed {e} for table {table_name}"
        if obj is not None:
            msg += f" last object: {obj}"
        if json_objs is not None:
            msg += f" json_objs: {json_objs}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error reading {table_name}: {e}")


def generate_id(prefix: str, num_bytes=24):
    random_bytes = secrets.token_bytes(num_bytes)
    random_string = base64.urlsafe_b64encode(random_bytes).rstrip(b'=').decode('utf-8')
    generated_id = f"{prefix}_{random_string}"
    logger.info(f"generated id: {generated_id}")
    return generated_id


def generate_id_from_upload_file(upload_file, prefix="file", length=24):
    spooled_file = upload_file.file
    spooled_file.seek(0)
    file_data = upload_file.filename.encode('utf-8') + spooled_file.read()
    sha256_hash = hashlib.sha256(file_data).digest()
    base64_encoded_hash = base64.urlsafe_b64encode(sha256_hash).rstrip(b'=').decode('utf-8')[:length]
    spooled_file.seek(0)

    delim = "_"
    if prefix == "file":
        delim = "-"

    return f"{prefix}{delim}{base64_encoded_hash}"