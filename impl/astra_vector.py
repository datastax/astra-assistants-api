import asyncio
import json
import logging
import os
import random
import tempfile
import time
from datetime import datetime
from random import randint
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import requests
from cassandra.concurrent import execute_concurrent, execute_concurrent_with_args
from fastapi import HTTPException

from cassandra import ConsistencyLevel, Unauthorized
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, DriverException, NoHostAvailable
from cassandra.policies import RetryPolicy
from cassandra.query import (
    UNSET_VALUE,
    SimpleStatement,
    ValueSequence,
    dict_factory,
    named_tuple_factory, PreparedStatement,
)
from openai.types.beta.threads.runs import RunStep, ToolCallsStepDetails
from pydantic import BaseModel, Field

from impl.model.assistant_object import AssistantObject
from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from impl.model.message_object import MessageObject
from impl.model.open_ai_file import OpenAIFile
from impl.model.run_object import RunObject
from impl.models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
)
from impl.services.inference_utils import get_embeddings
from openapi_server.models.message_content_text_object import MessageContentTextObject
from openapi_server.models.message_content_text_object_text import MessageContentTextObjectText
from openapi_server.models.run_object_required_action import RunObjectRequiredAction
from openapi_server.models.thread_object import ThreadObject



# Create a logger for this module.
logger = logging.getLogger(__name__)

CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "assistant_api")
CASSANDRA_USER = "token"
DEFAULT_DB_NAME = "assistant_api_db"

TOKEN_AUTH_FAILURE_MESSAGE = """
Unauthorized to connect to AstraDB. Please ensure you're passing a token starting with `ASTRACS:...` from https://astra.datastax.com and ensure it has the right scope.
"""


class Payload(BaseModel):
    args: Dict[str, Any]


class HandledResponse(BaseModel):
    status_code: int = Field(alias="status_code"),
    detail: str = Field(alias="detail"),
    retryable: bool = Field(alias="retryable"),


class AstraVectorDataStore:
    def __init__(self) -> None:
        # no longer create session on init, since we need a session per user
        # but maybe do it in local mode
        # self.client = self.create_db_client()
        pass

    async def create_db_client(self, token, dbid):
        self.client = CassandraClient(token, dbid)
        await self.client.async_setup()
        return self.client

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        query_results: List[QueryResult] = []
        for query in queries:
            if not query.top_k:
                query.top_k = 10
            data = await self.client.runQuery(query)

            if data is None:
                query_results.append(QueryResult(query=query.query, results=[]))
            else:
                results: List[DocumentChunkWithScore] = []
                for row in data:
                    doc_metadata = DocumentChunkMetadata(
                        source=row.source,
                        source_id=row.source_id,
                        document_id=row.document_id,
                        url=row.url,
                        created_at=row.created_at.isoformat(),
                        author=row.author,
                    )
                    document_chunk = DocumentChunkWithScore(
                        id=row.id,
                        text=row.content,
                        # TODO: add embedding to the response ?
                        # embedding=row.embedding,
                        score=float(row.score),
                        # score=float(1),
                        metadata=doc_metadata,
                    )
                    results.append(document_chunk)
                query_results.append(QueryResult(query=query.query, results=results))
        return query_results

    async def delete(
            self,
            ids: Optional[List[str]] = None,
            filter: Optional[DocumentMetadataFilter] = None,
            delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything in the datastore.
        Multiple parameters can be used at once.
        Returns whether the operation was successful.
        """
        if delete_all:
            try:
                await self.client._delete("documents", delete_all)
            except:
                return False
        elif ids:
            try:
                await self.client._delete_in("documents", "document_id", ids)
            except:
                return False
        elif filter:
            raise NotImplementedError
            # try:
            #    await self.client._delete_by_filters("documents", filter)
            # except:
            #    return False
        return True

    async def setupSession(self, token, dbid):
        self.dbid = dbid
        self.client = await self.create_db_client(token, dbid)
        return self.client


class VectorRetryPolicy(RetryPolicy):
    def on_read_timeout(
            self,
            query,
            consistency,
            required_responses,
            received_responses,
            data_retrieved,
            retry_num,
    ):
        if retry_num < 3:
            logger.info(f"retrying timeout {retry_num}")
            logger.info(f"query: {query}")
            return RetryPolicy.RETRY, consistency  # return a tuple
        else:
            return RetryPolicy.RETHROW, consistency

    def on_request_error(self, query, consistency, error, retry_num):
        if retry_num < 3:
            logger.info(f"retrying error {retry_num}")
            logger.info(f"query: {query}")
            return RetryPolicy.RETRY, consistency  # return a tuple
        else:
            return RetryPolicy.RETHROW, consistency

    def on_unavailable(
            self, query, consistency, required_replicas, alive_replicas, retry_num
    ):
        return RetryPolicy.RETHROW, consistency  # return a tuple

    def on_write_timeout(
            self,
            query,
            consistency,
            write_type,
            required_responses,
            received_responses,
            retry_num,
    ):
        return RetryPolicy.RETHROW, consistency  # return a tuple


class CassandraClient:

    def __init__(self, token, dbid=None) -> None:
        self.token = token
        self.dbid = dbid
        self.session = None  # Initialize session to None

    async def async_setup(self):
        if self.dbid is None:
            await self.get_or_create_db()

        # Attempt to connect synchronously (assuming connect is a sync method)
        session = self.connect()
        if session:
            self.session = session
            # Perform async table creation
            await self.create_table()
        else:
            raise Exception("Failed to connect to AstraDB")

    async def get_or_create_db(self):
        logger.info("get or create db")
        token = self.token

        url = f"https://api.astra.datastax.com/v2/databases"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers)
        handled_response = self.handle_response_errors(response)
        if handled_response is not None:
            logger.error(f"Failed to create AstraDBs {handled_response.detail}")
            raise HTTPException(detail=handled_response.detail, status_code=handled_response.status_code)
        response = response.json()

        logger.debug(f"{len(response) = }")
        logger.debug(f"{response = }")
        is_terminating = False
        for database in response:
            if database["info"]["name"] == DEFAULT_DB_NAME:
                status = database["status"]
                if status == "ACTIVE":
                    self.dbid = database["id"]
                    return
                if status == "HIBERNATED":
                    # running make keyspace will wake it up
                    self.dbid = database["id"]
                    await self.make_keyspace()
                    logger.info(f"Waking up hibernated db {database['id']}")
                    time.sleep(5)
                    await self.get_or_create_db()
                    return
                if status == "TERMINATING":
                    is_terminating = True
                else:
                    time.sleep(5)
                    logger.info(f"Waiting for {database['id']} to come up")
                    await self.get_or_create_db()
                    return

        if is_terminating:
            time.sleep(5)
            logger.info(f"Waiting for {database['id']} to terminate")
            await self.get_or_create_db()
            return

        logger.info(f"Creating db for {token}")
        # it's not there so create it
        payload = {
            "name": DEFAULT_DB_NAME,
            "tier": "serverless",
            "cloudProvider": "GCP",
            "keyspace": CASSANDRA_KEYSPACE,
            "region": "us-east1",
            "capacityUnits": 1,
            "user": "token",
            "password": token,
            "dbType": "vector",
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        logger.info(f"{url = }")
        logger.info(f"{headers = }")
        logger.info(f"{payload = }")
        response = requests.post(url, headers=headers, json=payload)
        handled_response = self.handle_response_errors(response)
        if handled_response is not None:
            logger.error(f"Failed to create AstraDBs {handled_response.detail}")
            raise HTTPException(detail=handled_response.detail, status_code=handled_response.status_code)
        self.dbid = response.headers.get("location")
        await self.get_or_create_db()
        return

    def handle_response_errors(self, response: requests.Response) -> HandledResponse:
        if response.status_code == 401:
            # Forward the auth error to return from FastAPI
            return HandledResponse(
                status_code=401,
                detail=f"{TOKEN_AUTH_FAILURE_MESSAGE}\nCould not access url: {e.response.url}. Detail: {e.response.text}",
                retryable=False,
            )
        elif response.status_code == 409:
            return HandledResponse(
                status_code=409,
                detail="Conflict",
                retryable=True,
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                # Forward the auth error to return from FastAPI
                return HandledResponse(
                    status_code=401,
                    detail=f"{TOKEN_AUTH_FAILURE_MESSAGE}\nCould not access url: {e.response.url}. Detail: {e.response.text}",
                    retryable=False,
                )
        try:
            response_dict = response.json()
        except json.JSONDecodeError:
            response_dict = None
        if response_dict is not None and "errors" in response_dict:
            # Handle the errors
            errors = response_dict["errors"]
            if errors[0]["message"]:
                if errors[0]["message"] == "JWT not valid":
                    return HandledResponse(
                        status_code=400,
                        detail=TOKEN_AUTH_FAILURE_MESSAGE
                                   + str(response_dict),
                        retryable=False,
                    )
                    # TODO - maybe link to registration page in this error message
                if errors[0]["message"] == "database is not in a valid state to perform requested action":
                    return HandledResponse(
                        status_code=400,
                        detail="database is not in a valid state to perform requested action",
                        retryable=True,
                    )
            logger.error(errors)
            return HandledResponse(
                status_code=400,
                detail=TOKEN_AUTH_FAILURE_MESSAGE
                       + str(response_dict),
                retryable=False,
            )
        else:
            return None


    def get_astra_bundle_url(self):
        dbid = self.dbid
        token = self.token

        # Define the URL
        url = f"https://api.astra.datastax.com/v2/databases/{dbid}/secureBundleURL"

        # Define the headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Define the payload (if any)
        payload = {}

        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        handled_response = self.handle_response_errors(response)
        if handled_response is not None:
            if handled_response.retryable:
                time.sleep(5)
                return self.get_astra_bundle_url()
            else:
                logger.error(f"Failed to get AstraDB bundle URL " + handled_response.detail)
                raise HTTPException(detail=handled_response.detail, status_code=handled_response.status_code)

        return response.json()["downloadURL"]

    def connect(self, retry=False):
        dbid = self.dbid
        token = self.token
        if dbid is not None:
            try:
                # connect to Astra
                bundlepath = f"/tmp/{dbid}.zip"
                if not os.path.exists(bundlepath):
                    url = self.get_astra_bundle_url()
                    if url:
                        # Download the secure connect bundle and extract it
                        r = requests.get(url)
                        with open(bundlepath, "wb") as f:
                            f.write(r.content)
                # Connect to the cluster
                cloud_config = {"secure_connect_bundle": bundlepath}
                auth_provider = PlainTextAuthProvider(CASSANDRA_USER, token)
                cluster = Cluster(
                    cloud=cloud_config, auth_provider=auth_provider, connect_timeout=60
                )
                session = cluster.connect()
                session.default_consistency_level = ConsistencyLevel.LOCAL_QUORUM
                return session
            except Unauthorized as e:
                raise HTTPException(401, f"{TOKEN_AUTH_FAILURE_MESSAGE}: {e}")

            except NoHostAvailable as e:
                for db_url, error in e.errors.items():
                    if isinstance(error, Unauthorized):
                        raise HTTPException(
                            401, f"{TOKEN_AUTH_FAILURE_MESSAGE}: {e}"
                        )
                raise

            except DriverException as e:
                logger.error(e)
                raise HTTPException(400, f"Failed to connect to cluster - the database may be hibernated")

            except Exception as e:
                logger.warning(f"Failed to connect to AstraDB: {e}")
                # sleep and retry
                time.sleep(5)
                if retry:
                    return self.connect(retry=False)
                else:
                    raise
        else:
            raise Exception("Failed to connect to AstraDB")

    async def make_keyspace(self):
        # Define the URL
        url = f"https://api.astra.datastax.com/v2/databases/{self.dbid}/keyspaces/{CASSANDRA_KEYSPACE}"

        # Define the headers
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Define the payload (if any)
        payload = {}

        # Make the POST request asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
        handled_response = self.handle_response_errors(response)
        if handled_response is not None:
            if handled_response.retryable:
                wait_time = random.uniform(5, 20)
                await asyncio.sleep(wait_time)  # Use asyncio.sleep for async sleep
                return await self.make_keyspace()  # Recursively call itself with await
            else:
                logger.error(f"Failed to create AstraDB keyspace" + handled_response.detail)
                raise HTTPException(detail=handled_response.detail, status_code=handled_response.status_code)

    def infer_embedding_dim(self, model, litellm_kwargs) -> int:
        """Dependency to infer embedding dimension based on headers by making a request"""
        # self.infer_embedding_dim_info.embedding_model.replace("-","_").replace(".","_").replace("/","_")
        embedding = get_embeddings(
            texts=["test"],
            model=model,
            **litellm_kwargs
        )
        return len(embedding[0])

    def maybe_alter_file_chunks(self, model, litellm_kwargs):
        # TODO: optimize - check cluster schema metadata to see if this alter is necessary
        dims = self.infer_embedding_dim(model, litellm_kwargs)
        model_string = model.replace("-", "_").replace(".", "_").replace("/", "_")
        try:
            self.session.execute(
                f"""alter TABLE assistant_api.file_chunks ADD embedding_{model_string} VECTOR<float, {dims}>;"""
            )
        except Exception as e:
            logger.warning(f"Exception adding column: {e}")
        try:
            statement = SimpleStatement(
                f"CREATE CUSTOM INDEX IF NOT EXISTS ON {CASSANDRA_KEYSPACE}.file_chunks (embedding_{model_string}) USING 'StorageAttachedIndex';",
                consistency_level=ConsistencyLevel.QUORUM,
            )
            self.session.execute(statement)
        except Exception as e:
            logger.warning(f"Exception adding index for column: {e}")

    async def create_table(self):
        try:
            await self.make_keyspace()

            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.assistants (
                    id text primary key,
                    created_at timestamp,
                    name text,
                    description text,
                    model text,
                    instructions text,
                    tools List<text>,
                    file_ids List<text>,
                    metadata Map<text, text>,
                    object text
            );"""
            )

            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.files(
                    id text primary key,
                    object text,
                    purpose text,
                    created_at timestamp,
                    filename text,
                    format text,
                    bytes int,
                    status text
            );"""
            )
            try:
                self.session.execute(
                    f"""alter TABLE assistant_api.files ADD embedding_model text;"""
                )
            except Exception as e:
                logger.warning(f"alter table attempt: {e}")


            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.file_chunks (
                    file_id text,
                    chunk_id text,
                    content text,
                    created_at timestamp,
                    embedding VECTOR<float, 1536>,
                    PRIMARY KEY ((file_id), chunk_id)
            );"""
            )

            try:
                self.session.execute(
                    f"""alter TABLE assistant_api.file_chunks ADD embedding_openai_text_embedding_ada_002 VECTOR<float, 1536>;"""
                )
            except Exception as e:
                logger.warning(f"alter table attempt: {e}")
            try:
                statement = SimpleStatement(
                    f"CREATE CUSTOM INDEX IF NOT EXISTS ON {CASSANDRA_KEYSPACE}.file_chunks (embedding_openai_text_embedding_ada_002) USING 'StorageAttachedIndex';",
                    consistency_level=ConsistencyLevel.QUORUM,
                )
                self.session.execute(statement)
            except Exception as e:
                logger.warning(f"index creation attempt: {e}")

            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.threads (
                    id text primary key,
                    object text,
                    created_at timestamp,
                    metadata Map<text, text>
            );"""
            )

            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.messages (
                    id text,
                    object text,
                    created_at timestamp,
                    thread_id text,
                    role text,
                    content List<text>,
                    assistant_id text,
                    run_id text,
                    file_ids List<text>,
                    metadata Map<text, text>,
                    PRIMARY KEY ((thread_id), id)
            );"""
            )

            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.runs(
                id text,
                object text,
                created_at timestamp,
                thread_id text,
                assistant_id text,
                status text,
                required_action text,
                last_error text,
                expires_at timestamp,
                started_at timestamp,
                cancelled_at timestamp,
                failed_at timestamp,
                completed_at timestamp,
                model text,
                instructions text,
                tools list<text>,
                file_ids list<text>,
                metadata map<text, text>,
                PRIMARY KEY((thread_id), id)
            ); """
            )

            self.session.execute(
                f"""create table if not exists {CASSANDRA_KEYSPACE}.run_steps(
                id text,
                assistant_id text,
                cancelled_at timestamp,
                completed_at timestamp,
                created_at timestamp,
                expired_at timestamp,
                failed_at timestamp,
                last_error text,
                metadata map<text, text>,
                object text,
                run_id text,
                status text,
                step_details text,
                thread_id text,
                type text,
                usage text,
                PRIMARY KEY((run_id), id)
            ); """
            )


            statement = SimpleStatement(
                f"CREATE CUSTOM INDEX IF NOT EXISTS ON {CASSANDRA_KEYSPACE}.file_chunks (embedding) USING 'StorageAttachedIndex';",
                consistency_level=ConsistencyLevel.QUORUM,
            )
            self.session.execute(statement)

        except Exception as e:
            logger.warning(f"Exception creating table or index: {e}")
            raise e

    def delete_assistant(self, id):
        query_string = f"""
        DELETE FROM {CASSANDRA_KEYSPACE}.assistants WHERE id = ?;  
        """

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        bound = statement.bind((id,))
        self.session.execute(bound)
        return True

    def get_run_step(self, id, run_id):
        query_string = f"""
        SELECT * FROM {CASSANDRA_KEYSPACE}.run_steps WHERE id = ? and run_id = ?;  
        """

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.row_factory = dict_factory
        bound = statement.bind(
            (
                id,
                run_id,
            )
        )
        rows = self.session.execute(bound)
        result = [dict(row) for row in rows]
        if result is None or len(result) == 0:
            return None
        json_rows = result[0]
        self.session.row_factory = named_tuple_factory

        metadata = json_rows["metadata"]
        if metadata is None:
            metadata = {}

        cancelled_at = json_rows["cancelled_at"]
        completed_at = json_rows["completed_at"]
        created_at = json_rows["created_at"]
        expired_at = json_rows["expired_at"]
        failed_at = json_rows["failed_at"]

        if cancelled_at is not None:
            cancelled_at = int(cancelled_at.timestamp() * 1000)
        if completed_at is not None:
            completed_at = int(completed_at.timestamp() * 1000)
        if created_at is not None:
            created_at = int(created_at.timestamp() * 1000)
        if expired_at is not None:
            expired_at = int(expired_at.timestamp() * 1000)
        if failed_at is not None:
            failed_at = int(failed_at.timestamp() * 1000)

        try:
            step_details = ToolCallsStepDetails.parse_raw(json_rows["step_details"])
            run_step = RunStep(
                id=json_rows["id"],
                assistant_id=json_rows["assistant_id"],
                cancelled_at=cancelled_at,
                completed_at=completed_at,
                created_at=created_at,
                expired_at=expired_at,
                failed_at=failed_at,
                last_error=json_rows["last_error"],
                metadata=metadata,
                object=json_rows["object"],
                run_id=json_rows["run_id"],
                status=json_rows["status"],
                step_details=step_details,
                thread_id=json_rows["thread_id"],
                type=json_rows["type"],
                usage=json_rows["usage"],
            )
            return run_step
        except Exception as e:
            logger.error(f"Error parsing run step: {e}")
            raise e

    def get_run(self, id, thread_id):
        query_string = f"""
        SELECT * FROM {CASSANDRA_KEYSPACE}.runs WHERE id = ? and thread_id = ?;  
        """

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.row_factory = dict_factory
        bound = statement.bind(
            (
                id,
                thread_id,
            )
        )
        rows = self.session.execute(bound)
        result = [dict(row) for row in rows]
        if result is None or len(result) == 0:
            return None
        json_rows = result[0]
        self.session.row_factory = named_tuple_factory

        toolsJson = json_rows["tools"]
        tools = []
        if toolsJson is not None:
            for json_string in toolsJson:
                tools.append(AssistantObjectToolsInner.parse_raw(json_string))

        metadata = json_rows["metadata"]
        if metadata is None:
            metadata = {}

        file_ids = json_rows["file_ids"]
        if file_ids is None:
            file_ids = []

        created_at = int(json_rows["created_at"].timestamp() * 1000)
        expires_at = int(json_rows["expires_at"].timestamp() * 1000)
        started_at = int(json_rows["started_at"].timestamp() * 1000)
        cancelled_at = int(json_rows["cancelled_at"].timestamp() * 1000)
        completed_at = int(json_rows["completed_at"].timestamp() * 1000)
        failed_at = int(json_rows["failed_at"].timestamp() * 1000)
        required_action = None

        instructions = json_rows["instructions"]
        if instructions is None:
            instructions = ""

        required_action_object = None
        if required_action is not None:
            required_action_object = RunObjectRequiredAction.parse_raw(required_action)
        run = RunObject(
            id=json_rows["id"],
            object="thread.run",
            created_at=created_at,
            thread_id=json_rows["thread_id"],
            assistant_id=json_rows["assistant_id"],
            status=json_rows["status"],
            required_action=required_action_object,
            last_error=json_rows["last_error"],
            expires_at=expires_at,
            started_at=started_at,
            cancelled_at=cancelled_at,
            failed_at=failed_at,
            completed_at=completed_at,
            model=json_rows["model"],
            instructions=instructions,
            tools=tools,
            file_ids=file_ids,
            metadata=metadata,
        )
        return run

    def get_assistant(self, id):
        logger.info(f"getting assistant from db for id: {id}")
        query_string = f"""
        SELECT * FROM {CASSANDRA_KEYSPACE}.assistants WHERE id = ?;  
        """

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.row_factory = dict_factory
        bound = statement.bind((id,))
        rows = self.session.execute(bound)
        result = [dict(row) for row in rows]
        if result is None or len(result) == 0:
            return None
        json_row = result[0]
        logger.info(f"fetched this row: {json_row}")
        self.session.row_factory = named_tuple_factory

        toolsJson = json_row["tools"]
        tools = []
        if toolsJson is not None:
            for json_string in toolsJson:
                tools.append(AssistantObjectToolsInner.parse_raw(json_string))

        metadata = json_row["metadata"]
        if metadata is None:
            metadata = {}

        file_ids = json_row["file_ids"]
        if file_ids is None:
            file_ids = []

        description = json_row["description"]
        if description is None:
            description = ""

        name = json_row["name"]
        if name is None:
            name = ""

        model = json_row["model"]
        if model is None:
            model = ""

        instructions = json_row["instructions"]
        if instructions is None:
            instructions = ""

        created_at = int(json_row["created_at"].timestamp() * 1000)
        assistant = AssistantObject(
            id=id,
            created_at=created_at,
            name=name,
            description=description,
            model=model,
            instructions=instructions,
            tools=tools,
            file_ids=file_ids,
            metadata=metadata,
            object=json_row["object"],
        )

        logger.info(f"parsed assistant from row: {assistant}")
        return assistant


    def delete_by_pk(self, key, value, table):
        query_string = f"""
        DELETE FROM {CASSANDRA_KEYSPACE}.{table} WHERE {key} = ?;  
        """

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        bound = statement.bind((value,))
        self.session.execute(bound)
        return True


    def delete_by_pks(self, keys, values, table):
        query_string = f"DELETE FROM {CASSANDRA_KEYSPACE}.{table} WHERE "
        i = 0
        for key in keys:
            query_string += f"{key} = ?"
            if i < len(keys) - 1:
                query_string += " AND "
            i += 1

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        bound = statement.bind(values)
        self.session.execute(bound)
        return True


    def update_run_status(self, id, thread_id, status):
        query_string = f"""
        UPDATE {CASSANDRA_KEYSPACE}.runs SET status = ? WHERE id = ? and thread_id = ?;  
        """

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        bound = statement.bind(
            (
                status,
                id,
                thread_id,
            )
        )
        self.session.execute(bound)
        return True


    def upsert_run_step(self, run_step : RunStep):
        query_string = f"""insert into {CASSANDRA_KEYSPACE}.run_steps(
            id,
            assistant_id,
            cancelled_at,
            completed_at,
            created_at,
            expired_at,
            failed_at,
            last_error,
            metadata,
            object,
            run_id,
            status,
            step_details,
            thread_id,
            type,
            usage
            ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        );"""
        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM

        id = run_step.id
        assistant_id = run_step.assistant_id
        cancelled_at = run_step.cancelled_at
        completed_at = run_step.completed_at
        created_at = run_step.created_at
        expired_at = run_step.expired_at
        failed_at = run_step.failed_at
        last_error = run_step.last_error
        metadata = run_step.metadata
        object = run_step.object
        run_id = run_step.run_id
        status = run_step.status
        step_details = run_step.step_details
        thread_id = run_step.thread_id
        type = run_step.type
        usage = run_step.usage

        self.session.execute(
            statement,
            (
                id,
                assistant_id,
                cancelled_at,
                completed_at,
                created_at,
                expired_at,
                failed_at,
                last_error,
                metadata,
                object,
                run_id,
                status,
                step_details.json(),
                thread_id,
                type,
                usage
            ),
        )


    def upsert_run(
            self,
            id,
            object,
            created_at,
            thread_id,
            assistant_id,
            status,
            required_action,
            last_error,
            expires_at,
            started_at,
            cancelled_at,
            failed_at,
            completed_at,
            model,
            instructions,
            tools,
            file_ids,
            metadata,
    ):
        query_string = f"""insert into {CASSANDRA_KEYSPACE}.runs(
                    id,
                    object,
                    created_at,
                    thread_id,
                    assistant_id,
                    status,
                    required_action,
                    last_error,
                    expires_at,
                    started_at,
                    cancelled_at,
                    failed_at,
                    completed_at,
                    model,
                    instructions,
                    tools,
                    file_ids,
                    metadata
            ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            );"""
        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM

        toolsJson = []
        for tool in tools:
            toolsJson.append(tool.json())

        self.session.execute(
            statement,
            (
                id,
                object,
                created_at,
                thread_id,
                assistant_id,
                status,
                required_action,
                last_error,
                expires_at,
                started_at,
                cancelled_at,
                failed_at,
                completed_at,
                model,
                instructions,
                toolsJson,
                file_ids,
                metadata,
            ),
        )

        if instructions is None:
            instructions = ""

        required_action_object = None
        if required_action is not None:
            required_action_object = RunObjectRequiredAction.parse_raw(required_action)
        # TODO add support for RunCompletionUsage
        return RunObject(
            id=id,
            object=object,
            created_at=created_at,
            thread_id=thread_id,
            assistant_id=assistant_id,
            status=status,
            required_action=required_action_object,
            last_error=last_error,
            expires_at=expires_at,
            started_at=started_at,
            cancelled_at=cancelled_at,
            failed_at=failed_at,
            completed_at=completed_at,
            model=model,
            instructions=instructions,
            tools=tools,
            file_ids=file_ids,
            metadata=metadata,
            usage=None,
        )

    def upsert_message(
            self,
            id,
            object,
            created_at,
            thread_id,
            role,
            content,
            assistant_id,
            run_id,
            file_ids,
            metadata,
    ):
        if created_at is None:
            created_at = UNSET_VALUE
        if role is None:
            role = UNSET_VALUE
        if assistant_id is None:
            assistant_id = UNSET_VALUE
        if run_id is None:
            run_id = UNSET_VALUE
        if file_ids is None:
            file_ids = UNSET_VALUE
        if metadata is None:
            metadata = {}

        query_string = f"""insert into {CASSANDRA_KEYSPACE}.messages(
                    id,
                    object,
                    created_at,
                    thread_id,
                    role,
                    content,
                    assistant_id,
                    run_id,
                    file_ids,
                    metadata
            ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            );"""
        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.execute(
            statement,
            (
                id,
                object,
                created_at,
                thread_id,
                role,
                content,
                assistant_id,
                run_id,
                file_ids,
                metadata,
            ),
        )

        message = self.get_message(thread_id, id)
        logger.info(f"upserted message: {message}")
        return message

    def get_message(self, thread_id, message_id):
        rows = self.selectFromTableByPK(table="messages", partitionKeys=["thread_id", "id"],
                                        args={"thread_id": thread_id, "id": message_id})
        if len(rows) == 0:
            raise HTTPException(status_code=404, detail=f"Message not found {thread_id} {message_id}")

        row = rows[0]

        raw_content = row['content']
        content = []
        if raw_content is not None and len(raw_content) > 0:
            text = MessageContentTextObjectText(value=raw_content[0], annotations=[])
            content = [MessageContentTextObject(text=text, type="text")]

        metadata = row['metadata']
        if metadata is None:
            metadata = {}

        file_ids = row['file_ids']
        if file_ids is None:
            file_ids = []

        created_at = int(row["created_at"].timestamp() * 1000)
        message_object = MessageObject(
            id=row['id'],
            object=row['object'],
            created_at=created_at,
            thread_id=row['thread_id'],
            role=row['role'],
            content=content,
            assistant_id=row['assistant_id'],
            run_id=row['run_id'],
            file_ids=file_ids,
            metadata=metadata
        )
        return message_object

    def upsert_content_only_file(
            self, id, created_at, object, purpose, filename, format, bytes, content,
    ):
        self.upsert_chunks_content_only(id, content, created_at)
        status = "uploaded"
        query_string = f"""insert into {CASSANDRA_KEYSPACE}.files (
                    id,
                    object,
                    purpose,
                    created_at,
                    filename,
                    format,
                    bytes,
                    status
            ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?
            );"""

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.execute(
            statement,
            (id, object, purpose, created_at, filename, format, bytes, status),
        )
        file = OpenAIFile(
            id=id,
            object=object,
            purpose=purpose,
            created_at=created_at,
            filename=filename,
            format=format,
            bytes=bytes,
            status=status,
        )
        return file

    def upsert_file(
            self, id, created_at, object, purpose, filename, format, bytes, chunks, embedding_model, **litellm_kwargs,
    ):
        self.upsert_chunks(chunks, embedding_model, **litellm_kwargs)
        status = "processed"

        query_string = f"""insert into {CASSANDRA_KEYSPACE}.files (
                    id,
                    object,
                    purpose,
                    created_at,
                    filename,
                    format,
                    bytes,
                    status,
                    embedding_model
            ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?
            );"""

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.execute(
            statement,
            (id, object, purpose, created_at, filename, format, bytes, status, embedding_model),
        )
        file = OpenAIFile(
            id=id,
            object=object,
            purpose=purpose,
            created_at=created_at,
            filename=filename,
            format=format,
            bytes=bytes,
            status=status,
            embedding_model=embedding_model,
        )
        return file

    def upsert_thread(
            self,
            id,
            object,
            created_at,
            metadata
    ):
        query_string = f"""insert into {CASSANDRA_KEYSPACE}.threads (
                    id,
                    object,
                    created_at,
                    metadata
            ) VALUES (
            ?, ?, ?, ?
            );"""

        if object is None:
            object = UNSET_VALUE

        if created_at is None:
            created_at = UNSET_VALUE

        if metadata is None:
            metadata = UNSET_VALUE

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.execute(
            statement,
            (
                id,
                object,
                created_at,
                metadata
            ),
        )
        return self.get_thread(id)

    def get_thread(self, id):
        rows = self.selectFromTableByPK(table="threads", partitionKeys=["id"], args={"id": id})
        if rows is not None and len(rows) > 0:
            row = rows[0]
            created_at = row["created_at"]
            if created_at is not None:
                created_at = int(created_at.timestamp() * 1000)

            metadata = row["metadata"]
            if metadata is None:
                metadata = {}

            return ThreadObject(
                id=row["id"],
                object=row["object"],
                created_at=created_at,
                metadata=metadata
            )
        else:
            raise HTTPException(status_code=404, detail=f"Thread not found with id {id}")

    def upsert_assistant(
            self,
            id,
            created_at,
            name,
            description,
            model,
            instructions,
            tools,
            file_ids,
            metadata,
            object,
    ):
        # TODO: figure out how to parse tools
        logger.info(f"going to upsert assistant with id: {id} and model:{model}")
        query_string = f"""insert into {CASSANDRA_KEYSPACE}.assistants (
                    id,
                    created_at,
                    name,
                    description,
                    model,
                    instructions,
                    file_ids,
                    metadata,
                    object,
                    tools
            ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ? 
            );"""

        if name is None:
            name = UNSET_VALUE

        if instructions is None:
            instructions = UNSET_VALUE

        if model is None:
            model = UNSET_VALUE

        toolsJson = []
        for tool in tools:
            toolsJson.append(tool.json())

        statement = self.session.prepare(query_string)
        statement.consistency_level = ConsistencyLevel.QUORUM
        try:
            response = self.session.execute(
                statement,
                (
                    id,
                    created_at,
                    name,
                    description,
                    model,
                    instructions,
                    file_ids,
                    metadata,
                    object,
                    toolsJson
                ),
            )
        except Exception as e:
            logger.error(f"failed to upsert assistant: {id} {e}")
            raise e

    def __del__(self):
        # close the connection when the client is destroyed
        self.session.shutdown()

    # TODO: make these async
    def selectAllFromTable(self, table):
        queryString = f"""SELECT * FROM {CASSANDRA_KEYSPACE}.{table} limit 1000"""
        statement = self.session.prepare(queryString)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.row_factory = dict_factory
        rows = self.session.execute(statement)
        json_rows = [dict(row) for row in rows]
        self.session.row_factory = named_tuple_factory
        return json_rows

    def selectFromTableByPK(self, table, partitionKeys, args, limit=None, order=None):
        limitString = ""
        if limit is not None:
            limitString = f"limit {limit}"
        queryString = f"""SELECT * FROM {CASSANDRA_KEYSPACE}.{table} WHERE """
        partitionKeyValues = []
        for column in partitionKeys:
            # TODO support other types (single quotes are for strings)
            queryString += f"{column} = ? AND "
            partitionKeyValues.append(args[column])
        # remove the last AND
        queryString = queryString[:-4]
        if order is not None:
            i = 0
            for key, value in order.items():
                if i == 0:
                    queryString += f" ORDER BY "
                queryString += f"{key} {value} ,"
                i += 1
            # remove the last comma
            queryString = queryString[:-1]
            # TODO: figure out how to do a migration to get rid of ALLOW FILTERING
            queryString = queryString + limitString + " ALLOW FILTERING;"
        else:
            queryString = queryString + limitString
        statement = self.session.prepare(queryString)
        statement.consistency_level = ConsistencyLevel.QUORUM
        preparedStatement = statement.bind(partitionKeyValues)
        self.session.row_factory = dict_factory
        rows = self.session.execute(preparedStatement)
        json_rows = [dict(row) for row in rows]
        self.session.row_factory = named_tuple_factory
        return json_rows

    async def runQuery(self, query):
        filters = ""
        if query.filter:
            filter = query.filter
            # TODO, change to WHERE when syntax changes
            filters = " WHERE "
            if filter.document_id:
                filters += f" document_id = '{filter.document_id}' AND"
            if filter.source:
                filters += f" source = '{filter.source}' AND"
            if filter.source_id:
                filters += f" source_id = '{filter.source_id}' AND"
            if filter.author:
                filters += f" author = '{filter.author}' AND"
            if filter.start_date:
                filters += f" created_at >= '{filter.start_date}' AND"
            if filter.end_date:
                filters += f" created_at <= '{filter.end_date}' AND"
            filters = filters[:-4]

        try:
            self.session.row_factory = named_tuple_factory
            queryString = f"""SELECT id, content, embedding, document_id, 
            source, source_id, url, author, created_at, 
            similarity_cosine(?, embedding) as score
            from {CASSANDRA_KEYSPACE}.documents {filters} 
            ORDER BY embedding ann of {query.embedding} 
            LIMIT {query.top_k};"""
            statement = self.session.prepare(queryString)
            statement.consistency_level = ConsistencyLevel.QUORUM
            boundStatement = statement.bind([query.embedding])
            resultset = self.session.execute(boundStatement)
            return resultset

        except Exception as e:
            logger.warning(f"Exception during query (retrying): {e}")
            raise
            # TODO: Do we want this instead?
            # sleep 10 seconds and retry
            # time.sleep(10)
            # await self.runQuery(query)

    def upsert_chunks(self, chunks: Dict[str, List[DocumentChunk]], model: str, **litellm_kwargs: Any) -> List[str]:
        """
        Takes in a dict of document_ids to list of document chunks and inserts them into the database.
        Return a list of document ids.

                    file_id text,
                    chunk_id text,
                    content text,
                    created_at timestamp,
                    embedding VECTOR<float,EMB_SIZE>
        """
        statements_and_params = []
        for document_id, document_chunks in chunks.items():
            for chunk in document_chunks:
                json = {
                    "file_id": document_id,
                    "chunk_id": chunk.id,
                    "content": chunk.text,
                    f"embedding": chunk.embedding,
                }
                # self.do_upsert_chunks(json, model, **litellm_kwargs)
                statements_and_params = self.queue_up_chunks(statements_and_params, json, model, **litellm_kwargs)
        self.upsert_chunks_concurrently(statements_and_params)

    def load_auth_file(self, file_id):
        rows = self.selectFromTableByPK(table="file_chunks", partitionKeys=["file_id", "chunk_id"],
                                        args={"file_id": file_id, "chunk_id": "0"})
        content = rows[0]["content"]

        # write content to tmp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file.write(content.encode("utf-8"))
        return tmp_file.name

    def upsert_chunks_content_only(self, file_id, content, created_at):
        table = "file_chunks"
        try:
            chunk_id = "0"

            queryString = f"""
                        insert into {CASSANDRA_KEYSPACE}.{table} 
                        (file_id, chunk_id, content, created_at) 
                        VALUES (%s, %s, %s, %s);
                    """
            statement = SimpleStatement(
                queryString, consistency_level=ConsistencyLevel.QUORUM
            )

            self.session.execute(
                statement,
                (
                    file_id,
                    chunk_id,
                    content,
                    created_at
                ),
            )
        except Exception as e:
            logger.warning(f"Exception inserting into table: {e}")
            raise

    def queue_up_chunks(self, statements_and_params: [PreparedStatement], json: dict[str, Any], embedding_model,
                        **litellm_kwargs):
        statement = self.make_chunks_statement(embedding_model, json, litellm_kwargs)
        statements_and_params.append(
            (
                statement,
                (
                    json["file_id"],
                    json["chunk_id"],
                    json["content"],
                    json["created_at"],
                    json["embedding"].tolist(),
                )
            )
        )
        return statements_and_params

    def upsert_chunks_concurrently(self, statements_and_params: [SimpleStatement]):
        results = execute_concurrent(
            self.session, statements_and_params, concurrency=100)

        for (success, result) in results:
            if not success:
                logger.warning(f"Exception inserting into table: {result}")
                raise

    def do_upsert_chunks(self, json: dict[str, Any], embedding_model, **litellm_kwargs):
        statement = self.make_chunks_statement(embedding_model, json, litellm_kwargs)

        try:
            self.session.execute(
                statement,
                (
                    json["file_id"],
                    json["chunk_id"],
                    json["content"],
                    json["created_at"],
                    json["embedding"].tolist(),
                ),
            )
        except Exception as e:
            logger.warning(f"Exception inserting into table: {e}")
            raise

    def make_chunks_statement(self, embedding_model, json, litellm_kwargs):
        """
            Takes in a list of documents and inserts them into the table.
                "file_id": document_id,
                "chunk_id": chunk.id,
                "content": chunk.text,
                "created_at" : epoch,
                "embedding": chunk.embedding,
            """
        table = "file_chunks"
        if not json.get("created_at"):
            json["created_at"] = datetime.now()
        else:
            json["created_at"] = json["created_at"][0]
        json["embedding"] = np.array(json["embedding"])
        columns = self.get_columns(table)
        model_string = embedding_model.replace("-", "_").replace(".", "_").replace("/", "_")
        missing = True
        for column in columns:
            if column["column_name"] == f"embedding_{model_string}":
                missing = False
                break

        if missing:
            self.maybe_alter_file_chunks(embedding_model, litellm_kwargs)

        queryString = f"""
            insert into {CASSANDRA_KEYSPACE}.{table} 
            (file_id, chunk_id, content, created_at, embedding_{model_string}) 
            VALUES (?, ?, ?, ?, ?);
        """
        statement = self.session.prepare(
            queryString
        )
        statement.consistency_level = ConsistencyLevel.QUORUM
        return statement

    async def _delete_by_filters(self, table: str, filter: DocumentMetadataFilter):
        """
        Deletes rows in the table that match the filter.
        """

        filters = "WHERE"
        if filter.document_id:
            filters += f" document_id = '{filter.document_id}' AND"
        if filter.source:
            filters += f" source = '{filter.source}' AND"
        if filter.source_id:
            filters += f" source_id = '{filter.source_id}' AND"
        if filter.author:
            filters += f" author = '{filter.author}' AND"
        if filter.start_date:
            filters += f" created_at >= '{filter.start_date}' AND"
        if filter.end_date:
            filters += f" created_at <= '{filter.end_date}' AND"
        filters = filters[:-4]
        self.session.execute(f"DELETE FROM {table} {filters}")

    async def _delete(self, table, delete_all):
        if delete_all:
            self.session.execute(f"TRUNCATE TABLE {CASSANDRA_KEYSPACE}.{table}")
        else:
            raise NotImplementedError

    def truncate_table(self, table) -> None:
        self.session.execute(f"TRUNCATE TABLE {CASSANDRA_KEYSPACE}.{table}")

    async def _delete_in(self, table: str, column: str, doc_ids: List[str]):
        """
        Deletes rows in the table that match the ids.
        """
        try:
            query = (
                f"SELECT id FROM {CASSANDRA_KEYSPACE}.{table} WHERE {column} IN (%s)"
            )
            statement = SimpleStatement(
                query, consistency_level=ConsistencyLevel.QUORUM
            )
            parameters = ValueSequence(doc_ids)
            rows = self.session.execute(statement, parameters)

            ids = ValueSequence([row.id for row in rows])

            if len(ids) == 0:
                logger.info(f"DocIds not found : {doc_ids}")
                return

            self.session.execute(
                f"DELETE FROM {CASSANDRA_KEYSPACE}.{table} WHERE id IN (%s)", ids
            )
        except Exception as e:
            logger.warning(f"Exception deleting from table: {e}")
            raise

    async def get_tables_async(self, keyspace):
        return self.get_tables(keyspace)

    async def get_keyspaces_async(self):
        return self.get_keyspaces()

    def get_keyspaces(self):
        queryString = "SELECT DISTINCT keyspace_name FROM system_schema.tables"
        statement = self.session.prepare(queryString)
        statement.consistency_level = ConsistencyLevel.QUORUM
        rows = self.session.execute(statement)
        keyspaces = [row.keyspace_name for row in rows]
        keyspaces.remove("system_auth")
        keyspaces.remove("system_schema")
        keyspaces.remove("system")
        keyspaces.remove("data_endpoint_auth")
        keyspaces.remove("system_traces")
        keyspaces.remove("datastax_sla")
        return keyspaces

    async def get_columns_async(self, keyspace, table):
        return self.get_columns(keyspace, table)

    def get_tables(self, keyspace):
        queryString = f"""SELECT table_name FROM system_schema.tables WHERE keyspace_name='{keyspace}'"""
        statement = self.session.prepare(queryString)
        statement.consistency_level = ConsistencyLevel.QUORUM
        rows = self.session.execute(statement)
        tables = [row.table_name for row in rows]
        if keyspace == CASSANDRA_KEYSPACE and "documents" in tables:
            tables.remove("documents")
        return tables

    def get_indexes(self, table):
        queryString = f"""
        SELECT options FROM system_schema.indexes 
        WHERE keyspace_name='{CASSANDRA_KEYSPACE}' 
        and table_name = '{table}'
        and kind = 'CUSTOM' ALLOW FILTERING;
        """
        statement = self.session.prepare(queryString)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.row_factory = dict_factory
        rows = self.session.execute(statement)
        indexes = [row["options"] for row in rows]
        indexed_columns = []
        for index in indexes:
            options = dict(index)
            # TODO - extract whether it's dot vs cosine for vector types
            if "StorageAttachedIndex" in options["class_name"]:
                indexed_columns.append(options["target"])
        self.session.row_factory = named_tuple_factory
        return indexed_columns

    def get_columns(self, table):
        queryString = f"""select column_name, kind, type, position from system_schema."columns" WHERE keyspace_name = '{CASSANDRA_KEYSPACE}' and table_name = '{table}';"""
        statement = self.session.prepare(queryString)
        statement.consistency_level = ConsistencyLevel.QUORUM
        self.session.row_factory = dict_factory
        rows = self.session.execute(statement)
        json_rows = [dict(row) for row in rows]
        self.session.row_factory = named_tuple_factory
        return json_rows

    def annSearch(
            self,
            table,
            vector_index_column,
            search_string,
            litellm_kwargs: Dict[str, Any],
            embedding_model: str,
            embedding_api_key: str,
            partitions,
            # Todo: make this configurable or based on model token limit
            limit=20,
    ):

        queryString = f"SELECT "
        embeddings = []
        columns = self.get_columns(table)
        indexes = self.get_indexes(table)

        model_string = embedding_model.replace("-", "_").replace(".", "_").replace("/", "_")

        missing = True
        for column in columns:
            if column["column_name"] == f"embedding_{model_string}":
                missing = False
                break

        if missing:
            raise HTTPException(f"Missing file embeddings for {model_string}, please resubmit the file.")

        # TODO: we may have to check if there aren't any populated embeddings for the model as well

        vector_index_column = vector_index_column + "_" + model_string

        for column in columns:
            if (
                    vector_index_column in indexes
                    and column["column_name"] == vector_index_column
            ):
                litellm_kwargs_embedding = litellm_kwargs.copy()
                litellm_kwargs_embedding["api_key"] = embedding_api_key
                embeddings.append(get_embeddings([search_string], model=embedding_model, **litellm_kwargs_embedding)[0])
                queryString += f"similarity_cosine(?, {column['column_name']}) as score, "
            elif 'embedding' not in column['column_name'] and column['column_name'] != 'created_at':
                queryString += f"{column['column_name']}, "
        queryString = queryString[:-2]

        queryString += f""" FROM {CASSANDRA_KEYSPACE}.{table} """
        if len(partitions) > 1:
            return self.handle_multiple_partitions(embeddings, limit, queryString, vector_index_column, partitions)
        else:
            return self.finish_ann_query_and_get_json(embeddings, limit, queryString, vector_index_column, partitions)

    # TODO: make this async and or fix the data model
    def handle_multiple_partitions(self, embeddings, limit, queryString, vector_index_column, partitions):
        json_rows = []

        queryString += f"WHERE file_id = ? "
        queryString += f"ORDER BY "
        queryString += f"""
                {vector_index_column} ann of ?
                """
        # TODO make limit configurable
        queryString += f"LIMIT {limit}"
        statement = self.session.prepare(queryString)
        statement.retry_policy = VectorRetryPolicy()
        statement.consistency_level = ConsistencyLevel.LOCAL_ONE
        self.session.row_factory = dict_factory
        parameters = []
        for partition in partitions:
            parameters.append([embeddings[0], partition, embeddings[0]])
        rows = execute_concurrent_with_args(self.session, statement, parameters, concurrency=100)

        json_rows = []
        for (success, result) in rows:
            if not success:
                logger.error(f"problem with async query: {result}")  # result will be an Exception
            else:
                for row in result:
                    json_rows.append(dict(row))
        json_rows = [
            {k: v for k, v in row.items() if k not in [vector_index_column]}
            for row in json_rows
        ]
        self.session.row_factory = named_tuple_factory

        #sort json_rows by score
        json_rows = sorted(json_rows, key=lambda x: x["score"], reverse=True)

        #trim limit
        json_rows = json_rows[:limit]

        return json_rows

    def finish_ann_query_and_get_json(self, embeddings, limit, queryString, vector_index_column, partitions):
        queryString += f"WHERE file_id = '{partitions[0]}' "
        queryString += f"ORDER BY "
        queryString += f"""
                {vector_index_column} ann of ?
                """
        # TODO make limit configurable
        queryString += f"LIMIT {limit}"
        statement = self.session.prepare(queryString)
        statement.retry_policy = VectorRetryPolicy()
        statement.consistency_level = ConsistencyLevel.LOCAL_ONE
        boundStatement = statement.bind([embeddings[0], embeddings[0]])
        self.session.row_factory = dict_factory
        json_rows = self.execute_and_get_json(boundStatement, vector_index_column)
        return json_rows

    def execute_and_get_json(self, boundStatement, vector_index_column, tries=0):
        try:
            rows = self.session.execute(boundStatement, timeout=100)
            json_rows = [dict(row) for row in rows]
            json_rows = [
                {k: v for k, v in row.items() if k not in [vector_index_column]}
                for row in json_rows
            ]
            self.session.row_factory = named_tuple_factory
            return json_rows
        except Exception as e:
            if tries < 3:
                logger.warning(f"Exception during query (retrying): {e}")
                time.sleep(1)
                return self.execute_and_get_json(boundStatement, vector_index_column, tries + 1)
            else:
                raise HTTPException(status_code=500, detail=f"Exception during recall")
