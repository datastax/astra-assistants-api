# Assistants API

A backend implementation of the OpenAI beta Assistants API with support for persistent threads, files, assistants, messages, and more generated from the OpenAPI spec. Compatible with existing OpenAI apps via the OpenAI SDKs with a single line of code:

**Last Updated 11/14/23**:

Here are all the OpenAI endpoints, most of the stateful beta endpoints are implemented and all the simple stateless services are simply proxied to OpenAI:

|Endpoint | Implemented | Stateless / Proxy | Roadmap|
|---------|-------------|-----------------|--------|
|/chat/completions - post | X | X |  | 
|/completions - post | X | X |  | 
|/edits - post | X | X |  | 
|/images/generations - post | X | X |  | 
|/images/edits - post | X | X |  | 
|/images/variations - post | X | X |  | 
|/embeddings - post | X | X |  | 
|/audio/speech - post | X | X |  | 
|/audio/transcriptions - post | X | X |  | 
|/audio/translations - post | X | X |  | 
|/files - get | X | X |  | 
|/files - post | X | X |  | 
|/files/{file_id} - delete | X | X |  | 
|/files/{file_id} - get | X | X |  | 
|/files/{file_id}/content - get | X | X |  | 
|/fine_tuning/jobs - post | X | X |  | 
|/fine_tuning/jobs - get | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id} - get | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id}/events - get | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id}/cancel - post | X | X |  | 
|/fine-tunes - post | X | X |  | 
|/fine-tunes - get | X | X |  | 
|/fine-tunes/{fine_tune_id} - get | X | X |  | 
|/fine-tunes/{fine_tune_id}/cancel - post | X | X |  | 
|/fine-tunes/{fine_tune_id}/events - get | X | X |  | 
|/models - get | X | X |  | 
|/models/{model} - get | X | X |  | 
|/models/{model} - delete | X | X |  | 
|/moderations - post | X | X |  | 
|/assistants - get | X |  |  | 
|/assistants - post | X |  |  | 
|/assistants/{assistant_id} - get | X |  |  | 
|/assistants/{assistant_id} - post | X |  |  | 
|/assistants/{assistant_id} - delete | X |  |  | 
|/threads - post | X |  |  | 
|/threads/{thread_id} - get | |  | X | 
|/threads/{thread_id} - post | |  | X | 
|/threads/{thread_id} - delete | |  | X | 
|/threads/{thread_id}/messages - get | X |  |  | 
|/threads/{thread_id}/messages - post | X |  |  | 
|/threads/{thread_id}/messages/{message_id} - get | |  | X | 
|/threads/{thread_id}/messages/{message_id} - post | |  | X | 
|/threads/runs - post | |  | X | 
|/threads/{thread_id}/runs - get | X |  |  | 
|/threads/{thread_id}/runs - post | X |  |  | 
|/threads/{thread_id}/runs/{run_id} - get | X |  |  | 
|/threads/{thread_id}/runs/{run_id} - post | |  | X | 
|/threads/{thread_id}/runs/{run_id}/submit_tool_outputs - post | |  | X | 
|/threads/{thread_id}/runs/{run_id}/cancel - post | |  | X | 
|/threads/{thread_id}/runs/{run_id}/steps - get | |  | X | 
|/threads/{thread_id}/runs/{run_id}/steps/{step_id} - get | |  | X | 
|/assistants/{assistant_id}/files - get | |  | X | 
|/assistants/{assistant_id}/files - post | |  | X | 
|/assistants/{assistant_id}/files/{file_id} - get | |  | X | 
|/assistants/{assistant_id}/files/{file_id} - delete | |  | X | 
|/threads/{thread_id}/messages/{message_id}/files - get | |  | X | 
|/threads/{thread_id}/messages/{message_id}/files/{file_id} - get | |  | X | 

40 out of 57 endpoints are implemented, 70%
