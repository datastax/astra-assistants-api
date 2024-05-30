Here are all the OpenAI endpoints, most of the stateful beta endpoints are implemented and all the simple stateless services are simply proxied to OpenAI:

|Endpoint | Implemented | Stateless / Proxy | Roadmap|
|---------|-------------|-----------------|--------|
|/chat/completions - post | X | X |  | 
|/completions - post | X | X |  | 
|/images/generations - post | X | X |  | 
|/images/edits - post | X | X |  | 
|/images/variations - post | X | X |  | 
|/embeddings - post | X | X |  | 
|/audio/speech - post | X | X |  | 
|/audio/transcriptions - post | X | X |  | 
|/audio/translations - post | X | X |  | 
|/files - get | X |  |  | 
|/files - post | X |  |  | 
|/files/{file_id} - delete | X |  |  | 
|/files/{file_id} - get | X |  |  | 
|/files/{file_id}/content - get | X |  |  | 
|/fine_tuning/jobs - post | X | X |  | 
|/fine_tuning/jobs - get | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id} - get | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id}/events - get | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id}/cancel - post | X | X |  | 
|/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints - get | X | X |  | 
|/models - get | X |  |  | 
|/models/{model} - get | X |  |  | 
|/models/{model} - delete | X |  |  | 
|/moderations - post | X | X |  | 
|/assistants - get | X |  |  | 
|/assistants - post | X |  |  | 
|/assistants/{assistant_id} - get | X |  |  | 
|/assistants/{assistant_id} - post | X |  |  | 
|/assistants/{assistant_id} - delete | X |  |  | 
|/threads - post | X |  |  | 
|/threads/{thread_id} - get | X |  |  | 
|/threads/{thread_id} - post | X |  |  | 
|/threads/{thread_id} - delete | X |  |  | 
|/threads/{thread_id}/messages - get | X |  |  | 
|/threads/{thread_id}/messages - post | X |  |  | 
|/threads/{thread_id}/messages/{message_id} - get | X |  |  | 
|/threads/{thread_id}/messages/{message_id} - post | X |  |  | 
|/threads/{thread_id}/messages/{message_id} - delete | X |  |  | 
|/threads/runs - post | X |  |  | 
|/threads/{thread_id}/runs - get | X |  |  | 
|/threads/{thread_id}/runs - post | X |  |  | 
|/threads/{thread_id}/runs/{run_id} - get | X |  |  | 
|/threads/{thread_id}/runs/{run_id} - post | |  | X | 
not implemented: /threads/{thread_id}/runs/{run_id} post
|/threads/{thread_id}/runs/{run_id}/submit_tool_outputs - post | X |  |  | 
|/threads/{thread_id}/runs/{run_id}/cancel - post | |  | X | 
not implemented: /threads/{thread_id}/runs/{run_id}/cancel post
|/threads/{thread_id}/runs/{run_id}/steps - get | |  | X | 
not implemented: /threads/{thread_id}/runs/{run_id}/steps get
|/threads/{thread_id}/runs/{run_id}/steps/{step_id} - get | |  | X | 
not implemented: /threads/{thread_id}/runs/{run_id}/steps/{step_id} get
|/vector_stores - get | |  | X | 
not implemented: /vector_stores get
|/vector_stores - post | X |  |  | 
|/vector_stores/{vector_store_id} - get | X |  |  | 
|/vector_stores/{vector_store_id} - post | |  | X | 
not implemented: /vector_stores/{vector_store_id} post
|/vector_stores/{vector_store_id} - delete | |  | X | 
not implemented: /vector_stores/{vector_store_id} delete
|/vector_stores/{vector_store_id}/files - get | X |  |  | 
|/vector_stores/{vector_store_id}/files - post | X |  |  | 
|/vector_stores/{vector_store_id}/files/{file_id} - get | |  | X | 
not implemented: /vector_stores/{vector_store_id}/files/{file_id} get
|/vector_stores/{vector_store_id}/files/{file_id} - delete | |  | X | 
not implemented: /vector_stores/{vector_store_id}/files/{file_id} delete
|/vector_stores/{vector_store_id}/file_batches - post | |  | X | 
not implemented: /vector_stores/{vector_store_id}/file_batches post
|/vector_stores/{vector_store_id}/file_batches/{batch_id} - get | |  | X | 
not implemented: /vector_stores/{vector_store_id}/file_batches/{batch_id} get
|/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel - post | |  | X | 
not implemented: /vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel post
|/vector_stores/{vector_store_id}/file_batches/{batch_id}/files - get | |  | X | 
not implemented: /vector_stores/{vector_store_id}/file_batches/{batch_id}/files get
|/batches - post | |  | X | 
not implemented: /batches post
|/batches - get | |  | X | 
not implemented: /batches get
|/batches/{batch_id} - get | |  | X | 
not implemented: /batches/{batch_id} get
|/batches/{batch_id}/cancel - post | |  | X | 
not implemented: /batches/{batch_id}/cancel post

47 out of 64 endpoints are implemented, 73%
