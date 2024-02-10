# streaming_assistants

The official OpenAI Assistants API does not yet support streaming (although this functionality has been hinted at since launch back in November). Streaming is critical for a large subset of genai use cases and there has been significant feedback that it's the major blocker for addption of the Assistants API from many. We decided that we (and our users) couldn't wait so we implemented streaming support in Astra Assistants API.

# How to use    

Because streaming is not supported in the API, it also isn't supported by the OpenAI SDKs. Rather than forking the SDK project, we created this shim which we will maintain at least until the official implementation catches up. Expect a similar project for js/ts coming soon.

Install streaming_assistants using your python package manager of choice:

```
poetry add streaming_assistants
```


import and patch your client:

```
from openai import OpenAI
from streaming_assistants import patch

client = patch(OpenAI())

```


After creating your run status will go to `generating`:
```
run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=assistant_id,
)

while (True):
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run.id
    )
    if run.status == 'failed':
        raise ValueError("Run is in failed state")
    if run.status == 'completed' or run.status == 'generating':
        break
    time.sleep(1)
```

At this point you can call `client.beta.threads.messages.list` with streaming=True:

```
response = client.beta.threads.messages.list(
    thread_id=thread_id,
    stream=True,
)
```

process the streaming response:

```
for part in response:
    print(part.data[0].content[0].delta.value)
```


## Compatibility

We've done our best to come up with the likeliest design for what OpenAI will release. We also attempted to work in the open on the implementation by sharing this design doc starting a discussion in OpenAI's openapi repo. There are a couple of other projects that are interested in how this functionality will officially be supported. See related tickets here: 

That said, we had to make some design decisions that may or may not match what OpenAI will do in their official implementation.


As soon as OpenAI releases official streaming support we will close the compatibility gap as soon as possible while doing our best to support existing users and to avoid breaking changes. This will be a tricky needle to thread but believe that giving folks *an option* today will be worth the trouble tomorrow.
