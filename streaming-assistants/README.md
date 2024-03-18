# streaming_assistants

~The official OpenAI Assistants API does not yet support streaming (although this functionality has been hinted at since launch back in November). Streaming is critical for a large subset of genai use cases and there has been significant feedback that it's the major blocker for addption of the Assistants API from many. We decided that we (and our users) couldn't wait so we implemented streaming support in Astra Assistants API.~

OpenAI has now added streaming support with streaming runs. This libriary will now mainly be used to streamline multi-llm use. We will continue to support our old streaming messages approach for existing users.

# How to use    

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
