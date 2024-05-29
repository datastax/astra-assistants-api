def print_embedding(model, client):
    text="Draw your favorite animal."
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    print(f'text> {text}')
    print(f'embedding-model-{model}>\n{response.data[0].embedding}')
    print(f'size{len(response.data[0].embedding)}')


def test_embedding_cohere(openai_client):
    model = "cohere/embed-english-v3.0"
    print_embedding(model, openai_client)

def test_embedding_titan(openai_client):
    model = "amazon.titan-embed-text-v1"
    print_embedding(model, openai_client)


def test_embedding_ada_002(openai_client):
    model = "text-embedding-ada-002"
    print_embedding(model, openai_client)


def test_embedding_3_small(openai_client):
    model="text-embedding-3-small"
    print_embedding(model, openai_client)

def test_embedding_3_small(openai_client):
    model="text-embedding-3-large"
    print_embedding(model, openai_client)
