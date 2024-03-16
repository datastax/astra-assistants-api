import logging

logger = logging.getLogger(__name__)

def print_embedding(model, client):
    text="Draw your favorite animal."
    response = client.embeddings.create(
        model=model,
        input=[text],
    )
    logger.info(f'text> {text}')
    logger.info(f'embedding-model-{model}>\n{response.data[0].embedding}')
    logger.info(f'size{len(response.data[0].embedding)}')
    text2="Draw your least favorite animal."
    response = client.embeddings.create(
        model=model,
        input=[text, text2],
    )
    logger.info(f'text> {text}')

    assert len(response.data) == 2

    for datum in response.data:
        logger.info(f'embedding-model-{model}>\n{datum.embedding}')



def test_embedding_cohere(patched_openai_client):
    model = "cohere/embed-english-v3.0"
    print_embedding(model, patched_openai_client)

def test_embedding_titan(patched_openai_client):
    model = "amazon.titan-embed-text-v1"
    print_embedding(model, patched_openai_client)


def test_embedding_ada_002(patched_openai_client):
    model = "text-embedding-ada-002"
    print_embedding(model, patched_openai_client)


def test_embedding_3_small(patched_openai_client):
    model="text-embedding-3-small"
    print_embedding(model, patched_openai_client)

def test_embedding_3_small(patched_openai_client):
    model="text-embedding-3-large"
    print_embedding(model, patched_openai_client)
