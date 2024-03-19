import logging

logger = logging.getLogger(__name__)
def test_file_embedding(patched_openai_client):
    file = patched_openai_client.files.create(
        file=open(
            "./tests/fixtures/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
        embedding_model="text-embedding-3-large",
    )
    logger.info(file)

def test_file_embedding_python(patched_openai_client):
    file = patched_openai_client.files.create(
        file=open(
            "./tests/fixtures/sample.py",
            "rb",
        ),
        purpose="assistants",
        embedding_model="text-embedding-3-large",
    )
    logger.info(file)

def test_file_embedding_invalid(patched_openai_client):
    try:
        patched_openai_client.files.create(
            file=open(
                "./tests/fixtures/invalid",
                "rb",
            ),
            purpose="assistants",
            embedding_model="text-embedding-3-large",
        )
    except Exception as e:
        assert e.status_code == 400
        logger.info(e)

def test_file_embedding_png(patched_openai_client):
    try:
        patched_openai_client.files.create(
            file=open(
                "./tests/fixtures/fake.png",
                "rb",
            ),
            purpose="assistants",
            embedding_model="text-embedding-3-large",
        )
    except Exception as e:
        assert e.status_code == 400
        logger.info(e)