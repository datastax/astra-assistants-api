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

def test_file_embedding_no_extension(patched_openai_client):
    patched_openai_client.files.delete(file_id='e3b0c442-98fc-1c14-9afb-f4c8996fb924')
    file = patched_openai_client.files.create(
        file=open(
            "./tests/fixtures/Dockerfile",
            "rb",
        ),
        purpose="assistants",
        embedding_model="text-embedding-3-large",
    )
    logger.info(file)

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