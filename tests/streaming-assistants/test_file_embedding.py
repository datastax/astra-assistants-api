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