def test_file_embedding(openai_client):
    file = openai_client.files.create(
        file=open(
            "./tests/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
        embedding_model="text-embedding-3-large",
    )
    print(file)


def test_with_custom_timeout(openai_client):
    file_id = openai_client.with_options(
        timeout=80 * 1000,  # Set a custom timeout
    ).files.create(
        file=open("./tests/language_models_are_unsupervised_multitask_learners.pdf","rb"),
        purpose="assistants"
    ).id
    assert file_id is not None