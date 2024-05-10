#!/bin/bash

openapi-generator-cli generate --skip-validate-spec -i source_openapi.yaml -g python-fastapi -o ./  -p fastapiImplementationPackage=impl -t templates -p packageName=openapi_server_v2
#openapi-generator-cli generate --skip-validate-spec -i extracted_openapi_assistant_id_deep.yaml -g python-fastapi -o ./  -p fastapiImplementationPackage=impl -t templates
#openapi-generator-cli generate --skip-validate-spec -i extracted_openapi_assistant_id_deep.yaml -g python-fastapi -o ./  -p fastapiImplementationPackage=impl

rm -rf ./openapi_server_v2/*

mv ./src/openapi_server_v2/* ./openapi_server_v2

# Remove the now-empty src directory
rm -rf ./src

rm -rf .flake8 .openapi-generator-ignore .openapi-generator/ docker-compose.yaml openapi.yaml openapitools.json requirements.txt setup.cfg tests/*.py

git checkout -- pyproject.toml .gitignore Dockerfile README.md
