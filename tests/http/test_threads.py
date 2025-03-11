from model_v2.assistants_api_tool_choice_option import AssistantsApiToolChoiceOption
from model_v2.create_run_request import CreateRunRequest
from model_v2.create_thread_and_run_request import CreateThreadAndRunRequest


def test_CreateRunRequest():

    obj = {
        "assistant_id": "assistant_id",
        "thread_id": "thread_id",
        "content": "content",
        "tool_choice": 'auto'
    }
    tool_choice = AssistantsApiToolChoiceOption.from_dict(obj.get("tool_choice")) if obj.get("tool_choice") is not None else None
    bytes = b'{"assistant_id": "asst_AQtHeQJYZ-Ic5ZfAq7RKZb3fVypuQNGH", "additional_instructions": null, "tool_choice": "auto", "stream": true}'

    crr = CreateRunRequest(**obj)
    crr = CreateRunRequest.from_dict(obj)
    crr = CreateRunRequest.from_json(bytes.decode('utf-8'))
    ctarr = CreateThreadAndRunRequest(**obj)
    ctarr = CreateThreadAndRunRequest.from_dict(obj)
    ctarr = CreateThreadAndRunRequest.from_json(bytes.decode('utf-8'))
