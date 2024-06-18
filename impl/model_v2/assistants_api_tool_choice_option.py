from typing_extensions import override

from openapi_server_v2.models.assistants_api_tool_choice_option import AssistantsApiToolChoiceOption as GeneratedAssistantsApiToolChoiceOption

class AssistantsApiToolChoiceOption(GeneratedAssistantsApiToolChoiceOption):

    @override
    def __init__(self, *args, **kwargs) -> None:
        if args:
            if len(args) > 1:
                raise ValueError("If a position argument is used, only 1 is allowed to set `actual_instance`")
            if kwargs:
                raise ValueError("If a position argument is used, keyword arguments cannot be used.")
            super().__init__(actual_instance=args[0])
        else:
            super().__init__(**kwargs)
            super().__init__(actual_instance=self.from_dict(kwargs).actual_instance)
