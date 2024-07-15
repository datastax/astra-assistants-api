from abc import ABC, abstractmethod
import inspect
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    # has an id and output text
    cache_id: str = Field(str, description="The cache ID of the tool call.")
    output: str = Field(str, description="The output of the tool call.")


class ToolInterface(ABC):
    @abstractmethod
    def call(self, arguments: BaseModel) -> ToolResult:
        pass

    def name(self):
        return self.to_function()['function']['name']

    def tool_choice_object(self):
        return {"type": "function", "function": {"name": self.name()}}

    def get_model(self):
        call_sig = inspect.signature(self.call)
        # Get the first parameter of the call method
        return list(call_sig.parameters.values())[0].annotation

    def to_function(self):
        call_sig = inspect.signature(self.call)

        parameters = {}
        for name, param in call_sig.parameters.items():
            if param.annotation != param.empty:
                param_type = param.annotation
                if issubclass(param_type, BaseModel):
                    parameters = param_type.schema()
                else:
                    parameters = {
                        "type": "object",
                        "properties": {
                            name: {
                                "type": param.annotation.__name__.lower(),
                                "description": f"The {name} parameter."
                            }
                        },
                        "required": [name for name, param in call_sig.parameters.items() if param.default == inspect.Parameter.empty]
                    }
            else:
                parameters = {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": "string",  # Defaulting to string if type is not specified
                            "description": f"The {name} parameter."
                        }
                    },
                    "required": [name for name, param in call_sig.parameters.items() if param.default == inspect.Parameter.empty]
               }

        function = {
            "type": "function",
            "function": {
                "name": self.__class__.__name__,
                "description": f"{self.__class__.__name__} function.",
                "parameters": parameters
            }
        }
        # print(json.dumps(function))
        return function
