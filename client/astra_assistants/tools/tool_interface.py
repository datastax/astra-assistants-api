from abc import ABC, abstractmethod
import inspect

class ToolInterface(ABC):
    @abstractmethod
    def search(self, query):
        pass
class ToolInterface(ABC):
    @abstractmethod
    def call(self, query):
        pass

    def name(self):
        return self.to_function()['function']['name']

    def tool_choice_object(self):
        return {"type": "function", "function": {"name": self.name()}}

    def to_function(self):
        search_sig = inspect.signature(self.call)
        parameters = {
            name: {
                "type": "string",  # Assuming all parameters are strings for simplicity
                "description": f"The {name} parameter."
            }
            for name, param in search_sig.parameters.items()
        }
        required = [name for name, param in search_sig.parameters.items() if param.default == inspect.Parameter.empty]

        return {
            "type": "function",
            "function": {
                "name": self.__class__.__name__,
                "description": f"{self.__class__.__name__} function.",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
        }