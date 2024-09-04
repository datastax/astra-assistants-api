import os
from threading import Event

from astra_assistants.tools.structured_code.lsp.constants import VSCODE_DEFAULT_INITIALIZE, TIMEOUT_SECONDS
from astra_assistants.tools.structured_code.lsp.lsp_session import LspSession, PUBLISH_DIAGNOSTICS
from astra_assistants.tools.structured_code.lsp.util import convert_keys_to_snake_case


class LspSessionManager:
    def __init__(self, module="ruff server"):
        self.diagnostics = []
        self.ls_session = LspSession(cwd=os.getcwd(), module=module)
        self.ls_session.initialize(VSCODE_DEFAULT_INITIALIZE)
        assert "serverInfo" in self.ls_session.server_capabilities
        self.done = Event()
        self.ls_session.set_notification_callback(PUBLISH_DIAGNOSTICS, self._handler)

    def _handler(self, params):
        self.diagnostics = params
        self.done.set()

    # Note this is not thread safe
    def send_notification(self, notification_name, payload):
        self.done = Event()
        self.ls_session.send_notification(notification_name, payload)

        # Wait to receive all notifications.
        self.done.wait(TIMEOUT_SECONDS)

        return convert_keys_to_snake_case(self.diagnostics)

    def send_request(self, request_name, payload):
        fut = self.ls_session.send_request(request_name, payload)
        result = fut.result(TIMEOUT_SECONDS)
        return result

    def close(self):
        self.ls_session.close()
