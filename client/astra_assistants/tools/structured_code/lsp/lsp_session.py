import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Callable, TypeVar, Optional

from pylsp_jsonrpc.dispatchers import MethodDispatcher
from pylsp_jsonrpc.endpoint import Endpoint
from pylsp_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter

from astra_assistants.tools.structured_code.lsp.constants import VSCODE_DEFAULT_INITIALIZE

LSP_EXIT_TIMEOUT = 5000


PUBLISH_DIAGNOSTICS = "textDocument/publishDiagnostics"
CODE_ACTION = "textDocument/codeAction"
WINDOW_LOG_MESSAGE = "window/logMessage"
WINDOW_SHOW_MESSAGE = "window/showMessage"

T = TypeVar("T")


def unwrap(option: T | None) -> T:
    if option is None:
        raise ValueError("Option is None")
    return option


class LspSession(MethodDispatcher):
    """Send and Receive messages over LSP."""

    def __init__(self, cwd: str, module: str):
        self.cwd = cwd
        self.module = module

        self._endpoint: Endpoint
        self._thread_pool: ThreadPoolExecutor = ThreadPoolExecutor()
        self._sub: Optional[subprocess.Popen] = None
        self._reader: JsonRpcStreamReader | None = None
        self._writer: JsonRpcStreamWriter | None = None
        self._notification_callbacks: dict[str, Callable] = {}
        self.server_capabilities = None
        self.start()

    def start(self):
        """Context manager entrypoint.

        shell=True needed for pytest-cov to work in subprocess.
        """
        if self._sub is not None:
            raise RuntimeError("LSP session already started")
        self._sub = subprocess.Popen(
            #[sys.executable, "-m", self.module],
            #[sys.executable, "-m", "ruff", "server", "-v"],
            [sys.executable, "-m", "ruff", "server"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=self.cwd,
            env=os.environ,
        )

        self._writer = JsonRpcStreamWriter(self._sub.stdin)
        self._reader = JsonRpcStreamReader(self._sub.stdout)
        threading.Thread(target=self._read_stderr, daemon=True).start()

        dispatcher = {
            PUBLISH_DIAGNOSTICS: self._publish_diagnostics,
            WINDOW_SHOW_MESSAGE: self._window_show_message,
            WINDOW_LOG_MESSAGE: self._window_log_message,
        }
        self._endpoint = Endpoint(dispatcher, self._writer.write)
        self._thread_pool.submit(self._reader.listen, self._endpoint.consume)
        return self

    def _read_stderr(self):
        """Handle stderr in a non-blocking way."""
        for line in self._sub.stderr:
            print(f"STDERR: {line.strip()}")

    def close(self):
        self.shutdown(True)
        unwrap(self._sub).terminate()
        unwrap(self._sub).wait()
        self._endpoint.shutdown()  # type: ignore[union-attr]
        self._thread_pool.shutdown()
        unwrap(self._writer).close()  # type: ignore[attr-defined]
        unwrap(self._reader).close()  # type: ignore[attr-defined]
        self._sub.terminate()

    def initialize(
            self,
            initialize_params=None,
            process_server_capabilities=None,
    ):
        """Sends the initialize request to LSP server."""
        if initialize_params is None:
            initialize_params = VSCODE_DEFAULT_INITIALIZE
        server_initialized = Event()

        def _after_initialize(fut):
            self.server_capabilities = fut.result()
            if process_server_capabilities:
                process_server_capabilities(self.server_capabilities)
            self.initialized()
            server_initialized.set()

        self.send_request(
            "initialize",
            params=(
                initialize_params
                if initialize_params is not None
                else VSCODE_DEFAULT_INITIALIZE
            ),
            handle_response=_after_initialize,
        )

        server_initialized.wait()

    def initialized(self, initialized_params=None):
        """Sends the initialized notification to LSP server."""
        if initialized_params is None:
            initialized_params = {}
        self._endpoint.notify("initialized", initialized_params)

    def shutdown(self, should_exit, exit_timeout: float = LSP_EXIT_TIMEOUT):
        """Sends the shutdown request to LSP server."""

        def _after_shutdown(_):
            if should_exit:
                self.exit_lsp(exit_timeout)

        self.send_request("shutdown", handle_response=_after_shutdown)

    def exit_lsp(self, exit_timeout: float = LSP_EXIT_TIMEOUT):
        """Handles LSP server process exit."""
        self._endpoint.notify("exit")
        assert unwrap(self._sub).wait(exit_timeout) == 0

    def notify_did_change(self, did_change_params):
        """Sends did change notification to LSP Server."""
        self.send_notification("textDocument/didChange", params=did_change_params)

    def notify_did_save(self, did_save_params):
        """Sends did save notification to LSP Server."""
        self.send_notification("textDocument/didSave", params=did_save_params)

    def notify_did_open(self, did_open_params):
        """Sends did open notification to LSP Server."""
        self.send_notification("textDocument/didOpen", params=did_open_params)

    def notify_did_close(self, did_close_params):
        """Sends did close notification to LSP Server."""
        self.send_notification("textDocument/didClose", params=did_close_params)

    def set_notification_callback(self, notification_name, callback):
        """Set custom LS notification handler."""
        self._notification_callbacks[notification_name] = callback

    def get_notification_callback(self, notification_name):
        """Gets callback if set or default callback for a given LS notification."""
        try:
            return self._notification_callbacks[notification_name]
        except KeyError:

            def _default_handler(_params):
                """Default notification handler."""

            return _default_handler

    def _publish_diagnostics(self, publish_diagnostics_params):
        """Internal handler for text document publish diagnostics."""
        return self._handle_notification(
            PUBLISH_DIAGNOSTICS, publish_diagnostics_params
        )

    def _window_log_message(self, window_log_message_params):
        """Internal handler for window log message."""
        return self._handle_notification(WINDOW_LOG_MESSAGE, window_log_message_params)

    def _window_show_message(self, window_show_message_params):
        """Internal handler for window show message."""
        return self._handle_notification(
            WINDOW_SHOW_MESSAGE, window_show_message_params
        )

    def _handle_notification(self, notification_name, params):
        """Internal handler for notifications."""
        fut: Future = Future()

        def _handler():
            callback = self.get_notification_callback(notification_name)
            callback(params)
            fut.set_result(None)

        self._thread_pool.submit(_handler)
        return fut

    def send_request(self, name, params=None, handle_response=lambda f: f.done()):
        """Sends {name} request to the LSP server."""
        fut = self._endpoint.request(name, params)
        fut.add_done_callback(handle_response)
        return fut

    def send_notification(self, name, params=None):
        """Sends {name} notification to the LSP server."""
        self._endpoint.notify(name, params)