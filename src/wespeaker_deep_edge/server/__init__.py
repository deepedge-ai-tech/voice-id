"""WeSpeaker WebSocket 服务端。"""

from .template_manager import TemplateManager
from .ws_server import WSServer

__all__ = ["WSServer", "TemplateManager"]
