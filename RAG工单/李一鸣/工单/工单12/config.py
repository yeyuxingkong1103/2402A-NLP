from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ReActConfig:
    model: str = os.getenv("REACT_MODEL", "deepseek-chat")
    max_iterations: int = int(os.getenv("REACT_MAX_ITERATIONS", "6"))
    temperature: float = float(os.getenv("REACT_TEMPERATURE", "0.2"))


config = ReActConfig()
