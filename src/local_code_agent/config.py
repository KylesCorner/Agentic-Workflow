from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# Look for global .env_agent file in home directory
home_dir = os.path.expanduser("~")
global_env_file = os.path.join(home_dir, ".env_agent")

# Create a list of env files to load, with global file first (so local can override)
env_files = [global_env_file, ".env"] if os.path.exists(global_env_file) else [".env"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_files,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-coder:latest"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    tavily_api_key: str | None = None
    code_agent_max_tool_iterations: int = 20
    code_agent_max_tool_output_chars: int = 40_000


settings = Settings()
