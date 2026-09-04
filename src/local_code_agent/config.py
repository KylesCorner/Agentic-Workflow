from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-coder:latest"
    tavily_api_key: str | None = None
    code_agent_max_tool_iterations: int = 20
    code_agent_max_tool_output_chars: int = 40_000


settings = Settings()
