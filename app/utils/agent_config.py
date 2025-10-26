"""Agent configuration loader from YAML files.

Supports loading large prompts and greetings that exceed environment variable limits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import structlog
import yaml


logger = structlog.get_logger(__name__)


@dataclass
class AgentConfig:
    """Agent configuration loaded from YAML file.

    Fields:
        instructions: Agent system prompt/instructions (required)
        greeting: Optional welcome message spoken at call start
        metadata: Optional metadata for documentation purposes
    """

    instructions: str = "You are a helpful voice assistant."
    greeting: Optional[str] = None
    metadata: Optional[Dict] = None

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "AgentConfig":
        """Load agent configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            AgentConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML file is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {file_path}")

        logger.info("Loading agent config from YAML", file_path=str(file_path))

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise ValueError("YAML file must contain a dictionary")

            # Extract fields
            instructions = data.get("instructions")
            greeting = data.get("greeting")
            metadata = data.get("metadata")

            # Validate instructions (required field)
            if not instructions:
                raise ValueError("'instructions' field is required in YAML config")
            if not isinstance(instructions, str):
                raise ValueError("'instructions' field must be a string")

            # Strip whitespace from multi-line strings
            instructions = instructions.strip()
            if greeting:
                greeting = greeting.strip()

            logger.info(
                "Agent config loaded successfully",
                instructions_length=len(instructions),
                has_greeting=greeting is not None,
                metadata=metadata
            )

            return cls(
                instructions=instructions,
                greeting=greeting,
                metadata=metadata
            )

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load agent config: {e}")

    @classmethod
    def from_yaml_or_none(cls, file_path: Optional[str | Path]) -> Optional["AgentConfig"]:
        """Load agent config from YAML file, or return None if file not specified.

        FAIL-FAST STRATEGY: If file_path is specified but loading fails, raises exception.
        No fallback to defaults when file is specified.

        Args:
            file_path: Optional path to YAML configuration file

        Returns:
            AgentConfig instance if file_path specified, None otherwise

        Raises:
            FileNotFoundError: If specified file doesn't exist
            ValueError: If YAML file is invalid
        """
        if not file_path:
            logger.info("No agent config file specified")
            return None

        # If file_path is specified, loading MUST succeed (fail-fast)
        return cls.from_yaml(file_path)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/debugging.

        Returns:
            Dictionary representation
        """
        return {
            "instructions": self.instructions[:100] + "..." if len(self.instructions) > 100 else self.instructions,
            "greeting": self.greeting[:100] + "..." if self.greeting and len(self.greeting) > 100 else self.greeting,
            "metadata": self.metadata,
            "instructions_length": len(self.instructions),
            "greeting_length": len(self.greeting) if self.greeting else 0
        }
