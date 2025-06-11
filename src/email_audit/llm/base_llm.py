import abc
import os
from typing import Optional, Type, Union, List
from pydantic import BaseModel
from pathlib import Path

class MediaInput(BaseModel):
    """Represents a media input for multimodal LLM processing."""
    type: str  # "image" or "document" 
    path: Path
    content_type: str
    description: Optional[str] = None

class MultimodalPrompt(BaseModel):
    """Represents a multimodal prompt with text and media inputs."""
    text: str
    media_inputs: List[MediaInput] = []
    
    @property
    def has_media(self) -> bool:
        return len(self.media_inputs) > 0
    
    @property
    def image_count(self) -> int:
        return len([m for m in self.media_inputs if m.type == "image"])
    
    @property
    def document_count(self) -> int:
        return len([m for m in self.media_inputs if m.type == "document"])

class BaseLLM(abc.ABC):
    def __init__(self, api_key: Optional[str], model_name: str, temperature: float):
        """
        Initializes the base LLM.

        Args:
            api_key: The API key for the LLM provider. If None, it will be fetched from the environment variable.
            model_name: The name of the model to use.
            temperature: The temperature to use for generation.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    @abc.abstractmethod
    async def ainvoke(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        """
        Invokes the language model with the given prompt.

        Args:
            prompt: The prompt to send to the model.
            schema: An optional Pydantic schema. If provided, the LLM is expected
                    to return a response that can be parsed into this schema.
                    If not provided, the LLM should return a string.

        Returns:
            A Pydantic model instance if a schema is provided and parsing is successful,
            or a string if no schema is provided or parsing fails (though ideally,
            implementations should try to enforce schema compliance or handle errors).
            Returns None if the invocation fails.
        """
        pass

    @abc.abstractmethod
    async def ainvoke_multimodal(self, prompt: MultimodalPrompt, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        """
        Invokes the language model with a multimodal prompt containing text and media.

        Args:
            prompt: The multimodal prompt with text and media inputs.
            schema: An optional Pydantic schema for structured output.

        Returns:
            A Pydantic model instance if a schema is provided and parsing is successful,
            or a string if no schema is provided.
            Returns None if the invocation fails.
        """
        pass

    @property
    @abc.abstractmethod
    def supports_multimodal(self) -> bool:
        """Returns True if this LLM supports multimodal inputs."""
        pass

    @property
    @abc.abstractmethod
    def supported_image_types(self) -> List[str]:
        """Returns list of supported image MIME types."""
        pass

    @property
    @abc.abstractmethod
    def supported_document_types(self) -> List[str]:
        """Returns list of supported document MIME types."""
        pass

    def can_process_media(self, content_type: str, media_type: str) -> bool:
        """Check if this LLM can process the given media type and content type."""
        if not self.supports_multimodal:
            return False
        
        if media_type == "image":
            return content_type in self.supported_image_types
        elif media_type == "document":
            return content_type in self.supported_document_types
        
        return False

    @staticmethod
    def _get_env_var(name: str) -> Optional[str]:
        """Helper to get environment variables."""
        return os.getenv(name)
