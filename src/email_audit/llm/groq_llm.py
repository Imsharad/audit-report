import json
import os
from typing import Optional, Type, Union, Dict, Any, List
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI
from groq import Groq

from .base_llm import BaseLLM, MultimodalPrompt
from loguru import logger

class GroqLLM(BaseLLM):
    """Groq LLM implementation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.7
    ):
        """
        Initialize the Groq LLM client.

        Args:
            api_key: Optional API key. If not provided, will attempt to load from GROQ_API_KEY env var.
            model_name: The model to use (default: mixtral-8x7b-32768).
            temperature: Sampling temperature (default: 0.7).
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided and GROQ_API_KEY environment variable not set")
        
        self.model_name = model_name
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        logger.info(f"Initialized Groq LLM with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 8096) -> str:
        """
        Generate a response using the Groq model.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated response text.
        """
        try:
            logger.debug(f"Generating response with Groq model {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response with Groq: {str(e)}")
            raise

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get information about available Groq models.

        Returns:
            Dictionary containing model information.
        """
        return {
            "mixtral-8x7b-32768": {
                "description": "Mixtral 8x7B with 32K context window",
                "max_tokens": 32768,
                "supports_streaming": True
            },
            "llama2-70b-4096": {
                "description": "Llama 2 70B with 4K context window",
                "max_tokens": 4096,
                "supports_streaming": True
            }
        }

    async def ainvoke(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        logger.debug(f"GroqLLM invoking model {self.model_name} with temperature {self.temperature}")
        try:
            messages = [{"role": "user", "content": prompt}]
            if schema:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "structured_output",
                            "description": f"Parses the output according to the provided schema: {schema.__name__}",
                            "parameters": schema.model_json_schema()
                        }
                    }
                ]
                tool_choice = {"type": "function", "function": {"name": "structured_output"}}

                logger.debug(f"Using tool choice: {tool_choice}")
                logger.debug(f"Schema for tool: {schema.model_json_schema()}")

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=8192
                )

                logger.debug(f"Groq response: {response}")

                if response.choices and response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    if tool_call.function.name == "structured_output":
                        try:
                            data = json.loads(tool_call.function.arguments)
                            return schema(**data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSONDecodeError parsing Groq response: {e}")
                            return None
                        except ValidationError as e:
                            logger.error(f"ValidationError for Groq response: {e}")
                            return None
                else:
                    logger.warning("No tool calls found in Groq response.")
                    return None

            else:
                # No schema, expect string output
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=8192
                )
                logger.debug(f"Groq response (no schema): {response}")
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    logger.warning("No content found in response when no schema was provided.")
                    return None

        except Exception as e:
            logger.error(f"Error invoking Groq model {self.model_name}: {e}")
            return None 

    @property
    def supports_multimodal(self) -> bool:
        """Groq doesn't support multimodal inputs currently."""
        return False

    @property
    def supported_image_types(self) -> List[str]:
        """Groq doesn't support image types currently."""
        return []

    @property
    def supported_document_types(self) -> List[str]:
        """Groq doesn't support document types currently."""
        return []

    async def ainvoke_multimodal(self, prompt: MultimodalPrompt, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        """Multimodal not supported by Groq - fallback to text only."""
        logger.warning("Multimodal not supported by Groq, falling back to text-only processing")
        return await self.ainvoke(prompt.text, schema) 