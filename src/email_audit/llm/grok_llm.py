import json
import os
from typing import Optional, Type, Union, List
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

from .base_llm import BaseLLM, MultimodalPrompt
from loguru import logger

class GrokLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "grok-3-beta", temperature: float = 0.0):
        super().__init__(api_key, model_name, temperature)
        self.api_key = api_key or self._get_env_var("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("Grok API key not found. Please set GROK_API_KEY environment variable or pass it as an argument.")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

    async def ainvoke(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        logger.debug(f"GrokLLM invoking model {self.model_name} with temperature {self.temperature}")
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
                )

                logger.debug(f"Grok response: {response}")

                if response.choices and response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    if tool_call.function.name == "structured_output":
                        try:
                            data = json.loads(tool_call.function.arguments)
                            return schema(**data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSONDecodeError parsing Grok response: {e}")
                            return None
                        except ValidationError as e:
                            logger.error(f"ValidationError for Grok response: {e}")
                            return None
                else:
                    logger.warning("No tool calls found in Grok response.")
                    return None

            else:
                # No schema, expect string output
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                logger.debug(f"Grok response (no schema): {response}")
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    logger.warning("No content found in response when no schema was provided.")
                    return None

        except Exception as e:
            logger.error(f"Error invoking Grok model {self.model_name}: {e}")
            return None

    @property
    def supports_multimodal(self) -> bool:
        """Grok doesn't support multimodal inputs currently."""
        return False

    @property
    def supported_image_types(self) -> List[str]:
        """Grok doesn't support image types currently."""
        return []

    @property
    def supported_document_types(self) -> List[str]:
        """Grok doesn't support document types currently."""
        return []

    async def ainvoke_multimodal(self, prompt: MultimodalPrompt, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        """Multimodal not supported by Grok - fallback to text only."""
        logger.warning("Multimodal not supported by Grok, falling back to text-only processing")
        return await self.ainvoke(prompt.text, schema) 