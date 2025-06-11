import json
import os
import re # Added import re
from typing import Optional, Type, Any, Dict, Union, List
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI # Use AsyncOpenAI for asynchronous operations

from .base_llm import BaseLLM, MultimodalPrompt
from loguru import logger

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", temperature: float = 0.0):
        super().__init__(api_key, model_name, temperature)
        self.api_key = api_key or self._get_env_var("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it as an argument.")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def ainvoke(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        logger.debug(f"OpenAILLM invoking model {self.model_name} with temperature {self.temperature}")
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

                logger.debug(f"OpenAI response: {response}")

                if response.choices and response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    if tool_call.function.name == "structured_output":
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                            return schema(**arguments)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSONDecodeError parsing arguments: {tool_call.function.arguments}, error: {e}")
                            logger.error(f"Attempting to parse a more resilient way, raw arguments: {tool_call.function.arguments}")
                            # Attempt to fix common JSON issues, like trailing commas or escaped quotes.
                            # This is a basic attempt; more sophisticated error handling might be needed.
                            try:
                                # Replace escaped single quotes if they are causing issues with simplejson
                                fixed_args = tool_call.function.arguments.replace("\\'", "'")
                                # Remove trailing commas before closing curly or square brackets
                                fixed_args = re.sub(r',\s*([}\]])', r'\1', fixed_args)
                                arguments = json.loads(fixed_args)
                                return schema(**arguments)
                            except Exception as inner_e:
                                logger.error(f"Could not parse arguments even after attempting to fix them: {inner_e}")
                                logger.error(f"Problematic raw arguments: {tool_call.function.arguments}")
                                return None # Or raise an error, or return the raw string
                        except ValidationError as e:
                            logger.error(f"Pydantic ValidationError: {e}, arguments: {tool_call.function.arguments}")
                            return None # Or raise an error
                    else:
                        logger.warning(f"Expected tool call 'structured_output', got '{tool_call.function.name}'")
                        return None
                else:
                    logger.warning("No tool calls found in response when schema was provided.")
                    # Fallback or error handling: what if the model doesn't use the tool?
                    # Try to parse the content directly if available, or return None/error.
                    if response.choices and response.choices[0].message.content:
                        logger.warning("Falling back to parsing content directly as JSON.")
                        try:
                            # This assumes the model might return JSON even without the tool call.
                            # This is less reliable.
                            raw_content = response.choices[0].message.content
                            parsed_content = json.loads(raw_content)
                            return schema(**parsed_content)
                        except Exception as e:
                            logger.error(f"Could not parse content as JSON: {e}, content: {raw_content}")
                            return raw_content # Return raw content as string if parsing fails
                    return None
            else:
                # No schema, expect string output
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                logger.debug(f"OpenAI response (no schema): {response}")
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    logger.warning("No content found in response when no schema was provided.")
                    return None

        except Exception as e:
            logger.error(f"Error invoking OpenAI model {self.model_name}: {e}")
            # Consider re-raising, or returning a specific error object
            return None

    @property
    def supports_multimodal(self) -> bool:
        """OpenAI GPT-4 Vision supports multimodal inputs."""
        return "vision" in self.model_name.lower() or "gpt-4" in self.model_name.lower()

    @property
    def supported_image_types(self) -> List[str]:
        """Supported image types for OpenAI."""
        return ["image/jpeg", "image/png", "image/gif", "image/webp"]

    @property
    def supported_document_types(self) -> List[str]:
        """OpenAI doesn't directly support document types yet."""
        return []

    async def ainvoke_multimodal(self, prompt: MultimodalPrompt, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        """Multimodal support for OpenAI - placeholder implementation."""
        logger.warning("Multimodal support not fully implemented for OpenAI LLM yet")
        # For now, fall back to text-only processing
        return await self.ainvoke(prompt.text, schema)
