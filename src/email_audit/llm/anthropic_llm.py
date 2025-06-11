import json
import os
import base64
from typing import Optional, Type, Any, Dict, Union, List
from pydantic import BaseModel, ValidationError
from anthropic import AsyncAnthropic # Use AsyncAnthropic for asynchronous operations
from pathlib import Path

from .base_llm import BaseLLM, MultimodalPrompt, MediaInput
from loguru import logger

def _extract_json_from_text(text: str) -> Optional[str]:
    """Extracts a JSON object or array from a string, even if it's embedded in other text."""
    # Find the first opening brace or bracket
    first_brace = text.find('{')
    first_bracket = text.find('[')

    if first_brace == -1 and first_bracket == -1:
        return None  # No JSON object or array found

    # Determine the starting position of the JSON
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start = first_brace
        start_char = '{'
        end_char = '}'
    else:
        start = first_bracket
        start_char = '['
        end_char = ']'

    # Find the matching closing brace/bracket by counting nesting levels
    nesting_level = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == start_char:
            nesting_level += 1
        elif text[i] == end_char:
            nesting_level -= 1
            if nesting_level == 0:
                end = i
                break
    
    if end != -1:
        return text[start:end+1]
    
    return None # Unmatched brackets/braces

class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-opus-20240229", temperature: float = 0.0):
        super().__init__(api_key, model_name, temperature)
        self.api_key = api_key or self._get_env_var("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable or pass it as an argument.")
        self.client = AsyncAnthropic(api_key=self.api_key)

    @property
    def supports_multimodal(self) -> bool:
        """Anthropic Claude supports multimodal inputs."""
        return True

    @property
    def supported_image_types(self) -> List[str]:
        """Supported image types for Anthropic Claude."""
        return ["image/jpeg", "image/png", "image/gif", "image/webp"]

    @property
    def supported_document_types(self) -> List[str]:
        """Anthropic Claude doesn't directly support document types yet."""
        return []

    async def ainvoke(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        logger.debug(f"AnthropicLLM invoking model {self.model_name} with temperature {self.temperature}")
        try:
            system_prompt = ""
            if schema:
                # Instruct the model to return JSON matching the schema
                # This is a common approach for Anthropic models when specific tool use/function calling is not as mature or desired.
                schema_json = schema.model_json_schema()
                # Remove "title" from schema_json if present, as it can sometimes confuse the model
                if "title" in schema_json:
                    del schema_json["title"]

                system_prompt = (
                    "You are a helpful assistant that always responds in JSON format. "
                    "Please provide a response that strictly adheres to the following JSON schema. "
                    "Do not include any explanatory text or markdown formatting before or after the JSON object. "
                    "The entire response must be a single valid JSON object.\n"
                    f"JSON Schema:\n{json.dumps(schema_json)}"
                )
                logger.debug(f"Anthropic system prompt for schema {schema.__name__}:\n{system_prompt}")


            messages = [{"role": "user", "content": prompt}]

            response = await self.client.messages.create(
                model=self.model_name,  # Use the model name from the instance
                max_tokens=4096,  # Adjusted to be within the model's limit
                temperature=self.temperature,
                system=system_prompt if system_prompt else None, # System prompt for structured JSON output
                messages=messages
            )

            logger.debug(f"Anthropic response: {response}")

            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                # Assuming the first content block is the one we want, and it's of type TextBlock
                raw_response_content = response.content[0].text if hasattr(response.content[0], 'text') else None

                if not raw_response_content:
                    logger.warning("No text content found in Anthropic response block.")
                    return None

                logger.debug(f"Raw Anthropic response content: {raw_response_content}")

                if schema:
                    try:
                        # First, extract the JSON part of the response
                        json_str = _extract_json_from_text(raw_response_content)
                        
                        if not json_str:
                            logger.error(f"Could not extract JSON from Anthropic response: {raw_response_content}")
                            return raw_response_content
                            
                        # Now, parse the extracted string
                        data = json.loads(json_str)
                        return schema(**data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSONDecodeError parsing Anthropic response: {e}, content: {raw_response_content}")
                        # Fallback: return the raw string if JSON parsing fails, so it can be inspected
                        return raw_response_content
                    except ValidationError as e:
                        logger.error(f"Pydantic ValidationError for Anthropic response: {e}, content: {raw_response_content}")
                        # Fallback: return the raw string
                        return raw_response_content
                else:
                    # No schema, return the raw text content
                    return raw_response_content
            else:
                logger.warning("No content blocks found in Anthropic response.")
                return None

        except Exception as e:
            logger.error(f"Error invoking Anthropic model {self.model_name}: {e}")
            # Consider re-raising or returning a specific error object
            return None

    async def ainvoke_multimodal(self, prompt: MultimodalPrompt, schema: Optional[Type[BaseModel]] = None) -> Optional[Union[BaseModel, str]]:
        """Invoke Anthropic Claude with multimodal prompt containing images."""
        logger.debug(f"AnthropicLLM multimodal invoke with {prompt.image_count} images")
        
        try:
            system_prompt = ""
            if schema:
                schema_json = schema.model_json_schema()
                if "title" in schema_json:
                    del schema_json["title"]

                system_prompt = (
                    "You are a helpful assistant that always responds in JSON format. "
                    "Please provide a response that strictly adheres to the following JSON schema. "
                    "Do not include any explanatory text or markdown formatting before or after the JSON object. "
                    "The entire response must be a single valid JSON object.\n"
                    f"JSON Schema:\n{json.dumps(schema_json)}"
                )

            # Build content array with text and images
            content = []
            
            # Add text content
            content.append({
                "type": "text",
                "text": prompt.text
            })
            
            # Add image content
            for media in prompt.media_inputs:
                if media.type == "image" and self.can_process_media(media.content_type, "image"):
                    try:
                        # Read and encode image
                        image_data = media.path.read_bytes()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media.content_type,
                                "data": base64_image
                            }
                        })
                        
                        logger.debug(f"Added image {media.path.name} to multimodal prompt")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {media.path}: {e}")
                        continue
                else:
                    logger.warning(f"Unsupported media type {media.type} or content type {media.content_type}")

            messages = [{"role": "user", "content": content}]

            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=self.temperature,
                system=system_prompt if system_prompt else None,
                messages=messages
            )

            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                raw_response_content = response.content[0].text if hasattr(response.content[0], 'text') else None

                if not raw_response_content:
                    logger.warning("No text content found in Anthropic multimodal response.")
                    return None

                if schema:
                    try:
                        # Extract the JSON part of the response
                        json_str = _extract_json_from_text(raw_response_content)
                        
                        if not json_str:
                            logger.error(f"Could not extract JSON from Anthropic multimodal response: {raw_response_content}")
                            return raw_response_content
                            
                        data = json.loads(json_str)
                        return schema(**data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSONDecodeError parsing multimodal Anthropic response: {e}, content: {raw_response_content}")
                        return raw_response_content
                    except ValidationError as e:
                        logger.error(f"Pydantic ValidationError for multimodal Anthropic response: {e}, content: {raw_response_content}")
                        return raw_response_content
                else:
                    return raw_response_content
            else:
                logger.warning("No content blocks found in Anthropic multimodal response.")
                return None

        except Exception as e:
            logger.error(f"Error invoking Anthropic multimodal: {e}")
            return None
