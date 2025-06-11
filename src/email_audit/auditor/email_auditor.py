import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from loguru import logger
from dotenv import load_dotenv, find_dotenv
# from langchain_openai import ChatOpenAI # Removed
from ..llm.llm_factory import LLMFactory
from ..llm.base_llm import BaseLLM
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(find_dotenv())

class Email(BaseModel):
    """Represents a single email message in a conversation thread."""
    sender: str = Field(..., description="The sender's name or email address.")
    timestamp: str = Field(..., description="The timestamp of when the email was sent.")
    recipient: str = Field(..., description="The primary recipient's email address.")
    cc: List[str] = Field(default_factory=list, description="A list of CC'd recipients.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The main content of the email body.")

class EmailConversation(BaseModel):
    """Represents a structured email conversation thread."""
    email_conversations: List[Email] = Field(..., description="A list of email messages, ordered chronologically (oldest first).")

class StepResult(BaseModel):
    """Structured result for a single audit step."""
    step_id: str = Field(..., description="The unique identifier for the audit step.")
    title: str = Field(..., description="The title of the audit step.")
    passed: bool = Field(..., description="Whether the audit criteria for this step were met (score >= 0.7).")
    score: float = Field(..., description="The score assigned for this step, from 0.0 to 1.0.")
    analysis: str = Field(..., description="The detailed analysis of how the conversation performed against the criteria.")
    reasoning: str = Field(..., description="The reasoning behind the assigned score.")
    improvements: Optional[str] = Field(None, description="Suggestions for improvement, if any.")

class AuditReport(BaseModel):
    """The full audit report containing results for all steps."""
    results: List[StepResult] = Field(..., description="A list of results for each audit step performed.")

class RefinedAuditReport(BaseModel):
    """The refined and corrected full audit report, reviewed by a judge model."""
    chain_of_thought: List[str] = Field(..., description="A step-by-step reasoning of the judge model's decisions.")
    results: List[StepResult] = Field(..., description="A list of the corrected results for each audit step performed.")

class EmailAuditor:
    def __init__(self, config_path: str = 'src/email_audit/auditor/audit_config.json'):
        # Define default model names
        DEFAULT_OPENAI_PRIMARY_MODEL = "gpt-4"
        DEFAULT_OPENAI_REASONING_MODEL = "gpt-4"
        DEFAULT_OPENAI_DETAIL_MODEL = "gpt-4" # Though detail_llm isn't directly used by audit_email's main flow
        DEFAULT_ANTHROPIC_PRIMARY_MODEL = "claude-3-opus-20240229"
        DEFAULT_ANTHROPIC_REASONING_MODEL = "claude-3-opus-20240229"
        DEFAULT_ANTHROPIC_DETAIL_MODEL = "claude-3-opus-20240229"
        DEFAULT_GROQ_PRIMARY_MODEL = "llama-3.3-70b-versatile"
        DEFAULT_GROQ_REASONING_MODEL = "llama-3.3-70b-versatile"
        DEFAULT_GROQ_DETAIL_MODEL = "llama-3.3-70b-versatile"
        DEFAULT_JUDGE_MODEL = "claude-3-opus-20240229" # Default judge model

        try:
            logger.debug("Initializing LLMs using LLMFactory...")

            # Primary LLM
            primary_llm_provider = os.getenv('PRIMARY_LLM_PROVIDER', 'anthropic').lower()
            if primary_llm_provider == 'openai':
                default_primary_model = DEFAULT_OPENAI_PRIMARY_MODEL
            elif primary_llm_provider == 'anthropic':
                default_primary_model = DEFAULT_ANTHROPIC_PRIMARY_MODEL
            elif primary_llm_provider == 'groq':
                default_primary_model = DEFAULT_GROQ_PRIMARY_MODEL
            else:
                default_primary_model = DEFAULT_OPENAI_PRIMARY_MODEL
            primary_llm_model_name = os.getenv(
                f'{primary_llm_provider.upper()}_PRIMARY_MODEL',
                default_primary_model
            )
            self.primary_llm: BaseLLM = LLMFactory.create_llm(
                provider=primary_llm_provider,
                model_name=primary_llm_model_name,
                temperature=0.0
            )
            logger.debug(f"Initialized primary_llm with {primary_llm_provider}:{primary_llm_model_name}")

            # Reasoning LLM
            reasoning_llm_provider = 'anthropic' # Force Anthropic for reasoning
            # reasoning_llm_provider = os.getenv('REASONING_LLM_PROVIDER', 'anthropic').lower()
            if reasoning_llm_provider == 'openai':
                default_reasoning_model = DEFAULT_OPENAI_REASONING_MODEL
            elif reasoning_llm_provider == 'anthropic':
                default_reasoning_model = DEFAULT_ANTHROPIC_REASONING_MODEL
            elif reasoning_llm_provider == 'groq':
                default_reasoning_model = DEFAULT_GROQ_REASONING_MODEL
            else:
                default_reasoning_model = DEFAULT_OPENAI_REASONING_MODEL
            reasoning_llm_model_name = os.getenv(
                f'{reasoning_llm_provider.upper()}_REASONING_MODEL',
                default_reasoning_model
            )
            self.reasoning_llm: BaseLLM = LLMFactory.create_llm(
                provider=reasoning_llm_provider,
                model_name=reasoning_llm_model_name,
                temperature=0.3
            )
            logger.debug(f"Initialized reasoning_llm with {reasoning_llm_provider}:{reasoning_llm_model_name}")

            # Detail LLM
            detail_llm_provider = os.getenv('DETAIL_LLM_PROVIDER', 'anthropic').lower()
            if detail_llm_provider == 'openai':
                default_detail_model = DEFAULT_OPENAI_DETAIL_MODEL
            elif detail_llm_provider == 'anthropic':
                default_detail_model = DEFAULT_ANTHROPIC_DETAIL_MODEL
            elif detail_llm_provider == 'groq':
                default_detail_model = DEFAULT_GROQ_DETAIL_MODEL
            else:
                default_detail_model = DEFAULT_OPENAI_DETAIL_MODEL
            detail_llm_model_name = os.getenv(
                f'{detail_llm_provider.upper()}_DETAIL_MODEL',
                default_detail_model
            )
            self.detail_llm: BaseLLM = LLMFactory.create_llm( # This LLM is initialized but not used in the current audit_email flow
                provider=detail_llm_provider,
                model_name=detail_llm_model_name,
                temperature=0.1
            )
            logger.debug(f"Initialized detail_llm with {detail_llm_provider}:{detail_llm_model_name}")

            # Judge LLM
            judge_llm_provider = os.getenv('JUDGE_LLM_PROVIDER', 'anthropic').lower()
            judge_llm_model_name = os.getenv('JUDGE_LLM_MODEL', DEFAULT_JUDGE_MODEL)
            self.judge_llm: BaseLLM = LLMFactory.create_llm(
                provider=judge_llm_provider,
                model_name=judge_llm_model_name,
                temperature=0.0
            )
            logger.debug(f"Initialized judge_llm with {judge_llm_provider}:{judge_llm_model_name}")

            logger.debug("All LLMs initialized successfully via factory")
        except Exception as e:
            logger.error(f"Error initializing LLMs via factory: {str(e)}")
            raise
        
        # Load audit steps from the configuration file
        self.audit_steps = self._load_audit_config(config_path)

    def _load_audit_config(self, config_path: str) -> List[Dict[str, Any]]:
        """Loads audit steps from a JSON config file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Flatten the nested structure into a list of audit steps
            audit_steps = []
            for category, data in config.items():
                for audit in data.get("audits", []):
                    # Add the category to each audit step if it's not already there
                    if "category" not in audit:
                        audit["category"] = category
                    audit_steps.append(audit)
            
            logger.info(f"Successfully loaded and flattened {len(audit_steps)} audit steps from {config_path}")
            return audit_steps
        except FileNotFoundError:
            logger.error(f"Audit config file not found at {config_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {config_path}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading audit config: {e}")
            return []

    async def audit_email(self, html_path: Path) -> Dict[str, Any]:
        """
        Audits an email by directly parsing the HTML file, structuring the content,
        and then performing a comprehensive analysis. This updated method avoids
        browser automation for significant speed improvements.
        
        Args:
            html_path: Path to the HTML file to analyze (from eml-html folder)
            
        Returns:
            Dictionary containing the audit results
        """
        try:
            # Step 1: Direct HTML Parsing (Fast)
            logger.info(f"Directly parsing HTML file: {html_path}")
            if not html_path.exists():
                raise FileNotFoundError(f"HTML file not found at {html_path}")
            if not str(html_path).endswith('.html'):
                raise ValueError(f"Expected HTML file, got {html_path}")

            html_content = html_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            # Use get_text() to extract all text content, which is faster and simpler
            email_text_content = soup.get_text(separator='\n', strip=True)

            # Step 2: Structure the conversation with a reliable, structured LLM call
            logger.info("Structuring conversation from raw text using a LLM...")
            # structuring_llm = self.primary_llm.with_structured_output(EmailConversation) # Old way
            
            structuring_prompt = f"""
            Based on the raw text extracted from an email HTML file, your task is to parse it into a chronological list of email messages. Pay close attention to headers like "From:", "Sent:", "To:", "Cc:", and "Subject:". The messages are typically in reverse chronological order in the text; please return them in chronological order (oldest first).

            Raw Text Content (first 20000 characters):
            ---
            {email_text_content[:20000]}
            ---

            Please return the data as a JSON object conforming to the required schema.
            """
            
            # structured_data = await structuring_llm.ainvoke(structuring_prompt) # Old way
            structured_data = await self.primary_llm.ainvoke(structuring_prompt, schema=EmailConversation)
            if not isinstance(structured_data, EmailConversation):
                logger.error(f"Failed to get structured EmailConversation. Received type: {type(structured_data)}. Content: {structured_data}")
                # Try to parse from text if the model failed to use the tool
                if isinstance(structured_data, str):
                    try:
                        # TODO: This is a fallback, but we should investigate why the model isn't using the tool
                        logger.warning("Falling back to parsing EmailConversation from text.")
                        # structured_data = EmailConversation.model_validate_json(structured_data)
                        structured_data = await self.primary_llm.ainvoke(structuring_prompt, schema=EmailConversation)
                    except ValidationError as e:
                        logger.error(f"Failed to parse EmailConversation. Raw response: {structured_data}")
                        return None, "Failed to parse conversation structure from email."
        
            logger.debug(f"Structured conversation extracted: {structured_data.model_dump_json(indent=2)}")

            # Step 3: Perform the detailed audit using the structured data
            messages = {
                "messages": [
                    {
                        "timestamp": msg.timestamp,
                        "sender": msg.sender,
                        "recipients": [msg.recipient] + msg.cc,
                        "subject": msg.subject,
                        "content": msg.body,
                    }
                    for msg in structured_data.email_conversations
                ]
            }

            audit_prompt = f"""
Analyze the following email conversation based on a comprehensive set of audit criteria.

Conversation History (chronological order):
{json.dumps(messages['messages'], indent=2, default=str)}

Please evaluate the conversation against each of the following audit steps. Call the `structured_output` tool to provide the results in the required format.

Audit Criteria:
"""
            for step in self.audit_steps:
                audit_prompt += f"""
- Step ID: {step['id']}
  - Title: {step['title']}
  - Purpose: {step['purpose']}
  - Prompt: {step['prompt']}
"""
            audit_prompt += """
For each step, provide:
1. A boolean 'passed' field (true if score is >= 0.7).
2. A float 'score' from 0.0 to 1.0.
3. A detailed 'analysis' of what happened.
4. The 'reasoning' for your score.
5. Concrete 'improvements' if applicable.

You must call the `structured_output` function with the results of your analysis.
"""
            
            # structured_llm = self.reasoning_llm.with_structured_output(AuditReport) # Old way

            logger.info("Performing comprehensive audit with a single, structured LLM call...")
            # comprehensive_report = await structured_llm.ainvoke(analysis_task_prompt) # Old way
            initial_audit_report = await self.reasoning_llm.ainvoke(audit_prompt, schema=AuditReport)
            logger.debug(f"Initial audit report generated: {initial_audit_report.model_dump_json(indent=2)}")

            # Step 4: Refine the audit with a "judge" model
            logger.info("Refining the audit with a judge model...")
            refinement_prompt = f"""
You are the Chief Auditor. Your role is to review an initial audit of an email conversation and provide a corrected, final judgment. The initial auditor may have made mistakes, been too lenient, or too harsh.

Your task is to review the original conversation and the initial audit report. Then, generate a refined and corrected version of the report.

**First, provide a step-by-step chain of thought.** For each audit point, explain your reasoning for either agreeing with or changing the initial assessment. Be specific.

**After your chain of thought, provide the final, corrected audit** as a JSON object conforming to the `RefinedAuditReport` schema.

Original Conversation:
```json
{json.dumps(messages, indent=2)}
```

Initial Audit Report to Review:
```json
{initial_audit_report.model_dump_json(indent=2)}
```

Begin your response with your chain of thought, and then provide the final JSON object.
"""
            
            refined_audit_report = await self.judge_llm.ainvoke(refinement_prompt, schema=RefinedAuditReport)
            logger.debug(f"Refined audit report generated: {refined_audit_report.model_dump_json(indent=2)}")

            step_metadata = {step['id']: step for step in self.audit_steps}
            final_audit_results = []
            for result_pydantic in refined_audit_report.results:
                result_dict = result_pydantic.model_dump()
                metadata = step_metadata.get(result_dict['step_id'])
                if metadata:
                    result_dict['is_critical'] = metadata.get('isCritical', False)
                    result_dict['category'] = metadata.get('category', 'Uncategorized')
                    result_dict['max_score'] = metadata.get('max_score', 1.0) # Assuming max_score is in config
                final_audit_results.append(result_dict)

            # Return the final, judge-refined results
            overall_score = sum(r['score'] * r['max_score'] for r in final_audit_results) / sum(r['max_score'] for r in final_audit_results) if final_audit_results else 0
            
            return {
                "score": overall_score,
                "detailed_results": final_audit_results,
                "chain_of_thought": refined_audit_report.chain_of_thought,
                "conversation_history": [msg.model_dump() for msg in structured_data.email_conversations],
                "initial_audit_report": initial_audit_report.model_dump() # For comparison
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred during the audit process: {e}")
            # Consider more specific error handling or re-raising
            raise 