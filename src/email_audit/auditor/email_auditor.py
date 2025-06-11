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

            # Step 2: Perform the detailed audit using the structured data
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
            comprehensive_report = await self.reasoning_llm.ainvoke(audit_prompt, schema=AuditReport)
            if not isinstance(comprehensive_report, AuditReport):
                logger.error(f"Failed to get structured AuditReport. Received type: {type(comprehensive_report)}. Content: {comprehensive_report}")
                # Or, if LLM client returns raw string on parsing error:
                # if isinstance(comprehensive_report, str):
                #    logger.error(f"Failed to parse AuditReport. Raw response: {comprehensive_report}")
                raise ValueError("Could not parse audit report structure from reasoning_llm.")
            
            # Step 3b: Refine the audit with a "judge" LLM
            logger.info("Refining the audit with a judge LLM...")
            
            initial_report_json = json.dumps([result.model_dump() for result in comprehensive_report.results], indent=2)

            judging_prompt = f"""
You are an expert quality assurance auditor. Your task is to review an email conversation and an initial automated audit report.
Your goal is to identify inaccuracies, missed details, or misinterpretations in the first report and produce a more accurate, refined version.

**Original Email Content:**
---
{email_text_content[:20000]}
---

**Initial Audit Report (JSON):**
---
{initial_report_json}
---

**Your Task:**
Carefully compare the initial audit report against the original email content. Pay close attention to context. For example, if the initial report penalizes the agent for not offering a service that was clearly not applicable, you must correct it. Conversely, if the report misses a clear failure by the agent, you must identify and score it correctly.

Provide a refined, corrected version of the full audit report. Ensure your output is a JSON object that conforms to the required schema, containing the complete list of corrected audit steps.
"""
            
            refined_report = await self.judge_llm.ainvoke(judging_prompt, schema=RefinedAuditReport)
            if not isinstance(refined_report, RefinedAuditReport):
                logger.warning(f"Failed to get structured RefinedAuditReport. Falling back to the original report. Received type: {type(refined_report)}")
                # Fallback to the original report if judge fails
                final_comprehensive_report = comprehensive_report
            else:
                logger.info("Successfully refined the audit report.")
                # The judge's output is a list of StepResult, so we create an AuditReport instance from it
                final_comprehensive_report = AuditReport(results=refined_report.results)

            step_metadata = {step['id']: step for step in self.audit_steps}
            audit_results = []
            
            for result_pydantic in final_comprehensive_report.results:
                result_dict = result_pydantic.model_dump()
                metadata = step_metadata.get(result_dict['step_id'])
                
                if metadata:
                    result_dict['is_critical'] = metadata['isCritical']
                    result_dict['category'] = metadata['category']
                    result_dict['max_score'] = 1.0
                    audit_results.append(result_dict)

            # Step 4: Calculate scores and prepare final report
            total_score = sum(res['score'] * res['max_score'] for res in audit_results)
            max_score = len(self.audit_steps)
            overall_score = total_score / max_score if max_score > 0 else 0
            
            return {
                "context": self._extract_context(audit_results),
                "participants": self._extract_participants(messages['messages']),
                "tone": self._analyze_tone(audit_results),
                "security": self._check_security(audit_results),
                "effectiveness": self._assess_effectiveness(audit_results),
                "recommendations": self._generate_recommendations(audit_results),
                "score": overall_score,
                "detailed_results": audit_results,
                "reasoning": self._generate_reasoning(audit_results)
            }
            
        except Exception as e:
            logger.error(f"Error during efficient HTML-based audit: {str(e)}")
            raise
    
    def _extract_context_for_step(self, messages: List[Dict[str, Any]], step: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context for a specific audit step."""
        context = {
            "relevant_messages": [],
            "key_events": [],
            "participants": set()
        }
        
        for message in messages:
            # Add message if it's relevant to the step
            if self._is_message_relevant(message, step):
                context["relevant_messages"].append(message)
                context["participants"].add(message["sender"])
                if message["recipients"]:
                    context["participants"].update(message["recipients"])
                
                # Extract key events
                if self._is_key_event(message, step):
                    context["key_events"].append({
                        "timestamp": message["timestamp"],
                        "type": self._get_event_type(message, step),
                        "description": message["content"][:100] + "..."
                    })
        
        return context
    
    def _is_message_relevant(self, message: Dict[str, Any], step: Dict[str, Any]) -> bool:
        """Determine if a message is relevant to a specific audit step."""
        # Add logic to determine relevance based on step type
        if step["category"] == "PNR":
            return any(keyword in message["content"].lower() for keyword in ["itinerary", "flight", "booking", "reservation"])
        elif step["category"] == "communication":
            return True  # All messages are relevant for communication analysis
        elif step["category"] == "policy and service":
            return any(keyword in message["content"].lower() for keyword in ["policy", "service", "requirement", "visa"])
        return False
    
    def _is_key_event(self, message: Dict[str, Any], step: Dict[str, Any]) -> bool:
        """Determine if a message represents a key event for the audit step."""
        # Add logic to identify key events based on step type
        if step["id"] == "limo_offering":
            return "limo" in message["content"].lower() or "car service" in message["content"].lower()
        elif step["id"] == "transit_visa_advisory":
            return "visa" in message["content"].lower() or "transit" in message["content"].lower()
        return False
    
    def _get_event_type(self, message: Dict[str, Any], step: Dict[str, Any]) -> str:
        """Get the type of event for a message."""
        # Add logic to categorize events based on step type
        if step["id"] == "limo_offering":
            return "limo_service_mentioned"
        elif step["id"] == "transit_visa_advisory":
            return "visa_requirement_discussed"
        return "general_message"
    
    def _generate_reasoning(self, audit_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive reasoning for the audit results."""
        reasoning_parts = []
        
        # Group results by category
        category_results = {}
        for result in audit_results:
            category = result["category"]
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Generate reasoning for each category
        for category, results in category_results.items():
            reasoning_parts.append(f"\n{category.upper()} Analysis:")
            for result in results:
                reasoning_parts.append(f"\n{result['title']}:")
                reasoning_parts.append(f"Score: {result['score']}")
                reasoning_parts.append(f"Analysis: {result['analysis']}")
                if not result["passed"]:
                    reasoning_parts.append(f"Areas for Improvement: {result.get('improvements', 'None specified')}")
        
        return "\n".join(reasoning_parts)
    
    def _extract_context(self, audit_results: List[Dict[str, Any]]) -> str:
        """Extract context from audit results."""
        context_parts = []
        for result in audit_results:
            if result["category"] == "PNR":
                context_parts.append(f"{result['title']}: {result['analysis']}")
        return " | ".join(context_parts) if context_parts else "No specific context found"
    
    def _extract_participants(self, messages: List[Dict[str, Any]]) -> str:
        """Extract participants from email thread."""
        # This is a simple implementation - you might want to enhance it
        return "Participants extracted from email thread"
    
    def _analyze_tone(self, audit_results: List[Dict[str, Any]]) -> str:
        """Analyze the tone from audit results."""
        tone_parts = []
        for result in audit_results:
            if result["category"] == "communication":
                tone_parts.append(f"{result['title']}: {result['analysis']}")
        return " | ".join(tone_parts) if tone_parts else "No tone analysis available"
    
    def _check_security(self, audit_results: List[Dict[str, Any]]) -> str:
        """Check security aspects from audit results."""
        security_parts = []
        for result in audit_results:
            if "sensitive" in result["analysis"].lower() or "security" in result["analysis"].lower():
                security_parts.append(f"{result['title']}: {result['analysis']}")
        return " | ".join(security_parts) if security_parts else "No security concerns found"
    
    def _assess_effectiveness(self, audit_results: List[Dict[str, Any]]) -> str:
        """Assess communication effectiveness from audit results."""
        effectiveness_parts = []
        for result in audit_results:
            if result["category"] in ["communication", "policy and service"]:
                effectiveness_parts.append(f"{result['title']}: {result['analysis']}")
        return " | ".join(effectiveness_parts) if effectiveness_parts else "No effectiveness assessment available"
    
    def _generate_recommendations(self, audit_results: List[Dict[str, Any]]) -> str:
        """Generate recommendations based on audit results."""
        recommendations = []
        for result in audit_results:
            if not result["passed"] and result["is_critical"]:
                recommendations.append(f"Critical: {result['title']} - {result['analysis']}")
            elif not result["passed"]:
                recommendations.append(f"Improvement: {result['title']} - {result['analysis']}")
        return " | ".join(recommendations) if recommendations else "No specific recommendations" 