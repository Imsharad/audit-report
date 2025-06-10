import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(find_dotenv('.env.local'))

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

class EmailAuditor:
    def __init__(self):
        # Get API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Debug logging
        logger.debug(f"API Key present: {bool(openai_api_key)}")
        logger.debug(f"API Key length: {len(openai_api_key) if openai_api_key else 0}")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize multiple models for different aspects of analysis
        try:
            logger.debug("Initializing primary_llm...")
            self.primary_llm = ChatOpenAI(
                model='gpt-4o',
                temperature=0.0,  # For factual analysis
                api_key=openai_api_key
            )
            logger.debug("Initializing reasoning_llm...")
            self.reasoning_llm = ChatOpenAI(
                model='gpt-4o',
                temperature=0.3,  # For nuanced understanding and reasoning
                api_key=openai_api_key
            )
            logger.debug("Initializing detail_llm...")
            self.detail_llm = ChatOpenAI(
                model='gpt-4o',
                temperature=0.1,  # For detailed analysis
                api_key=openai_api_key
            )
            logger.debug("All LLMs initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLMs: {str(e)}")
            raise
        
        # Define audit steps
self.audit_steps = [
    # Category: PNR Fields (and similar)
    {
        "id": "logical_itinerary",
        "title": "Logical Itinerary (Time window, Routing, Connections)",
        "purpose": "Check if the itinerary's time window, routing, and connections are logical and meet client needs.",
        "prompt": "Analyze the email conversation to determine if the proposed/booked itinerary is logical. Consider:\n        - Are flight times reasonable (e.g., not too early/late unless requested)?\n        - Is the routing efficient, or are there unnecessary layovers?\n        - Are connection times adequate (not too short, not excessively long)?\n        - Does it align with any explicitly stated travel needs or constraints by the client?\n        Provide a score from 0.0 to 1.0 (1.0 for perfectly logical, 0.0 for illogical or problematic). Detail any issues found.",
        "isCritical": False,
        "score": 3.00,
        "category": "PNR Fields",
        "model": "reasoning"
    },
    {
        "id": "frequent_flyer_loyalty",
        "title": "Frequent Flyer - Air & Loyalty Membership - Car, Hotels",
        "purpose": "Check if frequent flyer and loyalty memberships for air, car, and hotels were requested and correctly applied.",
        "prompt": "Scan the conversation for mentions of frequent flyer numbers (e.g., for airlines) or loyalty program memberships (for car rentals, hotels).\n        - Was the client asked for these details if not initially provided for a relevant booking?\n        - If provided, is there evidence they were applied to the booking?\n        - If not mentioned or applied, was it appropriate for the services booked?\n        Provide a score from 0.0 to 1.0 (1.0 for perfect handling, 0.0 for missed opportunity or error). Explain your reasoning.",
        "isCritical": False,
        "score": 5.40,
        "category": "PNR Fields",
        "model": "detail"
    },
    {
        "id": "preferences_seat_meal",
        "title": "Preferences – Seat, Meal",
        "purpose": "Check if client's seat and meal preferences were requested and applied.",
        "prompt": "Review the email exchange for discussions about seat preferences (e.g., aisle, window) and meal preferences (e.g., vegetarian, kosher) for flights.\n        - Were these preferences solicited from the client?\n        - If specified by the client, is there confirmation they were applied to the booking?\n        - If not mentioned, was it a missed opportunity for personalization?\n        Provide a score from 0.0 to 1.0 (1.0 for full adherence, 0.0 for omission). Detail your findings.",
        "isCritical": False,
        "score": 5.40,
        "category": "PNR Fields",
        "model": "detail"
    },
    {
        "id": "pnr_documentation_p5h_reference",
        "title": "PNR Documentation in P5H & Reference (Receive Field)",
        "purpose": "Check for correct PNR documentation, specifically P5H and Receive Field references.",
        "prompt": "Examine the conversation, especially agent's messages, for internal PNR documentation notes or references. Specifically look for:\n        - Mentions of P5H updates or requirements.\n        - Use of a 'Receive Field' or similar mechanism for tracking client communication or requests within the PNR.\n        - Is there any indication that standard PNR documentation practices were followed or missed?\n        This might require inferring from agent's technical language or system interaction hints. Provide a score from 0.0 to 1.0. Explain the basis of your score.",
        "isCritical": False,
        "score": 3.00,
        "category": "PNR Fields",
        "model": "detail"
    },
    # Category: Client Policy and Service
    {
        "id": "limo_offering_address_phone",
        "title": "Offered Limo where applicable and Ensuring complete address & phone number",
        "purpose": "Check if limo service was offered where applicable, and if so, was complete address and phone number captured.",
        "prompt": "Analyze the conversation to determine if limo service was applicable (e.g., based on flight class, client policy, or specific client mentioned in emails).\n        - If applicable, was limo service proactively offered by the agent?\n        - If the client accepted or inquired about limo service, did the agent request and confirm a complete pickup/drop-off address and a valid phone number?\n        - If not applicable, note that.\n        Provide a score from 0.0 to 1.0 (1.0 if handled perfectly, 0.0 if missed or handled poorly). Explain your reasoning, including applicability.",
        "isCritical": True,
        "score": 5.40,
        "category": "Client Policy and Service",
        "model": "reasoning"
    },
    {
        "id": "transit_visa_advisory",
        "title": "Transit Visa advisory",
        "purpose": "Check if transit visa requirements were properly advised for any layovers.",
        "prompt": "Review the itinerary details discussed or booked in the email conversation.\n        - Identify all layover/transit points.\n        - Was the client advised about potential transit visa requirements for these layovers?\n        - Was the advice accurate and clear?\n        - If no layovers, or if visa requirements are not applicable (e.g., domestic travel within a visa-free zone for the client's nationality - if known), note that.\n        Provide a score from 0.0 to 1.0 (1.0 for accurate and proactive advice, 0.0 for missing or incorrect advice). Detail findings.",
        "isCritical": True,
        "score": 5.40,
        "category": "Client Policy and Service",
        "model": "reasoning"
    },
    {
        "id": "apptivo_case_communication",
        "title": "Captured correct Apptivo case number & used Apptivo for communication",
        "purpose": "Check if the correct Apptivo case number was captured and if Apptivo was used for communication.",
        "prompt": "Examine the email conversation for:\n        - Explicit mentions or references to an Apptivo case number. Is it correctly formatted or referenced consistently?\n        - Indications that Apptivo (or a similar CRM/case management tool by that name) is the system of record for communications (e.g., \"I've updated the case in Apptivo\").\n        - Is there any discrepancy or lack of clarity regarding the case number?\n        Provide a score from 0.0 to 1.0. Explain your assessment.",
        "isCritical": False,
        "score": 3.00,
        "category": "Client Policy and Service",
        "model": "detail"
    },
    {
        "id": "options_class_as_per_policy",
        "title": "Options & class of service as per client policy",
        "purpose": "Check if flight/service options and class of service align with client policy.",
        "prompt": "Review the options provided to the client (e.g., flight choices, hotel grades, car types) and the class of service.\n        - Is there any mention of a specific client travel policy (e.g., \"must be economy class,\" \"preferred airlines are X, Y, Z\")?\n        - Do the provided options and booked services adhere to this policy?\n        - If no explicit policy is mentioned, were reasonable and standard options provided? Were multiple options offered if appropriate?\n        Provide a score from 0.0 to 1.0 (1.0 for full compliance/reasonableness, 0.0 for deviation or lack of options). Explain.",
        "isCritical": True,
        "score": 5.40,
        "category": "Client Policy and Service",
        "model": "reasoning"
    },
    {
        "id": "noshow_cancellation_advice",
        "title": "No Show / Cancellations / Changes to be advised correctly",
        "purpose": "Check if advice on no-show, cancellation, and change policies was correctly provided.",
        "prompt": "Analyze the conversation for any discussion related to booking changes, cancellations, or potential no-shows.\n        - Was the client clearly informed about the rules, fees, and implications associated with changes, cancellations, or no-shows for their specific booking?\n        - Was this information provided proactively, especially for restrictive tickets/bookings?\n        Provide a score from 0.0 to 1.0 (1.0 for clear and correct advice, 0.0 for missing or incorrect advice). Detail your findings.",
        "isCritical": True,
        "score": 5.40,
        "category": "Client Policy and Service",
        "model": "reasoning"
    },
    {
        "id": "corporate_deal_application",
        "title": "Corporate deal application for Air/Car/Hotel",
        "purpose": "Check if applicable corporate deals for air, car, or hotel were applied.",
        "prompt": "Scan the emails for mentions of corporate rates, client-specific deals, or negotiated fares/perks for air travel, car rentals, or hotel stays.\n        - Is there evidence that such deals were considered or applied if available to the client/company?\n        - If a corporate deal was mentioned or implied as potentially applicable, was it correctly factored into the quotation or booking?\n        Provide a score from 0.0 to 1.0. Explain your reasoning.",
        "isCritical": False,
        "score": 3.00,
        "category": "Client Policy and Service",
        "model": "detail"
    },
    {
        "id": "client_specifics_sop",
        "title": "Client Specifics (if any) & Standard Operating Procedure related",
        "purpose": "Check adherence to any mentioned client-specific needs or general SOPs.",
        "prompt": "Review the entire conversation for:\n        - Any explicit client-specific requests, preferences, or instructions that go beyond standard service (e.g., \"always book me on carrier X,\" \"my department code is Y\"). Were these acknowledged and addressed?\n        - Any indications of adherence to, or deviation from, general Standard Operating Procedures (SOPs) for a travel agency (this might be inferred from professionalism, process steps mentioned, etc.).\n        Provide a score from 0.0 to 1.0. This is a general assessment; focus on explicitly mentioned client needs first. Explain.",
        "isCritical": False,
        "score": 3.00,
        "category": "Client Policy and Service",
        "model": "reasoning"
    },
    {
        "id": "din_entry_applicable",
        "title": "DIN entry as applicable",
        "purpose": "Check if DIN (Delegated Identification Number or similar) entry was made if applicable.",
        "prompt": "Analyze the agent's communications for any mention or requirement of a DIN entry or similar specific identifier related to the booking or client profile.\n        - If the context suggests a DIN or special code should be recorded (e.g., for corporate tracking, specific client type), is there any sign it was done?\n        - This is a specialized check; if no information, assume not applicable unless context implies otherwise.\n        Provide a score from 0.0 to 1.0. Explain if evidence is found or why it's deemed not applicable.",
        "isCritical": False,
        "score": 5.40,
        "category": "Client Policy and Service",
        "model": "detail"
    },
    # Category: Accounting
    {
        "id": "applied_commission",
        "title": "Applied Commission as applicable (Retained / Parted)",
        "purpose": "Check if commission was applied correctly (e.g., retained or parted) as per requirements.",
        "prompt": "Review the conversation for any details or agent notes related to commission on the booking.\n        - Is there any indication of how commission was handled (e.g., retained by agency, parted with another entity)?\n        - Does the handling seem appropriate given the context (if enough context exists)?\n        This is often an internal detail. If no information is present, score based on whether any red flags appear or if it seems unaddressed where it should be. Provide a score from 0.0 to 1.0. Explain.",
        "isCritical": True,
        "score": 5.40,
        "category": "Accounting",
        "model": "detail"
    },
    {
        "id": "accurate_reissuance",
        "title": "Process accurate re-issuance (Tax code, add collect, WFRF, WFR)",
        "purpose": "Check if any ticket re-issuance was processed accurately, including tax codes, additional collections, WFRF, WFR.",
        "prompt": "If the conversation involves a ticket re-issuance (exchange or modification):\n        - Were correct tax codes applied?\n        - Was any additional collection of fare/fees handled properly?\n        - Are there mentions of WFRF (Waiver and Favor Requisition Form) or WFR (Waiver and Favor Record) if applicable for the change?\n        - Does the re-issuance process appear to be accurate based on the details provided?\n        If no re-issuance, mark as not applicable. Provide a score from 0.0 to 1.0 for accuracy if re-issuance occurred. Explain.",
        "isCritical": True,
        "score": 5.40,
        "category": "Accounting",
        "model": "detail"
    },
    {
        "id": "correct_form_of_payment",
        "title": "Correct form of payment",
        "purpose": "Check if the correct form of payment was used/captured as per client instructions or policy.",
        "prompt": "Analyze the conversation for details about the form of payment (FOP).\n        - Was the client asked for their preferred FOP?\n        - If a specific FOP was requested (e.g., \"use my Amex ending in 1234,\" \"bill to company account\"), was it acknowledged and seemingly used?\n        - Are there any discrepancies or issues noted regarding payment?\n        Provide a score from 0.0 to 1.0 (1.0 for correct handling, 0.0 for errors or omissions). Explain.",
        "isCritical": True,
        "score": 5.40,
        "category": "Accounting",
        "model": "detail"
    },
    {
        "id": "accurate_accounting_line_pac",
        "title": "Process accurate accounting line - PAC",
        "purpose": "Check if the accounting line (PAC) was processed accurately.",
        "prompt": "Review agent's notes or system references for mentions of an accounting line, PAC (Passenger Account Control), or similar financial reconciliation codes.\n        - Is there evidence that the correct accounting information was recorded for the transaction?\n        - This is highly internal. If no specific information, assess if any related financial details seem misaligned.\n        Provide a score from 0.0 to 1.0. Explain your reasoning.",
        "isCritical": True,
        "score": 3.00,
        "category": "Accounting",
        "model": "detail"
    },
    {
        "id": "correct_pos_indicator_fee",
        "title": "Correct POS indicator & POS fee",
        "purpose": "Check for correct Point of Sale (POS) indicator and application of any POS fees.",
        "prompt": "Examine the context for clues about the Point of Sale (e.g., country of booking).\n        - Is there any indication that the POS indicator was set correctly?\n        - If POS-specific fees apply, were they accounted for?\n        This is technical; if no information, assume neutral unless errors in related areas (like currency or taxes) suggest a POS issue. Provide a score from 0.0 to 1.0. Explain.",
        "isCritical": False,
        "score": 3.00,
        "category": "Accounting",
        "model": "detail"
    },
    {
        "id": "correct_service_fee_selection",
        "title": "Correct Service fee selection (Air, Car, Hotel, Other)",
        "purpose": "Check if the correct type and amount of service fee was applied for the services booked.",
        "prompt": "Analyze the quotation and booking details for service fees.\n        - Was a service fee applied? If so, for which services (Air, Car, Hotel, Other)?\n        - Does the type and amount of service fee seem appropriate for the transaction and stated policies (if any)?\n        - Was the service fee clearly communicated to the client?\n        Provide a score from 0.0 to 1.0 (1.0 for correct and clear fee application, 0.0 for errors or lack of clarity). Explain.",
        "isCritical": True,
        "score": 5.40,
        "category": "Accounting",
        "model": "reasoning"
    },
    # Category: Communication
    {
        "id": "overall_communication_email",
        "title": "Overall communication in the email",
        "purpose": "Assess the overall quality of communication in the email exchange.",
        "prompt": "Evaluate the overall communication quality from the agent. Consider:\n        - Clarity and conciseness of information.\n        - Professionalism and tone.\n        - Empathy and helpfulness, especially if issues arose.\n        - Grammar and spelling.\n        - Timeliness of responses (if discernible).\n        - Avoidance of jargon where possible, or clear explanation if used.\n        - Does the agent guide the client effectively?\n        Provide a score from 0.0 to 1.0 (1.0 for excellent communication, 0.0 for poor communication). Summarize strengths and weaknesses.",
        "isCritical": False,
        "score": 3.00,
        "category": "Communication",
        "model": "reasoning"
    },
    {
        "id": "reason_codes_low_fare_script",
        "title": "Capture accurate reason codes (Missed/Realized) & Low Fare in the script",
        "purpose": "Check if accurate reason codes (Missed/Realized Savings) and Low Fare information were captured/communicated.",
        "prompt": "Review the conversation for:\n        - Mentions of reason codes, especially for 'missed savings' or 'realized savings'.\n        - Discussion or use of a 'low fare script' or similar process to justify fare choices.\n        - Was the client informed if a lower fare was available but not chosen due to policy or preference?\n        - If a low fare was found, was it highlighted?\n        Provide a score from 0.0 to 1.0. Explain findings.",
        "isCritical": False,
        "score": 3.00,
        "category": "Communication",
        "model": "reasoning"
    },
    {
        "id": "cwt_itinerary_clipboard_sabre_format",
        "title": "Used CWT Itinerary / Clipboard / Sabre format as per client",
        "purpose": "Check if the itinerary was presented in the client-specified format (e.g., CWT Itinerary, Clipboard, Sabre).",
        "prompt": "Examine how itinerary information was presented to the client.\n        - Is there any mention of a required format (e.g., \"please send in CWT format,\" \"use Sabre clipboard text\")?\n        - Does the presented itinerary seem to follow a structured, standard travel agency format?\n        - If a specific format was requested, was it used?\n        Provide a score from 0.0 to 1.0. Explain.",
        "isCritical": False,
        "score": 5.40,
        "category": "Communication",
        "model": "detail"
    },
    {
        "id": "cross_upsell_opportunity",
        "title": "Utilized cross sell & up sell opportunity (Hotel, Car, Insurance)",
        "purpose": "Check if the agent attempted to cross-sell or up-sell relevant services.",
        "prompt": "Analyze the client's request and the services booked.\n        - If the primary request was for flights, did the agent proactively offer or inquire about hotel, car rental, or travel insurance needs?\n        - If the client booked a basic service, was there an appropriate attempt to offer an upgrade or enhanced option (upsell)?\n        - This does not apply if the client explicitly stated they *only* need one service or if they initiated requests for all components.\n        Provide a score from 0.0 to 1.0 (1.0 for good attempt, 0.0 for missed opportunity). Explain.",
        "isCritical": False,
        "score": 3.00,
        "category": "Communication",
        "model": "reasoning"
    },
    {
        "id": "quotation_based_on_request",
        "title": "Quotation based on request (Date, Time, City Pair)",
        "purpose": "Check if the quotation accurately reflects the client's request regarding date, time, and city pair.",
        "prompt": "Compare the client's initial travel request (dates, times, origin, destination) with the quotation provided by the agent.\n        - Does the quotation directly address the client's specified parameters?\n        - If alternatives were offered, was it clear why, and was the original request also addressed or acknowledged?\n        - Are there any discrepancies in dates, times, or locations between the request and the quote?\n        Provide a score from 0.0 to 1.0 (1.0 for perfect alignment, 0.0 for significant mismatch). Detail any deviations.",
        "isCritical": True,
        "score": 5.40,
        "category": "Communication",
        "model": "detail"
    }
]
    
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
            structuring_llm = self.primary_llm.with_structured_output(EmailConversation)
            
            structuring_prompt = f"""
            Based on the raw text extracted from an email HTML file, your task is to parse it into a chronological list of email messages. Pay close attention to headers like "From:", "Sent:", "To:", "Cc:", and "Subject:". The messages are typically in reverse chronological order in the text; please return them in chronological order (oldest first).

            Raw Text Content (first 20000 characters):
            ---
            {email_text_content[:20000]}
            ---

            Please return the data as a JSON object conforming to the required schema.
            """
            
            structured_data = await structuring_llm.ainvoke(structuring_prompt)

            messages = {
                "messages": [
                    {
                        "timestamp": msg.timestamp,
                        "sender": msg.sender,
                        "recipients": [msg.recipient] + msg.cc,
                        "subject": msg.subject,
                        "content": msg.body,
                        "attachments": [],  # Assuming no attachment parsing for now
                        "images": []        # Assuming no image parsing for now
                    }
                    for msg in structured_data.email_conversations
                ]
            }
            logger.info(f"Successfully structured {len(messages['messages'])} messages.")

            # Step 3: Perform the comprehensive audit (this part is already efficient)
            logger.info("Building comprehensive analysis prompt...")
            analysis_task_prompt = f"""
Analyze the following email conversation based on a comprehensive set of audit criteria.

Conversation History (chronological order):
{json.dumps(messages['messages'], indent=2, default=str)}

Please evaluate the conversation against each of the following audit steps and provide a structured JSON response that conforms to the required format.

Audit Criteria:
"""
            for step in self.audit_steps:
                analysis_task_prompt += f"""
- Step ID: {step['id']}
  - Title: {step['title']}
  - Purpose: {step['purpose']}
  - Prompt: {step['prompt']}
"""
            analysis_task_prompt += """
For each step, provide:
1. A boolean 'passed' field (true if score is >= 0.7).
2. A float 'score' from 0.0 to 1.0.
3. A detailed 'analysis' of what happened.
4. The 'reasoning' for your score.
5. Concrete 'improvements' if applicable.
"""
            structured_llm = self.reasoning_llm.with_structured_output(AuditReport)

            logger.info("Performing comprehensive audit with a single, structured LLM call...")
            comprehensive_report = await structured_llm.ainvoke(analysis_task_prompt)
            
            step_metadata = {step['id']: step for step in self.audit_steps}
            audit_results = []
            
            for result_pydantic in comprehensive_report.results:
                result_dict = result_pydantic.model_dump()
                metadata = step_metadata.get(result_dict['step_id'])
                
                if metadata:
                    result_dict['is_critical'] = metadata['isCritical']
                    result_dict['category'] = metadata['category']
                    result_dict['max_score'] = metadata['score']
                    audit_results.append(result_dict)

            # Step 4: Calculate scores and prepare final report
            total_score = sum(res['max_score'] for res in audit_results if res['passed'])
            max_score = sum(step['score'] for step in self.audit_steps)
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
                "reasoning": self._generate_reasoning(audit_results),
                "conversation_history": messages['messages']
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