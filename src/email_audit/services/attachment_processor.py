from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from loguru import logger
import json

from ..parser.models import Attachment, AttachmentFlag, AttachmentManifest
from ..llm.base_llm import BaseLLM, MultimodalPrompt, MediaInput
from pydantic import BaseModel, Field

class AttachmentAnalysis(BaseModel):
    """Results of analyzing a single attachment."""
    filename: str
    content_type: str
    size_bytes: int
    analysis_type: str  # "image", "document", "text", "skipped"
    findings: List[str] = Field(default_factory=list)
    security_concerns: List[str] = Field(default_factory=list)
    extracted_text: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_ms: int = 0

class AttachmentBatchAnalysis(BaseModel):
    """Results of analyzing a batch of attachments."""
    total_attachments: int
    processed_attachments: int
    skipped_attachments: int
    analyses: List[AttachmentAnalysis] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    total_processing_time_ms: int = 0

class AttachmentProcessor:
    """Service for processing email attachments with LLM analysis."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.image_analysis_prompt = self._get_image_analysis_prompt()
        self.document_analysis_prompt = self._get_document_analysis_prompt()
    
    async def process_attachments(self, manifest: AttachmentManifest, case_dir: Path) -> AttachmentBatchAnalysis:
        """Process all approved attachments in a manifest."""
        start_time = asyncio.get_event_loop().time()
        
        # Filter to approved attachments only
        approved_attachments = manifest.approved_attachments
        
        batch_analysis = AttachmentBatchAnalysis(
            total_attachments=len(manifest.attachments),
            processed_attachments=0,
            skipped_attachments=len(manifest.attachments) - len(approved_attachments)
        )
        
        # Process attachments in parallel (but limit concurrency)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent processing
        tasks = []
        
        for attachment in approved_attachments:
            task = self._process_single_attachment_with_semaphore(
                semaphore, attachment, case_dir
            )
            tasks.append(task)
        
        # Wait for all processing to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, AttachmentAnalysis):
                batch_analysis.analyses.append(result)
                batch_analysis.processed_attachments += 1
            else:
                logger.error(f"Error processing attachment: {result}")
        
        # Generate summary
        end_time = asyncio.get_event_loop().time()
        batch_analysis.total_processing_time_ms = int((end_time - start_time) * 1000)
        batch_analysis.summary = self._generate_batch_summary(batch_analysis)
        
        logger.info(f"Processed {batch_analysis.processed_attachments}/{batch_analysis.total_attachments} attachments")
        
        return batch_analysis
    
    async def _process_single_attachment_with_semaphore(
        self, semaphore: asyncio.Semaphore, attachment: Attachment, case_dir: Path
    ) -> AttachmentAnalysis:
        """Process single attachment with concurrency control."""
        async with semaphore:
            return await self._process_single_attachment(attachment, case_dir)
    
    async def _process_single_attachment(self, attachment: Attachment, case_dir: Path) -> AttachmentAnalysis:
        """Process a single attachment."""
        start_time = asyncio.get_event_loop().time()
        
        analysis = AttachmentAnalysis(
            filename=attachment.filename or f"attachment_{attachment.part_number}",
            content_type=attachment.content_type,
            size_bytes=attachment.size_bytes,
            analysis_type="skipped"
        )
        
        try:
            # Determine analysis type
            if attachment.content_type.startswith("image/"):
                analysis.analysis_type = "image"
                await self._analyze_image(attachment, analysis)
            elif attachment.content_type == "application/pdf":
                analysis.analysis_type = "document"
                await self._analyze_document(attachment, analysis)
            elif attachment.content_type.startswith("text/"):
                analysis.analysis_type = "text"
                await self._analyze_text(attachment, analysis)
            else:
                analysis.analysis_type = "skipped"
                analysis.findings.append("Unsupported content type for analysis")
                
        except Exception as e:
            logger.error(f"Error analyzing {attachment.filename}: {e}")
            analysis.security_concerns.append(f"Processing error: {str(e)}")
        
        # Calculate processing time
        end_time = asyncio.get_event_loop().time()
        analysis.processing_time_ms = int((end_time - start_time) * 1000)
        
        return analysis
    
    async def _analyze_image(self, attachment: Attachment, analysis: AttachmentAnalysis) -> None:
        """Analyze an image attachment using multimodal LLM."""
        if not self.llm.supports_multimodal:
            analysis.findings.append("LLM does not support image analysis")
            return
        
        if not attachment.extracted_path or not attachment.extracted_path.exists():
            analysis.security_concerns.append("Image file not found or not extracted")
            return
        
        try:
            # Create multimodal prompt
            media_input = MediaInput(
                type="image",
                path=attachment.extracted_path,
                content_type=attachment.content_type,
                description=f"Email attachment: {attachment.filename}"
            )
            
            prompt = MultimodalPrompt(
                text=self.image_analysis_prompt,
                media_inputs=[media_input]
            )
            
            # Get analysis from LLM
            response = await self.llm.ainvoke_multimodal(prompt)
            
            if response:
                # Parse the response (assuming structured JSON response)
                if isinstance(response, str):
                    try:
                        parsed = json.loads(response)
                        analysis.findings = parsed.get("findings", [])
                        analysis.security_concerns = parsed.get("security_concerns", [])
                        analysis.extracted_text = parsed.get("extracted_text")
                        analysis.confidence_score = parsed.get("confidence_score", 0.5)
                    except json.JSONDecodeError:
                        # Fallback to raw text analysis
                        analysis.findings = [response[:500]]  # Truncate if too long
                        analysis.confidence_score = 0.3
                else:
                    # Handle structured response
                    if hasattr(response, 'findings'):
                        analysis.findings = response.findings
                    if hasattr(response, 'security_concerns'):
                        analysis.security_concerns = response.security_concerns
                
                logger.debug(f"Analyzed image {attachment.filename}: {len(analysis.findings)} findings")
            
        except Exception as e:
            logger.error(f"Error in image analysis for {attachment.filename}: {e}")
            analysis.security_concerns.append(f"Image analysis failed: {str(e)}")
    
    async def _analyze_document(self, attachment: Attachment, analysis: AttachmentAnalysis) -> None:
        """Analyze a document attachment (placeholder for now)."""
        # For PDF analysis, we would need additional libraries like PyPDF2 or similar
        # For now, basic metadata analysis
        analysis.findings.append("PDF document detected")
        analysis.findings.append(f"File size: {attachment.size_bytes} bytes")
        
        # Basic security checks
        if attachment.size_bytes > 50 * 1024 * 1024:  # 50MB
            analysis.security_concerns.append("Large PDF file - potential resource consumption")
        
        analysis.confidence_score = 0.6
    
    async def _analyze_text(self, attachment: Attachment, analysis: AttachmentAnalysis) -> None:
        """Analyze a text attachment."""
        if not attachment.extracted_path or not attachment.extracted_path.exists():
            analysis.security_concerns.append("Text file not found or not extracted")
            return
        
        try:
            # Read text content
            text_content = attachment.extracted_path.read_text(encoding='utf-8', errors='ignore')
            analysis.extracted_text = text_content[:1000]  # First 1000 chars
            
            # Basic text analysis
            line_count = len(text_content.split('\n'))
            char_count = len(text_content)
            
            analysis.findings.append(f"Text file with {line_count} lines, {char_count} characters")
            
            # Look for potential security concerns in text
            suspicious_patterns = [
                "password", "credential", "api_key", "secret", "token",
                "confidential", "private", "internal", "restricted"
            ]
            
            text_lower = text_content.lower()
            found_patterns = [pattern for pattern in suspicious_patterns if pattern in text_lower]
            
            if found_patterns:
                analysis.security_concerns.append(f"Potentially sensitive content detected: {', '.join(found_patterns)}")
            
            analysis.confidence_score = 0.8
            
        except Exception as e:
            logger.error(f"Error analyzing text file {attachment.filename}: {e}")
            analysis.security_concerns.append(f"Text analysis failed: {str(e)}")
    
    def _generate_batch_summary(self, batch_analysis: AttachmentBatchAnalysis) -> Dict[str, Any]:
        """Generate summary statistics for the batch analysis."""
        total_findings = sum(len(a.findings) for a in batch_analysis.analyses)
        total_concerns = sum(len(a.security_concerns) for a in batch_analysis.analyses)
        avg_confidence = sum(a.confidence_score for a in batch_analysis.analyses) / max(len(batch_analysis.analyses), 1)
        
        analysis_types = {}
        for analysis in batch_analysis.analyses:
            analysis_types[analysis.analysis_type] = analysis_types.get(analysis.analysis_type, 0) + 1
        
        return {
            "total_findings": total_findings,
            "total_security_concerns": total_concerns,
            "average_confidence_score": round(avg_confidence, 2),
            "analysis_types": analysis_types,
            "processing_time_per_attachment_ms": batch_analysis.total_processing_time_ms // max(batch_analysis.processed_attachments, 1)
        }
    
    def _get_image_analysis_prompt(self) -> str:
        """Get the prompt template for image analysis."""
        return """
Analyze this email attachment image for the following:

1. Content description: What does the image show?
2. Text extraction: Extract any readable text from the image
3. Security concerns: Look for:
   - Screenshots of sensitive information (credentials, personal data)
   - Suspicious QR codes or links
   - Phishing attempts or fake interfaces
   - Inappropriate or offensive content
4. Email context: How might this image be used in an email attack?

Please respond with JSON in this format:
{
    "findings": ["List of key observations about the image"],
    "security_concerns": ["List of potential security issues"],
    "extracted_text": "Any text found in the image",
    "confidence_score": 0.95
}
"""
    
    def _get_document_analysis_prompt(self) -> str:
        """Get the prompt template for document analysis."""
        return """
Analyze this email attachment document for:

1. Document type and structure
2. Potential security risks
3. Content summary
4. Metadata concerns

Respond with JSON format as specified in image analysis.
""" 