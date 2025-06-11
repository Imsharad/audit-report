from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

class AttachmentType(Enum):
    """Types of attachments supported by the system."""
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_JPG = "image/jpg"
    APPLICATION_PDF = "application/pdf"
    TEXT_PLAIN = "text/plain"
    TEXT_HTML = "text/html"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    OTHER = "other"

class AttachmentFlag(Enum):
    """Flags for attachment processing status."""
    OVERSIZED = "oversized"
    PASSWORD_PROTECTED = "password_protected"
    UNSUPPORTED_TYPE = "unsupported_type"
    APPROVED = "approved"

class Attachment(BaseModel):
    """Represents a single attachment from an email."""
    filename: Optional[str] = Field(None, description="Original filename of the attachment")
    content_type: str = Field(..., description="MIME content type")
    size_bytes: int = Field(..., description="Size of attachment in bytes")
    content_id: Optional[str] = Field(None, description="Content-ID header if present")
    content_disposition: Optional[str] = Field(None, description="Content-Disposition header")
    part_number: int = Field(..., description="Sequential number of this part in the email")
    flags: List[AttachmentFlag] = Field(default_factory=list, description="Processing flags")
    extracted_path: Optional[Path] = Field(None, description="Path where attachment was saved")
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

class EmailPart(BaseModel):
    """Represents any part of an email (body content or attachment)."""
    part_number: int = Field(..., description="Sequential number of this part")
    content_type: str = Field(..., description="MIME content type") 
    content: Optional[str] = Field(None, description="Text content if applicable")
    filename: Optional[str] = Field(None, description="Filename if this is an attachment")
    size_bytes: int = Field(0, description="Size in bytes")
    is_attachment: bool = Field(False, description="Whether this is an attachment")
    is_body: bool = Field(False, description="Whether this is email body content")
    
class AttachmentManifest(BaseModel):
    """Complete inventory of all email parts and attachments."""
    email_file: str = Field(..., description="Source EML filename")
    total_parts: int = Field(..., description="Total number of MIME parts found")
    body_parts: List[EmailPart] = Field(default_factory=list, description="Email body parts (text/html)")
    attachments: List[Attachment] = Field(default_factory=list, description="All attachments")
    processing_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of processing results")
    
    @property
    def attachment_count(self) -> int:
        """Total number of attachments."""
        return len(self.attachments)
    
    @property
    def approved_attachments(self) -> List[Attachment]:
        """List of attachments approved for processing."""
        return [att for att in self.attachments if AttachmentFlag.APPROVED in att.flags]
    
    @property
    def flagged_attachments(self) -> List[Attachment]:
        """List of attachments with processing concerns."""
        return [att for att in self.attachments 
                if any(flag != AttachmentFlag.APPROVED for flag in att.flags)]

# Approved MIME types for processing
APPROVED_MIME_TYPES = {
    "image/png",
    "image/jpeg", 
    "image/jpg",
    "application/pdf",
    "text/plain",
    "text/html"
}

# Maximum attachment size (10MB)
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024

# Password-protected file heuristics (filename patterns)
PASSWORD_PROTECTED_PATTERNS = [
    "password",
    "encrypted", 
    "protected",
    "secure",
    ".p7m",
    ".p7s"
] 