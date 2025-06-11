from pathlib import Path
from typing import Optional, Tuple
import email
from email import policy
from email.message import Message
from bs4 import BeautifulSoup
from loguru import logger
import base64
import os

from .models import (
    AttachmentManifest, 
    Attachment, 
    EmailPart, 
    AttachmentFlag,
    APPROVED_MIME_TYPES,
    MAX_ATTACHMENT_SIZE,
    PASSWORD_PROTECTED_PATTERNS
)

class EMLParser:
    def __init__(self):
        # Create a new policy with no line length limit
        self.policy = policy.default.clone(
            max_line_length=None  # This is the correct way to set no line length limit
        )
    
    def parse_with_attachments(self, eml_path: Path, output_dir: Path) -> Tuple[str, AttachmentManifest]:
        """
        Parse an EML file with full MIME awareness, extracting attachments and generating manifest.
        
        Args:
            eml_path: Path to the EML file
            output_dir: Directory to save extracted attachments
            
        Returns:
            Tuple of (HTML content, AttachmentManifest)
        """
        try:
            # Read and parse the EML file
            with open(eml_path, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=self.policy)
            
            # Create attachments directory
            attachments_dir = output_dir / "attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize manifest
            manifest = AttachmentManifest(
                email_file=eml_path.name,
                total_parts=0,
                processing_summary={}
            )
            
            # Process all MIME parts
            part_number = 0
            html_content = None
            
            for part in msg.walk():
                part_number += 1
                content_type = part.get_content_type()
                
                # Process body parts
                if content_type in ["text/html", "text/plain"]:
                    email_part = EmailPart(
                        part_number=part_number,
                        content_type=content_type,
                        content=part.get_content(),
                        size_bytes=len(str(part)),
                        is_body=True
                    )
                    manifest.body_parts.append(email_part)
                    
                    # Extract HTML for primary processing
                    if content_type == "text/html" and html_content is None:
                        html_content = part.get_content()
                
                # Process attachments
                elif part.get_filename() or self._is_attachment(part):
                    attachment = self._process_attachment(part, part_number, attachments_dir)
                    if attachment:
                        manifest.attachments.append(attachment)
            
            manifest.total_parts = part_number
            
            # If no HTML content found, convert text to HTML
            if not html_content:
                text_content = self._get_text_content(manifest.body_parts)
                html_content = self._text_to_html(text_content)
            
            # Generate processing summary
            manifest.processing_summary = self._generate_processing_summary(manifest)
            
            logger.info(f"Parsed {eml_path.name}: {manifest.total_parts} parts, {manifest.attachment_count} attachments")
            
            return html_content, manifest
            
        except Exception as e:
            logger.error(f"Error parsing EML file {eml_path}: {str(e)}")
            raise
    
    def convert_to_html(self, eml_path: Path) -> str:
        """
        Legacy method for backward compatibility.
        Convert an EML file to HTML format without attachment processing.
        """
        try:
            # Read and parse the EML file
            with open(eml_path, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=self.policy)
            
            # Get the HTML content
            html_content = self._extract_html_content(msg)
            if not html_content:
                # If no HTML content found, convert text to HTML
                text_content = self._extract_text_content(msg)
                html_content = self._text_to_html(text_content)
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error parsing EML file {eml_path}: {str(e)}")
            raise
    
    def _process_attachment(self, part: Message, part_number: int, output_dir: Path) -> Optional[Attachment]:
        """Process a single attachment part."""
        try:
            filename = part.get_filename()
            content_type = part.get_content_type()
            
            # Get content disposition and ID
            content_disposition = part.get('Content-Disposition', '')
            content_id = part.get('Content-ID', None)
            
            # Get attachment content
            content = part.get_payload(decode=True)
            if not content:
                logger.warning(f"No content found for attachment part {part_number}")
                return None
            
            size_bytes = len(content)
            
            # Create attachment object
            attachment = Attachment(
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                content_id=content_id,
                content_disposition=content_disposition,
                part_number=part_number
            )
            
            # Apply processing flags
            self._apply_attachment_flags(attachment)
            
            # Save if approved
            if AttachmentFlag.APPROVED in attachment.flags:
                safe_filename = self._sanitize_filename(filename or f"attachment_{part_number}")
                attachment_path = output_dir / safe_filename
                
                # Handle filename conflicts
                counter = 1
                while attachment_path.exists():
                    name_parts = safe_filename.rsplit('.', 1)
                    if len(name_parts) == 2:
                        safe_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        safe_filename = f"{safe_filename}_{counter}"
                    attachment_path = output_dir / safe_filename
                    counter += 1
                
                # Write attachment
                attachment_path.write_bytes(content)
                attachment.extracted_path = attachment_path
                logger.info(f"Extracted attachment: {safe_filename} ({size_bytes} bytes)")
            
            return attachment
            
        except Exception as e:
            logger.error(f"Error processing attachment part {part_number}: {str(e)}")
            return None
    
    def _apply_attachment_flags(self, attachment: Attachment) -> None:
        """Apply processing flags to an attachment based on size, type, and security heuristics."""
        flags = []
        
        # Check size
        if attachment.size_bytes > MAX_ATTACHMENT_SIZE:
            flags.append(AttachmentFlag.OVERSIZED)
        
        # Check content type
        if attachment.content_type not in APPROVED_MIME_TYPES:
            flags.append(AttachmentFlag.UNSUPPORTED_TYPE)
        
        # Check for password protection heuristics
        if attachment.filename:
            filename_lower = attachment.filename.lower()
            if any(pattern in filename_lower for pattern in PASSWORD_PROTECTED_PATTERNS):
                flags.append(AttachmentFlag.PASSWORD_PROTECTED)
        
        # If no blocking flags, approve for processing
        blocking_flags = {AttachmentFlag.OVERSIZED, AttachmentFlag.UNSUPPORTED_TYPE, AttachmentFlag.PASSWORD_PROTECTED}
        if not any(flag in blocking_flags for flag in flags):
            flags.append(AttachmentFlag.APPROVED)
        
        attachment.flags = flags
    
    def _is_attachment(self, part: Message) -> bool:
        """Determine if a message part is an attachment."""
        content_disposition = part.get('Content-Disposition', '')
        return 'attachment' in content_disposition.lower()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage."""
        # Remove dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
        
        # Ensure it's not empty and doesn't start with dot
        if not sanitized or sanitized.startswith('.'):
            sanitized = f"attachment_{sanitized}"
        
        return sanitized
    
    def _get_text_content(self, body_parts: list) -> str:
        """Extract text content from body parts."""
        for part in body_parts:
            if part.content_type == "text/plain" and part.content:
                return part.content
        return ""
    
    def _generate_processing_summary(self, manifest: AttachmentManifest) -> dict:
        """Generate processing summary for the manifest."""
        return {
            "total_attachments": manifest.attachment_count,
            "approved_attachments": len(manifest.approved_attachments),
            "flagged_attachments": len(manifest.flagged_attachments),
            "body_parts": len(manifest.body_parts),
            "flag_breakdown": self._get_flag_breakdown(manifest.attachments)
        }
    
    def _get_flag_breakdown(self, attachments: list) -> dict:
        """Get breakdown of attachment flags."""
        breakdown = {}
        for attachment in attachments:
            for flag in attachment.flags:
                breakdown[flag.value] = breakdown.get(flag.value, 0) + 1
        return breakdown

    def _extract_html_content(self, msg: email.message.Message) -> Optional[str]:
        """Extract HTML content from the email message."""
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                return part.get_content()
        return None
    
    def _extract_text_content(self, msg: email.message.Message) -> str:
        """Extract text content from the email message."""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_content()
        return ""
    
    def _text_to_html(self, text: str) -> str:
        """Convert plain text to HTML format."""
        # Create a basic HTML structure
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Converted Email</title>
        </head>
        <body>
            <pre>{text}</pre>
        </body>
        </html>
        """
        return html 