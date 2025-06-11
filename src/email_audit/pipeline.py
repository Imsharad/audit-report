from pathlib import Path
from loguru import logger
from typing import List, Dict, Any
import json
import csv # Added import
from datetime import datetime
import os
import asyncio
from dotenv import load_dotenv

from .parser.eml_parser import EMLParser
from .parser.models import AttachmentManifest
from .auditor.email_auditor import EmailAuditor
from .reporter.report_generator import ReportGenerator
from .utils.state_manager import StateManager
from .services.attachment_processor import AttachmentProcessor
from .llm.llm_factory import LLMFactory

# Load environment variables
load_dotenv()

class EmailAuditPipeline:
    def __init__(self, input_dir: str = None, html_dir: str = None, reports_dir: str = None):
        # Use environment variables with fallback to default values
        self.input_dir = Path(input_dir or os.getenv('INPUT_DIR', 'eml-input'))
        self.html_dir = Path(html_dir or os.getenv('HTML_DIR', 'eml-html'))
        self.reports_dir = Path(reports_dir or os.getenv('REPORTS_DIR', 'reports'))
        
        # Create directories if they don't exist
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = EMLParser()
        self.email_auditor = EmailAuditor()
        self.reporter = ReportGenerator(self.email_auditor.audit_steps)
        self.state_manager = StateManager()
        
        # Initialize LLM for attachment processing
        llm_provider = os.getenv('LLM_PROVIDER', 'anthropic')
        llm_model = os.getenv('LLM_MODEL', 'claude-3-opus-20240229')
        llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.0'))
        self.llm = LLMFactory.create_llm(llm_provider, llm_model, llm_temperature)
        self.attachment_processor = AttachmentProcessor(self.llm)
        
        # Configure logger
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE', 'pipeline.log')
        logger.add(log_file, rotation="1 day", level=log_level)
    
    async def process_eml_file(self, eml_path: Path) -> Dict[str, Any]:
        """Process a single EML file through the pipeline with full MIME awareness."""
        try:
            # Check if file has already been processed
            if self.state_manager.is_processed(eml_path):
                case_number = self.state_manager.get_case_number(eml_path)
                logger.info(f"File {eml_path.name} already processed in case {case_number}")
                return {
                    "status": "skipped",
                    "eml_file": eml_path.name,
                    "case_number": case_number,
                    "reason": "Already processed"
                }
            
            # Create new case folder
            case_number = self.state_manager.create_case_folder(eml_path)
            case_dir = Path("processed_cases") / case_number
            logger.info(f"Created new case folder: {case_number}")
            
            # Step 1: Parse EML with full MIME awareness
            html_content, attachment_manifest = self.parser.parse_with_attachments(
                eml_path, case_dir
            )
            
            # Save HTML content
            html_path = self.html_dir / f"{eml_path.stem}.html"
            html_path.write_text(html_content)
            
            # Step 2: Process attachments with LLM analysis
            attachment_analysis = None
            if attachment_manifest.attachment_count > 0:
                logger.info(f"Processing {attachment_manifest.attachment_count} attachments")
                attachment_analysis = await self.attachment_processor.process_attachments(
                    attachment_manifest, case_dir
                )
            
            # Step 3: Browser-based audit (existing)
            audit_results = await self.email_auditor.audit_email(html_path)
            
            # Step 4: Generate comprehensive report
            report = self.reporter.generate_report(
                eml_path.name,
                audit_results,
                datetime.now().isoformat(),
                attachment_manifest=attachment_manifest,
                attachment_analysis=attachment_analysis
            )
            
            # Save report
            report_path = self.reports_dir / f"{eml_path.stem}_report.json"
            report_path.write_text(json.dumps(report, indent=2, default=str))

            # Generate and save CSV report
            csv_data = self.reporter.generate_csv_report(
                case_number, 
                eml_path.name, 
                audit_results, 
                report["timestamp"],
                attachment_manifest=attachment_manifest,
                attachment_analysis=attachment_analysis
            )
            csv_report_path = self.reports_dir / f"{eml_path.stem}_report.csv"
            with open(csv_report_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerows(csv_data)
            
            # Save attachment manifest
            manifest_path = case_dir / "attachment_manifest.json"
            manifest_path.write_text(attachment_manifest.model_dump_json(indent=2))
            
            # Save attachment analysis if available
            if attachment_analysis:
                analysis_path = case_dir / "attachment_analysis.json"
                analysis_path.write_text(attachment_analysis.model_dump_json(indent=2))
            
            # Move files to case folder
            new_paths = self.state_manager.move_to_case_folder(
                case_number,
                eml_path,
                html_path,
                report_path,
                csv_report_path
            )
            
            # Clean up temporary files
            html_path.unlink()
            
            # Prepare paths for return
            returned_paths = {
                "eml": str(new_paths["eml"]),
                "html": str(new_paths["html"]),
                "json_report": str(new_paths["report"]),
                "attachment_manifest": str(manifest_path),
            }
            if "csv_report" in new_paths and new_paths["csv_report"]:
                returned_paths["csv_report"] = str(new_paths["csv_report"])
            if attachment_analysis:
                returned_paths["attachment_analysis"] = str(analysis_path)

            # Log processing summary
            logger.info(f"Processed {eml_path.name}: {attachment_manifest.total_parts} MIME parts, "
                       f"{attachment_manifest.attachment_count} attachments")
            
            if attachment_analysis:
                logger.info(f"Attachment analysis: {attachment_analysis.processed_attachments} processed, "
                           f"{attachment_analysis.summary.get('total_security_concerns', 0)} security concerns")

            return {
                "status": "success",
                "eml_file": eml_path.name,
                "case_number": case_number,
                "paths": returned_paths,
                "processing_summary": {
                    "mime_parts": attachment_manifest.total_parts,
                    "attachments": attachment_manifest.attachment_count,
                    "approved_attachments": len(attachment_manifest.approved_attachments),
                    "flagged_attachments": len(attachment_manifest.flagged_attachments),
                    "attachment_processing_time_ms": attachment_analysis.total_processing_time_ms if attachment_analysis else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {eml_path.name}: {str(e)}")
            return {
                "status": "error",
                "eml_file": eml_path.name,
                "error": str(e)
            }
    
    async def run(self) -> List[Dict[str, Any]]:
        """Run the pipeline on all EML files in the input directory."""
        results = []
        
        for eml_path in self.input_dir.glob("*.eml"):
            logger.info(f"Processing {eml_path.name}")
            result = await self.process_eml_file(eml_path)
            results.append(result)
            
        return results

async def main():
    pipeline = EmailAuditPipeline()
    results = await pipeline.run()
    
    # Print summary
    processed = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"Pipeline completed:")
    logger.info(f"- Processed: {processed} files")
    logger.info(f"- Skipped: {skipped} files")
    logger.info(f"- Failed: {failed} files")
    
    # Print processing statistics
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        total_attachments = sum(r.get("processing_summary", {}).get("attachments", 0) for r in successful_results)
        total_approved = sum(r.get("processing_summary", {}).get("approved_attachments", 0) for r in successful_results)
        total_flagged = sum(r.get("processing_summary", {}).get("flagged_attachments", 0) for r in successful_results)
        
        logger.info(f"Attachment processing:")
        logger.info(f"- Total attachments: {total_attachments}")
        logger.info(f"- Approved for analysis: {total_approved}")
        logger.info(f"- Flagged: {total_flagged}")
    
    # Print case information
    cases = pipeline.state_manager.list_cases()
    logger.info(f"Total cases: {len(cases)}")

if __name__ == "__main__":
    asyncio.run(main()) 