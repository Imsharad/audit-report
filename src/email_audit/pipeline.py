from pathlib import Path
from loguru import logger
from typing import List, Dict, Any
import json
import csv # Added import
from datetime import datetime
import os
import asyncio
from dotenv import load_dotenv, find_dotenv

from .parser.eml_parser import EMLParser
from .auditor.email_auditor import EmailAuditor
from .reporter.report_generator import ReportGenerator
from .utils.state_manager import StateManager

# Load environment variables
load_dotenv(find_dotenv('.env.local'))

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
        self.reporter = ReportGenerator()
        self.state_manager = StateManager()
        
        # Configure logger
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE', 'pipeline.log')
        logger.add(log_file, rotation="1 day", level=log_level)
    
    async def process_eml_file(self, eml_path: Path) -> Dict[str, Any]:
        """Process a single EML file through the pipeline."""
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
            logger.info(f"Created new case folder: {case_number}")
            
            # Step 1: Convert EML to HTML
            html_content = self.parser.convert_to_html(eml_path)
            html_path = self.html_dir / f"{eml_path.stem}.html"
            html_path.write_text(html_content)
            
            # Step 2: Browser-based audit
            audit_results = await self.email_auditor.audit_email(html_path)
            
            # Step 3: Generate report
            report = self.reporter.generate_report(
                eml_path.name,
                audit_results,
                datetime.now().isoformat()
            )
            
            # Save report
            report_path = self.reports_dir / f"{eml_path.stem}_report.json"
            report_path.write_text(json.dumps(report, indent=2))

            # Generate and save CSV report
            csv_data = self.reporter.generate_csv_report(case_number, eml_path.name, audit_results, report["timestamp"]) # Added case_number
            csv_report_path = self.reports_dir / f"{eml_path.stem}_report.csv"
            with open(csv_report_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            # Move files to case folder
            new_paths = self.state_manager.move_to_case_folder(
                case_number,
                eml_path,
                html_path,
                report_path,
                csv_report_path # Pass CSV report path
            )
            
            # Clean up temporary files
            html_path.unlink()
            report_path.unlink()
            if csv_report_path.exists(): # Ensure it exists before trying to unlink
                csv_report_path.unlink()
            
            # Prepare paths for return, ensuring csv_report is handled correctly
            returned_paths = {
                "eml": str(new_paths["eml"]),
                "html": str(new_paths["html"]),
                "json_report": str(new_paths["report"]) # Renamed for clarity
            }
            if "csv_report" in new_paths and new_paths["csv_report"]:
                returned_paths["csv_report"] = str(new_paths["csv_report"])

            return {
                "status": "success",
                "eml_file": eml_path.name,
                "case_number": case_number,
                "paths": returned_paths
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
    
    # Print case information
    cases = pipeline.state_manager.list_cases()
    logger.info(f"Total cases: {len(cases)}")

if __name__ == "__main__":
    asyncio.run(main()) 