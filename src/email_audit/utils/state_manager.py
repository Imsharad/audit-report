import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
from loguru import logger

class StateManager:
    def __init__(self, processed_dir: str = "processed_cases"):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.processed_dir / "processing_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load the processing state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {"processed_files": {}}
            self._save_state()
    
    def _save_state(self):
        """Save the current processing state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_processed(self, eml_path: Path) -> bool:
        """Check if a file has already been processed."""
        return str(eml_path.absolute()) in self.state["processed_files"]
    
    def get_case_number(self, eml_path: Path) -> Optional[str]:
        """Get the case number for a processed file."""
        return self.state["processed_files"].get(str(eml_path.absolute()))
    
    def create_case_folder(self, eml_path: Path) -> str:
        """Create a new case folder and return its number."""
        # Generate case number based on timestamp
        case_number = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        case_dir = self.processed_dir / case_number
        
        # Create case directory structure
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "eml").mkdir(exist_ok=True)
        (case_dir / "html").mkdir(exist_ok=True)
        (case_dir / "reports").mkdir(exist_ok=True)
        
        return case_number
    
    def move_to_case_folder(
        self,
        case_number: str,
        eml_path: Path,
        html_path: Path,
        report_path: Path,
        csv_report_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Move processed files to their case folder."""
        case_dir = self.processed_dir / case_number
        
        # Define new paths
        new_paths = {
            "eml": case_dir / "eml" / eml_path.name,
            "html": case_dir / "html" / html_path.name,
            "report": case_dir / "reports" / report_path.name
        }
        
        # Move files
        shutil.copy2(eml_path, new_paths["eml"])
        shutil.copy2(html_path, new_paths["html"])
        shutil.copy2(report_path, new_paths["report"])

        if csv_report_path and csv_report_path.exists():
            new_paths["csv_report"] = case_dir / "reports" / csv_report_path.name
            shutil.copy2(csv_report_path, new_paths["csv_report"])
        
        # Update state
        self.state["processed_files"][str(eml_path.absolute())] = case_number
        self._save_state()
        
        return new_paths
    
    def get_case_info(self, case_number: str) -> Dict[str, Any]:
        """Get information about a specific case."""
        case_dir = self.processed_dir / case_number
        if not case_dir.exists():
            return {}
        
        return {
            "case_number": case_number,
            "created_at": datetime.fromtimestamp(case_dir.stat().st_ctime).isoformat(),
            "files": {
                "eml": list((case_dir / "eml").glob("*.eml")),
                "html": list((case_dir / "html").glob("*.html")),
                "json_reports": list((case_dir / "reports").glob("*.json")),
                "csv_reports": list((case_dir / "reports").glob("*.csv"))
            }
        }
    
    def list_cases(self) -> Dict[str, Dict[str, Any]]:
        """List all processed cases."""
        cases = {}
        for case_dir in self.processed_dir.glob("CASE_*"):
            if case_dir.is_dir():
                cases[case_dir.name] = self.get_case_info(case_dir.name)
        return cases 