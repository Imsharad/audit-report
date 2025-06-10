from typing import Dict, List, Any
from datetime import datetime
import re
from .csv_headers import AUDIT_CSV_HEADERS

class ReportGenerator:
    def generate_report(
        self,
        email_name: str,
        audit_results: Dict[str, Any],
        timestamp: str
    ) -> Dict[str, Any]:
        """
        Generate an audit report for an email.
        
        Args:
            email_name: Name of the email file
            audit_results: Dictionary containing browser audit results
            timestamp: Timestamp of the audit
            
        Returns:
            Dictionary containing the complete audit report
        """
        # Get score from browser audit
        score = audit_results["score"]
        
        # Generate report
        report = {
            "email_name": email_name,
            "timestamp": timestamp,
            "overall_score": round(score, 2),
            "audit_results": audit_results,
            "summary": self._generate_summary(audit_results)
        }
        
        return report
    
    def _generate_summary(self, audit_results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the audit results."""
        summary_parts = []
        
        # Add audit summary
        summary_parts.append("Email Analysis Summary:")
        summary_parts.append(f"1. Context: {audit_results['context']}")
        summary_parts.append(f"2. Participants: {audit_results['participants']}")
        summary_parts.append(f"3. Tone: {audit_results['tone']}")
        summary_parts.append(f"4. Security: {audit_results['security']}")
        summary_parts.append(f"5. Communication: {audit_results['effectiveness']}")
        summary_parts.append(f"6. Recommendations: {audit_results['recommendations']}")
        
        return "\n".join(summary_parts) 