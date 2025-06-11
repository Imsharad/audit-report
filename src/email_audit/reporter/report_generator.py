from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from .csv_headers import AUDIT_CSV_HEADERS

class ReportGenerator:
    def __init__(self, audit_steps: List[Dict[str, Any]]):
        self.TITLE_TO_ID_MAPPING = self._create_title_to_id_mapping(audit_steps)
        self.CATEGORY_MAPPING = {
            "PNR Fields": "PNR Fields",
            "Client Policy and Service": "Client Policy and Service",
            "Accounting": "Accounting",
            "Communication": "Communication"
        }

    def _create_title_to_id_mapping(self, audit_steps: List[Dict[str, Any]]) -> Dict[str, str]:
        """Dynamically creates the title-to-ID mapping from audit steps."""
        mapping = {}
        for step in audit_steps:
            mapping[step['title']] = step['id']
        # Also add the special "Oppurtunities" mapping if the corresponding step exists
        for step in audit_steps:
            if step['id'] == "cross_upsell_opportunity":
                mapping["Oppurtunities"] = "cross_upsell_opportunity"
                break
        return mapping

    def generate_csv_report(
        self, 
        case_number: str, 
        email_name: str, 
        audit_results: Dict[str, Any], 
        timestamp: str,
        attachment_manifest: Optional[Any] = None,
        attachment_analysis: Optional[Any] = None
    ) -> List[List[Any]]:
        rows: List[List[Any]] = [AUDIT_CSV_HEADERS]
        data_row: List[Any] = [""] * len(AUDIT_CSV_HEADERS)

        header_to_index = {header: i for i, header in enumerate(AUDIT_CSV_HEADERS)}
        detailed_steps_list = audit_results.get('detailed_results', [])
        detailed_steps_map = {step['id']: step for step in detailed_steps_list if 'id' in step}

        # Date of Audit
        try:
            dt_object = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            data_row[header_to_index["Date of Audit"]] = dt_object.strftime('%d-%b-%y')
        except (ValueError, KeyError):
            data_row[header_to_index["Date of Audit"]] = ""

        # Transaction Date
        try:
            conv_history = audit_results.get('conversation_history', [{}])
            if conv_history and isinstance(conv_history, list) and conv_history[0].get('timestamp'):
                trans_dt_object = datetime.fromisoformat(conv_history[0]['timestamp'].replace("Z", "+00:00"))
                data_row[header_to_index["Transaction Date"]] = trans_dt_object.strftime('%d-%b-%y')
            else:
                data_row[header_to_index["Transaction Date"]] = ""
        except (ValueError, KeyError, IndexError):
            data_row[header_to_index["Transaction Date"]] = ""

        # Transaction ID
        transaction_id = ""
        conv_history = audit_results.get('conversation_history', [{}])
        if conv_history and isinstance(conv_history, list) and conv_history[0].get('transaction_id'):
            transaction_id = conv_history[0]['transaction_id']

        if not transaction_id:
            match = re.search(r"([A-Z0-9]{6,})", email_name) # More generic ID pattern
            if match:
                transaction_id = match.group(1)

        if not transaction_id:
            transaction_id = audit_results.get('transaction_id', "")
        data_row[header_to_index["Transaction ID"]] = transaction_id

        # Transaction Type
        data_row[header_to_index["Transaction Type"]] = detailed_steps_map.get('transaction_type_identification', {}).get('analysis', '')

        # Agent name
        data_row[header_to_index["Agent name"]] = detailed_steps_map.get('agent_name_extraction', {}).get('analysis', '')

        # Team Leader, LOB, Observer's Name - set to ""
        data_row[header_to_index["Team Leader"]] = ""
        data_row[header_to_index["LOB"]] = ""
        data_row[header_to_index["Observer's Name"]] = ""

        # Audit ID
        data_row[header_to_index["Audit ID"]] = case_number

        # Apptivo (Case Number string)
        data_row[header_to_index["Apptivo"]] = detailed_steps_map.get('apptivo_case_communication', {}).get('analysis', '')

        # Direct Score Mapping for audit step related columns
        for header, step_id in self.TITLE_TO_ID_MAPPING.items():
            if header in header_to_index: # Ensure header is in current AUDIT_CSV_HEADERS
                step_result = detailed_steps_map.get(step_id)
                if step_result and isinstance(step_result, dict) and 'score' in step_result and 'max_score' in step_result:
                    try:
                        achieved_score = float(step_result['score']) * float(step_result['max_score'])
                        data_row[header_to_index[header]] = round(achieved_score, 2)
                    except (ValueError, TypeError):
                         data_row[header_to_index[header]] = 0.0
                else:
                    data_row[header_to_index[header]] = 0.0

        # Calculated Scores
        # Max Score (Overall Max Score)
        overall_max_score = sum(float(step.get('max_score', 0)) for step in detailed_steps_list if isinstance(step, dict))
        data_row[header_to_index["Max Score"]] = round(overall_max_score, 2)

        # Quality Score (Overall Achieved Score as Percentage String)
        total_achieved_score = sum(float(step.get('score', 0)) * float(step.get('max_score', 0)) for step in detailed_steps_list if isinstance(step, dict))
        percentage = (total_achieved_score / overall_max_score * 100) if overall_max_score > 0 else 0
        data_row[header_to_index["Quality Score"]] = f"{round(percentage)}%"

        # FATAL Transaction
        is_fatal_flag = any(
            isinstance(step, dict) and step.get('is_critical', False) and float(step.get('score', 1.0)) < 0.7
            for step in detailed_steps_list
        )
        data_row[header_to_index["FATAL Transaction"]] = "Fatal Error" if is_fatal_flag else "NO"

        # Score without Fatal
        fatal_flag_for_score = data_row[header_to_index["FATAL Transaction"]] == "Fatal Error"
        data_row[header_to_index["Score without Fatal"]] = "NA" if fatal_flag_for_score else data_row[header_to_index["Quality Score"]]

        # Category Scores
        for csv_col_header, category_name_in_json in self.CATEGORY_MAPPING.items():
            if csv_col_header in header_to_index:
                category_total = sum(
                    float(res.get('score', 0)) * float(res.get('max_score', 0))
                    for res in detailed_steps_list
                    if isinstance(res, dict) and res.get('category') == category_name_in_json
                )
                data_row[header_to_index[csv_col_header]] = round(category_total, 2)

        # FEEDBACK - Enhanced with attachment findings
        feedback_items = []
        for step in detailed_steps_list:
            if isinstance(step, dict) and float(step.get('score', 1.0)) < 0.7:
                improvement_text = step.get('improvements') or step.get('analysis')
                if improvement_text: # Only add if there's something to say
                    # Sanitize the text to prevent breaking CSV format
                    sanitized_text = improvement_text.replace('"', "'").replace('\n', ' ').replace('\r', '')
                    feedback_items.append(f"{step.get('title', 'Unknown Step')}: {sanitized_text}")
        
        # Add attachment security concerns to feedback
        if attachment_analysis:
            for analysis in attachment_analysis.analyses:
                if analysis.security_concerns:
                    concerns_text = "; ".join(analysis.security_concerns)
                    sanitized_concerns = concerns_text.replace('"', "'").replace('\n', ' ').replace('\r', '')
                    feedback_items.append(f"Attachment {analysis.filename}: {sanitized_concerns}")
        
        data_row[header_to_index["FEEDBACK"]] = "; ".join(feedback_items) if feedback_items else ""

        # Ensure all data row elements are appropriately typed (string or number)
        # For now, numerical scores are floats. If specific string formatting (e.g. "5.40") is needed:
        # for i, val in enumerate(data_row):
        #     if isinstance(val, float):
        #         data_row[i] = f"{val:.2f}"

        rows.append(data_row)
        return rows

    def generate_report(
        self,
        email_name: str,
        audit_results: Dict[str, Any],
        timestamp: str,
        attachment_manifest: Optional[Any] = None,
        attachment_analysis: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate an audit report for an email with attachment data.
        """
        # Get score from browser audit
        score = audit_results.get("score", audit_results.get("overall_score", 0))
        
        # Generate base report
        report = {
            "email_name": email_name,
            "timestamp": timestamp,
            "overall_score": round(float(score), 2) if isinstance(score, (int, float, str)) and str(score).replace('.', '', 1).isdigit() else 0.0,
            "audit_results": audit_results,
            "summary": self._generate_summary(audit_results)
        }
        
        # Add attachment data if available
        if attachment_manifest:
            report["attachment_manifest"] = {
                "total_parts": attachment_manifest.total_parts,
                "attachment_count": attachment_manifest.attachment_count,
                "approved_attachments": len(attachment_manifest.approved_attachments),
                "flagged_attachments": len(attachment_manifest.flagged_attachments),
                "processing_summary": attachment_manifest.processing_summary
            }
        
        if attachment_analysis:
            report["attachment_analysis"] = {
                "total_attachments": attachment_analysis.total_attachments,
                "processed_attachments": attachment_analysis.processed_attachments,
                "skipped_attachments": attachment_analysis.skipped_attachments,
                "total_processing_time_ms": attachment_analysis.total_processing_time_ms,
                "summary": attachment_analysis.summary,
                "security_concerns_found": sum(len(a.security_concerns) for a in attachment_analysis.analyses),
                "high_confidence_findings": len([a for a in attachment_analysis.analyses if a.confidence_score > 0.7])
            }
        
        return report
    
    def _generate_summary(self, audit_results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the audit results."""
        summary_parts = []
        
        # Add audit summary (This might need updates based on new audit_results structure)
        summary_parts.append("Email Analysis Summary:")
        # Example: Use a high-level summary if available, or fallback
        summary_parts.append(f"1. Context: {audit_results.get('context', 'N/A')}")
        summary_parts.append(f"2. Participants: {audit_results.get('participants', 'N/A')}")
        summary_parts.append(f"3. Tone: {audit_results.get('tone', 'N/A')}")
        summary_parts.append(f"4. Security: {audit_results.get('security', 'N/A')}") # Assuming these keys might still exist
        summary_parts.append(f"5. Communication: {audit_results.get('effectiveness', 'N/A')}")

        # Recommendations from detailed_results for CSV could be used here too
        feedback_items_summary = []
        detailed_steps_list_summary = audit_results.get('detailed_results', [])
        for step in detailed_steps_list_summary:
            if isinstance(step, dict) and float(step.get('score', 1.0)) < 0.7:
                improvement_text = step.get('improvements') or step.get('analysis')
                if improvement_text:
                     feedback_items_summary.append(f"{step.get('title', 'Issue')}: {improvement_text}")

        if feedback_items_summary:
            summary_parts.append("6. Key Recommendations:\n  - " + "\n  - ".join(feedback_items_summary))
        elif audit_results.get('recommendations'): # Fallback to old recommendations
             summary_parts.append(f"6. Recommendations: {audit_results['recommendations']}")
        else:
            summary_parts.append("6. Recommendations: No specific recommendations.")

        return "\n".join(summary_parts)

# Note: The original generate_report and _generate_summary methods might need
# further adjustments if the structure of 'audit_results' (especially 'score'
# and 'recommendations' at the top level) has significantly changed due to the
# focus on 'detailed_results' for the CSV. The changes made above are minimal
# attempts to make them robust to missing keys or use new feedback style.
# The subtask primarily focuses on generate_csv_report.