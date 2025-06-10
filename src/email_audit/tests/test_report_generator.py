import unittest
from datetime import datetime # Though not directly used in test data, good for context
from src.email_audit.reporter.report_generator import ReportGenerator
from src.email_audit.reporter.csv_headers import AUDIT_CSV_HEADERS

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        """Setup method to instantiate ReportGenerator and AUDIT_CSV_HEADERS indices."""
        self.report_generator = ReportGenerator()
        # Create a local mapping for header indices for convenience in tests
        self.header_indices = {header: i for i, header in enumerate(AUDIT_CSV_HEADERS)}

    def test_generate_csv_report_structure_and_basic_mapping(self):
        sample_email_name = "TestEmail_TXN123_report.eml"
        sample_timestamp = "2024-07-31T15:45:30.123Z" # ISO 8601 format
        sample_audit_results = {
            "score": 85.5,
            "recommendations": "Consider improving clarity.",
            "apptivo_case_number": "ACN789",
            "context": "Test context",
            "participants": "test@example.com",
            "tone": "Neutral",
            "security": "OK",
            "effectiveness": "Good",
            "some_other_metric": "Value for other metric",
            "another_finding": "Details of another finding",
            "Agent name": "Test Agent" # Example of a directly mapped field
        }

        csv_output = self.report_generator.generate_csv_report(
            sample_email_name, sample_audit_results, sample_timestamp
        )

        self.assertIsInstance(csv_output, list)
        self.assertEqual(len(csv_output), 2, "CSV output should have 1 header row and 1 data row")

        # Test header row
        self.assertEqual(csv_output[0], AUDIT_CSV_HEADERS)

        # Test data row
        data_row = csv_output[1]
        self.assertEqual(len(data_row), len(AUDIT_CSV_HEADERS))

        # Specific field assertions using self.header_indices
        self.assertEqual(data_row[self.header_indices["Date of Audit"]], "31-Jul-24")
        self.assertEqual(data_row[self.header_indices["Transaction ID"]], "TXN123")
        self.assertEqual(data_row[self.header_indices["Apptivo"]], "ACN789")
        self.assertEqual(data_row[self.header_indices["Quality Score"]], "86%") # 85.5 rounds to "86%"
        self.assertEqual(data_row[self.header_indices["FATAL Transaction"]], "NO")
        self.assertEqual(data_row[self.header_indices["FEEDBACK"]], "Consider improving clarity.")
        self.assertEqual(data_row[self.header_indices["Max Score"]], 86.4) # As per implementation

        # Test a field that might be generically mapped from audit_results
        self.assertEqual(data_row[self.header_indices["Agent name"]], "Test Agent")

        # Test a field not in audit_results and not specially handled (should be empty)
        # Example: 'LOB' (assuming no specific logic for it and not in sample_audit_results)
        if "LOB" not in sample_audit_results: # Ensure it's truly not in sample data for this check
             self.assertEqual(data_row[self.header_indices["LOB"]], "")


    def test_generate_csv_report_fatal_error_and_missing_data(self):
        sample_email_name = "FatalEmail_report.eml" # No txn ID in name
        sample_timestamp = "2024-08-01T10:00:00Z"
        sample_audit_results = {
            "score": 40.2, # Low score for fatal error
            "recommendations": ["Rec1", "Rec2"], # Test list recommendations
            # 'apptivo_case_number' is missing
            "transaction_id": "TRN_FALLBACK" # Test fallback for txn_id
        }

        csv_output = self.report_generator.generate_csv_report(
            sample_email_name, sample_audit_results, sample_timestamp
        )

        self.assertEqual(len(csv_output), 2) # Header + 1 data row
        data_row = csv_output[1]

        self.assertEqual(data_row[self.header_indices["Date of Audit"]], "01-Aug-24")
        self.assertEqual(data_row[self.header_indices["FATAL Transaction"]], "Fatal Error")
        self.assertEqual(data_row[self.header_indices["Quality Score"]], "40%") # 40.2 becomes "40%"
        self.assertEqual(data_row[self.header_indices["Apptivo"]], "") # Missing, should be empty
        self.assertEqual(data_row[self.header_indices["FEEDBACK"]], "Rec1, Rec2") # List joined

        # Transaction ID: not in email name pattern, but present in audit_results
        self.assertEqual(data_row[self.header_indices["Transaction ID"]], "TRN_FALLBACK")

        # Test a field that is neither in audit_results nor specially handled
        self.assertEqual(data_row[self.header_indices["Team Leader"]], "")

if __name__ == '__main__':
    unittest.main()
