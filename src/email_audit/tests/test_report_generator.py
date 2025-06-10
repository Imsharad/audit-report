import unittest
from src.email_audit.reporter.report_generator import ReportGenerator
from src.email_audit.reporter.csv_headers import AUDIT_CSV_HEADERS
# datetime import might be useful if we were constructing datetime objects for inputs,
# but current ReportGenerator takes ISO strings.

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        """Setup method to instantiate ReportGenerator and common test data."""
        self.report_generator = ReportGenerator()
        self.header_indices = {header: i for i, header in enumerate(AUDIT_CSV_HEADERS)}
        self.case_number = "CASE_TEST_123"
        self.sample_timestamp = "2024-07-31T15:45:30.123Z"

        # Sample detailed_results for general use, can be overridden in specific tests
        self.base_detailed_results = [
            {
                'id': 'transaction_type_identification', 'title': 'Transaction Type (New/Existing)',
                'analysis': 'NEW', 'score': 1.0, 'max_score': 0, # max_score 0 if only for analysis
                'category': 'Metadata', 'is_critical': False, 'improvements': None
            },
            {
                'id': 'agent_name_extraction', 'title': 'Agent Name Extraction',
                'analysis': 'Agent Smith', 'score': 1.0, 'max_score': 0,
                'category': 'Metadata', 'is_critical': False, 'improvements': None
            },
            {
                'id': 'apptivo_case_communication',
                'title': 'Captured correct Apptivo case number & used Apptivo for communication',
                'analysis': 'Case ABC12345', 'score': 0.9, 'max_score': 3.00, # Has score and analysis
                'category': 'Client Policy and Service', 'is_critical': False, 'improvements': "Ensure full ID is captured."
            },
            {
                'id': 'logical_itinerary', 'title': 'Logical Itinerary (Time window, Routing, Connections)',
                'analysis': 'Itinerary seems fine.', 'score': 0.9, 'max_score': 3.00,
                'category': 'PNR Fields', 'is_critical': False, 'improvements': None
            },
            {
                'id': 'applied_commission', 'title': 'Applied Commission as applicable (Retained / Parted)',
                'analysis': 'Commission applied.', 'score': 1.0, 'max_score': 2.00,
                'category': 'Accounting', 'is_critical': True, 'improvements': None # Critical but passed
            },
            {
                'id': 'cross_upsell_opportunity', 'title': 'Utilized cross sell & up sell opportunity (Hotel, Car, Insurance)',
                'analysis': 'Offered hotel.', 'score': 0.5, 'max_score': 2.00, # For "Oppurtunities"
                'category': 'Client Policy and Service', 'is_critical': False, 'improvements': "Explore car options."
            },
            {
                'id': 'overall_email_communication', 'title': 'Overall communication in the email',
                'analysis': 'Good clarity.', 'score': 0.8, 'max_score': 5.00,
                'category': 'Communication', 'is_critical': False, 'improvements': "Slightly more concise."
            }
        ]

        self.base_conversation_history = [
            {'timestamp': '2024-03-15T10:00:00Z', 'transaction_id': 'TID_CONVO123', 'sender': 'client@example.com', 'body_preview': 'Hello...'},
            {'timestamp': '2024-03-15T10:05:00Z', 'transaction_id': 'TID_CONVO123', 'sender': 'agent@example.com', 'body_preview': 'Hi there...'}
        ]

    def test_generate_csv_report_structure_and_basic_mapping(self):
        sample_email_name = "TestEmail_NWCHVC_report.eml" # NWCHVC is a potential fallback ID

        sample_audit_results = {
            "conversation_history": self.base_conversation_history,
            "detailed_results": self.base_detailed_results,
            # Top-level 'transaction_id' for fallback test if convo history one isn't picked (it should be)
            "transaction_id": "TOP_LEVEL_TXN_IGNORE"
        }

        csv_output = self.report_generator.generate_csv_report(
            self.case_number, sample_email_name, sample_audit_results, self.sample_timestamp
        )

        self.assertIsInstance(csv_output, list)
        self.assertEqual(len(csv_output), 2, "CSV output should have 1 header row and 1 data row")
        self.assertEqual(csv_output[0], AUDIT_CSV_HEADERS, "CSV headers do not match expected")

        data_row = csv_output[1]
        self.assertEqual(len(data_row), len(AUDIT_CSV_HEADERS), "Data row length mismatch")

        # Assertions for specific fields
        self.assertEqual(data_row[self.header_indices["Date of Audit"]], "31-Jul-24")
        self.assertEqual(data_row[self.header_indices["Audit ID"]], self.case_number)

        # Fields from conversation_history
        self.assertEqual(data_row[self.header_indices["Transaction Date"]], "15-Mar-24")
        self.assertEqual(data_row[self.header_indices["Transaction ID"]], "TID_CONVO123")

        # Fields from detailed_steps_map (analysis part)
        self.assertEqual(data_row[self.header_indices["Transaction Type"]], "NEW")
        self.assertEqual(data_row[self.header_indices["Agent name"]], "Agent Smith")
        self.assertEqual(data_row[self.header_indices["Apptivo"]], "Case ABC12345") # From apptivo_case_communication analysis

        # Assert fixed empty fields
        self.assertEqual(data_row[self.header_indices["Team Leader"]], "")
        self.assertEqual(data_row[self.header_indices["LOB"]], "")
        self.assertEqual(data_row[self.header_indices["Observer's Name"]], "")

        # Assert individual scores from TITLE_TO_ID_MAPPING
        for header_title, step_id in self.report_generator.TITLE_TO_ID_MAPPING.items():
            mock_step = next((s for s in self.base_detailed_results if s['id'] == step_id), None)
            expected_score = 0.0
            if mock_step and 'score' in mock_step and 'max_score' in mock_step:
                expected_score = round(float(mock_step['score']) * float(mock_step['max_score']), 2)

            # CSV output for scores should be float (or string convertible to float)
            actual_val_str = data_row[self.header_indices[header_title]]
            self.assertEqual(float(actual_val_str), expected_score, f"Score mismatch for header: {header_title}")

        # Assert "Oppurtunities" specifically (it uses 'cross_upsell_opportunity' step)
        opp_step = next(s for s in self.base_detailed_results if s['id'] == 'cross_upsell_opportunity')
        expected_opp_score = round(float(opp_step['score']) * float(opp_step['max_score']), 2)
        self.assertEqual(float(data_row[self.header_indices["Oppurtunities"]]), expected_opp_score)

        # Assert Max Score (Overall)
        expected_max_score = sum(float(s.get('max_score', 0)) for s in self.base_detailed_results)
        self.assertEqual(float(data_row[self.header_indices["Max Score"]]), round(expected_max_score, 2))

        # Assert Quality Score
        total_achieved = sum(float(s.get('score', 0)) * float(s.get('max_score', 0)) for s in self.base_detailed_results)
        expected_quality_perc = (total_achieved / expected_max_score * 100) if expected_max_score > 0 else 0
        self.assertEqual(data_row[self.header_indices["Quality Score"]], f"{round(expected_quality_perc)}%")

        # Assert FATAL Transaction
        self.assertEqual(data_row[self.header_indices["FATAL Transaction"]], "NO") # No critical steps failed

        # Assert Score without Fatal
        self.assertEqual(data_row[self.header_indices["Score without Fatal"]], f"{round(expected_quality_perc)}%")

        # Assert Category Scores
        for cat_header, cat_json_name in self.report_generator.CATEGORY_MAPPING.items():
            expected_cat_sum = sum(
                float(s.get('score', 0)) * float(s.get('max_score', 0))
                for s in self.base_detailed_results if s.get('category') == cat_json_name
            )
            actual_cat_val_str = data_row[self.header_indices[cat_header]]
            self.assertEqual(float(actual_cat_val_str), round(expected_cat_sum, 2), f"Category score mismatch for {cat_header}")

        # Assert FEEDBACK
        expected_feedback_items = [
            "Captured correct Apptivo case number & used Apptivo for communication: Ensure full ID is captured.",
            "Utilized cross sell & up sell opportunity (Hotel, Car, Insurance): Explore car options.",
            "Overall communication in the email: Slightly more concise."
        ]
        self.assertEqual(data_row[self.header_indices["FEEDBACK"]], "; ".join(expected_feedback_items))


    def test_generate_csv_report_fatal_error_and_missing_data(self):
        sample_email_name = "UrgentHelp.eml" # No obvious ID in name
        fatal_timestamp = "2024-08-01T10:00:00Z"

        # Modify detailed_results for fatal error and missing pieces
        custom_detailed_results = [
            { # Critical step failed
                'id': 'applied_commission', 'title': 'Applied Commission as applicable (Retained / Parted)',
                'analysis': 'Commission missed.', 'score': 0.0, 'max_score': 5.00, # Score 0
                'category': 'Accounting', 'is_critical': True, 'improvements': "CRITICAL: Commission was not applied."
            },
            { # To test missing analysis for agent name
                'id': 'agent_name_extraction', 'title': 'Agent Name Extraction',
                'analysis': '', 'score': 1.0, 'max_score': 0, # Empty analysis
                'category': 'Metadata', 'is_critical': False, 'improvements': None
            },
            { # Step from TITLE_TO_ID_MAPPING but will be missing from detailed_results for this test
              # 'logical_itinerary' will be omitted from this list
            },
            { # Normal step for calculation base
                'id': 'pnr_documentation_p5h_reference', 'title': 'PNR Documentation in P5H & Reference (Receive Field)',
                'analysis': 'Documented.', 'score': 1.0, 'max_score': 3.00,
                'category': 'PNR Fields', 'is_critical': False, 'improvements': None
            }
        ]

        sample_audit_results = {
            "conversation_history": [], # Empty, to test fallback for Transaction ID and Date
            "detailed_results": custom_detailed_results,
            "transaction_id": "FALLBACK_TXN_ID" # Test fallback for Transaction ID from top level
        }

        csv_output = self.report_generator.generate_csv_report(
            self.case_number, sample_email_name, sample_audit_results, fatal_timestamp
        )
        data_row = csv_output[1]

        # Assertions for fatal error and missing data scenarios
        self.assertEqual(data_row[self.header_indices["Date of Audit"]], "01-Aug-24")
        self.assertEqual(data_row[self.header_indices["Transaction Date"]], "", "Transaction Date should be empty")

        # Transaction ID: Fallback to audit_results.transaction_id, then regex (none here), then empty
        self.assertEqual(data_row[self.header_indices["Transaction ID"]], "FALLBACK_TXN_ID")

        self.assertEqual(data_row[self.header_indices["Agent name"]], "", "Agent name should be empty")

        # Assert score for a step in TITLE_TO_ID_MAPPING but missing from detailed_results
        self.assertEqual(float(data_row[self.header_indices["Logical Itinerary (Time window, Routing, Connections)"]]), 0.0)

        # FATAL Transaction
        self.assertEqual(data_row[self.header_indices["FATAL Transaction"]], "Fatal Error")

        # Score without Fatal
        self.assertEqual(data_row[self.header_indices["Score without Fatal"]], "NA")

        # Quality Score
        expected_max_score_fatal = sum(float(s.get('max_score', 0)) for s in custom_detailed_results if 'id' in s) # only count valid steps
        total_achieved_fatal = sum(float(s.get('score', 0)) * float(s.get('max_score', 0)) for s in custom_detailed_results if 'id' in s)
        expected_quality_perc_fatal = (total_achieved_fatal / expected_max_score_fatal * 100) if expected_max_score_fatal > 0 else 0
        self.assertEqual(data_row[self.header_indices["Quality Score"]], f"{round(expected_quality_perc_fatal)}%")

        # FEEDBACK for fatal case
        self.assertEqual(data_row[self.header_indices["FEEDBACK"]], "Applied Commission as applicable (Retained / Parted): CRITICAL: Commission was not applied.")

if __name__ == '__main__':
    unittest.main()
