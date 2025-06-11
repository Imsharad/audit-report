
### **Task 1: [EPIC] Instrument the Pipeline for Full MIME Awareness**

**User Story:** As an Auditor, I need the system to parse any `.eml` file and identify all its constituent parts (HTML, text, images, PDFs, etc.), creating a complete inventory of the email's contents so we can process attachments in subsequent work.

**Acceptance Criteria:**
*   Refactor `src/email_audit/parser/eml_parser.py`. The primary method must now walk the entire MIME tree and output a structured list of all detected parts, not just the first `text/html` part.
*   Introduce Pydantic models for `Attachment` and `AttachmentManifest` to standardize the representation of non-body content.
*   The pipeline (`pipeline.py`) must be updated to call the new parser and save an `attachments_manifest.json` in the case folder for every processed email.
*   Implement a basic classification and allow-list mechanism. The manifest should flag attachments that are oversized (>10MB), password-protected (based on file name heuristics for now), or have a MIME type not on the approved list (`.png`, `.jpg`, `.pdf`, etc.).
*   Update logging to report the count and types of all attachments found in each email.

**Outcome:** The system is no longer blind. It can see and catalog every attachment, setting the stage for intelligent processing.

---

### **Task 2: [EPIC] Implement Text-Based Content Extraction & Integrate into Audit Chain**

**User Story:** As a Judge, I need the system to automatically extract text from PDF and image attachments and include this text in the audit context, enabling both the primary Auditor and myself to make more accurate, evidence-based assessments.

**Acceptance Criteria:**
*   Create a new, self-contained `AttachmentProcessor` in `src/email_audit/processor/`.
*   Integrate `PyMuPDF` for native text extraction from PDFs and `pytesseract` for baseline OCR on images.
*   Modify `EmailAuditor.audit_email` to accept a new "Unified Context" object that includes the email body *and* the extracted text from all attachments.
*   Update the prompts for both the `reasoning_llm` and the `judge_llm` to explicitly reference and use the provided attachment context.
*   Write a key integration test: an email where the body is vague but an attachment contains a critical detail. The test must prove that the initial auditor might fail but the Judge, using the attachment context, corrects the final score.

**Outcome:** The "brain" of the system can now read. Audit accuracy increases immediately as it's no longer limited to the email body.

---

### **Task 3: [EPIC] Integrate Multimodal Vision for Advanced Content Analysis**

**User Story:** As an Auditor, when standard OCR fails on a low-quality image or a scanned PDF, I need the system to fall back to a powerful Vision LLM to "read" the attachment, ensuring we don't miss critical context from challenging documents.

**Acceptance Criteria:**
*   Update the `BaseLLM` interface and relevant implementations (`AnthropicLLM`, `OpenAILLM`) to support multimodal inputs (i.e., sending image data alongside text prompts).
*   Enhance the `AttachmentProcessor`: If `pytesseract` returns low-confidence or empty text, the processor should, if a feature flag is enabled, call the designated vision-capable LLM with the image bytes.
*   Implement cost and performance logging for all vision LLM calls to monitor usage.
*   The extracted data from the vision model is fed into the same "Unified Context" object, making this enhancement transparent to the downstream auditor.
*   An end-to-end test with a scanned (image-only) PDF must pass, demonstrating successful extraction and auditing where the text-only approach would have failed.

**Outcome:** The system can now handle the most common and frustrating edge cases (scans, complex layouts), dramatically reducing the need for human intervention.

---

### **Task 4: [EPIC] Productionize with Secure, Scalable Ingestion Channels**

**User Story:** As a Platform Operator, I need to prepare the attachment-aware pipeline for client use by hardening its security, adding monitoring, and building flexible, secure ingestion channels.

**Acceptance Criteria:**
*   **Secure Ingestion PoC:** Implement a "forward-to-mailbox" ingestion channel. A dedicated script using IMAP can poll a mailbox, download `.eml` files, and feed them into the existing pipeline.
*   **Secure Ingestion (Primary):** Implement a robust ingestion mechanism using the Microsoft Graph API with OAuth 2.0. This should be the recommended method for clients. Your code will request read-only access to a specific mailbox, fetch messages, and process them.
*   **Hardening:** Ensure the pipeline is secure. By default, it must not fetch remote HTTP resources found in email bodies. All file processing (parsing, extraction) should occur in a sandboxed/ephemeral environment.
*   **Monitoring:** Add metrics to track key pipeline health indicators: attachments processed vs. failed, OCR success rate, number of vision model fallbacks, average processing time per email.
*   Draft client-facing documentation explaining the new attachment-processing capabilities and detailing the security-first design of the ingestion methods.

**Outcome:** The pipeline is transformed from a local script into a secure, scalable, and client-ready service.