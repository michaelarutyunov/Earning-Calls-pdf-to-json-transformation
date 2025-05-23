# Quarterly Earning Calls (QEC) PDF to JSON transformation

A demonstration of a workflow to transform earning call transcripts into a json object.

**Goal**: Create a json object that represents the flow of conversation in the presentation and Q&A sections of a transcript.

**Challenge**: Earning call transcripts do not have stardard formatting making regex based extraction difficult.

**Solution**: Use LLM to extract most relevant infomation about the transcript.

**Guardrails**: Keep API call costs to a minimum, particularly reduce the count of output tokens.

**Key Steps**:
- Normalize the text imported from PDF
- Enrich of the text with formatting tags
- Capture potential attributions using SpaCy to create additional focus for LLM
- API call to LLM to extract call specs and attributions
- Create json object with clean speakers and utterances
- Optional: create clean text file from JSON

**Files and Folders**
config.json - configuration for the main script
pdf_processing.py - main script
create_standard_text.py - optional script
transcript_pdf - source documents
final_json - main script outputs
standardized_text - optional script outputs

**Notes**:
- Code requires anthropic API to be saved in .env file in the same folder

**Next steps**:
- Structuring (e.g. by dialogue where question-answers are grouped by analyst)
- Isolate multiple questions asked within the same analyst's utterance and align with specific parts of the bank response
- Optimal chunking strategies, e.g. for RAG implementation, topic modelling etc.
- Maintaining a consolidated list of real examples of attributions and use that instead of LLM
- More robust LLM output validation
- Testing possibiity to run SLM with structured outputs instead of API call