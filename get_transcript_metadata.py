from datetime import time
import pymupdf
from dotenv import load_dotenv
import anthropic
import os
import re
import argparse
import json
import time

load_dotenv()

def get_folder_paths():
    """Get folder paths from folder_config.json."""
    config_paths = "folder_config.json"
    
    try:
        with open(config_paths, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("folder_paths", {})
    except FileNotFoundError:
        print(f"Configuration file {config_paths} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_paths}.")
        return {}

def get_test_mode():
    """Get folder paths from folder_config.json."""
    config_paths = "folder_config.json"

    try:
        with open(config_paths, 'r', encoding='utf-8') as f:
            config = json.load(f)
            test_mode = config.get("test_mode", {})
            return test_mode.get("enabled", False), test_mode.get("file_name", None)
    except FileNotFoundError:
        print(f"Configuration file {config_paths} not found.")
        return False, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_paths}.")
        return False, None

def extract_text_from_pdf(path):
    """Extract text from PDF file."""
    doc = pymupdf.open(path)
    print(f"pages found in {path}: {doc.page_count}")

    full_text = ""

    def clean_text(text):
        """Clean text by handling encoding issues and removing problematic characters."""
        # Handle encoding issues with round-trip conversion
        try:
            # Convert to bytes and back with explicit error handling
            text_bytes = text.encode('utf-8', errors='ignore')
            text = text_bytes.decode('utf-8', errors='ignore')
        except (UnicodeError, AttributeError):
            pass

        # Dictionary of other common substitutions for financial documents
        replacements = {
            '�': '',
            '\ufffd': '',
            '\u2022': '•',  # bullet point
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2013': '-',  # en-dash
            '\u2014': '--',  # em-dash
            '\u00a0': ' <NBSP> ',  # non-breaking space
            '\f': ' <PAGEBREAK> ',  # form feed / page break
            '\n': ' <LINEBREAK> ',  # line break
            '\t': ' <TAB> ',  # tab
        }

        # Apply all replacements
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # Preserve multiple spaces for regex pattern identification
        text = re.sub(r'[ ]{2,}', ' <MULTISPACE> ', text)

        # Pattern for spaced headers
        pattern = r'(?<!\S)([A-Z](\s+)[A-Z](\s+[A-Z])+)(?!\S)'

        def fix_spaced_header(match):
            # Get the original text with spaces
            spaced_text = match.group(0)

            # First approach: Split by multiple spaces (2 or more)
            if re.search(r'\s{2,}', spaced_text):
                words = re.split(r'\s{2,}', spaced_text)
                condensed_words = [re.sub(r'\s+', '', word) for word in words]
                return ' '.join(condensed_words)
            else:
                # For headers with single spaces between letters but no clear word boundaries
                # Try to identify common words or insert spaces at logical points
                # Example: "Q 2 2 0 2 3 E A R N I N G S" -> "Q2 2023 EARNINGS"
                condensed = re.sub(r'\s+', '', spaced_text)

                # Insert spaces before uppercase letters that follow lowercase or numbers
                # This helps separate words like "EarningsCall" -> "Earnings Call"
                spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', condensed)

                # For dates like "2 0 2 3", keep them together
                # This is a simplified approach and might need adjustment
                spaced = re.sub(r'(\d)\s+(\d)', r'\1\2', spaced)

                return spaced

        text = re.sub(pattern, fix_spaced_header, text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text("text")
        # print(f"page {page_num + 1} of {pdf_path} has {len(page_text)} characters")
        page_text = clean_text(page_text)
        if page_text:
            full_text += page_text + "\n<PAGE_BREAK>\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1} of {path}")

    doc.close()
    
    return full_text


def API_call_anthropic(text):
    # Construct the prompt according to the specified format
    prompt = f"""User: You are an AI assistant specialized in extracting specific information from earning call transcripts. Your task is to analyze the following transcript, extract key information according to the guidelines provided, and format it as JSON according to the specified schema.

Here is the transcript to analyze:

<transcript>
{text}
</transcript>

The transcript has been preprocessed with special markers:
- Multiple spaces are marked as <MULTISPACE>
- Line breaks are marked as <LINEBREAK>
- Tab characters are marked as <TAB>
- Page breaks are marked as <PAGEBREAK>

Your goal is to carefully extract the following detailed information and format it into a JSON object:
1. Bank name
2. Call date
3. Reporting period (in Q-YYYY format, e.g., "Q1-2023")
4. Participants,including full names, misspelled names, title(s) with all variations found in the document, and company(ies) with all variations found in the document
5. Regular expression pattern for the speaker to separate who is speaking from what they're saying
6. All sections in the document in order of appearance
7. Presentation section details including section title, section start page number, section end page number
8. Q&A section details including section title, section start page number, section end page number

Follow these guidelines:

1. For participant names, use the most frequently used spelling for the "name" field and include all variations, including acronyms, potential misspellings or grammatical errors, in the "misspelled_names" array.
2. For participants, list out each mention of a participant with their name and title, then synthesize that information.
3. For participant titles, always search and include all variations found in the document, including acronyms, potential misspellings or grammatical errors.
4. For participant companies, always search and include all variations found in the document, including acronyms, potential misspellings or grammatical errors.
5. For sections flow, only include the titles of the sections. Include all section names in order of appearance in the transcript.
6. For the presentation and q&a sections details, always include the start and end page numbers of the section.
7. Pay close attention to the formatting requirements, especially for the reporting period.
8. If any information is unclear or not explicitly stated in the transcript, use "Not clearly stated" as the value.
9. It should be possible to parse the JSON object from the response.

Guidelines for the speaker_regex_pattern:
- Regex pattern should reliably identify speaker notations regardless of whether they include company information and job titles.
- Must match formats: "Name - Company - Title", "Name - Company", "Name - Title", and standalone names
- Names typically begin with capital letters (e.g., "John Smith") and may appear in ALL UPPERCASE (e.g., "JOHN SMITH")
- Account for the formatting markers around speaker names
- Company names and titles may contain a mix of capital and lowercase letters
- Use flexible spacing around dashes (don't require multiple whitespace)
- Include support for apostrophes, commas, periods, ampersands, capital letters in all segments
- Don't require ending colons in the pattern itself
- Test your pattern against actual examples from the transcript before finalizing

Apply the guidelines and provide only the JSON object as your final response, with no additional markdown, text or explanations.

Here's an example of the expected JSON structure (with generic placeholders):

{{
  "bank_name": "Example Bank",
  "call_date": "YYYY-MM-DD",
  "reporting_period": "QX-YYYY",
  "speaker_regex_pattern": "EXAMPLE_REGEX_PATTERN",
  "participants": [
    {{
      "speaker_name": "John Doe",
      "speaker_misspelled_names": ["Jon Doe", "John Do"],
      "speaker_title": ["Chief Executive Officer", "CEO"],
      "speaker_company": ["Example Bank", "Example Bank Inc."]
    }}
  ],
  "all_document_sections": ["Presentation", "Question and Answer", "Disclaimer"],
  "presentation_section_details": [
    {{
      "presentation_section_title": "Presentation",
      "presentation_section_start_page_number": "3",
      "presentation_section_end_page_number": "5"
    }}
  ],
  "question_and_answer_section_details": [
    {{
      "question_and_answer_section_title": "Question and Answer",
      "question_and_answer_section_start_page_number": "6",
      "question_and_answer_section_end_page_number": "12"
    }}
  ]
}}

Please proceed with your analysis and JSON formatting of the transcript information.
"""

    api_key = os.getenv('ANTHROPIC_API_KEY')
    client = anthropic.Anthropic(api_key=api_key)
    unique_prompt = f"{prompt}\n\n[Request timestamp: {time.time()}]"

    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2048,
            system="You are a financial expert specializing in formatting earnings call transcripts into a structured JSON object.",
            messages=[
                {"role": "user", "content": unique_prompt}
            ],
            temperature=0,
            top_p=0.7,
            top_k=20
        )

        # Get token counts from the API response
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate cost in USD: https://www.anthropic.com/pricing#anthropic-api
        input_cost_per_million = 3.0
        output_cost_per_million = 15.0

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost

        # Get the response text
        if not message.content or len(message.content) == 0:
            return {
                "error": "Empty response from API"
            }
        
        response_text = message.content[0].text
        
        # Try to parse the response as JSON
        try:
            import json
            json_response = json.loads(response_text)
            parsed_successfully = True
        except json.JSONDecodeError:
            parsed_successfully = False
            json_response = None

        return {
            "response": response_text,
            "parsed_json": json_response if parsed_successfully else None,
            "json_parsed_successfully": parsed_successfully,
            "token_counts": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "cost_estimate": {
                "total_cost_usd": total_cost
            }
        }

    except Exception as e:
        return {
            "error": str(e)
        }


"""
def main():
    # Main function to process a PDF transcript and get AI analysis.
    
    parser = argparse.ArgumentParser(description='Extract and analyze earnings call transcript using Claude API.')
    parser.add_argument('pdf_path', help='Path to the PDF transcript')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Extract text from PDF
    print(f"Extracting text from {args.pdf_path}...")
    full_text = extract_text_from_pdf(args.pdf_path)
    
    # Make API call to Claude
    print("Sending text to Claude API for analysis...")
    result = API_call_anthropic(full_text)
    
    # Check for errors
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print token usage and cost
    print(f"Input tokens: {result['token_counts']['input_tokens']}")
    print(f"Output tokens: {result['token_counts']['output_tokens']}")
    print(f"Total tokens: {result['token_counts']['total_tokens']}")
    print(f"Estimated cost: ${result['cost_estimate']['total_cost_usd']:.6f}")
    
    # Save or print the result
    if args.output:
        output_path = args.output
    else:
        # Use the output_file_name function to generate a filename based on the input PDF
        output_path = output_file_name(args.pdf_path)
        print(f"No output file specified, using generated filename: {output_path}")
    
    if result["json_parsed_successfully"]:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result["parsed_json"], f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        print("\nRaw response (could not parse as JSON):")
        print(result["response"])
        # Save raw response to file with _raw suffix
        raw_output_path = output_path.replace(".json", "_raw.txt")
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(result["response"])
        print(f"Raw response saved to {raw_output_path}")


if __name__ == "__main__":
    main()

# Test code - comment out or remove this section when done testing
"""


def test_mode(transcripts_pdf_folder, transcripts_cleantxt_folder, metadata_folder, file_name):
    # test flow
    source_file_path = f"{transcripts_pdf_folder}/{file_name}"
    print(f"Testing with file: {source_file_path}")
    
    text_markup = extract_text_from_pdf(source_file_path)  # Extract
    cleantxt_file_path = f"{transcripts_cleantxt_folder}/{file_name.replace('.pdf', '_clean.txt')}"
    with open(cleantxt_file_path, 'w', encoding='utf-8') as f:
        f.write(text_markup)

    call_response = API_call_anthropic(text_markup)  # API call

    # Check for errors
    if "error" in call_response:
        print(f"Error: {call_response['error']}")
        return
    
    # Print token usage and cost
    print(f"Input tokens: {call_response['token_counts']['input_tokens']}")
    print(f"Output tokens: {call_response['token_counts']['output_tokens']}")
    print(f"Total tokens: {call_response['token_counts']['total_tokens']}")
    print(f"Estimated cost: ${call_response['cost_estimate']['total_cost_usd']:.6f}")
    
    # Save the call_response
    metadata_file_path = f"{metadata_folder}/{file_name.replace('.pdf', '_metadata.json')}"

    if call_response["json_parsed_successfully"]:
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(call_response["parsed_json"], f, indent=2)
        # Add key "path_to_transcript" to metadata at the top level in metadata_file_path
        metadata = json.load(open(metadata_file_path, 'r', encoding='utf-8'))
        metadata["path_to_transcript_txt"] = cleantxt_file_path
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Results saved to {metadata_file_path}")

    else:
        print("Could not parse response as JSON. Saving raw response...")
        with open(metadata_file_path.replace('.json', '_raw.txt'), 'w', encoding='utf-8') as f:
            f.write(call_response["response"])
        print(f"Raw response saved to {metadata_file_path.replace('.json', '_raw.txt')}")

# Test mode

test_mode_enabled, test_file_name = get_test_mode()

if test_mode_enabled:
    transcripts_pdf_folder = get_folder_paths()["transcripts_pdf_folder"]
    transcripts_cleantxt_folder = get_folder_paths()["transcripts_cleantxt_folder"]
    metadata_folder = get_folder_paths()["metadata_folder"]
    test_mode(transcripts_pdf_folder, transcripts_cleantxt_folder, metadata_folder, test_file_name)
else:
    print("Test mode is not enabled. Skipping test.")
