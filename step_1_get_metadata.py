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
            return test_mode.get("enabled", False), test_mode.get("file_name", None), test_mode.get("debug_mode", False)
    except FileNotFoundError:
        print(f"Configuration file {config_paths} not found.")
        return False, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_paths}.")
        return False, None


def get_api_setup():
    """Get API setup from folder_config.json."""
    config_paths = "folder_config.json"

    try:
        with open(config_paths, 'r', encoding='utf-8') as f:
            config = json.load(f)
            api_setup = config.get("api_setup", {})
            return api_setup.get("api_key_file", None), api_setup.get("api_key_name", None)
    except FileNotFoundError:
        print(f"Configuration file {config_paths} not found.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_paths}.")
        return None, None


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

    # If more than 3 same punctuation characters are found in a row, replace them with a single instance of that punctuation character
    # text = re.sub(r'\.{3,}', '', text)
    # text = re.sub(r'(\. ){3,}', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_text_from_pdf(path, debug_mode=False):
    """Extract text from PDF file."""

    if debug_mode:
        print(f"Extracting text...")

    doc = pymupdf.open(path)
    print(f"pages found in {path}: {doc.page_count}")

    full_text = "" 

    # Extract text from each page
    if debug_mode:
        print(f"Extracting and cleaning text from each page...")

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


def extract_dict_from_pdf(pdf_path, debug_mode=False):
    """Extract text with formatting from PDF, including text from images."""

    if debug_mode:
        print(f"Extracting formatted text with images...")

    doc = pymupdf.open(pdf_path)
    full_text = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # Get all text including from images
        complete_text = page.get_text("text")

        # Get text with formatting information
        page_dict = page.get_text("dict")
        formatted_blocks = {}

        # Process each block to capture formatting
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Get the text content
                        text = span["text"]
                        if not text.strip():
                            continue

                        # Check for formatting
                        is_bold = False
                        if "font" in span:
                            is_bold = "bold" in span["font"].lower()
                        if not is_bold and "flags" in span:
                            is_bold = (span["flags"] & 2) > 0

                        is_italic = False
                        if "font" in span:
                            is_italic = "italic" in span["font"].lower()
                        if not is_italic and "flags" in span:
                            is_italic = (span["flags"] & 4) > 0

                        # Store formatting info for this text snippet
                        if is_bold or is_italic:
                            formatted_blocks[text] = (is_bold, is_italic)

        # Apply formatting to the complete text
        for text, (is_bold, is_italic) in formatted_blocks.items():
            # Try to find and replace exact text
            if text in complete_text:
                formatted_text = text
                if is_bold:
                    formatted_text = f"<BOLD>{formatted_text}</BOLD>"
                if is_italic:
                    formatted_text = f"<ITALIC>{formatted_text}</ITALIC>"
                complete_text = complete_text.replace(text, formatted_text)

        # Clean the text
        if debug_mode:
            print(f"Cleaning text from page {page_num+1}...")
        page_text = clean_text(complete_text)

        if page_text:
            full_text += page_text + "\n<PAGE_BREAK>\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1} of {pdf_path}")

    doc.close()
    return full_text


def API_call_anthropic(text, debug_mode=False):
    # Construct the prompt according to the specified format
    prompt = f"""User: You are an AI assistant specialized in extracting specific information from earning call transcripts. Your task is to analyze the following transcript, extract key information according to the guidelines provided, and format it as JSON according to the specified schema.

Here is the transcript to analyze:

<transcript>
{text}
</transcript>

The text contains several formatting tags:
- Line breaks are marked as <LINEBREAK>
- Bold text is marked as <BOLD>
- Italic text is marked as <ITALIC>
- Multispace is marked as <MULTISPACE>
- Tab characters are marked as <TAB>
- Page breaks are marked as <PAGEBREAK>

Your goal is to carefully extract the following detailed information and format it into a JSON object:
1. Bank name
2. Call date
3. Reporting period (in Q-YYYY format, e.g., "Q1-2023")
4. Participants' full names, misspelled names, job title(s) with all variations found in the document, and company(ies) with all variations found in the document
5. Participants' speaker flags that show how the speaker is marked in the transcript to attribute the speech to the correct speaker
6. All sections in the document in order of appearance
7. Presentation section details including section title, section start page number, section end page number
8. Q&A section details including section title, section start page number, section end page number

Follow these guidelines:

1. For participant names, use the most frequently used spelling for the "name" field and include all variations, including acronyms, potential misspellings or grammatical errors, in the "misspelled_names" array.
2. For participant titles, always search and include all variations found in the document, including acronyms, potential misspellings or grammatical errors.
3. For participant companies, always search and include all variations found in the document, including acronyms, potential misspellings or grammatical errors.
4. For speaker flags, if necessary include the adjacent formatting tags like <LINEBREAK> etc.
5. For sections flow, only include the titles of the sections. Include all section names in order of appearance in the transcript.
6. For the presentation and q&a sections details, always include the start and end page numbers of the section.
7. Pay close attention to the formatting requirements, especially for the reporting period.
8. If any information is unclear or not explicitly stated in the transcript, use "Not clearly stated" as the value.
9. It should be possible to parse the JSON object from the response.

Apply the guidelines and provide only the JSON object as your final response, with no additional markdown, text or explanations.

Here's an example of the expected JSON structure (with generic placeholders):

{{
  "bank_name": "Example Bank",
  "call_date": "YYYY-MM-DD",
  "reporting_period": "QX-YYYY",
  "participants": [
    {{
      "speaker_name": "John Doe",
      "speaker_misspelled_names": ["Jon Doe", "John Do"],
      "speaker_title": ["Chief Executive Officer", "CEO"],
      "speaker_company": ["Example Bank", "Example Bank Inc.", "EB"],
      "speaker_flags": ["<LINEBREAK> JOHN DOE:", "<LINEBREAK> John Doe - CEO - Example Bank <LINEBREAK>", "<LINEBREAK> John Doe - EB - CEO <LINEBREAK>"]
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
    if debug_mode:
        print(f"Setting up API call...")

    api_key_name, api_key_file = get_api_setup()
    api_key = os.getenv(api_key_name)
    client = anthropic.Anthropic(api_key=api_key)
    unique_prompt = f"{prompt}\n\n[Request timestamp: {time.time()}]"

    if debug_mode:
        print(f"Running API call...")

    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4096,
            system="You are a financial expert specializing in formatting earnings call transcripts into a structured JSON object.",
            messages=[
                {"role": "user", "content": unique_prompt}
            ],
            temperature=0,
            top_p=0.7,
            top_k=20
        )

        if debug_mode:
            print(f"Processing API response...")

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


def test_mode(transcripts_pdf_folder, transcripts_cleantxt_folder, metadata_folder, file_name, debug_mode):
    # test flow
    source_file_path = f"{transcripts_pdf_folder}/{file_name}"
    print(f"Testing with file: {source_file_path}")

    text_markup = extract_dict_from_pdf(source_file_path, debug_mode)
    cleantxt_file_path = f"{transcripts_cleantxt_folder}/{file_name.replace('.pdf', '_clean.txt')}"
    with open(cleantxt_file_path, 'w', encoding='utf-8') as f:
        f.write(text_markup)

    call_response = API_call_anthropic(text_markup, debug_mode)  # API call

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


test_mode_enabled, test_file_name, debug_mode = get_test_mode()

if test_mode_enabled:
    transcripts_pdf_folder = get_folder_paths()["transcripts_pdf_folder"]
    transcripts_cleantxt_folder = get_folder_paths()["transcripts_cleantxt_folder"]
    metadata_folder = get_folder_paths()["metadata_folder"]
    test_mode(transcripts_pdf_folder, transcripts_cleantxt_folder, metadata_folder, test_file_name, debug_mode)
else:
    print("Test mode is not enabled. Skipping test.")
