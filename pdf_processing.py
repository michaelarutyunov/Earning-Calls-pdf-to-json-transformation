### IMPORTS ###

import argparse
from logging import config
import pymupdf
import anthropic
import copy
import json
import os
import re
import uuid
from dotenv import load_dotenv

load_dotenv()

### CONFIGURATION ###

def load_config(config_path="config.json"):
    """
    Load configuration from JSON file and return as a dictionary.
    
    Args:
        config_path (str): Path to the configuration file. Default is "config.json"
        
    Returns:
        dict: Complete configuration dictionary or empty dict if file not found or invalid
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            return config_data
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_path}.")
        return {}

def get_folder_paths(config_data):
    """Extract folder paths from config dictionary."""
    folder_paths = config_data.get("folder_paths", {})
    return {
        "transcripts_pdf_folder": folder_paths.get("transcripts_pdf_folder", None),
        "transcripts_cleantxt_folder": folder_paths.get("transcripts_cleantxt_folder", None),
        "metadata_folder": folder_paths.get("metadata_folder", None),
        "utterances_folder": folder_paths.get("utterances_folder", None),
        "final_json_folder": folder_paths.get("final_json_folder", None)
    }

def get_test_mode_info(config_data):
    """Extract test mode information from config dictionary."""
    test_mode = config_data.get("test_mode", {})
    return {
        "enabled": test_mode.get("enabled", False),
        "file_name": test_mode.get("file_name", None),
        "debug_mode": test_mode.get("debug_mode", False)
    }

def get_api_setup(config_data):
    """Extract API setup from config dictionary."""
    api_setup = config_data.get("api_setup", {})
    api_key_name = api_setup.get("api_key_name", None)
    model = api_setup.get("model", None)
    input_cost_per_million = api_setup.get("input_cost_per_million", 0)
    output_cost_per_million = api_setup.get("output_cost_per_million", 0)
    
    return api_key_name, model, input_cost_per_million, output_cost_per_million

def get_cleaning_parameters(config_data):
    """Extract cleaning parameters from config dictionary."""
    cleaning_parameters = config_data.get("cleaning_parameters", {})
    return {
        "keep_bold_tags": cleaning_parameters.get("keep_bold_tags", False),
        "keep_italics_tags": cleaning_parameters.get("keep_italics_tags", False),
    }


### TEXT INGESTION ###

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

    # Remove repeating punctuation characters
    # Handle cases where punctuation is separated by spaces
    text = re.sub(r'([.!?](\s+))\1{2,}', '', text)
    
    # Handle continuous repeating punctuation (like "............")
    text = re.sub(r'([.!?])\1{2,}', '', text)
    
    # Handle cases where dots are mixed with spaces in long sequences
    text = re.sub(r'([.]\s*){2,}', '', text)
    
    # Handle other punctuation characters that might repeat
    # Use re.escape to properly escape special regex characters
    #for punct in ['-', '=', '*', '/', '\\']:
    #    escaped_punct = re.escape(punct)
    #    text = re.sub(r'({0})\1{{2,}}'.format(escaped_punct), punct * 3, text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

def extract_text_with_formatting(pdf_path, config_data, debug_mode=False):
    """Extract text with formatting from PDF, including text from images."""

    doc = pymupdf.open(pdf_path)
    full_text = ""

    cleaning_params = get_cleaning_parameters(config_data)
    keep_bold_tags = cleaning_params["keep_bold_tags"]
    keep_italics_tags = cleaning_params["keep_italics_tags"]
    keep_underline_tags = cleaning_params.get("keep_underline_tags", True)
    
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

                        is_underline = False
                        if "font" in span:
                            is_underline = "underline" in span["font"].lower()
                        if not is_underline and "flags" in span:
                            is_underline = (span["flags"] & 8) > 0

                        # Store formatting info for this text snippet
                        if is_bold or is_italic or is_underline:
                            formatted_blocks[text] = (is_bold, is_italic, is_underline)

        # Apply formatting to the complete text
        for text, (is_bold, is_italic, is_underline) in formatted_blocks.items():
            # Try to find and replace exact text
            if text in complete_text:
                formatted_text = text
                if keep_bold_tags and is_bold:
                    formatted_text = f"<BOLD>{formatted_text}</BOLD>"
                if keep_italics_tags and is_italic:
                    formatted_text = f"<ITALIC>{formatted_text}</ITALIC>"
                if keep_underline_tags and is_underline:
                    formatted_text = f"<UNDERLINE>{formatted_text}</UNDERLINE>"
                complete_text = complete_text.replace(text, formatted_text)

        # Clean the text
        page_text = clean_text(complete_text)

        if page_text:
            full_text += page_text + "\n<PAGE_BREAK>\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1} of {pdf_path}")

    if debug_mode:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'extracted_text.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Extracted text saved to {file_path}")

    doc.close()
    return full_text


### SPEAKER ATTRIBUTION EXTRACTION ###

def API_call(text, config_data, debug_mode=False):
    # Construct the prompt according to the specified format
    prompt = f"""User: You are an AI assistant specialized in extracting speaker attributions from earning call transcripts.

    <task>
    Extract all speaker attributions from the following transcript.    
    Focus only on the text segments that identify who is speaking before their statements.
    For each speaker attribution, return the speaker's name, their title/role if available, and the company they work for.
    Always check the WHOLE transcript from start to end for ALL variations of attributions for every speaker.
    Return call details like bank name, call date and reporting period.
    Return header and footer patterns with formatting tags.
    Format the results as a json object.
    </task>

    <formatting_tags>
    Text contains formatting tags that can help identify who is speaking:
    - Line breaks are marked as <LINEBREAK>
    - Bold text is marked as <BOLD>
    - Multispace is marked as <MULTISPACE>
    </formatting_tags>
    
    <attribution_search_guidelines>
    Guidelines for searching for speaker attribution:
    - Always check the WHOLE transcript from start to end for ALL variations of attributions for every speaker.
    - Speaker attribution always starts with the speaker full name.
    - Speaker's name in the attribution can include speaker's job title and company.
    - Speaker attribution is normally preceded by a <LINEBREAK> tag or a combination of <MULTISPACE> and <LINEBREAK> tags.
    - Speaker attribution normally ends with a formatting tag or a punctuation mark.
    - Speaker attribution NEVER contains a speech.
    - Attribution for operator should always contain the word "Operator".
    </sattribution_search_guidelines>

    <attribution_format_guidelines>
    Guidelines for formatting speaker attribution:
    - Speaker's name, job title and company can be in normal, bold, italic, underlined text, upper and title case, may include punctuations marks like a dash, a colon, a period, or an apostrophe.
    - Speaker's name, job title and company may be separated with a punctuation mark like a dash, a colon, or formatting tags, or a combination of these.
    - Speaker's name, job title and company cannot be separated by a text segment that is not a formatting tag.
    - Speaker's name can include middle names and initials.
    - Speaker's name can be followed by a punctuation and separated from it by a formatting tag or a space.
    - If speaker's name is separated from the job title or company by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    - If speaker's name is followed by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    </attribution_format_guidelines>

    <transcript>    
    Here is the transcript to analyze:
    {text}
    </transcript>

    <examples>
    Here is an example of a speaker attribution with variations in formatting:
    * name followed by a colon, both in bold: <BOLD>SPEAKER NAME: </BOLD>
    * name followed by a colon, both in bold: <BOLD>SPEAKER NAME<BOLD>: </BOLD></BOLD>
    * name followed by a line break, then by a colon in bold: SPEAKER NAME <LINEBREAK> <BOLD>: </BOLD>
    * name in bold followed by a line break, then by a colon: <BOLD>SPEAKER NAME</BOLD> <LINEBREAK> :
    * name in plain text followed by a colon in bold: SPEAKER NAME<BOLD>: </BOLD>
    If found, all such variations should be included in the output.

    Here is an example of a text where the speaker attribution and company are separated by a speech:
    * <MULTISPACE> <LINEBREAK> Name Surname <MULTISPACE> <LINEBREAK> Good morning, everybody. My first question relates to <MULTISPACE> <LINEBREAK> (Company Name) <MULTISPACE> <LINEBREAK> capital return.
    In this case the company name should be ignored and only speaker name should be a part of attribution.

    Here is an example variations of the valid speaker name formats for the same speaker:
    * <LINEBREAK> <BOLD>Name Surname </BOLD> <LINEBREAK>
    * <LINEBREAK> <BOLD>Name M. Surname </BOLD> <LINEBREAK>
    * <LINEBREAK> <BOLD>Name Middlename Surname </BOLD> <LINEBREAK>
    * <LINEBREAK> <BOLD>Name Surname-Surname </BOLD> <LINEBREAK>   
    </examples>

    <guidelines>
    Follow these guidelines:
    1. IMPORTANT: for each speaker, check the whole transcript from start to end for all variations of their speaker attributions.
    2. Check every instance when a full name is mentioned in the transcript and verify if it is a speaker attribution using attribution_search_guidelines.
    3. Include each and every speaker attribution found in the transcript.
    4. Include all variations of speaker names found in the transcript.
    5. Include all variations of speaker titles found in the transcript.
    6. Include all variations of speaker companies found in the transcript.
    7. Pay close attention to the formatting requirements, especially for the reporting period (QX-YYYY) and the header and footer of the transcript.
    8. Include the header and footer patterns with formatting tags in the output.
    </guidelines>

    Here's an example of the expected JSON structure (with generic placeholders):
    <jsonexample>
    {{
    "bank_name": "Example Bank",
    "call_date": "YYYY-MM-DD",
    "reporting_period": "QX-YYYY",
    "header_pattern": "HEADER_PATTERN",
    "footer_pattern": "FOOTER_PATTERN",
    "participants": [
        {{
        "speaker_name_variants": ["John Doe", "Jon Doe", "John Do"],
        "speaker_title_variants": ["Chief Executive Officer", "CEO"],
        "speaker_company_variants": ["Example Bank", "Example Bank Inc.", "EB"],
        "speaker_attributions": ["<LINEBREAK> JOHN DOE:", "<LINEBREAK> John Doe - CEO - Example Bank <LINEBREAK>", "<LINEBREAK> John Doe - EB - CEO <LINEBREAK>"]
        }}
    ]
    }}
    </jsonexample>

    <json_validation>
    Your response must be valid, parseable JSON. Ensure:
    - Use single curly braces for objects, not double
    - All strings are properly quoted
    - No trailing commas in arrays or objects
    - All keys and values follow proper JSON syntax
    - Test your JSON structure mentally before providing it as output
    </json_validation>

    It should be possible to parse the JSON object from the response.
    Provide only the JSON object as your final response, with no additional text or explanations.
    """
    api_key_name, model, input_cost_per_million, output_cost_per_million = get_api_setup(config_data)
    api_key = os.getenv(api_key_name)
    client = anthropic.Anthropic(api_key=api_key)
    # unique_prompt = f"{prompt}\n\n[Request timestamp: {time.time()}]"

    if debug_mode:
        print(f"Running API call using {model}...")

    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4096,
            system="You are a financial expert specializing in formatting earnings call transcripts into a structured JSON object.",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0.7,
            top_k=20
        )

        # Get token counts from the API response and calculate cost
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        input_cost = (input_tokens / 1_000_000) * float(input_cost_per_million)
        output_cost = (output_tokens / 1_000_000) * float(output_cost_per_million)
        total_cost = input_cost + output_cost

        # Get the response text
        if not message.content or len(message.content) == 0:
            return {
                "error": "Empty response from API"
            }

        response_text = message.content[0].text
        
        # Always save the raw response for debugging
        if debug_mode:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            raw_response_path = os.path.join(script_dir, 'api_response_raw.txt')
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"Raw API response saved to {raw_response_path}")

        # Try to parse the response as JSON
        try:
            # Check if the response is wrapped in markdown code blocks
            clean_response = response_text
            if response_text.startswith("```json"):
                end_marker = "```"
                end_pos = response_text.rfind(end_marker)
                if end_pos > 0:
                    clean_response = response_text[7:end_pos].strip()
                    
                    if debug_mode:
                        cleaned_path = os.path.join(script_dir, 'api_response_cleaned.txt')
                        with open(cleaned_path, 'w', encoding='utf-8') as f:
                            f.write(clean_response)
                        print(f"Cleaned API response saved to {cleaned_path}")
            
            json_response = json.loads(clean_response)
            parsed_successfully = True
            
            if debug_mode:
                parsed_path = os.path.join(script_dir, 'api_response_parsed.json')
                with open(parsed_path, 'w', encoding='utf-8') as f:
                    json.dump(json_response, f, indent=2)
                print(f"Parsed JSON saved to {parsed_path}")
                
        except json.JSONDecodeError as e:
            parsed_successfully = False
            json_response = None
            
            if debug_mode:
                print(f"JSON parsing error: {str(e)}")
                error_path = os.path.join(script_dir, 'api_response_error.txt')
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                    f.write("\n\n--- JSON PARSE ERROR ---\n")
                    f.write(str(e))
                print(f"Error details saved to {error_path}")

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
        if debug_mode:
            print(f"API call exception: {str(e)}")
        return {
            "error": str(e)
        }


### TEXT PARSING ###

"""
def remove_unused_sections(metadata_file: str, debug_mode: bool) -> str:
    
    Remove sections before the presentation start page.
    
    Args:
        transcript_text (str): Full transcript text
        metadata_file (str): Path to the metadata file
        
    Returns:
        str: Transcript text starting from the presentation start page
        
    Raises:
        KeyError: If required metadata fields are missing
        ValueError: If page numbers are invalid
    
    if debug_mode:
        print(f"Removing unused sections...")

    metadata = load_metadata(metadata_file, debug_mode)
    text = load_transcript(metadata['path_to_transcript_txt'], debug_mode)

    try:
        presentation_start_page = int(metadata['presentation_section_details'][0]['presentation_section_start_page_number'])
    except (KeyError, IndexError) as e:
        raise KeyError(f"Required metadata fields are missing: {str(e)}")

    # print(presentation_start_page)

    pages = text.split('<PAGE_BREAK>')
    if presentation_start_page < 1 or presentation_start_page > len(pages):
        raise ValueError(f"Invalid start page number: {presentation_start_page}, total pages={len(pages)}")

    # Select pages starting from the presentation start page
    selected_text = "<PAGE_BREAK>".join(pages[presentation_start_page - 1:])

    return selected_text
"""

def get_utterances(text, api_response, debug_mode=False):
    """
    Extract utterances from transcript text using speaker attributions from API response.
    
    Args:
        text (str): The transcript text
        api_response (dict): The response from the API call containing speaker attributions
        debug_mode (bool): Whether to print debug information
        
    Returns:
        list: List of dictionaries containing speaker and their utterance
    """
    
    # Check if api_response is None or doesn't have parsed_json
    if api_response is None:
        if debug_mode:
            print("API response is None")
        return []
    
    # Get parsed JSON from API response
    parsed_json = api_response.get("parsed_json")
    
    # Check if parsed_json is None
    if parsed_json is None:
        if debug_mode:
            print("Parsed JSON is None - JSON parsing likely failed")
            if "response" in api_response:
                print("Raw API response:", api_response["response"][:200] + "...")  # Print first 200 chars
        return []
    
    # Check if participants key exists
    if "participants" not in parsed_json:
        if debug_mode:
            print("No 'participants' key in parsed JSON")
            print("Available keys:", list(parsed_json.keys()))
        return []
    
    # Collect all speaker attributions into a list
    all_attributions = []
    for participant in parsed_json.get("participants", []):
        speaker_name = participant.get("speaker_name_variants", ["Unknown"])[0]
        for attribution in participant.get("speaker_attributions", []):
            all_attributions.append({
                "speaker_name": speaker_name,
                "attribution": attribution
            })

    if debug_mode:
        print(f"Found {len(all_attributions)} speaker attributions")
        print("Writing all attributions to all_attributions.txt...")
        
        # Create a file to save all attributions for debugging
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'all_attributions.txt')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Total attributions found: {len(all_attributions)}\n\n")
            for i, attr in enumerate(all_attributions):
                f.write(f"Attribution {i+1}:\n")
                f.write(f"  Speaker: {attr['speaker_name']}\n")
                f.write(f"  Text: {attr['attribution']}\n\n")
        
    if not all_attributions:
        return []
    
    # Find all matches for all attributions directly in the text
    all_matches = []
    for attr_info in all_attributions:
        attribution = attr_info["attribution"]
        # Find all occurrences of this attribution in the text
        start_pos = 0
        while True:
            pos = text.find(attribution, start_pos)
            if pos == -1:
                break
            all_matches.append({
                "speaker_name": attr_info["speaker_name"],
                "start": pos,
                "end": pos + len(attribution)
            })
            start_pos = pos + 1  # Move past the current match to find the next one
    
    # Sort matches by their position in the text
    all_matches.sort(key=lambda x: x["start"])
    
    if debug_mode:
        print(f"Found {len(all_matches)} total matches in the text")
    
    if not all_matches:
        return []
    
    # Extract utterances
    utterances = []
    for i in range(len(all_matches) - 1):
        current_match = all_matches[i]
        next_match = all_matches[i + 1]
        
        # Get the text between the end of the current attribution and the start of the next
        utterance_text = text[current_match["end"]:next_match["start"]]
        
        utterances.append({
            "speaker": current_match["speaker_name"],
            "utterance": utterance_text.strip()
        })
    
    # Handle the last speaker's utterance (from last attribution to end of text)
    last_match = all_matches[-1]
    last_utterance = text[last_match["end"]:]
    utterances.append({
        "speaker": last_match["speaker_name"],
        "utterance": last_utterance.strip()
    })
    
    if debug_mode:
        print(f"Extracted {len(utterances)} utterances")

    #
    
    return utterances

def clean_utterances(utterances: list, api_response: dict) -> list:
    cleaned_utterances = []

    # Remove empty utterances
    utterances = [utterance for utterance in utterances if utterance['utterance'].strip()]

    # Remove header from all utterances
    header_pattern = api_response.get("header_pattern", None) if isinstance(api_response, dict) else None
    if header_pattern:
        cleaned_header_pattern = re.sub(r'^(?:<MULTISPACE>|<LINEBREAK>|<BOLD>)+\s*', '', header_pattern)
        utterances = [utterance for utterance in utterances if not re.search(cleaned_header_pattern, utterance['utterance'], re.IGNORECASE)]

        utterances = [utterance for utterance in utterances if not re.search(cleaned_header_pattern, utterance['utterance'])]

    # Remove footer from all utterances
    footer_pattern = api_response.get("footer_pattern")
    if footer_pattern:
        cleaned_footer_pattern = re.sub(r'^(?:<MULTISPACE>|<LINEBREAK>|<BOLD>)+\s*', '', footer_pattern)
        utterances = [utterance for utterance in utterances if not re.search(cleaned_footer_pattern, utterance['utterance'], re.IGNORECASE)]

        utterances = [utterance for utterance in utterances if not re.search(cleaned_footer_pattern, utterance['utterance'])]
    
    debug_counts = {
        'angle_brackets': 0,
        'parentheses': 0,
        # 'double_quotes': 0,
        # 'single_quotes': 0,
        'double_slashes': 0,
        'backslashes': 0
    }

    for utterance in utterances:
        text = utterance['utterance']

        # Remove text within <>
        angle_brackets_count = len(re.findall(r'<[^>]*>', text))
        text = re.sub(r'<[^>]*>', '', text)
        debug_counts['angle_brackets'] += angle_brackets_count

        # Remove text within ()
        parentheses_count = len(re.findall(r'\([^)]*\)', text))
        text = re.sub(r'\([^)]*\)', '', text)
        debug_counts['parentheses'] += parentheses_count

        # Remove text within //
        double_slashes_count = len(re.findall(r'//[^/]*//', text))
        text = re.sub(r'//[^/]*//', '', text)
        debug_counts['double_slashes'] += double_slashes_count

        # Remove text within \\
        backslashes_count = len(re.findall(r'\\[^\\]*\\', text))
        text = re.sub(r'\\[^\\]*\\', '', text)
        debug_counts['backslashes'] += backslashes_count

        # Remove extra white spaces
        text = ' '.join(text.split())

        # Create a new utterance with a UUID if it doesn't have one
        cleaned_utterance = {
            'speaker': utterance['speaker'],
            'utterance': text
        }
        
        # Add UUID if it exists in the original utterance, otherwise generate a new one
        if 'uuid' in utterance:
            cleaned_utterance['uuid'] = utterance['uuid']
        else:
            cleaned_utterance['uuid'] = str(uuid.uuid4())
            
        cleaned_utterances.append(cleaned_utterance)

    return cleaned_utterances

def create_and_save_final_json(api_response, cleaned_utterances, output_path, debug_mode=False):
    """
    Combine the API response and cleaned utterances into a final JSON structure.
    
    Args:
        api_response (dict): The response from the API call containing speaker attributions and metadata
        cleaned_utterances (list): List of cleaned utterances with speaker and text
        debug_mode (bool): Whether to print debug information
        
    Returns:
        dict: Combined JSON with metadata from API and cleaned utterances
    """
    if debug_mode:
        print("Creating final JSON from API response and cleaned utterances...")
    
    # Start with the API response as the base, ensuring we have a dictionary
    parsed_json = api_response.get("parsed_json")
    if parsed_json is None:
        if debug_mode:
            print("Warning: parsed_json is None, creating empty dictionary")
        final_json = {}
    else:
        final_json = copy.deepcopy(parsed_json)
    
    # Add the utterances to the final JSON
    final_json["utterances"] = cleaned_utterances
    
    if debug_mode:
        print(f"Final JSON created with {len(cleaned_utterances)} utterances")
    
    # Save the final JSON to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2)
    print(f"Final JSON saved to {output_path}")
    
    return final_json


### MAIN FUNCTION ###

def main():
    # Main function to process a PDF transcript and get AI analysis.
    
    parser = argparse.ArgumentParser(description='Extract and analyze earnings call transcript using Claude API.')
    parser.add_argument('pdf_path', nargs='?', help='Path to the PDF transcript (optional, will use test file if not provided)')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    
    args = parser.parse_args()

    # Load config
    config_data = load_config()

    # Extract test mode and debug mode information
    test_mode_info = get_test_mode_info(config_data)
    test_mode_enabled = test_mode_info['enabled']
    debug_mode = test_mode_info['debug_mode']

    # Extract folder paths
    folder_paths = get_folder_paths(config_data)
    transcripts_pdf_folder = folder_paths['transcripts_pdf_folder']
    final_json_folder = folder_paths['final_json_folder']

    # Determine the file to process
    if args.pdf_path:
        file_path = args.pdf_path
        file_name = file_path.split('/')[-1]
    elif test_mode_enabled:
        print("No PDF path provided, using test file from config.")
        file_name = test_mode_info['file_name']
        file_path = f"{transcripts_pdf_folder}/{file_name}"
    else:
        print("Error: No PDF path provided and test mode is not enabled.")
        return

    # Extract text from PDF
    print(f"Extracting text from {file_path}...")
    full_text = extract_text_with_formatting(file_path, config_data, debug_mode)
    
    # Make API call
    print("Sending text to API for analysis...")
    api_response = API_call(full_text, config_data, debug_mode)
    
    # Check for errors in the API call result
    if "error" in api_response:
        print(f"Error: {api_response['error']}")
        return
    
    # Get utterances
    utterances = get_utterances(full_text, api_response, debug_mode)
    
    # Clean utterances
    cleaned_utterances = clean_utterances(utterances, api_response)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"{final_json_folder}/{file_name.replace('.pdf', '_final.json')}"
    
    # Create and save final JSON
    create_and_save_final_json(api_response, cleaned_utterances, output_path, debug_mode)

    # Check if token_counts is available
    if "token_counts" in api_response:
        print(f"Input tokens: {api_response['token_counts']['input_tokens']}")
        print(f"Output tokens: {api_response['token_counts']['output_tokens']}")
        print(f"Total tokens: {api_response['token_counts']['total_tokens']}")
        print(f"Estimated cost: ${api_response['cost_estimate']['total_cost_usd']:.2f}")
    else:
        print("Token counts not available in the API response.")


if __name__ == "__main__":
    main()