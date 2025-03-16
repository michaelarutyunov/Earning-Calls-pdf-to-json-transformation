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
    # First handle cases where punctuation is separated by spaces
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

    if debug_mode:
        print(f"Extracting formatted text with images...")

    doc = pymupdf.open(pdf_path)
    full_text = ""

    cleaning_params = get_cleaning_parameters(config_data)
    keep_bold_tags = cleaning_params["keep_bold_tags"]
    keep_italics_tags = cleaning_params["keep_italics_tags"]
    
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
                if keep_bold_tags and is_bold:
                    formatted_text = f"<BOLD>{formatted_text}</BOLD>"
                if keep_italics_tags and is_italic:
                    formatted_text = f"<ITALIC>{formatted_text}</ITALIC>"
                complete_text = complete_text.replace(text, formatted_text)

        # Clean the text
        page_text = clean_text(complete_text)

        if page_text:
            full_text += page_text + "\n<PAGE_BREAK>\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1} of {pdf_path}")

    # make whole text lowercase
    #full_text = full_text.lower()
    
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
    Return call details like bank name, call date and reporting period.
    Format the results as a json object.
    </task>

    <speaker_attribution_guidelines>
    Guidelines for speaker attribution:
    - Speaker attribution always starts with the speaker's name
    - Speaker's name in the attribution can be in normal or bold text, upper or title case. 
    - Speaker's name in the attribution can be followed by the speaker's title and company.
    - If the speaker's name in the attribution is followed by the speaker's title and company, there may be a separator like a dash or a colon between the name and the title/company
    - Speaker attribution often preceded by a <LINEBREAK> tag.
    - Speaker attribution may include punctuations marks like a dash, a colon, a period, or an apostrophe. Pay close attention to these punctuations marks and tags around them.
    - Speaker attribution may multiple different tags, which may appear inconsistently across the attributions. Ensure you capture ALL variations of speaker attributions.
    - Speaker attributions for the same speaker may appear in inconsistent formats. Ensure you capture ALL variations of speaker attributions, even if they appear only once in the transcript.
    - Attribution for operator should always contain the word "Operator".
    </speaker_attribution_guidelines>

    <formatting_tags>
    Here are the formatting tags used in the transcript:
    - Line breaks are marked as <LINEBREAK>
    - Bold text is marked as <BOLD>
    - Italic text is marked as <ITALIC>
    - Multispace is marked as <MULTISPACE>
    - Tab characters are marked as <TAB>
    - Page breaks are marked as <PAGEBREAK>
    </formatting_tags>

    Here is the transcript to analyze:
    <transcript>
    {text}
    </transcript>

    <inconsistent_formatting_examples>
    Here are some examples of inconsistent formatting:
    * <BOLD>SPEAKER NAME: </BOLD>
    * <BOLD>SPEAKER NAME<BOLD>: </BOLD></BOLD>
    * SPEAKER NAME <LINEBREAK> <BOLD>: </BOLD>
    * <BOLD>SPEAKER NAME</BOLD> <LINEBREAK> :
    </inconsistent_formatting_examples>

    Follow these guidelines:
    <guidelines>
    1. Include all speaker attributions found in the transcript even if they appear in inconsistent formats. Do not miss any speakers or attributions even if they appear only once in the transcript.
    2. Include all variations of speaker names found in the transcript.
    3. Include all variations of speaker titles found in the transcript.
    4. Include all variations of speaker companies found in the transcript.
    5. Pay close attention to the formatting requirements, especially for the reporting period (QX-YYYY)
    </guidelines>

    <completeness>
    Your extraction should be comprehensive - ensure you capture ALL speaker attributions, even if there are more than five variations of the same speaker attribution.
    </completeness>

    <testing>
    After extracting all attributions, double-check the transcript for any missed patterns or variations in how speakers are indicated.
    </testing>

    Here's an example of the expected JSON structure (with generic placeholders):
    <jsonexample>
    {{
    "bank_name": "Example Bank",
    "call_date": "YYYY-MM-DD",
    "reporting_period": "QX-YYYY",
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

def get_utterances_old(metadata_file: str, debug_mode: bool) -> list:
    """
    Extract utterances from transcript text using a speaker regex pattern.
    For each match, extract the text between the current and next match as an utterance.
    
    Args:
        metadata_file (str): Path to the metadata file with the speaker regex pattern
        
    Returns:
        Optional[dict]: Dictionary containing speaker and their utterance, or None if no speaker is found
        
    Raises:
        KeyError: If required metadata fields are missing
    """

    # Load metadata and transcript
    try:
        metadata = load_metadata(metadata_file, debug_mode)
        text = load_transcript(metadata['path_to_transcript_txt'], debug_mode)

        speaker_regex_pattern = re.compile(metadata['speaker_regex_pattern'], re.MULTILINE)

        matches = list(speaker_regex_pattern.finditer(text))
    except KeyError as e:
        raise KeyError(f"Required metadata fields are missing: {str(e)}")

    if not matches:
        if debug_mode:
            print("No matches found in get_utterances")
        return []

    if debug_mode:
        print(f"get_utterances found {len(matches)} matches")

    # Prepare a set of all possible speaker names (normalized to lowercase)
    valid_speakers = set()
    for participant in metadata['participants']:
        # Add the primary speaker name
        valid_speakers.add(participant['speaker_name'].lower())
        # Add any misspelled variations
        if 'speaker_misspelled_names' in participant and participant['speaker_misspelled_names']:
            for variant in participant['speaker_misspelled_names']:
                valid_speakers.add(variant.lower())

    # SAVE VALID SPEAKER SET TO FILE
    if debug_mode:
        print("Writing valid speakers to file...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'valid_speakers.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            for speaker in valid_speakers:
                f.write(speaker + '\n')

    # Filter matches to only include valid speakers and collect the sequence of speakers
    filtered_matches = []
    speaker_sequence = []

    for match in matches:
        if debug_mode:
            if match.lastindex is None or match.lastindex < 1:
                raise ValueError("Regex pattern does not contain a capturing group for the speaker name.")

        if match.group(1) is None:
            # Handle the case where nothing was captured
            continue

        full_speaker_text = match.group(1).strip().lower()
        match_found = False

        for valid_name in valid_speakers:
            if valid_name in full_speaker_text:
                filtered_matches.append(match)
                speaker_sequence.append(valid_name)
                match_found = True
                break

        if not match_found:
            for valid_name in valid_speakers:
                words_in_valid_name = valid_name.split()
                if all(word in full_speaker_text for word in words_in_valid_name):
                    filtered_matches.append(match)
                    speaker_sequence.append(valid_name)
                    break

    # Extract utterances using the filtered matches

    if debug_mode:
        print(f"{len(filtered_matches)} matches contain valid speakers")
        for i, match in enumerate(matches):
            print(f"Match {i}: {match.group(0)}")
            print(f"Group 1: {match.group(1)}")

    result = []
    for i in range(len(filtered_matches) - 1):
        speaker = speaker_sequence[i].title()
        start_pos = filtered_matches[i].end()
        end_pos = filtered_matches[i + 1].start()
        utterance = text[start_pos:end_pos].strip()
        utterance_id = str(uuid.uuid4())

        result.append({
            "uuid": utterance_id,
            "speaker": speaker,
            "utterance": utterance
        })

    if filtered_matches:
        speaker = speaker_sequence[-1].title()
        utterance = text[matches[-1].end():].strip()
        utterance_id = str(uuid.uuid4())
        result.append({
            "uuid": utterance_id,
            "speaker": speaker,
            "utterance": utterance
        })

    return result

def clean_utterances(utterances: list, debug_mode: bool) -> list:
    cleaned_utterances = []

    # Remove empty utterances
    utterances = [utterance for utterance in utterances if utterance['utterance'].strip()]

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

    if debug_mode:
        print("Saving debug counts to clean_debug.txt")
        with open('clean_debug.txt', 'w', encoding='utf-8') as debug_file:
            for key, count in debug_counts.items():
                debug_file.write(f"{key}: {count}\n")

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
    result = API_call(full_text, config_data, debug_mode)
    
    # Check for errors in the API call result
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Get utterances
    utterances = get_utterances(full_text, result, debug_mode)
    
    # Clean utterances
    cleaned_utterances = clean_utterances(utterances, debug_mode)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"{final_json_folder}/{file_name.replace('.pdf', '_final.json')}"
    
    # Create and save final JSON
    create_and_save_final_json(result, cleaned_utterances, output_path, debug_mode)

    # Check if token_counts is available
    if "token_counts" in result:
        print(f"Input tokens: {result['token_counts']['input_tokens']}")
        print(f"Output tokens: {result['token_counts']['output_tokens']}")
        print(f"Total tokens: {result['token_counts']['total_tokens']}")
        print(f"Estimated cost: ${result['cost_estimate']['total_cost_usd']:.6f}")
    else:
        print("Token counts not available in the API response.")


if __name__ == "__main__":
    main()