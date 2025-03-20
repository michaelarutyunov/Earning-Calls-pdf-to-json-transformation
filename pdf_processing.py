### IMPORTS ###

import spacy
import argparse
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
        #"transcripts_cleantxt_folder": folder_paths.get("transcripts_cleantxt_folder", None),
        #"metadata_folder": folder_paths.get("metadata_folder", None),
        #"utterances_folder": folder_paths.get("utterances_folder", None),
        "final_json_folder": folder_paths.get("final_json_folder", None),
    }

def get_test_mode_info(config_data):
    """Extract test mode information from config dictionary."""
    test_mode = config_data.get("test_mode", {})
    return {
        "enabled": test_mode.get("enabled", False),
        "file_name": test_mode.get("file_name", None),
        "debug_mode": test_mode.get("debug_mode", False),
        "diagnostics_folder": test_mode.get("diagnostics_folder", None)
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
        "keep_underline_tags": cleaning_parameters.get("keep_underline_tags", False),
        "keep_capitalization_tags": cleaning_parameters.get("keep_capitalization_tags", False)
    }


### PDF IMPORT AND TEXT PRE-PROCESSING ###

# Move punctuation outside of bold tags and adjust colons from "word :" to "word: "
def normalize_punctuation(text):
    """Move punctuation outside of bold tags."""
    # Manage colons
    # text = re.sub(r'(\w):', r'\1 :', text) # add space before colon
    text = re.sub(r':(\w)', r': \1', text)  # add space after colon
    text = re.sub(r'(\w+)\s+:', r'\1:', text)  # remove space before colon

    # Remove isolated punctuation marks like " : "
    text = re.sub(r'\s+([,.;:!?])\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# Optimize TEXT tags (TAG_2, TAG_4, TAG_3)
def optimize_text_tags(text):
    """Optimize text tags by removing unnecessary formatting tags."""

    # Remove TEXT tags with single letter, number or punctuation (from "<TAG> [char] <TAG>" to "<TAG>")
    text = re.sub(r'<TAG_2>\s*[A-Za-z]\s*<TAG_2>', r' <TAG_2> ', text, flags=re.IGNORECASE)
    text = re.sub(r'<TAG_2>\s*(\d+)\s*<TAG_2>', r' <TAG_2>', text, flags=re.IGNORECASE)
    text = re.sub(r'<TAG_2>\s*[,.;:!?]\s*<TAG_2>', r' <TAG_2>', text, flags=re.IGNORECASE)
    text = re.sub(r'<TAG_4>\s*[A-Za-z]\s*<TAG_4>', r' <TAG_4> ', text, flags=re.IGNORECASE)
    text = re.sub(r'<TAG_4>\s*(\d+)\s*<TAG_4>', r' <TAG_4> ', text, flags=re.IGNORECASE)

    # optimize multiple TEXT tags in a row, keep only one
    text = re.sub(r'(<TAG_4>\s*){2,}', r'\1', text)
    text = re.sub(r'(<TAG_3>\s*){2,}', r'\1', text)
    text = re.sub(r'(<TAG_2>\s*){2,}', r'\1', text)

    # Ensure single space between tags
    text = re.sub(r'>\s+<', '> <', text)
    text = re.sub(r'><', '> <', text)
    
    return text

# Optimize WORD tags (BOLD, ITALIC, UNDERLINE)
def optimize_word_tags(text):
    """Clean up text by removing unnecessary formatting tags."""

    # Remove WORD tags if they are around punctuation, numbers and single letters (from "<BOLD-> [char]] <-BOLD>" to "")
    text = re.sub(r'<BOLD->\s*([:,.;?!])\s*<-BOLD>', r' \1 ', text)
    text = re.sub(r'<BOLD->\s*([A-Za-z])\s*<-BOLD>', r' \1 ', text)
    text = re.sub(r'<BOLD->\s*(\d+)\s*<-BOLD>', r' \1 ', text)
    
    # Optimize multiple WORD tags
    text = re.sub('<-BOLD>\s*<BOLD->', '', text)
    text = re.sub('<-ITALIC>\s*<ITALIC->', '', text)
    text = re.sub('<-UNDERLINE>\s*<UNDERLINE->', '', text)

    # Ensure single space between tags
    text = re.sub(r'>\s+<', '> <', text)
    text = re.sub(r'><', '> <', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Format spaced headers (from "Q 2 2 0 2 3 E A R N I N G S" to "Q2 2023 EARNINGS")
def format_spaced_headers(text):
    # Pattern to identify spaced headers - letters with spaces between them
    spaced_header_pattern = r'(?<!\S)(?<!<)([A-Z](\s+[A-Z])+)(?![^<]*>)(?!\S)'

    # Function to process each matched header
    def process_header(match):
        spaced_text = match.group(0)

        # First approach: Split by multiple spaces (2 or more)
        if re.search(r'\s{2,}', spaced_text):
            words = re.split(r'\s{2,}', spaced_text)
            condensed_words = [re.sub(r'\s+', '', word) for word in words]
            return ' '.join(condensed_words)
        else:
            # For headers with single spaces between letters but no clear word boundaries
            condensed = re.sub(r'\s+', '', spaced_text)

            # Insert spaces before uppercase letters that follow lowercase or numbers
            spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', condensed)

            # For dates like "2 0 2 3", keep them together
            spaced = re.sub(r'(\d)\s+(\d)', r'\1\2', spaced)

            return spaced

    # Apply the regex substitution with the processing function
    text = re.sub(spaced_header_pattern, process_header, text)

    return text

# Remove repeating punctuation characters
def remove_repeating_punctuation(text):
    """Remove repeating punctuation characters."""
    # Handle cases where punctuation is separated by spaces
    text = re.sub(r'([.!?](\s+))\1{2,}', '', text)
    
    # Handle continuous repeating punctuation (like "............")
    text = re.sub(r'([.!?])\1{2,}', '', text)
    
    # Handle cases where dots are mixed with spaces in long sequences
    text = re.sub(r'([.]\s*){2,}', '', text)
    
    return text

# Change WORD1 WORD2 into Word1 Word2
def normalize_adjacent_uppercase_words(text):
    """Convert likely names (adjacent uppercase words) to Title Case."""
    # Pattern for two or more adjacent uppercase words, optionally with a middle initial
    pattern = r'\b([A-Z][A-Z\'\-]+)(\s+[A-Z]\.?\s+)?(\s+[A-Z][A-Z\'\-]+)\b'

    def convert_to_title(match):
        first = match.group(1).title()
        middle = match.group(2) if match.group(2) else ''
        last = match.group(3).title()
        return f"{first}{middle}{last}"

    # Change "OPERATOR" to "Operator"
    text = re.sub(r'\bOPERATOR\b', 'Operator', text)
    
    return re.sub(pattern, convert_to_title, text)

# Replace special characters, add TEXT tags
def add_text_tags(text):
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
        '\u00a9': '',  # copyright symbol
        '\u00a0': ' <NBSP> ',  # non-breaking space
        '\f': ' <PAGEBREAK> ',  # form feed / page break
        '\n': ' <TAG_2> ',  # line break
        '\t': ' <TAB> ',  # tab
    }

    # Apply all replacements
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Preserve multiple spaces for regex pattern identification
    text = re.sub(r'[ ]{2,}', ' <TAG_3> ', text)
    
    # Consolidate <TAG_3> <TAG_2> into <TAG_4>
    text = re.sub(r'<TAG_3>\s*<TAG_2>', ' <TAG_4> ', text)
    # text = re.sub(r'<TAG_2> <TAG_3>', '<TAG_4>', text) # optional

    # Ensure single space between tags
    text = re.sub(r'>\s+<', '> <', text)
    text = re.sub(r'><', '> <', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip leading/trailing whitespace
    # text = text.strip()

    return text

# Extract text with formatting from PDF, including text from images
def text_processing_pipeline(pdf_path, config_data, debug_mode=False):
    """Extract text with formatting from PDF, including text from images."""

    doc = pymupdf.open(pdf_path)
    full_text = ""

    # Get cleaning parameters
    cleaning_params = get_cleaning_parameters(config_data)
    keep_bold_tags = cleaning_params["keep_bold_tags"]
    keep_italics_tags = cleaning_params["keep_italics_tags"]
    keep_underline_tags = cleaning_params["keep_underline_tags"]
    # keep_capitalization_tags = cleaning_params["keep_capitalization_tags"]
    
    for page_num in range(doc.page_count):
        page = doc[page_num]

        # Get all text including from images
        page_text = page.get_text("text")

        # 1st step: initial cleanup and TEXT tagging
        page_text = add_text_tags(page_text)  # initial cleanup and TEXT tagging
        page_text = optimize_text_tags(page_text) # optimize text tags
        page_text = format_spaced_headers(page_text) # format spaced headers
        page_text = remove_repeating_punctuation(page_text) # remove repeating punctuation
        
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

        # 2nd step: apply tags to the page text
        for text, (is_bold, is_italic, is_underline) in formatted_blocks.items():
            # Try to find and replace exact text
            if text in page_text:
                formatted_text = text
                if keep_bold_tags and is_bold:
                    formatted_text = f" <BOLD-> {formatted_text} <-BOLD> "
                if keep_italics_tags and is_italic:
                    formatted_text = f" <ITALIC-> {formatted_text} <-ITALIC> "
                if keep_underline_tags and is_underline:
                    formatted_text = f" <UNDERLINE-> {formatted_text} <-UNDERLINE> "
                page_text = page_text.replace(text, formatted_text)

        # 3rd step: post-tagging cleanup      
        page_text = optimize_word_tags(page_text) # optimize word tags
        page_text = normalize_punctuation(page_text) # normalize punctuation
        page_text = normalize_adjacent_uppercase_words(page_text)  # bring all names into title case

        # Add page with page break to full text
        if page_text:
            full_text += page_text + "\n<PAGE_BREAK>\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1} of {pdf_path}")

    
    if debug_mode:
        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]
        if not os.path.exists(diagnostics_folder):
            os.makedirs(diagnostics_folder)
            print(f"Created directory: {diagnostics_folder}")
        file_path = os.path.join(diagnostics_folder, 'extracted_text.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Formatted text saved to {file_path}")

    doc.close()
    return full_text


### SPEAKER ATTRIBUTION EXTRACTION USING LLM ###

# Extract potential speaker names using SpaCy PERSON entities to support LLM extraction
def get_spacy_person_tags(text, nlp, config_data, debug_mode=False):
    # Process the text with SpaCy
    doc = nlp(text)

    # Extract all potential speaker names (PERSON entities)
    potential_speakers = []
    potential_attributions = []
    spacy_patterns = ""

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start_char = max(0, ent.start_char - 2)
            name = text[start_char:ent.end_char]

            potential_speakers.append({
                "name": name,
                "start": start_char,
                "end": ent.end_char
            })

    formatting_tags = ["<TAG_2>", "<TAG_4>", "<TAG_3>", "<BOLD->"]
    punctuation_marks = [":", " -", ",", ">"]
    
    for speaker in potential_speakers:
        # Get pre- and post-context for finding attribution start and end
        pre_context = text[max(0, speaker["start"] - 15):speaker["start"]]
        post_context = text[speaker["end"]:min(len(text), speaker["end"] + 15)]
        
        has_tag_before = any(tag in pre_context for tag in formatting_tags)

        if has_tag_before:
            start_idx = speaker["start"]
            for tag in formatting_tags:
                tag_pos = pre_context.rfind(tag)
                if tag_pos != -1: # if tag is found in pre-context
                    start_idx = speaker["start"] - (len(pre_context) - tag_pos)
                    break

            # Find the end index, looking for colon
            end_idx = speaker["end"]
            for mark in punctuation_marks:
                mark_pos = post_context.find(mark)
                if mark_pos != -1: # if punctuation mark is found in post-context
                    potential_end = speaker["end"] + mark_pos + 1  # Include the punctuation mark
                    if end_idx == speaker["end"] or potential_end < end_idx:  # Take the earliest punctuation
                        end_idx = potential_end
                        break

            potential_attribution = text[start_idx:end_idx]
            potential_attributions.append(potential_attribution)

    for i, attr in enumerate(potential_attributions):
        spacy_patterns += f"Pattern {i+1}: {attr}\n"
        context_start = max(0, text.find(attr) - 10)
        context_end = min(len(text), text.find(attr) + len(attr) + 10)
        context = text[context_start:context_end]
        spacy_patterns += f"Context: \"{context}\"\n\n"
        
    if debug_mode:
        print(f"Found {len(potential_speakers)} potential speakers")
        print(f"Extracted {len(potential_attributions)} potential attributions")

        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]
        if not os.path.exists(diagnostics_folder):
            os.makedirs(diagnostics_folder)
            print(f"Created directory: {diagnostics_folder}")
        file_path = os.path.join(diagnostics_folder, 'spacy_patterns.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(spacy_patterns))

    return potential_speakers, potential_attributions, spacy_patterns

# API call to extract speaker attributions
def API_call(text, spacy_patterns, config_data, debug_mode=False):
    # Construct the prompt according to the specified format
    prompt = f"""User: You are an AI assistant specialized in extracting speaker attributions from earning call transcripts.

    <goal>
    Extract all speaker attributions and call details from the transcript.    
    </goal>

    Here is the transcript to analyze:
    <transcript>
    {text}
    </transcript>

    Here are the potential speaker attributions extracted by SpaCy:
    <spacy_suggestions>
    {spacy_patterns}
    </spacy_suggestions>

    <instructions>
    Follow these step by step instructions:
    Step 1: Find the names of all call participants, including variations and misspellings. Use the spacy suggestions to help you.
    Step 2: For each participant, find their job title and companies, including variations.
    Step 3: Go through the whole text in small overlapping chunks to idenitify all variants of speaker attributions including leading and tailing tags, names, titles and companies (if available), punctuation marks.
    Step 4. For each speaker with a single attribution check again for other attributions with different formatting.
    Step 5. Identify call details like bank name, call date and reporting period.
    Step 6. Identify header and footer patterns.
    Step 7. Return results as a json object.
    </instructions>

    <formatting_tags>
    Formatting tags should be treated as text and should be added to the attribution.
    - Bold text is surrounded by <BOLD-> [speaker attribution or text or punctuation] <-BOLD>
    - Line breaks are marked as <TAG_2>
    - Paragraph breaks are marked as <TAG_4>
    - Multispaces are marked as <TAG_3>
    </formatting_tags>

    <attribution_description>
    The speaker attribution :
    1. There are usually two or more variants of the attribution formats for the same speaker. Always include all variants in the output.
    2. Attribution must start with a leading formatting tag like <TAG_2> or <BOLD-> or a combination of tags.
    3. Attribution includes one of the following:
        a) [Speaker Name and Surname] or [Name, Middle Name Initial and Surname]
        b) [Speaker Name and Surname] or [Name, Middle Name Initial and Surname], followed by a [job title], [company], or [job title and company]
    4. The job title and company, if present, can be separated from the speaker name and from each other by a punctuation mark like a colon or a dash, a formatting tag like <BOLD-> or <TAG_2> or <TAG_2>, or a combination.
    5. Attribution must end with a punctuation mark like a colon or a dash, a formatting tag like <-BOLD> or <TAG_2>, or both.
    6. Attribution never includes the text of the speech of the speaker.
    </attribution_description>
    
    <attribution_search_guidelines>
    Guidelines for searching for speaker attribution:
    - Always check the whole transcript from the beginning to the end.
    - Always look for attributions everywhere in paragraphs, both at the beginning, middle and at the end of paragraphs, particularly after sentence endings.
    - Always include all variations of attributions for each speaker even if the difference is in one character.   
    - Pay attention to the job title and company name variations.
    - Speaker attributions always alternate with the speaker's speech.
    - Speaker attributions cannot appear next to each other. If they do, they are not speaker attributions. There should always be a speech between attributions.
    - If two adjacent speeches are from the same speaker, then there must be an attribution for another speaker between them. Find it.
    - All variants of operator attibutions should always contain the word "Operator" and variations of leading and trailing formatting tags.
    - Always include ALL attributions variants even if minor.
    
    Additionally:
    - If speaker's name is separated from the job title or company by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    - If speaker's name is followed by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    </attribution_search_guidelines>
    
    <examples>    
    Examples of attributions that contain speaker name only:
    * "(attribution starts) <TAG_2> Name Surname (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) Name Surname: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Name Surname: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_4> Name Surname: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <BOLD-> Name Surname <-BOLD> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <BOLD-> Name Surname: <-BOLD> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Name Surname <TAG_2> (attribution ends) Thank you. I'd like to present our quarterly results."
    If the same speaker has varios attributions, then all attributions should be included into the output.

    Examples of attributions that contain speaker name with job title, company, or job title and company:
    * "(attribution starts) <BOLD-> Name Surname - Company - Job Title <-BOLD> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <BOLD-> Name Surname <-BOLD> - CEO, TechCorp: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> <BOLD-> Name Surname, CFO: <-BOLD> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Name Surname • Senior VP: <TAG_2> (attribution ends) Thank you. I'd like to present our quarterly results."

    Example of complex attributions with multiple tags and punctuation:
    * "(attribution starts) Jamie Dimon <TAG_3> <TAG_2> Chairman & Chief Executive Officer, JPMorgan Chase & Co. <TAG_3><TAG_2> (attribution ends)"
    
    There are often multiple attribution formatting variations of the same speaker, for example:
    * "(attribution starts) <TAG_2> Name Surname: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <BOLD-> Name Surname: <-BOLD> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <BOLD-> Name Surname <-BOLD> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Name Surname <TAG_2> (attribution ends) Thank you. I'd like to present our quarterly results."
    In such cases all variants should be included in the output.

    There are often multiple variations of the Operator attributions, for example:
    * "<TAG_2> OPERATOR:"
    * "<TAG_2> OPERATOR <TAG_2>"
    * "<BOLD-> OPERATOR: <-BOLD>"
    In such cases all variants should be included in the output.
    
    Spelling and Name Variations:
    * "(attribution begins here) <TAG_4> Michael J. Thompson: <TAG_2> (attribution ends here) Thank you. I'd like to present our quarterly results."
    * "(attribution begins here) <TAG_4> Mike Thompson: <TAG_2> (attribution ends here) Thank you. I'd like to present our quarterly results."
    In such cases all variants should be included in the output.
    
    Job Title and Company Separation:
    * "(attribution begins here) <TAG_4> Name Surname: <TAG_2> (attribution ends here) Thank you. I am (Job Title) and I'd like present our (Company Name) quarterly results."
    In such cases only "<TAG_4> Name Surname: <TAG_2>" should be considered attribution because there is a text between the name and job title.
    
    Company Name Variations:
    * "(attribution begins here) <TAG_2> Name Surname (Full company name) <TAG_2> (attribution ends here) Thank you. I'd like to present our quarterly results."
    * "(attribution begins here) <TAG_2> Name Surname (Company name abbreviation) <TAG_2> (attribution ends here) Thank you. I'd like to present our quarterly results."
    In such cases all variants should be included in the output.
    </examples>

    REMEMBER: The output should include ALL variants of the attributions for every speaker.
    
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
        "speaker_attributions": ["<TAG_2> JOHN DOE:", "<BOLD-> John Doe - CEO - Example Bank <-BOLD>", "<TAG_2> John Doe - EB - CEO <TAG_2>"]
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
            model="claude-3-7-sonnet-20250219",  # claude-3-opus-20240229 claude-3-7-sonnet-20250219
            max_tokens=4096,
            system="You are an expert in finding speaker attributions in the earnings call transcripts.",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=0.8,
            # top_k=20
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

        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]  # always use this folder for diagnostics
        
        # Always save the raw response for debugging
        if debug_mode:
            raw_response_path = os.path.join(diagnostics_folder, 'api_response_raw.txt')
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
                        cleaned_path = os.path.join(diagnostics_folder, 'api_response_cleaned.txt')
                        with open(cleaned_path, 'w', encoding='utf-8') as f:
                            f.write(clean_response)
                        print(f"Cleaned API response saved to {cleaned_path}")
            
            json_response = json.loads(clean_response)
            parsed_successfully = True
            
            if debug_mode:
                parsed_path = os.path.join(diagnostics_folder, 'api_response_parsed.json')
                with open(parsed_path, 'w', encoding='utf-8') as f:
                    json.dump(json_response, f, indent=2)
                print(f"Parsed JSON saved to {parsed_path}")
                
        except json.JSONDecodeError as e:
            parsed_successfully = False
            json_response = None
            
            if debug_mode:
                print(f"JSON parsing error: {str(e)}")
                error_path = os.path.join(diagnostics_folder, 'api_response_error.txt')
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


### TEXT PARSING AND CLEANING ###

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

def get_utterances(text, api_response, config_data, debug_mode=False):
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
        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]
        file_path = os.path.join(diagnostics_folder, 'all_attributions.txt')
        
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

    # Count for each cleaning type specificed below
    debug_counts = {
        'angle_brackets': 0,
        'parentheses': 0,
        # 'double_quotes': 0,
        # 'single_quotes': 0,
        'double_slashes': 0,
        'backslashes': 0
    }

    # This should clear the utterance from all tags
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

    # Remove header pattern from each utterance's text (not removing the whole utterance)
    header_pattern = api_response.get("header_pattern", None) if isinstance(api_response, dict) else None
    if header_pattern:  # Only process if header_pattern exists
        cleaned_header_pattern = re.sub(r'<(?:TAG_2|TAG_3|TAG_4|BOLD-|-BOLD)>', '', header_pattern)
        for utterance in utterances:
            # Replace the matching pattern with an empty string, keeping the rest of the text
            cleaned_text = re.sub(cleaned_header_pattern, '', utterance['utterance'], flags=re.IGNORECASE)
            # Create a new utterance with the cleaned text
            cleaned_utterance = utterance.copy()  # Copy to preserve other fields
            cleaned_utterance['utterance'] = cleaned_text
            cleaned_utterances.append(cleaned_utterance)

    # Remove footer from all utterances
    footer_pattern = api_response.get("footer_pattern", None) if isinstance(api_response, dict) else None
    if footer_pattern:  # Only process if footer_pattern exists
        cleaned_footer_pattern = re.sub(r'<(?:TAG_2|TAG_3|TAG_4|BOLD-|-BOLD)>', '', footer_pattern)
        for utterance in utterances:
            # Replace the matching pattern with an empty string, keeping the rest of the text
            cleaned_text = re.sub(cleaned_footer_pattern, '', utterance['utterance'], flags=re.IGNORECASE)
            # Create a new utterance with the cleaned text
            cleaned_utterance = utterance.copy()  # Copy to preserve other fields
            cleaned_utterance['utterance'] = cleaned_text
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
    full_text = text_processing_pipeline(file_path, config_data, debug_mode)
    
    # Make API call
    print("Sending text to API for analysis...")
    # nlp = spacy.load("en_core_web_trf")  # transformer model for better accuracy
    nlp = spacy.load("en_core_web_sm")
    potential_speakers, potential_attributions, spacy_patterns = get_spacy_person_tags(full_text, nlp, config_data, debug_mode)
    api_response = API_call(full_text, spacy_patterns, config_data, debug_mode)
    
    # Check for errors in the API call result
    if "error" in api_response:
        print(f"Error: {api_response['error']}")
        return
    
    # Get utterances
    utterances = get_utterances(full_text, api_response, config_data, debug_mode)
    
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