### IMPORTS ###

import spacy
from spacy.matcher import Matcher
from spacy.symbols import ORTH
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

### 1. CONFIGURATION ###

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


### 2. PDF IMPORT AND TEXT PRE-PROCESSING ###

# Change case of adjacent uppercase words
def normalize_adjacent_uppercase_words(text):
    """Convert likely names (adjacent uppercase words) to Title Case."""
    # Change "OPERATOR" to "Operator"
    text = re.sub(r'\bOPERATOR\b', 'Operator', text)

    # Pattern for two or more adjacent uppercase words, optionally with a middle initial
    pattern = r'\b([A-Z][A-Z\'\-]+)(\s+[A-Z]\.?\s+)?(\s+[A-Z][A-Z\'\-]+)\b' # original
    #pattern = r'\b([A-Z][A-Z\'\-]+)(?:\s+([A-Z]\.?)?\s+)?([A-Z][A-Z\'\-]+)\b'  # supposed to tackle apostrophes

    def convert_to_title(match):
        first = match.group(1).title()
        middle = match.group(2) if match.group(2) else ''
        last = match.group(3).title()
        return f"{first}{middle}{last}"
    
    return re.sub(pattern, convert_to_title, text)

# Clean special characters
def clean_special_characters(text):
    """Clean text by handling encoding issues and removing problematic characters."""
    # Handle encoding issues with round-trip conversion
    try:
        # Convert to bytes and back with explicit error handling
        text_bytes = text.encode('utf-8', errors='ignore')
        text = text_bytes.decode('utf-8', errors='ignore')
    except (UnicodeError, AttributeError):
        pass

    # Dictionary common substitutions
    replacements = {
        '\u00a0': ' ',  # non-breaking space
        '\ufffd': '',   # replacement character
        '\u201a': ',',  # single low-9 quotation mark
        '\u201b': ',',  # single high-reversed-9 quotation mark
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2013': '-',  # en-dash
        '\u2014': '--', # em-dash
        '\u2015': '-',  # horizontal bar
        '\u2016': '||', # double vertical bar
        '\u2017': '_',  # underscore
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u2026': '.', # ellipsis
        
        '�': '',       # unknown character
        '\u2022': '•',  # bullet point
        '\u00a9': '',   # copyright symbol

        '\s+': ' ',     # replace multiple spaces with a single space
    }

    # Apply all replacements
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove leading and trailing whitespace
    text = text.strip()

    return text

# Remove lines that contain only numbers or only capital letters
def remove_lines_with_only_numbers_or_capital_letters(text):
    """Remove lines that contain only numbers or only capital letters."""
    return re.sub(r'^[0-9]+$', '', text)

# Remove page numbers
def clean_page_numbers(text):
    """Remove different formats of page numbers from text."""

    # Define patterns for different page number formats
    patterns = [
        # r'\b\d+\b',                           # Simple numbers: 1, 2, 3
        r'\bpage\s+\d+\b',                    # "page n": page 1, page 2
        # r'\bp\.\s*\d+\b',                     # "p.n": p.1, p. 2
        # r'\b[ivxlcdm]+\b',                    # Roman numerals: i, ii, iii, iv
        # r'\b\d+\s*-\s*\d+\b',                 # Page ranges: 1-5, 10-15
        # r'\b\d+\s*-\s*\d+\b',                 # Section numbering: 1-1, 1-2
        # r'\b0*\d+\b',                         # Leading zeros: 001, 002
        r'\bPage\s+\d+\s+of\s+\d+\b',         # "Page n of m": Page 1 of 10
        # r'\b(?:Page|Pg\.?|P\.)\s+\d+\s+of\s+\d+\b'  # Variations: Pg 1 of 10, P. 1 of 10
    ]

    # Combine all patterns with OR operator and make case-insensitive
    combined_pattern = '|'.join(patterns)

    # Remove matched patterns from text
    text = re.sub(combined_pattern, '', text, flags=re.IGNORECASE)

    # Remove extra whitespace that might be left behind
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

# Main text processing pipeline
def text_processing_pipeline(pdf_path, config_data=None, debug_mode=False):
    """Extract text with formatting from PDF, including text from images."""

    doc = pymupdf.open(pdf_path)
    interim_text = ""
    document_text = ""
    page_text = ""

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        interim_text += page_text + "\n<PAGE_BREAK>\n"
        
        page_text = clean_special_characters(page_text)
        
        lines = page_text.split('\n')
        cleaned_lines = []
        cleaned_lines_with_tags = []
        
        for line in lines:
            cleaned_line = clean_page_numbers(line)
            cleaned_line = re.sub(r'^[\s!"#$%&\'*+,-./:;<=>?@[\\\]^_`{|}~]+', '', cleaned_line)  # (remove leading punctuation and spaces)
            cleaned_line = normalize_adjacent_uppercase_words(cleaned_line) # (change WORD1 WORD2 into Word1 Word2)
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip() # (normalize spaces)
            
            # remove lines that contain only one symbol after removing extra spaces
            if len(cleaned_line.strip()) > 1: #or \
                #not cleaned_line.strip().isdigit() or \
                #not (cleaned_line.strip().isupper() or \
                #not cleaned_line.strip().isalpha()) or \
                #not "Page" in cleaned_line:

                cleaned_lines.append(cleaned_line)
                cleaned_lines_with_tags.append(cleaned_line)  # + " <TAG_2>"
        
            # Remove lines that contain tags only
            if re.match(r'^\s*<[^>]+>\s*$', cleaned_line):
                continue
            
        # Cleaning
        # page_text = normalize_adjacent_uppercase_words(page_text)  # bring all names into title case

        # Collate lines and add to full text
        page_text = ' <TAG_2> '.join(cleaned_lines_with_tags)
        document_text += page_text + "\n<PAGE_BREAK> "

    
    # Get cleaning parameters
    #cleaning_params = get_cleaning_parameters(config_data)
    #keep_bold_tags = cleaning_params["keep_bold_tags"]
    #keep_italics_tags = cleaning_params["keep_italics_tags"]
    #keep_underline_tags = cleaning_params["keep_underline_tags"]
    
    if debug_mode:
        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]
        if not os.path.exists(diagnostics_folder):
            os.makedirs(diagnostics_folder)
            print(f"Created directory: {diagnostics_folder}")
            
        file_path = os.path.join(diagnostics_folder, 'interim_text.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(interim_text)
            
        file_path = os.path.join(diagnostics_folder, 'document_text.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(document_text)

        print(f"Formatted text saved to {file_path}")

    doc.close()
    return document_text


### 3. EXTRACTING POTENTIAL SPEAKER ATTRIBUTIONS USING SPACY NER AND MATCHER ###

# Main function to extract speaker attributions
def extract_speaker_attributions(text, nlp, config_data=None, debug_mode=False):
    """
    Extract potential speaker attributions using both SpaCy NER and custom matcher.
    Returns formatted string to pass to LLM.
    """
    # Lists to collect attributions from different methods
    attributions_ner = []
    attributions_matcher = []

    # Add custom tags to SpaCy tokenizer
    custom_tags = ["<TAG_2>", "<TAG_3>", "<TAG_4>", "<BOLD->", "<-BOLD>"]
    for tag in custom_tags:
        special_case = [{ORTH: tag}]
        nlp.tokenizer.add_special_case(tag, special_case)

    # PART 1: Extract patterns using SpaCy NER
    # Remove XML/HTML tags for better NER processing
    untagged_text = re.sub(r'<[^>]+>', ' ', text)
    untagged_text = re.sub(r'\s+', ' ', untagged_text)
    doc = nlp(untagged_text)

    # Get all PERSON entities
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            attributions_ner.append({
                "name": ent.text,
                "source": "NER"
            })

    # PART 2: Extract patterns using SpaCy matcher
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Add various patterns to the matcher
    # Pattern 1: Name and punctuation
    matcher.add("NAME_opSPACE_PUNCT", [
        [
            {"POS": "PROPN"},
            {"POS": "PROPN", "OP": "+"},
            {"IS_SPACE": True, "OP": "?"},
            {"IS_PUNCT": True, "TEXT": {"NOT_IN": [",", "."]}}
        ]
    ])

    # Pattern 2: Name and tag
    matcher.add("NAME_opSPACE_TAG", [
        [
            {"POS": "PROPN"},
            {"POS": "PROPN", "OP": "+"},
            {"IS_SPACE": True, "OP": "?"},
            {"TEXT": "<TAG_2>"}
        ]
    ])

    # Pattern 3: Name, punctuation, and tag
    matcher.add("NAME_opSPACE_PUNCT_opSPACE_TAG", [
        [
            {"POS": "PROPN"},
            {"POS": "PROPN", "OP": "+"},
            {"IS_SPACE": True, "OP": "*"},
            {"IS_PUNCT": True},
            {"IS_SPACE": True, "OP": "*"},
            {"TEXT": {"REGEX": "<[^>]+>"}}
        ]
    ])

    # Pattern 4: Newline name and tag
    matcher.add("NEWLINE_NAME_opSPACE_TAG", [
        [
            {"IS_SENT_START": True},
            {"POS": "PROPN"},
            {"POS": "PROPN", "OP": "+"},
            {"IS_SPACE": True, "OP": "*"},
            {"TEXT": {"REGEX": "<[^>]+>"}}
        ]
    ])

    # Pattern 5: Name with non-name text
    matcher.add("NEWLINE_NAME_op-von_SURNAME_opSPACE_TAG", [
        [
            {"IS_SENT_START": True},
            {"POS": "PROPN", "OP": "+"},
            {"POS": {"NOT_IN": ["PROPN"]}, "LENGTH": {"<=": 3}, "OP": "{,3}"},
            {"POS": "PROPN", "OP": "+"},
            {"IS_SPACE": True, "OP": "*"},
            {"TEXT": {"REGEX": "<[^>]+>"}}
        ]
    ])

    # Apply matcher
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_text = span.text

        if span.text:
            # Create a pattern dictionary
            pattern = {
                "name": matched_text,
                "source": "MATCHER"
            }

            # Check if pattern is already in the ner list or matcher list
            if not any(p["name"] == matched_text for p in attributions_ner):
                if not any(p["name"] == matched_text for p in attributions_matcher):
                    attributions_matcher.append(pattern)
            
            # Check if this pattern is already in the matcher list
            #if not any(p["name"] == matched_text for p in attributions_matcher):
            #    attributions_matcher.append(pattern)

    # Combine attributions and remove duplicates
    all_attributions = attributions_ner + attributions_matcher
    unique_attributions = remove_duplicates_by_name(all_attributions)

    # Format the unique attributions for LLM
    formatted_output = format_attribution_list(unique_attributions)

    # Debug output if requested
    if debug_mode and config_data:
        print(f"{len(attributions_ner)} attributions found using NER")
        print(f"{len(attributions_matcher)} attributions found using MATCHER")
        print(f"{len(unique_attributions)} total unique attributions passed to LLM")

        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]
        if not os.path.exists(diagnostics_folder):
            os.makedirs(diagnostics_folder)
            print(f"Created directory: {diagnostics_folder}")

        file_path = os.path.join(diagnostics_folder, 'attributions_from_spacy.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)

    return formatted_output

# Remove duplicate attributions based on name field.
def remove_duplicates_by_name(attributions):
    """
    Remove duplicate attributions based on name field.
    Returns a list of unique attributions.
    """
    seen_names = set()
    unique_attributions = []

    for attribution in attributions:
        name = attribution["name"]
        if name not in seen_names:
            seen_names.add(name)
            unique_attributions.append(attribution)

    return unique_attributions

# Format a list of attributions into a string for LLM.
def format_attribution_list(attributions):
    """
    Format a list of attributions into a string for LLM.
    """
    formatted_output = ""
    for i, attribution in enumerate(attributions):
        formatted_output += f"Pattern {i+1}: {attribution['name']}\n"

    return formatted_output

# Extract potential attributions for "Operator" with preceeding formatting tags
def get_operator_attributions(text):
    operator_pattern = r'(<[^>]+>)+\s*Operator'
    operator_attributions = ""

    for match in re.finditer(operator_pattern, text):
        operator_with_tags = match.group(0)
        operator_attributions += f"Operator with tags: {operator_with_tags}\n"

    # remove duplicates
    operator_attributions = list(set(operator_attributions))

    return operator_attributions


### 4. ATTRIBUTION EXTRACTION USING LLM ###

# API call to extract speaker attributions
def API_call(text, spacy_attributions, operator_attributions, config_data=None, debug_mode=False):
    # Construct the prompt according to the specified format
    prompt = f"""User: You are an AI assistant specialized in extracting speaker attributions from a bank's earnings call transcripts.

    <goal>
    Extract all speaker attributions and call details from the transcript.    
    </goal>

    This transcript is comprised of:
    - speaker attributions: attributions that identify the speaker of the speech
    - speaker speech: speech delivered by the speaker
    - other text: other text that is not speech by any participant
    <transcript>
    {text}
    </transcript>
    
    Here is a list of potential speaker attributions extracted by SpaCy:
    <spacy_suggestions>
    {spacy_attributions}
    </spacy_suggestions>
    Some of these attributions may be incorrect, some may be correct, some may be missing.

    Here are the potential operator attributions:
    <operator_suggestions>
    {operator_attributions}
    </operator_suggestions>

    <instructions>
    Follow these step by step instructions:
    Step 1: Find the names of all call participants, including variations and misspellings. Use the SpaCy suggestions to help you.
    Step 2: For each participant, find all variations of their job title and companies.
    Step 3: Go through the whole text in small overlapping chunks to idenitify all variants of speaker attributions including leading and tailing tags, names, titles and companies (if available), punctuation marks.
    Step 4. For each speaker with a single attribution check again for other attributions with different formatting.
    Step 5. Make sure that for each speaker you identified attributions with different formatting, or a different spelling of the name, job title and company for each speaker.
    Step 6. Make sure that you identified attributions for all speakers.
    Step 6. Identify call bank name, call date and reporting period.
    Step 7. Identify the first 3 non-speaker words of the last utterance in the transcript, if any.
    Step 8. Identify header that repeat throughout the transcript, if any, and remove all tags and "\n" from it for use in the json.
    Step 9. Identify footer that repeat throughout the transcript, if any, and remove all tags and "\n" from it for use in the json.
    Step 10. Return results as a json object.
    </instructions>

    <formatting_tags>
    Tags like <TAG_2>, <PAGE_BREAK> should be added to the attribution if they are present and help to identify the speaker.
    </formatting_tags>

    <attribution_description>
    The speaker attribution :
    1. There may be two, three or more variants of the attribution formats for the same speaker. Always include all variants in the output.
    2. Attribution should start with a tag like <TAG_2> or a speaker's name.
    3. Attribution includes one of the following:
        a) [Speaker Name and Surname] or [Name, Middle Name Initial and Surname]
        b) [Speaker Name and Surname] or [Name, Middle Name Initial and Surname], followed by a [job title], [company], or [job title and company]
    4. The job title and company, if present, can be separated from the speaker name and from each other by a punctuation mark like a colon or a dash, a formatting tag like <TAG_2>, or a combination.
    5. Attribution must end with a punctuation mark like a colon or a dash, a formatting tag like <TAG_2>, or both.
    6. If a text that fits to the above description is found in the speech of some speaker, then it is not an attribution.
    7. If attribution is followed by a speaker's job title or company, then attribution should be updated to include the job title or company.
    8. Attribution never includes the text of the speech of the speaker.
    9. Attribution never appears in the speech.
    </attribution_description>
    
    <attribution_search_guidelines>
    Guidelines for searching for speaker attribution:
    - Always check the whole transcript from the beginning to the end.
    - Always look for attributions everywhere in paragraphs, both at the beginning, middle and at the end of paragraphs, particularly after sentence endings.
    - Always include all variations of attributions for each speaker even if the difference is in one character.   
    - Pay attention to the job title and company name variations.
    - Speaker attributions always alternate with the speaker's speech.
    - Speaker attributions cannot appear one after another without any speech between them. If they do, they are not speaker attributions. There should always be a speech between attributions.
    - If two adjacent speeches are from the same speaker, then there must be an attribution for another speaker between them. Find it.
    - All variants of operator attibutions should always contain the word "Operator" and variations of leading and trailing formatting tags.
    - Always include ALL attributions variants even if differences are minimal.
    
    Additionally:
    - If speaker's name is separated from the job title or company by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    - If speaker's name is followed by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    </attribution_search_guidelines>
    
    <examples>    
    Often, there are multiple attribution formats for the same speaker, for example:
    * "(attribution starts) Full name <TAG_2> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) Full name: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) Full name: <TAG_2> (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) Full name - (attribution ends) Thank you. I'd like to present our quarterly results."
    If the same speaker has various attributions, then all attributions should be included into the output.

    Examples of attributions that contain speaker name with job title, company, or job title and company:
    * "(attribution starts) <TAG_2> Full name - Company - Job Title (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Full name - Company - Job Title (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Full name - CEO, TechCorp: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Full name, CFO: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Full name • Senior VP: (attribution ends) Thank you. I'd like to present our quarterly results."
    * "(attribution starts) <TAG_2> Full name • Senior VP: (attribution ends) Thank you. I'd like to present our quarterly results."

    Example of complex attributions with multiple tags and punctuation:
    * "(attribution starts) Jamie Dimon <TAG_3> <TAG_2> Chairman & Chief Executive Officer, JPMorgan Chase & Co. <TAG_3><TAG_2> (attribution ends)"
    
    There are often multiple variations of the Operator attributions, for example:
    * "OPERATOR:"
    * "OPERATOR <TAG_2>"
    In such cases all variants should be included in the output.
    
    Spelling and Name Variations:
    * "(attribution begins here) Michael J. Thompson: (attribution ends here)"
    * "(attribution begins here) Mike Thompson: (attribution ends here)"
    In such cases all variants should be included in the output.
       
    Company Name Variations:
    * "(attribution begins here) Full name - (Bank of America Securities) <TAG_2> (attribution ends here) Thank you. I'd like to present our quarterly results."
    * "(attribution begins here) Full name - (BofA) <TAG_2> (attribution ends here) Thank you. I'd like to present our quarterly results."
    * Transcript may contain mukltiple variants of the company name spelling, such as: "Company Name", "Company Name Bank", "Company Name Securities", "Company Name abbreviation"
    In such cases all variants should be included in the output.

    Job Title Variations:
    * "(attribution begins here) Full name - Company Name - Group Head, Business Banking (attribution ends here) Thank you. I'd like to present our quarterly results."
    * "(attribution begins here) Full name - Company Name - Group Head, Country Business Banking (attribution ends here) Thank you. I'd like to present our quarterly results."
    In such cases all variants should be included in the output.

    Example of creating attributions for speakers with variations in name, job title and company:
    * Attribution found in the transcript: "<TAG_2> Full name - Company Name - Job Title <TAG_2>"
    * Company name variants found in the text: "Company Name Securities", "Company Name abbreviation"
    * In such case, create additional attributions and add them to the output for given speaker:
        * "<TAG_2> Full name - Company Name Securities - Job Title <TAG_2>"
        * "<TAG_2> Full name - Company Name abbreviation - Job Title <TAG_2>"        
    Same method should be used for job title variations.
         
    Job Title and Company Separation:
    * "(attribution begins here) Full name: <TAG_2> (attribution ends here) Thank you. I am (Job Title) and I'd like present our (Company Name) quarterly results."
    In such cases only "Full name: <TAG_2>" should be considered attribution because there is a text between the name and job title.

    Non-speaker words in the last utterance
    * In this utterance the first 3 non-speaker words are "Certain", "statements" and "in": "This does conclude First Quarter 2023 Earnings Review Call. You may now disconnect at any time. Certain statements in this document are \"forward looking statements\"",
    * In this utterance the first 3 non-speaker words are "Disclaimer", "This" and "transcript": "Thank you very much for your questions. For any follow-up questions, please reach out to Investor Relations. Disclaimer This transcript contains forward-looking statements."
    * In this utterance there are no non-speaker words: "This does conclude First Quarter 2023 Earnings Review Call. You may now disconnect at any time."
    * Other examples of words that may indicate the beginning of a non-speaker text: "disclaimer", "cautionary statement", "forward-looking statements".
    
    Examples of names that are part of speech and not attributions:
    * This is not an attribution, this is a speech by some other speaker: "Thank you, Full name <TAG_2>"
    </examples>

    REMEMBER: The output should include ALL variants of the attributions for every speaker.
    
    Here's an example of the expected JSON structure (with generic placeholders):
    <jsonexample>
    {{
    "hosting_bank_name": "Example Bank",
    "call_date": {{
        "description": "The date of the earnings call in YYYY-MM-DD format",
        "type": "string",
        "format": "date",
        "pattern": "^\\d{4}-\\d{2}-\\d{2}$" 
    }},
    "reporting_period": {{
        "description": "The reporting period of the earnings call in Q-YYYY format",
        "type": "string",
        "pattern": "^Q[1-4]-\\d{4}$"
    }},
    "header_pattern": "HEADER_PATTERN_WITHOUT_TAGS_AND_NEWLINES",
    "footer_pattern": "FOOTER_PATTERN_WITHOUT_TAGS_AND_NEWLINES",
    "first_3_non_speaker_words_in_last_utterance": "FIRST_3_NON_SPEAKER_WORDS_OF_LAST_UTTERANCE",
    "participants": [
        {{
        "speaker_name_variants": ["John Doe", "Jon Doe", "John Do"],
        "speaker_title_variants": ["Chief Executive Officer", "CEO"],
        "speaker_company_variants": ["Example Bank", "Example Bank Inc.", "EB"],
        "speaker_attributions": ["JOHN DOE: <TAG_2>", "John Doe - CEO - Example Bank", "John Doe - EB - CEO <TAG_2>"]
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
    api_key_name, model_name, input_cost_per_million, output_cost_per_million = get_api_setup(config_data)
    api_key = os.getenv(api_key_name)
    client = anthropic.Anthropic(api_key=api_key)

    if debug_mode:
        print(f"Running API call using {model_name}...")

    try:
        message = client.messages.create(
            model=model_name,  # claude-3-opus-20240229 claude-3-7-sonnet-20250219
            max_tokens=4096,
            system="You are an expert in finding speaker attributions in the earnings call transcripts.",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0.7,
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


### 5. API RESPONSE PROCESSING ###

# De-duplicate leading tags in attributions (from <TAG_1> <TAG_1> to <TAG_1>)
def remove_leading_duplicate_tags(attribution):
    # Pattern to match any opening tag
    tag_pattern = r'<[A-Z_0-9]+>'

    # Find all tags at the beginning of the string
    match = re.match(r'^(\s*(' + tag_pattern + r'\s*)+)', attribution)

    if match:
        # Get the entire matched section (all leading tags)
        leading_tags_section = match.group(1)

        # Find all individual tags in this section
        tags = re.findall(tag_pattern, leading_tags_section)

        # Remove duplicates while preserving order
        unique_tags = []
        for tag in tags:
            if not unique_tags or tag != unique_tags[-1]:
                unique_tags.append(tag)

        # Create the new leading section with single space between tags
        new_leading_section = ' '.join(unique_tags) + ' '

        # Replace the original leading section with the deduplicated one
        result = attribution.replace(leading_tags_section, new_leading_section, 1)
        return result

    return attribution

# Get utterances from API response
def get_utterances(text, api_response, config_data=None, debug_mode=False):
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
        print(f"{len(all_attributions)} speaker attributions found in LLM response")
        
        # Create a file to save all attributions for debugging
        diagnostics_folder = get_test_mode_info(config_data)["diagnostics_folder"]
        file_path = os.path.join(diagnostics_folder, 'attributions_from_LLM.txt')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Total attributions found: {len(all_attributions)}\n\n")
            for i, attr in enumerate(all_attributions):
                f.write(f"Attribution {i+1}:\n")
                f.write(f"  Speaker: {attr['speaker_name']}\n")
                f.write(f"  Text: {attr['attribution']}\n\n")
        
    if not all_attributions:
        return []

    # De-duplicate leading tags in attributions
    all_attributions = [{
        "speaker_name": attr["speaker_name"],
        "attribution": remove_leading_duplicate_tags(attr["attribution"])
    } for attr in all_attributions]
    
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
        print(f"{len(utterances)} utterances created incl. empty")
    
    return utterances

# Clean utterances
def clean_utterances(utterances: list, api_response: dict, debug_mode=False) -> list:
    cleaned_utterances = []

    # Remove empty utterances
    utterances = [utterance for utterance in utterances if utterance['utterance'].strip()]

    # Remove utterances that only contain capital letters
    utterances = [utterance for utterance in utterances if not utterance['utterance'].isupper()]
    
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

    if debug_mode:
        print(f"{len(utterances)} utterances left after cleaning")
        
    return cleaned_utterances

# Clean last utterance from non-speaker text
def clean_last_utterance(cleaned_utterances: list, api_response: dict) -> list:
    # Check if cleaned_utterances is empty
    if not cleaned_utterances:
        print("No utterances to clean!")
        return cleaned_utterances

    # Get non-speaker text
    parsed_json = api_response.get("parsed_json", {})
    non_speaker_text = parsed_json.get("first_3_non_speaker_words_in_last_utterance")

    if not non_speaker_text:
        print("No non-speaker text to clean!")
        return cleaned_utterances
    
    # clean last_utterance_tokens
    #last_utterance_tokens_cleaned = last_utterance_tokens.replace("<TAG_2>", "").strip()
    #last_utterance_tokens_cleaned = last_utterance_tokens_cleaned.split("\n")[0].strip()   

    # Clean up the non-speaker text and last utterance for comparison
    non_speaker_text = ' '.join(non_speaker_text.split()).lower()
    last_utterance = cleaned_utterances[-1]['utterance']
    last_utterance_lower = last_utterance.lower()
    
    # Find the position of non-speaker text
    start_index = last_utterance_lower.find(non_speaker_text)
    
    if start_index != -1:
        # Keep only the part of last_utterance that comes before the match
        cleaned_text = last_utterance[:start_index].strip()
        # Update the last utterance in the list
        cleaned_utterances[-1]['utterance'] = cleaned_text
    else:
        print("No non-speaker text found")
    
    return cleaned_utterances

# Clean header from each utterance
def clean_header(cleaned_utterances: list, api_response: dict, debug_mode=False) -> list:
    parsed_json = api_response.get("parsed_json", {})
    header_pattern = parsed_json.get("header_pattern")

    if not header_pattern:
        print("No header pattern to clean!")
        return cleaned_utterances

    # Clean header pattern
    header_pattern_normalized = ' '.join(header_pattern.split())

    # Remove header text
    for utterance in cleaned_utterances:
        utterance_text = utterance['utterance']

        # Simply replace the header pattern if found
        if header_pattern_normalized in utterance_text:
            utterance['utterance'] = utterance_text.replace(header_pattern_normalized, "").strip()
        else:
            # Try case-insensitive match as fallback
            pattern_lower = header_pattern_normalized.lower()
            text_lower = ' '.join(utterance_text.split()).lower()

            if pattern_lower in text_lower:
                # Find position in lowercase version
                start_pos = text_lower.find(pattern_lower)
                end_pos = start_pos + len(pattern_lower)

                # Map the positions to original text
                normalized_pos = 0
                start_original = 0
                end_original = len(utterance_text)

                # Find start and end in original text
                for i, char in enumerate(utterance_text):
                    if normalized_pos == start_pos:
                        start_original = i
                    if normalized_pos == end_pos:
                        end_original = i
                        break

                    if not char.isspace() or (i > 0 and utterance_text[i - 1].isspace() and not char.isspace()):
                        normalized_pos += 1

                # Remove header from original text
                utterance['utterance'] = (
                    utterance_text[:start_original].strip() +
                    utterance_text[end_original:].strip()
                ).strip()

    return cleaned_utterances

# Reconcile repeated speaker attributions in adjacent utterances
def reconcile_repeated_speaker_attributions(cleaned_utterances: list) -> list:
    """
    Reconcile repeated speaker attributions in adjacent utterances.
    
    If the same speaker appears in consecutive utterances, combine their utterances
    and remove the duplicate speaker entry.
    
    Args:
        cleaned_utterances (list): List of utterance dictionaries with 'speaker' and 'utterance' keys
        
    Returns:
        list: Reconciled list of utterances with duplicates combined
    """
    if not cleaned_utterances or len(cleaned_utterances) < 2:
        return cleaned_utterances
    
    # Create a new list to store reconciled utterances
    reconciled_utterances = [cleaned_utterances[0]]
    
    # Start from the second utterance
    i = 1
    while i < len(cleaned_utterances):
        current_utterance = cleaned_utterances[i]
        previous_utterance = reconciled_utterances[-1]
        
        # Check if current speaker is the same as previous speaker
        if current_utterance['speaker'] == previous_utterance['speaker']:
            # Combine the utterances
            combined_text = previous_utterance['utterance'] + " " + current_utterance['utterance']
            previous_utterance['utterance'] = combined_text.strip()
            # Skip adding this utterance since we combined it
        else:
            # Different speaker, add to reconciled list
            reconciled_utterances.append(current_utterance)
        
        # Move to next utterance
        i += 1
    
    return reconciled_utterances

# Remove speaker job title and company from the utterance - TODO
def remove_speaker_job_title_and_company_from_utterance(cleaned_utterances: list) -> list:
    """
    Remove speaker job title and company from the utterance.
    
    Args:
        cleaned_utterances (list): List of utterance dictionaries with 'speaker' and 'utterance' keys
        
    Returns:
        list: List of utterances with speaker job title and company removed
    """
    for utterance in cleaned_utterances:
        utterance['utterance'] = utterance['utterance'].replace(utterance['speaker_job_title'], "").replace(utterance['speaker_company'], "")

    return cleaned_utterances

# Save final json
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
        print(f"{len(cleaned_utterances)} utterances in the final JSON")
    
    # Save the final JSON to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2)
    print(f"Final JSON saved to {output_path}")
    
    return final_json


### 6. MAIN FUNCTION ###

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
    

    # Extract potential speaker attributions
    print("Extracting speaker attributions using SpaCy...")
    # nlp = spacy.load("en_core_web_trf")  # optional: transformer model for better accuracy
    nlp = spacy.load("en_core_web_md")
    spacy_attributions = extract_speaker_attributions(full_text, nlp, config_data, debug_mode)
    operator_attributions = get_operator_attributions(full_text)

    # Make API call
    print("Sending text to LLM...")
    api_response = API_call(full_text, spacy_attributions, operator_attributions, config_data, debug_mode)
    
    # Check for errors in the API call result
    if "error" in api_response:
        print(f"Error: {api_response['error']}")
        return
    
    # Get utterances
    print("Processing utterances from API response...")
    utterances = get_utterances(full_text, api_response, config_data, debug_mode)
    
    # Clean utterances
    cleaned_utterances = clean_utterances(utterances, api_response, debug_mode)
    cleaned_utterances = clean_last_utterance(cleaned_utterances, api_response)
    cleaned_utterances = clean_header(cleaned_utterances, api_response, debug_mode)
    cleaned_utterances = reconcile_repeated_speaker_attributions(cleaned_utterances)
    # cleaned_utterances = remove_speaker_job_title_and_company_from_utterance(cleaned_utterances)
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