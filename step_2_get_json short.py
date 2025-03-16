import json
import re
import copy
import uuid
import os
import anthropic
import time
from typing import Dict, Optional


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
    
def load_metadata(metadata_file: str, debug_mode: bool) -> Dict:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_file (str): Path to the metadata JSON file
        
    Returns:
        dict: Parsed metadata
        
    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if debug_mode:
        print(f"Loading metadata...")
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in metadata file: {e.msg}", e.doc, e.pos)

def load_transcript(text_file: str, debug_mode: bool) -> str:
    """
    Load transcript from file trying different encodings.
    
    Args:
        transcript_file (str): Path to the transcript file
        
    Returns:
        str: Content of the transcript file
        
    Raises:
        ValueError: If file cannot be read with any of the supported encodings
        FileNotFoundError: If file does not exist
    """
    if debug_mode:
        print(f"Loading transcript...")
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(text_file, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file not found: {text_file}")
    raise ValueError(f"Could not read file with any of the encodings: {encodings}")

def create_regex_pattern(metadata_file: str, debug_mode: bool) -> str:
    """
    Create a regex pattern for the speaker to separate who is speaking from what they're saying.
    """
    metadata = load_metadata(metadata_file, debug_mode)

    # create a list of all the speaker flags
    speaker_attributions = []
    for participant in metadata['participants']:
        speaker_attributions.extend(participant['speaker_attributions'])

    # make an API call
    prompt = f"""You are an expert in creating regular expressions.
Your task is to create a regex that can match all the speaker attributions included into the list.

Here is the list of speaker attributions:
<speaker_attributions>
{speaker_attributions}
</speaker_attributions>

<capture_group_instructions>
Important requirements:
1. The regex must include capturing groups (using parentheses) that extract the relevant information.
2. The regex MUST capture the speaker's name in the FIRST capturing group.
3. For examples like "<BOLD>JENNIFER LANDIS: </BOLD>", your regex should capture "JENNIFER LANDIS" as a group.
4. For examples like "<LINEBREAK> John Doe - CEO - Example Bank <LINEBREAK>", your regex should capture the name, title, and organization in separate groups.
5. Any HTML tags or formatting elements should NOT be included in the capturing group for the speaker name.
6. The pattern should match the entire structure but isolate the important information in capture groups.
7. Do not use unnecessary capturing groups that would shift the speaker name to a group other than group(1).
</capture_group_instructions>

<backslash_instructions>
1. Your regex MUST include proper escape sequences using backslashes (\) where needed.
2. If your regex doesn't contain ANY backslashes, it is almost certainly incorrect for parsing structured text.
3. When providing regex patterns, use single backslashes for regex escapes, not double backslashes.
4. For example, use '\s' for whitespace, not '\\s'.
5. Make sure your pattern works with standard regex syntax that uses single backslashes for special character sequences.
6. Return the regex pattern with proper backslash escapes as it would be used directly in a regex engine, not in a programming language string.
</backslash_instructions>

You response should include only regex.
Do not include comments and analysis into the response.
"""
    if debug_mode:
        print(f"Setting up API call...")
        
    api_key = os.getenv('ANTHROPIC_API_KEY')
    client = anthropic.Anthropic(api_key=api_key)
    unique_prompt = f"{prompt}\n\n[Request timestamp: {time.time()}]"

    if debug_mode:
        print(f"Running API call...")
    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            system="You are an expert in creating regex based on text examples.",
            messages=[{"role": "user", "content": unique_prompt}],
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

        regex_pattern = message.content[0].text
        if debug_mode:
            print(f"Regex pattern from LLM: {regex_pattern}")

        # Append the regex pattern to the metadata file
        with open(metadata_file, 'w', encoding='utf-8') as f:
            metadata['speaker_regex_pattern'] = regex_pattern
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        return {
            "regex_pattern": regex_pattern,
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
        print(f"Error in create_regex_pattern: {e}")
        return None


def remove_unused_sections(metadata_file: str, debug_mode: bool) -> str:
    """
    Remove sections before the presentation start page.
    
    Args:
        transcript_text (str): Full transcript text
        metadata_file (str): Path to the metadata file
        
    Returns:
        str: Transcript text starting from the presentation start page
        
    Raises:
        KeyError: If required metadata fields are missing
        ValueError: If page numbers are invalid
    """
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

def get_utterances(metadata_file: str, debug_mode: bool) -> Optional[str]:
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
        for name in participant['speaker_name_variants']:
            valid_speakers.add(name.lower())

   
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
    debug_counts = {
        'angle_brackets': 0,
        'parentheses': 0,
        #'double_quotes': 0,
        #'single_quotes': 0,
        'double_slashes': 0,
        'backslashes': 0
    }

    if debug_mode:
        print(f"Cleaning utterances...")

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

        # Remove text within ""
        # double_quotes_count = len(re.findall(r'"[^"]*"', text))
        # text = re.sub(r'"[^"]*"', '', text)
        # debug_counts['double_quotes'] += double_quotes_count

        # Remove text within ''
        # single_quotes_count = len(re.findall(r"'[^']*'", text))
        # text = re.sub(r"'[^']*'", '', text)
        # debug_counts['single_quotes'] += single_quotes_count

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

        cleaned_utterances.append({
            'uuid': utterance['uuid'],
            'speaker': utterance['speaker'],
            'utterance': text
        })

    if debug_mode:
        print("Saving debug counts to clean_debug.txt")
        with open('clean_debug.txt', 'w', encoding='utf-8') as debug_file:
            for key, count in debug_counts.items():
                debug_file.write(f"{key}: {count}\n")

    return cleaned_utterances
    
def create_output_json(metadata_file, utterances_file_path, debug_mode: bool):
    
    # Load the original metadata
    metadata = load_metadata(metadata_file, debug_mode)

    # Create a copy of the original metadata
    output_metadata = copy.deepcopy(metadata)
    
    # Initialize the "utterances" container
    output_metadata['utterances'] = []

    # Function to get utterances
    try:
        if debug_mode:
            print("Getting utterances with get_utterances...")
        utterances = get_utterances(metadata_file, debug_mode)
        if debug_mode:
            print("Cleaning utterances with clean_utterances...")
        utterances = clean_utterances(utterances, debug_mode)
        if debug_mode:
            print("Adding utterances to metadata...")
        if utterances:
            for utterance in utterances:
                output_metadata['utterances'].append({
                    "uuid": utterance['uuid'],
                    "speaker": utterance['speaker'],
                    "utterance": utterance['utterance']
                })
    except Exception as e:
        print(f"Error in get_utterances: {e}")

    # Save the updated metadata with utterances to a new file
    with open(utterances_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_metadata, f, ensure_ascii=False, indent=4)

    return output_metadata


if __name__ == '__main__':

    test_mode_enabled, test_file_name, debug_mode = get_test_mode()

    if test_mode_enabled:
        print("Test mode is enabled. Running test.")
        folder_paths = get_folder_paths()
        metadata_file = folder_paths['metadata_folder'] + '/' + test_file_name.replace('.pdf', '_metadata_short.json')    
        transcript_txt_file_path = load_metadata(metadata_file, debug_mode)['path_to_transcript_txt']
        utterances_file_path = folder_paths['utterances_folder'] + '/' + test_file_name.replace('.pdf', '_utterances_short.json')
    else:
        print("Test mode is not enabled. Skipping test.")
        exit()
    
    try:
        # transcript_text = load_transcript(transcript_txt_file_path, debug_mode)
        # transcript_text = remove_unused_sections(metadata_file, debug_mode)
        create_regex_pattern(metadata_file, debug_mode)
        output_metadata = create_output_json(metadata_file, utterances_file_path, debug_mode)

    except Exception as e:
        print(f"Error: {str(e)}")
