import json
import re
import copy
import uuid
import os
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
            return test_mode.get("enabled", False), test_mode.get("file_name", None)
    except FileNotFoundError:
        print(f"Configuration file {config_paths} not found.")
        return False, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_paths}.")
        return False, None
    
def load_metadata(metadata_file: str) -> Dict:
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
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in metadata file: {e.msg}", e.doc, e.pos)

def load_transcript(text_file: str) -> str:
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

def remove_unused_sections(metadata_file: str) -> str:
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
    metadata = load_metadata(metadata_file)
    text = load_transcript(metadata['path_to_transcript_txt'])
    
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

def get_utterances(metadata_file: str) -> Optional[str]:
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
        metadata = load_metadata(metadata_file)
        text = load_transcript(metadata['path_to_transcript_txt'])
        speaker_pattern = re.compile(metadata['speaker_regex_pattern'], re.MULTILINE)
        matches = list(speaker_pattern.finditer(text))
    except KeyError as e:
        raise KeyError(f"Required metadata fields are missing: {str(e)}")

    if not matches:
        return []

    print(f"Found {len(matches)} matches")
    
    # Prepare a set of all possible speaker names (normalized to lowercase)
    valid_speakers = set()
    for participant in metadata['participants']:
        # Add the primary speaker name
        valid_speakers.add(participant['speaker_name'].lower())
        # Add any misspelled variations
        if 'speaker_misspelled_names' in participant and participant['speaker_misspelled_names']:
            for variant in participant['speaker_misspelled_names']:
                valid_speakers.add(variant.lower())

    """
    # SAVE VALID SPEAKER SET TO FILE
    print("Writing valid speakers to file...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'valid_speakers.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for speaker in valid_speakers:
            f.write(speaker + '\n')
    """
    
    # Filter matches to only include valid speakers and collect the sequence of speakers
    filtered_matches = []
    speaker_sequence = []

    for match in matches:
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

def clean_utterances(utterances: list) -> list:
    cleaned_utterances = []
    debug_counts = {
        'angle_brackets': 0,
        'parentheses': 0,
        #'double_quotes': 0,
        #'single_quotes': 0,
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

    # Save debug counts to clean_debug.txt
    #with open('clean_debug.txt', 'w', encoding='utf-8') as debug_file:
    #    for key, count in debug_counts.items():
    #        debug_file.write(f"{key}: {count}\n")

    return cleaned_utterances
    
def create_output_json(metadata_file, utterances_file_path):
    
    # Load the original metadata
    metadata = load_metadata(metadata_file)

    # Create a copy of the original metadata
    output_metadata = copy.deepcopy(metadata)
    
    # Initialize the "utterances" container
    output_metadata['utterances'] = []

    # Function to get utterances
    try:
        print("Getting utterances...")
        utterances = get_utterances(metadata_file)
        print("Cleaning utterances...")
        utterances = clean_utterances(utterances)
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

    test_mode_enabled, test_file_name = get_test_mode()

    if test_mode_enabled:
        print("Test mode is enabled. Running test.")
        folder_paths = get_folder_paths()
        metadata_file = folder_paths['metadata_folder'] + '/' + test_file_name.replace('.pdf', '_metadata.json')    
        transcript_txt_file_path = load_metadata(metadata_file)['path_to_transcript_txt']
        utterances_file_path = folder_paths['utterances_folder'] + '/' + test_file_name.replace('.pdf', '_utterances.json')
    else:
        print("Test mode is not enabled. Skipping test.")
        exit()
    
    try:
        transcript_text = load_transcript(transcript_txt_file_path)
        transcript_text = remove_unused_sections(metadata_file)
        output_metadata = create_output_json(metadata_file, utterances_file_path)
        print(f"Output metadata file created")
    except Exception as e:
        print(f"Error: {str(e)}")
