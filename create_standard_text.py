import json
import os
from datetime import datetime

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
        # "transcripts_cleantxt_folder": folder_paths.get("transcripts_cleantxt_folder", None),
        # "metadata_folder": folder_paths.get("metadata_folder", None),
        # "utterances_folder": folder_paths.get("utterances_folder", None),
        "final_json_folder": folder_paths.get("final_json_folder", None)
    }


def get_json_file_selection(config_data):
    """
    Gets the JSON files saved in the 'final_json' folder, displays them in the terminal,
    and allows the user to select one.
    
    Args:
        config_data (dict): Configuration dictionary containing folder paths
        
    Returns:
        str: The selected JSON file name or None if no files found or selection failed
    """
    # Get the final_json folder path from config
    folder_paths = get_folder_paths(config_data)
    final_json_folder = folder_paths.get("final_json_folder")
    
    if not final_json_folder or not os.path.exists(final_json_folder):
        print(f"Error: Final JSON folder '{final_json_folder}' not found.")
        return None
    
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(final_json_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in '{final_json_folder}'.")
        return None
    
    # Display the files with numbers for selection
    print("\nAvailable JSON files:")
    for i, file_name in enumerate(json_files, 1):
        print(f"{i}. {file_name}")
    
    # Get user selection
    while True:
        try:
            selection = input("\nEnter the number of the file to process (or 'q' to quit): ")
            
            if selection.lower() == 'q':
                return None
            
            selection_index = int(selection) - 1
            if 0 <= selection_index < len(json_files):
                selected_file = json_files[selection_index]
                print(f"Selected: {selected_file}")
                return selected_file
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(json_files)}.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")


def get_file_stats(config_data, selected_json_file):
    """
    Get statistics about the selected JSON file.
    
    Args:
        config_data (dict): Configuration dictionary containing folder paths
        selected_json_file (str): The name of the selected JSON file

    Returns:
        dict: Statistics about the selected JSON file
    """
    # Get the folder paths from config
    folder_paths = get_folder_paths(config_data)
    final_json_folder = folder_paths.get("final_json_folder")
    
    # Construct the full path to the JSON file
    json_file_path = os.path.join(final_json_folder, selected_json_file)
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count the number of utterances
    utterances = data.get("utterances", [])
    utterance_count = len(utterances)

    # Count total number of symbols in all utterances
    total_symbols = sum(len(utterance.get("utterance", "")) for utterance in utterances)

    # Count total number of words in all utterances
    total_words = sum(len(utterance.get("utterance", "").split()) for utterance in utterances)

    # Count unique speakers
    unique_speakers = set(utterance.get("speaker", "") for utterance in utterances)
    unique_speaker_count = len(unique_speakers)
    
    return {
        "file_name": selected_json_file,
        "utterance_count": utterance_count,
        "total_symbols": total_symbols,
        "total_words": total_words,
        "unique_speaker_count": unique_speaker_count
    }


def create_standard_text_file(config_data, selected_json_file):
    """
    Creates a clean text file from the selected JSON file with standardized formatting.
    
    Args:
        config_data (dict): Configuration dictionary containing folder paths
        selected_json_file (str): The name of the selected JSON file
        
    Returns:
        str: Path to the created text file or None if processing failed
    """

    # Today's date and time
    today = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # Get the folder paths from config
    folder_paths = get_folder_paths(config_data)
    final_json_folder = folder_paths.get("final_json_folder")
    
    if not final_json_folder or not os.path.exists(final_json_folder):
        print(f"Error: Final JSON folder '{final_json_folder}' not found.")
        return None
    
    # Construct the full path to the JSON file
    json_file_path = os.path.join(final_json_folder, selected_json_file)
    
    # Create the output file name by replacing _final.json with _text_<date>.txt
    output_file_name = selected_json_file.replace('_final.json', '_text_' + today + '.txt')
    
    # Create standardized_text folder if it doesn't exist
    standardized_text_folder = os.path.join(os.path.dirname(final_json_folder), "standardized_text")
    if not os.path.exists(standardized_text_folder):
        os.makedirs(standardized_text_folder)
        print(f"Created directory: {standardized_text_folder}")

    # Get file stats
    file_stats = get_file_stats(config_data, selected_json_file)
    utterance_count = file_stats.get("utterance_count")
    total_symbols = file_stats.get("total_symbols")
    total_words = file_stats.get("total_words")

    
    # Save the output file in the standardized_text folder
    output_file_path = os.path.join(standardized_text_folder, output_file_name)
    
    try:
        # Load the JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata
        bank_name = data.get("bank_name", "Unknown Bank")
        reporting_period = data.get("reporting_period", "Unknown Period")
        call_date = data.get("call_date", "Unknown Date")
        unique_speaker_count = file_stats.get("unique_speaker_count")
        # Get utterances and participants
        utterances = data.get("utterances", [])
        participants = data.get("participants", [])
        
        # Create a dictionary to quickly look up speaker information by name
        speaker_info = {}
        for participant in participants:
            name_variants = participant.get("speaker_name_variants", [])
            if name_variants:
                speaker_name = name_variants[0]
                title_variants = participant.get("speaker_title_variants", [])
                company_variants = participant.get("speaker_company_variants", [])
                
                speaker_info[speaker_name] = {
                    "title": title_variants[0] if title_variants else "",
                    "company": company_variants[0] if company_variants else ""
                }
        
        # Write the formatted text file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Write stats
            f.write("[TRANSCRIPT SPECS]\n\n")
            f.write(f"Created on: {today}\n")
            f.write(f"File Name: {selected_json_file}\n")
            f.write(f"Unique Speakers: {unique_speaker_count}\n")
            f.write(f"Utterance Count: {utterance_count}\n")
            f.write(f"Total Symbols: {total_symbols}\n")
            f.write(f"Total Words: {total_words}\n\n")

            # Write header
            f.write("[CALL DETAILS]\n\n")
            f.write(f"Bank: {bank_name}\n")
            f.write(f"Reporting Period: {reporting_period}\n")
            f.write(f"Call Date: {call_date}\n\n")

            f.write("[TRANSCRIPT START]\n\n")
            
            # Write each utterance
            for utterance in utterances:
                speaker_name = utterance.get("speaker", "Unknown Speaker")
                speaker_data = speaker_info.get(speaker_name, {"title": "", "company": ""})
                
                f.write(f"Speaker: {speaker_name}\n")
                if speaker_data['title']:
                    f.write(f"Job Title: {speaker_data['title']}\n")
                if speaker_data['company']:
                    f.write(f"Company: {speaker_data['company']}\n")
                f.write(f"Utterance: {utterance.get('utterance', '')}\n\n")

            f.write("[TRANSCRIPT END]\n\n")
            
        print(f"Standard text file created: {output_file_path}")
        return output_file_path
    
    except Exception as e:
        print(f"Error creating standard text file: {str(e)}")
        return None


def main():
    # Load configuration
    config_data = load_config()

    # Get JSON file selection
    selected_json_file = get_json_file_selection(config_data)

    # Create standard text file
    create_standard_text_file(config_data, selected_json_file)


if __name__ == "__main__":
    main()