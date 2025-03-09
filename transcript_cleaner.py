import json
import re
import os
import argparse
import pymupdf


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    doc = pymupdf.open(pdf_path)
    print(f"pages found in {pdf_path}: {doc.page_count}")

    full_text = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text("text")
        print(f"page {page_num + 1} of {pdf_path} has {len(page_text)} characters")
        if page_text:
            full_text += page_text + "\n<PAGE_BREAK>\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1} of {pdf_path}")

    doc.close()
    return full_text


def clean_text(full_text, config):
    """Clean text according to configuration."""
    # Use the full text directly since it's already a string with page breaks
    all_text = full_text

    # Remove patterns specified in config
    for pattern in config['cleaning_rules']['remove_patterns']:
        all_text = re.sub(pattern, '', all_text, flags=re.MULTILINE)

    # Process the text line by line to handle sections and speakers
    lines = all_text.split('\n')
    processed_lines = []
    truncated_lines = []  # Will store only lines up to the end marker

    # Initialize variables for section tracking
    keep_line = False  # Start with False until we find a section to keep
    current_section = None
    current_speaker = None
    seen_sections = set()  # Track sections we've seen to avoid duplication
    found_end_marker = False

    # Define section marker patterns
    section_markers = config['content_extraction'].get('section_markers', {})
    presentation_pattern = re.compile(section_markers.get('presentation_start', 'PRESENTATION')) if 'presentation_start' in section_markers else None
    qa_pattern = re.compile(section_markers.get('qa_start', 'QUESTION AND ANSWER')) if 'qa_start' in section_markers else None

    # Get end marker pattern - this is critical for truncating the transcript
    end_marker = section_markers.get('end_marker')
    end_pattern = re.compile(end_marker, re.IGNORECASE) if end_marker else None

    # Get speaker pattern
    speaker_pattern = None
    if 'speaker_pattern' in config['content_extraction']:
        speaker_pattern = re.compile(config['content_extraction']['speaker_pattern'])

    # Extract keep sections
    keep_sections = [re.escape(section) for section in config['cleaning_rules'].get('keep_sections', [])]
    keep_section_pattern = re.compile('|'.join(keep_sections)) if keep_sections else None

    # Extract ignore sections
    ignore_sections = [re.escape(section) for section in config['cleaning_rules'].get('ignore_sections', [])]
    ignore_section_pattern = re.compile('|'.join(ignore_sections)) if ignore_sections else None

    # If no section markers are defined, process all text
    if not presentation_pattern and not qa_pattern and not keep_section_pattern:
        keep_line = True
        print("No section markers defined, processing all text")

    # Flag for duplicate header removal
    remove_duplicate_headers = config['processing'].get('remove_duplicate_headers', False)

    # Process each line
    for line in lines:
        # Check for end marker - stop processing if found
        if end_pattern and end_pattern.search(line):
            # Include the end marker line itself
            truncated_lines.append(line)
            found_end_marker = True
            print(f"Found end marker: {line}")
            break

        # Add line to truncated_lines (we'll process these later)
        truncated_lines.append(line)

    # Only process the truncated lines
    for line in truncated_lines:
        # Handle page breaks
        if "<PAGE_BREAK>" in line:
            continue

        # Skip empty lines
        if not line.strip():
            continue

        # Check if line is a section header
        if keep_section_pattern and keep_section_pattern.search(line):
            section_name = line.strip()

            # Skip duplicate section headers if enabled
            if remove_duplicate_headers and section_name in seen_sections:
                continue

            seen_sections.add(section_name)
            current_section = section_name
            processed_lines.append(f"\n\n{current_section}\n")
            keep_line = True
            continue

        # Skip ignore sections
        if ignore_section_pattern and ignore_section_pattern.search(line):
            keep_line = False
            continue

        # Process presentation section
        if presentation_pattern and presentation_pattern.search(line):
            keep_line = True
            section_name = "PRESENTATION"

            # Skip duplicate section headers if enabled
            if remove_duplicate_headers and section_name in seen_sections:
                continue

            seen_sections.add(section_name)
            current_section = section_name
            processed_lines.append("\n\nPRESENTATION\n")

        # Process Q&A section
        if qa_pattern and qa_pattern.search(line):
            keep_line = True
            section_name = "QUESTION AND ANSWER"

            # Skip duplicate section headers if enabled
            if remove_duplicate_headers and section_name in seen_sections:
                continue

            seen_sections.add(section_name)
            current_section = section_name
            processed_lines.append("\n\nQUESTION AND ANSWER\n")

        # Only process lines if we're in a section to keep
        if keep_line:
            # Check if line contains a speaker
            if speaker_pattern:
                speaker_match = speaker_pattern.search(line)
                if speaker_match:
                    # Extract speaker name based on the format specified
                    if 'speaker_format' in config['content_extraction'] and isinstance(config['content_extraction']['speaker_format'], str):
                        format_template = config['content_extraction']['speaker_format']
                        speaker_name = format_template.format(*speaker_match.groups())
                    else:
                        # Default to first group if format not specified
                        speaker_name = speaker_match.group(1)

                    # Apply speaker corrections if any
                    speaker_corrections = config['content_extraction'].get('speaker_name_corrections', {})
                    if speaker_name in speaker_corrections:
                        speaker_name = speaker_corrections[speaker_name]

                    current_speaker = speaker_name.strip()

                    # Extract and clean the utterance
                    utterance = line[speaker_match.end():].strip()
                    if utterance:
                        processed_lines.append(f"\n{current_speaker}: {utterance}")
                    else:
                        processed_lines.append(f"\n{current_speaker}:")
                else:
                    # This is a continuation of previous speaker's text or just regular text
                    if current_speaker and line.strip():
                        # Check if the last line was from the same speaker
                        if processed_lines and processed_lines[-1].startswith(f"\n{current_speaker}:"):
                            # Append to the existing utterance
                            processed_lines[-1] += f" {line.strip()}"
                        else:
                            # It's a continuation paragraph
                            processed_lines.append(f" {line.strip()}")
                    else:
                        # No speaker pattern and no current speaker, just add the line
                        processed_lines.append(f"\n{line.strip()}")
            else:
                # No speaker pattern defined, just add the line
                processed_lines.append(f"\n{line.strip()}")

    # Join all processed lines
    cleaned_text = ''.join(processed_lines)

    # Handle special characters
    for char, replacement in config['processing'].get('special_characters', {}).items():
        cleaned_text = cleaned_text.replace(char, replacement)

    # Remove multiple consecutive newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # Log whether we found the end marker
    if found_end_marker:
        print("Successfully truncated transcript at the end marker")
    else:
        print("Warning: End marker not found in transcript")

    return cleaned_text


def extract_metadata(text, config):
    """Extract metadata from the transcript text based on configuration."""
    metadata = {}

    # Extract date
    if 'date_format' in config['metadata']:
        date_pattern = re.compile(config['metadata']['date_format'])
        date_match = date_pattern.search(text)
        if date_match:
            metadata['date'] = date_match.group(0)

    # Extract title/heading
    if 'heading_format' in config['metadata']:
        heading_pattern = re.compile(config['metadata']['heading_format'])
        heading_match = heading_pattern.search(text)
        if heading_match:
            metadata['heading'] = heading_match.group(0)

    # Extract other custom metadata fields
    for key, pattern in config['metadata'].items():
        if key not in ['date_format', 'heading_format', 'footer_format'] and pattern:
            try:
                pattern_re = re.compile(pattern)
                match = pattern_re.search(text)
                if match:
                    metadata[key.replace('_format', '')] = match.group(0)
                    # If there are capture groups, add them as well
                    if match.groups():
                        for i, group in enumerate(match.groups(), 1):
                            if group:
                                metadata[f"{key.replace('_format', '')}_group_{i}"] = group
            except re.error as e:
                print(f"Error compiling regex pattern for {key}: {e}")

    return metadata


def process_transcript(pdf_path, config_path, output_path=None):
    """Process the transcript and save to output file."""
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Extract text from PDF
    full_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(full_text)} characters from PDF")

    # Extract metadata first (before cleaning)
    metadata = extract_metadata(full_text, config)
    print(f"Extracted metadata: {metadata}")

    # Clean text according to configuration
    cleaned_text = clean_text(full_text, config)
    print(f"Cleaned text has {len(cleaned_text)} characters")

    # Determine output file path
    if not output_path:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"{base_name}_clean.txt"

    # Save cleaned text to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"Cleaned transcript saved to {output_path}")

    # Optionally, save metadata to a separate file
    """
    if metadata and len(metadata) > 0:
        metadata_path = f"{os.path.splitext(output_path)[0]}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    """

    return output_path


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Clean bank earnings call transcripts.')
    parser.add_argument('pdf_path', help='Path to the PDF transcript')
    parser.add_argument('config_path', help='Path to the configuration file')
    parser.add_argument('--output', '-o', help='Output file path (optional)')

    args = parser.parse_args()

    process_transcript(args.pdf_path, args.config_path, args.output)


if __name__ == "__main__":
    main()

# python transcript_cleaner.py pdf_transcripts/2023_q4_us_3_citi_transcript.pdf citi_config.json
