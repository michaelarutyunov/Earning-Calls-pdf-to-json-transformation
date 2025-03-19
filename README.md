# Quarterly Earning Calls (QEC) PDF to JSON transformation


UBS - 50
CTG - 77


    Guidelines for searching for speaker attribution:
    - Always check the WHOLE transcript from the beginning to the end for ALL variations of attributions for every speaker name and it's variations.
    - Speaker attribution is always separate from the speech by a formatting tag or a punctuation mark.
    - Speaker attribution always starts with the speaker name or it's variations.
    - Look for speaker changes in conversational context (e.g., after a question or statement by another speaker).
    - Pay special attention to paragraphs that begin with a name, especially after a question.
    - Look for paragraphs that begin with a known speaker name and it's variations followed immediately by speech.
    - When a speaker name or it's variation appears at the beginning of a paragraph and is immediately followed by words like "Well," "Thank you," "Look," etc., this is likely an attribution.
    - Consider name variations such as with/without middle initials or abbreviated names.
    - Speaker's name in the attribution can include speaker's job title and company.
    - Speaker attribution is normally preceded by a <LINEBREAK> tag or a combination of <MULTISPACE> and <LINEBREAK> tags, but may appear with NO formatting tags or punctuation at all.
    - Speaker attribution normally ends with a formatting tag or a punctuation mark.
    - Speaker attribution never contains a speech.
    - Two speaker attributions cannot appear next two each other. If they do, they are not speaker attributions.
    - Attribution for operator should always contain the word "Operator".

---------
    <attribution_search_guidelines>
    Guidelines for searching for speaker attribution:
    - IMPORTANT: always check the WHOLE transcript from the beginning to the end for ALL variations of attributions for EVERY name variation of EACH speaker.
    - Look for paragraphs that begin with a known speaker name or it's variations immediately followed by a speech.
    - Look for speaker changes in conversational context (e.g., after a question or statement by another speaker).   
    - Pay attention to the job title and company name variations.
    - Speaker attributions cannot appear next two each other. If they do, they are not speaker attributions. There should always be a speech between attributions.
    - Attribution for operator should always contain the word "Operator".

    Attributions may come in a variety of formats and often inconsistent within the same transcript:
    Here are potential variations in speaker's name, job title and company formatting:
    - can have spelling variations in different parts of the transcript, including with/without middle initials, abbreviated names, inconsistent use of punctuation. 
    - can be in normal, bold, italic, underlined text, upper and title case.
    - can include punctuations marks like a dash, a colon, a period, an apostrophe, a a special symbol, formatting tags, or a combination of these.
    - cannot be separated by a text segment that is not a formatting tag.
    - can be followed by a punctuation and can be separated from it by a formatting tag or a space.
    
    Additionally:
    - If speaker's name is separated from the job title or company by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    - If speaker's name is followed by a text segment that is not a formatting tag, then only speaker name should be a part of attribution.
    </attribution_search_guidelines>

   - Speaker attribution may sometimes appear with minimal or no special formatting tags.


    - Speaker attribution is normally preceded by a <BOLD-> tag or a <LINEBREAK> tag or a combination of <MULTISPACE> and <LINEBREAK> tags, but may appear with NO preceded formatting tags.
    - Speaker attribution normally ends with a formatting tag or a punctuation mark, which may be separated from the name by an empty space.
    - Speaker attribution is always separate from the speech by a formatting tag or a punctuation mark.
    - Speaker attribution always starts with the speaker name or it's variations.
    - When a speaker name or it's variation appears at the beginning of a paragraph and is immediately followed by words like "Well," "Thank you," "Look," etc., this is likely an attribution.
-         


    All these examples may be preceded by a <LINEBREAK> tag or a combination of <MULTISPACE> and <LINEBREAK> tags.
    All these examples may have a <LINEBREAK> tag or a combination of <MULTISPACE> and <LINEBREAK> tags between the attribution and the speech.
    Here is an example of a text where the speaker attribution and company are separated by a speech:
    * <MULTISPACE> <LINEBREAK> Name Surname <MULTISPACE> <LINEBREAK> Good morning, everybody. My first question relates to <MULTISPACE> <LINEBREAK> (Company Name) <MULTISPACE> <LINEBREAK> capital return.
    In this case the company name should be ignored and only speaker name should be a part of attribution.

Examples

Basic Speaker Attribution:
"John Smith: Hello everyone, thank you for joining today's call."
"Mary Johnson: Thank you, John. I'd like to present our quarterly results."

Formatting Variations:
"<BOLD-> ROBERT JONES <-BOLD> - CEO, TechCorp: We're pleased to announce our newest product line."
"<BOLD-> Sarah Williams, CFO: <-BOLD> The financial outlook remains strong despite market challenges."

Spelling and Name Variations:
"Michael J. Thompson: Our research team has made significant progress."
"Mike Thompson: As I mentioned earlier, this breakthrough has been years in the making."

Punctuation Inconsistencies:
"Dr. Jennifer Lee; Chief Scientific Officer: The clinical trials show promising results."
"Dr. J. Lee: These findings support our initial hypothesis about the treatment."

Job Title Separation:
"Amanda Chen: I'm happy to share our (GlobalBrands) marketing strategy for the upcoming quarter." - Only "Amanda Chen" should be considered attribution

Special Characters and Tags:
"<LINESPACE> Thomas O'Hara • Senior VP: The merger will be finalized next month."
"<MULTISPACE> <LINESPACE> Tom O'Hara <MULTISPACE> <LINESPACE> As I previously stated, we expect synergies to be realized by Q4."

Case Sensitivity and Company Name Variations:
"MARK ROBERTSON (Google Cloud): We're investing heavily in AI infrastructure."
"Mark Robertson (GCP): The capabilities we're building will transform the industry."


    1. IMPORTANT: for each speaker name and it's variations, check the whole transcript from start to finish to capture all variations of their speaker attributions.
    2. Find every instance when a speaker name or it's variations is mentioned in the transcript in any format and preceded or followed by a formatting tag or a punctuation mark.
    3. Include each and every speaker attribution found in the transcript even if it is located at the end of the transcript.
    4. Include all variations of speaker names found in the transcript, including variations in spelling, formatting, and punctuation.
    5. Include all variations of speaker titles found in the transcript.
    6. Include all variations of speaker companies found in the transcript.
    7. Pay close attention to the formatting requirements, especially for the reporting period (QX-YYYY) and the header and footer of the transcript.
    8. Include the header and footer patterns with formatting tags in the output.
   

       <task>
    Extract all speaker attributions from the following transcript.    
    Focus only on the text segments that identify who is speaking before their statements.
    Look for text segments anywhere in paragraphs, both at the beginning and at the end of paragraphs, particularly after sentence endings.
    Return complete and comprehensive list of all speaker attributions, with speaker name variations, their title/role if available, and the company they work for.
    Remove duplicates.
    Return call details like bank name, call date and reporting period.
    Return header and footer patterns without formatting tags.
    Format the results as a json object.
    </task>

    <task>
    Follow these step by step instructions:
    Step 1: Find the names of all call participants, including their variations.
    Step 2: For each participant, find their job title and companies, including variations.
    Step 3: Go through the whole text in small overlapping chunks to idenitify all variants of speaker attributions including leading and tailing tags, names, titles and companies (if available), punctuation marks.
    Step 4. Go throught the whole text paragraph by paragraph to idenitify speaker attributions that may appear in the middle, end, beginning of the paragraph.
    Step 5. Verify that no attribution is missing.
    Step 6. Verify conversation flow makes sense.
    Step 7. Identify call details like bank name, call date and reporting period.
    Step 8. Identify header and footer patterns
    Step 9. Return results as a json object
    </task>