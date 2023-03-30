import re
import os
import yaml
import nltk
import pdfbox
import torch
import openpyxl
import json
import datetime
from docx2pdf import convert
from typing import List, Set
from nltk.corpus import stopwords
from transformers import pipeline
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

""" Defining Functions """


# Here the keyword is the entity group that user chooses from the dropdown menu
def document_extraction(keyword: str, regex: str, file: str, proximity_stop_words=None, limit=0, exact_match=True,
                        duplicates=True, direction="both"):
    # Matches dates in the formats "yyyy-mm-dd" or "yyyy/mm/dd"
    pattern1 = r'\d{4}[-/]\d{2}[-/]\d{2}'

    # Matches dates in the formats "dd-MMM-yyyy" or "dd/MMM/yyyy"
    pattern2 = r'\d{2}[-/][A-Za-z]{3}[-/]\d{4}'

    # Matches dates in the format "Month dd, yyyy"
    pattern3 = r'[A-Za-z]+\s+\d{1,2},\s+\d{4}'

    # Matches dates in the format "20th - 22nd January 2023"
    pattern4 = r'\d{1,2}(?:st|nd|rd|th)\s*-\s*\d{1,2}(?:st|nd|rd|th)\s+[A-Za-z]+\s+\d{4}'

    # Matches dates in the format "22nd January 2023"
    pattern5 = r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}'

    pattern6 = r'\d{4}'

    pattern7 = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'

    pattern8 = r'(January|February|March|April|May|June|July|August|September|October|November|December)'

    patterns = {
        'Price': r'[\$£€¥]\s?\d+',
        'Date': f'({pattern1}|{pattern2}|{pattern3}|{pattern4}|{pattern5}|{pattern6}|{pattern7}|{pattern8})',
        'URL': r'\b((?:https?|ftp)://[^\s/$.?#].[^\s]*)\b',
        'Email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'Phone Number': r'\+?\d{1,2}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', }

    # defining a function to convert files to txt from pdf and docx
    def Text_Extraction(file):
        """
            Parameters
            ----------
            file_path : str
                Path of the Pdf file or Docx file

            Returns
            -------
            str
                Extracted text from the input file
        """
        if file.endswith(".pdf"):
            p = pdfbox.PDFBox()
            extracted_text = p.extract_text(file)
        elif file.endswith(".docx"):
            a = file.split('.')[-2]
            convert(file, file + a + ".pdf")
            p = pdfbox.PDFBox()
            extracted_text = p.extract_text(file + a + ".pdf")
        elif file.endswith(".txt"):
            with open(file, "r") as file:
                extracted_text = file.read()
        else:
            print("File Format Not Supported")

        return extracted_text

    # differentiate between normal txt and other files
    def check_format(file):
        if file.endswith(".pdf") or file.endswith(".docx"):
            Text_Extraction(file)
            text_file = os.path.splitext(file)[0] + ".txt"
            with open(text_file, "r", encoding="utf-8") as f:
                return f.read()
        elif file.endswith(".txt"):
            with open(file, "r") as file:
                return file.read()
        elif file.endswith(""):
            return file
        else:
            print("File Format Not Supported")

    # Classifying text
    tokenizer = AutoTokenizer.from_pretrained("obsei-ai/sell-buy-intent-classifier-bert-mini")
    model = AutoModelForSequenceClassification.from_pretrained("obsei-ai/sell-buy-intent-classifier-bert-mini")
    classifier = pipeline('text-classification', tokenizer=tokenizer, model=model)

    def Classify_text(file):
        # Checking the file type
        contents = check_format(file)

        # defining model
        # Creating a text-classification pipeline
        # the output will be a label and score
        Final_output = classifier(contents)

        return Final_output

    # Auto-responding to the user based on classified text
    # This auto-responder will respond to generalized text (e.g., emails)
    def auto_respond(text):
        # Classify the text and extract the label with the highest score
        labels = Classify_text(text)
        if not labels:
            raise ValueError("Unable to classify input text")

        # Check if the highest score is above the threshold of 0.6
        highest_score = max(labels, key=lambda x: x['score'])['score']
        if highest_score < 0.6:
            # Do nothing if score is below 0.6
            return "This Person will get back to you as soon as they see this. Have a Good Day :)"

        label = max(labels, key=lambda x: x['score'])['label']

        # Write a conditional statement to determine the response based on the classification
        if label == "LABEL_0":
            return "Thank you for your interest in selling. Our team will get back to you shortly."
        elif label == "LABEL_1":
            return "Thank you for your interest in buying. Our team will get back to you shortly."
        else:
            return "This Person will get back to you as soon as they see this. Have a Good Day :) "

    # Tagging using NER model
    def ner_tag(text):
        # # Classify the text and extract the label with the highest score
        # labels = Classify_text(text)
        # if not labels:
        #     raise ValueError("Unable to classify input text")
        # label = max(labels, key=lambda x: x['score'])['label']
        # # Apply NER tagging if the label is correct
        # if label == "LABEL_0":
        #     # Import the model and apply NER tagging
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        # apply ner tagging
        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        output = nlp(text)
        entities = [(e['entity_group'], e['word']) for e in output]
        # entities = [e['word'] for e in output]
        if not entities:
            raise ValueError("No entities found in the input text")
        return entities

    def extract_entities(entities, group_label):
        # Define the entity_groups dictionary with keys and values swapped
        entity_groups = {
            'Person': 'PER',
            'Organization': 'ORG',
            'Location': 'LOC',
            'Other': 'MISC'
        }

        # Check if group label is valid
        if group_label not in entity_groups.keys():
            raise ValueError("Invalid group label")

        # Extract the entities with the specified group label
        extracted_entities = [e[1] for e in entities if e[0] == entity_groups[group_label]]

        # If no entities were found for the specified group label, try the MISC label
        if not extracted_entities:
            if group_label != 'Other':
                extracted_entities = extract_entities(entities, 'Other')

        return extracted_entities

    def cleaning_body(text):
        """
        Cleans the text by removing unwanted characters and tokens.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        text1 = text.replace(">", "").strip()
        text2 = text1.replace("<", "").strip()
        text3 = text2.replace("/", "").strip()
        text4 = text3.replace("=", "").strip()
        text6 = text4.replace("+", "").strip()
        text7 = text6.replace("*", "").strip()
        tokens = nltk.word_tokenize(text7)
        cleaned_text = ' '.join(tokens)

        return cleaned_text

    def extract_keyword_with_limit(text, keywords, limit, regex, exact_match, duplicates, direction):

        text1 = []
        start1 = []
        end1 = []
        start_indices = []
        end_indices = []
        regex_start = []  # list to store start indices of regex matches
        regex_end = []  # list to store end indices of regex matches

        all_matches = []

        for keyword in keywords:
            matches = []

            # Determine the regular expression for the keyword based on the exact_match parameter
            if exact_match:
                keyword_regex = "\\b" + keyword + "\\b"
            else:
                keyword_regex = keyword

            for match in re.finditer(keyword_regex, text):
                try:
                    start_indices.append(match.start())
                    end_indices.append(match.end())
                except ValueError:
                    # if no match found for a pattern, append 0 to both lists
                    start_indices.append(0)
                    end_indices.append(0)

                    if direction == "forward":
                        start = match.start()
                        end = min(match.end() + limit, len(text))
                    elif direction == "backward":
                        start = max(match.start() - limit, 0)
                        end = match.end()
                    elif direction == "both":
                        start = max(match.start() - limit // 2, 0)
                        end = min(match.end() + limit // 2, len(text))
                    else:
                        raise ValueError("Invalid direction provided. Choose from 'forward', 'backward' or 'both'")
                    substring = text[start:end]
                    text1.append(substring)
                    matches = re.findall(regex, substring)
                    for match_str in matches:
                        if match_str in matches and not duplicates:
                            continue
                        try:
                            match_start = substring.index(match_str)
                            match_end = match_start + len(match_str)
                            reg_start = start + match_start
                            reg_end = start + match_end
                            regex_start.append(reg_start)
                            regex_end.append(reg_end)
                        except ValueError:
                            # if no match found for a pattern, append 0 to both lists
                            regex_start.append(0)
                            regex_end.append(0)

                    # if no value was appended to regex_start and regex_end for the current keyword, append 0
                    if len(regex_start) <= len(keywords):
                        regex_start.append(0)
                        regex_end.append(0)
                    all_matches.append(matches)
                    try:
                        start1.append(start)
                        end1.append(end)
                    except ValueError:
                        start1.append(0)
                        end1.append(0)
            else:
                all_matches.append([])
        if regex_start == []:
            regex_start = 0
        if regex_end == []:
            regex_end = 0

        return (text1, start1, end1, start_indices, end_indices, regex_start, regex_end, all_matches)

    def extract_regex(keywords: list, regex: str, text: str, proximity_stop_words=None, limit=0, exact_match=True,
                      duplicates=True, direction="both"):
        """Extracts all regex patterns within the proximity of a keyword in a text.

        Args:
            keywords (list): The keywords to search for in the text.
            regex (str): The regular expression pattern to search for within the proximity of the keyword.
            text (str): The text to search within.
            proximity_stop_words (List[str]): A list of stop words that define the boundaries of the proximity.
            exact_match (bool): If True, the keyword is searched as a whole word only.
            duplicates (bool): If True, duplicate regex patterns are included in the output.
            direction (str): The direction to search for the proximity of the keyword. Choose from 'forward', 'backward', or 'both'.

        Returns:
            List[str]: A list of all regex patterns found within the proximity of the keyword in the text.
        """
        """Extracts all regex patterns within the proximity of a keyword in a text and returns the results in JSON format."""

        proximity_text1 = []
        proximity_start1 = []
        proximity_end1 = []
        start_indices = []
        end_indices = []
        regex_start = []
        regex_end = []
        proximate_patterns_by_keyword = {}

        for keyword in keywords:
            # Determine the regular expression for the keyword based on the exact_match parameter
            proximate_patterns = []
            if exact_match:
                keyword_regex = r'\b' + keyword + r'\b'
            else:
                keyword_regex = keyword

            # Find all occurrences of the keyword in the text
            matches = re.finditer(keyword_regex, text)

            # Convert the proximity_stop_words list to a set for faster lookup
            stop_words = set(proximity_stop_words)

            # Create a set to store unique patterns
            unique_patterns: Set[str] = set()

            # For each occurrence of the keyword, extract the regex pattern in its proximity
            for match in matches:
                # Get the start and end indices of the keyword in the text
                start, end = match.start(), match.end()
                try:
                    start_indices.append(start)
                    end_indices.append(end)
                except ValueError:
                    # if no match found for a pattern, append 0 to both lists
                    start_indices.append(0)
                    end_indices.append(0)

                # Find the start and end indices of the proximity based on the direction parameter
                # if direction == 'both':
                if direction == 'both':
                    proximity_start = start
                    proximity_end = end
                    found_start_stop_word = False
                    found_end_stop_word = False

                    # Look for stop words in the left direction
                    for i in range(start - 1, -1, -1):
                        if text[i] in stop_words:
                            proximity_start = i + 1
                            found_start_stop_word = True
                            break

                    # Look for stop words in the right direction
                    for i in range(end, len(text)):
                        if text[i] in stop_words:
                            proximity_end = i
                            found_end_stop_word = True
                            break

                    # If a stop word was not found in either direction,
                    # extend the proximity range to the end of the text on that side
                    if not found_start_stop_word and start > 0:
                        proximity_start = 0
                    if not found_end_stop_word and end < len(text):
                        proximity_end = len(text)

                    # Add checks to ensure that proximity_start and proximity_end
                    # are valid index values for the text string
                    if proximity_start < 0:
                        proximity_start = 0
                    if proximity_end > len(text):
                        proximity_end = len(text)
                    if proximity_start > proximity_end:
                        proximity_start, proximity_end = proximity_end, proximity_start

                elif direction == 'forward':
                    proximity_start = end
                    for i, char in enumerate(text[end:], start=end):
                        if char in stop_words:
                            proximity_end = i
                            break
                    else:
                        proximity_end = len(text)
                elif direction == 'backward':
                    proximity_end = start
                    for i in range(start - 1, -1, -1):
                        if text[i] in stop_words:
                            proximity_start = i + 1
                            break
                    else:
                        proximity_start = 0
                else:
                    raise ValueError("Invalid direction provided. Choose from 'forward', 'backward' or 'both'")

                # Extract the proximity text and find the regex pattern within it
                proximity_text = text[proximity_start:proximity_end]
                patterns = re.findall(regex, proximity_text)

                # appending the text and start end indices of the text
                proximity_text1.append(proximity_text)
                try:
                    proximity_start1.append(proximity_start)
                    proximity_end1.append(proximity_end)
                except ValueError:
                    proximity_start1.append(0)
                    proximity_end1.append(0)
                # Check each pattern for duplicates and add them to the output list
                for match_str in patterns:
                    try:
                        match_start = proximity_text.index(match_str)
                        match_end = match_start + len(match_str)
                        reg_start = start + match_start
                        reg_end = start + match_end
                        regex_start.append(reg_start)
                        regex_end.append(reg_end)
                    except ValueError:
                        # if no match found for a pattern, append 0 to both lists
                        regex_start.append(0)
                        regex_end.append(0)

                    if match_str not in unique_patterns:
                        unique_patterns.add(match_str)
                        proximate_patterns.append(match_str)
                    elif duplicates:
                        proximate_patterns.append(match_str)
                        # if no value was appended to regex_start and regex_end for the current keyword, append 0
                if len(regex_start) <= len(keywords):
                    regex_start.append(0)
                    regex_end.append(0)
            proximate_patterns_by_keyword[keyword] = proximate_patterns

        return list(
            proximate_patterns_by_keyword.values()), proximity_text1, proximity_start1, proximity_end1, start_indices, end_indices, regex_start, regex_end

    # Clean the text
    max_seq_length = 700
    try:
        Text = cleaning_body(check_format(file))
    except ValueError:
        Text = "The PDF contents could not be extracted"
        print(Text)
        exit


    # Split the input text into chunks of maximum sequence length
    chunks = [Text[i:i + max_seq_length] for i in range(0, len(Text), max_seq_length)]

    # Initialize a dictionary to store the results for each keyword/entity

    # Pass each chunk to BERT and get the results
    for chunk in chunks:
        # Tokenize and pad the chunk
        tokenized_chunk = tokenizer.encode(chunk, add_special_tokens=True)
        padded_chunk = torch.tensor(tokenized_chunk + [0] * (max_seq_length - len(tokenized_chunk)))

        # Get the relevant text for each keyword/entity in the chunk
        if ner_tag(chunk) is None:
            print("No Entities were found in the Text")
            break
        keywords = extract_entities(ner_tag(chunk), keyword)
        regex_pattern = patterns.get(regex)
        if proximity_stop_words:
            Relevant_text, proximity_text1, proximity_start1, proximity_end1, start_indices, end_indices, regex_start, regex_end = extract_regex(
                keywords, regex_pattern, chunk, proximity_stop_words, exact_match, duplicates, direction)
        else:
            proximity_text1, proximity_start1, proximity_end1, start_indices, end_indices, regex_start, regex_end, Relevant_text = extract_keyword_with_limit(
                chunk, keywords, limit, regex_pattern, exact_match, duplicates, direction)
        print(proximity_text1)
        print(Relevant_text)

        # Add the results to the dictionary for each keyword
        for kw in keywords:
            kw_dict = {
                "name": kw,
                "total": len(keywords),
            }

    # Create a dictionary for the results and add the list of keyword dictionaries
    results_dict = {
        "timestamp": str(datetime.datetime.now()),
        "status": "200",
        "message": "success",
        "entity": keyword,
        "regex": regex,
        "total": len(keywords),
        "keywords": [
            {
                "name": kw_dict,
                "total": i,
                "data": [
                    {
                        "Entity": {
                            "type": keyword,
                            "start": start_indices[i],
                            "end": end_indices[i]
                        },
                        "text_in_proximity": {
                            "text": proximity_text1[i].encode('unicode_escape').decode('unicode_escape'),
                            "start": proximity_start1[i],
                            "end": proximity_end1[i]
                        },
                        "value": {
                            "extract_value": [value for value in list(Relevant_text)[i]],
                            "start": regex_start[i],
                            "end": regex_end[i]
                        }
                    }
                ]
            }
            for i, kw_dict in enumerate(keywords)
        ]
    }

    # Convert the output dictionary to JSON format
    output_json = json.dumps(results_dict, indent=4, ensure_ascii=False)

    # Print the JSON string
    return output_json