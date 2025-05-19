import streamlit as st
import os
import tempfile
import xml.etree.ElementTree as ET
import pandas as pd
import uuid
import xml.dom.minidom as minidom
import shutil
import time
import requests
import json
import logging
import datetime
import io
from pathlib import Path

# Set up enhanced logging for the application
def setup_logging():
    """Set up enhanced logging for the application"""
    # Corrected: log_dir is defined here
    # This local log_dir is used to create log_filepath
    log_dir_local = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir_local, exist_ok=True)
    log_filename = f"mqxliff_translator_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath_local = os.path.join(log_dir_local, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath_local, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger_local = logging.getLogger('mqxliff_translator')
    
    if not hasattr(st.session_state, 'log_capture_string'):
        st.session_state.log_capture_string = io.StringIO()
        string_handler = logging.StreamHandler(st.session_state.log_capture_string)
        string_handler.setLevel(logging.INFO)
        string_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger_local.addHandler(string_handler)
    
    return logger_local, log_filepath_local # Return log_filepath which contains the path

# Initialize logging
logger, log_filepath = setup_logging() # log_filepath is now globally available

# Log start of application
logger.info("=" * 80)
logger.info("MemoQ Translator Application Started")
logger.info("=" * 80)

# Helper function to log to both the logger and the session state
def log_message(message, level="info"):
    """Log a message to both the logger and the session state"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "success":
        logger.info(f"SUCCESS: {message}")
    elif level == "debug":
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            logger.debug(message) # Use the global logger instance
    
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    if level != "debug" or (hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode):
        st.session_state.logs.append({
            'message': message,
            'type': level,
            'timestamp': timestamp
        })

# Configure streamlit page
st.set_page_config(
    page_title="MemoQ Translation Assistant",
    page_icon="📝",
    layout="wide"
)

# Function to create backup of uploaded file
def create_backup(file_object, file_name):
    """Create a backup of the uploaded file"""
    log_message(f"Creating backup of file: {file_name}")
    
    try:
        backup_dir = os.path.join(os.getcwd(), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(file_name)
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        file_object.seek(0)
        with open(backup_path, 'wb') as backup_file:
            shutil.copyfileobj(file_object, backup_file)
        file_object.seek(0)
        log_message(f"Backup created successfully at: {backup_path}")
        return backup_path
    except Exception as e:
        error_msg = f"Error creating backup: {str(e)}"
        log_message(error_msg, level="error")
        return None

# Function to extract segments from XLIFF file
def extract_translatable_segments(xliff_content):
    """Extract translatable segments from XLIFF file"""
    try:
        log_message("Extracting translatable segments from XLIFF file")
        try:
            ET.register_namespace('mq', 'MQXliff')
            ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
            root = ET.fromstring(xliff_content)
            log_message(f"Successfully parsed XLIFF with ElementTree")
            ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
            file_nodes = root.findall('.//file')
            if not file_nodes:
                file_nodes = root.findall('.//x:file', ns)
            if not file_nodes:
                log_message("Could not find any file nodes in the XLIFF file", level="error")
                return None, None, None, []
            file_node = file_nodes[0]
            source_lang = file_node.get('source-language') or file_node.attrib.get('source-language')
            target_lang = file_node.get('target-language') or file_node.attrib.get('target-language')
            document_name = file_node.get('original') or file_node.attrib.get('original')
            log_message(f"File info: source_lang={source_lang}, target_lang={target_lang}, document_name={document_name}")
            trans_units = root.findall('.//x:trans-unit', ns)
            if not trans_units:
                trans_units = root.findall('.//trans-unit')
            log_message(f"Found {len(trans_units)} translation units")
            segments = []
            for trans_unit in trans_units:
                try:
                    segment_id = trans_unit.get('id') or trans_unit.attrib.get('id') or str(uuid.uuid4())
                    status = trans_unit.get('{MQXliff}status') or \
                             trans_unit.attrib.get('{MQXliff}status') or \
                             trans_unit.get('mq:status') or \
                             trans_unit.attrib.get('mq:status')
                    source_element = trans_unit.find('.//x:source', ns) or trans_unit.find('source')
                    target_element = trans_unit.find('.//x:target', ns) or trans_unit.find('target')
                    if source_element is not None:
                        source_text = source_element.text or ''.join(source_element.itertext()) or ""
                        segments.append({'id': segment_id, 'source': source_text, 'status': status or 'Unknown'})
                except Exception as segment_error:
                    log_message(f"Error processing segment: {str(segment_error)}", level="warning")
            log_message(f"Successfully extracted {len(segments)} segments from XLIFF file")
            if hasattr(st.session_state, 'override_source_lang') and st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
                old_source_lang = source_lang
                source_lang = get_language_code(st.session_state.override_source_lang)
                log_message(f"Source language override: {old_source_lang} -> {source_lang}")
            if hasattr(st.session_state, 'override_target_lang') and st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
                old_target_lang = target_lang
                target_lang = get_language_code(st.session_state.override_target_lang)
                log_message(f"Target language override: {old_target_lang} -> {target_lang}")
            return source_lang, target_lang, document_name, segments
        except Exception as et_error:
            log_message(f"ElementTree approach failed: {str(et_error)}", level="error")
            try:
                log_message("Trying XLIFF parsing with minidom")
                dom = minidom.parseString(xliff_content)
                file_nodes = dom.getElementsByTagName('file')
                if not file_nodes:
                    log_message("No file nodes found with minidom approach", level="error")
                    return None, None, None, []
                file_node = file_nodes[0]
                source_lang = file_node.getAttribute('source-language')
                target_lang = file_node.getAttribute('target-language')
                document_name = file_node.getAttribute('original')
                log_message(f"File info from minidom: source_lang={source_lang}, target_lang={target_lang}")
                trans_units = dom.getElementsByTagName('trans-unit')
                log_message(f"Found {len(trans_units)} translation units with minidom")
                segments = []
                for trans_unit in trans_units:
                    try:
                        segment_id = trans_unit.getAttribute('id') or str(uuid.uuid4())
                        status = trans_unit.getAttribute('mq:status')
                        source_elements = trans_unit.getElementsByTagName('source')
                        if source_elements:
                            source_text = "".join(node.data for node in source_elements[0].childNodes if node.nodeType == node.TEXT_NODE)
                            segments.append({'id': segment_id, 'source': source_text, 'status': status or 'Unknown'})
                    except Exception as segment_error:
                        log_message(f"Error processing segment with minidom: {str(segment_error)}", level="warning")
                log_message(f"Successfully extracted {len(segments)} segments with minidom")
                if hasattr(st.session_state, 'override_source_lang') and st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
                    old_source_lang = source_lang
                    source_lang = get_language_code(st.session_state.override_source_lang)
                    log_message(f"Source language override: {old_source_lang} -> {source_lang}")
                if hasattr(st.session_state, 'override_target_lang') and st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
                    old_target_lang = target_lang
                    target_lang = get_language_code(st.session_state.override_target_lang)
                    log_message(f"Target language override: {old_target_lang} -> {target_lang}")
                return source_lang, target_lang, document_name, segments
            except Exception as dom_error:
                log_message(f"minidom approach also failed: {str(dom_error)}", level="error")
                return None, None, None, []
    except Exception as e:
        log_message(f"Error parsing XLIFF file: {str(e)}", level="error")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", level="error")
        return None, None, None, []

# Function for TM matching
def extract_tm_matches(tmx_content, source_lang, target_lang, source_segments, match_threshold):
    """Extract translation memory matches from TMX file"""
    try:
        log_message(f"Finding TM matches with threshold {match_threshold}%")
        tree = ET.ElementTree(ET.fromstring(tmx_content))
        root = tree.getroot()
        tm_matches = []
        base_source_lang = source_lang.split('-')[0]
        base_target_lang = target_lang.split('-')[0]
        for tu in root.findall('.//tu'):
            source_segment_tm = None # Renamed to avoid conflict
            target_segment_tm = None # Renamed
            for tuv in tu.findall('.//tuv'):
                lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang', tuv.get('lang', ''))
                base_lang = lang.split('-')[0]
                seg_element = tuv.find('.//seg') # Corrected variable name
                if seg_element is not None and seg_element.text:
                    if base_lang == base_source_lang or lang == source_lang:
                        source_segment_tm = seg_element.text
                    if base_lang == base_target_lang or lang == target_lang:
                        target_segment_tm = seg_element.text
            if source_segment_tm and target_segment_tm:
                for segment_data in source_segments: # Renamed to avoid conflict
                    similarity = calculate_similarity(segment_data['source'], source_segment_tm)
                    if similarity >= match_threshold:
                        tm_matches.append({
                            'sourceText': source_segment_tm,
                            'targetText': target_segment_tm,
                            'similarity': similarity
                        })
        tm_matches.sort(key=lambda x: x['similarity'], reverse=True)
        unique_matches = []
        seen = set()
        for match in tm_matches:
            key = f"{match['sourceText']}|{match['targetText']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
                if len(unique_matches) >= 5:
                    break
        log_message(f"Found {len(unique_matches)} TM matches above threshold")
        return unique_matches
    except Exception as e:
        log_message(f"Error extracting TM matches: {str(e)}", level="error")
        return []

# Simple similarity calculation
def calculate_similarity(text1, text2):
    """Simple similarity calculation (percentage-based)"""
    if not text1 or not text2: # Handle empty strings
        return 0
    if text1 == text2:
        return 100
    words1 = text1.split()
    words2 = text2.split()
    if not words1 or not words2: # Handle if one string becomes empty after split
        return 0
    matches = sum(1 for word1 in words1 if word1 in words2)
    similarity = (matches / max(len(words1), len(words2))) * 100
    return round(similarity)

# Function to extract terminology
def extract_terminology(csv_content, source_segments):
    """Extract terminology matches from CSV file"""
    try:
        log_message("Extracting terminology matches")
        # Fixed: Use io.StringIO for pandas to read string as CSV
        df = pd.read_csv(io.StringIO(csv_content))
        
        if len(df.columns) < 2:
            log_message("CSV file must have at least 2 columns for terminology", level="warning")
            return []
        
        term_matches = []
        seen_terms = set()
        
        source_col_name = df.columns[0]
        target_col_name = df.columns[1]

        for _, row in df.iterrows():
            source_term = str(row[source_col_name])
            target_term = str(row[target_col_name])
            
            if not source_term or pd.isna(source_term) or not target_term or pd.isna(target_term):
                continue
                
            for segment in source_segments:
                if source_term.lower() in segment['source'].lower():
                    key = f"{source_term}|{target_term}"
                    if key not in seen_terms:
                        seen_terms.add(key)
                        term_matches.append({
                            'source': source_term,
                            'target': target_term
                        })
                        # Optimization: if a term is found, no need to check other segments for the same term pair
                        # However, a term might be relevant to multiple segments. So, we let it find all occurrences.
        
        log_message(f"Found {len(term_matches)} terminology matches")
        return term_matches
    except Exception as e:
        # Log the error with more details, including a snippet of csv_content if it's small enough
        csv_snippet = csv_content[:200] + "..." if len(csv_content) > 200 else csv_content
        log_message(f"Error extracting terminology: {str(e)}. CSV snippet: {csv_snippet}", level="error")
        import traceback
        log_message(f"Terminology extraction traceback: {traceback.format_exc()}", level="error")
        return []


# Get language name from code
def get_language_name(lang_code):
    """Get full language name from code"""
    language_map = {
        'bg': 'Bulgarian', 'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
        'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French',
        'ga': 'Irish', 'hr': 'Croatian', 'hu': 'Hungarian', 'it': 'Italian', 'lt': 'Lithuanian',
        'lv': 'Latvian', 'mt': 'Maltese', 'nl': 'Dutch', 'pl': 'Polish', 'pt': 'Portuguese',
        'ro': 'Romanian', 'sk': 'Slovak', 'sl': 'Slovenian', 'sv': 'Swedish', 'tr': 'Turkish',
        'en-US': 'English (US)', 'en-GB': 'English (UK)', 'fr-FR': 'French (France)',
        'fr-CA': 'French (Canada)', 'de-DE': 'German (Germany)', 'de-AT': 'German (Austria)',
        'de-CH': 'German (Switzerland)', 'es-ES': 'Spanish (Spain)', 'es-MX': 'Spanish (Mexico)',
        'pt-PT': 'Portuguese (Portugal)', 'pt-BR': 'Portuguese (Brazil)', 'zh': 'Chinese',
        'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi', 'ru': 'Russian',
        'uk': 'Ukrainian', 'he': 'Hebrew', 'th': 'Thai', 'vi': 'Vietnamese', 'no': 'Norwegian'
    }
    if lang_code not in language_map and '-' in lang_code:
        base_lang = lang_code.split('-')[0]
        if base_lang in language_map:
            return f"{language_map[base_lang]} ({lang_code.split('-')[1]})"
    return language_map.get(lang_code, lang_code)

# Function to get language code from name
def get_language_code(language_name):
    """Get language code from name"""
    language_map = {name: code for code, name in get_language_name(None).items() if isinstance(code, str)} # Inverse map
    # Re-populate for direct name to code mapping based on the structure of get_language_name
    inv_map = {
        'Bulgarian': 'bg', 'Czech': 'cs', 'Danish': 'da', 'German': 'de', 'Greek': 'el',
        'English': 'en', 'Spanish': 'es', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr',
        'Irish': 'ga', 'Croatian': 'hr', 'Hungarian': 'hu', 'Italian': 'it', 'Lithuanian': 'lt',
        'Latvian': 'lv', 'Maltese': 'mt', 'Dutch': 'nl', 'Polish': 'pl', 'Portuguese': 'pt',
        'Romanian': 'ro', 'Slovak': 'sk', 'Slovenian': 'sl', 'Swedish': 'sv', 'Turkish': 'tr',
        'English (US)': 'en-US', 'English (UK)': 'en-GB', 'French (France)': 'fr-FR',
        'French (Canada)': 'fr-CA', 'German (Germany)': 'de-DE', 'German (Austria)': 'de-AT',
        'German (Switzerland)': 'de-CH', 'Spanish (Spain)': 'es-ES', 'Spanish (Mexico)': 'es-MX',
        'Portuguese (Portugal)': 'pt-PT', 'Portuguese (Brazil)': 'pt-BR', 'Chinese': 'zh',
        'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi', 'Russian': 'ru',
        'Ukrainian': 'uk', 'Hebrew': 'he', 'Thai': 'th', 'Vietnamese': 'vi', 'Norwegian': 'no'
    }
    return inv_map.get(language_name, language_name)


# Get the list of EU languages + Turkish for the dropdowns
def get_language_options():
    """Get list of languages for dropdowns"""
    # This list is based on the keys of the inv_map in get_language_code for consistency
    # or directly from a predefined list as before.
    eu_languages_plus = [
        'Bulgarian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 
        'Finnish', 'French', 'German', 'Greek', 'Hungarian', 'Irish', 
        'Italian', 'Latvian', 'Lithuanian', 'Maltese', 'Polish', 
        'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Spanish', 
        'Swedish', 'Turkish'
    ]
    variants = [
        'English (US)', 'English (UK)', 'French (France)', 'French (Canada)',
        'German (Germany)', 'German (Austria)', 'German (Switzerland)',
        'Spanish (Spain)', 'Spanish (Mexico)', 'Portuguese (Portugal)', 
        'Portuguese (Brazil)'
    ]
    other_languages = [
        'Arabic', 'Chinese', 'Hindi', 'Japanese', 'Korean', 'Norwegian',
        'Russian', 'Ukrainian', 'Hebrew', 'Thai', 'Vietnamese'
    ]
    all_languages = sorted(list(set(eu_languages_plus + variants + other_languages))) # Use set to ensure unique names
    return all_languages

# Function to create AI prompt
def create_ai_prompt(prompt_template, source_lang, target_lang, document_name, batch, tm_matches, term_matches):
    """Create a prompt for the AI model"""
    try:
        log_message("Creating AI prompt")
        prompt = prompt_template + '\n\n' if prompt_template else ''
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        prompt += f"Translate from {source_lang_name} ({source_lang}) to {target_lang_name} ({target_lang}).\n\n"
        prompt += f"Document: {document_name}\n\n"
        if tm_matches:
            prompt += "TRANSLATION MEMORY EXAMPLES:\n"
            for match in tm_matches:
                prompt += f"Source: {match['sourceText']}\nTarget: {match['targetText']}\nSimilarity: {match['similarity']}%\n\n"
        if term_matches:
            prompt += "REQUIRED TERMINOLOGY:\n"
            for term in term_matches:
                prompt += f"{term['source']} → {term['target']}\n"
            prompt += "\n"
        prompt += "IMPORTANT INSTRUCTIONS:\n"
        prompt += "1. Translate each segment in order, maintaining the [number] at the beginning of each line.\n"
        prompt += "2. Preserve the original formatting, including HTML/XML tags, placeholders, and special markers.\n"
        prompt += "3. Ensure the translation maintains the same meaning and tone as the original.\n"
        prompt += "4. Use the terminology provided above consistently.\n"
        prompt += "5. Format your response as: [1] Translation for segment 1, [2] Translation for segment 2, etc.\n\n"
        prompt += "SEGMENTS TO TRANSLATE:\n"
        for i, segment in enumerate(batch):
            prompt += f"[{i+1}] {segment['source']}\n"
        log_message(f"Created prompt with {len(batch)} segments")
        prompt_dir = os.path.join(os.getcwd(), 'prompts')
        os.makedirs(prompt_dir, exist_ok=True)
        batch_index = st.session_state.current_batch if hasattr(st.session_state, 'current_batch') else 0
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_filename = f"prompt_batch_{batch_index}_{timestamp}.txt"
        prompt_path = os.path.join(prompt_dir, prompt_filename)
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        log_message(f"Saved full prompt to {prompt_path}")
        # ADDED: Log the prompt content
        log_message(f"AI Prompt Content (Batch {batch_index}):\n---BEGIN PROMPT---\n{prompt}\n---END PROMPT---", level="info")
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            log_message("=== FULL PROMPT START ===", level="debug")
            log_message(prompt, level="debug")
            log_message("=== FULL PROMPT END ===", level="debug")
        return prompt
    except Exception as e:
        log_message(f"Error creating prompt: {str(e)}", level="error")
        return ""

# Function to get translations from AI
def get_ai_translation(api_provider, api_key, model, prompt, source_lang, target_lang, temperature=0.3):
    """Get translations from AI model using direct API calls"""
    log_message(f"Sending request to {api_provider} API ({model}) with temperature {temperature}")
    
    batch_index = st.session_state.current_batch if hasattr(st.session_state, 'current_batch') else 0
    ai_response_content = "" # Initialize variable

    try:
        if api_provider == 'anthropic':
            headers = {
                "x-api-key": api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": model, "max_tokens": 4000, "temperature": temperature,
                "system": (
                    "You are a professional translator specializing in technical documents. "
                    "Translate precisely while preserving all formatting, tags, and special characters. "
                    f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                    "Pay special attention to cultural nuances and linguistic patterns."
                ),
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            if response.status_code != 200:
                error_message = f"Anthropic API Error: Status {response.status_code}, {response.text}"
                log_message(error_message, level="error")
                raise Exception(error_message)
            result = response.json()
            log_message("Received response from Anthropic API")
            ai_response_content = result["content"][0]["text"]
        elif api_provider == 'openai': # Corrected: Changed else to elif
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": (
                        "You are a professional translator specializing in technical documents. "
                        "Translate precisely while preserving all formatting, tags, and special characters. "
                        f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                        "Pay special attention to cultural nuances and linguistic patterns."
                    )},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000, "temperature": temperature
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120)
            if response.status_code != 200:
                error_message = f"OpenAI API Error: Status {response.status_code}, {response.text}"
                log_message(error_message, level="error")
                raise Exception(error_message)
            result = response.json()
            log_message("Received response from OpenAI API")
            ai_response_content = result["choices"][0]["message"]["content"]
        else:
            log_message(f"Unknown API Provider: {api_provider}", level="error")
            raise ValueError(f"Unsupported API provider: {api_provider}")

        # Common logic for saving and logging response
        response_dir = os.path.join(os.getcwd(), 'responses')
        os.makedirs(response_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        response_filename = f"response_batch_{batch_index}_{timestamp}.txt"
        response_path = os.path.join(response_dir, response_filename)
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(ai_response_content)
        log_message(f"Saved AI response to {response_path}")
        # ADDED: Log the AI response content
        log_message(f"AI Response Content (Batch {batch_index} - {api_provider}):\n---BEGIN RESPONSE---\n{ai_response_content}\n---END RESPONSE---", level="info")
        
        return ai_response_content
    
    except requests.exceptions.RequestException as e:
        error_message = f"Network error with {api_provider}: {str(e)}"
        log_message(error_message, level="error")
        raise Exception(error_message) # Re-raise to be caught by batch processing
    except json.JSONDecodeError as e:
        error_message = f"JSON parsing error with {api_provider} response: {str(e)}"
        log_message(error_message, level="error")
        raise Exception(error_message)
    except Exception as e: # Catch-all for other errors, including API specific ones after status check
        error_message = f"API Error with {api_provider}: {str(e)}"
        log_message(error_message, level="error")
        raise Exception(error_message)


# Function to parse AI response
def parse_ai_response(ai_response, batch):
    """Parse AI response to extract translations"""
    try:
        log_message("Parsing AI response")
        translations = {}
        lines = ai_response.split('\n')
        for i, segment in enumerate(batch):
            segment_number = i + 1
            pattern_variants = [
                f"\\[{segment_number}\\]\\s*(.+)",
                f"{segment_number}\\.\\s*(.+)",
                f"^\\s*{segment_number}\\s*[:\\-\\.]?\\s*(.+)"
            ]
            found = False
            for pattern in pattern_variants:
                for line in lines:
                    import re
                    match = re.match(pattern, line)
                    if match:
                        translations[segment['id']] = match.group(1).strip()
                        found = True
                        break
                if found: break
            if not found and i > 0:
                prev_id = batch[i-1]['id']
                if prev_id in translations:
                    for j, line in enumerate(lines):
                        if translations[prev_id] in line:
                            for k in range(j+1, len(lines)):
                                if lines[k].strip():
                                    translations[segment['id']] = lines[k].strip()
                                    found = True
                                    break
                            break
            if not found:
                log_message(f"Could not find translation for segment {segment_number} (ID: {segment['id']})", level="warning")
        log_message(f"Parsed {len(translations)} translations from AI response")
        return translations
    except Exception as e:
        log_message(f"Error parsing AI response: {str(e)}", level="error")
        return {}

# Function to update XLIFF with translations
def update_xliff_with_translations(xliff_content_str, translations): # Renamed variable
    """Update XLIFF file with translations"""
    try:
        log_message(f"Updating XLIFF file with {len(translations)} translations")
        ET.register_namespace('mq', 'MQXliff')
        ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
        tree = ET.ElementTree(ET.fromstring(xliff_content_str)) # Use renamed variable
        root = tree.getroot()
        ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
        updated_count = 0
        for trans_unit in root.findall('.//x:trans-unit', ns) if ns else root.findall('.//trans-unit'):
            segment_id = trans_unit.get('id')
            if segment_id in translations:
                target = trans_unit.find('.//x:target', ns) if ns else trans_unit.find('target')
                if target is None:
                    source = trans_unit.find('.//x:source', ns) if ns else trans_unit.find('source')
                    target_ns_prefix = "{urn:oasis:names:tc:xliff:document:1.2}" if 'x' in ns else "" # Handle missing ns
                    target = ET.SubElement(trans_unit, f'{target_ns_prefix}target')
                    if source is not None and source.get('{http://www.w3.org/XML/1998/namespace}space'):
                        target.set('{http://www.w3.org/XML/1998/namespace}space', 
                                  source.get('{http://www.w3.org/XML/1998/namespace}space'))
                target.text = translations[segment_id]
                status_attr_mq = '{MQXliff}status' if 'mq' in ns else 'mq:status' # Handle missing ns
                if status_attr_mq in trans_unit.attrib:
                    trans_unit.set(status_attr_mq, 'Translated')
                
                lastchanged_attr_mq = '{MQXliff}lastchangedtimestamp' if 'mq' in ns else 'mq:lastchangedtimestamp'
                if lastchanged_attr_mq in trans_unit.attrib:
                    # Corrected: Use timezone-aware datetime
                    trans_unit.set(lastchanged_attr_mq, 
                                  datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
                updated_count += 1
        xml_string = ET.tostring(root, encoding='utf-8', method='xml')
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        if not pretty_xml.startswith('<?xml version="1.0" encoding="UTF-8"?>'): # Ensure UTF-8, not utf-8
             pretty_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty_xml.split('\n', 1)[1]
        else: # Ensure encoding is uppercase UTF-8
            pretty_xml = pretty_xml.replace('encoding="utf-8"','encoding="UTF-8"')

        log_message(f"Successfully updated {updated_count} segments in XLIFF", level="success")
        return pretty_xml, updated_count
    except Exception as e:
        log_message(f"Error updating XLIFF: {str(e)}", level="error")
        import traceback
        log_message(f"XLIFF update traceback: {traceback.format_exc()}", level="error")
        return None, 0


# Function to save translations as text file
def save_translations_as_text(segments_list, translations, filename): # Renamed variable
    """Save translations as a text file when XLIFF update fails"""
    try:
        log_message("Saving translations as text file")
        output_dir = os.path.join(os.getcwd(), 'translated')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_translations_{timestamp}.txt"
        text_path = os.path.join(output_dir, output_filename)
        text_output = "# Translation Results\n\n"
        for seg_id, translation in translations.items():
            original = next((s['source'] for s in segments_list if s['id'] == seg_id), "Original source not found")
            text_output += f"ID: {seg_id}\nSource: {original}\nTarget: {translation}\n\n"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_output)
        log_message(f"Saved translations as text file: {text_path}")
        return text_path
    except Exception as e:
        log_message(f"Error saving translations as text: {str(e)}", level="error")
        return None

# Function to save XLIFF file
def save_translated_xliff(xliff_data, filename): # Renamed variable
    """Save translated XLIFF file"""
    try:
        log_message("Saving translated XLIFF file")
        output_dir = os.path.join(os.getcwd(), 'translated')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_translated_{timestamp}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'wb') as f: # open in binary mode
            if isinstance(xliff_data, str):
                f.write(xliff_data.encode('utf-8'))
            else: # Assuming it's bytes if not str
                f.write(xliff_data)
        log_message(f"Saved translated XLIFF file: {output_path}", level="success")
        return output_path
    except Exception as e:
        log_message(f"Error saving translated XLIFF file: {str(e)}", level="error")
        return None

# Main application
def main():
    st.sidebar.title("Log Information")
    # Corrected: Define log_dir using log_filepath which is global
    # log_filepath is defined after setup_logging() is called
    if 'log_filepath' in globals() and log_filepath and os.path.exists(log_filepath):
        current_log_dir = os.path.dirname(log_filepath) # Get directory from the full path
        st.sidebar.info(f"Log file: {os.path.basename(log_filepath)}")
        st.sidebar.info(f"Location: {current_log_dir}") # Use the derived log_dir
        
        with open(log_filepath, 'r', encoding='utf-8') as log_file_reader:
            log_content_display = log_file_reader.read() # Renamed
            st.sidebar.download_button(
                label="Download Log File",
                data=log_content_display,
                file_name=os.path.basename(log_filepath),
                mime="text/plain"
            )
    else:
        st.sidebar.warning("Log file not available yet or path is incorrect.")

    st.title("MemoQ Translation Assistant")
    st.markdown("Process MemoQ XLIFF files with Translation Memory, Terminology, and AI assistance")
    
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    st.session_state.debug_mode = debug_mode
    
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
        st.session_state.processing_complete = False
        st.session_state.progress = 0.0 # Ensure it's float
        st.session_state.current_batch = 0
        st.session_state.total_batches = 0
        st.session_state.logs = []
        st.session_state.translated_file_path = None
        st.session_state.xliff_content = None
        st.session_state.batch_results = []
        st.session_state.backup_path = None
        st.session_state.override_source_lang = "Auto-detect (from XLIFF)"
        st.session_state.override_target_lang = "Auto-detect (from XLIFF)"
    
    tab1, tab2, tab3 = st.tabs(["File Uploads & Settings", "Processing", "Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("File Uploads")
            xliff_file = st.file_uploader("MemoQ XLIFF File", type=["memoqxliff", "xliff", "mqxliff"], help="Upload a MemoQ XLIFF file to translate")
            tmx_file = st.file_uploader("Translation Memory (TMX)", type=["tmx"], help="Upload a TMX file with translation memory entries")
            csv_file = st.file_uploader("Terminology (CSV)", type=["csv"], help="Upload a CSV file with terminology entries")
            prompt_file = st.file_uploader("Custom Prompt (TXT)", type=["txt"], help="Upload a text file with custom prompt for the AI")
            st.subheader("Language Settings (Optional)")
            language_options = get_language_options()
            st.session_state.override_source_lang = st.selectbox("Override Source Language", ["Auto-detect (from XLIFF)"] + language_options, index=0, help="Override the source language specified in the XLIFF file")
            st.session_state.override_target_lang = st.selectbox("Override Target Language", ["Auto-detect (from XLIFF)"] + language_options, index=0, help="Override the target language specified in the XLIFF file")
        
        with col2:
            st.subheader("Translation Settings")
            api_provider = st.selectbox("AI Provider", ["anthropic", "openai"], help="Select the AI provider to use for translation")
            api_key = st.text_input("API Key", type="password", help="Enter your API key for the selected provider")
            if api_provider == "anthropic":
                model = st.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"])
            else: # OpenAI
                model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
            batch_size = st.slider("Batch Size", min_value=5, max_value=50, value=10, help="Number of segments to process in each batch")
            match_threshold = st.slider("TM Match Threshold (%)", min_value=60, max_value=100, value=75, help="Minimum similarity percentage for TM matches")
            temperature = st.slider("AI Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, help="Controls randomness in translation (0.0 = more deterministic, 1.0 = more creative)")
            custom_prompt_text = st.text_area("Additional prompt instructions (optional)", height=100, help="Add custom instructions to the AI prompt")
            start_button = st.button("Start Processing", disabled=st.session_state.processing_started)

        if st.session_state.processing_started and not st.session_state.processing_complete:
            st.subheader("Progress")
            # Ensure progress is a float between 0.0 and 1.0
            progress_value = float(st.session_state.progress) if isinstance(st.session_state.progress, (int, float)) else 0.0
            progress_bar_tab1 = st.progress(progress_value) # Renamed
            status_text_tab1 = st.empty() # Renamed
            if st.session_state.total_batches > 0:
                status_text_tab1.text(f"Batch {st.session_state.current_batch}/{st.session_state.total_batches} ({int(progress_value * 100)}%)")
            else:
                status_text_tab1.text("Preparing...")
    
    with tab2:
        st.subheader("Processing Status")
        if st.session_state.processing_started and not st.session_state.processing_complete:
            progress_value_tab2 = float(st.session_state.progress) if isinstance(st.session_state.progress, (int, float)) else 0.0
            progress_bar_tab2_ui = st.progress(progress_value_tab2) # Renamed
            status_text_tab2_ui = st.empty() # Renamed
            if st.session_state.total_batches > 0:
                status_text_tab2_ui.text(f"Processing batch {st.session_state.current_batch} of {st.session_state.total_batches}")
            else:
                status_text_tab2_ui.text("Preparing batches...")
            with st.spinner('Processing... Please wait'):
                st.info("Processing your file. This may take several minutes.")
        elif st.session_state.processing_complete:
             st.success("Processing finished. Check Results tab.")


        st.subheader("Process Log")
        log_container = st.container(height=400)
        with log_container:
            if 'logs' in st.session_state and st.session_state.logs:
                for log_item in st.session_state.logs: # Renamed
                    msg_color = "gray"
                    if log_item['type'] == 'info': msg_color = "blue"
                    elif log_item['type'] == 'error': msg_color = "red"
                    elif log_item['type'] == 'warning': msg_color = "orange"
                    elif log_item['type'] == 'success': msg_color = "green"
                    elif log_item['type'] == 'debug' and debug_mode : msg_color = "violet"
                    
                    if log_item['type'] != 'debug' or debug_mode: # Show debug only if mode is on
                         st.markdown(f"<span style='color:{msg_color};'>[{log_item['timestamp']}] {log_item['message']}</span>", unsafe_allow_html=True)
            else:
                st.info("No logs available yet.")
        
        if 'log_capture_string' in st.session_state:
            full_log_content = st.session_state.log_capture_string.getvalue() # Renamed
            st.download_button(label="Download Complete Log", data=full_log_content, file_name=f"detailed_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")
        
        if debug_mode:
            st.subheader("Debug Information")
            if st.session_state.xliff_content:
                with st.expander("XLIFF Preview (First 500 chars)"):
                    st.code(st.session_state.xliff_content[:500]) # Corrected variable name
            
            with st.expander("Raw Log Output (Complete from session)"):
                if 'log_capture_string' in st.session_state:
                    st.code(st.session_state.log_capture_string.getvalue())
                else:
                    st.info("No raw logs available in session.")
            
            prompt_dir_view = os.path.join(os.getcwd(), 'prompts') # Renamed
            if os.path.exists(prompt_dir_view):
                prompt_files_list = [f for f in os.listdir(prompt_dir_view) if f.endswith('.txt')] # Renamed
                if prompt_files_list:
                    with st.expander("View Prompts"):
                        selected_prompt_file = st.selectbox("Select prompt file to view", prompt_files_list) # Renamed
                        if selected_prompt_file: # Check if a file is selected
                            prompt_file_path_view = os.path.join(prompt_dir_view, selected_prompt_file) # Renamed
                            try:
                                with open(prompt_file_path_view, 'r', encoding='utf-8') as f_p:
                                    st.code(f_p.read())
                            except Exception as e_p_view: # Renamed
                                st.error(f"Error reading prompt file: {str(e_p_view)}")
            
            response_dir_view = os.path.join(os.getcwd(), 'responses') # Renamed
            if os.path.exists(response_dir_view):
                response_files_list = [f for f in os.listdir(response_dir_view) if f.endswith('.txt')] # Renamed
                if response_files_list:
                    with st.expander("View AI Responses"):
                        selected_response_file = st.selectbox("Select response file to view", response_files_list) # Renamed
                        if selected_response_file: # Check if a file is selected
                            response_file_path_view = os.path.join(response_dir_view, selected_response_file) # Renamed
                            try:
                                with open(response_file_path_view, 'r', encoding='utf-8') as f_r:
                                    st.code(f_r.read())
                            except Exception as e_r_view: # Renamed
                                st.error(f"Error reading response file: {str(e_r_view)}")
    
    with tab3:
        st.subheader("Translation Results")
        if st.session_state.processing_complete:
            st.success("Translation completed!") # Simplified message
            if st.session_state.translated_file_path and os.path.exists(st.session_state.translated_file_path):
                with open(st.session_state.translated_file_path, 'rb') as f_res_download: # Renamed
                    # Determine mime type based on file extension
                    mime_type = "application/xliff+xml" if st.session_state.translated_file_path.endswith(('.xliff', '.mqxliff')) else "text/plain"
                    st.download_button(label="Download Translated File", data=f_res_download, file_name=os.path.basename(st.session_state.translated_file_path), mime=mime_type)
            elif st.session_state.translated_file_path: # File path exists in session but not on disk
                 st.error(f"Translated file was expected at {st.session_state.translated_file_path} but not found. It might have been an error during saving.")

            if st.session_state.batch_results:
                st.subheader("Batch Summary")
                for i, batch_res in enumerate(st.session_state.batch_results): # Renamed
                    with st.expander(f"Batch {i+1}"):
                        col_b_res1, col_b_res2 = st.columns(2) # Renamed
                        with col_b_res1: st.metric("Processed", batch_res.get('segments_processed', 0))
                        with col_b_res2: st.metric("Translated", batch_res.get('translations_received', 0))
                        if 'error' in batch_res: st.error(f"Error: {batch_res['error']}")
        
        if st.session_state.processing_started or st.session_state.processing_complete : # Show reset if started or completed
            if st.button("Start New Translation"):
                # Reset all relevant session state variables
                for key in list(st.session_state.keys()):
                    if key not in ['debug_mode', 'override_source_lang', 'override_target_lang', 'log_capture_string']: # Persist these
                        del st.session_state[key]
                # Re-initialize critical states
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.progress = 0.0
                st.session_state.current_batch = 0
                st.session_state.total_batches = 0
                st.session_state.logs = []
                st.session_state.batch_results = []
                st.rerun()

    if start_button and not st.session_state.processing_started:
        if not xliff_file or not api_key: # Simplified validation for brevity
            st.error("Please upload XLIFF file and enter API Key.")
            if not xliff_file: log_message("XLIFF file not uploaded", level="error")
            if not tmx_file: log_message("TMX file not uploaded (optional, proceeding without)", level="warning") # Made optional for this check
            if not csv_file: log_message("CSV file not uploaded (optional, proceeding without)", level="warning") # Made optional for this check
            if not api_key: log_message("API key not provided", level="error")
            return

        st.session_state.processing_started = True
        st.session_state.processing_complete = False
        st.session_state.logs = [] # Clear previous logs from UI
        st.session_state.batch_results = []
        st.session_state.progress = 0.0
        st.session_state.current_batch = 0 # Initialize for the new run

        log_message("=" * 50)
        log_message(f"Starting translation process for XLIFF: {xliff_file.name}")
        if tmx_file: log_message(f"Using TMX: {tmx_file.name}")
        if csv_file: log_message(f"Using CSV: {csv_file.name}")
        log_message(f"Settings: Provider={api_provider}, Model={model}, Batch Size={batch_size}, Temp={temperature}, TM Threshold={match_threshold}%")
        if st.session_state.override_source_lang != "Auto-detect (from XLIFF)": log_message(f"Source lang override: {st.session_state.override_source_lang}")
        if st.session_state.override_target_lang != "Auto-detect (from XLIFF)": log_message(f"Target lang override: {st.session_state.override_target_lang}")
        log_message("=" * 50)
        
        # Force a rerun to update UI elements like progress bar visibility
        st.rerun()


    # This block will run when st.rerun() is called after pressing "Start Processing"
    if st.session_state.processing_started and not st.session_state.processing_complete and xliff_file:
        # This check ensures that we only proceed if files were available in the previous run context.
        # The actual file objects might not be directly accessible after rerun without re-upload.
        # This part of the logic might need adjustment based on how Streamlit handles file objects across reruns.
        # For simplicity, assuming xliff_content was stored in session_state if needed.
        # The current code re-reads files if processing_logic is here.
        # However, it's better to put the processing logic into a separate function called here if needed.
        # For now, the structure from the original code is followed, assuming `xliff_file` object is still valid.

        try:
            xliff_file_bytes = xliff_file.read() # Read once
            backup_main_path = create_backup(io.BytesIO(xliff_file_bytes), xliff_file.name) # Pass BytesIO for re-readability
            st.session_state.backup_path = backup_main_path
            if backup_main_path:
                # This UI update for backup might be tricky without a rerun, consider logging only
                log_message(f"Sidebar backup download should be available for {os.path.basename(backup_main_path)}")

            def decode_content(file_bytes, file_type_name="File"):
                try: return file_bytes.decode('utf-8'), "UTF-8"
                except UnicodeDecodeError:
                    log_message(f"Failed to decode {file_type_name} with UTF-8, trying UTF-16", level="warning")
                    try: return file_bytes.decode('utf-16'), "UTF-16"
                    except UnicodeDecodeError:
                        if file_bytes.startswith(b'\xff\xfe') or file_bytes.startswith(b'\xfe\xff'):
                            return file_bytes.decode('utf-16'), "UTF-16 with BOM"
                        log_message(f"UTF-16 failed for {file_type_name}, trying Latin-1 as fallback.", level="warning")
                        return file_bytes.decode('latin-1'), "Latin-1 (Fallback)"
            
            xliff_content_str_main, enc = decode_content(xliff_file_bytes, "XLIFF")
            log_message(f"Successfully decoded XLIFF file using {enc} encoding")
            st.session_state.xliff_content = xliff_content_str_main # Store raw XLIFF content

            tmx_content_str_main = ""
            if tmx_file:
                tmx_file_bytes = tmx_file.read()
                tmx_content_str_main, enc = decode_content(tmx_file_bytes, "TMX")
                log_message(f"Successfully decoded TMX file using {enc} encoding")

            csv_content_str_main = ""
            if csv_file:
                csv_file_bytes = csv_file.read()
                csv_content_str_main, enc = decode_content(csv_file_bytes, "CSV")
                log_message(f"Successfully decoded CSV file using {enc} encoding")

            prompt_template_str_main = ""
            if prompt_file:
                prompt_file_bytes = prompt_file.read()
                prompt_template_str_main, enc = decode_content(prompt_file_bytes, "Prompt")
                log_message(f"Successfully loaded prompt template using {enc} encoding")
            if custom_prompt_text:
                prompt_template_str_main = (prompt_template_str_main + "\n\n" + custom_prompt_text).strip()
                log_message("Added custom prompt text to template.")

            src_lang, trg_lang, doc_name, seg_list = extract_translatable_segments(st.session_state.xliff_content)
            if not seg_list:
                log_message("No translatable segments extracted. Aborting.", level="error")
                st.session_state.processing_complete = True; st.rerun(); return

            log_message(f"Translation: {get_language_name(src_lang)} ({src_lang}) -> {get_language_name(trg_lang)} ({trg_lang}) for doc: {doc_name}")
            
            main_batches = [seg_list[i:i+batch_size] for i in range(0, len(seg_list), batch_size)]
            st.session_state.total_batches = len(main_batches)
            log_message(f"Prepared {len(seg_list)} segments into {st.session_state.total_batches} batches.")

            all_translations_dict = {} # Renamed
            
            for idx, current_batch_data in enumerate(main_batches): # Renamed
                st.session_state.current_batch = idx + 1
                st.session_state.progress = (idx) / st.session_state.total_batches
                # Force UI update for progress (consider if st.rerun() is too disruptive here)
                # For now, we will let the UI elements in tabs read this state.
                # If a direct update is needed here, a placeholder + st.rerun() or st.experimental_rerun might be required.

                current_batch_log_msg = f"Processing Batch {st.session_state.current_batch}/{st.session_state.total_batches} ({len(current_batch_data)} segments)"
                log_message(current_batch_log_msg)
                batch_processing_result = {'batch_index': idx, 'segments_processed': len(current_batch_data), 'translations_received': 0} # Renamed
                
                try:
                    tm_match_list = [] # Renamed
                    if tmx_file: # Check if TMX file was provided
                         tm_match_list = extract_tm_matches(tmx_content_str_main, src_lang, trg_lang, current_batch_data, match_threshold)
                    
                    term_match_list = [] # Renamed
                    if csv_file: # Check if CSV file was provided
                        term_match_list = extract_terminology(csv_content_str_main, current_batch_data)
                    if not term_match_list: # Ensure it's an empty list if extraction fails or no file
                        term_match_list = []


                    ai_prompt_str = create_ai_prompt(prompt_template_str_main, src_lang, trg_lang, doc_name, current_batch_data, tm_match_list, term_match_list)
                    ai_response_str = get_ai_translation(api_provider, api_key, model, ai_prompt_str, src_lang, trg_lang, temperature)
                    
                    if not ai_response_str:
                        raise Exception("AI response was empty.")
                    
                    batch_translations = parse_ai_response(ai_response_str, current_batch_data)
                    all_translations_dict.update(batch_translations)
                    batch_processing_result['translations_received'] = len(batch_translations)
                    log_message(f"Batch {st.session_state.current_batch} processed. Received {len(batch_translations)} translations.")
                    
                except Exception as batch_e:
                    err_msg_batch = f"Error in Batch {st.session_state.current_batch}: {str(batch_e)}"
                    log_message(err_msg_batch, level="error")
                    batch_processing_result['error'] = str(batch_e)
                
                st.session_state.batch_results.append(batch_processing_result)
                st.session_state.progress = (idx + 1) / st.session_state.total_batches
                time.sleep(0.1) # Small delay
                # st.rerun() # To update progress after each batch - can be too frequent.

            log_message(f"All batches processed. Total translations gathered: {len(all_translations_dict)}.")
            
            if all_translations_dict:
                try:
                    updated_xliff_str, count = update_xliff_with_translations(st.session_state.xliff_content, all_translations_dict)
                    if updated_xliff_str and count > 0:
                        final_output_path = save_translated_xliff(updated_xliff_str, xliff_file.name)
                        if final_output_path:
                            st.session_state.translated_file_path = final_output_path
                            log_message(f"Successfully updated and saved XLIFF: {final_output_path}", level="success")
                        else:
                            log_message("Failed to save updated XLIFF. Saving translations as text.", level="error")
                            text_fallback_path = save_translations_as_text(seg_list, all_translations_dict, xliff_file.name)
                            if text_fallback_path: st.session_state.translated_file_path = text_fallback_path
                    else:
                        raise Exception(f"XLIFF update failed or returned no changes. Updated count: {count}")
                except Exception as final_save_e:
                    log_message(f"Error during final XLIFF update/save: {str(final_save_e)}. Saving as text.", level="error")
                    text_fallback_path_err = save_translations_as_text(seg_list, all_translations_dict, xliff_file.name)
                    if text_fallback_path_err: st.session_state.translated_file_path = text_fallback_path_err
            else:
                log_message("No translations were gathered from any batch. Check logs for errors.", level="warning")

            log_message("=" * 50)
            log_message(f"Translation process fully completed.")
            log_message(f"Total segments processed: {len(seg_list)}")
            log_message(f"Total segments translated in dictionary: {len(all_translations_dict)}")
            if st.session_state.translated_file_path:
                log_message(f"Final translated file: {st.session_state.translated_file_path}")
            else:
                log_message("No output file was generated due to errors or no translations.", level="warning")
            log_message("=" * 50)
            
            st.session_state.processing_complete = True
            st.session_state.progress = 1.0
            log_message("Process marked complete. Rerunning UI.", level="success")
            st.rerun()

        except Exception as e_outer: # Catch errors in the main processing block
            log_message(f"Outer processing error: {str(e_outer)}", level="error")
            import traceback
            log_message(f"Outer Traceback: {traceback.format_exc()}", level="error")
            st.session_state.processing_complete = True # Mark complete to stop retries
            st.session_state.processing_started = False # Allow user to try again
            st.error(f"A critical error occurred: {e_outer}")
            st.rerun()


if __name__ == "__main__":
    main()