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
    log_dir_local_var = os.path.join(os.getcwd(), 'logs') # Renamed to avoid conflict if user has global log_dir
    os.makedirs(log_dir_local_var, exist_ok=True)
    log_filename = f"mqxliff_translator_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath_local_var = os.path.join(log_dir_local_var, log_filename) # Use renamed local variable
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath_local_var, encoding='utf-8'), # Use renamed local variable
            logging.StreamHandler()
        ]
    )
    
    logger_instance = logging.getLogger('mqxliff_translator') # Renamed to avoid conflict
    
    if not hasattr(st.session_state, 'log_capture_string'):
        st.session_state.log_capture_string = io.StringIO()
        string_handler = logging.StreamHandler(st.session_state.log_capture_string)
        string_handler.setLevel(logging.INFO)
        string_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger_instance.addHandler(string_handler) # Use renamed local variable
    
    return logger_instance, log_filepath_local_var # Return the correct local variables

# Initialize logging
logger, log_filepath = setup_logging()

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
            logger.debug(message)
    
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    if level != "debug" or (hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode):
        st.session_state.logs.append({
            'message': message,
            'type': level,
            'timestamp': timestamp
        })


# Add this after your imports and logging setup, before the main functions
class TranslationContextCache:
    """Cache for storing and tracking previous batch translations as context for future batches"""
    
    def __init__(self, max_batches=3):
        """Initialize context cache with settings"""
        self.previous_translations = []  # List of {batch_id, source_texts, translations} dicts
        self.max_batches = max_batches
        
    def add_batch(self, batch_index, batch_segments, translations):
        """Add a new batch of translations to the context"""
        # Format the batch data for contextual reference
        batch_data = {
            'batch_index': batch_index,
            'segments': []
        }
        
        # Store each segment with its translation
        for segment in batch_segments:
            segment_id = segment['id']
            if segment_id in translations:
                batch_data['segments'].append({
                    'id': segment_id,
                    'source': segment['source'],
                    'translation': translations[segment_id]
                })
        
        # Only add if we have translations
        if batch_data['segments']:
            self.previous_translations.append(batch_data)
            
            # Maintain maximum context window - keep only most recent batches
            if len(self.previous_translations) > self.max_batches:
                self.previous_translations.pop(0)  # Remove oldest batch
                
    def get_context_for_prompt(self, max_examples=None):
        """Get formatted context string for inclusion in prompts"""
        if not self.previous_translations:
            return ""
            
        context_str = "PREVIOUSLY TRANSLATED SEGMENTS (FOR CONTEXT):\n"
        
        # Calculate how many examples we can include from each batch
        total_segments = sum(len(batch['segments']) for batch in self.previous_translations)
        examples_per_batch = max_examples // len(self.previous_translations) if max_examples else None
        
        # Add examples from each batch
        for batch in self.previous_translations:
            segments_to_include = batch['segments']
            if examples_per_batch and len(segments_to_include) > examples_per_batch:
                # If too many segments, take some from beginning and end
                half = examples_per_batch // 2
                segments_to_include = segments_to_include[:half] + segments_to_include[-half:]
                
            for seg in segments_to_include:
                context_str += f"Source: {seg['source']}\nTranslation: {seg['translation']}\n\n"
        
        return context_str + "\n"
        
    def get_stats(self):
        """Get statistics about the current context cache"""
        total_segments = sum(len(batch['segments']) for batch in self.previous_translations)
        return {
            'batches': len(self.previous_translations),
            'segments': total_segments
        }

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

# Function to extract segments from XLIFF file (User's Original Version)
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
                    status_keys = ['{MQXliff}status', 'mq:status'] # Add other potential keys if needed
                    status = None
                    for key in status_keys:
                        status = trans_unit.get(key) or trans_unit.attrib.get(key)
                        if status: break
                    
                    source_element = trans_unit.find('.//x:source', ns)
                    if source_element is None:
                        source_element = trans_unit.find('source')
                    
                    if source_element is not None:
                        source_text = source_element.text
                        if source_text is None or source_text.strip() == '':
                            source_text = ''.join(source_element.itertext())
                        if source_text is None: source_text = ""
                        
                        segments.append({'id': segment_id, 'source': source_text, 'status': status or 'Unknown'})
                except Exception as segment_error:
                    log_message(f"Error processing segment: {str(segment_error)}", level="warning")
            
            log_message(f"Successfully extracted {len(segments)} segments from XLIFF file")
            
            if hasattr(st.session_state, 'override_source_lang') and st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
                old_source_lang = source_lang; source_lang = get_language_code(st.session_state.override_source_lang)
                log_message(f"Source language override: {old_source_lang} -> {source_lang}")
            if hasattr(st.session_state, 'override_target_lang') and st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
                old_target_lang = target_lang; target_lang = get_language_code(st.session_state.override_target_lang)
                log_message(f"Target language override: {old_target_lang} -> {target_lang}")
            return source_lang, target_lang, document_name, segments
        
        except Exception as et_error:
            log_message(f"ElementTree approach failed: {str(et_error)}", level="error")
            try: # Fallback to minidom
                log_message("Trying XLIFF parsing with minidom")
                dom = minidom.parseString(xliff_content)
                file_nodes = dom.getElementsByTagName('file')
                if not file_nodes: return None, None, None, []
                file_node = file_nodes[0]
                source_lang = file_node.getAttribute('source-language')
                target_lang = file_node.getAttribute('target-language')
                document_name = file_node.getAttribute('original')
                trans_units = dom.getElementsByTagName('trans-unit')
                segments = []
                for trans_unit in trans_units:
                    try:
                        segment_id = trans_unit.getAttribute('id') or str(uuid.uuid4())
                        status = trans_unit.getAttribute('mq:status') # Assuming mq:status for minidom
                        source_elements = trans_unit.getElementsByTagName('source')
                        if source_elements:
                            source_text = "".join(node.data for node in source_elements[0].childNodes if node.nodeType == node.TEXT_NODE)
                            segments.append({'id': segment_id, 'source': source_text, 'status': status or 'Unknown'})
                    except Exception as segment_error_md: # Renamed variable
                        log_message(f"Error processing segment with minidom: {str(segment_error_md)}", level="warning")
                log_message(f"Successfully extracted {len(segments)} segments with minidom")
                # Language override logic for minidom path
                if hasattr(st.session_state, 'override_source_lang') and st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
                    old_source_lang = source_lang; source_lang = get_language_code(st.session_state.override_source_lang)
                    log_message(f"Source language override (minidom): {old_source_lang} -> {source_lang}")
                if hasattr(st.session_state, 'override_target_lang') and st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
                    old_target_lang = target_lang; target_lang = get_language_code(st.session_state.override_target_lang)
                    log_message(f"Target language override (minidom): {old_target_lang} -> {target_lang}")
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
            source_segment_tm, target_segment_tm = None, None # Renamed variables
            for tuv in tu.findall('.//tuv'):
                lang_attr = tuv.get('{http://www.w3.org/XML/1998/namespace}lang', tuv.get('lang', '')) # Renamed variable
                base_lang_attr = lang_attr.split('-')[0] # Renamed variable
                seg_node = tuv.find('.//seg') # Renamed variable
                if seg_node is not None and seg_node.text:
                    if base_lang_attr == base_source_lang or lang_attr == source_lang:
                        source_segment_tm = seg_node.text
                    if base_lang_attr == base_target_lang or lang_attr == target_lang:
                        target_segment_tm = seg_node.text
            if source_segment_tm and target_segment_tm:
                for current_segment in source_segments: # Renamed variable
                    similarity = calculate_similarity(current_segment['source'], source_segment_tm)
                    if similarity >= match_threshold:
                        tm_matches.append({'sourceText': source_segment_tm, 'targetText': target_segment_tm, 'similarity': similarity})
        tm_matches.sort(key=lambda x: x['similarity'], reverse=True)
        unique_matches = []
        seen_keys = set() # Renamed variable
        for match_item in tm_matches: # Renamed variable
            match_key = f"{match_item['sourceText']}|{match_item['targetText']}" # Renamed variable
            if match_key not in seen_keys:
                seen_keys.add(match_key)
                unique_matches.append(match_item)
                if len(unique_matches) >= 5: break
        log_message(f"Found {len(unique_matches)} TM matches above threshold")
        return unique_matches
    except Exception as e:
        log_message(f"Error extracting TM matches: {str(e)}", level="error")
        return []

# Simple similarity calculation
def calculate_similarity(text1, text2):
    """Simple similarity calculation (percentage-based)"""
    if text1 == text2: return 100
    if not text1 or not text2: return 0 # Handle empty or None strings
    words1 = text1.split()
    words2 = text2.split()
    if not words1 or not words2: return 0 # Handle if split results in empty lists
    common_words = set(words1) & set(words2) # More efficient way to count matches
    matches = len(common_words)
    # Original had sum(1 for word1 in words1 if word1 in words2) - this is fine too.
    # Using max(len(words1), len(words2)) in denominator can be problematic if one list is much larger.
    # A more standard approach like Jaccard or Dice might be better but sticking to original logic for now.
    if max(len(words1), len(words2)) == 0 : return 0 # Avoid division by zero
    similarity = (matches / max(len(words1), len(words2))) * 100
    return round(similarity)

# Function to extract terminology
def extract_terminology(csv_content, source_segments):
    """Extract terminology matches from CSV file"""
    try:
        log_message("Extracting terminology matches")
        # CORRECTED: Use io.StringIO
        df = pd.read_csv(io.StringIO(csv_content))
        
        if len(df.columns) < 2:
            log_message("CSV file must have at least 2 columns", level="warning")
            return []
        
        term_matches = []
        seen_terms = set()
        for _, row_data in df.iterrows(): # Renamed variable
            source_term = str(row_data[df.columns[0]])
            target_term = str(row_data[df.columns[1]])
            if not source_term or pd.isna(source_term) or not target_term or pd.isna(target_term) or source_term.strip()=="" or target_term.strip()=="":
                continue
            for segment_item in source_segments: # Renamed variable
                if source_term.lower() in segment_item['source'].lower():
                    term_key = f"{source_term}|{target_term}" # Renamed variable
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        term_matches.append({'source': source_term, 'target': target_term})
        log_message(f"Found {len(term_matches)} terminology matches")
        return term_matches
    except Exception as e:
        log_message(f"Error extracting terminology: {str(e)}", level="error")
        return []

# Get language name from code
def get_language_name(lang_code): # Original signature
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
    if lang_code is None: return language_map # Added for get_language_code helper
    if lang_code not in language_map and '-' in lang_code:
        base_lang, country = lang_code.split('-', 1) # Split only once
        if base_lang in language_map:
            return f"{language_map[base_lang]} ({country.upper()})" # Standardize country code display
    return language_map.get(lang_code, lang_code) # Return original code if not found

# Function to get language code from name
def get_language_code(language_name): # Original signature
    """Get language code from name"""
    # Create inverse map dynamically
    inv_map = {v: k for k, v in get_language_name(None).items()}
    return inv_map.get(language_name, language_name) # Return original name if no code found

# Get the list of language options for dropdowns
def get_language_options():
    """Get list of languages for dropdowns"""
    # Use values from the dynamically generated map to ensure consistency
    all_language_names = list(get_language_name(None).values())
    return sorted(list(set(all_language_names))) # Unique and sorted

# Function to create AI prompt
def create_ai_prompt(prompt_template, source_lang, target_lang, document_name, batch, 
                    tm_matches, term_matches, translation_context=None):
    """Create a prompt for the AI model, now with translation context"""
    try:
        batch_idx_log = st.session_state.get('current_batch', 0)
        log_message(f"Creating AI prompt for batch {batch_idx_log}")
        
        prompt = prompt_template + '\n\n' if prompt_template else ''
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        prompt += f"Translate from {source_lang_name} ({source_lang}) to {target_lang_name} ({target_lang}).\n\n"
        prompt += f"Document: {document_name}\n\n"

        # Add translation context if available
        if translation_context and hasattr(translation_context, 'get_context_for_prompt'):
            context_examples = translation_context.get_context_for_prompt(max_examples=10)
            if context_examples:
                prompt += context_examples
                context_stats = translation_context.get_stats()
                log_message(f"Added context from {context_stats['batches']} previous batches "
                          f"({context_stats['segments']} segments)")

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
        # Add this new instruction if context is available
        if translation_context and hasattr(translation_context, 'get_context_for_prompt') and translation_context.previous_translations:
            prompt += "5. Maintain consistency with the previously translated segments shown in the context section.\n"
            prompt += "6. Format your response as: [1] Translation for segment 1, [2] Translation for segment 2, etc.\n\n"
        else:
            prompt += "5. Format your response as: [1] Translation for segment 1, [2] Translation for segment 2, etc.\n\n"
        
        prompt += "SEGMENTS TO TRANSLATE:\n"
        for i, segment_item in enumerate(batch):
            prompt += f"[{i+1}] {segment_item['source']}\n"
        
        log_message(f"Created prompt with {len(batch)} segments for batch {batch_idx_log}")
        
        prompt_dir_path = os.path.join(os.getcwd(), 'prompts')
        os.makedirs(prompt_dir_path, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_filename_str = f"prompt_batch_{batch_idx_log}_{timestamp_str}.txt"
        full_prompt_path = os.path.join(prompt_dir_path, prompt_filename_str)
        
        with open(full_prompt_path, 'w', encoding='utf-8') as f_prompt:
            f_prompt.write(prompt)
        log_message(f"Saved full prompt to {full_prompt_path}")

        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            log_message(f"=== FULL PROMPT START (Batch {batch_idx_log}) ===", level="debug")
            log_message(prompt, level="debug")
            log_message(f"=== FULL PROMPT END (Batch {batch_idx_log}) ===", level="debug")
        return prompt
    except Exception as e:
        log_message(f"Error creating prompt: {str(e)}", level="error")
        return ""

# Function to get translations from AI
def get_ai_translation(api_provider, api_key, model, prompt, source_lang, target_lang, temperature=0.3):
    """Get translations from AI model using direct API calls"""
    batch_idx_log_resp = st.session_state.get('current_batch', 0) # Renamed for clarity
    log_message(f"Sending request to {api_provider} API ({model}) for batch {batch_idx_log_resp} with temp {temperature}")
    
    ai_response_text = "" # Initialize

    try:
        if api_provider == 'anthropic':
            headers = {"x-api-key": api_key, "content-type": "application/json", "anthropic-version": "2023-06-01"}
            
            # UPDATED: Using more detailed system prompt
            system_prompt_text = (
                "You are a professional translator specializing in technical documents. "
                "Translate precisely while preserving all formatting, tags, and special characters. "
                f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                "Pay special attention to cultural nuances and linguistic patterns."
            )
            
            data = {"model": model, "max_tokens": 4000, "temperature": temperature,
                    "system": system_prompt_text, "messages": [{"role": "user", "content": prompt}]}
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            if response.status_code != 200:
                raise Exception(f"Anthropic API Error: Status {response.status_code}, {response.text}")
            result = response.json()
            log_message(f"Received response from Anthropic API for batch {batch_idx_log_resp}")
            ai_response_text = result["content"][0]["text"]
        elif api_provider == 'openai': # Corrected: was 'else:'
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            
            # UPDATED: Using more detailed system prompt
            system_prompt_text_openai = (
                "You are a professional translator specializing in technical documents. "
                "Translate precisely while preserving all formatting, tags, and special characters. "
                f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                "Pay special attention to cultural nuances and linguistic patterns."
            )
            
            data = {"model": model, "messages": [{"role": "system", "content": system_prompt_text_openai}, {"role": "user", "content": prompt}],
                    "max_tokens": 4000, "temperature": temperature}
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120)
            if response.status_code != 200:
                raise Exception(f"OpenAI API Error: Status {response.status_code}, {response.text}")
            result = response.json()
            log_message(f"Received response from OpenAI API for batch {batch_idx_log_resp}")
            ai_response_text = result["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")

        # Common part for saving and logging response
        response_dir_path = os.path.join(os.getcwd(), 'responses') # Renamed variable
        os.makedirs(response_dir_path, exist_ok=True)
        timestamp_str_resp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Renamed variable
        response_filename_str = f"response_batch_{batch_idx_log_resp}_{timestamp_str_resp}.txt" # Renamed variable
        full_response_path = os.path.join(response_dir_path, response_filename_str) # Renamed variable
        
        with open(full_response_path, 'w', encoding='utf-8') as f_resp: # Renamed variable
            f_resp.write(ai_response_text)
        log_message(f"Saved AI response to {full_response_path}")

        # ADDED: Log the AI response content
        log_message(f"AI Response Content (Batch {batch_idx_log_resp} - {api_provider}):\n---BEGIN RESPONSE---\n{ai_response_text}\n---END RESPONSE---", level="info")
            
        return ai_response_text
    
    except Exception as e_api: # Renamed for clarity
        error_message_api = f"Error in get_ai_translation ({api_provider}, batch {batch_idx_log_resp}): {str(e_api)}" # Renamed
        log_message(error_message_api, level="error")
        raise Exception(error_message_api) # Re-raise for batch loop to catch


# Function to parse AI response
def parse_ai_response(ai_response, batch_segments): # Renamed variable
    """Parse AI response to extract translations"""
    try:
        log_message("Parsing AI response")
        translations = {}
        lines = ai_response.split('\n')
        for i, segment_data in enumerate(batch_segments): # Renamed variable
            segment_num = i + 1 # Renamed variable
            patterns = [f"\\[{segment_num}\\]\\s*(.+)", f"{segment_num}\\.\\s*(.+)", f"^\\s*{segment_num}\\s*[:\\-\\.]?\\s*(.+)"] # Renamed
            is_found = False # Renamed
            for p in patterns: # Renamed variable
                for line_item in lines: # Renamed variable
                    import re
                    match_obj = re.match(p, line_item) # Renamed
                    if match_obj:
                        translations[segment_data['id']] = match_obj.group(1).strip()
                        is_found = True; break
                if is_found: break
            if not is_found and i > 0:
                prev_segment_id = batch_segments[i-1]['id'] # Renamed
                if prev_segment_id in translations:
                    for j, line_item_outer in enumerate(lines): # Renamed
                        if translations[prev_segment_id] in line_item_outer:
                            for k_idx in range(j+1, len(lines)): # Renamed
                                if lines[k_idx].strip():
                                    translations[segment_data['id']] = lines[k_idx].strip(); is_found = True; break
                            if is_found: break
            if not is_found:
                log_message(f"Could not find translation for segment {segment_num} (ID: {segment_data['id']})", level="warning")
        log_message(f"Parsed {len(translations)} translations from AI response")
        return translations
    except Exception as e:
        log_message(f"Error parsing AI response: {str(e)}", level="error")
        return {}

# Function to update XLIFF with translations
def update_xliff_with_translations(xliff_data_str, translations_dict): # Renamed variables
    """Update XLIFF file with translations"""
    try:
        log_message(f"Updating XLIFF file with {len(translations_dict)} translations")
        ET.register_namespace('mq', 'MQXliff')
        ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
        tree = ET.ElementTree(ET.fromstring(xliff_data_str))
        root = tree.getroot()
        # More robust namespace handling (basic)
        doc_ns_map = {'x': 'urn:oasis:names:tc:xliff:document:1.2'} # Default
        if root.tag.startswith('{urn:oasis:names:tc:xliff:document:1.2}'):
            # Default namespace is XLIFF
            pass # ns_map['x'] is correct
        else: # No default XLIFF namespace found on root, clear 'x' or adapt
            doc_ns_map = {}


        updated_segment_count = 0 # Renamed variable
        # Adapt findall based on whether 'x' namespace is confirmed relevant
        find_path = './/x:trans-unit' if 'x' in doc_ns_map else './/trans-unit'
        
        for trans_unit_node in root.findall(find_path, doc_ns_map if 'x' in doc_ns_map else None): # Renamed
            segment_id_val = trans_unit_node.get('id') # Renamed
            if segment_id_val in translations_dict:
                target_node = trans_unit_node.find('.//x:target', doc_ns_map if 'x' in doc_ns_map else None) # Renamed
                if target_node is None and 'x' not in doc_ns_map : # Try without namespace if 'x' wasn't relevant
                    target_node = trans_unit_node.find('target')
                
                if target_node is None:
                    source_node = trans_unit_node.find('.//x:source', doc_ns_map if 'x' in doc_ns_map else None) # Renamed
                    if source_node is None and 'x' not in doc_ns_map: source_node = trans_unit_node.find('source')
                    
                    target_tag = "{urn:oasis:names:tc:xliff:document:1.2}target" if 'x' in doc_ns_map else "target"
                    target_node = ET.SubElement(trans_unit_node, target_tag)
                    if source_node is not None and source_node.get('{http://www.w3.org/XML/1998/namespace}space'):
                        target_node.set('{http://www.w3.org/XML/1998/namespace}space', source_node.get('{http://www.w3.org/XML/1998/namespace}space'))
                
                target_node.text = translations_dict[segment_id_val]
                
                # MQXliff specific attributes handling more carefully
                mq_status_attr = '{MQXliff}status' # Assuming the URI is fixed
                if trans_unit_node.get(mq_status_attr) is not None:
                    trans_unit_node.set(mq_status_attr, 'Translated')
                mq_timestamp_attr = '{MQXliff}lastchangedtimestamp'
                if trans_unit_node.get(mq_timestamp_attr) is not None:
                    trans_unit_node.set(mq_timestamp_attr, datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
                updated_segment_count += 1
        
        xml_output_str = ET.tostring(root, encoding='UTF-8', method='xml').decode('utf-8') # Renamed
        dom_parsed = minidom.parseString(xml_output_str) # Renamed
        pretty_xml_str = dom_parsed.toprettyxml(indent="  ") # Renamed
        
        # Ensure UTF-8 declaration
        if pretty_xml_str.startswith('<?xml version="1.0" ?>'):
            pretty_xml_str = pretty_xml_str.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>', 1)
        elif 'encoding="UTF-8"' not in pretty_xml_str.splitlines()[0]:
             pretty_xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty_xml_str.split('\n',1)[1]


        log_message(f"Successfully updated {updated_segment_count} segments in XLIFF", level="success")
        return pretty_xml_str, updated_segment_count
    except Exception as e:
        log_message(f"Error updating XLIFF: {str(e)}", level="error")
        import traceback; log_message(f"Update XLIFF Traceback: {traceback.format_exc()}", level="error")
        return None, 0

# Function to save translations as text file
def save_translations_as_text(segments_data_list, translations_map, output_filename_base): # Renamed variables
    """Save translations as a text file when XLIFF update fails"""
    try:
        log_message("Saving translations as text file")
        output_dir_path = os.path.join(os.getcwd(), 'translated') # Renamed
        os.makedirs(output_dir_path, exist_ok=True)
        timestamp_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Renamed
        name_part, ext_part = os.path.splitext(output_filename_base) # Renamed
        text_output_filename = f"{name_part}_translations_{timestamp_now}.txt" # Renamed
        full_text_path = os.path.join(output_dir_path, text_output_filename) # Renamed
        
        text_content = "# Translation Results\n\n" # Renamed
        for segment_id_key, translation_text in translations_map.items(): # Renamed
            original_source_text = next((s['source'] for s in segments_data_list if s['id'] == segment_id_key), "Original source not found") # Renamed
            text_content += f"ID: {segment_id_key}\nSource: {original_source_text}\nTarget: {translation_text}\n\n"
        
        with open(full_text_path, 'w', encoding='utf-8') as f_text: # Renamed
            f_text.write(text_content)
        log_message(f"Saved translations as text file: {full_text_path}")
        return full_text_path
    except Exception as e:
        log_message(f"Error saving translations as text: {str(e)}", level="error")
        return None

# Function to save XLIFF file
def save_translated_xliff(xliff_string_data, output_filename_base): # Renamed variables
    """Save translated XLIFF file"""
    try:
        log_message("Saving translated XLIFF file")
        output_dir_path_save = os.path.join(os.getcwd(), 'translated') # Renamed
        os.makedirs(output_dir_path_save, exist_ok=True)
        timestamp_now_save = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Renamed
        name_part_save, ext_part_save = os.path.splitext(output_filename_base) # Renamed
        final_output_filename = f"{name_part_save}_translated_{timestamp_now_save}{ext_part_save}" # Renamed
        full_output_path = os.path.join(output_dir_path_save, final_output_filename) # Renamed
        
        with open(full_output_path, 'wb') as f_xliff: # Renamed
            if isinstance(xliff_string_data, str):
                f_xliff.write(xliff_string_data.encode('utf-8'))
            else: # Should be bytes if not str, but ET.tostring with encoding='UTF-8' returns bytes.
                f_xliff.write(xliff_string_data) 
        log_message(f"Saved translated XLIFF file: {full_output_path}", level="success")
        return full_output_path
    except Exception as e:
        log_message(f"Error saving translated XLIFF file: {str(e)}", level="error")
        return None

# Main application
def main():
    st.sidebar.title("Log Information")
    # CORRECTED: Use os.path.dirname(log_filepath) to get the directory
    if 'log_filepath' in globals() and log_filepath and os.path.exists(log_filepath):
        actual_log_dir = os.path.dirname(log_filepath)
        st.sidebar.info(f"Log file: {os.path.basename(log_filepath)}")
        st.sidebar.info(f"Location: {actual_log_dir}") # Display derived directory
        try:
            with open(log_filepath, 'r', encoding='utf-8') as log_file_reader: # Added encoding
                log_content_download = log_file_reader.read() # Renamed
                st.sidebar.download_button(label="Download Log File", data=log_content_download,
                                           file_name=os.path.basename(log_filepath), mime="text/plain")
        except Exception as e_log_read_main: # Renamed
             st.sidebar.error(f"Error reading log for download: {str(e_log_read_main)}")
    else:
        st.sidebar.warning("Log file not available or path is incorrect.")

    st.title("MemoQ Translation Assistant")
    st.markdown("Process MemoQ XLIFF files with Translation Memory, Terminology, and AI assistance")
    
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.get('debug_mode', False))
    
    # Initialize session state variables robustly
    default_session_states = {
    'processing_started': False, 'processing_complete': False, 'progress': 0.0,
    'current_batch': 0, 'total_batches': 0, 'logs': [],
    'translated_file_path': None, 'xliff_content_main_run': None, 'batch_results': [], 
    'backup_path': None, 
    'override_source_lang': "Auto-detect (from XLIFF)",
    'override_target_lang': "Auto-detect (from XLIFF)",
    'translation_context': None  # Add this line for contextual caching
}
    for key_state, val_state in default_session_states.items(): # Renamed variables
        if key_state not in st.session_state: st.session_state[key_state] = val_state
    
    tab1, tab2, tab3 = st.tabs(["File Uploads & Settings", "Processing", "Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("File Uploads")
            # Use unique keys for file uploaders if not already done
            uploaded_xliff = st.file_uploader("MemoQ XLIFF File", type=["memoqxliff", "xliff", "mqxliff"], key="uploader_xliff")
            uploaded_tmx = st.file_uploader("Translation Memory (TMX)", type=["tmx"], key="uploader_tmx")
            uploaded_csv = st.file_uploader("Terminology (CSV)", type=["csv"], key="uploader_csv")
            uploaded_prompt_template = st.file_uploader("Custom Prompt (TXT)", type=["txt"], key="uploader_prompt") # Renamed
            
            st.subheader("Language Settings (Optional)")
            lang_opts = get_language_options() # Renamed
            # Ensure index is valid if language is not in options
            current_src_lang_idx = lang_opts.index(st.session_state.override_source_lang) + 1 if st.session_state.override_source_lang in lang_opts else 0
            current_trg_lang_idx = lang_opts.index(st.session_state.override_target_lang) + 1 if st.session_state.override_target_lang in lang_opts else 0

            st.session_state.override_source_lang = st.selectbox("Override Source Language", ["Auto-detect (from XLIFF)"] + lang_opts, index=current_src_lang_idx)
            st.session_state.override_target_lang = st.selectbox("Override Target Language", ["Auto-detect (from XLIFF)"] + lang_opts, index=current_trg_lang_idx)
        
        with col2:
            st.subheader("Translation Settings")
            sel_api_provider = st.selectbox("AI Provider", ["anthropic", "openai"], key="sel_api_provider") # Renamed
            inp_api_key = st.text_input("API Key", type="password", key="inp_api_key") # Renamed
            if sel_api_provider == "anthropic":
                sel_model = st.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"], key="sel_anthropic_model") # Removed future model
            else: # OpenAI
                sel_model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="sel_openai_model")

    # Add context settings
    context_enabled = st.checkbox("Enable Context Between Batches", 
                                value=st.session_state.get('context_enabled', True),
                                help="Use previous batch translations as context for future batches")
    
    if context_enabled:
        max_context_batches = st.slider("Max Context Batches", 
                                      min_value=1, max_value=5, value=st.session_state.get('max_context_batches', 3),
                                      help="Maximum number of previous batches to keep as context")
    else:
        max_context_batches = 0
        
    st.session_state.context_enabled = context_enabled
    st.session_state.max_context_batches = max_context_batches
            
            # Persist slider values using session_state.get
            val_batch_size = st.session_state.get('slider_batch_size', 10)
            val_match_threshold = st.session_state.get('slider_match_threshold', 75)
            val_temperature = st.session_state.get('slider_temperature', 0.0)
            
            st.session_state.slider_batch_size = st.slider("Batch Size", 5, 50, val_batch_size, key="main_batch_size_slider")
            st.session_state.slider_match_threshold = st.slider("TM Match Threshold (%)", 60, 100, val_match_threshold, key="main_match_thresh_slider")
            st.session_state.slider_temperature = st.slider("AI Temperature", 0.0, 1.0, val_temperature, step=0.1, key="main_temp_slider")
            
            st.session_state.custom_prompt_input = st.text_area("Additional prompt instructions (optional)", st.session_state.get('custom_prompt_input', ""), height=100, key="main_custom_prompt_area") # Renamed

            if st.button("Start Processing", disabled=st.session_state.processing_started, key="main_start_button"):
                if not uploaded_xliff: st.error("Please upload a MemoQ XLIFF file"); log_message("XLIFF missing", "error")
                elif not inp_api_key: st.error("Please enter an API Key"); log_message("API Key missing", "error")
                # Optional files TMX, CSV, Prompt - do not block if missing, but log
                else:
                    st.session_state.processing_started = True
                    st.session_state.processing_complete = False
                    st.session_state.logs = [] # Reset logs for new run display
                    st.session_state.batch_results = []
                    st.session_state.progress = 0.0
                    st.session_state.current_batch = 0 # Reset batch counter for the run

                    # Store file contents and settings in session_state for use after rerun
                    st.session_state.run_config = {
                        'xliff_name': uploaded_xliff.name,
                        'xliff_bytes': uploaded_xliff.getvalue(),
                        'tmx_name': uploaded_tmx.name if uploaded_tmx else None,
                        'tmx_bytes': uploaded_tmx.getvalue() if uploaded_tmx else None,
                        'csv_name': uploaded_csv.name if uploaded_csv else None,
                        'csv_bytes': uploaded_csv.getvalue() if uploaded_csv else None,
                        'prompt_template_bytes': uploaded_prompt_template.getvalue() if uploaded_prompt_template else None,
                        'custom_prompt_text': st.session_state.custom_prompt_input,
                        'api_provider': sel_api_provider,
                        'api_key': inp_api_key,
                        'model': sel_model,
                        'batch_size': st.session_state.slider_batch_size,
                        'match_threshold': st.session_state.slider_match_threshold,
                        'temperature': st.session_state.slider_temperature,
                        'override_source_lang': st.session_state.override_source_lang,
                        'override_target_lang': st.session_state.override_target_lang
                    }
                    log_message("=" * 50); log_message(f"Starting translation for: {uploaded_xliff.name}"); log_message("=" * 50)
                    st.rerun() # Trigger rerun to enter processing block

        if st.session_state.processing_started and not st.session_state.processing_complete:
            st.subheader("Progress")
            prog_bar_val = float(st.session_state.progress) # Renamed
            st.progress(prog_bar_val)
            status_text_ui = st.empty() # Renamed
            if st.session_state.total_batches > 0:
                status_text_ui.text(f"Batch {st.session_state.current_batch}/{st.session_state.total_batches} ({int(prog_bar_val * 100)}%)")
            else: status_text_ui.text("Preparing...")
    
    with tab2: # Processing Tab
        st.subheader("Processing Status & Log")
        if st.session_state.processing_started and not st.session_state.processing_complete:
            prog_bar_val_tab2 = float(st.session_state.progress) # Renamed
            st.progress(prog_bar_val_tab2)
            status_text_ui_tab2 = st.empty() # Renamed
            if st.session_state.total_batches > 0: status_text_ui_tab2.text(f"Processing batch {st.session_state.current_batch} of {st.session_state.total_batches}")
            else: status_text_ui_tab2.text("Preparing batches...")
            st.info("Processing... See detailed logs below. This may take time.")
        elif st.session_state.processing_complete:
            st.success("Processing complete. View results in the 'Results' tab.")

        log_display_container = st.container(height=400) # Renamed
        with log_display_container:
            if st.session_state.logs:
                for entry in st.session_state.logs: # Renamed
                    color_map = {'info': 'blue', 'error': 'red', 'warning': 'orange', 'success': 'green', 'debug': 'grey'}
                    log_color = color_map.get(entry['type'], 'black')
                    if entry['type'] != 'debug' or st.session_state.debug_mode:
                        st.markdown(f"<span style='color:{log_color};'>[{entry['timestamp']}] {entry['message']}</span>", unsafe_allow_html=True)
            else: st.info("Logs will appear here once processing starts.")
        
        if 'log_capture_string' in st.session_state: # For full in-memory log
            st.download_button("Download Full Session Log", st.session_state.log_capture_string.getvalue(), 
                               f"full_session_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", 
                               key="download_full_log_tab2")
        # Debug info (simplified)
        if st.session_state.debug_mode and st.session_state.get('xliff_content_main_run'):
            with st.expander("XLIFF Content (First 500 Chars)"): st.code(st.session_state.xliff_content_main_run[:500])
    
    with tab3: # Results Tab
        st.subheader("Translation Results")
        if st.session_state.processing_complete:
            st.success("Translation Process Finished!")
            if st.session_state.translated_file_path and os.path.exists(st.session_state.translated_file_path):
                with open(st.session_state.translated_file_path, 'rb') as f_download_res: # Renamed
                    mime = "application/xliff+xml" if st.session_state.translated_file_path.endswith(('.xliff', '.mqxliff', '.xml')) else "text/plain"
                    st.download_button("Download Translated File", f_download_res, os.path.basename(st.session_state.translated_file_path), mime, key="download_result_file")
            elif st.session_state.translated_file_path: # Path set but file missing
                st.error(f"Translated file was expected at {st.session_state.translated_file_path} but not found.")
            else: st.warning("No translated file was generated. Check logs for details.")

            if st.session_state.batch_results:
                st.subheader("Batch Processing Summary")
                for i, res_item in enumerate(st.session_state.batch_results): # Renamed
                    with st.expander(f"Batch {i+1} Details"):
                        st.metric("Segments Processed", res_item.get('segments_processed', 0))
                        st.metric("Translations Received", res_item.get('translations_received', 0))
                        if 'error' in res_item: st.error(f"Batch Error: {res_item['error']}")
        
        if st.session_state.processing_started or st.session_state.processing_complete:
            if st.button("Start New Translation", key="reset_from_results_tab"):
                # Reset relevant session state, keep uploader states and settings if desired by user
                keys_to_reset = ['processing_started', 'processing_complete', 'progress', 'current_batch', 
                                 'total_batches', 'logs', 'translated_file_path', 'xliff_content_main_run', 
                                 'batch_results', 'backup_path', 'run_config']
                for k_reset in keys_to_reset: # Renamed
                    if k_reset in st.session_state: del st.session_state[k_reset]
                # Re-initialize to defaults
                for key_state, val_state in default_session_states.items(): # Renamed variables
                     if key_state not in ['override_source_lang', 'override_target_lang', 'debug_mode']: # Persist these
                        st.session_state[key_state] = val_state
                st.rerun()

    # --- Main Processing Logic (after rerun from "Start Processing") ---
    if st.session_state.get('processing_started') and \
       not st.session_state.get('processing_complete') and \
       st.session_state.get('run_config'):
        
        run_cfg = st.session_state.run_config # Renamed for clarity

        try:
            log_message("Processing started with stored configuration...")
            # Create backup
            if run_cfg['xliff_bytes']:
                st.session_state.backup_path = create_backup(io.BytesIO(run_cfg['xliff_bytes']), run_cfg['xliff_name'])
                if st.session_state.backup_path: log_message(f"Backup created: {st.session_state.backup_path}")

            # Decode contents
            def decode_bytes_content(byte_data, file_desc): # Renamed
                if not byte_data: return "", "N/A (No Content)"
                try: return byte_data.decode('utf-8'), "UTF-8"
                except UnicodeDecodeError:
                    log_message(f"UTF-8 decode failed for {file_desc}, trying UTF-16.", "warning")
                    try: return byte_data.decode('utf-16'), "UTF-16"
                    except UnicodeDecodeError:
                        log_message(f"UTF-16 decode failed for {file_desc}, trying Latin-1.", "warning")
                        return byte_data.decode('latin-1', errors='replace'), "Latin-1"

            xliff_str_content, enc = decode_bytes_content(run_cfg['xliff_bytes'], "XLIFF")
            st.session_state.xliff_content_main_run = xliff_str_content # Store for potential debug view
            log_message(f"XLIFF ({run_cfg['xliff_name']}) decoded using {enc}.")

            tmx_str_content = ""
            if run_cfg['tmx_bytes']:
                tmx_str_content, enc = decode_bytes_content(run_cfg['tmx_bytes'], "TMX")
                log_message(f"TMX ({run_cfg['tmx_name']}) decoded using {enc}.")

            csv_str_content = ""
            if run_cfg['csv_bytes']:
                csv_str_content, enc = decode_bytes_content(run_cfg['csv_bytes'], "CSV")
                log_message(f"CSV ({run_cfg['csv_name']}) decoded using {enc}.")
            
            prompt_template_str = ""
            if run_cfg['prompt_template_bytes']:
                prompt_template_str, enc = decode_bytes_content(run_cfg['prompt_template_bytes'], "Prompt Template")
                log_message(f"Prompt template decoded using {enc}.")
            if run_cfg['custom_prompt_text']:
                prompt_template_str = (prompt_template_str + "\n\n" + run_cfg['custom_prompt_text']).strip()
                if prompt_template_str: log_message("Custom prompt text from text area applied.")


# Initialize context cache if enabled (ADD THIS HERE)
        if st.session_state.get('context_enabled', True):
            st.session_state.translation_context = TranslationContextCache(
                max_batches=st.session_state.get('max_context_batches', 3)
            )
            log_message(f"Initialized translation context cache with max {st.session_state.get('max_context_batches', 3)} batches")
        else:
            st.session_state.translation_context = None
            log_message("Translation context caching is disabled")

        # Extract Segments (existing code continues here)
        src_lang_main, trg_lang_main, doc_name_main, segments_list_main = extract_translatable_segments(xliff_str_content)

            # Extract Segments
            src_lang_main, trg_lang_main, doc_name_main, segments_list_main = extract_translatable_segments(xliff_str_content)
            if not segments_list_main:
                log_message("No translatable segments extracted. Aborting.", "error"); st.error("Segment extraction failed."); 
                st.session_state.processing_complete = True; st.session_state.processing_started = False; st.rerun(); return

            log_message(f"Extracted {len(segments_list_main)} segments. Doc: '{doc_name_main}', Langs: {src_lang_main} -> {trg_lang_main}")

            # Batching
            cfg_batch_size = run_cfg['batch_size'] # Renamed
            all_batches_main = [segments_list_main[i:i+cfg_batch_size] for i in range(0, len(segments_list_main), cfg_batch_size)] # Renamed
            st.session_state.total_batches = len(all_batches_main)
            log_message(f"Divided into {st.session_state.total_batches} batches of size up to {cfg_batch_size}.")

            master_translations = {} # Renamed
            for i, current_batch_list in enumerate(all_batches_main): # Renamed
                st.session_state.current_batch = i + 1 # Crucial for logging inside other functions
                st.session_state.progress = i / st.session_state.total_batches
                log_message(f"--- Starting Batch {st.session_state.current_batch}/{st.session_state.total_batches} ---")
                current_batch_result = {'batch_index': i, 'segments_processed': len(current_batch_list), 'translations_received': 0} # Renamed
                
                try:
                    current_tm_matches = [] # Renamed
                    if tmx_str_content: current_tm_matches = extract_tm_matches(tmx_str_content, src_lang_main, trg_lang_main, current_batch_list, run_cfg['match_threshold'])
                    
                    current_term_matches = [] # Renamed
                    if csv_str_content: current_term_matches = extract_terminology(csv_str_content, current_batch_list)
                    if not current_term_matches: current_term_matches = [] # Ensure it's a list

                    batch_ai_prompt = create_ai_prompt(
                    prompt_template_str, src_lang_main, trg_lang_main, doc_name_main, 
                    current_batch_list, current_tm_matches, current_term_matches,
                    translation_context=st.session_state.get('translation_context')  # ADD THIS PARAMETER
                )
                    batch_ai_response = get_ai_translation(run_cfg['api_provider'], run_cfg['api_key'], run_cfg['model'], batch_ai_prompt, src_lang_main, trg_lang_main, run_cfg['temperature'])
                    
                    if not batch_ai_response: raise Exception("AI returned an empty response for the batch.")
                    
                    parsed_translations_batch = parse_ai_response(batch_ai_response, current_batch_list) # Renamed
                    master_translations.update(parsed_translations_batch)
                    current_batch_result['translations_received'] = len(parsed_translations_batch)

# Add this batch to the context cache for future batches
                if st.session_state.get('translation_context') and parsed_translations_batch:
                    st.session_state.translation_context.add_batch(
                        i, current_batch_list, parsed_translations_batch
                    )
                    log_message(f"Added batch {i+1} translations to context cache")
                
                log_message(f"Batch {st.session_state.current_batch} finished. Received {len(parsed_translations_batch)} translations.")
                except Exception as e_batch_proc: # Renamed
                    log_message(f"Error in Batch {st.session_state.current_batch} processing: {str(e_batch_proc)}", "error")
                    current_batch_result['error'] = str(e_batch_proc)                
                st.session_state.batch_results.append(current_batch_result)
                st.session_state.progress = (i + 1) / st.session_state.total_batches
                # Consider a very short sleep or removing it if UI updates are handled by Streamlit's natural flow
                # time.sleep(0.05) 

            log_message(f"All batch processing complete. Total unique translations collected: {len(master_translations)}.")

            if master_translations:
                try:
                    final_xliff_str, num_updated_final = update_xliff_with_translations(xliff_str_content, master_translations) # Renamed
                    if final_xliff_str and num_updated_final > 0:
                        saved_final_path = save_translated_xliff(final_xliff_str, run_cfg['xliff_name']) # Renamed
                        if saved_final_path:
                            st.session_state.translated_file_path = saved_final_path
                            log_message(f"Successfully updated and saved translated XLIFF to: {saved_final_path}", "success")
                        else: raise Exception("Failed to save the final translated XLIFF file.")
                    else: raise Exception(f"XLIFF update yielded no changes or failed. Segments updated: {num_updated_final}.")
                except Exception as e_final_xliff_ops: # Renamed
                    log_message(f"Error during final XLIFF operations: {str(e_final_xliff_ops)}. Attempting to save as text.", "error")
                    text_fallback_final_path = save_translations_as_text(segments_list_main, master_translations, run_cfg['xliff_name']) # Renamed
                    if text_fallback_final_path: st.session_state.translated_file_path = text_fallback_final_path
            else:
                log_message("No translations were gathered from any batch. Output file not generated.", "warning")

            log_message("=" * 50); log_message("Translation process fully completed."); log_message("=" * 50)
            st.session_state.processing_complete = True
            st.session_state.progress = 1.0
            st.rerun()

        except Exception as e_outer_proc: # Renamed
            log_message(f"CRITICAL ERROR during main processing block: {str(e_outer_proc)}", "error")
            import traceback
            log_message(f"Outer Processing Traceback: {traceback.format_exc()}", "error")
            st.session_state.processing_complete = True # End processing
            st.session_state.processing_started = False # Allow user to retry
            st.error(f"A critical error stopped the process: {str(e_outer_proc)}")
            st.rerun()

if __name__ == "__main__":
    main()