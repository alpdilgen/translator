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
from pathlib import Path

# Set up logging configuration
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_filename = f"mqxliff_translator_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('mqxliff_translator')

# Log start of application
logger.info("=" * 80)
logger.info("MemoQ Translator Application Started")
logger.info("=" * 80)

# Configure streamlit page
st.set_page_config(
    page_title="MemoQ Translation Assistant",
    page_icon="📝",
    layout="wide"
)

# Function to create backup of uploaded file
def create_backup(file_object, file_name):
    """Create a backup of the uploaded file"""
    logger.info(f"Creating backup of file: {file_name}")
    
    try:
        # Create backup directory
        backup_dir = os.path.join(os.getcwd(), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create timestamp for unique backup
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create backup filename
        name, ext = os.path.splitext(file_name)
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Reset file pointer to beginning
        file_object.seek(0)
        
        # Write backup file
        with open(backup_path, 'wb') as backup_file:
            shutil.copyfileobj(file_object, backup_file)
        
        # Reset file pointer again for further processing
        file_object.seek(0)
        
        logger.info(f"Backup created successfully at: {backup_path}")
        return backup_path
    
    except Exception as e:
        error_msg = f"Error creating backup: {str(e)}"
        logger.error(error_msg)
        return None

# Function to extract segments from XLIFF file
def extract_translatable_segments(xliff_content):
    """Extract translatable segments from XLIFF file"""
    try:
        logger.info("Extracting translatable segments from XLIFF file")
        
        # Try first with ElementTree
        try:
            # Register MemoQ namespace
            ET.register_namespace('mq', 'MQXliff')
            ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
            
            # Parse XML
            root = ET.fromstring(xliff_content)
            logger.info(f"Successfully parsed XLIFF with ElementTree")
            
            # Get namespace
            ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
            
            # Extract file info
            file_nodes = root.findall('.//file')
            if not file_nodes:
                file_nodes = root.findall('.//x:file', ns)
            
            if not file_nodes:
                logger.error("Could not find any file nodes in the XLIFF file")
                return None, None, None, []
            
            file_node = file_nodes[0]
            
            # Get attributes
            source_lang = file_node.get('source-language')
            if source_lang is None:
                source_lang = file_node.attrib.get('source-language')
            
            target_lang = file_node.get('target-language')
            if target_lang is None:
                target_lang = file_node.attrib.get('target-language')
            
            document_name = file_node.get('original')
            if document_name is None:
                document_name = file_node.attrib.get('original')
            
            logger.info(f"File info: source_lang={source_lang}, target_lang={target_lang}, document_name={document_name}")
            
            # Find trans-unit nodes
            trans_units = []
            trans_units = root.findall('.//x:trans-unit', ns)
            if not trans_units:
                trans_units = root.findall('.//trans-unit')
            
            logger.info(f"Found {len(trans_units)} translation units")
            
            # Extract segments
            segments = []
            
            for trans_unit in trans_units:
                try:
                    # Get ID
                    segment_id = trans_unit.get('id')
                    if segment_id is None:
                        segment_id = trans_unit.attrib.get('id')
                    
                    if segment_id is None:
                        segment_id = str(uuid.uuid4())
                    
                    # Get status
                    status = trans_unit.get('{MQXliff}status')
                    if status is None:
                        status = trans_unit.attrib.get('{MQXliff}status')
                    if status is None:
                        status = trans_unit.get('mq:status')
                    if status is None:
                        status = trans_unit.attrib.get('mq:status')
                    
                    # Find source element
                    source_element = None
                    target_element = None
                    
                    source_element = trans_unit.find('.//x:source', ns)
                    if source_element is None:
                        source_element = trans_unit.find('source')
                    
                    target_element = trans_unit.find('.//x:target', ns)
                    if target_element is None:
                        target_element = trans_unit.find('target')
                    
                    if source_element is not None:
                        # Get source text
                        source_text = source_element.text
                        if source_text is None or source_text.strip() == '':
                            source_text = ''.join(source_element.itertext())
                        
                        if source_text is None:
                            source_text = ""
                        
                        # Check if target is empty
                        has_target = False
                        if target_element is not None:
                            target_text = target_element.text
                            if target_text is None:
                                target_text = ''.join(target_element.itertext())
                            
                            has_target = target_text is not None and target_text.strip() != ''
                        
                        # Add segment to list
                        segments.append({
                            'id': segment_id,
                            'source': source_text,
                            'status': status or 'Unknown'
                        })
                
                except Exception as segment_error:
                    logger.warning(f"Error processing segment: {str(segment_error)}")
                    continue
            
            logger.info(f"Successfully extracted {len(segments)} segments from XLIFF file")
            
            # Check if we should override languages based on session state
            if hasattr(st.session_state, 'override_source_lang'):
                if st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
                    old_source_lang = source_lang
                    source_lang = get_language_code(st.session_state.override_source_lang)
                    logger.info(f"Source language override: {old_source_lang} -> {source_lang}")
            
            if hasattr(st.session_state, 'override_target_lang'):
                if st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
                    old_target_lang = target_lang
                    target_lang = get_language_code(st.session_state.override_target_lang)
                    logger.info(f"Target language override: {old_target_lang} -> {target_lang}")
            
            return source_lang, target_lang, document_name, segments
        
        except Exception as et_error:
            logger.error(f"ElementTree approach failed: {str(et_error)}")
            
            # Try with minidom as fallback
            try:
                logger.info("Trying XLIFF parsing with minidom")
                dom = minidom.parseString(xliff_content)
                root = dom.documentElement
                
                # Get file node
                file_nodes = dom.getElementsByTagName('file')
                logger.info(f"Found {len(file_nodes)} file nodes with minidom")
                
                if not file_nodes:
                    logger.error("No file nodes found with minidom approach")
                    return None, None, None, []
                
                file_node = file_nodes[0]
                source_lang = file_node.getAttribute('source-language')
                target_lang = file_node.getAttribute('target-language')
                document_name = file_node.getAttribute('original')
                
                logger.info(f"File info from minidom: source_lang={source_lang}, target_lang={target_lang}")
                
                # Get all trans-units
                trans_units = dom.getElementsByTagName('trans-unit')
                logger.info(f"Found {len(trans_units)} translation units with minidom")
                
                segments = []
                for trans_unit in trans_units:
                    try:
                        segment_id = trans_unit.getAttribute('id')
                        status = trans_unit.getAttribute('mq:status')
                        
                        # Find source and target elements
                        source_elements = trans_unit.getElementsByTagName('source')
                        target_elements = trans_unit.getElementsByTagName('target')
                        
                        if source_elements:
                            source_element = source_elements[0]
                            source_text = ""
                            
                            # Extract text from source
                            for node in source_element.childNodes:
                                if node.nodeType == node.TEXT_NODE:
                                    source_text += node.data
                            
                            # Check if target is empty
                            has_target = False
                            if target_elements:
                                target_element = target_elements[0]
                                target_text = ""
                                
                                for node in target_element.childNodes:
                                    if node.nodeType == node.TEXT_NODE:
                                        target_text += node.data
                                
                                has_target = target_text.strip() != ""
                            
                            # Add segment to list
                            segments.append({
                                'id': segment_id,
                                'source': source_text,
                                'status': status or 'Unknown'
                            })
                    
                    except Exception as segment_error:
                        logger.warning(f"Error processing segment with minidom: {str(segment_error)}")
                        continue
                
                logger.info(f"Successfully extracted {len(segments)} segments with minidom")
                
                # Check if we should override languages based on session state
                if hasattr(st.session_state, 'override_source_lang'):
                    if st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
                        old_source_lang = source_lang
                        source_lang = get_language_code(st.session_state.override_source_lang)
                        logger.info(f"Source language override: {old_source_lang} -> {source_lang}")
                
                if hasattr(st.session_state, 'override_target_lang'):
                    if st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
                        old_target_lang = target_lang
                        target_lang = get_language_code(st.session_state.override_target_lang)
                        logger.info(f"Target language override: {old_target_lang} -> {target_lang}")
                
                return source_lang, target_lang, document_name, segments
            
            except Exception as dom_error:
                logger.error(f"minidom approach also failed: {str(dom_error)}")
                return None, None, None, []
    
    except Exception as e:
        logger.error(f"Error parsing XLIFF file: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, []

# Function for TM matching
def extract_tm_matches(tmx_content, source_lang, target_lang, source_segments, match_threshold):
    """Extract translation memory matches from TMX file"""
    try:
        logger.info(f"Finding TM matches with threshold {match_threshold}%")
        tree = ET.ElementTree(ET.fromstring(tmx_content))
        root = tree.getroot()
        
        # Extract translation units
        tm_matches = []
        
        # Check if source_lang or target_lang have country codes, and get base langs
        base_source_lang = source_lang.split('-')[0]
        base_target_lang = target_lang.split('-')[0]
        
        for tu in root.findall('.//tu'):
            source_segment = None
            target_segment = None
            
            for tuv in tu.findall('.//tuv'):
                lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang', tuv.get('lang', ''))
                base_lang = lang.split('-')[0]
                
                if base_lang == base_source_lang or lang == source_lang:
                    seg = tuv.find('.//seg')
                    if seg is not None and seg.text:
                        source_segment = seg.text
                
                if base_lang == base_target_lang or lang == target_lang:
                    seg = tuv.find('.//seg')
                    if seg is not None and seg.text:
                        target_segment = seg.text
            
            if source_segment and target_segment:
                # Calculate similarity for each segment
                for segment in source_segments:
                    similarity = calculate_similarity(segment['source'], source_segment)
                    if similarity >= match_threshold:
                        tm_matches.append({
                            'sourceText': source_segment,
                            'targetText': target_segment,
                            'similarity': similarity
                        })
        
        # Sort by similarity and deduplicate
        tm_matches.sort(key=lambda x: x['similarity'], reverse=True)
        unique_matches = []
        seen = set()
        
        for match in tm_matches:
            key = f"{match['sourceText']}|{match['targetText']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
                if len(unique_matches) >= 5:  # Limit to 5 matches
                    break
        
        logger.info(f"Found {len(unique_matches)} TM matches above threshold")
        return unique_matches
    except Exception as e:
        logger.error(f"Error extracting TM matches: {str(e)}")
        return []

# Simple similarity calculation
def calculate_similarity(text1, text2):
    """Simple similarity calculation (percentage-based)"""
    if text1 == text2:
        return 100
    
    words1 = text1.split()
    words2 = text2.split()
    
    matches = 0
    for word1 in words1:
        if word1 in words2:
            matches += 1
    
    similarity = (matches / max(len(words1), len(words2))) * 100
    return round(similarity)

# Function to extract terminology
def extract_terminology(csv_content, source_segments):
    """Extract terminology matches from CSV file"""
    try:
        logger.info("Extracting terminology matches")
        df = pd.read_csv(pd.StringIO(csv_content))
        
        if len(df.columns) < 2:
            logger.warning("CSV file must have at least 2 columns")
            return []
        
        # Extract source and target terms
        term_matches = []
        seen_terms = set()
        
        for _, row in df.iterrows():
            source_term = str(row[df.columns[0]])
            target_term = str(row[df.columns[1]])
            
            if not source_term or pd.isna(source_term) or not target_term or pd.isna(target_term):
                continue
                
            # Check if source term appears in any segment
            for segment in source_segments:
                if source_term.lower() in segment['source'].lower():
                    key = f"{source_term}|{target_term}"
                    if key not in seen_terms:
                        seen_terms.add(key)
                        term_matches.append({
                            'source': source_term,
                            'target': target_term
                        })
        
        logger.info(f"Found {len(term_matches)} terminology matches")
        return term_matches
    except Exception as e:
        logger.error(f"Error extracting terminology: {str(e)}")
        return []

# Get language name from code
def get_language_name(lang_code):
    """Get full language name from code"""
    language_map = {
        # EU languages
        'bg': 'Bulgarian',
        'cs': 'Czech',
        'da': 'Danish',
        'de': 'German',
        'el': 'Greek',
        'en': 'English',
        'es': 'Spanish',
        'et': 'Estonian',
        'fi': 'Finnish',
        'fr': 'French',
        'ga': 'Irish',
        'hr': 'Croatian',
        'hu': 'Hungarian',
        'it': 'Italian',
        'lt': 'Lithuanian',
        'lv': 'Latvian',
        'mt': 'Maltese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ro': 'Romanian',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'sv': 'Swedish',
        # Additional languages
        'tr': 'Turkish',
        # Common variants with country codes
        'en-US': 'English (US)',
        'en-GB': 'English (UK)',
        'fr-FR': 'French (France)',
        'fr-CA': 'French (Canada)',
        'de-DE': 'German (Germany)',
        'de-AT': 'German (Austria)',
        'de-CH': 'German (Switzerland)',
        'es-ES': 'Spanish (Spain)',
        'es-MX': 'Spanish (Mexico)',
        'pt-PT': 'Portuguese (Portugal)',
        'pt-BR': 'Portuguese (Brazil)',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ru': 'Russian',
        'uk': 'Ukrainian',
        'he': 'Hebrew',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'no': 'Norwegian'
    }
    
    # Handle country code variants by trying the base language code
    if lang_code not in language_map and '-' in lang_code:
        base_lang = lang_code.split('-')[0]
        if base_lang in language_map:
            return f"{language_map[base_lang]} ({lang_code.split('-')[1]})"
    
    return language_map.get(lang_code, lang_code)

# Function to get language code from name
def get_language_code(language_name):
    """Get language code from name"""
    language_map = {
        # EU languages
        'Bulgarian': 'bg',
        'Czech': 'cs',
        'Danish': 'da',
        'German': 'de',
        'Greek': 'el',
        'English': 'en',
        'Spanish': 'es',
        'Estonian': 'et',
        'Finnish': 'fi',
        'French': 'fr',
        'Irish': 'ga',
        'Croatian': 'hr',
        'Hungarian': 'hu',
        'Italian': 'it',
        'Lithuanian': 'lt',
        'Latvian': 'lv',
        'Maltese': 'mt',
        'Dutch': 'nl',
        'Polish': 'pl',
        'Portuguese': 'pt',
        'Romanian': 'ro',
        'Slovak': 'sk',
        'Slovenian': 'sl',
        'Swedish': 'sv',
        # Additional languages
        'Turkish': 'tr',
        # Common variants with country codes
        'English (US)': 'en-US',
        'English (UK)': 'en-GB',
        'French (France)': 'fr-FR',
        'French (Canada)': 'fr-CA',
        'German (Germany)': 'de-DE',
        'German (Austria)': 'de-AT',
        'German (Switzerland)': 'de-CH',
        'Spanish (Spain)': 'es-ES',
        'Spanish (Mexico)': 'es-MX',
        'Portuguese (Portugal)': 'pt-PT',
        'Portuguese (Brazil)': 'pt-BR',
        'Chinese': 'zh',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Arabic': 'ar',
        'Hindi': 'hi',
        'Russian': 'ru',
        'Ukrainian': 'uk',
        'Hebrew': 'he',
        'Thai': 'th',
        'Vietnamese': 'vi',
        'Norwegian': 'no'
    }
    
    return language_map.get(language_name, language_name)

# Get the list of EU languages + Turkish for the dropdowns
def get_language_options():
    """Get list of languages for dropdowns"""
    eu_languages_plus = [
        'Bulgarian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 
        'Finnish', 'French', 'German', 'Greek', 'Hungarian', 'Irish', 
        'Italian', 'Latvian', 'Lithuanian', 'Maltese', 'Polish', 
        'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Spanish', 
        'Swedish', 'Turkish'
    ]
    
    # Add common variants
    variants = [
        'English (US)', 'English (UK)', 'French (France)', 'French (Canada)',
        'German (Germany)', 'German (Austria)', 'German (Switzerland)',
        'Spanish (Spain)', 'Spanish (Mexico)', 'Portuguese (Portugal)', 
        'Portuguese (Brazil)'
    ]
    
    # Add other major languages
    other_languages = [
        'Arabic', 'Chinese', 'Hindi', 'Japanese', 'Korean', 'Norwegian',
        'Russian', 'Ukrainian', 'Hebrew', 'Thai', 'Vietnamese'
    ]
    
    # Combine all languages and sort alphabetically
    all_languages = sorted(eu_languages_plus + variants + other_languages)
    
    return all_languages

# Function to create AI prompt
def create_ai_prompt(prompt_template, source_lang, target_lang, document_name, batch, tm_matches, term_matches):
    """Create a prompt for the AI model"""
    try:
        logger.info("Creating AI prompt")
        prompt = prompt_template + '\n\n' if prompt_template else ''
        
        # Add source and target language information
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        prompt += f"Translate from {source_lang_name} ({source_lang}) to {target_lang_name} ({target_lang}).\n\n"
        
        # Add document info
        prompt += f"Document: {document_name}\n\n"
        
        # Add TM matches as examples
        if tm_matches:
            prompt += "TRANSLATION MEMORY EXAMPLES:\n"
            for match in tm_matches:
                prompt += f"Source: {match['sourceText']}\nTarget: {match['targetText']}\nSimilarity: {match['similarity']}%\n\n"
        
        # Add terminology requirements
        if term_matches:
            prompt += "REQUIRED TERMINOLOGY:\n"
            for term in term_matches:
                prompt += f"{term['source']} → {term['target']}\n"
            prompt += "\n"
        
        # Add instructions
        prompt += "IMPORTANT INSTRUCTIONS:\n"
        prompt += "1. Translate each segment in order, maintaining the [number] at the beginning of each line.\n"
        prompt += "2. Preserve the original formatting, including HTML/XML tags, placeholders, and special markers.\n"
        prompt += "3. Ensure the translation maintains the same meaning and tone as the original.\n"
        prompt += "4. Use the terminology provided above consistently.\n"
        prompt += "5. Format your response as: [1] Translation for segment 1, [2] Translation for segment 2, etc.\n\n"
        
        # Add segments to translate
        prompt += "SEGMENTS TO TRANSLATE:\n"
        for i, segment in enumerate(batch):
            prompt += f"[{i+1}] {segment['source']}\n"
        
        logger.info(f"Created prompt with {len(batch)} segments")
        return prompt
    except Exception as e:
        logger.error(f"Error creating prompt: {str(e)}")
        return ""

# Function to get translations from AI
def get_ai_translation(api_provider, api_key, model, prompt, source_lang, target_lang):
    """Get translations from AI model using direct API calls"""
    logger.info(f"Sending request to {api_provider} API ({model})")
    import requests
    import json
    
    try:
        if api_provider == 'anthropic':
            # Direct API call to Anthropic
            headers = {
                "x-api-key": api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": model,
                "max_tokens": 4000,
                "temperature": 0.3,
                "system": (
                    "You are a professional translator specializing in technical documents. "
                    "Translate precisely while preserving all formatting, tags, and special characters. "
                    f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                    "Pay special attention to cultural nuances and linguistic patterns."
                ),
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code != 200:
                error_message = f"Anthropic API Error: Status {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
            
            result = response.json()
            logger.info("Received response from Anthropic API")
            return result["content"][0]["text"]
        
        else:  # OpenAI
            # Direct API call to OpenAI
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator specializing in technical documents. "
                            "Translate precisely while preserving all formatting, tags, and special characters. "
                            f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                            "Pay special attention to cultural nuances and linguistic patterns."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code != 200:
                error_message = f"OpenAI API Error: Status {response.status_code}, {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
            
            result = response.json()
            logger.info("Received response from OpenAI API")
            return result["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        error_message = f"Network error: {str(e)}"
        logger.error(error_message)
        raise Exception(error_message)
    except json.JSONDecodeError as e:
        error_message = f"JSON parsing error: {str(e)}"
        logger.error(error_message)
        raise Exception(error_message)
    except Exception as e:
        error_message = f"API Error: {str(e)}"
        logger.error(error_message)
        raise Exception(error_message)

# Function to parse AI response
def parse_ai_response(ai_response, batch):
    """Parse AI response to extract translations"""
    try:
        logger.info("Parsing AI response")
        translations = {}
        lines = ai_response.split('\n')
        
        for i, segment in enumerate(batch):
            segment_number = i + 1
            # Pattern to match different numbering formats
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
                if found:
                    break
            
            # If no format matched, look for segment after previous one
            if not found and i > 0:
                prev_id = batch[i-1]['id']
                if prev_id in translations:
                    # Find index of previous translation
                    for j, line in enumerate(lines):
                        if translations[prev_id] in line:
                            # Take next non-empty line
                            for k in range(j+1, len(lines)):
                                if lines[k].strip():
                                    translations[segment['id']] = lines[k].strip()
                                    found = True
                                    break
                            break
            
            if not found:
                logger.warning(f"Could not find translation for segment {segment_number} (ID: {segment['id']})")
        
        logger.info(f"Parsed {len(translations)} translations from AI response")
        return translations
    except Exception as e:
        logger.error(f"Error parsing AI response: {str(e)}")
        return {}

# Function to update XLIFF with translations
def update_xliff_with_translations(xliff_content, translations):
    """Update XLIFF file with translations"""
    try:
        logger.info(f"Updating XLIFF file with {len(translations)} translations")
        # Register namespaces
        ET.register_namespace('mq', 'MQXliff')
        ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
        
        # Parse XML
        tree = ET.ElementTree(ET.fromstring(xliff_content))
        root = tree.getroot()
        
        # Get namespace
        ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
        
        # Update segments with translations
        updated_count = 0
        for trans_unit in root.findall('.//x:trans-unit', ns):
            segment_id = trans_unit.get('id')
            
            if segment_id in translations:
                # Find or create target element
                target = trans_unit.find('.//x:target', ns)
                if target is None:
                    # Create new target element
                    source = trans_unit.find('.//x:source', ns)
                    target = ET.SubElement(trans_unit, '{urn:oasis:names:tc:xliff:document:1.2}target')
                    if source is not None and source.get('{http://www.w3.org/XML/1998/namespace}space'):
                        target.set('{http://www.w3.org/XML/1998/namespace}space', 
                                  source.get('{http://www.w3.org/XML/1998/namespace}space'))
                
                # Update target text
                target.text = translations[segment_id]
                
                # Update status if it exists
                if '{MQXliff}status' in trans_unit.attrib:
                    trans_unit.set('{MQXliff}status', 'Translated')
                
                # Update timestamp if it exists
                if '{MQXliff}lastchangedtimestamp' in trans_unit.attrib:
                    from datetime import datetime
                    trans_unit.set('{MQXliff}lastchangedtimestamp', 
                                  datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
                
                updated_count += 1
        
        # Convert to string with proper formatting
        xml_string = ET.tostring(root, encoding='utf-8', method='xml')
        
        # Use minidom for pretty printing
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Fix XML declaration
        if not pretty_xml.startswith('<?xml version="1.0" encoding="UTF-8"?>'):
            pretty_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty_xml.split('\n', 1)[1]
        
        logger.info(f"Successfully updated {updated_count} segments in XLIFF")
        return pretty_xml, updated_count
    except Exception as e:
        logger.error(f"Error updating XLIFF: {str(e)}")
        return None, 0

# Function to save translations as text file
def save_translations_as_text(segments, translations, filename):
    """Save translations as a text file when XLIFF update fails"""
    try:
        logger.info("Saving translations as text file")
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'translated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_translations_{timestamp}.txt"
        text_path = os.path.join(output_dir, output_filename)
        
        # Create text content
        text_output = "# Translation Results\n\n"
        for seg_id, translation in translations.items():
            # Find original segment
            original = ""
            for segment in segments:
                if segment['id'] == seg_id:
                    original = segment['source']
                    break
            
            text_output += f"ID: {seg_id}\n"
            text_output += f"Source: {original}\n"
            text_output += f"Target: {translation}\n\n"
        
        # Write text file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_output)
        
        logger.info(f"Saved translations as text file: {text_path}")
        return text_path
    except Exception as e:
        logger.error(f"Error saving translations as text: {str(e)}")
        return None

# Function to save XLIFF file
def save_translated_xliff(xliff_content, filename):
    """Save translated XLIFF file"""
    try:
        logger.info("Saving translated XLIFF file")
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'translated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_translated_{timestamp}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save file
        with open(output_path, 'wb') as f:
            if isinstance(xliff_content, str):
                f.write(xliff_content.encode('utf-8'))
            else:
                f.write(xliff_content)
        
        logger.info(f"Saved translated XLIFF file: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving translated XLIFF file: {str(e)}")
        return None

# Display log file information
st.sidebar.title("Log Information")
if 'log_filepath' in locals():
    st.sidebar.info(f"Log file: {os.path.basename(log_filepath)}")
    st.sidebar.info(f"Location: {log_dir}")
    
    # Add download button for log file
    if os.path.exists(log_filepath):
        with open(log_filepath, 'r') as log_file:
            log_content = log_file.read()
            st.sidebar.download_button(
                label="Download Log File",
                data=log_content,
                file_name=os.path.basename(log_filepath),
                mime="text/plain"
            )

# Main application
def main():
    st.title("MemoQ Translation Assistant")
    st.markdown("Process MemoQ XLIFF files with Translation Memory, Terminology, and AI assistance")
    
    # Debug toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Initialize session state
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
        st.session_state.processing_complete = False
        st.session_state.progress = 0
        st.session_state.current_batch = 0
        st.session_state.total_batches = 0
        st.session_state.logs = []
        st.session_state.translated_file_path = None
        st.session_state.xliff_content = None
        st.session_state.batch_results = []
        st.session_state.backup_path = None
        st.session_state.override_source_lang = "Auto-detect (from XLIFF)"
        st.session_state.override_target_lang = "Auto-detect (from XLIFF)"
    
    # Create tab layout
    tab1, tab2, tab3 = st.tabs(["File Uploads & Settings", "Processing", "Results"])
    
    # Tab 1: File Uploads & Settings
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File Uploads")
            
            xliff_file = st.file_uploader("MemoQ XLIFF File", type=["memoqxliff", "xliff", "mqxliff"], 
                                         help="Upload a MemoQ XLIFF file to translate")
            
            tmx_file = st.file_uploader("Translation Memory (TMX)", type=["tmx"], 
                                       help="Upload a TMX file with translation memory entries")
            
            csv_file = st.file_uploader("Terminology (CSV)", type=["csv"], 
                                       help="Upload a CSV file with terminology entries")
            
            prompt_file = st.file_uploader("Custom Prompt (TXT)", type=["txt"], 
                                          help="Upload a text file with custom prompt for the AI")
            
            # Language Settings
            st.subheader("Language Settings (Optional)")
            
            # Get language options
            language_options = get_language_options()
            
            # Source language override
            st.session_state.override_source_lang = st.selectbox(
                "Override Source Language",
                ["Auto-detect (from XLIFF)"] + language_options,
                index=0,
                help="Override the source language specified in the XLIFF file"
            )
            
            # Target language override
            st.session_state.override_target_lang = st.selectbox(
                "Override Target Language", 
                ["Auto-detect (from XLIFF)"] + language_options,
                index=0,
                help="Override the target language specified in the XLIFF file"
            )
        
        with col2:
            st.subheader("Translation Settings")
            
            api_provider = st.selectbox("AI Provider", 
                                       ["anthropic", "openai"], 
                                       help="Select the AI provider to use for translation")
            
            api_key = st.text_input("API Key", 
                                   type="password", 
                                   help="Enter your API key for the selected provider")
            
            if api_provider == "anthropic":
                model = st.selectbox("Model", 
                                    ["claude-3-opus-20240229", "claude-3-sonnet-20240229", 
                                     "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620",
                                     "claude-3-7-sonnet-20250219"])
            else:
                model = st.selectbox("Model", 
                                    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
            
            batch_size = st.slider("Batch Size", 
                                  min_value=5, max_value=50, value=10, 
                                  help="Number of segments to process in each batch")
            
            match_threshold = st.slider("TM Match Threshold (%)", 
                                       min_value=60, max_value=100, value=75, 
                                       help="Minimum similarity percentage for TM matches")
            
            custom_prompt_text = st.text_area("Additional prompt instructions (optional)", 
                                             height=100, 
                                             help="Add custom instructions to the AI prompt")
            
            # Action button
            start_button = st.button("Start Processing", disabled=st.session_state.processing_started)
        
        # Progress bar in main tab
        if st.session_state.processing_started:
            st.subheader("Progress")
            progress_bar = st.progress(st.session_state.progress)
            status = st.empty()
            if st.session_state.total_batches > 0:
                status.text(f"Batch {st.session_state.current_batch}/{st.session_state.total_batches} ({int(st.session_state.progress * 100)}%)")
            else:
                status.text("Preparing...")
    
    # Tab 2: Processing
    with tab2:
        st.subheader("Processing Status")
        
        if st.session_state.processing_started:
            progress_bar = st.progress(st.session_state.progress)
            status = st.empty()
            if st.session_state.total_batches > 0:
                status.text(f"Processing batch {st.session_state.current_batch} of {st.session_state.total_batches}")
            else:
                status.text("Preparing batches...")
            
            if not st.session_state.processing_complete:
                with st.spinner('Processing... Please wait'):
                    st.info("Processing your file. This may take several minutes.")
        
        # Process log
        st.subheader("Process Log")
        log_container = st.container(height=300)
        with log_container:
            for log in st.session_state.logs:
                if log['type'] == 'info':
                    st.text(f"[{log['timestamp']}] {log['message']}")
                elif log['type'] == 'error':
                    st.error(log['message'])
                elif log['type'] == 'warning':
                    st.warning(log['message'])
                elif log['type'] == 'success':
                    st.success(log['message'])
        
        # Debug info
        if debug_mode:
            st.subheader("Debug Information")
            if st.session_state.xliff_content:
                with st.expander("XLIFF Preview"):
                    st.code(st.session_state.xliff_content[:500])
    
    # Tab 3: Results
    with tab3:
        st.subheader("Translation Results")
        
        if st.session_state.processing_complete:
            st.success("Translation completed successfully!")
            
            if st.session_state.translated_file_path:
                with open(st.session_state.translated_file_path, 'rb') as f:
                    st.download_button(
                        label="Download Translated File",
                        data=f,
                        file_name=os.path.basename(st.session_state.translated_file_path),
                        mime="application/xliff+xml"
                    )
            
            # Batch results
            if st.session_state.batch_results:
                st.subheader("Batch Summary")
                
                for i, result in enumerate(st.session_state.batch_results):
                    with st.expander(f"Batch {i+1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processed", result.get('segments_processed', 0))
                        with col2:
                            st.metric("Translated", result.get('translations_received', 0))
                        
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
        
        # Reset button
        if st.session_state.processing_started:
            if st.button("Start New Translation"):
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.progress = 0
                st.session_state.current_batch = 0
                st.session_state.total_batches = 0
                st.session_state.logs = []
                st.session_state.translated_file_path = None
                st.session_state.xliff_content = None
                st.session_state.batch_results = []
                # Keep the language override settings
                st.rerun()
    
    # Processing logic
    if start_button and not st.session_state.processing_started:
        # Validate inputs
        if not xliff_file:
            st.error("Please upload a MemoQ XLIFF file")
            logger.error("MemoQ XLIFF file not uploaded")
            return
        if not tmx_file:
            st.error("Please upload a TMX file")
            logger.error("TMX file not uploaded")
            return
        if not csv_file:
            st.error("Please upload a terminology CSV file")
            logger.error("Terminology CSV file not uploaded")
            return
        if not api_key:
            st.error("Please enter an API key")
            logger.error("API key not provided")
            return
        
        # Set processing state
        st.session_state.processing_started = True
        st.session_state.processing_complete = False
        st.session_state.logs = []
        st.session_state.batch_results = []
        st.session_state.progress = 0
        
        # Log processing start
        logger.info("=" * 50)
        logger.info(f"Starting translation process")
        logger.info(f"Files: XLIFF={xliff_file.name}, TMX={tmx_file.name}, CSV={csv_file.name}")
        logger.info(f"Settings: Provider={api_provider}, Model={model}, Batch Size={batch_size}, Match Threshold={match_threshold}%")
        
        # Log language override settings
        if st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
            logger.info(f"Source language override: {st.session_state.override_source_lang}")
        if st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
            logger.info(f"Target language override: {st.session_state.override_target_lang}")
            
        logger.info("=" * 50)
        
        try:
            # Read files with better encoding handling
            try:
                # Try to detect encoding and read XLIFF file
                xliff_bytes = xliff_file.read()
                
                # Create backup of original file
                backup_path = create_backup(xliff_file, xliff_file.name)
                st.session_state.backup_path = backup_path
                
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Created backup of XLIFF file at: {backup_path}",
                    'type': 'info',
                    'timestamp': timestamp
                })
                
                # Display backup information in sidebar
                if backup_path:
                    st.sidebar.success(f"Backup created: {os.path.basename(backup_path)}")
                    with open(backup_path, 'rb') as backup_file:
                        st.sidebar.download_button(
                            label="Download Backup File",
                            data=backup_file,
                            file_name=os.path.basename(backup_path),
                            mime="application/octet-stream"
                        )
                
                # Reset file pointer
                xliff_file.seek(0)
                
                # Try to decode with different encodings
                try:
                    xliff_content = xliff_bytes.decode('utf-8')
                    logger.info("Successfully decoded XLIFF file using UTF-8 encoding")
                except UnicodeDecodeError:
                    logger.warning("Failed to decode XLIFF with UTF-8, trying UTF-16")
                    try:
                        xliff_content = xliff_bytes.decode('utf-16')
                        logger.info("Successfully decoded XLIFF file using UTF-16 encoding")
                    except UnicodeDecodeError:
                        if xliff_bytes.startswith(b'\xff\xfe') or xliff_bytes.startswith(b'\xfe\xff'):
                            xliff_content = xliff_bytes.decode('utf-16')
                            logger.info("Successfully decoded XLIFF file using UTF-16 with BOM")
                        else:
                            xliff_content = xliff_bytes.decode('latin-1')
                            msg = "Could not determine the correct encoding for the XLIFF file. Using Latin-1 encoding as fallback."
                            logger.warning(msg)
                            st.warning(msg)
                
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Successfully loaded XLIFF file",
                    'type': 'info',
                    'timestamp': timestamp
                })
                
                # Read TMX file with similar handling
                tmx_bytes = tmx_file.read()
                try:
                    tmx_content = tmx_bytes.decode('utf-8')
                    logger.info("Successfully decoded TMX file using UTF-8 encoding")
                except UnicodeDecodeError:
                    logger.warning("Failed to decode TMX with UTF-8, trying UTF-16")
                    try:
                        tmx_content = tmx_bytes.decode('utf-16')
                        logger.info("Successfully decoded TMX file using UTF-16 encoding")
                    except UnicodeDecodeError:
                        if tmx_bytes.startswith(b'\xff\xfe') or tmx_bytes.startswith(b'\xfe\xff'):
                            tmx_content = tmx_bytes.decode('utf-16')
                            logger.info("Successfully decoded TMX file using UTF-16 with BOM")
                        else:
                            tmx_content = tmx_bytes.decode('latin-1')
                            msg = "Could not determine the correct encoding for the TMX file. Using Latin-1 encoding as fallback."
                            logger.warning(msg)
                            st.warning(msg)
                
                # Read CSV file with similar handling
                csv_bytes = csv_file.read()
                try:
                    csv_content = csv_bytes.decode('utf-8')
                    logger.info("Successfully decoded CSV file using UTF-8 encoding")
                except UnicodeDecodeError:
                    logger.warning("Failed to decode CSV with UTF-8, trying UTF-16")
                    try:
                        csv_content = csv_bytes.decode('utf-16')
                        logger.info("Successfully decoded CSV file using UTF-16 encoding")
                    except UnicodeDecodeError:
                        if csv_bytes.startswith(b'\xff\xfe') or csv_bytes.startswith(b'\xfe\xff'):
                            csv_content = csv_bytes.decode('utf-16')
                            logger.info("Successfully decoded CSV file using UTF-16 with BOM")
                        else:
                            csv_content = csv_bytes.decode('latin-1')
                            msg = "Could not determine the correct encoding for the CSV file. Using Latin-1 encoding as fallback."
                            logger.warning(msg)
                            st.warning(msg)
                
                # Get prompt template
                prompt_template = ""
                if prompt_file:
                    try:
                        prompt_bytes = prompt_file.read()
                        try:
                            prompt_template = prompt_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                prompt_template = prompt_bytes.decode('utf-16')
                            except UnicodeDecodeError:
                                prompt_template = prompt_bytes.decode('latin-1')
                        logger.info("Successfully loaded prompt template")
                    except Exception as prompt_error:
                        logger.warning(f"Error reading prompt file: {str(prompt_error)}")
                
                if custom_prompt_text:
                    prompt_template += "\n\n" + custom_prompt_text if prompt_template else custom_prompt_text
                    logger.info("Added custom prompt text")
            
            except Exception as file_error:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Error reading input files: {str(file_error)}",
                    'type': 'error',
                    'timestamp': timestamp
                })
                logger.error(f"Error reading input files: {str(file_error)}")
                st.session_state.processing_complete = True
                return
            
            # Store XLIFF content for later processing
            st.session_state.xliff_content = xliff_content
            
            # Add log
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Starting translation process",
                'type': 'info',
                'timestamp': timestamp
            })
            
            # Extract segments from XLIFF
            source_lang, target_lang, document_name, segments = extract_translatable_segments(xliff_content)
            
            if not segments:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': "No translatable segments found in the XLIFF file",
                    'type': 'error',
                    'timestamp': timestamp
                })
                logger.error("No translatable segments found in the XLIFF file")
                st.session_state.processing_complete = True
                return
            
            # Log language information
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            source_lang_name = get_language_name(source_lang)
            target_lang_name = get_language_name(target_lang)
            
            st.session_state.logs.append({
                'message': f"Translation: {source_lang_name} ({source_lang}) -> {target_lang_name} ({target_lang})",
                'type': 'info',
                'timestamp': timestamp
            })
            
            # Prepare batches
            batches = []
            for i in range(0, len(segments), batch_size):
                batches.append(segments[i:i+batch_size])
            
            st.session_state.total_batches = len(batches)
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Found {len(segments)} segments to translate in {len(batches)} batches",
                'type': 'info',
                'timestamp': timestamp
            })
            logger.info(f"Found {len(segments)} segments to translate in {len(batches)} batches")
            
            # Process batches
            all_translations = {}
            
            # Setup progress display
            progress_placeholder = st.empty()
            
            # Process each batch
            for batch_index, batch in enumerate(batches):
                # Update progress
                batch_progress = (batch_index) / len(batches)
                st.session_state.current_batch = batch_index + 1
                st.session_state.progress = batch_progress
                
                # Update progress display
                progress_placeholder.progress(batch_progress)
                
                # Process current batch
                batch_result = {'batch_index': batch_index}
                
                try:
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    msg = f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} segments)"
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    logger.info(msg)
                    
                    # Find TM matches
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    msg = f"Finding TM matches with threshold {match_threshold}%"
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    tm_matches = extract_tm_matches(tmx_content, source_lang, target_lang, batch, match_threshold)
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    msg = f"Found {len(tm_matches)} TM matches above threshold"
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Find terminology matches
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': "Identifying terminology matches",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    term_matches = extract_terminology(csv_content, batch)
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    msg = f"Found {len(term_matches)} relevant terminology entries"
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Create prompt
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': "Creating prompt with custom text, TM matches and terminology",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    prompt = create_ai_prompt(
                        prompt_template, source_lang, target_lang, document_name, 
                        batch, tm_matches, term_matches
                    )
                    
                    # Get translations from AI
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    msg = f"Sending request to {api_provider} API ({model})"
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    ai_response = get_ai_translation(
                        api_provider, api_key, model, prompt, source_lang, target_lang
                    )
                    
                    if not ai_response:
                        error_msg = "Failed to get translation from API"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': "Received translation response",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Parse AI response
                    translations = parse_ai_response(ai_response, batch)
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    msg = f"Parsed {len(translations)} translations from AI response"
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Add to all translations
                    all_translations.update(translations)
                    
                    # Update batch result
                    batch_result['segments_processed'] = len(batch)
                    batch_result['translations_received'] = len(translations)
                    
                    # Update progress at end of batch
                    batch_progress = (batch_index + 1) / len(batches)
                    st.session_state.current_batch = batch_index + 1
                    st.session_state.progress = batch_progress
                    
                    # Update progress display
                    progress_placeholder.progress(batch_progress)
                    
                    # Add batch result
                    st.session_state.batch_results.append(batch_result)
                    
                    # Small delay for UI refresh
                    time.sleep(0.1)
                    
                except Exception as e:
                    error_msg = f"Error processing batch {batch_index + 1}: {str(e)}"
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': error_msg,
                        'type': 'error',
                        'timestamp': timestamp
                    })
                    logger.error(error_msg)
                    batch_result['error'] = str(e)
                    st.session_state.batch_results.append(batch_result)
            
            # Update XLIFF with translations
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            msg = f"Updating XLIFF file with {len(all_translations)} translations"
            st.session_state.logs.append({
                'message': msg,
                'type': 'info',
                'timestamp': timestamp
            })
            logger.info(msg)
            
            # Try to update XLIFF
            try:
                updated_xliff, updated_count = update_xliff_with_translations(st.session_state.xliff_content, all_translations)
                
                if not updated_xliff:
                    raise Exception("Failed to update XLIFF file")
                
                # Save translated XLIFF
                final_path = save_translated_xliff(updated_xliff, xliff_file.name)
                
                if not final_path:
                    raise Exception("Failed to save translated XLIFF file")
                
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                msg = f"Updated {updated_count} segments in the XLIFF file"
                st.session_state.logs.append({
                    'message': msg,
                    'type': 'success',
                    'timestamp': timestamp
                })
                
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                msg = f"Saved translated XLIFF to {os.path.basename(final_path)}"
                st.session_state.logs.append({
                    'message': msg,
                    'type': 'success',
                    'timestamp': timestamp
                })
                
                # Store translated file path for download
                st.session_state.translated_file_path = final_path
                
            except Exception as update_error:
                # Log error
                error_msg = f"Error updating or saving XLIFF: {str(update_error)}"
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': error_msg,
                    'type': 'error',
                    'timestamp': timestamp
                })
                logger.error(error_msg)
                
                # Try to save as text file instead
                text_path = save_translations_as_text(segments, all_translations, xliff_file.name)
                
                if text_path:
                    msg = f"Saved translations as text file instead: {os.path.basename(text_path)}"
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': msg,
                        'type': 'warning',
                        'timestamp': timestamp
                    })
                    logger.warning(msg)
                    
                    # Store text file path for download
                    st.session_state.translated_file_path = text_path
            
            # Log completion
            logger.info("=" * 50)
            logger.info(f"Translation process completed")
            logger.info(f"Segments processed: {len(segments)}")
            logger.info(f"Segments translated: {len(all_translations)}")
            logger.info(f"Translated file: {st.session_state.translated_file_path}")
            logger.info("=" * 50)
            
            # Mark processing as complete
            st.session_state.processing_complete = True
            st.session_state.progress = 1.0
            
            # Show completion message
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': "Translation process completed successfully",
                'type': 'success',
                'timestamp': timestamp
            })
            
            # Refresh UI
            st.rerun()
            
        except Exception as e:
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Error: {str(e)}",
                'type': 'error',
                'timestamp': timestamp
            })
            logger.error(f"Error: {str(e)}")
            st.session_state.processing_complete = True
            st.rerun()

if __name__ == "__main__":
    main()