import streamlit as st
import os
import tempfile
import xml.etree.ElementTree as ET
import pandas as pd
import uuid
from anthropic import Anthropic
from openai import OpenAI
import xml.dom.minidom as minidom
import shutil
import time

st.set_page_config(
    page_title="MemoQ Translation Assistant",
    page_icon="📝",
    layout="wide"
)

# Utility functions
def extract_translatable_segments(xliff_content):
    """Extract translatable segments from XLIFF file"""
    try:
        # Register MemoQ namespace
        ET.register_namespace('mq', 'MQXliff')
        ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
        
        # Parse XML
        tree = ET.ElementTree(ET.fromstring(xliff_content))
        root = tree.getroot()
        
        # Get namespace
        ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
        
        # Extract file info
        file_nodes = root.findall('.//x:file', ns)
        if not file_nodes:
            return None, None, None, []
        
        file_node = file_nodes[0]
        source_lang = file_node.get('source-language', 'unknown')
        target_lang = file_node.get('target-language', 'unknown')
        document_name = file_node.get('original', 'Unknown document')
        
        # Extract segments
        segments = []
        for trans_unit in root.findall('.//x:trans-unit', ns):
            segment_id = trans_unit.get('id')
            status = trans_unit.get('{MQXliff}status', '')
            
            source_element = trans_unit.find('.//x:source', ns)
            target_element = trans_unit.find('.//x:target', ns)
            
            if source_element is not None:
                source_text = source_element.text or ""
                
                # Check if this segment needs translation
                has_target = (target_element is not None and 
                             target_element.text and 
                             target_element.text.strip())
                
                if status == 'NotStarted' or not has_target:
                    segments.append({
                        'id': segment_id,
                        'source': source_text,
                        'status': status
                    })
        
        return source_lang, target_lang, document_name, segments
    except Exception as e:
        st.error(f"Error parsing XLIFF file: {str(e)}")
        return None, None, None, []

def extract_tm_matches(tmx_content, source_lang, target_lang, source_segments, match_threshold):
    """Extract translation memory matches from TMX file"""
    try:
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
                # Calculate simple similarity for demonstration
                # In production, use better fuzzy matching algorithms
                for segment in source_segments:
                    similarity = calculate_similarity(segment['source'], source_segment)
                    if similarity >= match_threshold:
                        tm_matches.append({
                            'sourceText': source_segment,
                            'targetText': target_segment,
                            'similarity': similarity
                        })
        
        # Sort by similarity (highest first) and deduplicate
        tm_matches.sort(key=lambda x: x['similarity'], reverse=True)
        unique_matches = []
        seen = set()
        
        for match in tm_matches:
            key = f"{match['sourceText']}|{match['targetText']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
                if len(unique_matches) >= 5:  # Limit to 5 most relevant matches
                    break
        
        return unique_matches
    except Exception as e:
        st.error(f"Error processing TMX file: {str(e)}")
        return []

def calculate_similarity(text1, text2):
    """Simple similarity calculation (percentage-based)"""
    # In production, use better algorithms like Levenshtein distance
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

def extract_terminology(csv_content, source_segments):
    """Extract terminology matches from CSV file"""
    try:
        df = pd.read_csv(pd.StringIO(csv_content))
        
        if len(df.columns) < 2:
            st.warning("CSV file must have at least 2 columns (source term and target term)")
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
        
        return term_matches
    except Exception as e:
        st.error(f"Error processing terminology CSV: {str(e)}")
        return []

def create_ai_prompt(prompt_template, source_lang, target_lang, document_name, batch, tm_matches, term_matches):
    """Create a prompt for the AI model"""
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
    
    return prompt

def get_language_name(lang_code):
    """Get full language name from code"""
    language_names = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bg': 'Bulgarian',
        'ro': 'Romanian',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'fi': 'Finnish',
        'el': 'Greek',
        'hu': 'Hungarian',
        'no': 'Norwegian',
        'pl': 'Polish',
        'sv': 'Swedish',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'he': 'Hebrew',
        'th': 'Thai',
        'vi': 'Vietnamese'
    }
    
    # Handle country code variants
    base_lang = lang_code.split('-')[0]
    
    return language_names.get(base_lang) or language_names.get(lang_code) or lang_code

def get_ai_translation(api_provider, api_key, model, prompt, source_lang, target_lang):
    """Get translations from AI model"""
    try:
        if api_provider == 'anthropic':
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                system=(
                    "You are a professional translator specializing in technical documents. "
                    "Translate precisely while preserving all formatting, tags, and special characters. "
                    f"Ensure appropriate terminology consistency and grammatical correctness when translating from {source_lang} to {target_lang}. "
                    "Pay special attention to cultural nuances and linguistic patterns."
                ),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.content[0].text
        else:  # OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
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
                max_tokens=4000,
                temperature=0.3
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with AI API: {str(e)}")
        return None

def parse_ai_response(ai_response, batch):
    """Parse AI response to extract translations"""
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
            st.warning(f"Could not find translation for segment {segment_number} (ID: {segment['id']})")
    
    return translations

def update_xliff_with_translations(xliff_content, translations):
    """Update XLIFF file with translations"""
    try:
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
        
        return pretty_xml, updated_count
    except Exception as e:
        st.error(f"Error updating XLIFF: {str(e)}")
        return None, 0

# Main application
def main():
    st.title("MemoQ Translation Assistant")
    st.markdown("Process MemoQ XLIFF files with Translation Memory, Terminology, and AI assistance")
    
    # Add debugging toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Initialize session state for tracking
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
        st.session_state.tmp_dir = None
    
    # Create layout with tabs
    tab1, tab2, tab3 = st.tabs(["File Uploads & Settings", "Processing", "Results"])
    
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
            
        # Add overall progress indicator in main tab
        if st.session_state.processing_started and not st.session_state.processing_complete:
            st.subheader("Progress")
            progress_bar = st.progress(st.session_state.progress)
            status = st.empty()
            if st.session_state.total_batches > 0:
                status.text(f"Processing batch {st.session_state.current_batch} of {st.session_state.total_batches} ({int(st.session_state.progress * 100)}%)")
            else:
                status.text("Preparing to process...")
    
    with tab2:
        st.subheader("Processing Status")
        
        if st.session_state.processing_started and not st.session_state.processing_complete:
            progress_bar = st.progress(st.session_state.progress)
            status_text = st.empty()
            batch_status = st.empty()
            
            if st.session_state.total_batches > 0:
                status_text.markdown(f"**Status:** Processing batch {st.session_state.current_batch} of {st.session_state.total_batches}")
                batch_status.markdown(f"**Progress:** {int(st.session_state.progress * 100)}% complete")
            else:
                status_text.markdown("**Status:** Preparing batches...")
                batch_status.markdown("**Progress:** Initializing...")
            
            # Add spinner to show activity
            with st.spinner('Processing... Please wait'):
                st.info("The application is actively processing your file. This may take several minutes depending on file size and batch settings.")
        
        # Display log messages
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
        
        # Debug section only visible when debug mode is on
        if debug_mode:
            st.subheader("Debug Information")
            if st.session_state.xliff_content:
                with st.expander("XLIFF File Peek (first 500 chars)"):
                    st.code(st.session_state.xliff_content[:500])
            
            if 'error_traceback' in st.session_state:
                with st.expander("Error Traceback"):
                    st.code(st.session_state.error_traceback)
    
    with tab3:
        st.subheader("Translation Results")
        
        if st.session_state.processing_complete:
            st.success("Translation completed successfully!")
            
            if st.session_state.translated_file_path:
                with open(st.session_state.translated_file_path, 'rb') as f:
                    st.download_button(
                        label="Download Translated XLIFF",
                        data=f,
                        file_name=os.path.basename(st.session_state.translated_file_path),
                        mime="application/xliff+xml"
                    )
            
            # Display batch results
            if st.session_state.batch_results:
                st.subheader("Batch Processing Summary")
                
                for i, result in enumerate(st.session_state.batch_results):
                    with st.expander(f"Batch {i+1}: {result.get('segments_processed', 0)} segments"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processed", result.get('segments_processed', 0))
                        with col2:
                            st.metric("Translated", result.get('translations_received', 0))
                        
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
        
        # Reset button to start over
        if st.session_state.processing_started:
            if st.button("Start New Translation"):
                # Reset session state
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.progress = 0
                st.session_state.current_batch = 0
                st.session_state.total_batches = 0
                st.session_state.logs = []
                st.session_state.translated_file_path = None
                st.session_state.xliff_content = None
                st.session_state.batch_results = []
                # Rerun to refresh UI
                st.rerun()
    
    # Main processing logic
    if start_button and not st.session_state.processing_started:
        # Validate inputs
        if not xliff_file:
            st.error("Please upload a MemoQ XLIFF file")
            return
        if not tmx_file:
            st.error("Please upload a TMX file")
            return
        if not csv_file:
            st.error("Please upload a terminology CSV file")
            return
        if not api_key:
            st.error("Please enter an API key")
            return
        
        # Set processing state
        st.session_state.processing_started = True
        st.session_state.logs = []
        st.session_state.batch_results = []
        
        try:
            # Read files with better encoding handling
            try:
                # Try to detect encoding and read XLIFF file
                xliff_bytes = xliff_file.read()
                
                # First, try UTF-8
                try:
                    xliff_content = xliff_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # Try UTF-16
                    try:
                        xliff_content = xliff_bytes.decode('utf-16')
                    except UnicodeDecodeError:
                        # Try UTF-16 with BOM detection
                        if xliff_bytes.startswith(b'\xff\xfe') or xliff_bytes.startswith(b'\xfe\xff'):
                            xliff_content = xliff_bytes.decode('utf-16')
                        else:
                            # Try Latin-1 (should always work but might give wrong characters)
                            xliff_content = xliff_bytes.decode('latin-1')
                            st.warning("Could not determine the correct encoding for the XLIFF file. Using Latin-1 encoding as fallback, which may result in incorrect characters.")
                
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
                except UnicodeDecodeError:
                    try:
                        tmx_content = tmx_bytes.decode('utf-16')
                    except UnicodeDecodeError:
                        if tmx_bytes.startswith(b'\xff\xfe') or tmx_bytes.startswith(b'\xfe\xff'):
                            tmx_content = tmx_bytes.decode('utf-16')
                        else:
                            tmx_content = tmx_bytes.decode('latin-1')
                            st.warning("Could not determine the correct encoding for the TMX file. Using Latin-1 encoding as fallback.")
                
                # Read CSV file with similar handling
                csv_bytes = csv_file.read()
                try:
                    csv_content = csv_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        csv_content = csv_bytes.decode('utf-16')
                    except UnicodeDecodeError:
                        if csv_bytes.startswith(b'\xff\xfe') or csv_bytes.startswith(b'\xfe\xff'):
                            csv_content = csv_bytes.decode('utf-16')
                        else:
                            csv_content = csv_bytes.decode('latin-1')
                            st.warning("Could not determine the correct encoding for the CSV file. Using Latin-1 encoding as fallback.")
                
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
                    except Exception as prompt_error:
                        st.warning(f"Error reading prompt file: {str(prompt_error)}")
                
                if custom_prompt_text:
                    prompt_template += "\n\n" + custom_prompt_text if prompt_template else custom_prompt_text
            
            except Exception as file_error:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Error reading input files: {str(file_error)}",
                    'type': 'error',
                    'timestamp': timestamp
                })
                import traceback
                st.session_state.error_traceback = traceback.format_exc()
                st.error(f"Failed to read input files: {str(file_error)}")
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
            
            # Create backup with binary mode
            if st.session_state.tmp_dir is None:
                st.session_state.tmp_dir = tempfile.mkdtemp()
            backup_path = os.path.join(st.session_state.tmp_dir, f"{xliff_file.name}.backup")
            with open(backup_path, 'wb') as f:
                xliff_file.seek(0)
                shutil.copyfileobj(xliff_file, f)
                xliff_file.seek(0)  # Reset file pointer after reading
            
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Created backup of XLIFF file",
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
                st.session_state.processing_complete = True
                return
            
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
            
            # Create translated file path
            output_path = os.path.join(st.session_state.tmp_dir, f"{os.path.splitext(xliff_file.name)[0]}_translated{os.path.splitext(xliff_file.name)[1]}")
            
            # Process each batch
            all_translations = {}
            
            # Set up progress display
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Process batches
            for batch_index, batch in enumerate(batches):
                st.session_state.current_batch = batch_index + 1
                st.session_state.progress = batch_index / len(batches)
                
                # Update progress display
                progress_placeholder.progress(st.session_state.progress)
                if st.session_state.total_batches > 0:
                    status_placeholder.markdown(f"**Processing:** Batch {st.session_state.current_batch}/{st.session_state.total_batches} ({int(st.session_state.progress * 100)}%)")
                
                # Process the current batch
                batch_result = {'batch_index': batch_index}
                
                try:
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} segments)",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Find TM matches for this batch
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Finding TM matches with threshold {match_threshold}%",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    tm_matches = extract_tm_matches(tmx_content, source_lang, target_lang, batch, match_threshold)
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Found {len(tm_matches)} TM matches above threshold",
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
                    st.session_state.logs.append({
                        'message': f"Found {len(term_matches)} relevant terminology entries",
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
                    st.session_state.logs.append({
                        'message': f"Sending request to {api_provider} API ({model})",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    ai_response = get_ai_translation(
                        api_provider, api_key, model, prompt, source_lang, target_lang
                    )
                    
                    if not ai_response:
                        raise Exception("Failed to get translation from API")
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': "Received translation response",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Parse AI response
                    translations = parse_ai_response(ai_response, batch)
                    
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Parsed {len(translations)} translations from AI response",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Add to all translations
                    all_translations.update(translations)
                    
                    # Update batch result
                    batch_result['segments_processed'] = len(batch)
                    batch_result['translations_received'] = len(translations)
                    
                    # Update progress at the end of batch processing
                    st.session_state.current_batch = batch_index + 1
                    st.session_state.progress = (batch_index + 1) / len(batches)
                    
                    # Update progress display
                    progress_placeholder.progress(st.session_state.progress)
                    status_placeholder.markdown(f"**Processing:** Batch {st.session_state.current_batch}/{st.session_state.total_batches} ({int(st.session_state.progress * 100)}%)")
                    
                    # Add batch result
                    st.session_state.batch_results.append(batch_result)
                    
                    # Add a small sleep to let UI refresh
                    time.sleep(0.1)
                    
                except Exception as e:
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Error processing batch {batch_index + 1}: {str(e)}",
                        'type': 'error',
                        'timestamp': timestamp
                    })
                    batch_result['error'] = str(e)
                    st.session_state.batch_results.append(batch_result)
            
            # Update XLIFF with all translations
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Updating XLIFF file with {len(all_translations)} translations",
                'type': 'info',
                'timestamp': timestamp
            })
            
            updated_xliff, updated_count = update_xliff_with_translations(st.session_state.xliff_content, all_translations)
            
            if not updated_xliff:
                raise Exception("Failed to update XLIFF file")
            
            # Save final file in binary mode
            with open(output_path, 'wb') as f:
                if isinstance(updated_xliff, str):
                    f.write(updated_xliff.encode('utf-8'))
                else:
                    f.write(updated_xliff)
            
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Updated {updated_count} segments in the XLIFF file",
                'type': 'success',
                'timestamp': timestamp
            })
            
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Saved translated XLIFF to {os.path.basename(output_path)}",
                'type': 'success',
                'timestamp': timestamp
            })
            
            # Store translated file path for download
            st.session_state.translated_file_path = output_path
            
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
            
            # Switch to results tab and mark completion
            st.session_state.processing_complete = True
            st.session_state.progress = 1.0
            st.rerun()
            
        except Exception as e:
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Error: {str(e)}",
                'type': 'error',
                'timestamp': timestamp
            })
            st.session_state.processing_complete = True
            
            # Add traceback for debugging
            import traceback
            st.session_state.error_traceback = traceback.format_exc()

if __name__ == "__main__":
    main()
