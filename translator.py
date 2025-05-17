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
    """Extract translatable segments from XLIFF file with debug details"""
    try:
        # Try first with ET
        try:
            # Register MemoQ namespace
            ET.register_namespace('mq', 'MQXliff')
            ET.register_namespace('', 'urn:oasis:names:tc:xliff:document:1.2')
            
            # Parse XML
            root = ET.fromstring(xliff_content)
            
            # Get namespace
            ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
            
            # Extract file info with debugging
            file_nodes = root.findall('.//file')
            if not file_nodes:
                file_nodes = root.findall('.//x:file', ns)
            
            if not file_nodes:
                # Try without namespaces
                all_elements = list(root.iter())
                file_count = sum(1 for elem in all_elements if elem.tag.endswith('file'))
                
                # Try direct attribute checking for all elements
                all_file_like = []
                for elem in all_elements:
                    if ('source-language' in elem.attrib or 
                        'target-language' in elem.attrib or 
                        'original' in elem.attrib):
                        all_file_like.append(elem)
                
                if len(all_file_like) > 0:
                    file_nodes = all_file_like
            
            if not file_nodes:
                return None, None, None, []
            
            file_node = file_nodes[0]
            
            # Try to get attributes in different ways
            source_lang = file_node.get('source-language')
            if source_lang is None:
                source_lang = file_node.attrib.get('source-language')
            
            target_lang = file_node.get('target-language')
            if target_lang is None:
                target_lang = file_node.attrib.get('target-language')
            
            document_name = file_node.get('original')
            if document_name is None:
                document_name = file_node.attrib.get('original')
            
            # Find all trans-unit nodes with different approaches
            trans_units = []
            
            # Try with namespaces
            trans_units = root.findall('.//x:trans-unit', ns)
            if not trans_units:
                # Try without namespaces
                trans_units = root.findall('.//trans-unit')
            
            if not trans_units:
                # Try iterating through all elements
                all_elements = list(root.iter())
                trans_units = [elem for elem in all_elements if elem.tag.endswith('trans-unit')]
            
            # Extract segments
            segments = []
            
            for trans_unit in trans_units:
                try:
                    # Get ID through various approaches
                    segment_id = trans_unit.get('id')
                    if segment_id is None:
                        segment_id = trans_unit.attrib.get('id')
                    
                    if segment_id is None:
                        # Generate a random ID if none exists
                        segment_id = str(uuid.uuid4())
                    
                    # Get status through various approaches
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
                    
                    # Try with namespaces
                    source_element = trans_unit.find('.//x:source', ns)
                    if source_element is None:
                        source_element = trans_unit.find('source')
                    
                    target_element = trans_unit.find('.//x:target', ns)
                    if target_element is None:
                        target_element = trans_unit.find('target')
                    
                    # If still not found, look through children directly
                    if source_element is None:
                        for child in trans_unit:
                            if child.tag.endswith('source'):
                                source_element = child
                                break
                    
                    if target_element is None:
                        for child in trans_unit:
                            if child.tag.endswith('target'):
                                target_element = child
                                break
                    
                    if source_element is not None:
                        # Extract text content through various approaches
                        source_text = source_element.text
                        if source_text is None or source_text.strip() == '':
                            # Try to get all text from children
                            source_text = ''.join(source_element.itertext())
                        
                        if source_text is None:
                            source_text = ""
                        
                        # Check if target is empty or missing
                        has_target = False
                        if target_element is not None:
                            target_text = target_element.text
                            if target_text is None:
                                target_text = ''.join(target_element.itertext())
                            
                            has_target = target_text is not None and target_text.strip() != ''
                        
                        # Add to segments - include ALL segments for testing purposes
                        segments.append({
                            'id': segment_id,
                            'source': source_text,
                            'status': status or 'Unknown'
                        })
                
                except Exception as segment_error:
                    continue
            
            return source_lang, target_lang, document_name, segments
        
        except Exception as et_error:
            # Try with minidom as fallback
            try:
                dom = minidom.parseString(xliff_content)
                root = dom.documentElement
                
                # Get file node
                file_nodes = dom.getElementsByTagName('file')
                
                if not file_nodes:
                    return None, None, None, []
                
                file_node = file_nodes[0]
                source_lang = file_node.getAttribute('source-language')
                target_lang = file_node.getAttribute('target-language')
                document_name = file_node.getAttribute('original')
                
                # Get all trans-units
                trans_units = dom.getElementsByTagName('trans-unit')
                
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
                            
                            # Add to segments - include ALL segments for testing
                            segments.append({
                                'id': segment_id,
                                'source': source_text,
                                'status': status or 'Unknown'
                            })
                    
                    except Exception as segment_error:
                        continue
                
                return source_lang, target_lang, document_name, segments
            
            except Exception as dom_error:
                return None, None, None, []
    
    except Exception as e:
        import traceback
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
            # Simple initialization without extra parameters
            try:
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
            except TypeError as type_error:
                # This handles the proxies error specifically
                if "proxies" in str(type_error):
                    st.error("Anthropic client initialization error with proxies. Trying alternative initialization.")
                    # Try again with a different approach
                    import os
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    import anthropic
                    client = anthropic.Anthropic()
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
                else:
                    raise
            except Exception as api_error:
                # Get detailed error message
                error_message = f"Anthropic API Error: {str(api_error)}"
                st.error(error_message)
                raise Exception(error_message)
        else:  # OpenAI
            client = OpenAI(api_key=api_key)
            try:
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
            except Exception as api_error:
                # Get detailed error message
                error_message = f"OpenAI API Error: {str(api_error)}"
                if hasattr(api_error, 'status_code'):
                    error_message += f" (Status: {api_error.status_code})"
                
                st.error(error_message)
                raise Exception(error_message)
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        raise Exception(f"API Error: {str(e)}")

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
        return None, 0

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
        st.session_state.tmp_dir = None
    
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
                st.rerun()
    
    # Processing logic
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
        st.session_state.processing_complete = False
        st.session_state.logs = []
        st.session_state.batch_results = []
        st.session_state.progress = 0
        
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
                        pass
                
                if custom_prompt_text:
                    prompt_template += "\n\n" + custom_prompt_text if prompt_template else custom_prompt_text
            
            except Exception as file_error:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Error reading input files: {str(file_error)}",
                    'type': 'error',
                    'timestamp': timestamp
                })
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
            
            # Create backup of XLIFF file
            try:
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
            except Exception as backup_error:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Error creating backup: {str(backup_error)}",
                    'type': 'error',
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
                    st.session_state.logs.append({
                        'message': f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} segments)",
                        'type': 'info',
                        'timestamp': timestamp
                    })
                    
                    # Find TM matches
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
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Error processing batch {batch_index + 1}: {str(e)}",
                        'type': 'error',
                        'timestamp': timestamp
                    })
                    batch_result['error'] = str(e)
                    st.session_state.batch_results.append(batch_result)
            
            # Update XLIFF with translations
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Updating XLIFF file with {len(all_translations)} translations",
                'type': 'info',
                'timestamp': timestamp
            })
            
            try:
                updated_xliff, updated_count = update_xliff_with_translations(st.session_state.xliff_content, all_translations)
                
                if not updated_xliff:
                    raise Exception("Failed to update XLIFF file")
                
                # Save final file
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
            except Exception as update_error:
                timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                st.session_state.logs.append({
                    'message': f"Error updating XLIFF: {str(update_error)}",
                    'type': 'error',
                    'timestamp': timestamp
                })
                
                # Save as text file instead
                try:
                    text_output = "# Translation Results\n\n"
                    for seg_id, translation in all_translations.items():
                        # Find original segment
                        original = ""
                        for segment in segments:
                            if segment['id'] == seg_id:
                                original = segment['source']
                                break
                        
                        text_output += f"ID: {seg_id}\n"
                        text_output += f"Source: {original}\n"
                        text_output += f"Target: {translation}\n\n"
                    
                    text_path = os.path.join(st.session_state.tmp_dir, "translations.txt")
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text_output)
                    
                    st.session_state.translated_file_path = text_path
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Saved translations as text file instead: {text_path}",
                        'type': 'warning',
                        'timestamp': timestamp
                    })
                except Exception as text_save_error:
                    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                    st.session_state.logs.append({
                        'message': f"Failed to save translations as text: {str(text_save_error)}",
                        'type': 'error',
                        'timestamp': timestamp
                    })
            
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
            st.session_state.processing_complete = True

if __name__ == "__main__":
    main()
