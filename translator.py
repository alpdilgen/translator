# Function to get language name from code - Replace this with a more comprehensive version
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
        'vi': 'Vietnamese'
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
        'Vietnamese': 'vi'
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
        'Arabic', 'Chinese', 'Hindi', 'Japanese', 'Korean', 
        'Russian', 'Ukrainian', 'Hebrew', 'Thai', 'Vietnamese'
    ]
    
    # Combine all languages and sort alphabetically
    all_languages = sorted(eu_languages_plus + variants + other_languages)
    
    return all_languages

# Update the main function to include language selection options
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
        st.session_state.override_source_lang = None
        st.session_state.override_target_lang = None
    
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
            
            # Language Override Options
            st.subheader("Language Settings (Optional)")
            
            # Get language options
            language_options = get_language_options()
            
            # Source language override
            st.session_state.override_source_lang = st.selectbox(
                "Override Source Language",
                ["Auto-detect (from XLIFF)"] + language_options,
                help="Override the source language specified in the XLIFF file"
            )
            
            # Target language override
            st.session_state.override_target_lang = st.selectbox(
                "Override Target Language", 
                ["Auto-detect (from XLIFF)"] + language_options,
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
    
    # Rest of your existing code...
    # Tab 2: Processing
    with tab2:
        st.subheader("Processing Status")
        
        # Rest of your Tab 2 code...
    
    # Tab 3: Results
    with tab3:
        st.subheader("Translation Results")
        
        # Rest of your Tab 3 code...
    
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
        
        # Process files and start translation
        # ... rest of your existing processing code ...
        
        # Extract segments from XLIFF
        source_lang, target_lang, document_name, segments = extract_translatable_segments(xliff_content)
        
        # Handle language overrides
        if st.session_state.override_source_lang != "Auto-detect (from XLIFF)":
            source_lang = get_language_code(st.session_state.override_source_lang)
            logger.info(f"Source language overridden to: {source_lang}")
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Source language overridden to: {st.session_state.override_source_lang} ({source_lang})",
                'type': 'info',
                'timestamp': timestamp
            })
        
        if st.session_state.override_target_lang != "Auto-detect (from XLIFF)":
            target_lang = get_language_code(st.session_state.override_target_lang)
            logger.info(f"Target language overridden to: {target_lang}")
            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
            st.session_state.logs.append({
                'message': f"Target language overridden to: {st.session_state.override_target_lang} ({target_lang})",
                'type': 'info',
                'timestamp': timestamp
            })
        
        # Continue with the rest of your process...
        
        # ... rest of your existing code ...
