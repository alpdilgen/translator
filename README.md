MemoQ Translation Assistant
A Streamlit application for translating MemoQ XLIFF files using AI, Translation Memories, and Terminology lists.
Features

Process MemoQ XLIFF files (.memoqxliff, .xliff, .mqxliff)
Leverage existing Translation Memory (TMX) files
Apply terminology from CSV files
Use AI models (OpenAI or Anthropic Claude) for translation
Batch processing for efficient handling of large documents
Real-time progress tracking and detailed logs
Output properly formatted MemoQ XLIFF files

How It Works

Upload your MemoQ XLIFF file containing source text to be translated
Upload a TMX file with existing translations for reference
Upload a CSV file with terminology that must be used in translation
Optionally, upload a custom prompt file or add custom instructions
Configure AI provider, model, batch size, and match threshold
The application will:

Create a backup of your XLIFF file
Process segments in batches
Find TM matches based on your threshold
Identify terminology matches
Generate optimized AI prompts
Translate using the selected AI model
Update your XLIFF file with translations


Download the translated XLIFF file when complete
