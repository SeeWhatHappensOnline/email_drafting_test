import streamlit as st
import pandas as pd
import time
import google.generativeai as genai

st.set_page_config(
    page_title="CSV LLM Processor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CSV LLM Processor")
st.markdown("Upload a CSV, define a prompt with `{{message}}`, and get LLM responses for each row.")

# Configure Gemini with API key from secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please add your Google API key to `.streamlit/secrets.toml` as `GOOGLE_API_KEY`")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info("**Model:** gemini-2.5-flash-lite")
    
    # Rate limiting
    st.subheader("Rate Limiting")
    delay_between_calls = st.slider(
        "Delay between API calls (seconds)", 
        min_value=0.0, 
        max_value=5.0, 
        value=0.5, 
        step=0.1,
        help="Add delay to avoid rate limits"
    )
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Delimiter selection
    delimiter = st.selectbox("CSV Delimiter", [",", ";", "Auto-detect"], index=2)
    
    if uploaded_file is not None:
        # Determine delimiter
        if delimiter == "Auto-detect":
            # Read first line to detect delimiter
            first_line = uploaded_file.readline().decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            sep = ";" if first_line.count(";") > first_line.count(",") else ","
        else:
            sep = delimiter
        
        # Try different encodings
        try:
            df = pd.read_csv(uploaded_file, sep=sep)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1', sep=sep)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        st.success(f"Loaded {len(df)} rows")
        
        # Column selection
        columns = df.columns.tolist()
        message_column = st.selectbox(
            "Select the column containing messages",
            columns,
            index=columns.index("message") if "message" in columns else 0
        )
        
        # Preview
        st.markdown("**Preview (first 3 rows):**")
        st.dataframe(df.head(3), use_container_width=True)

with col2:
    st.subheader("✍️ Prompt Template")
    
    default_prompt = """You are a helpful assistant. 

Analyze the following message and provide a response:

{{message}}

Respond concisely and professionally."""
    
    prompt_template = st.text_area(
        "Enter your prompt (use {{message}} as placeholder)",
        value=default_prompt,
        height=300,
        help="The {{message}} placeholder will be replaced with the value from the selected column for each row"
    )
    
    if "{{message}}" not in prompt_template:
        st.warning("⚠️ Your prompt doesn't contain `{{message}}`. The message won't be inserted.")


# Function to call Gemini
def call_gemini(prompt: str, temperature: float) -> str:
    """Call Gemini 2.5 Flash Lite with the given prompt."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=temperature)
        )
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"


# Output column name
st.markdown("---")
output_column_name = st.text_input("Output column name", value="llm_response")

# Process button
if st.button("🚀 Process CSV", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    elif "{{message}}" not in prompt_template:
        st.error("Your prompt must contain the `{{message}}` placeholder.")
    else:
        # Process each row
        responses = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {total_rows}...")
            
            # Get the message value
            message_value = str(row[message_column]) if pd.notna(row[message_column]) else ""
            
            # Replace placeholder in prompt
            final_prompt = prompt_template.replace("{{message}}", message_value)
            
            # Call Gemini
            response = call_gemini(final_prompt, temperature)
            responses.append(response)
            
            # Rate limiting delay
            if delay_between_calls > 0 and idx < total_rows - 1:
                time.sleep(delay_between_calls)
        
        # Add responses to dataframe
        df[output_column_name] = responses
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Processing complete! Added {len(responses)} responses.")
        
        # Show results
        st.subheader("📊 Results")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv_output = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_output,
            file_name="processed_results.csv",
            mime="text/csv",
            type="primary"
        )

# Footer
st.markdown("---")
st.markdown("""
**Tips:**
- Use `{{message}}` in your prompt where you want the CSV column value inserted
- Adjust the delay between calls if you hit rate limits
- Lower temperature for more consistent outputs
""")
