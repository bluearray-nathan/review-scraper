import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Review AI Analyst", page_icon="🧠", layout="wide")

# --- DATA DICTIONARIES ---
COUNTRY_CODES = {
    "United Kingdom": "gb",
    "United States": "us",
    "Canada": "ca",
    "Australia": "au",
    "Germany": "de",
    "France": "fr",
    "Spain": "es",
    "Italy": "it",
    "India": "in",
    "Brazil": "br"
}

LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi"
}

# --- AUTHENTICATION ---
try:
    GENAI_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ Secrets not found. Please create a .streamlit/secrets.toml file.")
    st.stop()
except KeyError:
    st.error("⚠️ GEMINI_API_KEY not found in secrets.")
    st.stop()

# --- HELPER FUNCTIONS ---

def get_reviews(place_id, api_key, country_code, lang_code, max_pages=3):
    """
    Pulls reviews directly using a Place ID.
    """
    reviews_data = []
    
    params = {
        "engine": "google_maps_reviews",
        "place_id": place_id,
        "api_key": api_key,
        "sort_by": "newestFirst",
        "gl": country_code,
        "hl": lang_code
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
    except Exception as e:
        st.error(f"Error fetching data from SerpApi: {e}")
        return pd.DataFrame()
    
    page_count = 0
    
    # Progress indicator
    status_text = st.empty()
    
    while page_count < max_pages:
        status_text.caption(f"Scraping page {page_count+1}...")
        
        # Check for error in result
        if "error" in results:
            st.error(f"SerpApi Error: {results['error']}")
            break
            
        new_reviews = results.get("reviews", [])
        
        for review in new_reviews:
            reviews_data.append({
                "rating": review.get("rating"),
                "text": review.get("snippet", ""),
                "date": review.get("date"),
                "author": review.get("user", {}).get("name")
            })
            
        # Pagination Logic
        if "serpapi_pagination" not in results: break
        if "next_page_token" not in results["serpapi_pagination"]: break
            
        page_count += 1
        params["next_page_token"] = results["serpapi_pagination"]["next_page_token"]
        
        # 'place_id' conflicts with 'next_page_token', so we remove it for subsequent pages
        if "place_id" in params: del params["place_id"]
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
    status_text.empty()
    return pd.DataFrame(reviews_data)

def analyze_with_gemini(data_dict, lang_name):
    """
    Analyzes reviews using Gemini to find 5-10 pain points.
    """
    genai.configure(api_key=GENAI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare text for prompt
    prompt_context = ""
    for name, df in data_dict.items():
        # Filter for negative reviews (1-3 stars)
        neg_reviews = df[df['rating'] <= 3]['text'].tolist()
        
        if not neg_reviews:
            prompt_context += f"\n\n--- REVIEWS FOR {name.upper()} ---\n(No negative reviews found)\n"
            continue

        # We take up to 50 reviews
        formatted_reviews = "\n".join([f"- {r}" for r in neg_reviews[:50]])
        prompt_context += f"\n\n--- REVIEWS FOR {name.upper()} ---\n{formatted_reviews}\n"

    # --- PROMPTS ---
    if len(data_dict) > 1:
        # COMPETITOR MODE
        prompt = f"""
        You are a Strategic Analyst. I have reviews for two companies.
        The reviews are in {lang_name}. Please provide your analysis in {lang_name}.
        
        {prompt_context}
        
        Please provide a comparison report in Markdown.
        1. **Top 5-10 Pain Points for {list(data_dict.keys())[0]}:** List the most critical recurring issues.
        2. **Top 5-10 Pain Points for {list(data_dict.keys())[1]}:** List the most critical recurring issues.
        3. **Comparison Verdict:** What is the main difference in why customers are unhappy?
        """
    else:
        # SINGLE MODE
        prompt = f"""
        You are a CX Analyst. Analyze these negative reviews.
        The reviews are in {lang_name}. Please provide your analysis in {lang_name}.
        
        {prompt_context}
        
        Identify the **Top 5 to 10** distinct customer pain points.
        Do not list fewer than 5 unless the data is extremely sparse.
        
        For each pain point, provide:
        1. **Title**: Short and punchy.
        2. **Frequency**: Estimate if this is High, Medium, or Low frequency.
        3. **Explanation**: A brief explanation of the issue.
        4. **Quote**: A direct quote from one of the reviews.
        """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- MAIN UI ---
st.title("📍 Voice of Customer: AI Analyzer")
st.markdown("Enter a **Google Place ID** to analyze customer pain points.")

# SIDEBAR SETTINGS
with st.sidebar:
    st.header("🔑 API Settings")
    user_api_key = st.text_input("Enter SerpApi Key", type="password", help="Get this from serpapi.com")
    
    st.divider()
    
    st.header("🌍 Region Settings")
    selected_country_name = st.selectbox("Search Location", list(COUNTRY_CODES.keys()), index=0)
    country_code = COUNTRY_CODES[selected_country_name]
    
    selected_lang_name = st.selectbox("Language", list(LANGUAGE_CODES.keys()), index=0)
    lang_code = LANGUAGE_CODES[selected_lang_name]

# HELPER TEXT
st.info("💡 Don't know the Place ID? Use the [Google Place ID Finder](https://developers.google.com/maps/documentation/places/web-service/place-id) to find it.")

# MAIN INPUTS
col1, col2 = st.columns(2)
with col1:
    target_id = st.text_input("Main Place ID (Required)", placeholder="e.g. ChIJ...")
    target_name = "Main Business" # Default label since we don't have the name yet
with col2:
    competitor_id = st.text_input("Competitor Place ID (Optional)", placeholder="e.g. ChIJ...")
    competitor_name = "Competitor"

# RUN BUTTON
if st.button("Analyze Reviews", type="primary"):
    if not user_api_key:
        st.warning("Please enter your SerpApi key in the sidebar.")
    elif not target_id:
        st.warning("Please enter a Place ID.")
    else:
        results_store = {}
        
        try:
            with st.status("🚀 Starting Analysis...", expanded=True) as status:
                
                # 1. SCRAPE MAIN BUSINESS
                status.write(f"Fetching reviews for ID: {target_id}...")
                df_target = get_reviews(target_id, user_api_key, country_code, lang_code)
                
                if not df_target.empty:
                    results_store[target_name] = df_target
                    status.write(f"✅ Loaded {len(df_target)} reviews.")
                else:
                    status.update(label="Failed to load reviews", state="error")
                    st.error("No reviews found. Check the Place ID.")
                    st.stop()

                # 2. SCRAPE COMPETITOR
                if competitor_id:
                    status.write(f"Fetching reviews for Competitor ID: {competitor_id}...")
                    df_comp = get_reviews(competitor_id, user_api_key, country_code, lang_code)
                    if not df_comp.empty:
                        results_store[competitor_name] = df_comp
                        status.write(f"✅ Loaded {len(df_comp)} competitor reviews.")
                
                status.update(label="Scraping Complete! Running AI Analysis...", state="complete")

            # 3. AI ANALYSIS
            if results_store:
                st.divider()
                st.subheader("🧠 Top Pain Points Report")
                with st.spinner("Generating insights..."):
                    analysis = analyze_with_gemini(results_store, selected_lang_name)
                st.markdown(analysis)
                
                # 4. RAW DATA
                with st.expander("View Raw Data"):
                    tab1, tab2 = st.tabs(["Main Business", "Competitor"])
                    with tab1:
                        st.dataframe(results_store[target_name])
                    with tab2:
                        if competitor_name in results_store:
                            st.dataframe(results_store[competitor_name])
                        else:
                            st.write("No competitor data.")

        except Exception as e:
            st.error(f"An error occurred: {e}")




