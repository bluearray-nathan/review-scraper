import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Review AI Analyst", page_icon="🧠", layout="wide")

# --- DATA DICTIONARIES (Add more as needed) ---
# Maps "Display Name" -> "Google Code"
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

def get_place_id(business_name, api_key, country_code, lang_code):
    """Finds the Place ID using specific Country (gl) and Language (hl) settings."""
    if not business_name: return None
    
    params = {
        "engine": "google_maps",
        "q": business_name,
        "type": "search",
        "api_key": api_key,
        "num": 1,
        "gl": country_code,   # <--- Forces search from this country
        "hl": lang_code       # <--- Sets interface language
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # 1. Try Local Results (The standard map pins)
    if "local_results" in results and len(results["local_results"]) > 0:
        return results["local_results"][0].get("place_id")
        
    # 2. Backup: Check if it found a specific "Place" directly
    if "place_results" in results:
        return results["place_results"].get("place_id")
        
    return None

def get_reviews(place_id, api_key, country_code, lang_code, max_pages=3):
    """Pulls reviews using the selected language/region settings."""
    reviews_data = []
    
    params = {
        "engine": "google_maps_reviews",
        "place_id": place_id,
        "api_key": api_key,
        "sort_by": "newestFirst",
        "gl": country_code,   # <--- Critical for getting local review sorting
        "hl": lang_code       # <--- Returns reviews in this language (if translated)
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    page_count = 0
    
    while page_count < max_pages:
        new_reviews = results.get("reviews", [])
        for review in new_reviews:
            reviews_data.append({
                "rating": review.get("rating"),
                "text": review.get("snippet", ""),
                "date": review.get("date"),
                "author": review.get("user", {}).get("name")
            })
            
        if "serpapi_pagination" not in results: break
        if "next_page_token" not in results["serpapi_pagination"]: break
            
        page_count += 1
        params["next_page_token"] = results["serpapi_pagination"]["next_page_token"]
        if "place_id" in params: del params["place_id"]
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
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

    # --- UPDATED PROMPTS ---
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
st.markdown("Scrape Google Reviews and use AI to identify the **Top 5-10** customer pain points.")

# SIDEBAR SETTINGS
with st.sidebar:
    st.header("🔑 API Settings")
    user_api_key = st.text_input("Enter SerpApi Key", type="password", help="Get this from serpapi.com")
    
    st.divider()
    
    st.header("🌍 Search Settings")
    
    # Country Selector
    selected_country_name = st.selectbox("Search Location (Country)", list(COUNTRY_CODES.keys()), index=0)
    country_code = COUNTRY_CODES[selected_country_name]
    
    # Language Selector
    selected_lang_name = st.selectbox("Language", list(LANGUAGE_CODES.keys()), index=0)
    lang_code = LANGUAGE_CODES[selected_lang_name]
    
    st.info(f"Searching as if in: **{selected_country_name}**\nLanguage: **{selected_lang_name}**")

# MAIN INPUTS
col1, col2 = st.columns(2)
with col1:
    target_business = st.text_input("Main Business Name", placeholder="e.g. So Energy")
with col2:
    competitor_business = st.text_input("Competitor (Optional)", placeholder="e.g. Bulb Energy")

# RUN BUTTON
if st.button("Analyze Reviews", type="primary"):
    if not user_api_key:
        st.warning("Please enter your SerpApi key in the sidebar to proceed.")
    elif not target_business:
        st.warning("Please enter a business name.")
    else:
        results_store = {}
        
        try:
            with st.status(f"🔍 Analyzing {target_business}...", expanded=True) as status:
                # 1. Scrape Main Business
                status.write(f"Searching for '{target_business}' in {selected_country_name}...")
                
                place_id = get_place_id(target_business, user_api_key, country_code, lang_code)
                
                if place_id:
                    status.write("Fetching reviews...")
                    df_target = get_reviews(place_id, user_api_key, country_code, lang_code)
                    results_store[target_business] = df_target
                    status.write(f"✅ Found {len(df_target)} reviews.")
                else:
                    status.update(label="Business not found", state="error")
                    st.error(f"Could not find '{target_business}' in {selected_country_name}. Try adding the city name.")
                    st.stop()

                # 2. Scrape Competitor
                if competitor_business:
                    status.write(f"Searching for competitor: {competitor_business}...")
                    comp_id = get_place_id(competitor_business, user_api_key, country_code, lang_code)
                    if comp_id:
                        df_comp = get_reviews(comp_id, user_api_key, country_code, lang_code)
                        results_store[competitor_business] = df_comp
                        status.write(f"✅ Found {len(df_comp)} competitor reviews.")
                
                status.update(label="Scraping Complete! Running AI Analysis...", state="complete")

            # 3. AI Analysis
            if results_store:
                st.divider()
                st.subheader("🧠 Top Pain Points Report")
                with st.spinner("Generating insights..."):
                    # Pass the language name so Gemini answers in that language
                    analysis = analyze_with_gemini(results_store, selected_lang_name)
                st.markdown(analysis)
                
                # 4. Raw Data
                with st.expander("View Raw Data"):
                    tab1, tab2 = st.tabs(["Main Business", "Competitor"])
                    with tab1:
                        st.dataframe(results_store[target_business])
                    with tab2:
                        if competitor_business and competitor_business in results_store:
                            st.dataframe(results_store[competitor_business])
                        else:
                            st.write("No competitor data.")

        except Exception as e:
            st.error(f"An error occurred: {e}")




