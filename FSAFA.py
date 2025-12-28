"""
Project: Detecting Financial Statement Irregularities using Benford’s Law
File: benford_forensic_model.py
Type: Streamlit Application (Live Market Data Only + Modern FinTech UI)

INSTRUCTIONS:
1. Install libraries:
   pip install streamlit numpy pandas matplotlib scipy yfinance

2. Run the application:
   streamlit run benford_forensic_model.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import yfinance as yf

# --- SENSEX 30 TICKER MAPPING (BSE) ---
SENSEX_COMPANIES = {
    "Reliance Industries": "RELIANCE.BO",
    "TCS": "TCS.BO",
    "HDFC Bank": "HDFCBANK.BO",
    "ICICI Bank": "ICICIBANK.BO",
    "Infosys": "INFY.BO",
    "Hindustan Unilever": "HINDUNILVR.BO",
    "ITC": "ITC.BO",
    "SBI": "SBIN.BO",
    "Bharti Airtel": "BHARTIARTL.BO",
    "L&T": "LT.BO",
    "Kotak Mahindra Bank": "KOTAKBANK.BO",
    "Axis Bank": "AXISBANK.BO",
    "Asian Paints": "ASIANPAINT.BO",
    "HCL Technologies": "HCLTECH.BO",
    "Titan Company": "TITAN.BO",
    "Bajaj Finance": "BAJFINANCE.BO",
    "Sun Pharma": "SUNPHARMA.BO",
    "Maruti Suzuki": "MARUTI.BO",
    "UltraTech Cement": "ULTRACEMCO.BO",
    "Tata Motors": "TATAMOTORS.BO",
    "Mahindra & Mahindra": "M&M.BO",
    "Wipro": "WIPRO.BO",
    "NTPC": "NTPC.BO",
    "Power Grid Corp": "POWERGRID.BO",
    "Tata Steel": "TATASTEEL.BO",
    "IndusInd Bank": "INDUSINDBK.BO",
    "Nestle India": "NESTLEIND.BO",
    "Bajaj Finserv": "BAJAJFINSV.BO",
    "Tech Mahindra": "TECHM.BO",
    "JSW Steel": "JSWSTEEL.BO"
}

# --- Page Configuration & Custom Theme ---
st.set_page_config(
    page_title="Forensic Benford Analytics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Modern FinTech Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background - Clean Off-White */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, h2, h3 {
        color: #334155 !important;
    }
    
    /* Card/Metric Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    /* Primary Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }
    div.stButton > button:hover {
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
    }
    
    /* Text Input Styling - Visible Border */
    div[data-testid="stTextInput"] input {
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px;
        background-color: #f8fafc;
        color: #334155;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #4f46e5 !important;
        background-color: #ffffff;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
    }
    
    /* Info Box in Sidebar */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Core Logic Class ---
class BenfordDetector:
    def __init__(self):
        self.benford_probs = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        self.df = pd.DataFrame()
        self.data_source_name = ""

    def load_live_data(self, ticker_symbol, company_name=None):
        """Fetches real financial statements using Yahoo Finance."""

        try:
            stock = yf.Ticker(ticker_symbol)

            # If company name isn't provided, try to fetch it
            if not company_name:
                with st.spinner(f"Verifying ticker {ticker_symbol}..."):
                    try:
                        info = stock.info
                        company_name = info.get('longName', ticker_symbol)
                    except:
                        company_name = ticker_symbol

            self.data_source_name = f"{company_name} ({ticker_symbol})"

            # Fetch statements
            with st.spinner(f"Acquiring financial statements for {company_name}..."):
                # We fetch annual and quarterly to increase sample size
                stmts = [
                    stock.financials, stock.quarterly_financials,
                    stock.balance_sheet, stock.quarterly_balance_sheet,
                    stock.cashflow, stock.quarterly_cashflow
                ]

            all_values = []
            sources = []

            for stmt in stmts:
                if stmt is not None and not stmt.empty:
                    # Fill NaNs with 0
                    stmt = stmt.fillna(0)
                    # Flatten the dataframe to get all numbers
                    for col in stmt.columns:
                        for idx, val in stmt[col].items():
                            # We only care about absolute magnitude, not direction (debit/credit)
                            val = abs(float(val))
                            # Filter: Ignore 0, ignore small numbers (dates/years often appear as numbers)
                            # Ignore numbers < 10 to avoid single digit bias not related to Benford
                            if val > 10 and val != 0:
                                all_values.append(val)
                                sources.append(f"{str(idx)} ({str(col.year) if hasattr(col, 'year') else 'Period'})")

            if not all_values:
                st.error(f"No financial data found for {ticker_symbol}. It might be delisted or data is unavailable.")
                return False

            self.df = pd.DataFrame({
                'Amount': all_values,
                'Source': sources
            })
            self._process_digits()
            return True

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def _process_digits(self):
        """Extracts first digit."""
        # convert to string, remove decimal, strip leading zeros, take first char
        self.df['First_Digit'] = self.df['Amount'].astype(str).str.replace('.', '', regex=False).str.lstrip('0').str[0]

        # Force numeric and drop errors
        self.df['First_Digit'] = pd.to_numeric(self.df['First_Digit'], errors='coerce')
        self.df = self.df.dropna(subset=['First_Digit'])
        self.df = self.df[self.df['First_Digit'] > 0]
        self.df['First_Digit'] = self.df['First_Digit'].astype(int)

    def get_analysis_stats(self):
        """Returns stats for visualization."""
        counts = self.df['First_Digit'].value_counts().sort_index()
        total_count = len(self.df)

        results = []
        observed_counts = []
        expected_counts = []

        for d in range(1, 10):
            obs_count = counts.get(d, 0)
            obs_freq = obs_count / total_count if total_count > 0 else 0
            exp_freq = self.benford_probs[d]

            results.append({
                'Digit': d,
                'Observed %': obs_freq * 100,
                'Benford Expected %': exp_freq * 100
            })

            observed_counts.append(obs_count)
            expected_counts.append(exp_freq * total_count)

        return pd.DataFrame(results).set_index('Digit'), observed_counts, expected_counts, total_count

    def calculate_metrics(self, observed, expected):
        """Calculates Chi-Square and MAD."""
        if sum(observed) == 0: return 0, 0, 0 # Avoid div by zero

        chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)

        obs_props = np.array(observed) / sum(observed)
        exp_props = np.array(expected) / sum(expected)
        mad = np.mean(np.abs(obs_props - exp_props))

        return chi2_stat, p_val, mad

# --- UI Layout ---

# Sidebar Configuration
st.sidebar.markdown("## 📊 Analysis Control")

# Search Mode Selection
search_type = st.sidebar.radio(
    "Select Target",
    ["Quick Select (Sensex 30)", "Search Any Ticker"],
    help="Choose from a preset list or search specifically."
)

ticker_to_fetch = ""
company_name_display = ""
detector = BenfordDetector()
data_loaded = False

if search_type == "Quick Select (Sensex 30)":
    selected_company = st.sidebar.selectbox("Choose Company", list(SENSEX_COMPANIES.keys()))
    ticker_to_fetch = SENSEX_COMPANIES[selected_company]
    company_name_display = selected_company
else:
    # Search Bar Implementation
    st.sidebar.markdown("### Ticker Search")
    user_input = st.sidebar.text_input("Symbol", placeholder="e.g. ZOMATO").upper().strip()

    st.sidebar.info(
        "**Note:** Enter the NSE symbol for Indian stocks (e.g., `TATAMOTORS`, `INFY`, `ZOMATO`). "
        "The system automatically appends `.NS`."
    )

    if user_input:
        # Heuristic: If no suffix provided, assume India NSE (.NS)
        if not user_input.endswith(".NS") and not user_input.endswith(".BO") and not user_input.endswith(".BE"):
            ticker_to_fetch = f"{user_input}.NS"
        else:
            ticker_to_fetch = user_input
        company_name_display = None # Let the loader fetch the real name

st.sidebar.markdown("---")
if st.sidebar.button("RUN FORENSIC ANALYSIS"):
    if ticker_to_fetch:
        data_loaded = detector.load_live_data(ticker_to_fetch, company_name_display)
    else:
        st.sidebar.error("Please enter a ticker symbol.")

# Main Dashboard Area
st.title("Forensic Analytics Lab")
st.markdown(f"### Benford's Law Irregularity Detector")

if not data_loaded:
    # Empty State / Landing Page
    st.markdown("""
    <div style="background-color: white; padding: 30px; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 20px;">
        <h3 style="margin-top:0;">Ready to Analyze</h3>
        <p style="color: #64748b;">Select a company from the sidebar to begin scraping live financial statements.</p>
        <p style="color: #64748b; font-size: 0.9em;">This tool fetches Income Statements, Balance Sheets, and Cash Flows, extracts all numerical line items, and compares the leading digit frequency against Benford's Law distribution.</p>
    </div>
    """, unsafe_allow_html=True)

if data_loaded:
    df_results, obs_counts, exp_counts, N = detector.get_analysis_stats()
    chi2, p_val, mad = detector.calculate_metrics(obs_counts, exp_counts)

    st.divider()
    st.markdown(f"#### 🎯 Target: {detector.data_source_name}")

    # 1. Summary Metrics in a Grid
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sample Size (N)", N, help="Total count of financial variables extracted")
    col2.metric("Chi-Square Stat", f"{chi2:.2f}")

    is_sus = p_val < 0.05
    col3.metric("P-Value", f"{p_val:.4f}",
                delta="Significant Deviation" if is_sus else "Normal",
                delta_color="inverse" if is_sus else "normal")

    col4.metric("MAD Score", f"{mad:.4f}",
                delta="High Risk (>0.015)" if mad > 0.015 else "Low Risk",
                delta_color="inverse")

    if N < 200:
        st.warning(f"⚠️ **Low Data Volume:** N={N}. Statistical reliability increases significantly with N > 500.")

    # 2. Charts & Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 Visual Analysis", "📄 Methodology", "💾 Source Data"])

    with tab1:
        col_chart, col_details = st.columns([2, 1])

        with col_chart:
            # FinTech Style Chart
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#ffffff') # White background for the figure
            ax.set_facecolor('#ffffff')

            digits = df_results.index
            width = 0.4

            # Sophisticated Color Palette
            color_obs = '#4f46e5'  # Indigo 600
            color_exp = '#cbd5e1'  # Slate 300 (Gray)

            # Bars
            bar2 = ax.bar(digits + width/2, df_results['Benford Expected %'], width, label="Theoretical (Benford)", color=color_exp)
            bar1 = ax.bar(digits - width/2, df_results['Observed %'], width, label='Observed (Actual)', color=color_obs)

            # Trend Line
            ax.plot(digits, df_results['Benford Expected %'], color='#f59e0b', marker='o', markersize=6, linestyle='-', linewidth=2, alpha=0.8, label='Expected Curve')

            # Styling
            ax.set_xticks(digits)
            ax.set_ylabel('Frequency (%)', fontsize=10, fontweight='600', color='#475569')
            ax.set_xlabel('Leading Digit', fontsize=10, fontweight='600', color='#475569')
            ax.set_title("Leading Digit Distribution", fontsize=12, fontweight='700', color='#1e293b', pad=20)

            # Legend
            ax.legend(frameon=False, loc='upper right')

            # Clean up grid
            ax.grid(axis='y', linestyle='--', alpha=0.4, color='#e2e8f0')
            ax.grid(axis='x', alpha=0) # Hide x grid
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#cbd5e1')

            st.pyplot(fig)

        with col_details:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; height: 100%;">
                <h4 style="margin-top:0;">Diagnostic Report</h4>
            """, unsafe_allow_html=True)

            if mad > 0.015:
                st.error("🚩 **Result: NON-CONFORMING**")
                st.markdown(f"""
                The Mean Absolute Deviation (MAD) is **{mad:.4f}**, which exceeds the critical threshold of **0.015**.
                
                **Implications:**
                - The digit distribution does not follow natural logarithmic growth.
                - While this is not proof of fraud, it is a statistical anomaly warranting deeper forensic audit.
                """)
            else:
                st.success("✅ **Result: CONFORMING**")
                st.markdown(f"""
                The MAD score is **{mad:.4f}**, falling within the acceptable range.
                
                **Implications:**
                - The financial variables exhibit natural distribution patterns.
                - No systemic manipulation of line items detected at the aggregate level.
                """)

            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        ### Forensic Methodology
        
        **1. Data Acquisition**
        We utilize `yfinance` to scrape the publicly filed Annual and Quarterly financial statements (Income Statement, Balance Sheet, Cash Flow) for **{detector.data_source_name}**.
        
        **2. Data Cleaning**
        - **Extraction:** All numerical cells are flattened into a single array.
        - **Filtration:** We remove dates, null values, and integers < 10 (to avoid small-number bias).
        - **Digit Extraction:** The leading digit (1-9) is extracted from the remaining {N} variables.
        
        **3. Statistical Tests**
        - **Chi-Square:** Tests if the difference between Observed and Expected frequencies is significant.
        - **MAD (Mean Absolute Deviation):** Measures the average distance between the actual data and Benford's curve. MAD is generally preferred for large financial datasets as it is less sensitive to sample size than Chi-Square.
        """)

    with tab3:
        st.subheader("Extracted Financial Variables")
        st.dataframe(detector.df, use_container_width=True)

        csv = detector.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Extracted Data CSV",
            csv,
            "benford_analysis_data.csv",
            "text/csv",
            key='download-csv'
        )