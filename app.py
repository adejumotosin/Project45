import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import calendar
from scipy import stats
from hijri_converter import Hijri, Gregorian
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Seasonal Violence Patterns - Advanced",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
    .stat-sig {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 10px 0;
    }
    .insight-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📅 Advanced Seasonal Violence Patterns Analyzer")
st.markdown("""
Enhanced conflict analysis with **weather correlation**, **religious calendar tracking**, 
and **statistical significance testing**. Powered by ACLED data.
""")

# Sidebar controls
st.sidebar.header("🎯 Analysis Parameters")

# Country selection with coordinates
COUNTRIES_WITH_COORDS = {
    "Nigeria": (9.0820, 8.6753),
    "Somalia": (5.1521, 46.1996),
    "Ethiopia": (9.1450, 40.4897),
    "Democratic Republic of Congo": (-4.0383, 21.7587),
    "Sudan": (12.8628, 30.2176),
    "South Sudan": (6.8770, 31.3070),
    "Mali": (17.5707, -3.9962),
    "Burkina Faso": (12.2383, -1.5616),
    "Niger": (17.6078, 8.0817),
    "Kenya": (-0.0236, 37.9062),
    "Cameroon": (7.3697, 12.3547),
    "Mozambique": (-18.6657, 35.5296),
    "Central African Republic": (6.6111, 20.9394),
    "Myanmar": (21.9162, 95.9560),
    "Afghanistan": (33.9391, 67.7100),
    "Pakistan": (30.3753, 69.3451),
    "Yemen": (15.5527, 48.5164),
    "Syria": (34.8021, 38.9968),
    "Iraq": (33.2232, 43.6793),
    "Libya": (26.3351, 17.2283),
    "Lebanon": (33.8547, 35.8623),
    "Palestine": (31.9522, 35.2332)
}

selected_countries = st.sidebar.multiselect(
    "Select Countries (max 3 for performance)",
    list(COUNTRIES_WITH_COORDS.keys()),
    default=["Nigeria"],
    max_selections=3
)

# Date range
st.sidebar.subheader("📅 Date Range")
years_back = st.sidebar.slider("Years of historical data", 1, 5, 3)
end_date = datetime.now()
start_date = end_date - timedelta(days=365*years_back)

# Event type filters
EVENT_TYPES = [
    "Battles",
    "Violence against civilians",
    "Protests",
    "Riots",
    "Explosions/Remote violence",
    "Strategic developments"
]

selected_event_types = st.sidebar.multiselect(
    "Event Types",
    EVENT_TYPES,
    default=["Battles", "Violence against civilians", "Explosions/Remote violence"]
)

# Fatality threshold
min_fatalities = st.sidebar.slider(
    "Minimum fatalities per event",
    0, 50, 0
)

# Advanced options
st.sidebar.subheader("🔬 Advanced Features")
enable_weather = st.sidebar.checkbox("Enable Weather Correlation", value=True)
enable_religious = st.sidebar.checkbox("Enable Religious Calendar Analysis", value=True)
enable_stats = st.sidebar.checkbox("Enable Statistical Testing", value=True)

# Helper Functions

@st.cache_data(ttl=86400)  # Cache token for 24 hours
def get_acled_token(email, password):
    """Get ACLED OAuth access token"""
    token_url = "https://acleddata.com/oauth/token"
    
    data = {
        'username': email,
        'password': password,
        'grant_type': 'password',
        'client_id': 'acled'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=30)
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data['access_token'], token_data.get('refresh_token')
        else:
            st.error(f"❌ Authentication failed: {response.status_code}")
            st.error(f"Response: {response.text}")
            return None, None
    except Exception as e:
        st.error(f"❌ Error getting token: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)
def load_acled_data(countries, start_date, end_date, event_types, min_fatalities, email, password):
    """Fetch ACLED data with OAuth authentication"""
    
    # Get access token
    access_token, refresh_token = get_acled_token(email, password)
    
    if not access_token:
        st.error("Failed to authenticate with ACLED API. Check your credentials.")
        return pd.DataFrame()
    
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, country in enumerate(countries):
        status_text.text(f"Loading data for {country}...")
        progress_bar.progress((idx + 1) / len(countries))
        
        try:
            api_url = "https://acleddata.com/api/acled/read"
            
            params = {
                '_format': 'json',
                'country': country,
                'event_date': f'{start_date.strftime("%Y-%m-%d")}|{end_date.strftime("%Y-%m-%d")}',
                'event_date_where': 'BETWEEN',
                'event_type': '|'.join(event_types),
                'limit': 10000
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 200 and 'data' in data:
                    df_country = pd.DataFrame(data['data'])
                    all_data.append(df_country)
                else:
                    st.warning(f"No data returned for {country}")
            else:
                st.error(f"Error loading {country}: HTTP {response.status_code}")
                
        except Exception as e:
            st.error(f"Error loading {country}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Data cleaning
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Filter by fatalities
    if min_fatalities > 0:
        df = df[df['fatalities'] >= min_fatalities]
    
    # Extract temporal features
    df['year'] = df['event_date'].dt.year
    df['month'] = df['event_date'].dt.month
    df['month_name'] = df['event_date'].dt.month_name()
    df['day_of_week'] = df['event_date'].dt.dayofweek
    df['day_name'] = df['event_date'].dt.day_name()
    df['week_of_year'] = df['event_date'].dt.isocalendar().week
    df['day_of_year'] = df['event_date'].dt.dayofyear
    df['quarter'] = df['event_date'].dt.quarter
    
    # Seasons (Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    return df

@st.cache_data(ttl=86400)
def get_weather_data(lat, lon, start_date, end_date):
    """Fetch historical weather data from Open-Meteo"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'daily' in data:
            weather_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'precipitation': data['daily']['precipitation_sum'],
                'rainfall': data['daily']['rain_sum']
            })
            return weather_df
        return None
    except Exception as e:
        st.warning(f"Weather data unavailable: {str(e)}")
        return None

def is_ramadan(date):
    """Check if date falls in Ramadan"""
    try:
        hijri_date = Gregorian(date.year, date.month, date.day).to_hijri()
        return hijri_date.month == 9
    except:
        return False

def is_eid_al_fitr(date):
    """Check if date is near Eid al-Fitr (end of Ramadan)"""
    try:
        hijri_date = Gregorian(date.year, date.month, date.day).to_hijri()
        return hijri_date.month == 10 and hijri_date.day <= 3
    except:
        return False

def is_eid_al_adha(date):
    """Check if date is near Eid al-Adha"""
    try:
        hijri_date = Gregorian(date.year, date.month, date.day).to_hijri()
        return hijri_date.month == 12 and 10 <= hijri_date.day <= 13
    except:
        return False

def is_muharram(date):
    """Check if date is in Muharram (Islamic New Year)"""
    try:
        hijri_date = Gregorian(date.year, date.month, date.day).to_hijri()
        return hijri_date.month == 1
    except:
        return False

def add_religious_features(df):
    """Add religious calendar features"""
    df['is_ramadan'] = df['event_date'].apply(is_ramadan)
    df['is_eid_fitr'] = df['event_date'].apply(is_eid_al_fitr)
    df['is_eid_adha'] = df['event_date'].apply(is_eid_al_adha)
    df['is_muharram'] = df['event_date'].apply(is_muharram)
    
    # Christian holidays (fixed dates)
    df['is_christmas'] = ((df['month'] == 12) & (df['event_date'].dt.day.between(24, 26)))
    df['is_easter_season'] = df['month'].isin([3, 4])
    
    return df

# Load ACLED data
if not selected_countries:
    st.warning("⚠️ Please select at least one country")
    st.stop()

with st.spinner("🔐 Authenticating and loading ACLED data..."):
    df = load_acled_data(
        selected_countries, 
        start_date, 
        end_date, 
        selected_event_types, 
        min_fatalities,
        st.secrets["acled"]["email"],
        st.secrets["acled"]["password"]
    )

if df.empty:
    st.error("No data found. Try adjusting filters.")
    st.stop()

# Add religious calendar features
if enable_religious:
    with st.spinner("Adding religious calendar features..."):
        df = add_religious_features(df)

# Load weather data
weather_df = None
if enable_weather and len(selected_countries) == 1:
    country = selected_countries[0]
    lat, lon = COUNTRIES_WITH_COORDS[country]
    
    with st.spinner("Loading weather data..."):
        weather_df = get_weather_data(lat, lon, start_date, end_date)
    
    if weather_df is not None:
        df_with_weather = df.copy()
        df_with_weather['date_only'] = df_with_weather['event_date'].dt.date
        weather_df['date_only'] = weather_df['date'].dt.date
        
        df_with_weather = df_with_weather.merge(
            weather_df[['date_only', 'temp_max', 'precipitation', 'rainfall']],
            on='date_only',
            how='left'
        )
    else:
        df_with_weather = df.copy()
        enable_weather = False
elif enable_weather and len(selected_countries) > 1:
    st.info("ℹ️ Weather correlation available only for single-country analysis")
    enable_weather = False
    df_with_weather = df.copy()
else:
    df_with_weather = df.copy()

# Summary metrics
st.header("📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Events", f"{len(df):,}")

with col2:
    st.metric("Total Fatalities", f"{df['fatalities'].sum():,.0f}")

with col3:
    st.metric("Date Range", f"{(df['event_date'].max() - df['event_date'].min()).days} days")

with col4:
    st.metric("Deadliest Day", df.groupby('day_name')['fatalities'].sum().idxmax())

with col5:
    st.metric("Deadliest Month", df.groupby('month_name')['fatalities'].sum().idxmax())

# Additional metrics row
if enable_religious:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ramadan_events = df[df['is_ramadan']].shape[0]
        ramadan_rate = ramadan_events / len(df) * 100
        st.metric("Ramadan Events", f"{ramadan_events:,}", f"{ramadan_rate:.1f}% of total")
    
    with col2:
        eid_fitr_events = df[df['is_eid_fitr']].shape[0]
        st.metric("Eid al-Fitr Events", f"{eid_fitr_events:,}")
    
    with col3:
        eid_adha_events = df[df['is_eid_adha']].shape[0]
        st.metric("Eid al-Adha Events", f"{eid_adha_events:,}")
    
    with col4:
        muharram_events = df[df['is_muharram']].shape[0]
        st.metric("Muharram Events", f"{muharram_events:,}")

# REST OF THE CODE REMAINS EXACTLY THE SAME FROM YOUR ORIGINAL
# (All the tabs, visualizations, statistical tests, etc.)
# I'm keeping it short here, but copy everything from "# Tabs" onwards from your original code

# ... [Insert all your tab code here - it's identical to what you already have]

# Footer
st.markdown("---")
st.caption(f"""
**Data Sources**: ACLED (OAuth), Open-Meteo | **Built with**: Streamlit, Plotly, SciPy  
**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | 
**Countries**: {', '.join(selected_countries)} | **Events**: {len(df):,}
""")
Updated .streamlit/secrets.toml
[acled]
email = "your_email@example.com"
password = "your_password"
Key change: Replace api_key with email and password.
Key Changes Made
1. OAuth Token Function
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_acled_token(email, password):
    """Get ACLED OAuth access token"""
    token_url = "https://acleddata.com/oauth/token"
    
    data = {
        'username': email,
        'password': password,
        'grant_type': 'password',
        'client_id': 'acled'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    response = requests.post(token_url, headers=headers, data=data, timeout=30)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data['access_token'], token_data.get('refresh_token')
    else:
        st.error(f"Authentication failed: {response.status_code}")
        return None, None
2. Updated Data Loading Function
@st.cache_data(ttl=3600)
def load_acled_data(countries, start_date, end_date, event_types, min_fatalities, email, password):
    """Fetch ACLED data with OAuth authentication"""
    
    # Get access token
    access_token, refresh_token = get_acled_token(email, password)
    
    if not access_token:
        return pd.DataFrame()
    
    # ... rest of function
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(api_url, params=params, headers=headers, timeout=30)
3. Updated API Call
# Old (incorrect)
params = {
    'key': st.secrets["acled"]["api_key"],
    'email': st.secrets["acled"]["email"],
    # ...
}

# New (correct)
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

params = {
    '_format': 'json',
    'country': country,
    # ... (no key or email in params)