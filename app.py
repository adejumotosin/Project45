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
Enhanced conflict analysis with weather correlation, religious calendar tracking,
and statistical significance testing. Powered by ACLED data.
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
    df['is_easter_season'] = df['month'].isin([3, 4])  # Simplified
    
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
        # Merge weather with events
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
        non_ramadan_events = df[~df['is_ramadan']].shape[0]
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

# Tabs
tabs = ["📅 Calendar", "📈 Monthly", "🗓️ Weekly", "🌾 Seasonal"]

if enable_weather:
    tabs.append("🌦️ Weather")
if enable_religious:
    tabs.append("🕌 Religious")
if enable_stats:
    tabs.append("📊 Statistics")

tabs.append("🔮 Forecast")

selected_tab = st.tabs(tabs)
tab_idx = 0

# TAB: Calendar Heatmap
with selected_tab[tab_idx]:
    tab_idx += 1
    st.subheader("Calendar Heatmap: Violence Intensity")
    
    heatmap_data = df.groupby(['year', 'month']).agg({
        'fatalities': 'sum',
        'event_date': 'count'
    }).rename(columns={'event_date': 'events'}).reset_index()
    
    metric_choice = st.radio(
        "Display metric:",
        ["Fatalities", "Event Count", "Avg Fatalities/Event"],
        horizontal=True
    )
    
    if metric_choice == "Avg Fatalities/Event":
        heatmap_data['metric'] = heatmap_data['fatalities'] / heatmap_data['events']
    elif metric_choice == "Fatalities":
        heatmap_data['metric'] = heatmap_data['fatalities']
    else:
        heatmap_data['metric'] = heatmap_data['events']
    
    heatmap_pivot = heatmap_data.pivot(index='month', columns='year', values='metric')
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    heatmap_pivot.index = month_names
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Reds',
        hoverongaps=False,
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Value: %{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{metric_choice} Heatmap",
        xaxis_title="Year",
        yaxis_title="Month",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# TAB: Monthly Trends
with selected_tab[tab_idx]:
    tab_idx += 1
    st.subheader("Monthly Violence Trends")
    
    monthly_agg = df.groupby('month').agg({
        'fatalities': ['sum', 'mean'],
        'event_date': 'count'
    }).reset_index()
    
    monthly_agg.columns = ['month', 'total_fatalities', 'avg_fatalities', 'total_events']
    monthly_agg['month_name'] = monthly_agg['month'].apply(lambda x: calendar.month_name[x])
    monthly_agg['avg_fat_per_event'] = monthly_agg['total_fatalities'] / monthly_agg['total_events']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Events per Month', 'Fatalities per Month', 
                       'Avg Fatalities/Event', 'Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    fig.add_trace(
        go.Bar(x=monthly_agg['month_name'], y=monthly_agg['total_events'],
               marker_color='steelblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=monthly_agg['month_name'], y=monthly_agg['total_fatalities'],
               marker_color='crimson'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_agg['month_name'], y=monthly_agg['avg_fat_per_event'],
                  mode='lines+markers', line=dict(color='darkred', width=3)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=monthly_agg['month_name'], values=monthly_agg['total_events']),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Monthly Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical test for monthly differences
    if enable_stats:
        st.markdown("### 📊 Statistical Significance Test")
        
        monthly_groups = [df[df['month'] == m]['fatalities'].values for m in range(1, 13)]
        # Filter out empty groups
        monthly_groups = [g for g in monthly_groups if len(g) > 0]
        
        if len(monthly_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*monthly_groups)
            
            if p_value < 0.05:
                st.markdown(f'<div class="stat-sig">✅ <b>Statistically Significant</b>: Monthly fatality differences are significant (F={f_stat:.2f}, p={p_value:.4f})</div>', unsafe_allow_html=True)
            else:
                st.info(f"Monthly differences not statistically significant (F={f_stat:.2f}, p={p_value:.4f})")

# TAB: Weekly Patterns
with selected_tab[tab_idx]:
    tab_idx += 1
    st.subheader("Week-by-Week Analysis")
    
    weekly_agg = df.groupby('week_of_year').agg({
        'fatalities': 'sum',
        'event_date': 'count'
    }).reset_index()
    weekly_agg.columns = ['week', 'fatalities', 'events']
    
    weekly_agg['events_smooth'] = weekly_agg['events'].rolling(window=4, center=True).mean()
    weekly_agg['fatalities_smooth'] = weekly_agg['fatalities'].rolling(window=4, center=True).mean()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=weekly_agg['week'], y=weekly_agg['events_smooth'],
                  mode='lines', name='Events (4-week avg)',
                  line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=weekly_agg['week'], y=weekly_agg['fatalities_smooth'],
                  mode='lines', name='Fatalities (4-week avg)',
                  line=dict(color='red', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Week of Year")
    fig.update_yaxes(title_text="Events", secondary_y=False)
    fig.update_yaxes(title_text="Fatalities", secondary_y=True)
    fig.update_layout(title="Weekly Patterns", height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    st.markdown("### 📅 Day of Week Breakdown")
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_agg = df.groupby('day_name').agg({
        'fatalities': 'sum',
        'event_date': 'count'
    }).reset_index()
    dow_agg.columns = ['day_name', 'total_fatalities', 'total_events']
    dow_agg['day_name'] = pd.Categorical(dow_agg['day_name'], categories=day_order, ordered=True)
    dow_agg = dow_agg.sort_values('day_name')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(dow_agg, x='day_name', y='total_events',
                     title='Events by Day', color='total_events',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(dow_agg, x='day_name', y='total_fatalities',
                     title='Fatalities by Day', color='total_fatalities',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical test
    if enable_stats:
        dow_groups = [df[df['day_name'] == day]['fatalities'].values for day in day_order]
        dow_groups = [g for g in dow_groups if len(g) > 0]
        
        if len(dow_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*dow_groups)
            
            if p_value < 0.05:
                st.markdown(f'<div class="stat-sig">✅ Day-of-week differences are statistically significant (F={f_stat:.2f}, p={p_value:.4f})</div>', unsafe_allow_html=True)

# TAB: Seasonal Patterns
with selected_tab[tab_idx]:
    tab_idx += 1
    st.subheader("Seasonal Violence Analysis")
    
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_agg = df.groupby('season').agg({
        'fatalities': 'sum',
        'event_date': 'count'
    }).reset_index()
    seasonal_agg.columns = ['season', 'fatalities', 'events']
    seasonal_agg['season'] = pd.Categorical(seasonal_agg['season'], categories=season_order, ordered=True)
    seasonal_agg = seasonal_agg.sort_values('season')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(seasonal_agg, x='season', y='events',
                     title='Events by Season', color='events',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(seasonal_agg, x='season', y='fatalities',
                     title='Fatalities by Season', color='fatalities',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Agricultural context
    st.markdown("### 🌾 Agricultural Cycle Context")
    st.info("""
    **Typical patterns:**
    - **Winter (Dec-Feb)**: Dry season, easier military movement
    - **Spring (Mar-May)**: Planting season, labor/land disputes
    - **Summer (Jun-Aug)**: Growing season, water conflicts
    - **Fall (Sep-Nov)**: Harvest season, resource competition
    """)
    
    max_season = seasonal_agg.loc[seasonal_agg['fatalities'].idxmax()]
    st.markdown(f'<div class="insight-box">📌 <b>Peak Season</b>: {max_season["season"]} shows highest violence ({max_season["fatalities"]:,.0f} fatalities)</div>', unsafe_allow_html=True)
    
    # Statistical test
    if enable_stats:
        season_groups = [df[df['season'] == s]['fatalities'].values for s in season_order]
        season_groups = [g for g in season_groups if len(g) > 0]
        
        if len(season_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*season_groups)
            
            if p_value < 0.05:
                st.markdown(f'<div class="stat-sig">✅ Seasonal differences are statistically significant (F={f_stat:.2f}, p={p_value:.4f})</div>', unsafe_allow_html=True)
            else:
                st.info(f"Seasonal differences not statistically significant (p={p_value:.4f})")

# TAB: Weather Correlation
if enable_weather:
    with selected_tab[tab_idx]:
        tab_idx += 1
        st.subheader("🌦️ Weather-Violence Correlation")
        
        if weather_df is not None and 'temp_max' in df_with_weather.columns:
            # Daily aggregation
            daily_events = df_with_weather.groupby('date_only').agg({
                'event_date': 'count',
                'fatalities': 'sum',
                'temp_max': 'first',
                'precipitation': 'first',
                'rainfall': 'first'
            }).reset_index()
            daily_events.columns = ['date', 'events', 'fatalities', 'temp_max', 'precipitation', 'rainfall']
            
            # Remove outliers for cleaner visualization
            daily_events = daily_events.dropna(subset=['temp_max', 'precipitation'])
            
            # Temperature correlation
            st.markdown("### 🌡️ Temperature vs Violence")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot
                fig = px.scatter(daily_events, x='temp_max', y='events',
                               title='Temperature vs Event Count',
                               trendline='ols',
                               labels={'temp_max': 'Max Temperature (°C)', 'events': 'Events'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation
                temp_corr = daily_events['temp_max'].corr(daily_events['events'])
                st.metric("Temperature-Event Correlation", f"{temp_corr:.3f}")
            
            with col2:
                # Binned analysis
                daily_events['temp_bin'] = pd.cut(daily_events['temp_max'], bins=5)
                temp_binned = daily_events.groupby('temp_bin')['events'].mean().reset_index()
                
                fig = px.bar(temp_binned, x='temp_bin', y='events',
                            title='Average Events by Temperature Range',
                            labels={'temp_bin': 'Temperature Range', 'events': 'Avg Events'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Precipitation correlation
            st.markdown("### 🌧️ Rainfall vs Violence")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create dry/wet categories
                daily_events['rain_category'] = pd.cut(
                    daily_events['precipitation'],
                    bins=[-0.1, 0.1, 10, 50, 1000],
                    labels=['No Rain', 'Light', 'Moderate', 'Heavy']
                )
                
                rain_agg = daily_events.groupby('rain_category').agg({
                    'events': 'sum',
                    'fatalities': 'sum'
                }).reset_index()
                
                fig = px.bar(rain_agg, x='rain_category', y='events',
                            title='Events by Rainfall Category',
                            color='events', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Time series overlay
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Aggregate to weekly for cleaner viz
                daily_events['week'] = pd.to_datetime(daily_events['date']).dt.to_period('W').dt.start_time
                weekly = daily_events.groupby('week').agg({
                    'events': 'sum',
                    'precipitation': 'mean'
                }).reset_index()
                
                fig.add_trace(
                    go.Bar(x=weekly['week'], y=weekly['precipitation'],
                          name='Avg Precipitation', marker_color='lightblue'),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=weekly['week'], y=weekly['events'],
                              mode='lines', name='Events',
                              line=dict(color='red', width=2)),
                    secondary_y=True
                )
                
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=False)
                fig.update_yaxes(title_text="Events", secondary_y=True)
                fig.update_layout(title="Rainfall vs Events Over Time", height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical insights
            st.markdown("### 📊 Weather Correlations Summary")
            
            rain_corr = daily_events['precipitation'].corr(daily_events['events'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Temperature Correlation", f"{temp_corr:.3f}",
                         help="Correlation between max temperature and event count")
            
            with col2:
                st.metric("Precipitation Correlation", f"{rain_corr:.3f}",
                         help="Correlation between rainfall and event count")
            
            with col3:
                # Count rainy vs dry day events
                rainy_days = daily_events[daily_events['precipitation'] > 1]['events'].sum()
                dry_days = daily_events[daily_events['precipitation'] <= 1]['events'].sum()
                st.metric("Rainy vs Dry Events", f"{rainy_days / dry_days:.2f}x",
                         help="Event ratio: rainy days vs dry days")
            
            # Interpretation
            st.markdown("### 🔍 Interpretation")
            
            if abs(temp_corr) > 0.3:
                temp_dir = "increases" if temp_corr > 0 else "decreases"
                st.markdown(f'<div class="insight-box">🌡️ <b>Temperature Effect</b>: Violence {temp_dir} with higher temperatures (r={temp_corr:.3f})</div>', unsafe_allow_html=True)
            else:
                st.info("No strong temperature correlation detected")
            
            if abs(rain_corr) > 0.2:
                rain_dir = "increases" if rain_corr > 0 else "decreases"
                st.markdown(f'<div class="insight-box">🌧️ <b>Rainfall Effect</b>: Violence {rain_dir} during rainy periods (r={rain_corr:.3f})</div>', unsafe_allow_html=True)
            else:
                st.info("No strong rainfall correlation detected")
            
        else:
            st.warning("Weather data not available for this selection")

# TAB: Religious Calendar
if enable_religious:
    with selected_tab[tab_idx]:
        tab_idx += 1
        st.subheader("🕌 Religious Calendar Analysis")
        
        # Islamic calendar events
        st.markdown("### Islamic Calendar Events")
        
        religious_periods = {
            'Ramadan': 'is_ramadan',
            'Eid al-Fitr': 'is_eid_fitr',
            'Eid al-Adha': 'is_eid_adha',
            'Muharram': 'is_muharram'
        }
        
        religious_stats = []
        for period, col in religious_periods.items():
            if col in df.columns:
                period_events = df[df[col] == True]
                non_period_events = df[df[col] == False]
                
                if len(period_events) > 0 and len(non_period_events) > 0:
                    period_rate = len(period_events) / (df[col].sum() / len(df))
                    avg_fatalities = period_events['fatalities'].mean()
                    
                    religious_stats.append({
                        'Period': period,
                        'Events': len(period_events),
                        'Fatalities': period_events['fatalities'].sum(),
                        'Avg Fatalities/Event': avg_fatalities,
                        'Days': df[col].sum()
                    })
        
        if religious_stats:
            religious_df = pd.DataFrame(religious_stats)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(religious_df, x='Period', y='Events',
                            title='Events During Religious Periods',
                            color='Events', color_continuous_scale='Purples')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(religious_df, x='Period', y='Avg Fatalities/Event',
                            title='Average Fatalities per Event',
                            color='Avg Fatalities/Event', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical comparison
            st.markdown("### 📊 Statistical Comparison")
            
            for period, col in religious_periods.items():
                if col in df.columns and df[col].sum() > 0:
                    period_fatalities = df[df[col] == True]['fatalities'].values
                    non_period_fatalities = df[df[col] == False]['fatalities'].values
                    
                    if len(period_fatalities) > 10 and len(non_period_fatalities) > 10:
                        # T-test
                        t_stat, p_value = stats.ttest_ind(period_fatalities, non_period_fatalities)
                        
                        period_mean = period_fatalities.mean()
                        non_period_mean = non_period_fatalities.mean()
                        
                        if p_value < 0.05:
                            direction = "higher" if period_mean > non_period_mean else "lower"
                            st.markdown(f'<div class="stat-sig">✅ <b>{period}</b>: Fatalities are {direction} during this period (p={p_value:.4f}, avg: {period_mean:.2f} vs {non_period_mean:.2f})</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"{period}: No significant difference (p={p_value:.4f})")
            
            # Display table
            st.markdown("### 📋 Summary Table")
            st.dataframe(religious_df, use_container_width=True)
        
        # Christian holidays
        st.markdown("### Christian Calendar Events")
        
        if 'is_christmas' in df.columns:
            christmas_events = df[df['is_christmas'] == True]
            easter_events = df[df['is_easter_season'] == True]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Christmas Period Events", len(christmas_events),
                         f"{christmas_events['fatalities'].sum():,.0f} fatalities")
            
            with col2:
                st.metric("Easter Season Events", len(easter_events),
                         f"{easter_events['fatalities'].sum():,.0f} fatalities")

# TAB: Statistical Analysis
if enable_stats:
    with selected_tab[tab_idx]:
        tab_idx += 1
        st.subheader("📊 Statistical Significance Testing")
        
        st.markdown("""  
        This section performs rigorous statistical tests to determine if observed patterns  
        are statistically significant or could be due to random chance.  
        """)
        
        # Seasonal ANOVA
        st.markdown("### 1. Seasonal Variation (ANOVA)")
        
        season_groups = [df[df['season'] == s]['fatalities'].values for s in ['Winter', 'Spring', 'Summer', 'Fall']]
        season_groups = [g for g in season_groups if len(g) > 0]
        
        if len(season_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*season_groups)
            
            st.write(f"**F-statistic**: {f_stat:.4f}")
            st.write(f"**p-value**: {p_value:.4f}")
            
            if p_value < 0.05:
                st.markdown(f'<div class="stat-sig">✅ <b>Conclusion</b>: Seasonal differences in fatalities are statistically significant at α=0.05</div>', unsafe_allow_html=True)
            else:
                st.info("Seasonal differences are not statistically significant")
            
            # Effect size (eta-squared)
            season_df = df[['season', 'fatalities']].copy()
            grand_mean = season_df['fatalities'].mean()
            ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in season_groups])
            ss_total = sum([(x - grand_mean)**2 for g in season_groups for x in g])
            eta_squared = ss_between / ss_total
            
            st.write(f"**Effect Size (η²)**: {eta_squared:.4f}")
            st.caption("η² > 0.14 = large effect, 0.06-0.14 = medium, < 0.06 = small")
        
        # Monthly Kruskal-Wallis (non-parametric alternative)
        st.markdown("### 2. Monthly Variation (Kruskal-Wallis)")
        
        monthly_groups = [df[df['month'] == m]['fatalities'].values for m in range(1, 13)]
        monthly_groups = [g for g in monthly_groups if len(g) > 0]
        
        if len(monthly_groups) >= 2:
            h_stat, p_value = stats.kruskal(*monthly_groups)
            
            st.write(f"**H-statistic**: {h_stat:.4f}")
            st.write(f"**p-value**: {p_value:.4f}")
            
            if p_value < 0.05:
                st.markdown(f'<div class="stat-sig">✅ Monthly differences are statistically significant</div>', unsafe_allow_html=True)
            else:
                st.info("Monthly differences are not statistically significant")
        
        # Day of week Chi-square
        st.markdown("### 3. Day of Week Distribution (Chi-Square)")
        
        dow_counts = df['day_of_week'].value_counts().sort_index()
        expected = len(df) / 7
        
        chi2_stat, p_value = stats.chisquare(dow_counts.values, f_exp=[expected]*7)
        
        st.write(f"**χ² statistic**: {chi2_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        
        if p_value < 0.05:
            st.markdown(f'<div class="stat-sig">✅ Day-of-week distribution differs from uniform (some days have significantly more events)</div>', unsafe_allow_html=True)
        else:
            st.info("Day-of-week distribution is uniform (no significant differences)")
        
        # Trend analysis
        st.markdown("### 4. Temporal Trend (Mann-Kendall)")
        
        try:
            from scipy.stats import kendalltau
            
            monthly_totals = df.groupby(df['event_date'].dt.to_period('M'))['fatalities'].sum().reset_index()
            monthly_totals['time'] = range(len(monthly_totals))
            
            tau, p_value = kendalltau(monthly_totals['time'], monthly_totals['fatalities'])
            
            st.write(f"**Kendall's τ**: {tau:.4f}")
            st.write(f"**p-value**: {p_value:.4f}")
            
            if p_value < 0.05:
                trend_dir = "increasing" if tau > 0 else "decreasing"
                st.markdown(f'<div class="stat-sig">✅ Significant {trend_dir} trend over time</div>', unsafe_allow_html=True)
            else:
                st.info("No significant temporal trend detected")
        except ImportError:
            st.warning("Kendall test requires scipy >= 1.7")
        
        # Summary
        st.markdown("### 📋 Interpretation Guide")
        st.info("""
        **p-value < 0.05**: Pattern is statistically significant (unlikely due to chance)
        
        **p-value ≥ 0.05**: Pattern could be due to random variation
        
        **Important**: Statistical significance ≠ practical significance.   
        Always consider effect sizes and real-world context.
        """)

# TAB: Forecast
with selected_tab[tab_idx]:
    st.subheader("🔮 Risk Forecasting")
    
    # Calculate risk scores
    monthly_risk = df.groupby('month').agg({
        'event_date': 'count',
        'fatalities': 'sum'
    }).reset_index()
    monthly_risk.columns = ['month', 'events', 'fatalities']
    monthly_risk['month_name'] = monthly_risk['month'].apply(lambda x: calendar.month_name[x])
    
    # Normalize to 0-100
    monthly_risk['event_risk'] = (monthly_risk['events'] - monthly_risk['events'].min()) / \
                                  (monthly_risk['events'].max() - monthly_risk['events'].min()) * 100
    monthly_risk['fatality_risk'] = (monthly_risk['fatalities'] - monthly_risk['fatalities'].min()) / \
                                     (monthly_risk['fatalities'].max() - monthly_risk['fatalities'].min()) * 100
    monthly_risk['composite_risk'] = (monthly_risk['event_risk'] + monthly_risk['fatality_risk']) / 2
    
    # Current and next months
    current_month = datetime.now().month
    next_month = (current_month % 12) + 1
    
    current_risk = monthly_risk[monthly_risk['month'] == current_month]['composite_risk'].values[0]
    next_risk = monthly_risk[monthly_risk['month'] == next_month]['composite_risk'].values[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Current ({calendar.month_name[current_month]})",
            f"{current_risk:.1f}/100",
            help="Risk score based on historical patterns"
        )
    
    with col2:
        st.metric(
            f"Next Month ({calendar.month_name[next_month]})",
            f"{next_risk:.1f}/100",
            delta=f"{next_risk - current_risk:+.1f}"
        )
    
    with col3:
        # Year-over-year trend
        yearly_totals = df.groupby('year')['fatalities'].sum()
        if len(yearly_totals) >= 2:
            yoy_change = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2] * 100)
            st.metric("YoY Trend", f"{yoy_change:+.1f}%", help="Year-over-year fatality change")
    
    # Risk calendar
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_risk['month_name'],
        y=monthly_risk['composite_risk'],
        marker=dict(
            color=monthly_risk['composite_risk'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score", x=1.15)
        ),
        text=monthly_risk['composite_risk'].round(1),
        textposition='auto',
        name='Risk Score'
    ))
    
    # Highlight current and next month
    fig.add_shape(
        type="rect",
        x0=current_month-1.5, x1=current_month-0.5,
        y0=0, y1=100,
        fillcolor="blue", opacity=0.1,
        line=dict(color="blue", width=2)
    )
    
    fig.update_layout(
        title="Monthly Risk Profile",
        xaxis_title="Month",
        yaxis_title="Composite Risk Score (0-100)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### 📋 Risk Mitigation Recommendations")
    
    high_risk = monthly_risk[monthly_risk['composite_risk'] > 70]
    medium_risk = monthly_risk[(monthly_risk['composite_risk'] > 40) & (monthly_risk['composite_risk'] <= 70)]
    
    if len(high_risk) > 0:
        st.error(f"**High-Risk Months**: {', '.join(high_risk['month_name'].values)}")
        st.markdown("""
        **Recommended Actions:**
        - ⚠️ Increase security posture and personnel
        - 🚫 Restrict non-essential travel
        - 📦 Pre-position emergency supplies
        - 📡 Enhance intelligence collection
        - 🏥 Ensure medical evacuation readiness
        """)
    
    if len(medium_risk) > 0:
        st.warning(f"**Medium-Risk Months**: {', '.join(medium_risk['month_name'].values)}")
        st.markdown("""
        **Recommended Actions:**
        - 👀 Maintain heightened awareness
        - 📋 Review and update contingency plans
        - 📊 Monitor local developments closely
        - 🔄 Establish check-in protocols
        """)
    
    # 90-day outlook
    st.markdown("### 📆 90-Day Outlook")
    
    next_3_months = [(current_month + i - 1) % 12 + 1 for i in range(1, 4)]
    outlook_df = monthly_risk[monthly_risk['month'].isin(next_3_months)].copy()
    outlook_df = outlook_df.sort_values('month')
    
    for _, row in outlook_df.iterrows():
        risk_level = "🔴 High" if row['composite_risk'] > 70 else "🟡 Medium" if row['composite_risk'] > 40 else "🟢 Low"
        st.write(f"**{row['month_name']}**: {risk_level} ({row['composite_risk']:.1f}/100)")

# Export section
st.markdown("---")
st.header("📥 Data Export")

col1, col2, col3 = st.columns(3)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        "📄 Download Raw Data",
        data=csv,
        file_name=f"acled_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Monthly summary
    monthly_summary = df.groupby(['year', 'month']).agg({
        'event_date': 'count',
        'fatalities': 'sum'
    }).reset_index()
    monthly_summary.columns = ['Year', 'Month', 'Events', 'Fatalities']
    
    summary_csv = monthly_summary.to_csv(index=False)
    st.download_button(
        "📊 Download Monthly Summary",
        data=summary_csv,
        file_name=f"monthly_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col3:
    if enable_weather and weather_df is not None:
        weather_csv = weather_df.to_csv(index=False)
        st.download_button(
            "🌦️ Download Weather Data",
            data=weather_csv,
            file_name=f"weather_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Methodology
with st.expander("📖 Methodology & Technical Notes"):
    st.markdown("""
    ## Data Sources
    
    ### ACLED  
    - Human-verified conflict events database  
    - Daily updates with 1-2 week verification lag  
    - Standardized event categories and actor coding  
    
    ### Open-Meteo Weather API  
    - Historical weather data (2020-present)  
    - Temperature, precipitation, rainfall metrics  
    - Aggregated to daily resolution  
    
    ### Hijri Converter  
    - Islamic calendar date conversion  
    - Ramadan, Eid, Muharram identification  
    
    ## Statistical Methods  
    
    ### ANOVA (Analysis of Variance)  
    - Tests if group means differ significantly  
    - Used for seasonal/monthly comparisons  
    - Assumes normal distribution  
    
    ### Kruskal-Wallis Test  
    - Non-parametric alternative to ANOVA  
    - Doesn't assume normal distribution  
    - More robust for skewed data  
    
    ### Chi-Square Test  
    - Tests if observed distribution matches expected  
    - Used for day-of-week analysis  
    
    ### Mann-Kendall Trend Test  
    - Detects monotonic trends over time  
    - Non-parametric, robust to outliers  
    
    ### Pearson Correlation  
    - Measures linear relationship strength  
    - Used for weather-violence correlations  
    - Range: -1 (perfect negative) to +1 (perfect positive)  
    
    ## Risk Scoring  
    
    Composite risk score = (Event Risk + Fatality Risk) / 2  
    
    Where:  
    - Event Risk = Normalized event frequency (0-100)  
    - Fatality Risk = Normalized fatality count (0-100)  
    
    ## Limitations  
    
    1. **Reporting Bias**: Events in urban areas or with media presence are overrepresented  
    2. **Temporal Lag**: ACLED data has 1-2 week verification delay  
    3. **Weather Proxy**: Single point weather data may not represent entire country  
    4. **Religious Calendar**: Simplified calculations, may miss regional variations  
    5. **Correlation ≠ Causation**: Statistical relationships don't prove causal mechanisms  
    
    ## Best Practices  
    
    - Use multi-year data for pattern identification  
    - Combine statistical findings with local knowledge  
    - Consider effect sizes, not just p-values  
    - Validate findings against current intelligence  
    - Update analysis as new data becomes available  
    
    ## Citation  
    
    When using this analysis, cite:  
    
    > Armed Conflict Location & Event Data Project (ACLED); www.acleddata.com  
    >   
    > Open-Meteo Historical Weather API; www.open-meteo.com  
    """)

# Footer
st.markdown("---")
st.caption(f"""
Data Sources: ACLED (OAuth), Open-Meteo | Built with: Streamlit, Plotly, SciPy
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} |
Countries: {', '.join(selected_countries)} | Events: {len(df):,}
""")