import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import base64
import os
from prophet import Prophet
from fpdf import FPDF
import csv
import io

warnings.filterwarnings("ignore")
# Check if kaleido is available
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except:
    KALEIDO_AVAILABLE = False
# Page config
st.set_page_config(
    page_title="Global Oil Analytics Dashboard v2.3",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---

def load_profile():
    profile_path = "profile.jpg"
    if os.path.exists(profile_path):
        with open(profile_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            return f'<img src="data:image/jpeg;base64,{encoded}" style="width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid #1a365d;">'
    return '<div style="width:120px;height:120px;border-radius:50%;background:#e2e8f0;display:flex;align-items:center;justify-content:center;color:#718096;font-size:14px;">👤</div>'

def generate_pdf_report(title, metrics_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, "Model Performance Metrics", ln=True)
    pdf.set_font('Helvetica', '', 10)
    col_width = 90
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(col_width, 8, 'Model', border=1, fill=True)
    pdf.cell(col_width, 8, 'RMSE', border=1, fill=True, ln=True)
    for _, row in metrics_df.iterrows():
        pdf.cell(col_width, 8, str(row['Model']), border=1)
        pdf.cell(col_width, 8, str(row['RMSE']), border=1, ln=True)
    pdf.ln(10)
    valid = metrics_df[metrics_df['RMSE'] != 'N/A']
    if not valid.empty:
        best_model = valid.loc[valid['RMSE'].astype(float).idxmin(), 'Model']
        best_rmse = valid.loc[valid['RMSE'].astype(float).idxmin(), 'RMSE']
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 10, f"Best Model: {best_model} (RMSE: {best_rmse})", ln=True)
    pdf.set_y(-20)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Global Oil Analytics Dashboard | Gerson Japhet Fumbuka | INTI International University", ln=True, align='C')
    return pdf.output(dest='S').encode('latin1')

def convert_fig_to_png(fig):
    """Convert Plotly figure to PNG bytes."""
    try:
        import plotly.io as pio
        
        # Try kaleido engine (works locally and should work on Streamlit Cloud)
        img_bytes = pio.to_image(
            fig, 
            format="png", 
            width=1200, 
            height=600, 
            scale=2,
            engine="kaleido"
        )
        return img_bytes
        
    except Exception as e:
        # Log the error for debugging
        st.warning(f"⚠️ PNG export issue: {str(e)[:100]}")
        return None

# --- DATA LOADING FUNCTIONS ---

@st.cache_data(ttl=3600)
def load_production_data():
    """Load oil production data - using recent dates to match Yahoo Finance prices."""
    try:
        # Try to load from CSV if it exists and has recent data
        csv_path = "real_oil_data.csv"
        
        if os.path.exists(csv_path):
            # Read and parse CSV (same logic as before)
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find crude oil section
            start_idx = None
            for i, line in enumerate(lines):
                if 'crude oil including lease condensate production' in line.lower():
                    start_idx = i + 1
                    break
            
            if start_idx is not None:
                # Parse data (same as before)...
                # [Keep your existing CSV parsing logic here]
                # ...
                # If successful, return the dataframe
                # If it only has old data (pre-2018), we'll extend it below
                pass
    except:
        pass
    
    # FALLBACK: Generate synthetic production data with RECENT DATES (2019-2024)
    # This ensures overlap with Yahoo Finance price data
    st.info("ℹ️ Using simulated production data (2019-2024) to match Yahoo Finance prices")
    
    # Generate monthly data from 2019-2024
    dates = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    countries = ["Nigeria", "Angola", "Algeria", "Libya", "Egypt", 
                 "Saudi Arabia", "Russia", "USA", "Canada", "China", "Brazil"]
    
    # Base production values in kbpd (thousand barrels per day)
    base_production = {
        "Nigeria": 1800, "Angola": 1400, "Algeria": 1000, "Libya": 1200, "Egypt": 600,
        "Saudi Arabia": 10500, "Russia": 11200, "USA": 19000, "Canada": 5500, 
        "China": 3800, "Brazil": 3000
    }
    
    data = []
    np.random.seed(42)  # For reproducibility
    
    for country in countries:
        base = base_production.get(country, 1000)
        for i, date in enumerate(dates):
            # Add trend, seasonality, and noise
            trend = 0.5 * i  # Slight growth over time
            seasonal = 50 * np.sin(date.month / 12 * 2 * np.pi)  # Monthly seasonality
            noise = np.random.normal(0, 100)  # Random variation
            
            production = max(0, base + trend + seasonal + noise)
            
            # Determine region
            if country in ["Nigeria", "Angola", "Algeria", "Libya", "Egypt"]:
                region = "Africa"
            elif country in ["Saudi Arabia", "Iran", "Iraq", "Kuwait", "UAE"]:
                region = "Middle East"
            elif country in ["USA", "Canada", "Brazil", "Mexico", "Venezuela"]:
                region = "Americas"
            elif country in ["Russia", "Norway", "United Kingdom"]:
                region = "Europe"
            elif country in ["China", "India", "Indonesia"]:
                region = "Asia"
            else:
                region = "Global"
            
            data.append({
                "Country": country,
                "Date": date,
                "Production_kbpd": production,
                "Region": region
            })
    
    df = pd.DataFrame(data)
    st.success(f"✅ Loaded production  {len(df)} records for {df['Country'].nunique()} countries (2019-2024)")
    return df

@st.cache_data(ttl=3600)
def load_prices():
    tickers = ["BZ=F", "BRENTOIL=X", "CL=F"]
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5y", interval="1mo", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs('Close', axis=1, level=0, drop_level=True)
                else:
                    if 'Close' in df.columns:
                        df = df[['Close']]
                    elif 'Adj Close' in df.columns:
                        df = df[['Adj Close']]
                    else:
                        continue
                df.columns = ['Brent_Price_USD']
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
                df = df.dropna(subset=['Brent_Price_USD'])
                if not df.empty:
                    return df
        except:
            continue
    st.warning("⚠️ Could not fetch live prices. Using fallback data.")
    dates = pd.date_range(start="2018-01-01", end="2024-12-01", freq="MS")
    np.random.seed(42)
    base_price, volatility = 75.0, 15.0
    prices = []
    for i, d in enumerate(dates):
        trend = 0.002 * i
        seasonal = 5.0 * np.sin(2 * np.pi * d.month / 12)
        noise = np.random.normal(0, volatility)
        price = max(20, min(150, base_price + trend + seasonal + noise))
        prices.append(price)
    return pd.DataFrame({"Date": dates, "Brent_Price_USD": prices})

# --- FORECASTING FUNCTIONS ---
def forecast_simple(df_country, steps=12):
    x = np.arange(len(df_country))
    y = df_country["Production_kbpd"].values
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    future_x = np.arange(len(df_country), len(df_country) + steps)
    future_dates = [df_country["Date"].max() + timedelta(days=30*i) for i in range(1, steps+1)]
    fc_vals = poly(future_x)
    hist_df = df_country[["Date", "Production_kbpd"]].rename(columns={"Production_kbpd": "Forecast"})
    hist_df["Type"] = "Historical"
    fc_df = pd.DataFrame({"Date": future_dates, "Forecast": fc_vals, "Type": "Forecast"})
    return pd.concat([hist_df, fc_df])

def forecast_prophet(df_country, steps=12):
    try:
        df = df_country[["Date", "Production_kbpd"]].copy()
        df.columns = ['ds', 'y']
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
        model.fit(df)
        future = model.make_future_dataframe(periods=steps, freq='MS')
        forecast = model.predict(future)
        viz_df = pd.DataFrame({
            'Date': forecast['ds'],
            'Forecast': forecast['yhat'],
            'Lower_Bound': forecast['yhat_lower'],
            'Upper_Bound': forecast['yhat_upper'],
            'Type': ['Historical' if d < df['ds'].max() else 'Forecast' for d in forecast['ds']]
        })
        return viz_df, model, forecast
    except Exception as e:
        st.error(f"Prophet error: {e}")
        return None, None, None

def forecast_arima(df_country, steps=12):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        df = df_country.set_index('Date')['Production_kbpd'].dropna()
        if len(df) < 10:
            return None, None
        try:
            model = ARIMA(df, order=(1, 1, 1))
            results = model.fit()
        except:
            model = ARIMA(df, order=(1, 0, 1))
            results = model.fit()
        forecast_result = results.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=steps, freq='MS')
        hist_df = pd.DataFrame({'Date': df.index, 'Forecast': df.values, 'Lower_Bound': np.nan, 'Upper_Bound': np.nan, 'Type': 'Historical'})
        fc_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_mean.values, 'Lower_Bound': conf_int.iloc[:, 0].values, 'Upper_Bound': conf_int.iloc[:, 1].values, 'Type': 'Forecast'})
        return pd.concat([hist_df, fc_df]), results
    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        return None, None

# --- LOAD DATA FIRST (This fixes the UnboundLocalError) ---
prod_df = load_production_data()
price_df = load_prices()

# --- SIDEBAR (Now runs after data is loaded) ---
with st.sidebar:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown(load_profile(), unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Gerson Japhet Fumbuka</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-title">DBA Scholar<br>INTI International University<br>Nilai, Malaysia</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("📖 About This Dashboard"):
        st.markdown("""
        **Global Oil Analytics Dashboard v2.3** provides:
        - Real-time oil production monitoring
        - ML-powered forecasting with Prophet
        - Price correlation analysis
        - Interactive visualizations
        - Mobile-optimized interface
        - CSV/PDF export functionality
        """)
    
    st.divider()
    
    st.subheader("🎛️ Controls")
    
    # Region Selector
    region = st.selectbox(
        "Select Region",
        ["Global", "Africa", "Middle East", "Americas", "Asia", "Europe"],
        index=0
    )
    
    st.divider()
    
    # Country Selection
    st.subheader("🌍 Select Countries")
    
    # Filter countries based on selected region
    all_countries = sorted(prod_df['Country'].unique())
    
    if region == "Global":
        available_countries = all_countries
    elif region == "Africa":
        available_countries = ["Nigeria", "Angola", "Algeria", "Libya", "Egypt"]
    elif region == "Middle East":
        available_countries = ["Saudi Arabia", "Iran", "Iraq", "Kuwait", "UAE"]
    elif region == "Americas":
        available_countries = ["USA", "Canada", "Brazil", "Mexico", "Venezuela"]
    elif region == "Asia":
        available_countries = ["China", "Japan", "India", "Indonesia"]
    elif region == "Europe":
        available_countries = ["Russia", "Norway", "United Kingdom"]
    else:
        available_countries = all_countries
        
    # Filter available_countries to only those in data
    available_countries = [c for c in available_countries if c in all_countries]
    
    # Default selection
    default_selection = ["Nigeria", "USA", "Saudi Arabia", "Russia", "China"]
    default_selection = [c for c in default_selection if c in available_countries]
    
    selected_countries = st.multiselect(
        "Choose countries:",
        options=available_countries,
        default=default_selection[:3] if default_selection else [],
        help="Select one or more countries to analyze"
    )
    
    st.divider()
    
    show_fc = st.checkbox("📈 Show 12-Month Forecast", value=True)
    
    st.divider()
    
    with st.expander("❓ User Guide & Instructions Manual"):
        st.markdown("""
        ### 📖 Quick Start Guide
        **Step 1:** Select region from dropdown  
        **Step 2:** Choose countries to analyze  
        **Step 3:** Enable "Show 12-Month Forecast"  
        **Step 4:** Navigate tabs...
        """)

# --- MAIN CONTENT ---
st.title("🛢️ Global Oil Analytics Dashboard v2.3")
st.caption("ML forecasting • Real-time monitoring • CSV/PDF/Chart Export")

if not selected_countries:
    st.info("👈 Select countries in the sidebar to begin")
    st.stop()

filtered = prod_df[prod_df["Country"].isin(selected_countries)]
st.caption(f"📊 Showing: {', '.join(selected_countries)}")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Production", f"{filtered['Production_kbpd'].sum():,.0f} kbpd")
col2.metric("Avg per Country", f"{filtered['Production_kbpd'].mean():,.0f} kbpd")
col3.metric("Brent Price", f"${price_df['Brent_Price_USD'].iloc[-1]:.2f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "📈 Forecast", "💰 Correlation", "⚠️ Alerts & Export"])

# Tab 1: Map
with tab1:
    map_data = filtered.groupby("Country")["Production_kbpd"].mean().reset_index()
    fig_map = px.choropleth(
        map_data,
        locations="Country",
        locationmode="country names",
        color="Production_kbpd",
        color_continuous_scale="OrRd",
        title="Average Oil Production by Country",
        hover_name="Country"
    )
    st.plotly_chart(fig_map, width="stretch")
    
    map_png = convert_fig_to_png(fig_map)
    if map_png is not None:
        st.download_button(
            label="📸 Download Map as PNG",
            data=map_png,
            file_name=f"production_map_{datetime.now().strftime('%Y%m%d')}.png",
            mime="image/png"
        )
    
    csv_data = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Production CSV",
        data=csv_data,
        file_name=f"production_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Tab 2: Forecast
with tab2:
    if show_fc and len(selected_countries) == 1:
        country_name = selected_countries[0]
        country_df = prod_df[prod_df["Country"] == country_name].sort_values('Date')
        st.info("🤖 Multi-Model Benchmark")
        
        model_choice = st.selectbox(
            "Select Forecasting Model",
            ["Prophet (ML)", "ARIMA (Statistical)", "Linear (Baseline)"],
            index=0
        )
        
        with st.spinner(f"⏳ Training {model_choice}..."):
            fc_df = None
            if model_choice == "Prophet (ML)":
                fc_df, _, _ = forecast_prophet(country_df)
            elif model_choice == "ARIMA (Statistical)":
                fc_df, _ = forecast_arima(country_df)
            else:
                fc_df = forecast_simple(country_df)
        
        if fc_df is not None and not fc_df.empty:
            hist = fc_df[fc_df['Type'] == 'Historical']
            fc = fc_df[fc_df['Type'] == 'Forecast']
            
            fig = go.Figure()
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist['Date'], y=hist['Forecast'],
                    mode='lines', name='Historical',
                    line=dict(color='#1f77b4', width=2)
                ))
            if not fc.empty:
                fig.add_trace(go.Scatter(
                    x=fc['Date'], y=fc['Forecast'],
                    mode='lines', name=f'{model_choice} Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dot')
                ))
                if 'Lower_Bound' in fc.columns and not fc['Lower_Bound'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc['Date'], fc['Date'][::-1]]),
                        y=pd.concat([fc['Upper_Bound'], fc['Lower_Bound'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255,127,14,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% CI'
                    ))
            
            fig.update_layout(
                title=f"{model_choice} Forecast for {country_name}",
                xaxis_title="Month",
                yaxis_title="Production (kbpd)",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, width="stretch")
            
            forecast_png = convert_fig_to_png(fig)
            if forecast_png is not None:
                st.download_button(
                    label="📸 Download Forecast Chart as PNG",
                    data=forecast_png,
                    file_name=f"forecast_{country_name}_{datetime.now().strftime('%Y%m%d')}.png",
                    mime="image/png"
                )
            
           # Model Performance (Last 12 Years - since data is annual)
st.subheader("📊 Model Performance (Last 12 Years)")

# Get the actual production data for the selected country
# Create country_df for the selected country
if len(selected_countries) == 1:
    country_df = prod_df[prod_df['Country'] == selected_countries[0]].sort_values('Date')
else:
    st.error("Please select exactly one country for model performance analysis")
    st.stop()

# Now the existing line will work
actual_data = country_df[['Date', 'Production_kbpd']].copy()
actual_data = actual_data.sort_values('Date')

# Use last 12 years for validation (not months, since data is annual)
if len(actual_data) >= 12:
    test_data = actual_data.tail(12)
    train_data = actual_data.head(len(actual_data) - 12)
else:
    test_data = actual_data
    train_data = pd.DataFrame(columns=['Date', 'Production_kbpd'])

# Calculate RMSE for each model
metrics = []

# Linear Regression RMSE
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    if len(train_data) > 0:
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data['Production_kbpd'].values
        
        X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
        y_test = test_data['Production_kbpd'].values
        
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
        
        metrics.append({
            "Model": "Linear",
            "RMSE": f"{rmse_lr:,.2f}",
            "MAPE": f"{mape_lr:.2f}%"
        })
except Exception as e:
    metrics.append({"Model": "Linear", "RMSE": "N/A", "MAPE": "N/A"})

# Prophet RMSE
try:
    from prophet import Prophet
    from sklearn.metrics import mean_squared_error
    
    if len(train_data) > 0:
        df_p = train_data.rename(columns={'Date': 'ds', 'Production_kbpd': 'y'})
        model_p = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model_p.fit(df_p)
        
        future_p = model_p.make_future_dataframe(periods=len(test_data), freq='Y')
        forecast_p = model_p.predict(future_p)
        
        # Get predictions for test period
        pred_p = forecast_p.tail(len(test_data))['yhat'].values
        y_test = test_data['Production_kbpd'].values
        
        if len(pred_p) == len(y_test):
            rmse_p = np.sqrt(mean_squared_error(y_test, pred_p))
            mape_p = np.mean(np.abs((y_test - pred_p) / y_test)) * 100
            
            metrics.append({
                "Model": "Prophet",
                "RMSE": f"{rmse_p:,.2f}",
                "MAPE": f"{mape_p:.2f}%"
            })
        else:
            metrics.append({"Model": "Prophet", "RMSE": "N/A", "MAPE": "N/A"})
    else:
        metrics.append({"Model": "Prophet", "RMSE": "N/A", "MAPE": "N/A"})
except Exception as e:
    metrics.append({"Model": "Prophet", "RMSE": "N/A", "MAPE": "N/A"})

# ARIMA RMSE
try:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    
    if len(train_data) > 0:
        train_values = train_data['Production_kbpd'].values
        test_values = test_data['Production_kbpd'].values
        
        # Fit ARIMA model
        try:
            model_a = ARIMA(train_values, order=(1, 1, 1))
        except:
            model_a = ARIMA(train_values, order=(1, 0, 1))
        
        fitted_a = model_a.fit()
        
        # Forecast
        forecast_a = fitted_a.forecast(steps=len(test_values))
        
        rmse_a = np.sqrt(mean_squared_error(test_values, forecast_a))
        mape_a = np.mean(np.abs((test_values - forecast_a) / test_values)) * 100
        
        metrics.append({
            "Model": "ARIMA",
            "RMSE": f"{rmse_a:,.2f}",
            "MAPE": f"{mape_a:.2f}%"
        })
    else:
        metrics.append({"Model": "ARIMA", "RMSE": "N/A", "MAPE": "N/A"})
except Exception as e:
    metrics.append({"Model": "ARIMA", "RMSE": "N/A", "MAPE": "N/A"})

# Display metrics table
metrics_df = pd.DataFrame(metrics)
st.dataframe(metrics_df, width="stretch", hide_index=True)

# Find best model based on RMSE
valid_metrics = metrics_df[metrics_df['RMSE'] != 'N/A']
if not valid_metrics.empty:
    # Convert RMSE to numeric for comparison
    valid_metrics['RMSE_numeric'] = valid_metrics['RMSE'].str.replace(',', '').astype(float)
    best_model_row = valid_metrics.loc[valid_metrics['RMSE_numeric'].idxmin()]
    
    st.success(f"🏆 **Best Model:** {best_model_row['Model']} (RMSE: {best_model_row['RMSE']}, MAPE: {best_model_row['MAPE']})")
    
    # Add interpretation
    st.info(f"""
    **Interpretation:**
    - RMSE measures absolute error in kbpd
    - MAPE measures percentage error (lower is better)
    - For Russia's production scale (~10M kbpd), a MAPE < 10% is excellent
    """) 
            
# Tab 3: Correlation
with tab3:
    st.subheader("💰 Brent Price Correlation")
    try:
        merged = filtered.merge(price_df[["Date", "Brent_Price_USD"]], on="Date", how="inner")
        if not merged.empty:
            corr = merged.groupby("Date").agg({
                "Production_kbpd": "sum",
                "Brent_Price_USD": "mean"
            }).reset_index()
            coef = corr["Production_kbpd"].corr(corr["Brent_Price_USD"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=corr["Date"], y=corr["Production_kbpd"],
                mode="lines", name="Total Production",
                line=dict(color="#1f77b4", width=2), yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=corr["Date"], y=corr["Brent_Price_USD"],
                mode="lines", name="Brent Price",
                line=dict(color="#ff7f0e", width=2), yaxis="y2"
            ))
            fig.update_layout(
                title="Production vs Brent Price",
                yaxis=dict(title="Production (kbpd)"),
                yaxis2=dict(title="Price (USD)", overlaying="y", side="right"),
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, width="stretch")
            
            corr_png = convert_fig_to_png(fig)
            if corr_png is not None:
                st.download_button(
                    label="📸 Download Correlation Chart as PNG",
                    data=corr_png,
                    file_name=f"correlation_chart_{datetime.now().strftime('%Y%m%d')}.png",
                    mime="image/png"
                )
            
            col1, col2 = st.columns(2)
            col1.metric("Correlation", f"{coef:.3f}")
            col2.metric("Strength", "Weak" if abs(coef) < 0.3 else "Moderate" if abs(coef) < 0.7 else "Strong")
            
            csv_corr = corr.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Correlation CSV",
                data=csv_corr,
                file_name=f"correlation_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No overlapping dates for correlation analysis")
    except Exception as e:
        st.error(f"❌ Correlation error: {e}")

# Tab 4: Alerts & Export
with tab4:
    st.subheader("⚠️ Production Drop Alerts (>10% MoM)")
    alerts = []
    for country in selected_countries:
        cdf = prod_df[prod_df["Country"] == country].sort_values("Date")
        if len(cdf) >= 2:
            latest = cdf.iloc[-1]["Production_kbpd"]
            previous = cdf.iloc[-2]["Production_kbpd"]
            change = ((latest - previous) / previous) * 100
            if change < -10:
                alerts.append(f"{country}: ▼ {abs(change):.1f}% drop")
    
    if alerts:
        for a in alerts:
            st.error(f"🚨 {a}")
    else:
        st.success("✅ No significant drops detected")
    
    st.divider()
    st.subheader("📦 Bulk Export")
    col1, col2 = st.columns(2)
    with col1:
        csv_all = prod_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 All Production Data",
            data=csv_all,
            file_name=f"all_production_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col2:
        csv_prices = price_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Price Data",
            data=csv_prices,
            file_name=f"prices_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )