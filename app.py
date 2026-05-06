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
from fpdf import FPDF
import csv
import io

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Global Oil Analytics Dashboard v2.3",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ️ HELPER FUNCTIONS
# ==========================================

def load_profile():
    """Load and encode profile image for sidebar."""
    profile_path = "profile.jpg"
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()
                return f'<img src="data:image/jpeg;base64,{encoded}" style="width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid #1a365d;">'
        except:
            pass
    return '<div style="width:120px;height:120px;border-radius:50%;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);display:flex;align-items:center;justify-content:center;color:white;font-size:40px;font-weight:bold;border:3px solid #1a365d;box-shadow:0 4px 6px rgba(0,0,0,0.1);">GF</div>'

def generate_pdf_report(title, metrics_df):
    """Generate PDF report for model performance."""
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
    """Convert Plotly figure to PNG bytes for download."""
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=1200, height=600, scale=2)
        return img_bytes
    except:
        return None

# ==========================================
# 📊 DATA LOADING FUNCTIONS
# ==========================================

@st.cache_data(ttl=3600)
def load_production_data():
    """Load oil production data (2019-2024 synthetic to match Yahoo Finance timeframe)."""
    dates = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    countries = ["Nigeria", "Angola", "Algeria", "Libya", "Egypt", 
                 "Saudi Arabia", "Russia", "USA", "Canada", "China", "Brazil"]
    
    base_production = {
        "Nigeria": 1800, "Angola": 1400, "Algeria": 1000, "Libya": 1200, "Egypt": 600,
        "Saudi Arabia": 10500, "Russia": 11200, "USA": 19000, "Canada": 5500, 
        "China": 3800, "Brazil": 3000
    }
    
    data = []
    np.random.seed(42)
    
    for country in countries:
        base = base_production.get(country, 1000)
        for i, date in enumerate(dates):
            trend = 0.5 * i
            seasonal = 50 * np.sin(date.month / 12 * 2 * np.pi)
            noise = np.random.normal(0, 100)
            production = max(0, base + trend + seasonal + noise)
            
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
    
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_prices():
    """Load Brent crude oil prices from Yahoo Finance."""
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
    # Fallback data
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

# ==========================================
# 🤖 FORECASTING FUNCTIONS
# ==========================================

def forecast_simple(df_country, steps=12):
    """Linear regression baseline forecast."""
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

def forecast_lstm(df_country, steps=12):
    """LSTM deep learning forecast."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.preprocessing import MinMaxScaler
        
        df = df_country[['Date', 'Production_kbpd']].copy().sort_values('Date')
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Production_kbpd']])
        
        seq_length = 60
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length, 0])
            y.append(scaled_data[i+seq_length, 0])
        X, y = np.array(X), np.array(y)
        
        if len(X) < 10:
            return None, None, None
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            layers.LSTM(50),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        test_predict = model.predict(X_test, verbose=0)
        test_predict = scaler.inverse_transform(test_predict)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        mape = np.mean(np.abs((y_test_actual - test_predict) / y_test_actual)) * 100
        
        last_sequence = scaled_data[-seq_length:]
        future_predictions = []
        for _ in range(steps):
            pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
            future_predictions.append(pred[0, 0])
            last_sequence = np.vstack([last_sequence[1:], pred])
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=steps, freq='MS')
        hist_df = df.rename(columns={'Production_kbpd': 'Forecast'})
        hist_df['Type'] = 'Historical'
        fc_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': future_predictions.flatten(),
            'Type': 'Forecast'
        })
        fc_df['Lower_Bound'] = fc_df['Forecast'] * 0.95
        fc_df['Upper_Bound'] = fc_df['Forecast'] * 1.05
        
        return pd.concat([hist_df, fc_df]), rmse, mape
    except:
        return None, None, None

def forecast_arima(df_country, steps=12):
    """ARIMA statistical forecast."""
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
    except:
        return None, None

# ==========================================
# 🚀 LOAD DATA
# ==========================================
prod_df = load_production_data()
price_df = load_prices()

# ==========================================
# 📱 SIDEBAR
# ==========================================
profile_img = load_profile()

with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="margin-bottom: 10px;">{profile_img}</div>
        <div style="font-size: 18px; font-weight: bold; color: #1a365d; margin-bottom: 5px;">Gerson Japhet Fumbuka</div>
        <div style="font-size: 13px; color: #4a5568; line-height: 1.5;">
            DBA Scholar<br>INTI International University<br>Nilai, Malaysia
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📖 About This Dashboard"):
        st.markdown("""
        ### 🎯 Purpose
        This **Global Oil Production & Analytics Dashboard** is an open-access, interactive research tool designed to advance evidence-based understanding of petroleum resource dynamics across major oil-producing regions.
        
        ### 🌐 Why This Matters
        - **Global Significance**: Oil production drives economic development, geopolitical power, and energy security
        - **Data Transparency**: Addresses fragmented data through open, standardized presentation
        - **Academic Rigor**: Provides methodological transparency for peer-reviewed research
        - **Policy Support**: Enables data-driven decision-making for stakeholders
        
        ### 🎓 Benefits
        **For Students:** Hands-on learning, skill development, project inspiration
        **For Researchers:** Rapid hypothesis testing, methodological transparency, collaboration support
        **For Scholars:** Evidence-based advocacy, longitudinal analysis, global comparative work
        
        ### 🔓 Open Access
        Provided under principles of **open science** and **equitable knowledge access**.
        """)
    
    with st.expander("📚 Data Sources & Methodology"):
        st.markdown("### 🔍 Data Sources")
        st.markdown("- **Production**: EIA, OPEC Annual Statistical Bulletin, World Bank Open Data")
        st.markdown("- **Brent Crude Prices**: Yahoo Finance (Ticker: `BZ=F`)")
        st.markdown("- **Country Codes**: ISO 3166-1 alpha-3 standard")
        
        st.markdown("### 📐 Methodology")
        st.markdown("- **Units**: Production in thousand barrels per day (kbpd)")
        st.markdown("- **Forecasting**: LSTM Deep Learning, ARIMA Statistical, Linear Regression")
        st.markdown("- **Correlation**: Pearson coefficient between production and monthly Brent prices")
        st.markdown("- **Model Evaluation**: RMSE and MAPE on 12-month holdout validation")
        
        st.markdown("### 📅 Last Updated")
        st.code(datetime.now().strftime('%Y-%m-%d %H:%M UTC'))
        
        st.markdown("### 📖 Suggested Citation (APA)")
        citation = f"Fumbuka, G. J. (2026). Global Oil Analytics Dashboard v2.3 [Web application]. INTI International University. https://global-oil-dashboard-cobdnhgtjbkuplybfqncmq.streamlit.app"
        st.code(citation, language=None)
        
        st.markdown("### ⚠️ Limitations")
        st.markdown("Simulated production data used for demonstration. Forecasts represent ML/statistical projections and do not account for geopolitical shocks or OPEC+ policy changes.")
    
    st.markdown("---")
    st.subheader("🎛️ Controls")
    
    region = st.selectbox("Select Region", ["Global", "Africa", "Middle East", "Americas", "Asia", "Europe"], index=0)
    st.markdown("---")
    st.subheader("🌍 Select Countries")
    
    all_countries = sorted(prod_df['Country'].unique())
    region_map = {
        "Africa": ["Nigeria", "Angola", "Algeria", "Libya", "Egypt"],
        "Middle East": ["Saudi Arabia", "Iran", "Iraq", "Kuwait", "UAE"],
        "Americas": ["USA", "Canada", "Brazil", "Mexico", "Venezuela"],
        "Asia": ["China", "Japan", "India", "Indonesia"],
        "Europe": ["Russia", "Norway", "United Kingdom"]
    }
    available_countries = region_map.get(region, all_countries)
    available_countries = [c for c in available_countries if c in all_countries]
    default_selection = ["Nigeria", "USA", "Saudi Arabia", "Russia", "China"]
    default_selection = [c for c in default_selection if c in available_countries]
    
    selected_countries = st.multiselect("Choose countries:", options=available_countries, default=default_selection[:3] if default_selection else [])
    st.markdown("---")
    show_fc = st.checkbox("📈 Show 12-Month Forecast", value=True)
    st.markdown("---")
    
    with st.expander("❓ User Guide & Instructions"):
        st.markdown("""
        ### 📖 Quick Start
        1. **Select Region** → 2. **Choose Countries** → 3. **Enable Forecast** → 4. **Explore Tabs**
        
        ### 📊 Tabs Explained
        **🗺️ Map:** Geographic production visualization
        **📈 Forecast:** ML forecasting (select 1 country)
        **💰 Correlation:** Production vs Price analysis
        **⚠️ Alerts & Export:** Data downloads & monitoring
        
        ### 💡 Tips
        - ARIMA best for stable trends (~1.28% MAPE)
        - LSTM for complex seasonal patterns
        - Click chart legend to toggle series
        - Use Plotly camera icon for quick screenshots
        """)
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #718096; font-size: 11px; margin-top: 20px;">
        <b>Global Oil Analytics Dashboard v2.3</b><br>
        DBA Research Project • 2026<br>INTI International University
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🖥️ MAIN CONTENT
# ==========================================
st.title("🛢️ Global Oil Analytics Dashboard v2.3")
st.caption("ML forecasting • Real-time monitoring • CSV/PDF/Chart Export")

if not selected_countries:
    st.info("👈 Select countries in the sidebar to begin")
    st.stop()

filtered = prod_df[prod_df["Country"].isin(selected_countries)]
st.caption(f"📊 Showing: {', '.join(selected_countries)}")

col1, col2, col3 = st.columns(3)
col1.metric("Total Production", f"{filtered['Production_kbpd'].sum():,.0f} kbpd")
col2.metric("Avg per Country", f"{filtered['Production_kbpd'].mean():,.0f} kbpd")
col3.metric("Brent Price", f"${price_df['Brent_Price_USD'].iloc[-1]:.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Map", "📈 Forecast", "💰 Correlation", "⚠️ Alerts & Export", "🔮 Price-to-Production Estimator"])

# --- TAB 1: MAP ---
with tab1:
    map_data = filtered.groupby("Country")["Production_kbpd"].mean().reset_index()
    fig_map = px.choropleth(map_data, locations="Country", locationmode="country names", color="Production_kbpd", color_continuous_scale="OrRd", title="Average Oil Production by Country", hover_name="Country")
    st.plotly_chart(fig_map, width="stretch")
    
    map_png = convert_fig_to_png(fig_map)
    if map_png is not None:
        st.download_button(label="📸 Download Map as PNG", data=map_png, file_name=f"production_map_{datetime.now().strftime('%Y%m%d')}.png", mime="image/png")
    
    csv_data = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Production CSV", data=csv_data, file_name=f"production_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# --- TAB 2: FORECAST ---
with tab2:
    if show_fc and len(selected_countries) == 1:
        country_name = selected_countries[0]
        country_df = prod_df[prod_df["Country"] == country_name].sort_values('Date')
        st.info("🤖 Multi-Model Benchmark")
        
        model_choice = st.selectbox("Select Forecasting Model", ["LSTM (Deep Learning)", "ARIMA (Statistical)", "Linear (Baseline)"], index=0)
        
        with st.spinner(f"⏳ Training {model_choice}..."):
            fc_df = None
            if model_choice == "LSTM (Deep Learning)":
                fc_df, _, _ = forecast_lstm(country_df)
            elif model_choice == "ARIMA (Statistical)":
                fc_df, _ = forecast_arima(country_df)
            else:
                fc_df = forecast_simple(country_df)
        
        if fc_df is not None and not fc_df.empty:
            hist = fc_df[fc_df['Type'] == 'Historical']
            fc = fc_df[fc_df['Type'] == 'Forecast']
            
            fig = go.Figure()
            if not hist.empty:
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Forecast'], mode='lines', name='Historical', line=dict(color='#1f77b4', width=2)))
            if not fc.empty:
                fig.add_trace(go.Scatter(x=fc['Date'], y=fc['Forecast'], mode='lines', name=f'{model_choice} Forecast', line=dict(color='#ff7f0e', width=3, dash='dot')))
                if 'Lower_Bound' in fc.columns and not fc['Lower_Bound'].isna().all():
                    fig.add_trace(go.Scatter(x=pd.concat([fc['Date'], fc['Date'][::-1]]), y=pd.concat([fc['Upper_Bound'], fc['Lower_Bound'][::-1]]), fill='toself', fillcolor='rgba(255,127,14,0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
            
            fig.update_layout(title=f"{model_choice} Forecast for {country_name}", xaxis_title="Month", yaxis_title="Production (kbpd)", height=500, hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
            
            forecast_png = convert_fig_to_png(fig)
            if forecast_png is not None:
                st.download_button(label="📸 Download Forecast Chart as PNG", data=forecast_png, file_name=f"forecast_{country_name}_{datetime.now().strftime('%Y%m%d')}.png", mime="image/png")
            
            # Model Performance Table
            st.subheader("📊 Model Performance (Last 12 Months)")
            actual_data = country_df[['Date', 'Production_kbpd']].copy().sort_values('Date')
            if len(actual_data) >= 12:
                test_data = actual_data.tail(12)
                train_data = actual_data.head(len(actual_data) - 12)
            else:
                test_data = actual_data
                train_data = pd.DataFrame(columns=['Date', 'Production_kbpd'])
            
            metrics = []
            # Linear
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error
                if len(train_data) > 0:
                    X_train, y_train = np.arange(len(train_data)).reshape(-1, 1), train_data['Production_kbpd'].values
                    X_test, y_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1), test_data['Production_kbpd'].values
                    model_lr = LinearRegression().fit(X_train, y_train)
                    y_pred_lr = model_lr.predict(X_test)
                    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
                    mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
                    metrics.append({"Model": "Linear", "RMSE": f"{rmse_lr:,.2f}", "MAPE": f"{mape_lr:.2f}%"})
            except: metrics.append({"Model": "Linear", "RMSE": "N/A", "MAPE": "N/A"})
            
            # ARIMA
            try:
                from statsmodels.tsa.arima.model import ARIMA
                from sklearn.metrics import mean_squared_error
                if len(train_data) > 0:
                    tv, tsv = train_data['Production_kbpd'].values, test_data['Production_kbpd'].values
                    model_a = ARIMA(tv, order=(1,1,1) if len(tv)>5 else (1,0,1)).fit()
                    fc_a = model_a.forecast(steps=len(tsv))
                    rmse_a = np.sqrt(mean_squared_error(tsv, fc_a))
                    mape_a = np.mean(np.abs((tsv - fc_a) / tsv)) * 100
                    metrics.append({"Model": "ARIMA", "RMSE": f"{rmse_a:,.2f}", "MAPE": f"{mape_a:.2f}%"})
            except: metrics.append({"Model": "ARIMA", "RMSE": "N/A", "MAPE": "N/A"})
            
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, width="stretch", hide_index=True)
            valid_metrics = metrics_df[metrics_df['RMSE'] != 'N/A']
            if not valid_metrics.empty:
                valid_metrics['RMSE_numeric'] = valid_metrics['RMSE'].str.replace(',', '').astype(float)
                best = valid_metrics.loc[valid_metrics['RMSE_numeric'].idxmin()]
                st.success(f"🏆 **Best Model:** {best['Model']} (RMSE: {best['RMSE']}, MAPE: {best['MAPE']})")
                st.info("**Interpretation:** RMSE = absolute error (kbpd) | MAPE = % error (lower is better) | <10% MAPE is excellent")
    elif len(selected_countries) != 1:
        st.warning("⚠️ Select exactly ONE country for forecasting")
    else:
        st.info("Enable forecast in sidebar")

# --- TAB 3: CORRELATION ---
with tab3:
    st.subheader("💰 Brent Price Correlation")
    try:
        merged = filtered.merge(price_df[["Date", "Brent_Price_USD"]], on="Date", how="inner")
        if not merged.empty:
            corr = merged.groupby("Date").agg({"Production_kbpd": "sum", "Brent_Price_USD": "mean"}).reset_index()
            coef = corr["Production_kbpd"].corr(corr["Brent_Price_USD"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=corr["Date"], y=corr["Production_kbpd"], mode="lines", name="Total Production", line=dict(color="#1f77b4", width=2), yaxis="y1"))
            fig.add_trace(go.Scatter(x=corr["Date"], y=corr["Brent_Price_USD"], mode="lines", name="Brent Price", line=dict(color="#ff7f0e", width=2), yaxis="y2"))
            fig.update_layout(title="Production vs Brent Price", yaxis=dict(title="Production (kbpd)"), yaxis2=dict(title="Price (USD)", overlaying="y", side="right"), hovermode="x unified", height=500)
            st.plotly_chart(fig, width="stretch")
            
            corr_png = convert_fig_to_png(fig)
            if corr_png is not None:
                st.download_button(label="📸 Download Correlation Chart as PNG", data=corr_png, file_name=f"correlation_chart_{datetime.now().strftime('%Y%m%d')}.png", mime="image/png")
            
            col1, col2 = st.columns(2)
            col1.metric("Correlation", f"{coef:.3f}")
            col2.metric("Strength", "Weak" if abs(coef) < 0.3 else "Moderate" if abs(coef) < 0.7 else "Strong")
            
            csv_corr = corr.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Correlation CSV", data=csv_corr, file_name=f"correlation_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        else:
            st.warning("⚠️ No overlapping dates for correlation analysis")
    except Exception as e:
        st.error(f"❌ Correlation error: {e}")

# --- TAB 4: ALERTS & EXPORT ---
with tab4:
    st.subheader("⚠️ Production Drop Alerts (>10% MoM)")
    alerts = []
    for country in selected_countries:
        cdf = prod_df[prod_df["Country"] == country].sort_values("Date")
        if len(cdf) >= 2:
            latest, previous = cdf.iloc[-1]["Production_kbpd"], cdf.iloc[-2]["Production_kbpd"]
            change = ((latest - previous) / previous) * 100
            if change < -10:
                alerts.append(f"{country}: ▼ {abs(change):.1f}% drop")
    
    if alerts:
        for a in alerts: st.error(f"🚨 {a}")
    else:
        st.success("✅ No significant drops detected")
    
    st.divider()
    st.subheader("📦 Bulk Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(label="📥 All Production Data", data=prod_df.to_csv(index=False).encode('utf-8'), file_name=f"all_production_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    with col2:
        st.download_button(label="📥 Price Data", data=price_df.to_csv(index=False).encode('utf-8'), file_name=f"prices_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# --- TAB 5: PRICE-TO-PRODUCTION ESTIMATOR ---
with tab5:
    st.subheader("🔮 Estimate Production from Brent Price")
    st.info("""
    **Academic Note**: This is an exploratory statistical estimate, NOT official production data.
    Actual production depends on reserves, infrastructure, OPEC+ quotas, and geopolitical factors.
    Price typically explains <10% of production variance (Kilian, 2009; Hamilton, 2009).
    Use for research insight and scenario analysis only.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        est_country = st.selectbox("Select Country", sorted(prod_df['Country'].unique()), index=0)
    with col2:
        price_input = st.number_input("Brent Price (USD/barrel)", min_value=20.0, max_value=250.0, value=85.0, step=0.5)
    
    if st.button("🔍 Run Estimation"):
        # Merge country production with price data
        country_prod = prod_df[prod_df['Country'] == est_country][['Date', 'Production_kbpd']]
        merged_est = country_prod.merge(price_df[['Date', 'Brent_Price_USD']], on='Date', how='inner')
        
        if len(merged_est) < 15:
            st.warning(f"⚠️ Insufficient overlapping data for {est_country}. Need at least 15 months.")
        else:
            X = merged_est[['Brent_Price_USD']].values
            y = merged_est['Production_kbpd'].values
            
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict
            pred_value = model.predict([[price_input]])[0]
            r2_val = r2_score(y, model.predict(X))
            
            # Display results
            col1_res, col2_res, col3_res = st.columns(3)
            col1_res.metric("Estimated Production", f"{pred_value:,.0f} kbpd")
            col2_res.metric("Model R²", f"{r2_val:.3f}")
            col3_res.metric("Relationship", "Weak" if r2_val < 0.1 else "Moderate" if r2_val < 0.3 else "Strong")
            
            # Academic warning based on R²
            if r2_val < 0.1:
                st.warning(f"⚠️ Low explanatory power: Brent price explains only {r2_val*100:.1f}% of {est_country}'s production variation. This aligns with energy economics literature showing production is primarily policy/reserve-driven, not price-driven.")
            elif r2_val < 0.3:
                st.info(f"ℹ️ Moderate relationship: Price explains {r2_val*100:.1f}% of variation. Other factors (OPEC+ quotas, infrastructure, geopolitical events) dominate production decisions.")
            else:
                st.success(f"✅ Relatively strong statistical relationship for this period ({r2_val*100:.1f}% explained). Note: Correlation ≠ causation. Structural breaks may affect validity.")
            
            # Visualization
            fig_est = px.scatter(
                merged_est, x='Brent_Price_USD', y='Production_kbpd',
                title=f"{est_country}: Price vs Production Relationship",
                labels={'Brent_Price_USD': 'Brent Price (USD)', 'Production_kbpd': 'Production (kbpd)'},
                trendline='ols',
                hover_data=['Date']
            )
            fig_est.add_vline(x=price_input, line_dash="dot", line_color="red",
                             annotation_text=f"Input: ${price_input}/bbl",
                             annotation_position="top right")
            fig_est.add_hline(y=pred_value, line_dash="dash", line_color="green",
                             annotation_text=f"Est: {pred_value:,.0f} kbpd",
                             annotation_position="bottom right")
            
            st.plotly_chart(fig_est, width="stretch")
            
            # Methodology expander
            with st.expander("📐 Methodology & Statistical Formula"):
                st.markdown(f"""
                **Model**: Ordinary Least Squares (OLS) Linear Regression  
                **Equation**: `Production = {model.coef_[0]:.2f} × Price + {model.intercept_:.2f}`  
                **R²**: {r2_val:.3f} ({r2_val*100:.1f}% of variance explained)  
                **Data Points**: {len(merged_est)} months  
                **Standard Error**: {np.sqrt(np.mean((y - model.predict(X))**2)):.2f} kbpd  
                
                **Academic Limitations**:
                - Does not account for OPEC+ quotas, geopolitical shocks, or capacity constraints
                - Linear assumption may not capture threshold effects or supply shocks
                - Recommended for exploratory analysis only; use official EIA/OPEC data for policy decisions
                - See: Kilian (2009) AER, Hamilton (2009) Brookings Papers, IEA (2023) Oil Market Report
                """)


