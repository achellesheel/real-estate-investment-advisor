import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Script directory — all file paths are resolved relative to this
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Indian Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Google Font: Outfit, Slate Premium Dark Theme look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium visual borders */
    .metric-card {
        background: #F8F9FC;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E3E6F0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-bottom: 15px;
    }
    
    .investment-badge-good {
        background-color: #D4EDDA;
        color: #155724;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #C3E6CB;
    }

    .investment-badge-bad {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #F5C6CB;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #4E73DF;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #858796;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. CACHED DATA & PIPELINE LOADERS
# ----------------------------------------------------
@st.cache_data
def load_housing_data():
    """Load and cache raw housing data for filters and visual charts.
    Uses full CSV if available locally, falls back to the lightweight sample CSV on Cloud.
    """
    full_csv = os.path.join(SCRIPT_DIR, 'india_housing_prices.csv')
    sample_csv = os.path.join(SCRIPT_DIR, 'india_housing_prices_sample.csv')
    
    if os.path.exists(full_csv):
        df = pd.read_csv(full_csv).iloc[:, 0:23]
    elif os.path.exists(sample_csv):
        df = pd.read_csv(sample_csv)
    else:
        # Return empty dataframe with expected schema so app doesn't crash
        return pd.DataFrame()
    
    df = df.drop_duplicates()
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']
    df['Age_of_Property'] = 2025 - df['Year_Built']
    return df

@st.cache_resource
def load_ml_assets():
    """Load classifier, regressor, and preprocessing pipelines."""
    with open(os.path.join(SCRIPT_DIR, "preprocessing_assets.pkl"), "rb") as f:
        preproc = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, "best_classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, "best_regressor.pkl"), "rb") as f:
        regressor = pickle.load(f)
        
    return {
        "scaler": preproc["scaler"],
        "pt": preproc["power_transformer"],
        "locality_map": preproc["locality_map"],
        "city_map": preproc["city_map"],
        "global_mean_price": preproc["global_mean_price"],
        "continuous_cols": preproc["continuous_cols"],
        "feature_names": feature_names,
        "classifier": classifier,
        "regressor": regressor
    }

# Check if model files are trained and saved
if not os.path.exists(os.path.join(SCRIPT_DIR, "preprocessing_assets.pkl")) or not os.path.exists(os.path.join(SCRIPT_DIR, "best_classifier.pkl")):
    st.error("⚠️ MLflow model pipeline files not found! Please run the training pipeline first: `python train_and_track.py`")
    st.stop()

# Initialize data and pipelines
df_raw = load_housing_data()
ml_assets = load_ml_assets()

# ----------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/real-estate.png", width=80)
st.sidebar.title("Advisor Panel")
st.sidebar.markdown("Indian Real Estate Investment Evaluation & Value Forecasting")

menu = st.sidebar.radio(
    "Go To",
    ["📊 Market Insights", "🔮 Property Predictor", "🔍 Catalog Explorer"]
)

# ----------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW & INSIGHTS
# ----------------------------------------------------
if menu == "📊 Market Insights":
    st.title("📊 Indian Real Estate Market Insights")
    st.markdown("Explore macroeconomic benchmarks, pricing distributions, and geographical pricing indexes.")

    # High-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties Analyzed", f"{len(df_raw):,}")
    with col2:
        st.metric("Average Listing Price", f"₹{df_raw['Price_in_Lakhs'].mean():.2f} Lakhs")
    with col3:
        st.metric("Average Price / SqFt", f"₹{df_raw['Price_per_SqFt'].mean() * 100000:.0f} / SqFt")
    with col4:
        st.metric("Median Property Age", f"{df_raw['Age_of_Property'].median():.0f} Years")

    st.markdown("---")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Price per SqFt by State (Top 15)
        state_prices = df_raw.groupby('State')['Price_per_SqFt'].mean().reset_index()
        state_prices['Price_per_SqFt_Rs'] = state_prices['Price_per_SqFt'] * 100000
        state_prices = state_prices.sort_values(by='Price_per_SqFt_Rs', ascending=False)
        fig_state = px.bar(
            state_prices, x='Price_per_SqFt_Rs', y='State', orientation='h',
            title="Average Price per SqFt by State (INR)",
            labels={"Price_per_SqFt_Rs": "Price per SqFt (₹)", "State": "State"},
            color="Price_per_SqFt_Rs", color_continuous_scale="Viridis",
            height=450
        )
        st.plotly_chart(fig_state, use_container_width=True)

    with row1_col2:
        # Average Property Price by City (Top 15)
        city_prices = df_raw.groupby('City')['Price_in_Lakhs'].mean().reset_index().sort_values(by='Price_in_Lakhs', ascending=False).head(15)
        fig_city = px.bar(
            city_prices, x='City', y='Price_in_Lakhs',
            title="Top 15 Most Expensive Cities (Average Price in Lakhs)",
            labels={"Price_in_Lakhs": "Avg Price (₹ Lakhs)", "City": "City"},
            color="Price_in_Lakhs", color_continuous_scale="Cividis",
            height=450
        )
        st.plotly_chart(fig_city, use_container_width=True)

    st.markdown("---")
    
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Price Distribution
        fig_price_dist = px.histogram(
            df_raw, x="Price_in_Lakhs", nbins=50,
            title="Property Price Distribution Spread",
            labels={"Price_in_Lakhs": "Price (₹ Lakhs)"},
            color_discrete_sequence=["#4E73DF"]
        )
        st.plotly_chart(fig_price_dist, use_container_width=True)

    with row2_col2:
        # BHK Distribution
        bhk_counts = df_raw['BHK'].value_counts().reset_index()
        bhk_counts.columns = ['BHK', 'Count']
        fig_bhk = px.pie(
            bhk_counts, values='Count', names='BHK',
            title="BHK Size Configuration Share",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_bhk, use_container_width=True)

# ----------------------------------------------------
# TAB 2: PROPERTY ADVISOR PREDICTOR
# ----------------------------------------------------
elif menu == "🔮 Property Predictor":
    st.title("🔮 Real Estate Evaluator & Future Price Predictor")
    st.markdown("Enter property details below to predict investment health and forecast pricing appreciation.")

    # ── Cascaded dropdowns OUTSIDE the form so they react immediately ──
    st.subheader("📋 Property Specification Details")
    loc_col1, loc_col2, loc_col3 = st.columns(3)
    with loc_col1:
        state = st.selectbox("State", options=sorted(df_raw['State'].dropna().unique()), key="sel_state")
    with loc_col2:
        available_cities = sorted(df_raw[df_raw['State'] == state]['City'].dropna().unique())
        city = st.selectbox("City", options=available_cities, key="sel_city")
    with loc_col3:
        available_localities = sorted(df_raw[df_raw['City'] == city]['Locality'].dropna().unique())
        locality = st.selectbox("Locality", options=available_localities if available_localities else ["N/A"], key="sel_locality")

    # ── Rest of the form ──
    with st.form("property_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
            bhk = st.slider("BHK Config", 1, 5, 3)

        with col2:
            size = st.number_input("Size in SqFt", min_value=100.0, max_value=20000.0, value=1500.0, step=50.0)
            price = st.number_input("Current Listing Price (₹ in Lakhs)", min_value=5.0, max_value=5000.0, value=150.0, step=5.0)
            year_built = st.slider("Year Built", 1980, 2024, 2015)
            floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=3)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=10)

        with col3:
            facing = st.selectbox("Facing Direction", ["East", "North", "South", "West"])
            owner_type = st.selectbox("Owner Listing Type", ["Broker", "Builder", "Owner"])
            furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
            transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
            parking = st.radio("Dedicated Parking Space Available?", ["No", "Yes"], index=1)
            security = st.radio("Security Staff Present?", ["No", "Yes"], index=1)
            availability = st.selectbox("Availability Status", ["Under Construction", "Ready to Move"])

        st.subheader("🏡 Society Amenities")
        amenities = st.multiselect(
            "Select all active amenities in the property/society:",
            ["Clubhouse", "Garden", "Gym", "Playground", "Pool"],
            default=["Garden", "Gym"]
        )

        submit_btn = st.form_submit_button("🔍 Run Investment Valuation Analysis")

    if submit_btn:
        # Preprocessing user input to match model format
        scaler = ml_assets["scaler"]
        pt = ml_assets["pt"]
        locality_map = ml_assets["locality_map"]
        city_map = ml_assets["city_map"]
        global_mean_price = ml_assets["global_mean_price"]
        continuous_cols = ml_assets["continuous_cols"]
        feature_names = ml_assets["feature_names"]
        classifier = ml_assets["classifier"]
        regressor = ml_assets["regressor"]
        
        # Encodings
        transport_map = {'Low': 0, 'Medium': 1, 'High': 2}
        furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
        binary_map = {'No': 0, 'Yes': 1}
        availability_map = {'Under Construction': 0, 'Ready to Move': 1}
        
        # Compute variables
        price_per_sqft = price / size
        age_of_property = 2025 - year_built
        
        # Transform price_per_sqft
        price_per_sqft_transformed = pt.transform([[price_per_sqft]])[0][0]
        
        # Lookups
        locality_encoded = locality_map.get(locality, global_mean_price)
        city_encoded = city_map.get(city, global_mean_price)
        
        # Map values
        raw_inputs = {
            'BHK': bhk,
            'Size_in_SqFt': size,
            'Price_in_Lakhs': price,
            'Price_per_SqFt': price_per_sqft,
            'Year_Built': year_built,
            'Furnished_Status': furnished_map[furnished],
            'Floor_No': floor_no,
            'Total_Floors': total_floors,
            'Age_of_Property': age_of_property,
            'Nearby_Schools': 6, # Median or fixed average
            'Nearby_Hospitals': 5, # Median
            'Public_Transport_Accessibility': transport_map[transport],
            'Parking_Space': binary_map[parking],
            'Security': binary_map[security],
            'Availability_Status': availability_map[availability],
            'Price_per_SqFt_Transformed': price_per_sqft_transformed,
            'Locality_Target_Encoded': locality_encoded,
            'City_Target_Encoded': city_encoded,
            
            # Amenities dummies
            'Amenity_Clubhouse': 1 if 'Clubhouse' in amenities else 0,
            'Amenity_Garden': 1 if 'Garden' in amenities else 0,
            'Amenity_Gym': 1 if 'Gym' in amenities else 0,
            'Amenity_Playground': 1 if 'Playground' in amenities else 0,
            'Amenity_Pool': 1 if 'Pool' in amenities else 0,
            
            # Categories (one hot)
            'Facing_East': 1 if facing == 'East' else 0,
            'Facing_North': 1 if facing == 'North' else 0,
            'Facing_South': 1 if facing == 'South' else 0,
            'Facing_West': 1 if facing == 'West' else 0,
            'Owner_Type_Broker': 1 if owner_type == 'Broker' else 0,
            'Owner_Type_Builder': 1 if owner_type == 'Builder' else 0,
            'Owner_Type_Owner': 1 if owner_type == 'Owner' else 0,
            'Property_Type_Apartment': 1 if property_type == 'Apartment' else 0,
            'Property_Type_Independent House': 1 if property_type == 'Independent House' else 0,
            'Property_Type_Villa': 1 if property_type == 'Villa' else 0,
        }
        
        # Ensure all columns in feature_names are present
        for col in feature_names:
            if col not in raw_inputs:
                raw_inputs[col] = 0
                
        # Build DataFrame in precise column order
        X_new = pd.DataFrame([raw_inputs])[feature_names]
        
        # Scale continuous features
        X_new[continuous_cols] = scaler.transform(X_new[continuous_cols])
        
        # Make predictions
        pred_class = classifier.predict(X_new)[0]
        prob = classifier.predict_proba(X_new)[0][1] if hasattr(classifier, "predict_proba") else (1.0 if pred_class else 0.0)
        
        pred_future_price = regressor.predict(X_new)[0]
        
        # Output evaluation section
        st.markdown("---")
        st.header("🎯 Evaluation Results")
        
        res_col1, res_col2 = st.columns(2)
        
        # Classification evaluation display
        with res_col1:
            st.subheader("Classification: Investment Grade")
            if pred_class == 1:
                st.markdown("<div class='investment-badge-good'>🟢 GOOD INVESTMENT</div>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Score:** `{prob * 100:.2f}%` likelihood of outperforming local metrics.")
            else:
                st.markdown("<div class='investment-badge-bad'>🔴 NOT A RECOMMENDED INVESTMENT</div>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Score:** `{100 - (prob * 100):.2f}%` likelihood of underperforming local metrics.")
            
            # Price vs City Median rules benchmarks
            city_data = df_raw[df_raw['City'] == city]
            median_city_price = city_data['Price_in_Lakhs'].median() if not city_data.empty else global_mean_price
            median_city_sqft = city_data['Price_per_SqFt'].median() if not city_data.empty else (global_mean_price / 1500)
            
            st.markdown("### 📋 Rule-Based Investment Benchmarks:")
            st.write(f"- **Price vs. City Median Price**: Current price (₹{price:.1f}L) vs. City Median (₹{median_city_price:.1f}L) ➔ " + ("✅ Below Median (Favorable)" if price <= median_city_price else "❌ Above Median"))
            st.write(f"- **Price/SqFt vs. City Median Price/SqFt**: Unit price (₹{price_per_sqft*100000:.0f}) vs. City Median (₹{median_city_sqft*100000:.0f}) ➔ " + ("✅ Cheaper per SqFt" if price_per_sqft < median_city_sqft else "❌ More Expensive per SqFt"))
            # Multi-factor 3-tier benchmark
            if bhk >= 3 and len(amenities) >= 2:
                multi_label = "✅ Premium layout (≥3 BHK & ≥2 amenities)"
            elif bhk >= 2 or len(amenities) >= 1:
                multi_label = "⚠️ Standard configuration"
            else:
                multi_label = "❌ Minimal layout — low appeal"
            st.write(f"- **Multi-factor Score**: BHK ({bhk} BHK) and Amenities ({len(amenities)} selected) ➔ {multi_label}")

        # Regression valuation display
        with res_col2:
            st.subheader("Regression: 5-Year Price Forecast")
            
            # Forecast calculations
            fixed_8_price = price * (1.08 ** 5)
            
            appreciation_rate = 0.05
            appreciation_rate += 0.01 * (1 if property_type == "Villa" else 0)
            appreciation_rate += 0.01 * (1 if transport == "High" else 0)
            appreciation_rate += 0.005 * (1 if 6 >= 5 else 0) # school benchmark
            appreciation_rate += 0.005 * (1 if 5 >= 5 else 0) # hospital benchmark
            appreciation_rate += 0.01 * (1 if age_of_property <= 5 else 0)
            appreciation_rate -= 0.01 * (1 if age_of_property >= 20 else 0)
            formula_price = price * ((1 + appreciation_rate) ** 5)
            
            # CAGR — guard against negative/NaN predictions
            if pred_future_price > 0 and price > 0:
                cagr = ((pred_future_price / price) ** (1/5) - 1) * 100
            else:
                cagr = 0.0
            
            st.metric("Estimated Price in 5 Years (ML Model)", f"₹{pred_future_price:.2f} Lakhs", f"CAGR: {cagr:.2f}%")
            
            # Visual comparison of growth forecasts
            fig_compare = go.Figure(data=[
                go.Bar(name='Current Price', x=['Current'], y=[price], marker_color='#858796'),
                go.Bar(name='ML Model Prediction', x=['5-Year Forecast'], y=[pred_future_price], marker_color='#1CC88A'),
                go.Bar(name='Fixed 8% appreciation', x=['5-Year Forecast'], y=[fixed_8_price], marker_color='#36B9CC'),
                go.Bar(name='Domain Formula Baseline', x=['5-Year Forecast'], y=[formula_price], marker_color='#F6C23E')
            ])
            fig_compare.update_layout(
                title="Forecast Comparison (₹ Lakhs)",
                barmode='group',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Model Interpretation (Feature Importance)")
        
        # Calculate feature importance
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            coef = classifier.coef_
            if len(coef.shape) > 1:
                coef = coef[0]
            importances = np.abs(coef)
        else:
            importances = np.zeros(X_new.shape[1])
            
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)
        
        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            title="Top 10 Most Influential Features Driving the Predictions",
            color="Importance", color_continuous_scale="Blues",
            labels={"Importance": "Feature Influence Score", "Feature": "Feature Name"}
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------------------------------
# TAB 3: CATALOG EXPLORER & FILTERS
# ----------------------------------------------------
elif menu == "🔍 Catalog Explorer":
    st.title("🔍 Historical Property Catalog Explorer")
    st.markdown("Filter and search through the historical properties dataset.")
    
    # Filters Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        f_state = st.multiselect("Filter State", options=sorted(df_raw['State'].dropna().unique()))
    with col2:
        available_cities = df_raw[df_raw['State'].isin(f_state)]['City'].dropna().unique() if f_state else df_raw['City'].dropna().unique()
        f_city = st.multiselect("Filter City", options=sorted(available_cities))
    with col3:
        f_prop_type = st.multiselect("Filter Property Type", options=sorted(df_raw['Property_Type'].dropna().unique()))
    with col4:
        f_bhk = st.multiselect("Filter BHK", options=sorted(df_raw['BHK'].unique()), default=[2, 3, 4])
        
    price_min, price_max = float(df_raw['Price_in_Lakhs'].min()), float(df_raw['Price_in_Lakhs'].max())
    f_price = st.slider("Filter Price Range (₹ Lakhs)", price_min, price_max, (price_min, price_max / 3))
    
    # Filter dataset
    filtered_df = df_raw.copy()
    if f_state:
        filtered_df = filtered_df[filtered_df['State'].isin(f_state)]
    if f_city:
        filtered_df = filtered_df[filtered_df['City'].isin(f_city)]
    if f_prop_type:
        filtered_df = filtered_df[filtered_df['Property_Type'].isin(f_prop_type)]
    if f_bhk:
        filtered_df = filtered_df[filtered_df['BHK'].isin(f_bhk)]
        
    filtered_df = filtered_df[(filtered_df['Price_in_Lakhs'] >= f_price[0]) & (filtered_df['Price_in_Lakhs'] <= f_price[1])]
    
    st.metric("Properties Matching Filter Criteria", f"{len(filtered_df):,}")
    
    # Plotly Scatter plots of filtered data
    row_col1, row_col2 = st.columns(2)
    with row_col1:
        fig_scatter_size = px.scatter(
            filtered_df, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type",
            title="Price vs. Size Spread of Listings",
            labels={"Size_in_SqFt": "Size (SqFt)", "Price_in_Lakhs": "Price (₹ Lakhs)", "Property_Type": "Property Type"}
        )
        st.plotly_chart(fig_scatter_size, use_container_width=True)
        
    with row_col2:
        fig_scatter_age = px.scatter(
            filtered_df, x="Age_of_Property", y="Price_in_Lakhs", color="Furnished_Status",
            title="Price vs. Property Age (Years)",
            labels={"Age_of_Property": "Property Age (Years)", "Price_in_Lakhs": "Price (₹ Lakhs)", "Furnished_Status": "Furnished Status"}
        )
        st.plotly_chart(fig_scatter_age, use_container_width=True)

    # Data Table
    st.subheader("📋 Search Results Table")
    st.dataframe(
        filtered_df[['State', 'City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'Age_of_Property', 'Furnished_Status', 'Availability_Status']].head(100),
        use_container_width=True
    )
