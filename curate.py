import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import time
import re
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Curate", layout="wide")

st.markdown("""
<style>
/* Formal dark theme */
body, .stApp { background-color: #0E1117; color: #e6eef6; }
h1, h2, h3, h4 { color: #e6eef6; }
.stButton>button { background-color: #111827; color: #e6eef6; border: 1px solid #1f2937; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
plt.style.use('dark_background')

def show_preview_with_highlights(current_df, previous_df):
    if current_df.shape != previous_df.shape or not current_df.index.equals(previous_df.index):
        st.dataframe(current_df)
        st.info("Dataset Shape or Index Changed. Displaying Updated Dataset Without Cell-Level Highlights.")
        return
    
    cols = current_df.columns.intersection(previous_df.columns)
    prev = previous_df[cols].copy()
    curr = current_df[cols].copy()

    prev_na = prev.isna()
    curr_na = curr.isna()
    both_na = prev_na & curr_na
    neq = curr.ne(prev)
    mask = neq & (~both_na)

    st.dataframe(current_df.style.apply(lambda x: pd.DataFrame(np.where(mask, 'background-color: yellow; color: black;', ''), index=x.index, columns=x.columns), axis=None))

def find_column_by_name(raw_name, candidates, cutoff=0.55):
    raw = raw_name.lower().strip()
    cand_low = [c.lower() for c in candidates]
    matches = difflib.get_close_matches(raw, cand_low, n=1, cutoff=cutoff)
    if matches:
        return candidates[cand_low.index(matches[0])]
    for c in candidates:
        if raw in c.lower():
            return c
    return None

@st.cache_data
def cached_corr(df, numeric_cols):
    return df[numeric_cols].corr()

st.markdown("<h1 style='text-align: center;'>Curate</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #3082a5;'>Intelligent Data Analysis Toolkit</h4>", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'ml_best_model' not in st.session_state:
    st.session_state.ml_best_model = None
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'scroll_trigger' not in st.session_state:
    st.session_state.scroll_trigger = False

st.markdown("---")
st.header("How It Works")
st.markdown("This Tool Provides a Guided, Four-Step Workflow to Turn Your Raw Data Into Insights.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("1. Upload")
    st.info("Upload Your CSV or ZIP File to Begin.")
with col2:
    st.subheader("2. Clean")
    st.info("Automatically/Manually Clean Your Data.")
with col3:
    st.subheader("3. Explore")
    st.info("Visualize Your Data With Interactive Charts.")
with col4:
    st.subheader("4. Download")
    st.info("Download Your Cleaned Dataset.")

st.markdown("---")

if st.session_state.step == 1:
    if st.session_state.scroll_trigger:
        st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)
        st.session_state.scroll_trigger = False

    st.header("Step 1 : Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or ZIP", type=["csv","zip"])

    if uploaded_file is not None:
        try:
            with st.spinner("Reading File."):
                temp_df = None
                if uploaded_file.name.endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file, 'r') as z:
                        csvs = [f for f in z.namelist() if f.endswith('.csv')]
                        if not csvs:
                            st.error("No CSV Files in ZIP.")
                        else:
                            nm = csvs[0]
                            with z.open(nm) as fh:
                                temp_df = pd.read_csv(io.BytesIO(fh.read()))
                            st.success(f"Loaded '{nm}' From ZIP.")
                else:
                    temp_df = pd.read_csv(uploaded_file)
                    st.success("CSV Loaded.")
                
                if temp_df is not None:
                    st.session_state.df = temp_df
                    st.session_state.df_cleaned = temp_df.copy()
                    if 'auto_conv' in st.session_state:
                        del st.session_state['auto_conv']

        except Exception as e:
            st.error(f"Could Not Read File: {e}")
            st.session_state.df = None
            st.session_state.df_cleaned = None

    if st.session_state.df is not None:
        st.markdown("**Dataset Summary**")
        st.write(f"Rows: **{st.session_state.df.shape[0]}**, Columns: **{st.session_state.df.shape[1]}**")
        if st.checkbox("Show Raw Preview"):
            st.dataframe(st.session_state.df)
        
        df_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Dtype': st.session_state.df.dtypes.astype(str),
            'Non-null': st.session_state.df.count().values,
            'Missing': st.session_state.df.isnull().sum().values
        })
        st.dataframe(df_info)
        
        if st.button("Proceed to Clean"):
            st.session_state.step = 2
            st.session_state.scroll_trigger = True
            st.rerun()

if st.session_state.step == 2:
    if st.session_state.scroll_trigger:
        st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)
        st.session_state.scroll_trigger = False

    if st.session_state.df_cleaned is None:
        st.warning("Upload Dataset First.")
        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.session_state.scroll_trigger = True
            st.rerun()
    else:
        st.header("Step 2 : Clean and Preprocess")
        st.markdown("All Mutating Operations Show Processing Indicators. Preview Highlights Actual Changes.")

        if 'auto_conv' not in st.session_state:
            with st.spinner("Auto-Converting Columns."):
                converted = []
                df_temp = st.session_state.df_cleaned.copy()
                for col in list(df_temp.columns):
                    if df_temp[col].dtype == 'object':
                        cleaned = df_temp[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True).str.strip()
                        try:
                            conv = pd.to_numeric(cleaned, errors='coerce')
                            if conv.notna().sum() / len(conv) > 0.5:
                                df_temp[col] = conv
                                converted.append(col)
                        except Exception:
                            continue
                
                if converted:
                    st.session_state.df_cleaned = df_temp
                    st.info(f"Auto-Converted To Numeric : {', '.join(converted)}")
                st.session_state.auto_conv = True

        st.subheader("Preview")
        st.dataframe(st.session_state.df_cleaned)

        st.markdown("#### Drop Duplicates")
        if st.button("Drop Duplicate Rows"):
            before = st.session_state.df_cleaned.copy()
            with st.spinner("Dropping Duplicates."):
                st.session_state.df_cleaned.drop_duplicates(inplace=True)
                time.sleep(0.2)
            dropped = before.shape[0] - st.session_state.df_cleaned.shape[0]
            st.info(f"Dropped {dropped} Duplicate Rows. Wait Until Preview Is Rendered.")
            show_preview_with_highlights(st.session_state.df_cleaned, before)

        st.markdown("---")
        st.subheader("Missing Values — Options")
        numeric_cols = st.session_state.df_cleaned.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = st.session_state.df_cleaned.select_dtypes(exclude=np.number).columns.tolist()

        if st.button("Drop Rows With Any Missing"):
            before = st.session_state.df_cleaned.copy()
            with st.spinner("Dropping Rows."):
                st.session_state.df_cleaned.dropna(inplace=True)
                time.sleep(0.2)
            st.success("Rows Dropped.")
            show_preview_with_highlights(st.session_state.df_cleaned, before)

        if numeric_cols:
            st.markdown("**Numeric Imputation**")
            num_scope = st.radio("Apply To:", ("All Numeric Columns", "Specific Numeric Column"), key='num_scope')
            num_method = st.selectbox("Method:", ('Mean','Median','Mode'), key='num_method')
            
            if num_scope.startswith("Specific"):
                sel = st.selectbox("Select Numeric Column:", numeric_cols, key='num_sel')
                if st.button("Apply Numeric Imputation to Column"):
                    before = st.session_state.df_cleaned.copy()
                    with st.spinner("Applying Imputation."):
                        s = st.session_state.df_cleaned[sel]
                        if num_method == 'Mean':
                            s.fillna(s.mean(), inplace=True)
                        elif num_method == 'Median':
                            s.fillna(s.median(), inplace=True)
                        elif num_method == 'Mode':
                            m = s.mode().dropna()
                            s.fillna(m.iloc[0] if not m.empty else 0, inplace=True)
                        st.session_state.df_cleaned[sel] = s
                        time.sleep(0.2)
                    st.success("Imputation Applied. Wait Until Preview Is Rendered")
                    show_preview_with_highlights(st.session_state.df_cleaned, before)
            else:
                if st.button("Apply Numeric Imputation to All"):
                    before = st.session_state.df_cleaned.copy()
                    with st.spinner("Applying Numeric Imputation to All."):
                        for col in numeric_cols:
                            s = st.session_state.df_cleaned[col]
                            if s.isna().sum()==0: continue
                            if num_method == 'Mean':
                                s.fillna(s.mean(), inplace=True)
                            elif num_method == 'Median':
                                s.fillna(s.median(), inplace=True)
                            elif num_method == 'Mode':
                                m = s.mode().dropna()
                                s.fillna(m.iloc[0] if not m.empty else 0, inplace=True)
                            st.session_state.df_cleaned[col] = s
                        time.sleep(0.2)
                    st.success("Numeric Imputation Applied. Wait Until Preview Is Rendered.")
                    show_preview_with_highlights(st.session_state.df_cleaned, before)
        else:
            st.info("No Numeric Columns Detected for Imputation.")

        if categorical_cols:
            st.markdown("**Categorical Imputation**")
            cat_scope = st.radio("Apply To:", ("All Categorical Columns", "Specific Categorical Column"), key='cat_scope')
            cat_method = st.selectbox("Method:", ('Mode','Constant'), key='cat_method')
            if cat_scope.startswith("Specific"):
                sc = st.selectbox("Select Categorical Column:", categorical_cols, key='cat_sel')
                const_cat = st.text_input("Constant For Categorical (If Chosen):", "Missing", key='cat_const')
                if st.button("Apply Categorical Imputation to Column"):
                    before = st.session_state.df_cleaned.copy()
                    with st.spinner("Applying Categorical Imputation."):
                        s = st.session_state.df_cleaned[sc]
                        if cat_method == 'Mode':
                            m = s.mode().dropna()
                            s.fillna(m.iloc[0] if not m.empty else 'Missing', inplace=True)
                        elif cat_method == 'Constant':
                            s.fillna(const_cat if const_cat else 'Missing', inplace=True)
                        st.session_state.df_cleaned[sc] = s
                        time.sleep(0.2)
                    st.success("Categorical Imputation Applied. Wait Until Preview Is Rendered")
                    show_preview_with_highlights(st.session_state.df_cleaned, before)
            else:
                const_cat_all = st.text_input("Constant For All Categorical (If Chosen):", "Missing", key='cat_const_all')
                if st.button("Apply Categorical Imputation to All"):
                    before = st.session_state.df_cleaned.copy()
                    with st.spinner("Applying Categorical Imputation to All."):
                        for col in categorical_cols:
                            s = st.session_state.df_cleaned[col]
                            if s.isna().sum()==0: continue
                            if cat_method == 'Mode':
                                m = s.mode().dropna()
                                s.fillna(m.iloc[0] if not m.empty else 'Missing', inplace=True)
                            elif cat_method == 'Constant':
                                s.fillna(const_cat_all if const_cat_all else 'Missing', inplace=True)
                            st.session_state.df_cleaned[col] = s
                        time.sleep(0.2)
                    st.success("Categorical Imputation Applied. Wait Until Preview Is Rendered")
                    show_preview_with_highlights(st.session_state.df_cleaned, before)
        else:
            st.info("No Categorical Columns Detected for Imputation.")

        st.markdown("---")
        st.subheader("Type Conversion")
        conv_col = st.selectbox("Column to Convert:", st.session_state.df_cleaned.columns.tolist(), key='conv_col')
        conv_to = st.selectbox("Convert To:", ("object","int64","float64","datetime64[ns]"), key='conv_to')
        if st.button("Apply Type Conversion"):
            before = st.session_state.df_cleaned.copy()
            with st.spinner("Converting Type."):
                try:
                    if conv_to.startswith("datetime"):
                        st.session_state.df_cleaned[conv_col] = pd.to_datetime(st.session_state.df_cleaned[conv_col], errors='coerce')
                    else:
                        st.session_state.df_cleaned[conv_col] = st.session_state.df_cleaned[conv_col].astype(conv_to)
                    time.sleep(0.2)
                    st.success("Type Conversion Successful. Wait Until Preview Is Rendered")
                except Exception as e:
                    st.error(f"Conversion Failed: {e}")
            show_preview_with_highlights(st.session_state.df_cleaned, before)

        st.markdown("---")
        if st.button("Proceed to Explore & Visualize"):
            st.session_state.step = 3
            st.session_state.scroll_trigger = True
            st.rerun()

        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.session_state.scroll_trigger = True
            st.rerun()

if st.session_state.step == 3:
    if st.session_state.scroll_trigger:
        st.markdown('<script>window.scrollTo(0, document.querySelector("h2").offsetTop);</script>', unsafe_allow_html=True)
        st.session_state.scroll_trigger = False

    if st.session_state.df_cleaned is None:
        st.warning("Please Upload & Clean Data First.")
        if st.button("Back to Clean"):
            st.session_state.step = 2
            st.session_state.scroll_trigger = True
            st.rerun()
    else:
        st.header("Step 3 : Explore & Visualizations")
        df = st.session_state.df_cleaned.copy()
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

        st.markdown("---")
        st.subheader("Recommended Visualizations")

        potential_cat_cols = [c for c in categorical_cols if df[c].nunique() < 50]
        potential_value_cols = [c for c in numeric_cols if any(keyword in c.lower() for keyword in ['total', 'sales', 'revenue', 'price', 'amount'])]
        if not potential_value_cols:
            potential_value_cols = numeric_cols

        cat_col_rec = find_column_by_name("category", potential_cat_cols) or (potential_cat_cols[0] if potential_cat_cols else None)
        value_col_rec = potential_value_cols[0] if potential_value_cols else None
        time_col_rec = datetime_cols[0] if datetime_cols else None

        rec_cols = st.columns(2)
        
        if cat_col_rec:
            with rec_cols[0]:
                try:
                    counts = df[cat_col_rec].value_counts().reset_index().head(20)
                    counts.columns = [cat_col_rec, 'count']
                    fig = px.bar(counts, x=cat_col_rec, y='count', title=f"Top 20 Distribution of {cat_col_rec}", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot distribution for {cat_col_rec}: {e}")

        if time_col_rec and value_col_rec:
            with rec_cols[1]:
                try:
                    df_time = df.set_index(time_col_rec).resample('D')[value_col_rec].sum().reset_index()
                    fig = px.line(df_time, x=time_col_rec, y=value_col_rec, title=f"Total {value_col_rec} Over Time", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot time series: {e}")

        if cat_col_rec and value_col_rec:
            rec_cols2 = st.columns(2)
            with rec_cols2[0]:
                try:
                    agg_sum = df.groupby(cat_col_rec)[value_col_rec].sum().reset_index().sort_values(value_col_rec, ascending=False).head(20)
                    fig_bar_sum = px.bar(agg_sum, x=cat_col_rec, y=value_col_rec, title=f"Total {value_col_rec} by {cat_col_rec}", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_bar_sum, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot aggregation for {cat_col_rec} vs {value_col_rec}: {e}")
            with rec_cols2[1]:
                try:
                    fig_box = px.box(df, x=cat_col_rec, y=value_col_rec, title=f"{value_col_rec} Distribution by {cat_col_rec}", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create box plot for {cat_col_rec} vs {value_col_rec}: {e}")

        st.markdown("---")
        st.subheader("Custom Plot (Scroll Up To View Recommeded Plots)")

        plot_opts = ["Bar", "Histogram", "Scatter", "Line", "Box", "Violin"]
        plot_type = st.selectbox("Plot Type:", plot_opts, key='custom_plot_type')
        
        c1, c2, c3 = st.columns(3)
        with c1:
            x_axis = st.selectbox("X Axis:", [None] + cols, key='custom_x')
        with c2:
            y_axis = st.selectbox("Y Axis:", [None] + cols, key='custom_y')
        with c3:
            color_by = st.selectbox("Color By:", [None] + cols, key='custom_color')

        if st.button("Render Custom Plot"):
            if not x_axis:
                st.warning("Please select an X-axis.")
            else:
                with st.spinner("Rendering Custom Plot."):
                    try:
                        if plot_type == "Bar":
                            fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, title=f'Bar Plot of {x_axis}', template=PLOTLY_TEMPLATE)
                        elif plot_type == "Histogram":
                            fig = px.histogram(df, x=x_axis, color=color_by, title=f'Histogram of {x_axis}', template=PLOTLY_TEMPLATE)
                        elif plot_type == "Scatter":
                            if not y_axis: st.warning("Scatter plot requires a Y-axis."); fig=None
                            else: fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=f'Scatter Plot: {x_axis} vs {y_axis}', template=PLOTLY_TEMPLATE)
                        elif plot_type == "Line":
                            if not y_axis: st.warning("Line plot requires a Y-axis."); fig=None
                            else: fig = px.line(df, x=x_axis, y=y_axis, color=color_by, title=f'Line Plot: {x_axis} vs {y_axis}', template=PLOTLY_TEMPLATE)
                        elif plot_type == "Box":
                            fig = px.box(df, x=x_axis, y=y_axis, color=color_by, title=f'Box Plot of {y_axis or x_axis}', template=PLOTLY_TEMPLATE)
                        elif plot_type == "Violin":
                            fig = px.violin(df, x=x_axis, y=y_axis, color=color_by, title=f'Violin Plot of {y_axis or x_axis}', template=PLOTLY_TEMPLATE)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Plotting Error: {e}")
        st.markdown("---")

        if st.button("Proceed to Auto - ML & Predictions"):
            st.session_state.step = 4
            st.session_state.scroll_trigger = True
            st.rerun()

        if st.button("Back to Clean"):
            st.session_state.step = 2
            st.session_state.scroll_trigger = True
            st.rerun()

if st.session_state.step == 4:
    if st.session_state.scroll_trigger:
        st.markdown('<script>window.scrollTo(0, 0);</script>', unsafe_allow_html=True)
        st.session_state.scroll_trigger = False

    if st.session_state.df_cleaned is None:
        st.warning("Please Upload & Clean Data First.")
        if st.button("Back to Explore"):
            st.session_state.step = 3
            st.session_state.scroll_trigger = True
            st.rerun()
    else:
        st.header("Step 4 : Auto - ML (Quick Models) & Download")
        df = st.session_state.df_cleaned.copy()
        cols = df.columns.tolist()

        st.subheader("Auto - ML (Training Pipeline)")
        target = st.selectbox("Select Target Column:", [None] + cols, key='auto_target')
        if target:
            features_all = [c for c in cols if c != target]
            features = st.multiselect("Select Features (Defaults to all):", features_all, default=features_all)
            test_size = st.slider("Test Set Fraction:", 0.1, 0.5, 0.2, 0.05)
            if st.button("Train Quick Models"):
                if not features:
                    st.error("Please select at least one feature.")
                else:
                    X = df[features].copy()
                    y = df[target].copy()
                    
                    with st.spinner("Preparing and Training."):
                        data = pd.concat([X, y], axis=1).dropna()
                        if data.shape[0] < 20:
                            st.error(f"Not Enough Data ({data.shape[0]} rows) After Dropping NA to Train. Try imputing missing values.")
                        else:
                            X = data[features]
                            y = data[target]

                            is_class = False
                            if not pd.api.types.is_numeric_dtype(y):
                                is_class = True
                            else:
                                if pd.api.types.is_integer_dtype(y) and y.nunique() < 30:
                                    is_class = True
                            
                            X_enc = pd.get_dummies(X, drop_first=True, dummy_na=True)
                            st.session_state.ml_cols = X_enc.columns.tolist()
                            
                            X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=42)
                            
                            num_feats = X_enc.select_dtypes(include=np.number).columns.tolist()
                            st.session_state.ml_num_feats = num_feats
                            scaler = StandardScaler()
                            if num_feats:
                                X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
                                X_test[num_feats] = scaler.transform(X_test[num_feats])
                                st.session_state.ml_scaler = scaler

                            trained = False
                            if is_class:
                                st.info("Detected Classification Task.")
                                try:
                                    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                                    lr = LogisticRegression(max_iter=500, solver='liblinear')
                                    models = {'RandomForest': rf, 'LogisticRegression': lr}
                                    results = {}
                                    for name, model in models.items():
                                        model.fit(X_train, y_train)
                                        acc = accuracy_score(y_test, model.predict(X_test))
                                        results[name] = (acc, model)
                                    
                                    best_name = max(results, key=lambda k: results[k][0])
                                    best_acc, best_model = results[best_name]
                                    st.success(f"Best Model: {best_name} (Accuracy: {best_acc:.3f})")
                                    trained = True
                                    st.session_state.ml_best_model = best_model
                                    st.session_state.ml_features = features
                                    st.session_state.ml_is_class = is_class
                                    st.session_state.ml_target = target
                                except ValueError as e:
                                    st.error(f"Training failed: {e}")
                                    st.warning("Could not fit a classification model. Please check if your target column is truly categorical.")
                            else:
                                st.info("Detected Regression Task.")
                                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                lr = LinearRegression()
                                dt = DecisionTreeRegressor(random_state=42)
                                models = {'RandomForest': rf, 'LinearRegression': lr, 'DecisionTree': dt}
                                results = {}
                                for name, model in models.items():
                                    model.fit(X_train, y_train)
                                    r2 = r2_score(y_test, model.predict(X_test))
                                    results[name] = (r2, model)
                                
                                best_name = max(results, key=lambda k: results[k][0])
                                best_r2, best_model = results[best_name]
                                st.success(f"Best Model: {best_name} (R²: {best_r2:.3f})")
                                trained = True
                                st.session_state.ml_best_model = best_model
                                st.session_state.ml_features = features
                                st.session_state.ml_is_class = is_class
                                st.session_state.ml_target = target
                            
                            if trained:
                                st.success("Model Trained and Ready for Predictions.")

        if st.session_state.get('ml_best_model'):
            st.subheader("Make Predictions")
            model = st.session_state.ml_best_model
            features = st.session_state.ml_features
            enc_cols = st.session_state.ml_cols
            num_feats = st.session_state.ml_num_feats
            scaler = st.session_state.get('ml_scaler')
            source_df = st.session_state.df_cleaned

            with st.expander("Make predictions on new data", expanded=True):
                pred_df_manual = pd.DataFrame(columns=features)

                column_config = {}
                for feature in features:
                    if not pd.api.types.is_numeric_dtype(source_df[feature]):
                        unique_options = source_df[feature].unique().tolist()
                        column_config[feature] = st.column_config.SelectboxColumn(
                            f"Select {feature}",
                            help=f"Choose a valid category for {feature}.",
                            options=unique_options,
                            required=True
                        )
                
                edited_df = st.data_editor(
                    pred_df_manual,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config=column_config
                )

                if st.button("Predict on Entered Data"):
                    if edited_df.isnull().values.any():
                        st.warning("Please fill in all required fields before predicting.")
                    elif edited_df.empty:
                        st.warning("No Data Entered.")
                    else:
                        with st.spinner("Predicting."):
                            Xp = edited_df.copy()
                            for col in Xp.columns:
                                Xp[col] = Xp[col].astype(source_df[col].dtype)

                            Xp_enc = pd.get_dummies(Xp, drop_first=True, dummy_na=True)
                            
                            missing_cols = set(enc_cols) - set(Xp_enc.columns)
                            for c in missing_cols:
                                Xp_enc[c] = 0
                            Xp_enc = Xp_enc[enc_cols]
                            
                            if num_feats and scaler:
                                num_feats_in_pred = [f for f in num_feats if f in Xp_enc.columns]
                                if num_feats_in_pred:
                                    Xp_enc[num_feats_in_pred] = scaler.transform(Xp_enc[num_feats_in_pred])
                            
                            preds = model.predict(Xp_enc)
                            edited_df[st.session_state.ml_target] = preds
                            st.dataframe(edited_df)
                            
                            csv_pred = edited_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Predictions (CSV)", data=csv_pred, file_name="manual_predictions.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Download Cleaned Data")
        csv = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")
        
        if st.button("Restart Workflow (Clear Session)"):
            keys_to_clear = list(st.session_state.keys())
            for k in keys_to_clear:
                if k != 'page_load':
                    del st.session_state[k]
            st.rerun()

        if st.button("Back to Explore"):
            st.session_state.step = 3
            st.session_state.scroll_trigger = True
            st.rerun()
