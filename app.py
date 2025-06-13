import streamlit as st
import pandas as pd
import plotly.express as px
import gdown
import os

LINK_FILE = "datalink.txt"

def save_link(link):
    with open(LINK_FILE, "w") as f:
        f.write(link)

def read_link():
    if os.path.exists(LINK_FILE):
        with open(LINK_FILE, "r") as f:
            return f.read().strip()
    return ""

@st.cache_data(show_spinner=True)
def load_data_from_drive(gdrive_link):
    if not gdrive_link:
        return None
    if "id=" in gdrive_link:
        file_id = gdrive_link.split("id=")[-1].split("&")[0]
    elif "/d/" in gdrive_link:
        file_id = gdrive_link.split("/d/")[-1].split("/")[0]
    else:
        st.error("Invalid Google Drive link format. Please provide a proper shareable link.")
        return None
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "temp_data.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    os.remove(output)
    return df

st.set_page_config(layout="wide")
st.title("📊 Streamlined Sales Analysis Dashboard")

# Admin section
with st.expander("🔒 Admin: Update Dataset (Google Drive Link)", expanded=True):
    st.markdown("""
    **Instructions:**  
    1. Upload your (large) CSV to Google Drive.  
    2. Get the *shareable* link (Anyone with the link can view).  
    3. Paste the link below and click 'Set as Current Data'.  
    """)
    new_link = st.text_input("🔗 Paste Google Drive CSV shareable link here:", key="gdrive_link")
    set_link = st.button("Set as Current Data")
    if set_link and new_link:
        save_link(new_link)
        st.success("Google Drive link set as current data for all users. Please reload app to see updated data.")

# Get the current link (for everyone)
current_link = read_link()

if current_link:
    with st.spinner("Loading data from Google Drive..."):
        df_original = load_data_from_drive(current_link)
    if df_original is not None:
        st.success("Data loaded successfully!")
        # ----------- Dashboard code starts here ------------------

        # Data Cleaning
        df_original.replace(['#DIV/0!', 'N/A', 'na', 'NA', ''], pd.NA, inplace=True)
        if 'Invoice Date' in df_original.columns:
            df_original['Invoice Date'] = pd.to_datetime(df_original['Invoice Date'], errors='coerce')
            df_original['Month'] = df_original['Invoice Date'].dt.to_period('M').astype(str)
            df_original['Date'] = df_original['Invoice Date'].dt.date

        for col in ['discount%', 'coupon%', 'scheme%', 'margin%']:
            if col in df_original.columns:
                df_original[col] = pd.to_numeric(
                    df_original[col].astype(str).str.replace('%', '').str.replace(',', ''), errors='coerce'
                )

        def remove_outliers_iqr(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return series.where((series >= lower) & (series <= upper))

        for col in ['discount%', 'coupon%', 'scheme%', 'margin%']:
            if col in df_original.columns:
                df_original[col] = remove_outliers_iqr(df_original[col])

        num_cols = ['Sum of Invoice Amount With Tax', 'disount amt', 'magin', 'upfront disc']
        for col in num_cols:
            if col in df_original.columns:
                df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

        st.sidebar.header("🔍 General Filters")

        def all_selectbox(label, options, default_all=True, key=None):
            all_label = "All"
            if default_all:
                default = options
            else:
                default = []
            options_with_all = [all_label] + options
            selected = st.sidebar.multiselect(label, options_with_all, default=[all_label] if default_all else [], key=key)
            if all_label in selected or selected == []:
                return options
            return selected

        months = sorted(df_original['Month'].dropna().unique()) if 'Month' in df_original.columns else []
        selected_months = all_selectbox("Select Month(s)", months, default_all=True, key="months_sidebar") if months else []
        min_date, max_date = None, None
        if 'Date' in df_original.columns and not df_original['Date'].isnull().all():
            min_date, max_date = df_original['Date'].min(), df_original['Date'].max()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], key="date_range_sidebar") if min_date and max_date else [None, None]

        cities = sorted(df_original['City'].dropna().unique()) if 'City' in df_original.columns else []
        selected_cities = all_selectbox("Select Cities", cities, default_all=True, key="cities_sidebar") if cities else []

        stores = sorted(df_original['Store Full Name'].dropna().unique()) if 'Store Full Name' in df_original.columns else []
        selected_stores = all_selectbox("Select Stores", stores, default_all=True, key="stores_sidebar") if stores else []

        skus = sorted(df_original['Sku Name'].dropna().unique()) if 'Sku Name' in df_original.columns else []
        selected_skus = all_selectbox("Select SKUs", skus, default_all=True, key="sku_sidebar") if skus else []

        subcats = sorted(df_original['Sku Sub Category'].dropna().unique()) if 'Sku Sub Category' in df_original.columns else []
        selected_subcats = all_selectbox("Select Sub Categories", subcats, default_all=True, key="subcats_sidebar") if subcats else []

        remove_pl = st.sidebar.checkbox("Remove Pvt Label (PL) Products", key="pl_sidebar")

        # FILTER DATA
        df_filtered_global = df_original
        if months and selected_months:
            df_filtered_global = df_filtered_global[df_filtered_global['Month'].isin(selected_months)]
        if min_date and max_date and date_range[0] and date_range[1]:
            df_filtered_global = df_filtered_global[
                (df_filtered_global['Date'] >= date_range[0]) & (df_filtered_global['Date'] <= date_range[1])
            ]
        if cities and selected_cities:
            df_filtered_global = df_filtered_global[df_filtered_global['City'].isin(selected_cities)]
        if stores and selected_stores:
            df_filtered_global = df_filtered_global[df_filtered_global['Store Full Name'].isin(selected_stores)]
        if skus and selected_skus:
            df_filtered_global = df_filtered_global[df_filtered_global['Sku Name'].isin(selected_skus)]
        if subcats and selected_subcats:
            df_filtered_global = df_filtered_global[df_filtered_global['Sku Sub Category'].isin(selected_subcats)]
        if remove_pl and 'Pvt Label Tag' in df_filtered_global.columns:
            df_filtered_global = df_filtered_global[df_filtered_global['Pvt Label Tag'] != 'PL']

        st.sidebar.header("🧩 Multi-View Options")
        multi_view_enabled = st.sidebar.checkbox("Enable Multi-View (View Multiple Analyses)", key="multi_view")

        analysis_options = [
            "Monthly Revenue Trend",
            "Day-on-Day Trend",
            "Group-wise Summary",
            "Group-wise MoM & DoD",
            "PL vs Non-PL Analysis",
            "Top/Bottom 10% by Category",
            "Store-Level Profitability",
            "Category-Wise Contribution",
            "Discount Breakdown",
            "Time-Series by Store",
            "Customer Rank vs Spend"
        ]
        metric_options = [
            'Sum of Invoice Amount With Tax', 'disount amt', 'discount%',
            'coupon%', 'scheme%', 'magin', 'margin%', 'upfront disc'
        ]

        def analysis_block(analysis_type, df, idx=""):
            st.markdown(f"#### 🔎 Analysis: {analysis_type}")
            metric_cols = [m for m in metric_options if m in df.columns]
            if not metric_cols:
                st.warning("No suitable metric columns found in data.")
                return
            selected_metric = st.selectbox(f"Select Metric", metric_cols, key=f"metric_{idx}_{analysis_type}")

            fig = None
            table_data = None

            if analysis_type == "Day-on-Day Trend":
                group_cols = [c for c in ['Store Full Name', 'City', 'Sku Sub Category'] if c in df.columns]
                selected_group = st.selectbox("Group By (optional)", ["None"] + group_cols, key=f"group_dod_{idx}")
                # Local filter for city (optional)
                if 'City' in df.columns:
                    cities = sorted(df['City'].dropna().unique())
                    selected_cities = st.multiselect("Filter by City", cities, default=cities, key=f"cities_dod_{idx}")
                    df = df[df['City'].isin(selected_cities)]
                if selected_group != "None":
                    trend = df.groupby(['Date', selected_group])[selected_metric].mean().reset_index() \
                        if selected_metric.endswith('%') else df.groupby(['Date', selected_group])[selected_metric].sum().reset_index()
                    fig = px.line(trend, x='Date', y=selected_metric, color=selected_group)
                else:
                    trend = df.groupby('Date')[selected_metric].mean().reset_index() \
                        if selected_metric.endswith('%') else df.groupby('Date')[selected_metric].sum().reset_index()
                    fig = px.line(trend, x='Date', y=selected_metric, markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(trend, use_container_width=True)
                return

            if analysis_type == "Group-wise MoM & DoD":
                group_opts = [c for c in ['Store Full Name', 'Sku Sub Category', 'Sku Name'] if c in df.columns]
                if not group_opts:
                    st.warning("No suitable group columns found in data.")
                    return
                group_col = st.selectbox("Group By", group_opts, key=f"group_col_{idx}")
                group_values = sorted(df[group_col].dropna().unique())
                selected_values = all_selectbox(f"Select {group_col} values", group_values, default_all=True, key=f"group_val_{idx}")
                df = df[df[group_col].isin(selected_values)]
                df_sorted = df.sort_values(by='Invoice Date')

                st.markdown("##### Day-over-Day Actual Values")
                group_df = df_sorted.groupby(['Date', group_col])[selected_metric].mean().reset_index() if selected_metric.endswith('%') else df_sorted.groupby(['Date', group_col])[selected_metric].sum().reset_index()
                fig = px.line(group_df, x='Date', y=selected_metric, color=group_col)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(group_df, use_container_width=True)

                st.markdown("##### Month-over-Month Actual Values")
                mom_df = df_sorted.groupby(['Month', group_col])[selected_metric].mean().reset_index() if selected_metric.endswith('%') else df_sorted.groupby(['Month', group_col])[selected_metric].sum().reset_index()
                fig2 = px.bar(mom_df, x='Month', y=selected_metric, color=group_col, barmode='group')
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(mom_df, use_container_width=True)
                return

            if analysis_type == "Monthly Revenue Trend":
                trend = df.groupby('Month')[selected_metric].mean().reset_index() if selected_metric.endswith('%') else df.groupby('Month')[selected_metric].sum().reset_index()
                fig = px.line(trend, x='Month', y=selected_metric, markers=True)
                table_data = trend

            elif analysis_type == "Group-wise Summary":
                group_opts = [c for c in ['Sku Name', 'Store Full Name', 'City', 'Sku Sub Category'] if c in df.columns]
                if not group_opts:
                    st.warning("No suitable group columns found in data.")
                    return
                group_by = st.selectbox("Group By", group_opts, key=f"group_by_{idx}")
                search_val = st.text_input(f"Search {group_by} (optional)", key=f"search_{idx}")
                agg_funcs = {col: 'mean' if col.endswith('%') else 'sum' for col in metric_cols}
                grouped = df.groupby(group_by).agg(agg_funcs).reset_index()
                if search_val:
                    grouped = grouped[grouped[group_by].astype(str).str.contains(search_val, case=False)]
                grouped = grouped.sort_values(selected_metric, ascending=False)
                view_by_month = st.checkbox(f"View {group_by} by Month", key=f"monthwise_{idx}")
                if view_by_month:
                    grouped_month = df.groupby(['Month', group_by]).agg(agg_funcs).reset_index()
                    grouped_month = grouped_month.sort_values([group_by, 'Month'])
                    fig = px.bar(
                        grouped_month,
                        x='Month',
                        y=selected_metric,
                        color=group_by,
                        barmode='group',
                        title=f"{selected_metric} by {group_by} over Months"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(grouped_month, use_container_width=True)
                else:
                    fig = px.bar(grouped.head(20), x=group_by, y=selected_metric, color=selected_metric)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(grouped, use_container_width=True)
                return

            elif analysis_type == "PL vs Non-PL Analysis" and 'Pvt Label Tag' in df.columns:
                tag = df.dropna(subset=['Pvt Label Tag'])
                tag = tag.groupby(['Month', 'Pvt Label Tag'])[selected_metric].mean().reset_index() if selected_metric.endswith('%') else tag.groupby(['Month', 'Pvt Label Tag'])[selected_metric].sum().reset_index()
                fig = px.bar(tag, x='Month', y=selected_metric, color='Pvt Label Tag')
                table_data = tag

            elif analysis_type == "Store-Level Profitability" and 'Store Full Name' in df.columns:
                store = df.groupby('Store Full Name')[selected_metric].sum().reset_index()
                fig = px.bar(store, x='Store Full Name', y=selected_metric, color=selected_metric)
                table_data = store

            elif analysis_type == "Category-Wise Contribution" and 'Sku Sub Category' in df.columns:
                cat = df.groupby('Sku Sub Category')[selected_metric].mean().reset_index() if selected_metric.endswith('%') else df.groupby('Sku Sub Category')[selected_metric].sum().reset_index()
                fig = px.pie(cat, names='Sku Sub Category', values=selected_metric)
                table_data = cat

            elif analysis_type == "Discount Breakdown":
                disc_cols = [col for col in ['discount%', 'coupon%', 'scheme%', 'upfront disc', 'Sku Name'] if col in df.columns]
                if set(['Sku Name']).issubset(disc_cols):
                    disc = df[disc_cols]
                    disc_avg = disc.groupby('Sku Name').mean().reset_index()
                    disc_melted = pd.melt(disc_avg, id_vars='Sku Name', var_name='Type', value_name='Value')
                    fig = px.bar(disc_melted, x='Sku Name', y='Value', color='Type', barmode='stack')
                    table_data = disc_melted

            elif analysis_type == "Time-Series by Store" and 'Store Full Name' in df.columns:
                stores = sorted(df['Store Full Name'].dropna().unique())
                selected_stores = all_selectbox("Choose Stores", stores, default_all=True, key=f"stores_{idx}")
                filtered = df[df['Store Full Name'].isin(selected_stores)]
                trend = filtered.groupby(['Month', 'Store Full Name'])[selected_metric].mean().reset_index() if selected_metric.endswith('%') else filtered.groupby(['Month', 'Store Full Name'])[selected_metric].sum().reset_index()
                fig = px.line(trend, x='Month', y=selected_metric, color='Store Full Name')
                table_data = trend

            elif analysis_type == "Customer Rank vs Spend" and set(['Customer ID', 'Sum of Customer Rank', 'Sum of Invoice Amount With Tax']).issubset(df.columns):
                spend = df.groupby('Customer ID').agg({
                    'Sum of Customer Rank': 'mean',
                    'Sum of Invoice Amount With Tax': 'sum'
                }).reset_index()
                fig = px.scatter(spend, x='Sum of Customer Rank', y='Sum of Invoice Amount With Tax', hover_name='Customer ID')
                table_data = spend

            elif analysis_type == "Top/Bottom 10% by Category":
                st.markdown("Get the top or bottom 10% performers in a category based on a metric.")
                cat_opts = [c for c in ['Store Full Name', 'Month', 'Sku Sub Category', 'Sku Name', 'City', 'Generic Flag', 'Pvt Label Tag', 'Customer ID'] if c in df.columns]
                if not cat_opts: st.warning("No categories available."); return
                category = st.selectbox(
                    "Category to Rank By",
                    cat_opts,
                    key=f"topcat_{idx}"
                )
                metric = st.selectbox(
                    "Metric to Use for Ranking",
                    metric_cols,
                    key=f"topmetric_{idx}"
                )
                top_or_bottom = st.radio(
                    "Show:",
                    ["Top 10% (Best performers)", "Bottom 10% (Worst performers)"],
                    key=f"toporbottom_{idx}"
                )

                if category == "Customer ID" and top_or_bottom == "Bottom 10% (Worst performers)":
                    st.warning("Showing worst performing customers is disabled.")
                    return

                agg_func = 'mean' if metric.endswith('%') else 'sum'
                summary_df = df.groupby(category)[metric].agg(agg_func).reset_index()
                summary_df = summary_df.dropna(subset=[metric])
                n_cat = summary_df.shape[0]

                few_option_categories = ['City', 'Month']
                if (category in few_option_categories and n_cat <= 12) or n_cat <= 12:
                    top_n = 3
                    bottom_n = 3
                    if top_or_bottom == "Top 10% (Best performers)":
                        filtered = summary_df.nlargest(top_n, metric)
                        filtered = filtered.sort_values(metric, ascending=False)
                    else:
                        if category == "Customer ID":
                            st.warning('Worst performing customers are not shown.')
                            return
                        filtered = summary_df.nsmallest(bottom_n, metric)
                        filtered = filtered.sort_values(metric, ascending=True)
                    st.markdown(f"##### Top {top_n} and Bottom {bottom_n} for '{category}' by '{metric}'")
                else:
                    quantile_val = 0.9 if top_or_bottom == "Top 10% (Best performers)" else 0.1
                    threshold = summary_df[metric].quantile(quantile_val)
                    if top_or_bottom == "Top 10% (Best performers)":
                        filtered = summary_df[summary_df[metric] >= threshold]
                        filtered = filtered.sort_values(metric, ascending=False)
                        st.markdown(f"##### Top 10% for '{category}' by '{metric}'")
                    else:
                        if category == "Customer ID":
                            st.warning('Worst performing customers are not shown.')
                            return
                        filtered = summary_df[summary_df[metric] <= threshold]
                        filtered = filtered.sort_values(metric, ascending=True)
                        st.markdown(f"##### Bottom 10% for '{category}' by '{metric}'")

                fig = px.bar(filtered, x=category, y=metric, color=metric)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(filtered, use_container_width=True)
                return

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if table_data is not None:
                st.dataframe(table_data, use_container_width=True)

        if multi_view_enabled:
            st.info("You can select up to 4 analyses to view. If you select 4, they'll be shown in a 2x2 grid.")
            selected_analyses = st.multiselect("Select Analyses", analysis_options, default=[analysis_options[0]])
            n = len(selected_analyses)
            if n == 0:
                st.warning("Please select at least one analysis.")
            elif n == 1:
                with st.expander(f"🔎 {selected_analyses[0]}", expanded=True):
                    analysis_block(selected_analyses[0], df_filtered_global, idx="0")
            elif n == 2:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander(f"🔎 {selected_analyses[0]}", expanded=True):
                        analysis_block(selected_analyses[0], df_filtered_global, idx="0")
                with col2:
                    with st.expander(f"🔎 {selected_analyses[1]}", expanded=True):
                        analysis_block(selected_analyses[1], df_filtered_global, idx="1")
            elif n == 3:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander(f"🔎 {selected_analyses[0]}", expanded=True):
                        analysis_block(selected_analyses[0], df_filtered_global, idx="0")
                    with st.expander(f"🔎 {selected_analyses[2]}", expanded=True):
                        analysis_block(selected_analyses[2], df_filtered_global, idx="2")
                with col2:
                    with st.expander(f"🔎 {selected_analyses[1]}", expanded=True):
                        analysis_block(selected_analyses[1], df_filtered_global, idx="1")
            elif n == 4:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander(f"🔎 {selected_analyses[0]}", expanded=True):
                        analysis_block(selected_analyses[0], df_filtered_global, idx="0")
                    with st.expander(f"🔎 {selected_analyses[2]}", expanded=True):
                        analysis_block(selected_analyses[2], df_filtered_global, idx="2")
                with col2:
                    with st.expander(f"🔎 {selected_analyses[1]}", expanded=True):
                        analysis_block(selected_analyses[1], df_filtered_global, idx="1")
                    with st.expander(f"🔎 {selected_analyses[3]}", expanded=True):
                        analysis_block(selected_analyses[3], df_filtered_global, idx="3")
            else:
                st.warning("Currently, only up to 4 analyses can be viewed at once. Please select 4 or fewer.")
        else:
            analysis_type = st.sidebar.selectbox("📈 Select Analysis", analysis_options)
            with st.container():
                analysis_block(analysis_type, df_filtered_global)
        # ----------- Dashboard code ends here ------------------
    else:
        st.error("Failed to load data. Check your link and try again.")
else:
    st.info("No data loaded yet. Please use the 'Admin: Update Dataset' section above to load your CSV from Google Drive.")