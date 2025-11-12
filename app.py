import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Automated EDA Tool", layout="wide")

st.title("📊 Automated EDA (Exploratory Data Analysis) Tool")
st.markdown("Upload your dataset and generate instant statistical and visual insights!")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📋 Basic Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("📈 Statistical Summary")
    st.write(df.describe(include='all'))

    st.subheader("🕳️ Missing Values Summary")
    st.write(df.isnull().sum())

    st.subheader("🔗 Correlation Heatmap (Numerical Columns)")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for correlation analysis.")

    st.subheader("📉 Distribution Plots (First 3 Numeric Columns)")
    for col in numeric_df.columns[:3]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    st.subheader("🧮 Pair Plot (First 4 Numeric Columns)")
    if numeric_df.shape[1] >= 2:
        fig = sns.pairplot(numeric_df.iloc[:, :4])
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for pair plot.")

    # Download cleaned report
    st.subheader("⬇️ Download EDA Summary")
    eda_summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    
    csv = eda_summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download EDA Summary as CSV",
        data=csv,
        file_name='eda_summary.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Upload a CSV file to begin your EDA journey.")
