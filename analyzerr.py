import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# ------------------------
# Helper Functions
# ------------------------

def extract_keywords(texts, n=10):
    """Extract top keywords using TF-IDF"""
    if not any(isinstance(t, str) and t.strip() for t in texts):
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(texts)
    indices = X.sum(axis=0).argsort()[0, -n:].tolist()[0]
    return [vectorizer.get_feature_names_out()[i] for i in indices[::-1]]

def cluster_feedback(texts, num_clusters=3):
    """Cluster feedback into themes"""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < num_clusters:
        return {"General": texts}

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    clusters = {f"Theme {i+1}": [] for i in range(num_clusters)}
    for i, label in enumerate(kmeans.labels_):
        clusters[f"Theme {label+1}"].append(texts[i])
    return clusters

def generate_wordcloud(texts, filename):
    """Generate and save wordcloud"""
    combined_text = " ".join([str(t) for t in texts if isinstance(t, str)])
    if not combined_text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    wc.to_file(filename)
    return filename

def extract_quotes(texts, top_n=5):
    """Get most common exact feedback quotes"""
    counter = Counter([t.strip() for t in texts if isinstance(t, str) and t.strip()])
    return counter.most_common(top_n)

def generate_pdf_report(per_column_analysis, overall_summary, wordcloud_paths, output_path="feedback_report.pdf"):
    """Generate a structured PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Student Feedback Analysis Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(5)

    # Per-Column Analysis
    for col, analysis in per_column_analysis.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, f"Question: {col}", ln=True)
        pdf.set_font("Arial", size=11)

        # Keywords
        pdf.multi_cell(0, 8, f" Keywords: {', '.join(analysis['keywords'])}")

        # Clusters
        for theme, feedbacks in analysis['clusters'].items():
            pdf.multi_cell(0, 8, f" {theme}:")
            for fb in feedbacks[:3]:  # Show sample quotes
                pdf.multi_cell(0, 8, f"   - {fb}")

        # Quotes
        pdf.multi_cell(0, 8, " Student Quotes:")
        for quote, count in analysis['quotes']:
            pdf.multi_cell(0, 8, f'   - "{quote}" ({count} mentions)')

        # Wordcloud
        if analysis['wordcloud'] and os.path.exists(analysis['wordcloud']):
            pdf.ln(3)
            pdf.image(analysis['wordcloud'], w=150)
        pdf.ln(5)

    # Overall Summary
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Overall Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, overall_summary)

    pdf.output(output_path)

# ------------------------
# Streamlit App
# ------------------------

st.title(" Student Feedback Analyzer")

uploaded_file = st.file_uploader("Upload your feedback CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Data Loaded:", df.head())

    per_column_analysis = {}
    overall_texts = []

    for col in df.columns:
        texts = df[col].dropna().astype(str).tolist()
        if not any(texts):
            continue

        keywords = extract_keywords(texts)
        clusters = cluster_feedback(texts)
        quotes = extract_quotes(texts)
        wc_path = generate_wordcloud(texts, f"wordcloud_{col}.png")
        overall_texts.extend(texts)

        per_column_analysis[col] = {
            "keywords": keywords,
            "clusters": clusters,
            "quotes": quotes,
            "wordcloud": wc_path
        }

        # Show in Streamlit
        st.subheader(f" Analysis for {col}")
        st.write(" Keywords:", keywords)
        st.write(" Clusters:")
        for theme, feedbacks in clusters.items():
            st.write(f"**{theme}:**")
            st.write(feedbacks[:3])
        st.write(" Top Quotes:", quotes)
        if wc_path:
            st.image(wc_path)

    # Overall Summary
    all_keywords = []
    for col, analysis in per_column_analysis.items():
        all_keywords.extend(analysis["keywords"])
    overall_summary = "Overall, students highlighted issues around: " + ", ".join(set(all_keywords))

    st.subheader(" Overall Summary")
    st.write(overall_summary)

    # Export PDF
    if st.button(" Export PDF Report"):
        generate_pdf_report(per_column_analysis, overall_summary, 
                            [a["wordcloud"] for a in per_column_analysis.values()], 
                            "feedback_report.pdf")
        st.success(" PDF Generated! Check feedback_report.pdf in your folder.")



