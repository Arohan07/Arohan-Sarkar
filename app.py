#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import re
import warnings
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from gnews import GNews
from ddgs import DDGS
from transformers import pipeline

warnings.filterwarnings("ignore")

# =========================================================
# 1. LOAD DATASET & TRAIN ML PIPELINE
# =========================================================
print("🌿 Booting Neural Pipeline & Loading Datasets...")

fake_path = "C:/Users/Arohan07/Downloads/data/Fake.csv" if os.path.exists("C:/Users/Arohan07/Downloads/data/Fake.csv") else "Fake.csv"
true_path = "C:/Users/Arohan07/Downloads/data/True.csv" if os.path.exists("C:/Users/Arohan07/Downloads/data/True.csv") else "True.csv"

try:
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 1  # 1 = Fake
    true_df['label'] = 0  # 0 = Real

    def clean_news_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"^.*?\(\s*Reuters\s*\)\s*-\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^.*?\(\s*AP\s*\)\s*-\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    fake_df['text'] = fake_df['text'].apply(clean_news_text)
    true_df['text'] = true_df['text'].apply(clean_news_text)

    fake_df = fake_df[fake_df['text'].str.len() > 100]
    true_df = true_df[true_df['text'].str.len() > 100]

    min_count = min(len(fake_df), len(true_df), 2000)
    fake_sample = fake_df.sample(n=min_count, random_state=42)
    true_sample = true_df.sample(n=min_count, random_state=42)

    df = pd.concat([fake_sample, true_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['text'] = df['text'].fillna('')

    X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_vec, y_train)

    def analyze_emotion_lightweight(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return {"label": "POSITIVE / OPTIMISTIC", "score": polarity}
    elif polarity < -0.3:
        return {"label": "NEGATIVE / SENSATIONAL", "score": abs(polarity)}
    else:
        return {"label": "NEUTRAL / INFORMATIONAL", "score": 1.0 - abs(polarity)}
    print("✅ Model Training & Pipeline Complete!")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")

google_news = GNews(language='en', country='US', max_results=5)


# =========================================================
# 2. HELPER FUNCTIONS FOR LIVE SEARCH & ANALYSIS
# =========================================================
def extract_search_keywords(text):
    if not text:
        return ""
    first_sentence = text.strip().split('\n')[0].split('.')[0]
    clean_text = re.sub(r'[^\w\s]', '', first_sentence)
    words = [w for w in clean_text.split() if len(w) > 2]
    return " ".join(words[:8])


def search_multi_engine(raw_text):
    combined_results = []
    seen_titles = set()
    query = extract_search_keywords(raw_text)

    if not query:
        return combined_results

    # Google News Search
    try:
        gnews_hits = google_news.get_news(query)
        for item in gnews_hits:
            title = item.get('title', '')
            publisher = item.get('publisher', {}).get('title', 'Google News')
            if title and title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                combined_results.append({'engine': 'Google News', 'title': title, 'source': publisher})
    except Exception:
        pass

    # DuckDuckGo Search
    try:
        with DDGS() as ddgs:
            ddg_hits = list(ddgs.news(keywords=query, max_results=5))
            if not ddg_hits:
                ddg_hits = list(ddgs.text(keywords=query, max_results=5))

            for item in ddg_hits:
                title = item.get('title', '')
                source = item.get('source', item.get('href', 'DuckDuckGo Web'))
                if title and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    combined_results.append({'engine': 'DuckDuckGo', 'title': title, 'source': source})
    except Exception:
        pass

    return combined_results


# =========================================================
# 3. CORE PREDICTION FUNCTION FOR GRADIO
# =========================================================
def verify_article_ui(user_input):
    if not user_input.strip():
        return "Please enter news text to analyze.", "", "", "", ""

    # ML Classifier Prediction
    vec_text = vectorizer.transform([user_input])
    proba = model.predict_proba(vec_text)[0]

    if proba[1] > 0.55:
        pred = 1
        local_confidence = proba[1] * 100
        stance_str = "Fake Pattern Detected"
    else:
        pred = 0
        local_confidence = proba[0] * 100
        stance_str = "Real Pattern Detected"

    # Multi-Engine Web Search
    live_results = search_multi_engine(user_input)
    has_web_matches = len(live_results) > 0

    # Subjectivity
    blob = TextBlob(user_input)
    subjectivity = blob.sentiment.subjectivity * 100

    # RoBERTa Emotion
    raw_emotions = emotion_analyzer(user_input[:512])
    emotions_list = raw_emotions[0] if isinstance(raw_emotions[0], list) else raw_emotions
    top_emotion = analyze_emotion_lightweight(user_input[:512])

    # Final Verdict Logic
    if pred == 0 and has_web_matches:
        verdict = "✅ AUTHENTIC ARTICLE\nVerified by structural patterns & confirmed by live search hits across news index databases."
    elif pred == 0 and not has_web_matches:
        verdict = "✅ LIKELY AUTHENTIC\nLinguistic structure matches objective journalistic patterns, though not explicitly indexed in live news feeds."
    elif pred == 1 and not has_web_matches:
        verdict = "⚠️ POTENTIAL DISINFORMATION / FAKE NEWS\nFlagged by ML model due to sensationalist phrasing and zero verified mainstream web coverage."
    else:
        verdict = "🟡 MIXED / SENSATIONAL COVERAGE\nModel flagged sensational phrasing, but live coverage exists on web indices. Manual check recommended."

    # Format Web Matches
    sources_text = ""
    if live_results:
        for idx, res in enumerate(live_results[:5], 1):
            sources_text += f"{idx}. [{res['engine']} | {res['source']}] {res['title']}\n"
    else:
        sources_text = "No matching mainstream coverage detected across Google News or DuckDuckGo."

    return (
        verdict,
        f"{local_confidence:.1f}% ({stance_str})",
        f"{subjectivity:.1f}% (0% = Objective, 100% = Opinion)",
        f"{top_emotion['label'].upper()} ({top_emotion['score']*100:.1f}% Intensity)",
        sources_text
    )


# =========================================================
# 4. WEBPAGE DESIGN & EMBEDDED JUPYTER UI
# =========================================================
# Custom Dark Emerald Theme Styling
custom_css = """
body, .gradio-container { background-color: #0b130e !important; color: #e2f1e7 !important; }
h1, h2, h3 { color: #2ecc71 !important; }
textarea, input { background-color: #122117 !important; color: #e2f1e7 !important; border: 1px solid #27ae60 !important; }
button { background: linear-gradient(135deg, #27ae60, #2ecc71) !important; color: #0b130e !important; font-weight: bold !important; }
"""

with gr.Blocks(css=custom_css, title="VeritasAI Verification Portal") as demo:
    gr.Markdown("# 🌿 VeritasAI Verification Portal")
    gr.Markdown("Multi-Layer AI Fake News & Propaganda Detector grounded with Live Search.")

    with gr.Row():
        input_text = gr.Textbox(
            lines=5, 
            placeholder="Paste news headline or full article paragraph here...", 
            label="Article Text Input"
        )

    verify_btn = gr.Button("Run Multi-Layer Verification", variant="primary")

    with gr.Row():
        verdict_output = gr.Textbox(label="Final Analysis Verdict", lines=3)

    with gr.Row():
        ml_conf = gr.Textbox(label="ML Stance Confidence")
        subj_score = gr.Textbox(label="Subjectivity Score")
        emo_score = gr.Textbox(label="Dominant Emotion")

    with gr.Row():
        web_refs = gr.Textbox(label="Live Web References (Google News & DuckDuckGo)", lines=5)

    verify_btn.click(
        fn=verify_article_ui, 
        inputs=[input_text], 
        outputs=[verdict_output, ml_conf, subj_score, emo_score, web_refs]
    )

# Launch webpage inside Notebook cell AND generate a public web link
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

