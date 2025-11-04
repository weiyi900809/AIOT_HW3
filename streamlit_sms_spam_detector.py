# SMS Spam Detector with Streamlit UI
"""
使用方式：
執行時於命令列輸入
    streamlit run streamlit_sms_spam_detector.py
並將 sms_spam_no_header.csv 放在同目錄下
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io

st.set_page_config(page_title="SMS Spam Detector", layout="wide")

st.title("📧 SMS Spam Detector 垃圾郵件偵測 (Logistic Regression)")

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv('sms_spam_no_header.csv', header=None, names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    return df

df = load_data()

# 資料探索區
with st.sidebar:
    st.header('🔍 資料探索')
    st.dataframe(df.head(10))
    st.write(f"樣本數: {df.shape[0]}, Ham: {sum(df['label']=='ham')}, Spam: {sum(df['label']=='spam')}")

    st.write('---')
    fig1, ax1 = plt.subplots()
    ax1.pie(df['label'].value_counts(), labels=['ham', 'spam'], autopct="%.1f%%", colors=['skyblue', 'salmon'])
    st.pyplot(fig1)
    st.write('類別分布')

    st.write('---')
    st.write('訊息長度分布')
    fig2, ax2 = plt.subplots()
    for label, color in [('ham','blue'),('spam','red')]:
        ax2.hist(df[df['label'] == label]['message_length'], bins=50, alpha=0.7, label=label, color=color, density=True)
    ax2.legend(); ax2.set_xlabel('訊息長度'); ax2.set_ylabel('密度')
    st.pyplot(fig2)

    st.write('---')
    st.write('Ham & Spam 文字雲')
    ham_text = ' '.join(df[df['label']=='ham']['message'])
    spam_text = ' '.join(df[df['label']=='spam']['message'])
    col1, col2 = st.columns(2)
    with col1:
        st.write('Ham')
        wc1 = WordCloud(width=250, height=180, background_color='white').generate(ham_text)
        st.image(wc1.to_array())
    with col2:
        st.write('Spam')
        wc2 = WordCloud(width=250, height=180, background_color='white').generate(spam_text)
        st.image(wc2.to_array())

# 訓練與測試模型
@st.cache_resource(show_spinner=True)
def train_model(df):
    X = df['message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer, X_test, y_test, X_test_tfidf

model, vectorizer, X_test, y_test, X_test_tfidf = train_model(df)
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)[:,1]
acc = accuracy_score(y_test, y_pred)

# 主頁結果
st.subheader('🎯 模型效能')
st.write(f'**測試集準確率:** {acc*100:.2f}%')

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Ham 準確率", f"{accuracy_score(y_test[y_test==0], y_pred[y_test==0])*100:.2f}%")
with colB:
    st.metric("Spam 準確率", f"{accuracy_score(y_test[y_test==1], y_pred[y_test==1])*100:.2f}%")
with colC:
    st.metric("AUC 分數", f"{auc(*roc_curve(y_test, y_proba)[:2]):.3f}")

report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)
st.write('**詳細分類報告:**')
st.dataframe(pd.DataFrame(report).T)

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax_cm)
ax_cm.set_xlabel('預測'); ax_cm.set_ylabel('實際'); ax_cm.set_title('混淆矩陣')
st.pyplot(fig_cm)

# ROC 曲線
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.3f}', color='orange')
ax_roc.plot([0,1],[0,1],'--',color='grey')
ax_roc.set_xlabel('假陽性率(FPR)')
ax_roc.set_ylabel('真陽性率(TPR)')
ax_roc.legend()
ax_roc.set_title('ROC 曲線')
st.pyplot(fig_roc)

# 用戶互動預測
st.subheader("📝 試試看你的訊息!")
user_input = st.text_area("請輸入一則 SMS 內容：", "Congratulations! You've won a free prize. Call now!")
if user_input:
    arr = vectorizer.transform([user_input])
    pre = model.predict(arr)[0]
    proba = model.predict_proba(arr)[0]
    st.write(f'預測結果：**{"Spam" if pre==1 else "Ham"}** (Spam 機率: {proba[1]:.2%}, Ham 機率: {proba[0]:.2%})')

# 特徵重要性
st.subheader("🔍 重要特徵分析")
features = vectorizer.get_feature_names_out()
importances = np.abs(model.coef_[0])
idx = np.argsort(importances)[-20:][::-1]
fig_imp, ax_imp = plt.subplots(figsize=(6,5))
sns.barplot(y=features[idx], x=importances[idx], palette='viridis', ax=ax_imp)
ax_imp.set_title('Top 20 文字特徵影響力')
st.pyplot(fig_imp)
