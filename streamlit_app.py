#!/usr/bin/env python3
"""
Logistic Regression Phishing Detector - Streamlit 應用
互動式可視化和預測界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 頁面配置
st.set_page_config(
    page_title="🎣 釣魚網站偵測系統",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義樣式
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-safe {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    .prediction-danger {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# 快取載入模型
@st.cache_resource
def load_models():
    """載入訓練好的模型"""
    try:
        model = joblib.load('phishing_logistic_model.pkl')
        scaler = joblib.load('phishing_scaler.pkl')
        metrics = joblib.load('phishing_metrics.pkl')
        return model, scaler, metrics
    except FileNotFoundError:
        return None, None, None

def main():
    st.title("🎣 釣魚網站偵測系統")
    st.subheader("基於 Logistic Regression 的釣魚攻擊識別")
    
    # 側邊欄導航
    page = st.sidebar.radio("選擇功能", ["📊 儀表板", "🔮 即時預測", "📈 模型評估", "ℹ️ 系統說明"])
    
    model, scaler, metrics = load_models()
    
    if model is None:
        st.error("❌ 找不到模型檔案，請先執行 train_phishing_model.py")
        return
    
    # ============================================
    # 頁面 1: 儀表板
    # ============================================
    if page == "📊 儀表板":
        st.header("📊 模型性能儀表板")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 測試準確率", f"{metrics['test_accuracy']:.2%}", 
                     delta=f"{(metrics['test_accuracy']-metrics['train_accuracy'])*100:.2f}%")
        with col2:
            st.metric("🎯 測試精準率", f"{metrics['test_precision']:.2%}")
        with col3:
            st.metric("🎯 測試召回率", f"{metrics['test_recall']:.2%}")
        with col4:
            st.metric("🎯 F1 分數", f"{metrics['test_f1']:.4f}")
        
        st.divider()
        
        # 性能對比表
        st.subheader("訓練集 vs 測試集性能對比")
        comparison_df = pd.DataFrame({
            '指標': ['準確率', '精準率', '召回率', 'F1 分數', 'AUC'],
            '訓練集': [metrics['train_accuracy'], metrics['train_precision'], 
                     metrics['train_recall'], metrics['train_f1'], metrics['train_auc']],
            '測試集': [metrics['test_accuracy'], metrics['test_precision'], 
                     metrics['test_recall'], metrics['test_f1'], metrics['test_auc']]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # 視覺化
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.set_index('指標')[['訓練集', '測試集']].plot(kind='bar', ax=ax, color=['#667eea', '#764ba2'])
        ax.set_title('模型性能對比', fontweight='bold', fontsize=14)
        ax.set_ylabel('分數')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.info(f"✓ 5 折交叉驗證平均準確率: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    # ============================================
    # 頁面 2: 即時預測
    # ============================================
    elif page == "🔮 即時預測":
        st.header("🔮 即時釣魚網站檢測")
        
        st.info("💡 輸入釣魚網站特徵值進行即時預測")
        
        # 範例特徵數 (根據實際資料集調整)
        num_features = scaler.n_features_in_
        
        col1, col2 = st.columns(2)
        
        feature_values = []
        for i in range(num_features):
            with col1 if i % 2 == 0 else col2:
                value = st.slider(
                    f"特徵 {i+1}",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1
                )
                feature_values.append(value)
        
        if st.button("🔍 進行預測", use_container_width=True):
            # 準備輸入
            features_array = np.array(feature_values).reshape(1, -1)
            
            # 進行預測
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            
            st.divider()
            st.subheader("預測結果")
            
            if prediction == 1:
                st.markdown(
                    f"<div class='prediction-danger'><h3>⚠️ 釣魚網站 (Phishing)</h3><p>置信度: {probability[1]:.2%}</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='prediction-safe'><h3>✅ 正常網站 (Legitimate)</h3><p>置信度: {probability[0]:.2%}</p></div>",
                    unsafe_allow_html=True
                )
            
            # 詳細概率
            col1, col2 = st.columns(2)
            with col1:
                st.metric("正常網站概率", f"{probability[0]:.2%}")
            with col2:
                st.metric("釣魚網站概率", f"{probability[1]:.2%}")
    
    # ============================================
    # 頁面 3: 模型評估
    # ============================================
    elif page == "📈 模型評估":
        st.header("📈 詳細模型評估")
        
        tab1, tab2, tab3 = st.tabs(["性能指標", "特徵分析", "圖表"])
        
        with tab1:
            st.subheader("完整評估指標")
            metrics_df = pd.DataFrame({
                '指標': list(metrics.keys()),
                '數值': [f"{v:.4f}" for v in metrics.values()]
            })
            st.dataframe(metrics_df, use_container_width=True)
        
        with tab2:
            st.subheader("模型係數")
            coefficients = model.coef_[0]
            coef_df = pd.DataFrame({
                '特徵': [f"特徵 {i+1}" for i in range(len(coefficients))],
                '係數': coefficients
            }).sort_values('係數', key=abs, ascending=False)
            
            st.dataframe(coef_df, use_container_width=True)
            
            # 係數視覺化
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#28a745' if x > 0 else '#dc3545' for x in coef_df['係數']]
            ax.barh(coef_df['特徵'], coef_df['係數'], color=colors)
            ax.set_xlabel('係數值')
            ax.set_title('特徵係數 (綠=正相關釣魚, 紅=負相關釣魚)', fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            st.pyplot(fig)
        
        with tab3:
            st.subheader("訓練過程圖表")
            
            # 顯示之前生成的圖表
            try:
                img1 = plt.imread('01_feature_distribution.png')
                st.image(img1, caption='特徵分佈', use_container_width=True)
            except:
                st.warning("特徵分佈圖表未找到")
            
            try:
                img2 = plt.imread('02_confusion_matrix_roc.png')
                st.image(img2, caption='混淆矩陣與 ROC 曲線', use_container_width=True)
            except:
                st.warning("混淆矩陣圖表未找到")
    
    # ============================================
    # 頁面 4: 系統說明
    # ============================================
    elif page == "ℹ️ 系統說明":
        st.header("ℹ️ 系統說明")
        
        st.subheader("📝 項目簡介")
        st.write("""
        這是一個基於 **Logistic Regression** 的釣魚網站自動偵測系統。
        
        **核心功能:**
        - 🤖 使用邏輯迴歸進行二元分類
        - 📊 自動化前處理與特徵標準化
        - 📈 完整的模型評估與驗證
        - 🔮 即時預測與置信度展示
        """)
        
        st.subheader("🔧 前處理步驟")
        st.write("""
        1. **資料載入** - 從 CSV 讀取釣魚網站資料
        2. **異常值檢測** - 使用 IQR 方法識別異常值
        3. **特徵標準化** - StandardScaler 正規化所有特徵
        4. **數據分割** - 80:20 訓練/測試分割（分層抽樣）
        5. **模型訓練** - Logistic Regression 最大似然估計
        6. **模型評估** - 多指標評估與交叉驗證
        """)
        
        st.subheader("📊 評估指標說明")
        
        metrics_info = {
            "準確率 (Accuracy)": "正確預測佔總預測的比例",
            "精準率 (Precision)": "在所有預測為釣魚的中，真正是釣魚的比例",
            "召回率 (Recall)": "在所有真正的釣魚中，被正確識別的比例",
            "F1 分數": "精準率和召回率的調和平均數",
            "AUC": "ROC 曲線下的面積，越接近 1 越好"
        }
        
        for metric, explanation in metrics_info.items():
            st.write(f"**{metric}**: {explanation}")
        
        st.subheader("🎯 使用指南")
        st.write("""
        1. 進入「即時預測」頁面
        2. 調整滑塊設定網站特徵值
        3. 點擊「進行預測」按鈕
        4. 查看預測結果與置信度
        """)

if __name__ == "__main__":
    main()
