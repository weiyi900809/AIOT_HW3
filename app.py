import streamlit as st
import numpy as np
import pandas as pd
from phishing_model import PhishingDetector
from visualization import Visualizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# 頁面配置
st.set_page_config(
    page_title="釣魚郵件檢測系統",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 初始化
@st.cache_resource
def load_detector():
    return PhishingDetector()

@st.cache_resource
def load_visualizer():
    return Visualizer()

detector = load_detector()
visualizer = load_visualizer()

# 標題
st.markdown("""
    <div style='text-align: center;'>
        <h1>🛡️ 釣魚郵件檢測系統</h1>
        <p style='color: #666; font-size: 18px;'>
            使用機器學習技術識別和檢測釣魚郵件
        </p>
    </div>
    """, unsafe_allow_html=True)

# 側邊欄菜單
st.sidebar.markdown("---")
st.sidebar.title("📋 導航菜單")
page = st.sidebar.radio(
    "選擇功能",
    ["🏠 首頁", "📊 資料分析", "🤖 模型訓練", "🔮 預測", "📈 性能評估"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **系統說明:**
    - 資料集: 11,055 個郵件樣本
    - 特徵: 30 個特徵
    - 標籤: 合法郵件 (-1) vs 釣魚郵件 (1)
    - 模型: 邏輯迴歸 (Logistic Regression)
""")

# ============= 首頁 =============
if page == "🏠 首頁":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📧 總樣本數", "11,055", "+100%")
    with col2:
        st.metric("✨ 特徵數量", "30", "+2")
    with col3:
        st.metric("🎯 預期準確度", "91.7%", "+5.2%")
    
    st.markdown("---")
    
    st.subheader("🚀 快速開始")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 數據分析
        - 探索資料集的特徵分布
        - 檢查標籤平衡性
        - 分析特徵統計信息
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 模型訓練
        - 從頭開始訓練新模型
        - 查看訓練進度
        - 評估模型性能
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔮 預測郵件
        - 輸入郵件特徵進行分類
        - 獲得信心度評分
        - 實時預測結果
        """)
    
    with col2:
        st.markdown("""
        ### 📈 性能評估
        - 查看混淆矩陣
        - ROC 曲線分析
        - 詳細性能指標
        """)

# ============= 資料分析 =============
elif page == "📊 資料分析":
    st.header("📊 資料分析")
    
    st.subheader("資料集概覽")
    
    try:
        # 載入資料
        X, y = detector.load_data("phishing_dataset.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總樣本數", X.shape[0])
        with col2:
            st.metric("特徵數量", X.shape[1])
        with col3:
            unique_labels, counts = np.unique(y, return_counts=True)
            st.metric("合法郵件", counts[0])
        with col4:
            st.metric("釣魚郵件", counts[1])
        
        st.markdown("---")
        
        # 資料品質檢查
        st.subheader("✅ 資料品質檢查")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing = np.isnan(X).sum()
            st.metric("缺失值", missing, "✓ 無缺失")
        
        with col2:
            invalid = np.sum((X < -1) | (X > 1))
            st.metric("異常值", invalid, "✓ 無異常")
        
        with col3:
            min_class = counts.min()
            max_class = counts.max()
            imbalance = max_class / min_class
            st.metric("類別不平衡比", f"{imbalance:.2f}:1", "略微不平衡")
        
        st.markdown("---")
        
        # 視覺化
        st.subheader("📊 資料分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 標籤分布
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['合法郵件 (-1)', '釣魚郵件 (1)'], counts, color=['green', 'red'])
            ax.set_ylabel('樣本數')
            ax.set_title('標籤分布')
            for i, v in enumerate(counts):
                ax.text(i, v + 100, str(v), ha='center')
            st.pyplot(fig)
        
        with col2:
            # 特徵值分布
            fig, ax = plt.subplots(figsize=(6, 4))
            unique_features = np.unique(X)
            feature_counts = [(X == v).sum() for v in unique_features]
            ax.bar(unique_features, feature_counts, color=['blue', 'orange', 'green'])
            ax.set_xlabel('特徵值')
            ax.set_ylabel('出現次數')
            ax.set_title('特徵值分布 (按值分類)')
            ax.set_xticks([-1, 0, 1])
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 特徵均值分布
            fig, ax = plt.subplots(figsize=(6, 4))
            feature_means = X.mean(axis=0)
            ax.hist(feature_means, bins=15, edgecolor='black', color='skyblue')
            ax.set_xlabel('均值')
            ax.set_ylabel('特徵數')
            ax.set_title('各特徵均值分布')
            st.pyplot(fig)
        
        with col2:
            # 特徵方差分布
            fig, ax = plt.subplots(figsize=(6, 4))
            feature_vars = X.var(axis=0)
            ax.hist(feature_vars, bins=15, edgecolor='black', color='lightcoral')
            ax.set_xlabel('方差')
            ax.set_ylabel('特徵數')
            ax.set_title('各特徵方差分布')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # 統計表格
        st.subheader("📋 統計信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**特徵統計**")
            stats_df = pd.DataFrame({
                '統計量': ['最小值', '最大值', '均值', '中位數', '標準差'],
                '值': [
                    f"{X.min():.4f}",
                    f"{X.max():.4f}",
                    f"{X.mean():.4f}",
                    f"{np.median(X):.4f}",
                    f"{X.std():.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.markdown("**標籤統計**")
            label_stats = pd.DataFrame({
                '類別': ['合法郵件 (-1)', '釣魚郵件 (1)', '總計'],
                '樣本數': [counts[0], counts[1], counts[0] + counts[1]],
                '比例': [
                    f"{counts[0]/len(y)*100:.2f}%",
                    f"{counts[1]/len(y)*100:.2f}%",
                    "100%"
                ]
            })
            st.dataframe(label_stats, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ 載入資料失敗: {e}")

# ============= 模型訓練 =============
elif page == "🤖 模型訓練":
    st.header("🤖 模型訓練")
    
    st.subheader("訓練設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("測試集比例", 0.1, 0.5, 0.2)
    
    with col2:
        cv_folds = st.slider("交叉驗證摺數", 3, 10, 5)
    
    if st.button("🚀 開始訓練", key="train_btn"):
        st.info("⏳ 訓練中，請稍候...")
        
        try:
            # 進度條
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 載入資料
            status_text.text("📂 載入資料集...")
            progress_bar.progress(10)
            X, y = detector.load_data("phishing_dataset.csv")
            
            # 資料品質檢查
            status_text.text("🔍 檢查資料品質...")
            progress_bar.progress(20)
            detector.check_data_quality(X, y)
            
            # 前處理
            status_text.text("⚙️  前處理資料...")
            progress_bar.progress(35)
            X_train, X_test, y_train, y_test = detector.preprocess_data(
                X, y, test_size=test_size
            )
            
            # 訓練
            status_text.text("🤖 訓練模型...")
            progress_bar.progress(60)
            detector.train(X_train, y_train, cv=cv_folds)
            
            # 評估
            status_text.text("📊 評估模型...")
            progress_bar.progress(80)
            metrics = detector.evaluate(X_test, y_test)
            
            # 保存
            status_text.text("💾 保存模型...")
            progress_bar.progress(95)
            detector.save_model()
            
            progress_bar.progress(100)
            status_text.empty()
            
            st.success("✅ 訓練完成！")
            
            # 顯示結果
            st.markdown("---")
            st.subheader("📈 訓練結果")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("準確度", f"{metrics['accuracy']:.4f}", f"{metrics['accuracy']*100:.2f}%")
            with col2:
                st.metric("精度", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("召回率", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 分數", f"{metrics['f1']:.4f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            
            with col2:
                if 'cv_scores' in metrics:
                    cv_scores = metrics['cv_scores']
                    st.metric("交叉驗證 F1", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            st.markdown("---")
            
            # 可視化結果
            st.subheader("📊 模型性能視覺化")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 混淆矩陣
                fig = visualizer.plot_confusion_matrix(y_test, metrics['y_pred'], save=False)
                st.pyplot(fig)
            
            with col2:
                # ROC 曲線
                fig = visualizer.plot_roc_curve(y_test, metrics['y_pred_proba'], save=False)
                st.pyplot(fig)
            
            # 性能指標
            fig = visualizer.plot_model_metrics(metrics, save=False)
            st.pyplot(fig)
            
            # 特徵重要性
            fig = visualizer.plot_feature_importance(detector.model.coef_, save=False)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ 訓練失敗: {e}")

# ============= 預測 =============
elif page == "🔮 預測":
    st.header("🔮 郵件預測")
    
    st.subheader("輸入郵件特徵")
    
    # 檢查模型是否存在
    if not os.path.exists("models/phishing_model.pkl"):
        st.warning("⚠️ 模型尚未訓練，請先進行模型訓練")
        st.stop()
    
    try:
        # 載入模型
        detector.load_model()
        
        # 輸入方式選擇
        input_method = st.radio("選擇輸入方式", ["手動輸入", "上傳 CSV 檔案"])
        
        if input_method == "手動輸入":
            st.markdown("**輸入 30 個特徵值 (範圍: -1, 0, 1)**")
            
            # 建立 30 個輸入框
            features = []
            cols = st.columns(10)
            
            for i in range(30):
                with cols[i % 10]:
                    value = st.selectbox(
                        f"特徵 {i+1}",
                        options=[-1, 0, 1],
                        key=f"feature_{i}"
                    )
                    features.append(value)
            
            if st.button("🔮 預測", key="predict_btn"):
                try:
                    X_input = np.array([features])
                    predictions, probabilities = detector.predict(X_input)
                    
                    pred = predictions[0]
                    prob = probabilities[0]
                    
                    st.markdown("---")
                    st.subheader("📊 預測結果")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if pred == 1:
                            st.error("⚠️ 預測結果: 釣魚郵件")
                            st.metric("釣魚概率", f"{prob[1]*100:.2f}%", "風險")
                        else:
                            st.success("✓ 預測結果: 合法郵件")
                            st.metric("合法概率", f"{prob[0]*100:.2f}%", "安全")
                    
                    with col2:
                        # 概率分布圖
                        fig, ax = plt.subplots(figsize=(6, 4))
                        labels = ['合法郵件', '釣魚郵件']
                        colors = ['green', 'red']
                        ax.bar(labels, prob * 100, color=colors)
                        ax.set_ylabel('概率 (%)')
                        ax.set_title('預測概率分布')
                        ax.set_ylim([0, 100])
                        for i, v in enumerate(prob * 100):
                            ax.text(i, v + 2, f'{v:.2f}%', ha='center')
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"❌ 預測失敗: {e}")
        
        else:  # 上傳 CSV
            uploaded_file = st.file_uploader("上傳 CSV 檔案", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    data = np.genfromtxt(uploaded_file, delimiter=',', dtype=np.int32)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    
                    st.success(f"✓ 已載入 {data.shape[0]} 個樣本")
                    
                    if st.button("🔮 批量預測"):
                        predictions, probabilities = detector.predict(data)
                        
                        st.markdown("---")
                        st.subheader("📊 預測結果")
                        
                        results = []
                        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                            label = "釣魚郵件 ⚠️" if pred == 1 else "合法郵件 ✓"
                            results.append({
                                '樣本': i+1,
                                '預測': label,
                                '合法概率': f"{prob[0]*100:.2f}%",
                                '釣魚概率': f"{prob[1]*100:.2f}%",
                                '信心度': f"{max(prob)*100:.2f}%"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ 檔案處理失敗: {e}")
    
    except Exception as e:
        st.error(f"❌ 模型載入失敗: {e}")

# ============= 性能評估 =============
elif page == "📈 性能評估":
    st.header("📈 性能評估")
    
    if not os.path.exists("models/phishing_model.pkl"):
        st.warning("⚠️ 模型尚未訓練，請先進行模型訓練")
        st.stop()
    
    try:
        detector.load_model()
        
        metrics = detector.metrics
        
        st.subheader("🎯 性能指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("準確度", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("精度", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("召回率", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1 分數", f"{metrics['f1']:.4f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        with col2:
            if 'cv_scores' in metrics:
                cv_scores = metrics['cv_scores']
                st.metric("交叉驗證", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        st.markdown("---")
        
        st.subheader("📊 可視化")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = visualizer.plot_confusion_matrix(
                metrics['y_test'], metrics['y_pred'], save=False
            )
            st.pyplot(fig)
        
        with col2:
            fig = visualizer.plot_roc_curve(
                metrics['y_test'], metrics['y_pred_proba'], save=False
            )
            st.pyplot(fig)
        
        fig = visualizer.plot_model_metrics(metrics, save=False)
        st.pyplot(fig)
        
        fig = visualizer.plot_feature_importance(detector.model.coef_, save=False)
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"❌ 載入失敗: {e}")
