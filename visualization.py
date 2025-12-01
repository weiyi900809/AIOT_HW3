import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from pathlib import Path

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    \"\"\"資料視覺化類別\"\"\"
    
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_data_distribution(self, X, y, save=True):
        \"\"\"繪製資料分布圖\"\"\"
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('資料集特徵分布分析', fontsize=16, fontweight='bold')
        
        # 標籤分布
        unique, counts = np.unique(y, return_counts=True)
        axes[0, 0].bar(['合法郵件 (-1)', '釣魚郵件 (1)'], counts)
        axes[0, 0].set_title('標籤分布')
        axes[0, 0].set_ylabel('樣本數')
        for i, v in enumerate(counts):
            axes[0, 0].text(i, v + 100, str(v), ha='center')
        
        # 特徵值分布
        unique_features = np.unique(X)
        feature_counts = [(X == v).sum() for v in unique_features]
        axes[0, 1].bar(unique_features, feature_counts)
        axes[0, 1].set_title('特徵值分布')
        axes[0, 1].set_xlabel('特徵值')
        axes[0, 1].set_ylabel('出現次數')
        
        # 每個特徵的均值
        feature_means = X.mean(axis=0)
        axes[1, 0].hist(feature_means, bins=20, edgecolor='black')
        axes[1, 0].set_title('特徵均值分布')
        axes[1, 0].set_xlabel('均值')
        axes[1, 0].set_ylabel('特徵數')
        
        # 特徵方差
        feature_vars = X.var(axis=0)
        axes[1, 1].hist(feature_vars, bins=20, edgecolor='black')
        axes[1, 1].set_title('特徵方差分布')
        axes[1, 1].set_xlabel('方差')
        axes[1, 1].set_ylabel('特徵數')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/data_distribution.png', dpi=300)
            print(f"✓ 已保存: {self.output_dir}/data_distribution.png")
        
        return fig
    
    def plot_confusion_matrix(self, y_test, y_pred, save=True):
        \"\"\"繪製混淆矩陣\"\"\"
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['合法郵件', '釣魚郵件'],
                   yticklabels=['合法郵件', '釣魚郵件'])
        
        ax.set_title('混淆矩陣', fontsize=14, fontweight='bold')
        ax.set_ylabel('實際標籤')
        ax.set_xlabel('預測標籤')
        
        # 添加性能指標
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        textstr = f'準確度: {accuracy:.3f}\\n'
        textstr += f'敏感性: {sensitivity:.3f}\\n'
        textstr += f'特異性: {specificity:.3f}'
        
        ax.text(1.5, 0.5, textstr, transform=ax.transData,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300)
            print(f"✓ 已保存: {self.output_dir}/confusion_matrix.png")
        
        return fig
    
    def plot_roc_curve(self, y_test, y_pred_proba, save=True):
        \"\"\"繪製 ROC 曲線\"\"\"
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲線 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='隨機分類器')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('偽正率 (False Positive Rate)')
        ax.set_ylabel('真正率 (True Positive Rate)')
        ax.set_title('ROC 曲線', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/roc_curve.png', dpi=300)
            print(f"✓ 已保存: {self.output_dir}/roc_curve.png")
        
        return fig
    
    def plot_model_metrics(self, metrics, save=True):
        \"\"\"繪製模型性能指標\"\"\"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('模型性能指標', fontsize=16, fontweight='bold')
        
        # 主要指標
        metric_names = ['準確度', '精度', '召回率', 'F1 分數']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0)
        ]
        
        axes[0, 0].bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_title('分類性能指標')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_ylabel('分數')
        for i, v in enumerate(metric_values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 交叉驗證分數
        if 'cv_scores' in metrics:
            cv_scores = metrics['cv_scores']
            axes[0, 1].boxplot([cv_scores], labels=['交叉驗證'])
            axes[0, 1].set_title('交叉驗證 F1 分數分布')
            axes[0, 1].set_ylabel('F1 分數')
            axes[0, 1].text(1, cv_scores.mean() + 0.02, f'均值: {cv_scores.mean():.3f}', ha='center')
        
        # ROC-AUC
        roc_auc = metrics.get('roc_auc', 0)
        axes[1, 0].barh(['ROC-AUC'], [roc_auc], color='#9467bd')
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_title('ROC-AUC 分數')
        axes[1, 0].text(roc_auc + 0.02, 0, f'{roc_auc:.3f}', va='center')
        
        # 混淆矩陣相關指標
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            axes[1, 1].bar(['敏感性', '特異性'], [sensitivity, specificity], color=['#17becf', '#bcbd22'])
            axes[1, 1].set_title('混淆矩陣衍生指標')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].set_ylabel('分數')
            axes[1, 1].text(0, sensitivity + 0.02, f'{sensitivity:.3f}', ha='center')
            axes[1, 1].text(1, specificity + 0.02, f'{specificity:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/model_metrics.png', dpi=300)
            print(f"✓ 已保存: {self.output_dir}/model_metrics.png")
        
        return fig
    
    def plot_feature_importance(self, coefficients, top_n=15, save=True):
        \"\"\"繪製特徵重要性\"\"\"
        # 計算特徵重要性（係數的絕對值）
        importance = np.abs(coefficients[0])
        indices = np.argsort(importance)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importance[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([f'特徵 {i+1}' for i in indices])
        ax.set_xlabel('重要性（係數絕對值）')
        ax.set_title(f'前 {top_n} 個最重要的特徵', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300)
            print(f"✓ 已保存: {self.output_dir}/feature_importance.png")
        
        return fig
