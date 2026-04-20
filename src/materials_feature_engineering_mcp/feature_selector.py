"""
特征选择工具
使用递归特征消除（RFE）结合交叉验证（CV）选择最佳特征组合
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import safe data loading function
from .utils import _load_data_safe


class FeatureSelector:
    """特征选择器类"""
    
    def __init__(self, task_type: str = 'regression', cv_folds: int = 5, 
                 min_features: int = 1, step: int = 1, scoring: str | None = None):
        """
        初始化特征选择器
        
        Args:
            task_type: 任务类型 ('regression' 或 'classification')
            cv_folds: 交叉验证折数
            min_features: 最少保留的特征数
            step: 每次迭代移除的特征数
            scoring: 评分指标 (None则使用默认指标)
        """
        self.task_type = task_type.lower()
        self.cv_folds = cv_folds
        self.min_features = min_features
        self.step = step
        self.scoring = scoring
        
        # 选择合适的模型和评分指标
        if self.task_type == 'regression':
            self.estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.scoring = scoring or 'neg_mean_squared_error'
            self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:  # classification
            self.estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.scoring = scoring or 'f1_weighted'
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        使用RFE-CV选择最佳特征组合
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            包含选择结果的字典
        """
        print(f"\n开始特征选择 (RFE-CV)...")
        print(f"任务类型: {self.task_type}")
        print(f"原始特征数: {X.shape[1]}")
        print(f"样本数: {X.shape[0]}")
        print(f"交叉验证折数: {self.cv_folds}")
        print()
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 执行RFE-CV
        print("执行递归特征消除 + 交叉验证...")
        rfecv = RFECV(
            estimator=self.estimator,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.min_features,
            n_jobs=-1,
            verbose=1
        )
        
        rfecv.fit(X_scaled, y)
        feature_counts = self._build_feature_count_sequence(X.shape[1])
        if len(feature_counts) != len(rfecv.cv_results_['mean_test_score']):
            feature_counts = list(range(1, len(rfecv.cv_results_['mean_test_score']) + 1))
        best_score_index = feature_counts.index(rfecv.n_features_)
        
        # 获取选择的特征
        selected_features = X.columns[rfecv.support_].tolist()
        rejected_features = X.columns[~rfecv.support_].tolist()
        
        print(f"\n✓ 特征选择完成！")
        print(f"  最优特征数: {rfecv.n_features_}")
        print(f"  选择的特征数: {len(selected_features)}")
        print(f"  拒绝的特征数: {len(rejected_features)}")
        print(f"  最佳CV评分: {rfecv.cv_results_['mean_test_score'][best_score_index]:.4f}")
        
        # 计算特征重要性（使用最终模型）
        feature_importances: Dict[str, float] = {}
        if hasattr(rfecv.estimator_, 'feature_importances_'):
            importances = rfecv.estimator_.feature_importances_  # type: ignore
            for feat, imp in zip(selected_features, importances):
                feature_importances[feat] = float(imp)
        
        # 构建结果字典
        result = {
            'selected_features': selected_features,
            'rejected_features': rejected_features,
            'n_features': rfecv.n_features_,
            'ranking': dict(zip(X.columns, rfecv.ranking_)),
            'feature_counts': feature_counts,
            'cv_scores': rfecv.cv_results_['mean_test_score'],
            'cv_scores_std': rfecv.cv_results_['std_test_score'],
            'feature_importances': feature_importances,
            'best_score': rfecv.cv_results_['mean_test_score'][best_score_index],
            'rfecv_object': rfecv,
            'scaler': scaler
        }
        
        return result
    
    def generate_report(self, result: Dict[str, Any], X: pd.DataFrame, 
                       output_path: str) -> str:
        """
        生成特征选择报告
        
        Args:
            result: 特征选择结果
            X: 原始特征数据
            output_path: 输出路径
            
        Returns:
            报告文件路径
        """
        print("\n生成特征选择报告...")
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        
        # 1. CV Score Curve
        ax1 = plt.subplot(2, 3, 1)
        n_features_range = result.get('feature_counts', self._build_feature_count_sequence(len(X.columns)))
        scores = result['cv_scores']
        scores_std = result['cv_scores_std']
        
        ax1.plot(n_features_range, scores, 'o-', linewidth=2, markersize=6)
        ax1.fill_between(n_features_range, 
                         scores - scores_std,
                         scores + scores_std, 
                         alpha=0.2)
        ax1.axvline(result['n_features'], color='r', linestyle='--', 
                   label=f'Optimal: {result["n_features"]} features')
        ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('CV Score', fontsize=12, fontweight='bold')
        ax1.set_title('RFE-CV Score Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Importance (selected features)
        if result['feature_importances']:
            ax2 = plt.subplot(2, 3, 2)
            importances_df = pd.DataFrame({
                'feature': list(result['feature_importances'].keys()),
                'importance': list(result['feature_importances'].values())
            }).sort_values('importance', ascending=True).tail(20)
            
            ax2.barh(importances_df['feature'], importances_df['importance'])
            ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax2.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Feature Ranking Distribution
        ax3 = plt.subplot(2, 3, 3)
        rankings = list(result['ranking'].values())
        ax3.hist(rankings, bins=min(20, max(rankings)), edgecolor='black', alpha=0.7)
        ax3.axvline(1, color='r', linestyle='--', linewidth=2, label='Selected (rank=1)')
        ax3.set_xlabel('Feature Rank', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
        ax3.set_title('Feature Ranking Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Selected vs Rejected Features
        ax4 = plt.subplot(2, 3, 4)
        sizes = [len(result['selected_features']), len(result['rejected_features'])]
        labels = [f"Selected ({sizes[0]})", f"Rejected ({sizes[1]})"]
        colors = ['#2ecc71', '#e74c3c']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Feature Selection Results', fontsize=14, fontweight='bold')
        
        # 5. CV Score Stability
        ax5 = plt.subplot(2, 3, 5)
        ax5.errorbar(range(len(scores)), scores, yerr=scores_std, 
                    fmt='o-', capsize=5, capthick=2, linewidth=2)
        ax5.axhline(result['best_score'], color='g', linestyle='--', 
                   label=f'Best: {result["best_score"]:.4f}')
        ax5.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax5.set_ylabel('CV Score ± Std', fontsize=12, fontweight='bold')
        ax5.set_title('Score Stability Analysis', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Information
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        📊 Feature Selection Summary
        
        Task Type: {self.task_type}
        Original Features: {len(X.columns)}
        Optimal Features: {result['n_features']}
        Retention Rate: {result['n_features']/len(X.columns)*100:.1f}%
        
        Best CV Score: {result['best_score']:.4f}
        Scoring Metric: {self.scoring}
        CV Folds: {self.cv_folds}
        
        Selected: {len(result['selected_features'])}
        Rejected: {len(result['rejected_features'])}
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # 保存图表
        base_path = output_path.rsplit('.csv', 1)[0] if output_path.endswith('.csv') else output_path
        report_path = f"{base_path}_feature_selection_report.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Report saved: {report_path}")
        
        return report_path
    
    def generate_html_report(self, result: Dict[str, Any], X: pd.DataFrame, output_path: str) -> str:
        """生成HTML格式的报告"""
        print("\n生成HTML报告...")
        
        base_path = output_path.rsplit('.csv', 1)[0] if output_path.endswith('.csv') else output_path
        html_path = f"{base_path}_feature_selection_report.html"
        
        # 计算统计信息
        retention_rate = result['n_features'] / len(X.columns) * 100
        
        # 生成HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Selection Report - RFE-CV</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .section {{
            padding: 40px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 40px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: visible;
            margin: 10px 0;
            position: relative;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 1s ease;
            border-radius: 15px;
            min-width: 50px;
        }}
        
        .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #333;
            font-weight: bold;
            font-size: 1.1em;
            white-space: nowrap;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
        }}
        
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .feature-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .feature-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .feature-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .feature-name {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 4px 8px;
            border-radius: 4px;
            color: #667eea;
            font-weight: bold;
        }}
        
        .importance-bar {{
            height: 20px;
            background: linear-gradient(90deg, #4caf50 0%, #8bc34a 100%);
            border-radius: 10px;
            min-width: 20px;
        }}
        
        .rank-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .rank-selected {{
            background: #4caf50;
            color: white;
        }}
        
        .rank-rejected {{
            background: #f44336;
            color: white;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        .metric-card h3 {{
            margin-bottom: 10px;
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 Feature Selection Report</h1>
            <p>RFE-CV (Recursive Feature Elimination with Cross-Validation)</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Task Type</div>
                <div class="stat-value">{self.task_type.upper()}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Original Features</div>
                <div class="stat-value">{len(X.columns)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Selected Features</div>
                <div class="stat-value">{result['n_features']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Retention Rate</div>
                <div class="stat-value">{retention_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best CV Score</div>
                <div class="stat-value">{result['best_score']:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">CV Folds</div>
                <div class="stat-value">{self.cv_folds}</div>
            </div>
        </div>
        
        <!-- Progress Section -->
        <div class="section">
            <h2 class="section-title">📊 Feature Reduction Progress</h2>
            <div class="metric-card">
                <h3>Original → Selected</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {retention_rate}%;"></div>
                    <div class="progress-text">{len(X.columns)} → {result['n_features']} features ({retention_rate:.1f}%)</div>
                </div>
            </div>
        </div>
        
        <!-- Selected Features -->
        <div class="section">
            <h2 class="section-title">✅ Selected Features ({len(result['selected_features'])})</h2>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th>Feature Name</th>
                        <th style="width: 150px;">Importance</th>
                        <th style="width: 200px;">Importance Bar</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add selected features
        max_importance = max(result['feature_importances'].values()) if result['feature_importances'] else 1
        for idx, feat in enumerate(result['selected_features'], 1):
            importance = result['feature_importances'].get(feat, 0)
            bar_width = (importance / max_importance * 100) if max_importance > 0 else 0
            html_content += f"""
                    <tr>
                        <td style="text-align: center;">{idx}</td>
                        <td><span class="feature-name">{feat}</span></td>
                        <td>{importance:.6f}</td>
                        <td>
                            <div class="importance-bar" style="width: {bar_width}%;"></div>
                        </td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <!-- Rejected Features -->
        <div class="section" style="background: #f8f9fa;">
            <h2 class="section-title">❌ Rejected Features ({len(result['rejected_features'])})</h2>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th>Feature Name</th>
                        <th style="width: 150px;">Rank</th>
                        <th style="width: 150px;">Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add rejected features
        rejected_with_rank = [(feat, result['ranking'][feat]) for feat in result['rejected_features']]
        rejected_with_rank.sort(key=lambda x: x[1])
        
        for idx, (feat, rank) in enumerate(rejected_with_rank, 1):
            html_content += f"""
                    <tr>
                        <td style="text-align: center;">{idx}</td>
                        <td><span class="feature-name">{feat}</span></td>
                        <td>{rank}</td>
                        <td><span class="rank-badge rank-rejected">REJECTED</span></td>
                    </tr>
"""
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <!-- Scoring Details -->
        <div class="section">
            <h2 class="section-title">📈 Cross-Validation Scoring Details</h2>
            <div class="metric-card">
                <h3>Scoring Metric: {self.scoring}</h3>
                <p>Number of iterations: {len(result['cv_scores'])}</p>
                <p>Best score achieved at {result['n_features']} features</p>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated by Materials Feature Engineering MCP Tool</p>
            <p>Report Type: RFE-CV Feature Selection</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved: {html_path}")
        return html_path

    def _build_feature_count_sequence(self, total_features: int) -> List[int]:
        if total_features <= self.min_features:
            return [total_features]

        remainder = (total_features - self.min_features) % self.step
        counts = [self.min_features]
        next_count = self.min_features + (remainder if remainder else self.step)

        while next_count <= total_features:
            counts.append(next_count)
            next_count += self.step

        return counts
    
    def save_results(self, result: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                    output_path: str, target_cols: List[str] | None = None, df_original: pd.DataFrame | None = None) -> Tuple[str, str, str, str]:
        """
        保存特征选择结果
        
        Args:
            result: 特征选择结果
            X: 原始特征数据
            y: 目标变量
            output_path: 输出路径
            target_cols: 所有目标列名列表（用于保存完整数据）
            df_original: 原始完整DataFrame（用于获取所有目标列）
            
        Returns:
            (数据文件路径, 报告文件路径, 详情文件路径)
        """
        # 保存选中的特征数据
        X_selected = X[result['selected_features']].copy()
        
        # 重新添加目标列
        if df_original is not None and target_cols and len(target_cols) > 0:
            # 从原始DataFrame获取所有目标列
            y_df = df_original[target_cols].copy()
            result_df = pd.concat([X_selected, y_df], axis=1)
        else:
            # 单个目标变量
            y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index, name=y.name if hasattr(y, 'name') else 'target')
            result_df = pd.concat([X_selected, y_series], axis=1)
        
        data_path = output_path
        result_df.to_csv(data_path, index=False)
        print(f"✓ 选中特征数据已保存: {data_path}")
        
        # 生成可视化报告
        report_path = self.generate_report(result, X, output_path)
        
        # 保存详细报告（文本格式）
        # 确保路径处理正确
        base_path = output_path.rsplit('.csv', 1)[0] if output_path.endswith('.csv') else output_path
        txt_report_path = f"{base_path}_feature_selection_details.txt"
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Feature Selection Detailed Report (RFE-CV)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task Type: {self.task_type}\n")
            f.write(f"Scoring Metric: {self.scoring}\n")
            f.write(f"CV Folds: {self.cv_folds}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Data Overview\n")
            f.write("=" * 80 + "\n")
            f.write(f"Original Features: {len(X.columns)}\n")
            f.write(f"Sample Size: {len(X)}\n")
            f.write(f"Optimal Features: {result['n_features']}\n")
            f.write(f"Retention Rate: {result['n_features']/len(X.columns)*100:.2f}%\n")
            f.write(f"Best CV Score: {result['best_score']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Selected Features ({len(result['selected_features'])})\n")
            f.write("=" * 80 + "\n")
            for i, feat in enumerate(result['selected_features'], 1):
                imp = result['feature_importances'].get(feat, 0)
                f.write(f"{i:3d}. {feat:50s}  Importance: {imp:.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Rejected Features ({len(result['rejected_features'])})\n")
            f.write("=" * 80 + "\n")
            
            # Sort rejected features by rank
            rejected_with_rank = [(feat, result['ranking'][feat]) 
                                  for feat in result['rejected_features']]
            rejected_with_rank.sort(key=lambda x: x[1])
            
            for i, (feat, rank) in enumerate(rejected_with_rank, 1):
                f.write(f"{i:3d}. {feat:50s}  Rank: {rank}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Cross-Validation Score Details\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Features':>10s}  {'Mean Score':>12s}  {'Std Dev':>12s}\n")
            f.write("-" * 80 + "\n")
            
            feature_counts = result.get('feature_counts', self._build_feature_count_sequence(len(X.columns)))
            for n_feat, score, std in zip(feature_counts, result['cv_scores'], result['cv_scores_std']):
                marker = " <- Optimal" if n_feat == result['n_features'] else ""
                f.write(f"{n_feat:>10d}  {score:>12.6f}  {std:>12.6f}{marker}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ 详细报告已保存: {txt_report_path}")
        
        # 生成HTML报告
        html_path = self.generate_html_report(result, X, output_path)
        
        return data_path, report_path, html_path, txt_report_path


def select_best_features(
    data_path: str,
    target_dims: int = 1,
    task_type: str = 'regression',
    cv_folds: int = 5,
    min_features: int = 1,
    step: int = 1,
    output_suffix: str = '_selected_features',
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用RFE-CV选择最佳特征组合
    
    Args:
        data_path: 数据文件路径
        target_dims: 目标变量维度（默认最后n列），默认为1
        task_type: 任务类型 ('regression' 或 'classification')
        cv_folds: 交叉验证折数
        min_features: 最少保留的特征数
        step: 每次迭代移除的特征数
        output_suffix: 输出文件后缀
        output_dir: 输出目录（可选，用于用户隔离）
        
    Returns:
        包含选择结果的字典
    """
    print("=" * 80)
    print("特征选择工具 (RFE-CV)")
    print("=" * 80)
    print(f"数据文件: {data_path}")
    print(f"目标维度: {target_dims} (最后{target_dims}列)")
    print(f"任务类型: {task_type}")
    print()
    
    # 读取数据（支持URL和本地文件）
    print("读取数据...")
    try:
        local_path = _load_data_safe(data_path)
        df = pd.read_csv(local_path)
        print(f"✓ 数据形状: {df.shape}")
    except Exception as e:
        raise ValueError(f"无法读取数据文件: {str(e)}")
    
    # 检查目标维度
    if target_dims <= 0 or target_dims >= len(df.columns):
        raise ValueError(f"目标维度必须在1到{len(df.columns)-1}之间，当前值: {target_dims}")
    
    # 分离特征和目标（目标变量在最后n列）
    target_columns = df.columns[-target_dims:].tolist()
    feature_columns = df.columns[:-target_dims].tolist()
    
    print(f"目标列: {target_columns}")
    print(f"特征列数: {len(feature_columns)}")
    
    X = df[feature_columns]
    
    # 如果有多个目标变量，只使用第一个进行特征选择
    if target_dims > 1:
        print(f"⚠️  检测到{target_dims}个目标变量，将使用第一个 '{target_columns[0]}' 进行特征选择")
        y = df[target_columns[0]]
    else:
        y = df[target_columns[0]]
    
    # 处理缺失值
    if X.isnull().values.any():  # type: ignore
        print("⚠️  检测到缺失值，使用均值填充...")
        X = X.fillna(X.mean())
    
    # 只保留数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(X.columns):
        print(f"⚠️  过滤掉 {len(X.columns) - len(numeric_cols)} 个非数值列")
        X = X[numeric_cols]
    
    print(f"✓ 特征数: {X.shape[1]}")
    print(f"✓ 样本数: {X.shape[0]}")
    
    # 创建特征选择器
    selector = FeatureSelector(
        task_type=task_type,
        cv_folds=cv_folds,
        min_features=min_features,
        step=step
    )
    
    # 执行特征选择
    result = selector.select_features(X, y)  # type: ignore
    
    # 保存结果 - 生成输出路径
    import os
    print(f"\n处理输出路径...")
    print(f"输入路径: {data_path}")
    
    # 获取输入文件名
    if data_path.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://')):
        filename = os.path.basename(data_path)
        if not filename.endswith('.csv'):
            filename = 'feature_selection_output.csv'
    else:
        filename = os.path.basename(data_path)
    
    # 生成输出文件名
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}{output_suffix}.csv"
    
    # 如果提供了output_dir（用户目录），使用它；否则使用原始路径所在目录
    if output_dir:
        output_path = os.path.join(output_dir, output_filename)
        print(f"✓ 使用用户输出目录: {output_dir}")
    else:
        # 兼容模式：没有output_dir时，保存到原始位置
        if data_path.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://')):
            output_path = os.path.abspath(output_filename)
        else:
            output_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), output_filename)
    
    print(f"输出路径: {output_path}")
    
    data_file, report_file, html_file, details_file = selector.save_results(
        result, X, y, output_path, target_columns, df  # type: ignore
    )
    
    print("\n" + "=" * 80)
    print("✓ 特征选择完成！")
    print("=" * 80)
    
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)  # type: ignore
    
    # 转换所有numpy类型为Python原生类型，以便JSON序列化
    def convert_to_native_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        else:
            return obj
    
    return {
        'selected_features': result['selected_features'],
        'rejected_features': result['rejected_features'],
        'n_selected_features': int(result['n_features']),  # numpy.int64 -> int
        'n_original_features': int(len(X_df.columns)),
        'retention_rate': f"{result['n_features']/len(X_df.columns)*100:.2f}%",
        'best_cv_score': float(result['best_score']),  # numpy.float64 -> float
        'output_file': str(data_file),
        'report_file': str(report_file),
        'html_report_file': str(html_file),
        'details_file': str(details_file),
        'task_type': str(task_type),
        'cv_folds': int(cv_folds)
    }
