"""
材料科学机器学习特征工程MCP工具 - 数据探索模块
包含数据摘要、缺失值分析、分布检查和目标变量分析功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
from urllib.parse import urlparse

# Import safe data loading function
from .utils import _load_data_safe, _validate_target_dims


class DataExplorer:
    """数据探索和预处理工具"""

    def __init__(self):
        self.data = None
        self.task_type = None
        self.target_dims = None
        self.target_columns = None
        self.feature_columns = None

    def load_data(self, data_path: str, task_type: str, target_dims: int):
        """
        加载数据并设置任务参数

        Args:
            data_path: 数据文件路径 (支持CSV, Excel, URL等)
            task_type: 任务类型 ("regression" 或 "classification")
            target_dims: 目标变量维度
        """
        # 使用安全加载函数处理URL和本地文件
        try:
            local_path = _load_data_safe(data_path)

            # 根据文件扩展名判断格式
            parsed_path = urlparse(local_path).path.lower()
            if parsed_path.endswith('.csv'):
                self.data = pd.read_csv(local_path)
            elif parsed_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(local_path)
            else:
                # 未能从扩展名判断，先尝试CSV，再回退到Excel
                try:
                    self.data = pd.read_csv(local_path)
                except Exception:
                    self.data = pd.read_excel(local_path)
        except Exception as e:
            raise ValueError(f"不支持的文件格式或无法解析，建议使用CSV或Excel文件。详细错误: {e}")

        self.task_type = task_type.lower()
        _validate_target_dims(len(self.data.columns), target_dims)
        self.target_dims = target_dims

        # 确定目标列和特征列
        self.target_columns = self.data.columns[-target_dims:].tolist()
        self.feature_columns = self.data.columns[:-target_dims].tolist()

        print(f"数据形状: {self.data.shape}")
        print(f"特征列数: {len(self.feature_columns)}")
        print(f"目标列: {self.target_columns}")

    def data_summary(self) -> Dict[str, Any]:
        """
        生成数据摘要报告
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        summary = {
            "基本信息": {
                "数据形状": self.data.shape,
                "特征数量": len(self.feature_columns),
                "目标变量数量": len(self.target_columns),
                "任务类型": self.task_type,
                "内存使用": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            "数据类型": {
                "数值型": len(self.data.select_dtypes(include=[np.number]).columns),
                "分类型": len(self.data.select_dtypes(include=['object', 'category']).columns),
                "日期型": len(self.data.select_dtypes(include=['datetime']).columns)
            },
            "数值统计": self.data[self.feature_columns].describe().to_dict(),
            "目标变量统计": self.data[self.target_columns].describe().to_dict()
        }

        return summary

    def missing_value_analysis(self) -> Dict[str, Any]:
        """
        缺失值分析
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        missing_info = {}

        # 整体缺失值统计
        total_missing = self.data.isnull().sum()
        missing_percent = (total_missing / len(self.data)) * 100

        missing_df = pd.DataFrame({
            '缺失数量': total_missing,
            '缺失百分比': missing_percent
        }).sort_values('缺失百分比', ascending=False)

        # 只显示有缺失值的列
        missing_df = missing_df[missing_df['缺失数量'] > 0]

        missing_info["缺失值统计"] = missing_df.to_dict()
        missing_info["缺失值模式"] = self._analyze_missing_patterns()
        missing_info["推荐处理策略"] = self._recommend_missing_strategies()

        return missing_info

    def _analyze_missing_patterns(self) -> Dict[str, Any]:
        """分析缺失值模式"""
        patterns = {}

        # 完全缺失的行和列
        completely_missing_rows = (self.data.isnull().sum(axis=1) == len(self.data.columns)).sum()
        completely_missing_cols = (self.data.isnull().sum() == len(self.data)).sum()

        patterns["完全缺失行数"] = completely_missing_rows
        patterns["完全缺失列数"] = completely_missing_cols

        # 缺失值相关性
        missing_corr = self.data.isnull().corr()
        high_corr_pairs = []

        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr_pairs.append({
                        'col1': missing_corr.columns[i],
                        'col2': missing_corr.columns[j],
                        'correlation': corr_val
                    })

        patterns["高相关缺失对"] = high_corr_pairs

        return patterns

    def _recommend_missing_strategies(self) -> Dict[str, str]:
        """推荐缺失值处理策略"""
        strategies = {}

        for col in self.data.columns:
            missing_pct = (self.data[col].isnull().sum() / len(self.data)) * 100

            if missing_pct == 0:
                continue
            elif missing_pct > 70:
                strategies[col] = "建议删除列 - 缺失率过高"
            elif missing_pct > 30:
                if self.data[col].dtype in ['object', 'category']:
                    strategies[col] = "创建'缺失'类别或使用模式填充"
                else:
                    strategies[col] = "KNN插补或模型预测插补"
            elif missing_pct > 10:
                if self.data[col].dtype in ['object', 'category']:
                    strategies[col] = "模式填充"
                else:
                    strategies[col] = "中位数填充或KNN插补"
            else:
                if self.data[col].dtype in ['object', 'category']:
                    strategies[col] = "模式填充"
                else:
                    strategies[col] = "均值或中位数填充"

        return strategies

    def distribution_analysis(self) -> Dict[str, Any]:
        """
        分布检查和分析
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        dist_info = {}

        # 数值型特征分布分析
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[np.number]).columns

        for col in numeric_features:
            col_data = self.data[col].dropna()

            # 基本统计
            dist_info[col] = {
                "基本统计": {
                    "均值": col_data.mean(),
                    "中位数": col_data.median(),
                    "标准差": col_data.std(),
                    "偏度": stats.skew(col_data),
                    "峰度": stats.kurtosis(col_data),
                    "最小值": col_data.min(),
                    "最大值": col_data.max()
                },
                "分布检验": self._test_distributions(col_data),
                "异常值检测": self._detect_outliers(col_data)
            }

        # 分类型特征分析
        categorical_features = self.data[self.feature_columns].select_dtypes(include=['object', 'category']).columns

        for col in categorical_features:
            value_counts = self.data[col].value_counts()
            dist_info[col] = {
                "唯一值数量": self.data[col].nunique(),
                "值分布": value_counts.to_dict(),
                "不平衡度": self._calculate_imbalance(value_counts)
            }

        return dist_info

    def _test_distributions(self, data: pd.Series) -> Dict[str, Any]:
        """测试数据分布"""
        results = {}
        sample_size = len(data)

        if sample_size < 3:
            results["正态性检验"] = {
                "状态": "skipped",
                "原因": "样本数少于3，无法执行 Shapiro-Wilk 检验",
                "样本数": sample_size
            }
            return results

        if data.nunique() <= 1:
            results["正态性检验"] = {
                "状态": "skipped",
                "原因": "样本值恒定，无法执行分布检验",
                "样本数": sample_size
            }
            return results

        # 正态性检验
        sample = data.sample(min(5000, sample_size), random_state=42) if sample_size > 5000 else data
        _, p_shapiro = stats.shapiro(sample)
        results["正态性检验"] = {
            "状态": "completed",
            "Shapiro-Wilk p值": p_shapiro,
            "是否正态": p_shapiro > 0.05
        }

        # 对数正态性检验
        if (data > 0).all():
            log_data = np.log(data)
            if len(log_data) >= 3 and log_data.nunique() > 1:
                log_sample = log_data.sample(min(5000, len(log_data)), random_state=42) if len(log_data) > 5000 else log_data
                _, p_log_shapiro = stats.shapiro(log_sample)
                results["对数正态性检验"] = {
                    "状态": "completed",
                    "p值": p_log_shapiro,
                    "是否对数正态": p_log_shapiro > 0.05
                }
            else:
                results["对数正态性检验"] = {
                    "状态": "skipped",
                    "原因": "对数变换后的样本不足或值恒定，跳过检验"
                }

        return results

    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """检测异常值"""
        if data.empty:
            return {"状态": "skipped", "原因": "无有效样本，跳过异常值检测"}

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        return {
            "IQR方法": {
                "异常值数量": len(outliers),
                "异常值比例": len(outliers) / len(data) * 100,
                "下界": lower_bound,
                "上界": upper_bound
            },
            "Z-score方法": {
                "异常值数量": len(data[np.abs(stats.zscore(data)) > 3]),
                "异常值比例": len(data[np.abs(stats.zscore(data)) > 3]) / len(data) * 100
            }
        }

    def _calculate_imbalance(self, value_counts: pd.Series) -> float:
        """计算分类不平衡度"""
        max_count = value_counts.max()
        min_count = value_counts.min()
        return max_count / min_count if min_count > 0 else float('inf')

    def target_analysis_and_preprocessing(self) -> Dict[str, Any]:
        """
        目标变量分析和预处理建议
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        target_info = {}
        preprocessing_recommendations = {}

        for target_col in self.target_columns:
            target_data = self.data[target_col].dropna()

            if self.task_type == "regression":
                # 回归任务目标变量分析
                target_info[target_col] = {
                    "类型": "回归目标",
                    "统计": {
                        "均值": target_data.mean(),
                        "中位数": target_data.median(),
                        "标准差": target_data.std(),
                        "偏度": stats.skew(target_data),
                        "峰度": stats.kurtosis(target_data),
                        "范围": [target_data.min(), target_data.max()]
                    },
                    "分布检验": self._test_distributions(target_data)
                }

                # 预处理建议
                skewness = stats.skew(target_data)
                if abs(skewness) > 2:
                    if skewness > 0:
                        preprocessing_recommendations[target_col] = "建议对数变换或Box-Cox变换减少右偏"
                    else:
                        preprocessing_recommendations[target_col] = "建议平方变换减少左偏"
                elif abs(skewness) > 1:
                    preprocessing_recommendations[target_col] = "轻度偏斜，可考虑平方根变换"
                else:
                    preprocessing_recommendations[target_col] = "分布较为正常，可直接使用或标准化"

            else:  # classification
                # 分类任务目标变量分析
                value_counts = target_data.value_counts()
                target_info[target_col] = {
                    "类型": "分类目标",
                    "类别数量": len(value_counts),
                    "类别分布": value_counts.to_dict(),
                    "类别比例": (value_counts / len(target_data)).to_dict(),
                    "不平衡度": self._calculate_imbalance(value_counts)
                }

                # 预处理建议
                imbalance_ratio = self._calculate_imbalance(value_counts)
                if imbalance_ratio > 10:
                    preprocessing_recommendations[target_col] = "严重不平衡，建议使用SMOTE、权重调整或欠采样"
                elif imbalance_ratio > 3:
                    preprocessing_recommendations[target_col] = "中度不平衡，建议使用类权重调整"
                else:
                    preprocessing_recommendations[target_col] = "分布较为平衡，可直接使用"

        return {
            "目标变量分析": target_info,
            "预处理建议": preprocessing_recommendations,
            "特征预处理建议": self._recommend_feature_preprocessing()
        }

    def _recommend_feature_preprocessing(self) -> Dict[str, Any]:
        """推荐特征预处理方法"""
        recommendations = {}

        # 数值型特征
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[np.number]).columns

        recommendations["数值型特征"] = {}
        for col in numeric_features:
            col_data = self.data[col].dropna()
            skewness = stats.skew(col_data)

            # 基于偏度和分布推荐预处理
            if abs(skewness) > 2:
                recommendations["数值型特征"][col] = {
                    "标准化": "RobustScaler (对异常值鲁棒)",
                    "变换": "对数变换或Box-Cox变换",
                    "原因": f"偏度 = {skewness:.2f}, 分布偏斜严重"
                }
            elif abs(skewness) > 1:
                recommendations["数值型特征"][col] = {
                    "标准化": "StandardScaler或RobustScaler",
                    "变换": "考虑平方根变换",
                    "原因": f"偏度 = {skewness:.2f}, 轻度偏斜"
                }
            else:
                recommendations["数值型特征"][col] = {
                    "标准化": "StandardScaler或MinMaxScaler",
                    "变换": "无需变换",
                    "原因": f"偏度 = {skewness:.2f}, 分布较正常"
                }

        # 分类型特征
        categorical_features = self.data[self.feature_columns].select_dtypes(include=['object', 'category']).columns

        recommendations["分类型特征"] = {}
        for col in categorical_features:
            unique_values = self.data[col].nunique()

            if unique_values > 50:
                recommendations["分类型特征"][col] = "高基数分类变量，建议目标编码或哈希编码"
            elif unique_values > 10:
                recommendations["分类型特征"][col] = "中等基数，建议独热编码或标签编码"
            else:
                recommendations["分类型特征"][col] = "低基数，建议独热编码"

        return recommendations

    def generate_processed_data(self, output_path: str) -> str:
        """
        生成预处理后的数据文件
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        processed_data = self.data.copy()

        # 应用基本预处理
        processing_log = []

        # 1. 处理缺失值
        missing_strategies = self._recommend_missing_strategies()
        for col, strategy in missing_strategies.items():
            if "删除" in strategy:
                processed_data = processed_data.drop(columns=[col])
                processing_log.append(f"删除列 {col}: {strategy}")
            elif "均值" in strategy:
                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
                processing_log.append(f"均值填充 {col}")
            elif "中位数" in strategy:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
                processing_log.append(f"中位数填充 {col}")
            elif "模式" in strategy:
                processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
                processing_log.append(f"模式填充 {col}")

        # 2. 标准化数值特征
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in self.target_columns]

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
            processing_log.append(f"标准化数值特征: {list(numeric_cols)}")

        # 保存处理后的数据
        processed_data.to_csv(output_path, index=False)

        # 保存处理日志
        log_path = output_path.replace('.csv', '_processing_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("数据预处理日志\n")
            f.write("=" * 50 + "\n")
            for log in processing_log:
                f.write(f"- {log}\n")
            f.write(f"\n处理后数据形状: {processed_data.shape}\n")
            f.write(f"保存路径: {output_path}\n")

        return f"预处理完成！数据保存至: {output_path}, 日志保存至: {log_path}"
