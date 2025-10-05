"""
材料科学机器学习特征工程MCP工具
使用FastMCP框架实现的数据探索工具
"""

import os
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from fastmcp import FastMCP,Context
from .data_explorer import DataExplorer
from .feature_generator import MaterialsFeatureGenerator
from .report_generator import OpenFEReportGenerator
from .feature_selector import select_best_features
from urllib.parse import urlparse
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif
from sklearn.preprocessing import LabelEncoder

# 创建FastMCP应用
mcp = FastMCP("Materials Feature Engineering Tool")


def _create_user_output_dir(user_id: Optional[str] = None) -> Tuple[str, str]:
    """
    创建基于用户ID和UUID的输出目录
    
    Args:
        user_id: 用户ID，如果为None则使用'anonymous'
        
    Returns:
        (output_dir, run_uuid): 输出目录路径和运行UUID
    """
    # 如果没有user_id，使用anonymous
    if not user_id or user_id.strip() == "":
        user_id = "anonymous"
    
    # 生成唯一的运行ID
    run_uuid = str(uuid.uuid4())
    
    # 创建输出目录: data/user_id/uuid/
    base_dir = Path("data")
    user_dir = base_dir / user_id
    output_dir = user_dir / run_uuid
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory created: {output_dir}")
    print(f"   User ID: {user_id}")
    print(f"   Run UUID: {run_uuid}")
    
    return str(output_dir), run_uuid


def _get_user_id_from_context(ctx: Optional[Context]) -> Optional[str]:
    """
    从Context中提取user_id
    
    Args:
        ctx: FastMCP的Context对象
        
    Returns:
        user_id或None
    """
    if ctx is not None:
        try:
            user_id = ctx.request_context.request.headers.get("user_id", None)  # type: ignore
            return user_id
        except Exception as e:
            print(f"⚠️  Warning: Could not extract user_id from context: {e}")
            return None
    return None


def _is_url(path: str) -> bool:
    """判断给定路径是否为URL。"""
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https", "ftp", "s3", "gs", "file")
    except Exception:
        return False


def _validate_data_path(data_path: str) -> None:
    """当为本地路径时进行存在性校验，URL则跳过本地校验。"""
    if _is_url(data_path):
        return
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")


def _json_safe(obj):
    """递归将numpy/pandas类型转换为可JSON序列化的原生Python类型。"""
    # 基本类型直接返回
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy标量
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        try:
            return obj.item()
        except Exception:
            return float(obj) if isinstance(obj, np.floating) else int(obj)

    # numpy数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas对象
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)

    # 容器类型
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    # 其他类型尝试直接JSON化，否则转字符串
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _auto_output_path(data_path: str, suffix: str = "enhanced") -> str:
    """为输出文件生成系统路径。

    - 对于本地路径：在当前工作目录的 outputs/ 下生成文件
    - 对于 URL：同样生成到当前工作目录的 outputs/ 下
    文件名规则：<basename>_<suffix>_<YYYYMMDD_HHMMSS>.csv
    """
    parsed_path = urlparse(data_path).path
    base = os.path.splitext(os.path.basename(parsed_path))[0] or "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    return os.path.join(outputs_dir, f"{base}_{suffix}_{ts}.csv")

@mcp.tool()
def explore_materials_data(
    data_path: str,
    task_type: str,
    target_dims: int,
    output_path: str = None # type: ignore
) -> Dict[str, Any]:
    """
    材料科学数据探索工具

    Args:
        data_path: 数据文件路径 (CSV或Excel格式)
        task_type: 任务类型 ("regression" 或 "classification")
        target_dims: 目标变量维度数量
        output_path: 可选的输出路径，用于保存预处理后的数据

    Returns:
        包含完整数据分析报告的字典
    """

    # 验证输入参数
    _validate_data_path(data_path)

    if task_type.lower() not in ["regression", "classification"]:
        raise ValueError("task_type 必须是 'regression' 或 'classification'")

    if target_dims <= 0:
        raise ValueError("target_dims 必须是正整数")

    # 创建数据探索器
    explorer = DataExplorer()

    try:
        # 加载数据
        explorer.load_data(data_path, task_type, target_dims)

        # 执行所有分析
        analysis_results = {
            "数据摘要": explorer.data_summary(),
            "缺失值分析": explorer.missing_value_analysis(),
            "分布分析": explorer.distribution_analysis(),
            "目标变量分析和预处理建议": explorer.target_analysis_and_preprocessing()
        }

        # 生成预处理数据（如果指定了输出路径）
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            processing_result = explorer.generate_processed_data(output_path)
            analysis_results["数据预处理"] = {
                "状态": "完成",
                "消息": processing_result,
                "输出文件": output_path
            }
        else:
            analysis_results["数据预处理"] = {
                "状态": "未执行",
                "消息": "未提供输出路径，跳过数据预处理"
            }

        return _json_safe(analysis_results) # type: ignore

    except Exception as e:
        return {
            "错误": str(e),
            "建议": "请检查数据文件格式和参数设置"
        }


@mcp.tool()
def quick_data_summary(data_path: str) -> Dict[str, Any]:
    """
    快速数据摘要工具 - 仅生成基本的数据概览

    Args:
        data_path: 数据文件路径

    Returns:
        基本数据摘要信息
    """
    _validate_data_path(data_path)

    explorer = DataExplorer()

    try:
        # 临时加载数据用于快速摘要
        explorer.load_data(data_path, "regression", 1)
        return _json_safe(explorer.data_summary()) # type: ignore
    except Exception as e:
        return {"错误": str(e)}



@mcp.tool()
def auto_generate_features_with_llm(
    data_path: str,
    sample_rows: int = 5,
    target_dims: int = 1,
    ctx: Context = None # type: ignore
) -> Dict[str, Any]:
    """
    使用LLM自动分析并生成材料特征（仅识别化学组成列，生成组成相关特征并去除源组成列）。

    Args:
        data_path: 数据文件路径
        sample_rows: 分析的样本行数
        target_dims: 目标变量维度（默认最后n列）

    Returns:
        完整的分析和特征生成结果
    """
    # 验证输入参数
    _validate_data_path(data_path)
    
    # 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    try:
        generator = MaterialsFeatureGenerator()
        generator.load_data(data_path)

        # 1. 分析列
        analysis_result = generator.analyze_columns_with_llm(sample_rows=sample_rows)

        # 2. 自动选择组成列进行特征生成（不依赖置信度，只要识别为化学组成就选择）
        selected_columns = {}
        for col, info in generator.identified_columns.items():
            if info.get("category") == "chemical_composition":
                selected_columns[col] = ["element_property", "stoichiometry", "valence_orbital"]

        # 3. 生成特征
        if selected_columns:
            # 先在内存中生成，再重排列顺序，最后保存
            enhanced_data = generator.create_enhanced_dataset(
                output_path=None,
                selected_columns=selected_columns
            )

            # 目标列移动到最后
            original_cols = list(generator.data.columns) # type: ignore
            target_cols: list = []
            if isinstance(target_dims, int) and target_dims > 0 and target_dims <= len(original_cols):
                target_cols = original_cols[-target_dims:]

            if target_cols:
                non_target_cols = [c for c in enhanced_data.columns if c not in target_cols]
                enhanced_data = enhanced_data[non_target_cols + target_cols]

            # 生成输出路径到用户目录
            input_filename = os.path.basename(data_path)
            base_name = os.path.splitext(input_filename)[0]
            output_filename = f"{base_name}_enhanced.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            enhanced_data.to_csv(output_path, index=False)
            print(f"\n✓ Enhanced data saved to: {output_path}")
            
            # 生成特征报告（也保存到用户目录）
            report_path = output_path.replace('.csv', '_feature_report.txt')
            try:
                generator._generate_feature_report(enhanced_data, report_path) # type: ignore
                print(f"✓ Feature report saved to: {report_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate feature report: {e}")

            # 统计新增特征列（与原始列集合对比）
            original_cols_set = set(generator.data.columns) # type: ignore
            new_feature_cols = [c for c in enhanced_data.columns if c not in original_cols_set]
            from .config import get_static_url,get_download_url
            output_path = get_download_url(output_path)
            report_path = get_static_url(report_path)
            result = {
                "status": "success",
                "input_file": data_path,
                "output_file": output_path,
                "feature_generated_report_html_path": report_path,
                "LLM_analysis": analysis_result,
                "selected_columns": selected_columns,
                "original_data_shape": generator.data.shape, # type: ignore
                "enhanced_data_shape": enhanced_data.shape,
                "new_features_count": len(new_feature_cols),
                "new_features_columns": new_feature_cols
            }
            
            return _json_safe(result)  # type: ignore
        else:
            result = {
                "status": "no_feature_generated",
                "reason": "未识别到合适的化学组成列",
                "LLM_analysis": analysis_result,
                "suggestion": "请检查数据或手动指定组成列",
            }
            return _json_safe(result) # type: ignore

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def _select_features_before_openfe(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "regression",
    max_features: int = 50,
    variance_threshold: float = 0.01
) -> List[str]:
    """
    在使用OpenFE之前进行初步特征筛选，减少计算复杂度
    
    Args:
        X: 特征数据框
        y: 目标变量
        task_type: 任务类型（'regression' 或 'classification'）
        max_features: 最多保留的特征数量
        variance_threshold: 方差阈值
        
    Returns:
        选中的特征列名列表
    """
    selected_features = []
    
    # 确保目标变量是数值类型
    if y.dtype == 'object' or y.dtype.name == 'object': # type: ignore
        print(f"目标变量类型为 {y.dtype}，尝试转换为数值类型")
        
        # 处理科学计数法字符串（如 '7.1×10-5'）
        def convert_sci_notation(val):
            """转换科学计数法字符串为数值"""
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            # 将科学计数法格式转换：7.1×10-5 -> 7.1e-5
            val_str = str(val)
            # 替换 ×10、x10、X10 为 e
            val_str = val_str.replace('×10', 'e').replace('x10', 'e').replace('X10', 'e')
            # 处理没有10的情况（如直接是 ×-5）
            val_str = val_str.replace('×', 'e').replace('x', 'e').replace('X', 'e')
            try:
                return float(val_str)
            except ValueError:
                return np.nan
        
        y = y.apply(convert_sci_notation) # type: ignore
        y = pd.Series(pd.to_numeric(y, errors='coerce'), index=y.index, name=y.name, dtype=np.float64) # type: ignore
        
        if y.isna().any(): # type: ignore
            nan_count = y.isna().sum() # type: ignore
            print(f"警告: 目标变量转换后有 {nan_count} 个NaN，使用中位数填充")
            median_val = float(y.median()) # type: ignore
            if not np.isnan(median_val): # type: ignore
                y = y.fillna(median_val) # type: ignore
            else:
                y = y.fillna(0) # type: ignore
    elif y.dtype not in [np.float64, np.float32, np.int64, np.int32]: # type: ignore
        # 确保是数值类型
        y = y.astype(np.float64) # type: ignore
    
    # 步骤1: 移除低方差特征
    print(f"原始特征数: {len(X.columns)}")
    
    # 只对数值型特征进行方差筛选
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = [col for col in X.columns if col not in numeric_features]
    
    if len(numeric_features) > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        X_numeric = X[numeric_features].fillna(X[numeric_features].mean())
        
        try:
            selector.fit(X_numeric)
            high_variance_features = [numeric_features[i] for i in range(len(numeric_features)) 
                                     if selector.variances_[i] > variance_threshold]
            print(f"经过方差筛选后的特征数: {len(high_variance_features)}")
        except Exception as e:
            print(f"方差筛选失败，保留所有数值特征: {e}")
            high_variance_features = numeric_features
    else:
        high_variance_features = []
    
    # 步骤2: 基于统计显著性筛选特征
    selected_numeric_features = high_variance_features
    if len(high_variance_features) > max_features:
        print(f"特征数超过{max_features}，进行进一步筛选...")
        
        # 确保数据类型一致性，并正确填充缺失值
        X_filtered = pd.DataFrame(X[high_variance_features].copy())
        
        # 逐列填充缺失值，确保数据类型安全
        for col in X_filtered.columns:
            col_data = X_filtered[col]
            if col_data.isna().any(): # type: ignore
                # 尝试使用均值填充，失败则使用中位数
                try:
                    fill_value = float(col_data.mean()) # type: ignore
                    if pd.isna(fill_value):
                        fill_value = float(col_data.median()) # type: ignore
                    if pd.isna(fill_value):
                        fill_value = 0.0
                    X_filtered.loc[:, col] = col_data.fillna(fill_value) # type: ignore
                except Exception:
                    X_filtered.loc[:, col] = col_data.fillna(0) # type: ignore
        
        # 确保所有列都是数值类型
        for col in X_filtered.columns:
            if X_filtered[col].dtype == 'object':
                try:
                    numeric_col = pd.to_numeric(X_filtered[col], errors='coerce') # type: ignore
                    if isinstance(numeric_col, pd.Series):
                        X_filtered.loc[:, col] = numeric_col.fillna(0)
                    else:
                        X_filtered.loc[:, col] = pd.Series(numeric_col).fillna(0) # type: ignore
                except Exception:
                    X_filtered.loc[:, col] = 0
        
        # 确保数据类型为 float64
        try:
            X_filtered = X_filtered.astype(np.float64)
        except Exception:
            # 如果批量转换失败，逐列转换
            for col in X_filtered.columns:
                try:
                    X_filtered.loc[:, col] = X_filtered[col].astype(np.float64)
                except Exception:
                    X_filtered.loc[:, col] = 0.0
        
        # 根据任务类型选择评分函数
        if task_type.lower() == "classification":
            score_func = f_classif
        else:
            score_func = f_regression
        
        try:
            selector = SelectKBest(score_func=score_func, k=min(max_features, len(high_variance_features)))
            selector.fit(X_filtered, y)
            selected_indices = selector.get_support(indices=True)
            if selected_indices is not None:
                selected_numeric_features = [high_variance_features[i] for i in selected_indices]
            else:
                selected_numeric_features = high_variance_features[:max_features]
            print(f"经过统计显著性筛选后的数值特征数: {len(selected_numeric_features)}")
        except Exception as e:
            print(f"统计筛选失败，保留高方差特征: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            selected_numeric_features = high_variance_features[:max_features]
    
    # 合并数值特征和非数值特征
    selected_features = selected_numeric_features + non_numeric_features
    print(f"最终筛选后的特征数: {len(selected_features)}")
    
    return selected_features


@mcp.tool()
def auto_feature_engineering_with_openfe(
    ctx: Context,
    data_path: str,
    target_dims: int = 1,
    task_type: str = "regression",
    n_features_before_openfe: int = 50,
    n_new_features: int = 10,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    use openfe to automatically generate features for high-dimensional data.
    
    workflow:
    1. load data and identify target variables (default last n columns)
    2. preliminary feature screening: reduce input feature dimension to manageable quantity
    3. use openfe to generate new features
    4. save enhanced data set
    
    Args:
        data_path: data file path (CSV or Excel format,url is also supported)
        target_dims: target variable dimension (default 1, last column is target variable)
        task_type: task type ('regression' or 'classification')
        n_features_before_openfe: number of features to retain before OpenFE processing (default 50)
        n_new_features: number of new features to generate with OpenFE (default 10)
        output_path: output file path (optional, if not provided, will be automatically generated)
        
    Returns:
        feature engineering result report
    """
    _validate_data_path(data_path)
    
    # 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    
    try:
        # 导入OpenFE（延迟导入，避免启动时的依赖问题）
        try:
            from openfe import OpenFE, transform # type: ignore
        except ImportError:
            return {
                "status": "failed",
                "error": "openfe library not installed, please run: pip install openfe"
            }
        
        # 1. 加载数据
        print(f"加载数据: {data_path}")
        
        # 判断是否为URL
        is_url = data_path.startswith(('http://', 'https://'))
        
        try:
            if data_path.endswith('.csv'):
                if is_url:
                    # 为URL添加超时和错误处理
                    import urllib.request
                    import urllib.error
                    import io
                    print("📥 正在从URL下载数据...")
                    req = urllib.request.Request(data_path, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read()
                    df = pd.read_csv(io.BytesIO(content))
                    print(f"✓ 成功从URL加载数据，形状: {df.shape}")
                else:
                    df = pd.read_csv(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                if is_url:
                    import urllib.request
                    import urllib.error
                    import io
                    print("📥 正在从URL下载数据...")
                    req = urllib.request.Request(data_path, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read()
                    df = pd.read_excel(io.BytesIO(content))
                    print(f"✓ 成功从URL加载数据，形状: {df.shape}")
                else:
                    df = pd.read_excel(data_path)
            else:
                raise ValueError("only support CSV or Excel format")
        except urllib.error.URLError as e:  # type: ignore
            raise ValueError(f"URL加载失败: {str(e)}. 请检查URL是否可访问")
        except Exception as e:
            if "timed out" in str(e).lower():
                raise ValueError(f"URL加载超时（30秒），请检查网络连接或尝试本地文件")
            raise
        
        print(f"original data shape: {df.shape}")
        
        # 2. 分离特征和目标
        if target_dims <= 0 or target_dims > len(df.columns):
            raise ValueError(f"target_dims必须在1到{len(df.columns)}之间")
        
        target_columns = df.columns[-target_dims:].tolist()
        feature_columns = df.columns[:-target_dims].tolist()
        
        print(f"特征列数: {len(feature_columns)}, 目标列: {target_columns}")
        
        X = pd.DataFrame(df[feature_columns].copy())
        y = df[target_columns[0]] if target_dims == 1 else df[target_columns]
        
        # 对于分类任务，编码目标变量
        if task_type.lower() == "classification" and target_dims == 1:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_series = pd.Series(y_encoded, index=y.index, name=y.name, dtype=np.float64)
        else:
            # 确保目标变量是数值类型
            if isinstance(y, pd.Series):
                y_temp = y.copy()
            else:
                y_temp = pd.Series(y.iloc[:, 0])
            
            # 处理科学计数法字符串（如 '7.1×10-5' 或 '7.1x10-5'）
            if y_temp.dtype == 'object':
                def convert_sci_notation(val):
                    """转换科学计数法字符串为数值"""
                    if pd.isna(val):
                        return np.nan
                    if isinstance(val, (int, float)):
                        return float(val)
                    # 将科学计数法格式转换：7.1×10-5 -> 7.1e-5
                    val_str = str(val)
                    # 替换 ×10、x10、X10 为 e
                    val_str = val_str.replace('×10', 'e').replace('x10', 'e').replace('X10', 'e')
                    # 处理没有10的情况（如直接是 ×-5）
                    val_str = val_str.replace('×', 'e').replace('x', 'e').replace('X', 'e')
                    try:
                        return float(val_str)
                    except ValueError:
                        return np.nan
                
                print(f"检测到科学计数法字符串格式，正在转换...")
                y_temp = y_temp.apply(convert_sci_notation)
            
            # 转换为数值类型
            y_series = pd.Series(pd.to_numeric(y_temp, errors='coerce'), index=y_temp.index, name=y_temp.name, dtype=np.float64)
            
            # 检查并填充转换失败的NaN值
            if y_series.isna().any():
                nan_count = y_series.isna().sum()
                print(f"警告: 目标变量中有 {nan_count} 个无法转换为数值的值，将使用中位数填充")
                median_val = y_series.median()
                if pd.isna(median_val): # type: ignore
                    print(f"警告: 无法计算中位数，使用0填充")
                    y_series = y_series.fillna(0)
                else:
                    y_series = y_series.fillna(median_val)
        
        # 3. 初步特征筛选
        print(f"\n开始初步特征筛选（目标: {n_features_before_openfe}个特征）...")
        selected_feature_names = _select_features_before_openfe(
            X, 
            y_series,
            task_type=task_type,
            max_features=n_features_before_openfe,
            variance_threshold=0.01
        )
        
        X_selected = pd.DataFrame(X[selected_feature_names].copy())
        
        # 清理列名以避免 OpenFE 冲突
        # OpenFE 对某些列名敏感（如 norm, index 等）
        original_col_names = list(X_selected.columns)
        safe_col_names = []
        col_name_mapping = {}
        
        for idx, col in enumerate(original_col_names):
            # 移除特殊字符，替换为下划线
            safe_name = str(col).replace(' ', '_').replace('-', '_').replace('.', '_')
            safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe_name)
            
            # 移除连续的下划线
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            safe_name = safe_name.strip('_')
            
            # 如果列名为空或全是下划线，使用默认名称
            if not safe_name or not any(c.isalnum() for c in safe_name):
                safe_name = f"feature_{idx}"
            
            # 避免 OpenFE 保留字或可能冲突的名称
            reserved_names = ['norm', 'index', 'level', 'values', 'count', 'sum', 'mean', 'min', 'max', 'std', 'var']
            if safe_name.lower() in reserved_names:
                safe_name = f"feat_{safe_name}"
            
            # 确保不以数字开头
            if safe_name and safe_name[0].isdigit():
                safe_name = f"feat_{safe_name}"
            
            # 确保唯一性
            original_safe_name = safe_name
            counter = 1
            while safe_name in safe_col_names:
                safe_name = f"{original_safe_name}_{counter}"
                counter += 1
            
            safe_col_names.append(safe_name)
            col_name_mapping[safe_name] = col
        
        X_selected.columns = safe_col_names
        print(f"列名已标准化: {len(original_col_names)} 列")
        
        # 填充缺失值
        for col in X_selected.columns:
            if X_selected[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X_selected.loc[:, col] = X_selected[col].fillna(X_selected[col].mean())
            else:
                mode_vals = X_selected[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 'missing'
                X_selected.loc[:, col] = X_selected[col].fillna(fill_val)
        
        print(f"筛选后的特征数据形状: {X_selected.shape}")
        
        # 4. 使用OpenFE生成新特征
        print(f"\n使用OpenFE生成 {n_new_features} 个新特征...")
        
        # 清理可能存在的临时文件
        temp_files = ['./openfe_tmp_data.feather', 'openfe_tmp_data.feather']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"已清理旧的临时文件: {temp_file}")
                except Exception:
                    pass
        
        ofe = OpenFE()
        
        # 准备训练数据
        train_x = X_selected
        train_y = y_series
        
        # OpenFE特征生成
        try:
            # 确保 train_y 是 DataFrame 格式（OpenFE 要求）
            train_y_df = pd.DataFrame(train_y) if isinstance(train_y, pd.Series) else train_y
            
            features = ofe.fit(
                data=train_x,
                label=train_y_df, # type: ignore
                n_jobs=1,  # 单进程避免序列化问题
                task=task_type,
                n_data_blocks=1,
                stage2_params={'verbose': -1},
                verbose=True
            )
            
            # 选择top n个新特征
            new_features_list = features[:min(n_new_features, len(features))]
            
            # 辅助函数：递归解析OpenFE特征对象树
            def parse_openfe_feature_tree(feat, depth=0):
                """递归解析OpenFE特征树，生成可读的表达式"""
                try:
                    type_name = type(feat).__name__
                    
                    # FNode - 原始特征叶子节点
                    if type_name == 'FNode':
                        if hasattr(feat, 'name') and feat.name:
                            return str(feat.name)
                        elif hasattr(feat, 'data') and feat.data is not None:
                            return str(feat.data)
                        else:
                            return "未知特征"
                    
                    # Node - 操作符节点
                    elif type_name == 'Node':
                        if not hasattr(feat, 'name'):
                            return str(feat)
                        
                        op_name = feat.name
                        children = feat.children if hasattr(feat, 'children') and feat.children else []
                        
                        # 调试输出
                        if depth == 0:  # 只在顶层打印
                            print(f"  [DEBUG] 解析特征: op={op_name}, children_count={len(children)}, has_children={hasattr(feat, 'children')}")
                        
                        if not children or len(children) == 0:
                            print(f"  [WARNING] 特征 {op_name} 没有子节点！")
                            return str(op_name)
                        
                        # 递归解析子节点
                        child_exprs = [parse_openfe_feature_tree(child, depth+1) for child in children]
                        
                        # 根据操作符类型格式化
                        if op_name in ['+', '-', '*', '/', '>', '<', '>=', '<=', '==', '!=']:
                            # 二元操作符
                            if len(child_exprs) == 2:
                                return f"({child_exprs[0]} {op_name} {child_exprs[1]})"
                            else:
                                return f"{op_name}({', '.join(child_exprs)})"
                        elif 'GroupBy' in op_name:
                            # GroupBy操作
                            return f"{op_name}({', '.join(child_exprs)})"
                        elif op_name in ['abs', 'sqrt', 'log', 'exp', 'sin', 'cos', 'residual', 'Round', 'round']:
                            # 单参数函数
                            if len(child_exprs) == 1:
                                return f"{op_name}({child_exprs[0]})"
                            else:
                                return f"{op_name}({', '.join(child_exprs)})"
                        elif op_name in ['max', 'min', 'mean', 'median', 'std', 'sum', 'var']:
                            # 聚合函数
                            return f"{op_name}({', '.join(child_exprs)})"
                        else:
                            # 其他操作
                            return f"{op_name}({', '.join(child_exprs)})"
                    else:
                        # 其他类型
                        return str(feat)
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] 解析特征时出错: {e}")
                    print(f"  [ERROR] Traceback: {traceback.format_exc()}")
                    return f"<解析错误: {str(e)}>"
            
            # 提取特征构造信息
            feature_descriptions = {}
            for idx, feat in enumerate(new_features_list):
                try:
                    feat_name = f"autoFE_f_{idx}"
                    # 使用树解析函数生成可读的特征描述
                    feat_desc = parse_openfe_feature_tree(feat)
                    
                    if feat_desc and feat_desc.strip() and not feat_desc.startswith('<'):
                        feature_descriptions[feat_name] = feat_desc.strip()
                    else:
                        feature_descriptions[feat_name] = f"组合特征 {idx}"
                        
                except Exception as e:
                    print(f"提取特征 {idx} 描述时出错: {e}")
                    feature_descriptions[feat_name] = f"组合特征 {idx}"
            
            print(f"已提取 {len(feature_descriptions)} 个特征的构造信息")
            
            # 调试：显示前3个特征的描述
            if feature_descriptions:
                print("前3个特征描述示例:")
                for i, (name, desc) in enumerate(list(feature_descriptions.items())[:3]):
                    print(f"  {name}: {desc[:100]}..." if len(desc) > 100 else f"  {name}: {desc}")
            
            # 生成新特征数据
            train_x_new_result, _ = transform(train_x, train_y, new_features_list, n_jobs=1)
            train_x_new = pd.DataFrame(train_x_new_result)
            
            print(f"OpenFE生成了 {len(train_x_new.columns) - len(train_x.columns)} 个新特征")
            
        except Exception as e:
            print(f"OpenFE特征生成出错: {e}")
            print("使用原始筛选后的特征继续...")
            train_x_new = train_x
            feature_descriptions = {}
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
        
        # 5. 合并数据并保存
        # 重新添加目标列
        result_df = pd.DataFrame(train_x_new.copy())
        for target_col in target_columns:
            result_df[target_col] = df[target_col].values
        
        # 生成输出路径到用户目录
        if output_path is None:
            # 从原始文件名生成输出文件名
            input_filename = os.path.basename(data_path)
            base_name = os.path.splitext(input_filename)[0]
            output_filename = f"{base_name}_openfe.csv"
            output_path = os.path.join(output_dir, output_filename)
        else:
            # 如果提供了output_path，将其放到用户目录下
            output_filename = os.path.basename(output_path)
            output_path = os.path.join(output_dir, output_filename)
        
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ Enhanced data saved to: {output_path}")
        
        # 保存特征描述报告（文本格式）
        if feature_descriptions:
            txt_report_path = output_path.replace('.csv', '_feature_descriptions.txt')
            try:
                with open(txt_report_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("OpenFE 自动特征工程报告\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"数据文件: {data_path}\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"任务类型: {task_type}\n")
                    f.write(f"生成特征数: {len(feature_descriptions)}\n")
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("特征构造详情\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for feat_name, feat_desc in feature_descriptions.items():
                        f.write(f"\n【{feat_name}】\n")
                        f.write(f"构造方式: {feat_desc}\n")
                        f.write("-" * 80 + "\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("说明:\n")
                    f.write("- 特征名称格式: autoFE_f_N (N为特征序号)\n")
                    f.write("- 构造方式显示了该特征由哪些原始特征通过何种运算生成\n")
                    f.write("- 常见运算: +（加）, -（减）, *（乘）, /（除）, ^（幂）等\n")
                    f.write("=" * 80 + "\n")
                
                print(f"文本报告已保存到: {txt_report_path}")
            except Exception as e:
                print(f"保存文本报告失败: {e}")
            
            # 生成HTML报告
            html_report_path = output_path.replace('.csv', '_report.html')
            try:
                report_generator = OpenFEReportGenerator()
                
                # 准备数据信息
                data_info = {
                    'data_file': os.path.basename(data_path),
                    'task_type': task_type,
                    'original_shape': df.shape,
                    'final_shape': result_df.shape,
                    'target_columns': target_columns
                }
                
                # 准备特征信息
                feature_info = {
                    'original_features': len(feature_columns),
                    'selected_features': len(selected_feature_names),
                    'selection_rate': f"{len(selected_feature_names)/len(feature_columns)*100:.1f}%",
                    'input_features': train_x.shape[1],
                    'output_features': train_x_new.shape[1],
                    'new_features': train_x_new.shape[1] - train_x.shape[1],
                    'target_new_features': n_new_features
                }
                
                # 生成HTML报告
                report_generator.generate_report(
                    html_report_path,
                    data_info,
                    feature_info,
                    feature_descriptions
                )
                
                print(f"📊 HTML报告已保存到: {html_report_path}")
            except Exception as e:
                print(f"生成HTML报告失败: {e}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
        from .config import get_download_url,get_static_url
        # 6. 生成特征报告
        report = {
            "status": "success",
            "input_file": data_path,
            "output_file": get_download_url(output_path),
            "original_data_shape": df.shape,
            "target_variables": target_columns,
            "task_type": task_type,
            "preliminary_feature_screening": {
                "original_features": len(feature_columns),
                "selected_features": len(selected_feature_names),
                "selection_rate": f"{len(selected_feature_names)/len(feature_columns)*100:.1f}%"
            },
            "OpenFE_feature_generation": {
                "input_features": train_x.shape[1],
                "output_features": train_x_new.shape[1],
                "new_features": train_x_new.shape[1] - train_x.shape[1],
                "target_new_features": n_new_features
            },
            "final_data_shape": result_df.shape,
            "feature_list": result_df.columns.tolist()
        }
        
        # 添加特征描述信息和报告路径
        if feature_descriptions:
            txt_path = output_path.replace('.csv', '_feature_descriptions.txt')
            html_path = output_path.replace('.csv', '_report.html')
            report["feature_construction_description"] = {
                "description": "the construction way of each new feature",
                "text_report": txt_path,
                "openfe_report_html_path": get_static_url(html_path),
                "feature_details": feature_descriptions
            }


        return _json_safe(report) # type: ignore
        
    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "error": str(e),
            "detailed_error": traceback.format_exc()
        }


@mcp.tool()
def select_optimal_features(
    ctx: Context,
    data_path: str,
    target_dims: int = 1,
    task_type: str = "regression",
    cv_folds: int = 5,
    min_features: int = 1,
    step: int = 1
) -> Dict[str, Any]:
    """
    use recursive feature elimination (RFE) combined with cross-validation to select the best feature combination.
    
    this tool uses the RFE-CV method to automatically select the feature subset that contributes most to model performance, can:
    - automatically determine the optimal number of features
    - provide the importance ranking of each feature
    - generate detailed visual reports
    - ensure the stability of the results through cross-validation
    
    Args:
        data_path: input data file path (CSV format)
        target_dims: target variable dimension (default last n columns), default 1
        task_type: task type, optional 'regression' (regression) or 'classification' (classification), default 'regression'
        cv_folds: cross-validation folds, default 5
        min_features: minimum number of features to retain, default 1
        step: number of features to remove in each iteration, default 1 (it is recommended to use a larger step for large datasets to speed up the process)
    
    Returns:
        a dictionary containing the following information:
        - selected_features: selected features list
        - rejected_features: rejected features list
        - n_selected_features: number of selected features
        - n_original_features: number of original features
        - retention_rate: feature retention rate
        - best_cv_score: best cross-validation score
        - output_file: output data file path
        - report_file: visualization report file path
        - details_file: detailed text report path
    
    Examples:
        # regression task (target variable in the last column)
        result = select_optimal_features(
            data_path="/path/to/data.csv",
            target_dims=1,
            task_type="regression",
            cv_folds=5
        )
        
        #  classification task (target variable in the last column)
        result = select_optimal_features(
            data_path="/path/to/data.csv",
            target_dims=1,
            task_type="classification",
            cv_folds=10,
            min_features=5,
            step=2
        )
        
        # multiple target variables (last 2 columns are targets, use the first one for selection)
        result = select_optimal_features(
            data_path="/path/to/data.csv",
            target_dims=2,
            task_type="regression"
        )
    """
    try:
        print(f"\n{'='*80}")
        print("特征选择工具 (RFE-CV)")
        print(f"{'='*80}\n")
        
        # 获取用户ID并创建输出目录
        user_id = _get_user_id_from_context(ctx)
        output_dir, run_uuid = _create_user_output_dir(user_id)
        
        # 调用特征选择函数
        result = select_best_features(
            data_path=data_path,
            target_dims=target_dims,
            task_type=task_type,
            cv_folds=cv_folds,
            min_features=min_features,
            step=step,
            output_dir=output_dir
        )
        
        print(f"\n{'='*80}")
        print("✓ 特征选择完成！")
        print(f"{'='*80}")
        print(f"\n输出文件:")
        print(f"  - Data: {result['output_file']}")
        print(f"  - PNG Report: {result['report_file']}")
        print(f"  - HTML Report: {result['html_report_file']}")
        print(f"  - Details: {result['details_file']}")
        print(f"\n特征统计:")
        print(f"  - Original Features: {result['n_original_features']}")
        print(f"  - Selected Features: {result['n_selected_features']}")
        print(f"  - Retention Rate: {result['retention_rate']}")
        print(f"  - Best CV Score: {result['best_cv_score']:.4f}")
        
        from .config import get_download_url,get_static_url
        # 确保返回值是JSON可序列化的
        return {
            'selected_features': [str(f) for f in result['selected_features']],
            'rejected_features': [str(f) for f in result['rejected_features']],
            'n_selected_features': int(result['n_selected_features']),
            'n_original_features': int(result['n_original_features']),
            'retention_rate': str(result['retention_rate']),
            'best_cv_score': float(result['best_cv_score']),
            'output_file': get_download_url(result['output_file']),
            'report_file': get_download_url(result['report_file']),
            'report_html_path': get_static_url(result['html_report_file']),
            'details_file': str(result['details_file']),
            'task_type': str(result['task_type']),
            'cv_folds': int(result['cv_folds']),
            'user_id': user_id if user_id else "anonymous",
            'run_uuid': run_uuid,
            'output_dir': output_dir
        }
        
    except Exception as e:
        import traceback
        error_msg = {
            "status": "failed",
            "error": str(e),
            "detailed_error": traceback.format_exc()
        }
        print(f"\n✗ error: {str(e)}")
        print(f"\ndetailed information:\n{traceback.format_exc()}")
        return error_msg


def main():
    """运行MCP服务器"""
    mcp.run()


if __name__ == "__main__":
    main()