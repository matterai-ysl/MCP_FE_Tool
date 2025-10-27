"""
Materials Science Machine Learning Feature Engineering MCP Tool
Data exploration tool implemented using the FastMCP framework
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

# Import utility functions
from .utils import _validate_data_path, _load_data_safe

# Create FastMCP application
mcp = FastMCP("Materials Feature Engineering Tool")


def _create_user_output_dir(user_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Create output directory based on user ID and UUID

    Args:
        user_id: User ID, if None then use 'anonymous'

    Returns:
        (output_dir, run_uuid): Output directory path and run UUID
    """
    # If no user_id, use anonymous
    if not user_id or user_id.strip() == "":
        user_id = "default"

    # Generate unique run ID
    run_uuid = str(uuid.uuid4())

    # Create output directory: data/user_id/uuid/
    base_dir = Path("data")
    user_dir = base_dir / user_id
    output_dir = user_dir / run_uuid

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory created: {output_dir}")
    print(f"   User ID: {user_id}")
    print(f"   Run UUID: {run_uuid}")

    return str(output_dir), run_uuid


def _get_user_id_from_context(ctx: Optional[Context]) -> Optional[str]:
    """
    Extract user_id from Context

    Args:
        ctx: FastMCP Context object

    Returns:
        user_id or None
    """
    if ctx is not None:
        try:
            user_id = ctx.request_context.request.headers.get("user_id", None)  # type: ignore
            return user_id
        except Exception as e:
            print(f"⚠️  Warning: Could not extract user_id from context: {e}")
            return None
    return None


def _json_safe(obj):
    """Recursively convert numpy/pandas types to native Python types that are JSON serializable."""
    # Return basic types directly
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        try:
            return obj.item()
        except Exception:
            return float(obj) if isinstance(obj, np.floating) else int(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas objects
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)

    # Container types
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    # Try to directly serialize other types to JSON, otherwise convert to string
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _auto_output_path(data_path: str, suffix: str = "enhanced") -> str:
    """Generate system path for output file.

    - For local paths: generate files under outputs/ in current working directory
    - For URLs: also generate to outputs/ in current working directory
    File naming rule: <basename>_<suffix>_<YYYYMMDD_HHMMSS>.csv
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
    ctx: Context,
    data_path: str,
    task_type: str,
    target_dims: int,
    enable_preprocessing: bool = False
) -> Dict[str, Any]:
    """
    Materials science data exploration tool

    Args:
        data_path: Data file path (CSV or Excel format)
        task_type: Task type ("regression" or "classification")
        target_dims: Number of target variable dimensions (always at the end of columns)
        enable_preprocessing: Whether to enable data preprocessing and save preprocessed file

    Returns:
        Dictionary containing complete data analysis report
    """

    # Validate input parameters
    _validate_data_path(data_path)

    if task_type.lower() not in ["regression", "classification"]:
        raise ValueError("task_type must be 'regression' or 'classification'")

    if target_dims <= 0:
        raise ValueError("target_dims must be a positive integer")

    # Get user ID and create output directory
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)

    # Create data explorer
    explorer = DataExplorer()

    try:
        # Load data
        explorer.load_data(data_path, task_type, target_dims)

        # Execute all analyses
        analysis_results = {
            "data_summary": explorer.data_summary(),
            "missing_value_analysis": explorer.missing_value_analysis(),
            "distribution_analysis": explorer.distribution_analysis(),
            "target_analysis_and_preprocessing_suggestions": explorer.target_analysis_and_preprocessing()
        }

        # Generate preprocessed data (if enabled)
        if enable_preprocessing:
            # Generate output filename based on original filename
            input_filename = os.path.basename(data_path)
            base_name = os.path.splitext(input_filename)[0]
            output_filename = f"{base_name}_preprocessed.csv"
            output_path = os.path.join(output_dir, output_filename)

            processing_result = explorer.generate_processed_data(output_path)

            from .config import get_download_url
            analysis_results["data_preprocessing"] = {
                "status": "completed",
                "message": processing_result,
                "output_file": get_download_url(output_path),
                "next_step_guide": "You can use the 'output_file' for next operations like generate_materials_basic_features, auto_feature_engineering_with_openfe, or select_optimal_features"
            }
            print(f"\n✓ Preprocessed data saved to: {output_path}")
            print(f"💡 Next: You can use this output_file for feature generation or selection tools")
        else:
            analysis_results["data_preprocessing"] = {
                "status": "not_executed",
                "message": "Preprocessing not enabled, skipping data preprocessing",
                "suggestion": "Set enable_preprocessing=True to generate preprocessed data for next steps"
            }

        return _json_safe(analysis_results) # type: ignore

    except Exception as e:
        return {
            "error": str(e),
            "suggestion": "Please check data file format and parameter settings"
        }


@mcp.tool()
def quick_data_summary(data_path: str) -> Dict[str, Any]:
    """
    Quick data summary tool - only generates basic data overview

    Args:
        data_path: Data file path

    Returns:
        Basic data summary information
    """
    _validate_data_path(data_path)

    explorer = DataExplorer()

    try:
        # Temporarily load data for quick summary
        explorer.load_data(data_path, "regression", 1)
        return _json_safe(explorer.data_summary()) # type: ignore
    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
def generate_materials_basic_features(
    data_path: str,
    sample_rows: int = 5,
    target_dims: int = 1,
    ctx: Context = None # type: ignore
) -> Dict[str, Any]:
    """
    Generate basic material features using LLM analysis (identify chemical composition columns, generate composition-related features via matminer, and remove source composition columns).

    Args:
        data_path: Data file path
        sample_rows: Number of sample rows for analysis
        target_dims: Target variable dimensions (default last n columns)

    Returns:
        Complete analysis and feature generation results
    """
    # Validate input parameters
    _validate_data_path(data_path)

    # Get user ID and create output directory
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    try:
        generator = MaterialsFeatureGenerator()
        generator.load_data(data_path)

        # 1. Analyze columns
        analysis_result = generator.analyze_columns_with_llm(sample_rows=sample_rows)

        # 2. Automatically select composition columns for feature generation (not dependent on confidence, select as long as identified as chemical composition)
        selected_columns = {}
        for col, info in generator.identified_columns.items():
            if info.get("category") == "chemical_composition":
                selected_columns[col] = ["element_property", "stoichiometry", "valence_orbital"]

        # 3. Generate features
        if selected_columns:
            # First generate in memory, then reorder columns, finally save
            enhanced_data = generator.create_enhanced_dataset(
                output_path=None,
                selected_columns=selected_columns
            )

            # Move target columns to the end
            original_cols = list(generator.data.columns) # type: ignore
            target_cols: list = []
            if isinstance(target_dims, int) and target_dims > 0 and target_dims <= len(original_cols):
                target_cols = original_cols[-target_dims:]

            if target_cols:
                non_target_cols = [c for c in enhanced_data.columns if c not in target_cols]
                enhanced_data = enhanced_data[non_target_cols + target_cols]

            # Generate output path to user directory
            input_filename = os.path.basename(data_path)
            base_name = os.path.splitext(input_filename)[0]
            output_filename = f"{base_name}_enhanced.csv"
            output_path = os.path.join(output_dir, output_filename)

            enhanced_data.to_csv(output_path, index=False)
            print(f"\n✓ Enhanced data saved to: {output_path}")

            # Generate feature reports (both text and HTML)
            txt_report_path = output_path.replace('.csv', '_feature_report.txt')
            html_report_path = output_path.replace('.csv', '_feature_report.html')
            try:
                generator._generate_feature_report(enhanced_data, txt_report_path) # type: ignore
                print(f"✓ Feature report (text) saved to: {txt_report_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate text feature report: {e}")

            try:
                generator._generate_html_feature_report(enhanced_data, html_report_path) # type: ignore
                print(f"✓ Feature report (HTML) saved to: {html_report_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate HTML feature report: {e}")

            # Count new feature columns (compared to original column set)
            original_cols_set = set(generator.data.columns) # type: ignore
            new_feature_cols = [c for c in enhanced_data.columns if c not in original_cols_set]
            from .config import get_static_url,get_download_url
            output_path = get_download_url(output_path)
            html_report_url = get_static_url(html_report_path)
            result = {
                "status": "success",
                "input_file": data_path,
                "output_file": output_path,
                # "feature_generated_report_txt_path": txt_report_url,
                "feature_generated_report_html_path": html_report_url,
                "LLM_analysis": analysis_result,
                "selected_columns": selected_columns,
                "original_data_shape": generator.data.shape, # type: ignore
                "enhanced_data_shape": enhanced_data.shape,
                "new_features_count": len(new_feature_cols),
                "new_features_columns": new_feature_cols,
                "next_step_guide": "You can use the 'output_file' for further feature engineering with auto_feature_engineering_with_openfe or feature selection with select_optimal_features. Open the HTML report to view detailed feature descriptions."
            }

            return _json_safe(result)  # type: ignore
        else:
            result = {
                "status": "no_feature_generated",
                "reason": "No suitable chemical composition columns identified",
                "LLM_analysis": analysis_result,
                "suggestion": "Please check data or manually specify composition columns",
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
    Preliminary feature selection before using OpenFE to reduce computational complexity

    Args:
        X: Feature dataframe
        y: Target variable
        task_type: Task type ('regression' or 'classification')
        max_features: Maximum number of features to retain
        variance_threshold: Variance threshold

    Returns:
        List of selected feature column names
    """
    selected_features = []

    # Ensure target variable is numeric type
    if y.dtype == 'object' or y.dtype.name == 'object': # type: ignore
        print(f"Target variable type is {y.dtype}, attempting to convert to numeric type")

        # Handle scientific notation strings (e.g., '7.1×10-5')
        def convert_sci_notation(val):
            """Convert scientific notation string to numeric value"""
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            # Convert scientific notation format: 7.1×10-5 -> 7.1e-5
            val_str = str(val)
            # Replace ×10、x10、X10 with e
            val_str = val_str.replace('×10', 'e').replace('x10', 'e').replace('X10', 'e')
            # Handle cases without 10 (e.g., directly ×-5)
            val_str = val_str.replace('×', 'e').replace('x', 'e').replace('X', 'e')
            try:
                return float(val_str)
            except ValueError:
                return np.nan
        
        y = y.apply(convert_sci_notation) # type: ignore
        y = pd.Series(pd.to_numeric(y, errors='coerce'), index=y.index, name=y.name, dtype=np.float64) # type: ignore

        if y.isna().any(): # type: ignore
            nan_count = y.isna().sum() # type: ignore
            print(f"Warning: Target variable has {nan_count} NaN values after conversion, filling with median")
            median_val = float(y.median()) # type: ignore
            if not np.isnan(median_val): # type: ignore
                y = y.fillna(median_val) # type: ignore
            else:
                y = y.fillna(0) # type: ignore
    elif y.dtype not in [np.float64, np.float32, np.int64, np.int32]: # type: ignore
        # Ensure numeric type
        y = y.astype(np.float64) # type: ignore

    # Step 1: Remove low-variance features
    print(f"Original feature count: {len(X.columns)}")

    # Only perform variance filtering on numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = [col for col in X.columns if col not in numeric_features]
    
    if len(numeric_features) > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        X_numeric = X[numeric_features].fillna(X[numeric_features].mean())
        
        try:
            selector.fit(X_numeric)
            high_variance_features = [numeric_features[i] for i in range(len(numeric_features))
                                     if selector.variances_[i] > variance_threshold]
            print(f"Feature count after variance filtering: {len(high_variance_features)}")
        except Exception as e:
            print(f"Variance filtering failed, retaining all numeric features: {e}")
            high_variance_features = numeric_features
    else:
        high_variance_features = []

    # Step 2: Filter features based on statistical significance
    selected_numeric_features = high_variance_features
    if len(high_variance_features) > max_features:
        print(f"Feature count exceeds {max_features}, performing further filtering...")

        # Ensure data type consistency and properly fill missing values
        X_filtered = pd.DataFrame(X[high_variance_features].copy())

        # Fill missing values column by column, ensuring data type safety
        for col in X_filtered.columns:
            col_data = X_filtered[col]
            if col_data.isna().any(): # type: ignore
                # Try to fill with mean, fall back to median
                try:
                    fill_value = float(col_data.mean()) # type: ignore
                    if pd.isna(fill_value):
                        fill_value = float(col_data.median()) # type: ignore
                    if pd.isna(fill_value):
                        fill_value = 0.0
                    X_filtered.loc[:, col] = col_data.fillna(fill_value) # type: ignore
                except Exception:
                    X_filtered.loc[:, col] = col_data.fillna(0) # type: ignore

        # Ensure all columns are numeric types
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


        # Ensure data type is float64
        try:
            X_filtered = X_filtered.astype(np.float64)
        except Exception:
            # If batch conversion fails, convert column by column
            for col in X_filtered.columns:
                try:
                    X_filtered.loc[:, col] = X_filtered[col].astype(np.float64)
                except Exception:
                    X_filtered.loc[:, col] = 0.0

        # Select scoring function based on task type
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
            print(f"Numeric feature count after statistical significance filtering: {len(selected_numeric_features)}")
        except Exception as e:
            print(f"Statistical filtering failed, retaining high-variance features: {e}")
            import traceback
            print(f"Detailed error information: {traceback.format_exc()}")
            selected_numeric_features = high_variance_features[:max_features]

    # Merge numeric and non-numeric features
    selected_features = selected_numeric_features + non_numeric_features
    print(f"Final feature count after filtering: {len(selected_features)}")
    
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


    # Get user ID and create output directory
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)

    try:
        # Import OpenFE (lazy import to avoid dependency issues at startup)
        try:
            from openfe import OpenFE, transform # type: ignore
        except ImportError:
            return {
                "status": "failed",
                "error": "openfe library not installed, please run: pip install openfe"
            }

        # 1. Load data
        print(f"Loading data: {data_path}")

        try:
            # Use safe loading function to handle both local files and URLs
            local_path = _load_data_safe(data_path)

            # Load data based on file extension
            if local_path.endswith('.csv'):
                df = pd.read_csv(local_path)
            elif local_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(local_path)
            else:
                # Try CSV first, then Excel
                try:
                    df = pd.read_csv(local_path)
                except Exception:
                    df = pd.read_excel(local_path)

            print(f"✓ Data loaded successfully, shape: {df.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")

        # 2. Separate features and targets
        if target_dims <= 0 or target_dims > len(df.columns):
            raise ValueError(f"target_dims must be between 1 and {len(df.columns)}")

        target_columns = df.columns[-target_dims:].tolist()
        feature_columns = df.columns[:-target_dims].tolist()

        print(f"Feature column count: {len(feature_columns)}, Target columns: {target_columns}")

        X = pd.DataFrame(df[feature_columns].copy())
        y = df[target_columns[0]] if target_dims == 1 else df[target_columns]

        # Check for text features (OpenFE does not support text features)
        text_columns = []
        for col in X.columns:
            # Check if column contains text data (object type and not numeric)
            if X[col].dtype == 'object':
                # Try to convert to numeric, if fails, it's text
                try:
                    pd.to_numeric(X[col], errors='raise')
                except (ValueError, TypeError):
                    # Check if it's actually text (not just numeric strings)
                    sample_values = X[col].dropna().head(10).astype(str)
                    # If any value contains letters (excluding scientific notation patterns)
                    has_text = any(
                        any(c.isalpha() for c in str(val) if c not in ['e', 'E', 'x', 'X', '×'])
                        for val in sample_values
                    )
                    if has_text:
                        text_columns.append(col)

        if text_columns:
            return {
                "status": "failed",
                "error": f"OpenFE does not support text features. Found text columns: {text_columns}. Please remove or encode these columns before using OpenFE, or use generate_materials_basic_features for chemical composition features.",
                "text_columns": text_columns,
                "suggestion": "You can: 1) Remove text columns, 2) Use one-hot encoding for categorical text, 3) Use generate_materials_basic_features if columns contain chemical formulas"
            }

        # For classification tasks, encode target variable
        if task_type.lower() == "classification" and target_dims == 1:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_series = pd.Series(y_encoded, index=y.index, name=y.name, dtype=np.float64)
        else:
            # Ensure target variable is numeric type
            if isinstance(y, pd.Series):
                y_temp = y.copy()
            else:
                y_temp = pd.Series(y.iloc[:, 0])

            # Handle scientific notation strings (e.g., '7.1×10-5' or '7.1x10-5')
            if y_temp.dtype == 'object':
                def convert_sci_notation(val):
                    """Convert scientific notation string to numeric value"""
                    if pd.isna(val):
                        return np.nan
                    if isinstance(val, (int, float)):
                        return float(val)
                    # Convert scientific notation format: 7.1×10-5 -> 7.1e-5
                    val_str = str(val)
                    # Replace ×10、x10、X10 with e
                    val_str = val_str.replace('×10', 'e').replace('x10', 'e').replace('X10', 'e')
                    # Handle cases without 10 (e.g., directly ×-5)
                    val_str = val_str.replace('×', 'e').replace('x', 'e').replace('X', 'e')
                    try:
                        return float(val_str)
                    except ValueError:
                        return np.nan

                print(f"Detected scientific notation string format, converting...")
                y_temp = y_temp.apply(convert_sci_notation)

            # Convert to numeric type
            y_series = pd.Series(pd.to_numeric(y_temp, errors='coerce'), index=y_temp.index, name=y_temp.name, dtype=np.float64)

            # Check and fill NaN values from failed conversions
            if y_series.isna().any():
                nan_count = y_series.isna().sum()
                print(f"Warning: Target variable has {nan_count} values that cannot be converted to numeric, will fill with median")
                median_val = y_series.median()
                if pd.isna(median_val): # type: ignore
                    print(f"Warning: Cannot calculate median, filling with 0")
                    y_series = y_series.fillna(0)
                else:
                    y_series = y_series.fillna(median_val)

        # 3. Preliminary feature screening
        print(f"\nStarting preliminary feature screening (target: {n_features_before_openfe} features)...")
        selected_feature_names = _select_features_before_openfe(
            X, 
            y_series,
            task_type=task_type,
            max_features=n_features_before_openfe,
            variance_threshold=0.01
        )
        
        X_selected = pd.DataFrame(X[selected_feature_names].copy())

        # Clean column names to avoid OpenFE conflicts
        # OpenFE is sensitive to certain column names (e.g., norm, index, etc.)
        original_col_names = list(X_selected.columns)
        safe_col_names = []
        col_name_mapping = {}

        for idx, col in enumerate(original_col_names):
            # Remove special characters, replace with underscore
            safe_name = str(col).replace(' ', '_').replace('-', '_').replace('.', '_')
            safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe_name)

            # Remove consecutive underscores
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            safe_name = safe_name.strip('_')

            # If column name is empty or all underscores, use default name
            if not safe_name or not any(c.isalnum() for c in safe_name):
                safe_name = f"feature_{idx}"

            # Avoid OpenFE reserved words or potentially conflicting names
            reserved_names = ['norm', 'index', 'level', 'values', 'count', 'sum', 'mean', 'min', 'max', 'std', 'var']
            if safe_name.lower() in reserved_names:
                safe_name = f"feat_{safe_name}"

            # Ensure it doesn't start with a digit
            if safe_name and safe_name[0].isdigit():
                safe_name = f"feat_{safe_name}"

            # Ensure uniqueness
            original_safe_name = safe_name
            counter = 1
            while safe_name in safe_col_names:
                safe_name = f"{original_safe_name}_{counter}"
                counter += 1

            safe_col_names.append(safe_name)
            col_name_mapping[safe_name] = col

        X_selected.columns = safe_col_names
        print(f"Column names standardized: {len(original_col_names)} columns")

        # Fill missing values
        for col in X_selected.columns:
            if X_selected[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X_selected.loc[:, col] = X_selected[col].fillna(X_selected[col].mean())
            else:
                mode_vals = X_selected[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 'missing'
                X_selected.loc[:, col] = X_selected[col].fillna(fill_val)

        print(f"Feature data shape after screening: {X_selected.shape}")

        # 4. Generate new features using OpenFE
        print(f"\nGenerating {n_new_features} new features using OpenFE...")

        # Clean up possible temporary files
        temp_files = ['./openfe_tmp_data.feather', 'openfe_tmp_data.feather']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Cleaned up old temporary file: {temp_file}")
                except Exception:
                    pass

        ofe = OpenFE()

        # Prepare training data
        train_x = X_selected
        train_y = y_series

        # OpenFE feature generation
        try:
            # Ensure train_y is DataFrame format (required by OpenFE)
            train_y_df = pd.DataFrame(train_y) if isinstance(train_y, pd.Series) else train_y
            
            features = ofe.fit(
                data=train_x,
                label=train_y_df, # type: ignore
                n_jobs=1,  # Single process to avoid serialization issues
                task=task_type,
                n_data_blocks=1,
                stage2_params={'verbose': -1},
                verbose=True
            )

            # Select top n new features
            new_features_list = features[:min(n_new_features, len(features))]

            # Helper function: recursively parse OpenFE feature object tree
            def parse_openfe_feature_tree(feat, depth=0):
                """Recursively parse OpenFE feature tree to generate readable expressions"""
                try:
                    type_name = type(feat).__name__

                    # FNode - original feature leaf node
                    if type_name == 'FNode':
                        if hasattr(feat, 'name') and feat.name:
                            return str(feat.name)
                        elif hasattr(feat, 'data') and feat.data is not None:
                            return str(feat.data)
                        else:
                            return "Unknown feature"

                    # Node - operator node
                    elif type_name == 'Node':
                        if not hasattr(feat, 'name'):
                            return str(feat)

                        op_name = feat.name
                        children = feat.children if hasattr(feat, 'children') and feat.children else []

                        # Debug output
                        if depth == 0:  # Only print at top level
                            print(f"  [DEBUG] Parsing feature: op={op_name}, children_count={len(children)}, has_children={hasattr(feat, 'children')}")

                        if not children or len(children) == 0:
                            print(f"  [WARNING] Feature {op_name} has no child nodes!")
                            return str(op_name)

                        # Recursively parse child nodes
                        child_exprs = [parse_openfe_feature_tree(child, depth+1) for child in children]

                        # Format based on operator type
                        if op_name in ['+', '-', '*', '/', '>', '<', '>=', '<=', '==', '!=']:
                            # Binary operators
                            if len(child_exprs) == 2:
                                return f"({child_exprs[0]} {op_name} {child_exprs[1]})"
                            else:
                                return f"{op_name}({', '.join(child_exprs)})"
                        elif 'GroupBy' in op_name:
                            # GroupBy operations
                            return f"{op_name}({', '.join(child_exprs)})"
                        elif op_name in ['abs', 'sqrt', 'log', 'exp', 'sin', 'cos', 'residual', 'Round', 'round']:
                            # Single-parameter functions
                            if len(child_exprs) == 1:
                                return f"{op_name}({child_exprs[0]})"
                            else:
                                return f"{op_name}({', '.join(child_exprs)})"
                        elif op_name in ['max', 'min', 'mean', 'median', 'std', 'sum', 'var']:
                            # Aggregation functions
                            return f"{op_name}({', '.join(child_exprs)})"
                        else:
                            # Other operations
                            return f"{op_name}({', '.join(child_exprs)})"
                    else:
                        # Other types
                        return str(feat)
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] Error parsing feature: {e}")
                    print(f"  [ERROR] Traceback: {traceback.format_exc()}")
                    return f"<Parsing error: {str(e)}>"

            # Extract feature construction information
            feature_descriptions = {}
            for idx, feat in enumerate(new_features_list):
                try:
                    feat_name = f"autoFE_f_{idx}"
                    # Use tree parsing function to generate readable feature description
                    feat_desc = parse_openfe_feature_tree(feat)

                    if feat_desc and feat_desc.strip() and not feat_desc.startswith('<'):
                        feature_descriptions[feat_name] = feat_desc.strip()
                    else:
                        feature_descriptions[feat_name] = f"Combined feature {idx}"

                except Exception as e:
                    print(f"Error extracting feature {idx} description: {e}")
                    feature_descriptions[feat_name] = f"Combined feature {idx}"

            print(f"Extracted construction information for {len(feature_descriptions)} features")

            # Debug: display first 3 feature descriptions
            if feature_descriptions:
                print("First 3 feature description examples:")
                for i, (name, desc) in enumerate(list(feature_descriptions.items())[:3]):
                    print(f"  {name}: {desc[:100]}..." if len(desc) > 100 else f"  {name}: {desc}")

            # Generate new feature data
            train_x_new_result, _ = transform(train_x, train_y, new_features_list, n_jobs=1)
            train_x_new = pd.DataFrame(train_x_new_result)

            print(f"OpenFE generated {len(train_x_new.columns) - len(train_x.columns)} new features")

        except Exception as e:
            print(f"OpenFE feature generation error: {e}")
            print("Continuing with original screened features...")
            train_x_new = train_x
            feature_descriptions = {}
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass


        # 5. Merge data and save
        # Re-add target columns
        result_df = pd.DataFrame(train_x_new.copy())
        for target_col in target_columns:
            result_df[target_col] = df[target_col].values

        # Generate output path to user directory
        if output_path is None:
            # Generate output filename from original filename
            input_filename = os.path.basename(data_path)
            base_name = os.path.splitext(input_filename)[0]
            output_filename = f"{base_name}_openfe.csv"
            output_path = os.path.join(output_dir, output_filename)
        else:
            # If output_path is provided, put it in user directory
            output_filename = os.path.basename(output_path)
            output_path = os.path.join(output_dir, output_filename)

        result_df.to_csv(output_path, index=False)
        print(f"\n✓ Enhanced data saved to: {output_path}")

        # Save feature description report (text format)
        if feature_descriptions:
            txt_report_path = output_path.replace('.csv', '_feature_descriptions.txt')
            try:
                with open(txt_report_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("OpenFE Automated Feature Engineering Report\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Data file: {data_path}\n")
                    f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Task type: {task_type}\n")
                    f.write(f"Generated features: {len(feature_descriptions)}\n")
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("Feature Construction Details\n")
                    f.write("=" * 80 + "\n\n")

                    for feat_name, feat_desc in feature_descriptions.items():
                        f.write(f"\n【{feat_name}】\n")
                        f.write(f"Construction method: {feat_desc}\n")
                        f.write("-" * 80 + "\n")

                    f.write("\n" + "=" * 80 + "\n")
                    f.write("Notes:\n")
                    f.write("- Feature name format: autoFE_f_N (N is feature index)\n")
                    f.write("- Construction method shows which original features and operations generated this feature\n")
                    f.write("- Common operations: + (add), - (subtract), * (multiply), / (divide), ^ (power), etc.\n")
                    f.write("=" * 80 + "\n")

                print(f"Text report saved to: {txt_report_path}")
            except Exception as e:
                print(f"Failed to save text report: {e}")

            # Generate HTML report
            html_report_path = output_path.replace('.csv', '_report.html')
            try:
                report_generator = OpenFEReportGenerator()

                # Prepare data information
                data_info = {
                    'data_file': os.path.basename(data_path),
                    'task_type': task_type,
                    'original_shape': df.shape,
                    'final_shape': result_df.shape,
                    'target_columns': target_columns
                }

                # Prepare feature information
                feature_info = {
                    'original_features': len(feature_columns),
                    'selected_features': len(selected_feature_names),
                    'selection_rate': f"{len(selected_feature_names)/len(feature_columns)*100:.1f}%",
                    'input_features': train_x.shape[1],
                    'output_features': train_x_new.shape[1],
                    'new_features': train_x_new.shape[1] - train_x.shape[1],
                    'target_new_features': n_new_features
                }

                # Generate HTML report
                report_generator.generate_report(
                    html_report_path,
                    data_info,
                    feature_info,
                    feature_descriptions
                )

                print(f"📊 HTML report saved to: {html_report_path}")
            except Exception as e:
                print(f"Failed to generate HTML report: {e}")
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
        from .config import get_download_url,get_static_url
        # 6. Generate feature report
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
            "feature_list": result_df.columns.tolist(),
            "next_step_guide": "You can use the 'output_file' for feature selection with select_optimal_features to choose the most important features"
        }

        # Add feature description information and report path
        if feature_descriptions:
            txt_path = output_path.replace('.csv', '_feature_descriptions.txt')
            html_path = output_path.replace('.csv', '_report.html')
            report["feature_construction_description"] = {
                "description": "the construction way of each new feature",
                # "text_report": txt_path,
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
        print("Feature Selection Tool (RFE-CV)")
        print(f"{'='*80}\n")

        # Get user ID and create output directory
        user_id = _get_user_id_from_context(ctx)
        output_dir, run_uuid = _create_user_output_dir(user_id)

        # Call feature selection function
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
        print("✓ Feature selection completed!")
        print(f"{'='*80}")
        print(f"\nOutput files:")
        print(f"  - Data: {result['output_file']}")
        print(f"  - PNG Report: {result['report_file']}")
        print(f"  - HTML Report: {result['html_report_file']}")
        print(f"  - Details: {result['details_file']}")
        print(f"\nFeature statistics:")
        print(f"  - Original Features: {result['n_original_features']}")
        print(f"  - Selected Features: {result['n_selected_features']}")
        print(f"  - Retention Rate: {result['retention_rate']}")
        print(f"  - Best CV Score: {result['best_cv_score']:.4f}")

        from .config import get_download_url,get_static_url
        # Ensure return value is JSON serializable
        return {
            'selected_features': [str(f) for f in result['selected_features']],
            # 'rejected_features': [str(f) for f in result['rejected_features']],
            'n_selected_features': int(result['n_selected_features']),
            'n_original_features': int(result['n_original_features']),
            # 'retention_rate': str(result['retention_rate']),
            'best_cv_score': float(result['best_cv_score']),
            'output_file': get_download_url(result['output_file']),
            'report_file': get_download_url(result['report_file']),
            'report_html_path': get_static_url(result['html_report_file']),
            'details_file': str(result['details_file']),
            'task_type': str(result['task_type']),
            'next_step_guide': "Feature selection completed! You can use the 'output_file' with optimally selected features for model training or further analysis"
            # 'cv_folds': int(result['cv_folds']),
            # 'user_id': user_id if user_id else "anonymous",
            # 'run_uuid': run_uuid,
            # 'output_dir': output_dir
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
    """Run MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()