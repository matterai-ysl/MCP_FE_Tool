"""
材料科学机器学习特征工程MCP工具 - 特征生成模块
使用LLM分析表格数据，并通过matminer生成材料特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, cast
import warnings
import json
import os
import re
from urllib.parse import urlparse
# LLM相关导入

from litellm import completion
from dotenv import load_dotenv
load_dotenv()

# matminer相关导入

from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from pymatgen.core import Composition
MATMINER_AVAILABLE = True


warnings.filterwarnings('ignore')


class MaterialsFeatureGenerator:
    """材料科学特征生成器"""

    def __init__(self, llm_model: str = "openai/gpt-4o", api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        初始化特征生成器

        Args:
            llm_model: LLM模型名称
            api_key: API密钥（如果不提供，将从环境变量获取）
        """
        self.llm_model = llm_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("BASE_URL")
        self.data = None
        self.identified_columns = {}


    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载数据文件

        Args:
            data_path: 数据文件路径（支持本地文件或URL）

        Returns:
            加载的DataFrame
        """
        # 根据路径（含URL）推断扩展名；无法推断时尝试CSV/Excel加载
        parsed_path = urlparse(data_path).path.lower()
        is_url = data_path.startswith(('http://', 'https://'))
        
        try:
            if parsed_path.endswith('.csv'):
                if is_url:
                    # 为URL添加超时和错误处理
                    import urllib.request
                    import urllib.error
                    import io
                    print("📥 正在从URL下载数据...")
                    req = urllib.request.Request(data_path, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read()
                    self.data = pd.read_csv(io.BytesIO(content))
                else:
                    self.data = pd.read_csv(data_path)
            elif parsed_path.endswith(('.xlsx', '.xls')):
                if is_url:
                    import urllib.request
                    import urllib.error
                    import io
                    print("📥 正在从URL下载数据...")
                    req = urllib.request.Request(data_path, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read()
                    self.data = pd.read_excel(io.BytesIO(content))
                else:
                    self.data = pd.read_excel(data_path)
            else:
                # 未能从扩展名判断，先尝试CSV，再回退到Excel
                if is_url:
                    import urllib.request
                    import urllib.error
                    import io
                    print("📥 正在从URL下载数据（尝试CSV格式）...")
                    req = urllib.request.Request(data_path, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read()
                    try:
                        self.data = pd.read_csv(io.BytesIO(content))
                    except Exception:
                        self.data = pd.read_excel(io.BytesIO(content))
                else:
                    try:
                        self.data = pd.read_csv(data_path)
                    except Exception:
                        self.data = pd.read_excel(data_path)
        except urllib.error.URLError as e:  # type: ignore
            raise ValueError(f"URL加载失败: {str(e)}. 请检查URL是否可访问")
        except Exception as e:
            if "timed out" in str(e).lower():
                raise ValueError(f"URL加载超时（30秒），请检查网络连接或尝试本地文件")
            raise ValueError(f"不支持的文件格式或无法解析，建议使用CSV或Excel文件。详细错误: {e}")

        print(f"✓ 已加载数据: {self.data.shape}")
        return self.data

    def analyze_columns_with_llm(self, sample_rows: int = 5) -> Dict[str, Any]:
        """
        使用LLM分析前几行，仅识别化学组成列
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        data = cast(pd.DataFrame, self.data)

        sample_data = data.head(sample_rows)
        data_str = sample_data.to_string(max_rows=sample_rows)

        # 构建提示词：仅识别化学组成列并返回列名
        prompt = f"""
作为材料科学专家，请分析下表前{sample_rows}行，仅判断是否存在“化学组成”列（如化学式 'Fe2O3', 'NaCl', 'Li1.05Mn0.95O2' 等）。

数据样本：
{data_str}

请只返回JSON，格式严格如下（不要添加额外文本）：
{{
  "composition_columns": ["列名1", "列名2"],
  "summary": "一句话说明判断依据"
}}
"""

        try:
            response = completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个材料科学和机器学习专家，擅长识别数据中的化学组成列。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                api_key=self.api_key,
                base_url=self.api_base
            )

            llm_response = response.choices[0].message.content # type: ignore
            print("llm_response", llm_response)
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL) # type: ignore
            if not json_match:
                raise ValueError("LLM未返回有效JSON")
            analysis_result = json.loads(json_match.group())

            composition_cols = analysis_result.get("composition_columns", []) or []
            self.identified_columns = {col: {"category": "chemical_composition", "confidence": "高"} for col in composition_cols}
            return analysis_result

        except Exception as e:
            # 简易规则回退：正则检测化学式
            print(f"LLM分析失败: {e}")
            composition_pattern = re.compile(r'^[A-Z][a-z]?\d*(?:\.?\d+)?(?:[A-Z][a-z]?\d*(?:\.?\d+)?)*$')
            candidate_cols: List[str] = []
            for col in self.data.columns:
                vals = self.data[col].dropna().astype(str).head(sample_rows).tolist()
                if vals and all(composition_pattern.match(v.replace(' ', '')) for v in vals):
                    candidate_cols.append(col)
            self.identified_columns = {col: {"category": "chemical_composition", "confidence": "中"} for col in candidate_cols}
            return {"composition_columns": candidate_cols, "summary": "使用简易规则回退"}

    def generate_composition_features(self, composition_column: str,
                                    feature_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        基于化学组成生成特征（test.py 风格流水线）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        data = cast(pd.DataFrame, self.data)

        if composition_column not in data.columns:
            raise ValueError(f"列 '{composition_column}' 不存在")

        # 直接使用 matminer 流水线
        df = data.copy()
        original_cols = list(df.columns)

        def to_composition(x: Any):
            try:
                # 去除空格，提升解析鲁棒性（如 "Fe2 O3" -> "Fe2O3"）
                return Composition(str(x).replace(" ", "")) if pd.notna(x) else None
            except Exception:
                return None

        df["composition"] = df[composition_column].apply(to_composition)

        ep = ElementProperty.from_preset("magpie")
        df = cast(pd.DataFrame, ep.featurize_dataframe(df, col_id="composition"))

        st = Stoichiometry()
        df = cast(pd.DataFrame, st.featurize_dataframe(df, "composition"))

        vo = ValenceOrbital()
        df = cast(pd.DataFrame, vo.featurize_dataframe(df, "composition"))

        # 基于 Composition 展开元素含量列
        el_amount_dicts: List[Dict[str, float]] = []
        for comp in df["composition"].tolist():
            if isinstance(comp, Composition):
                try:
                    el_amount_dicts.append(cast(Dict[str, float], comp.get_el_amt_dict()))
                except Exception:
                    el_amount_dicts.append({})
            else:
                el_amount_dicts.append({})

        all_elements: List[str] = sorted(list({el for d in el_amount_dicts for el in d.keys()}))
        elements_df = pd.DataFrame({el: [d.get(el, 0.0) for d in el_amount_dicts] for el in all_elements}, index=df.index)

        new_feature_cols = [c for c in df.columns if c not in original_cols and c != "composition"]
        features_df = cast(pd.DataFrame, df[new_feature_cols].copy())
        # 合并元素含量列
        features_df = cast(pd.DataFrame, pd.concat([features_df, elements_df], axis=1))
        print(f"成功生成 {len(features_df.columns)} 个组成特征")
        return features_df

    def generate_features_for_identified_columns(self,
                                               selected_columns: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        为识别出的列生成特征（仅支持化学组成列）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        data = cast(pd.DataFrame, self.data)

        all_features = pd.DataFrame(index=data.index)

        if selected_columns is None:
            selected_columns = {}
            for col, info in self.identified_columns.items():
                if info.get("category") == "chemical_composition":
                    selected_columns[col] = ["element_property", "stoichiometry", "valence_orbital"]

        for col in selected_columns.keys():
            if col not in data.columns:
                print(f"警告: 列 '{col}' 不存在，跳过")
                continue
            comp_features = self.generate_composition_features(col)
            all_features = cast(pd.DataFrame, pd.concat([all_features, comp_features], axis=1))
            print(f"为列 '{col}' 生成了 {len(comp_features.columns)} 个特征")

        return all_features

    def create_enhanced_dataset(self, output_path: Optional[str] = None,
                              selected_columns: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        创建包含原始数据和生成特征的增强数据集
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        data = cast(pd.DataFrame, self.data)

        generated_features = self.generate_features_for_identified_columns(selected_columns)
        enhanced_data = cast(pd.DataFrame, pd.concat([data, generated_features], axis=1))

        # 删除源字符串组成列（如果存在）。
        drop_candidates: List[str]
        if selected_columns is not None and len(selected_columns) > 0:
            drop_candidates = list(selected_columns.keys())
        else:
            drop_candidates = [col for col, info in self.identified_columns.items() if info.get("category") == "chemical_composition"]
        for col in drop_candidates:
            if col in enhanced_data.columns:
                enhanced_data.drop(columns=[col], inplace=True)

        if output_path:
            enhanced_data.to_csv(output_path, index=False)
            print(f"增强数据集已保存到: {output_path}")
            report_path = output_path.replace('.csv', '_feature_report.txt')
            self._generate_feature_report(enhanced_data, report_path)

        return enhanced_data

    def _generate_feature_report(self, enhanced_data: pd.DataFrame, report_path: str):
        """生成特征报告"""
        data = cast(pd.DataFrame, self.data)
        original_cols = len(data.columns)
        new_cols = len(enhanced_data.columns) - original_cols

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("材料科学特征生成报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"原始数据: {data.shape[0]} 行 × {original_cols} 列\n")
            f.write(f"增强数据: {enhanced_data.shape[0]} 行 × {enhanced_data.shape[1]} 列\n")
            f.write(f"新增特征: {new_cols} 个\n\n")

            f.write("列分析结果:\n")
            f.write("-" * 30 + "\n")
            for col, info in self.identified_columns.items():
                f.write(f"列名: {col}\n")
                f.write(f"类别: {info.get('category', 'unknown')}\n")
                f.write(f"置信度: {info.get('confidence', 'unknown')}\n")
                f.write(f"描述: {info.get('description', 'N/A')}\n")
                f.write("-" * 30 + "\n")

            f.write(f"\n生成的特征列:\n")
            for col in enhanced_data.columns:
                if col not in data.columns:
                    f.write(f"- {col}\n")

        print(f"特征报告已保存到: {report_path}")