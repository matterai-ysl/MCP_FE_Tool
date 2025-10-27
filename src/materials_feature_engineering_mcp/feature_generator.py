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

# Import safe data loading function
from .utils import _load_data_safe

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
            # 生成文本报告
            txt_report_path = output_path.replace('.csv', '_feature_report.txt')
            self._generate_feature_report(enhanced_data, txt_report_path)
            # 生成HTML报告
            html_report_path = output_path.replace('.csv', '_feature_report.html')
            self._generate_html_feature_report(enhanced_data, html_report_path)

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

    def _generate_html_feature_report(self, enhanced_data: pd.DataFrame, report_path: str):
        """Generate HTML format feature report with detailed explanations for each feature"""
        data = cast(pd.DataFrame, self.data)
        original_cols = len(data.columns)
        new_cols = len(enhanced_data.columns) - original_cols

        # Get generated feature columns
        generated_features = [col for col in enhanced_data.columns if col not in data.columns]

        def explain_feature(feature_name: str) -> str:
            """Generate explanation for a specific feature based on its naming pattern"""

            # MagpieData features - ElementProperty featurizer
            if 'MagpieData' in feature_name:
                parts = feature_name.replace('MagpieData ', '').split(' ', 1)
                if len(parts) == 2:
                    stat, prop = parts

                    # Statistical operation descriptions
                    stat_desc = {
                        'minimum': 'the minimum value',
                        'maximum': 'the maximum value',
                        'range': 'the range (max - min)',
                        'mean': 'the weighted average',
                        'avg_dev': 'the average deviation from mean',
                        'mode': 'the most common value'
                    }

                    # Property descriptions
                    prop_desc = {
                        'Number': 'atomic number (number of protons in nucleus)',
                        'MendeleevNumber': 'Mendeleev number (position in periodic table)',
                        'AtomicWeight': 'atomic weight (average mass of atoms)',
                        'MeltingT': 'melting temperature (temperature at which solid becomes liquid)',
                        'Column': 'group number in periodic table (vertical position)',
                        'Row': 'period number in periodic table (horizontal position)',
                        'CovalentRadius': 'covalent radius (atomic size in covalent bonds)',
                        'Electronegativity': 'electronegativity (ability to attract electrons)',
                        'NsValence': 'number of valence electrons in s orbital',
                        'NpValence': 'number of valence electrons in p orbital',
                        'NdValence': 'number of valence electrons in d orbital',
                        'NfValence': 'number of valence electrons in f orbital',
                        'NValence': 'total number of valence electrons',
                        'NsUnfilled': 'number of unfilled electrons in s orbital',
                        'NpUnfilled': 'number of unfilled electrons in p orbital',
                        'NdUnfilled': 'number of unfilled electrons in d orbital',
                        'NfUnfilled': 'number of unfilled electrons in f orbital',
                        'NUnfilled': 'total number of unfilled electrons',
                        'GSvolume_pa': 'ground state volume per atom (atomic volume in stable state)',
                        'GSbandgap': 'ground state band gap (energy gap between valence and conduction bands)',
                        'GSmagmom': 'ground state magnetic moment (measure of magnetism)',
                        'SpaceGroupNumber': 'space group number (crystal structure symmetry)'
                    }

                    stat_text = stat_desc.get(stat, stat)
                    prop_text = prop_desc.get(prop, prop)

                    return f"Calculates {stat_text} of {prop_text} across all elements in the composition, weighted by stoichiometry."

            # Stoichiometry features
            if '-norm' in feature_name:
                norm_num = feature_name.split('-')[0]
                return f"Computes the {norm_num}-norm of the elemental composition vector. This measures the compositional complexity and element distribution pattern."

            # ValenceOrbital features
            if 'valence electrons' in feature_name:
                if feature_name.startswith('avg'):
                    orbital = feature_name.split()[1]
                    return f"Average number of valence electrons in {orbital} orbital across all elements in the composition, weighted by stoichiometry."
                elif feature_name.startswith('frac'):
                    orbital = feature_name.split()[1]
                    return f"Fraction of total valence electrons that occupy the {orbital} orbital in the composition."

            # Element amount features
            # Check if it's a pure element symbol (1-2 characters, starts with uppercase)
            if len(feature_name) <= 2 and feature_name[0].isupper():
                return f"Amount of {feature_name} element in the chemical formula (0 if not present)."

            return "Feature generated from the composition."

        # Classify features by type
        element_property_features = [col for col in generated_features if 'MagpieData' in col]
        stoichiometry_features = [col for col in generated_features if '-norm' in col]
        valence_orbital_features = [col for col in generated_features if 'valence electrons' in col]
        element_amount_features = [col for col in generated_features
                                   if col not in element_property_features
                                   and col not in stoichiometry_features
                                   and col not in valence_orbital_features]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Materials Feature Engineering Report</title>
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
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .summary {{
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
        }}
        .summary h2 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .summary-item .label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .summary-item .value {{
            color: #667eea;
            font-size: 1.5em;
            font-weight: bold;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .featurizer-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .featurizer-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        .featurizer-card h3 {{
            color: #764ba2;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        .featurizer-card .description {{
            color: #555;
            line-height: 1.8;
            margin-bottom: 20px;
            font-size: 1.05em;
        }}
        .feature-list {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }}
        .feature-list h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        .feature-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .feature-tag {{
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            transition: background 0.3s ease;
        }}
        .feature-tag:hover {{
            background: #764ba2;
        }}
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .feature-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        .feature-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .feature-table tr:last-child td {{
            border-bottom: none;
        }}
        .feature-table tr:hover {{
            background: #f8f9fa;
        }}
        .feature-name {{
            font-family: 'Courier New', monospace;
            color: #764ba2;
            font-weight: 600;
        }}
        .feature-explanation {{
            color: #555;
            line-height: 1.6;
        }}
        .footer {{
            background: #2d3436;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Materials Feature Engineering Report</h1>
            <p>Feature Analysis Based on Matminer Library</p>
        </div>

        <div class="content">
            <div class="summary">
                <h2>Data Overview</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="label">Original Data Rows</div>
                        <div class="value">{data.shape[0]}</div>
                    </div>
                    <div class="summary-item">
                        <div class="label">Original Columns</div>
                        <div class="value">{original_cols}</div>
                    </div>
                    <div class="summary-item">
                        <div class="label">Enhanced Data Columns</div>
                        <div class="value">{enhanced_data.shape[1]}</div>
                    </div>
                    <div class="summary-item">
                        <div class="label">New Features Added</div>
                        <div class="value">{new_cols}</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Generated Features and Explanations</h2>
                <p style="margin-bottom: 20px; color: #555;">
                    This report lists all {len(generated_features)} features generated from the chemical composition data,
                    organized by feature type. Each feature name follows a specific naming convention that describes
                    what property it represents.
                </p>

                {''.join([f'''<div class="featurizer-card">
                    <h3>{category_name}</h3>
                    <p style="margin-bottom: 15px; color: #666;">{category_desc}</p>
                    <table class="feature-table">
                        <thead>
                            <tr>
                                <th style="width: 40%;">Feature Name</th>
                                <th style="width: 60%;">Explanation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([f'''<tr>
                                <td class="feature-name">{feat}</td>
                                <td class="feature-explanation">{explain_feature(feat)}</td>
                            </tr>''' for feat in features])}
                        </tbody>
                    </table>
                </div>''' for category_name, category_desc, features in [
                    ('ElementProperty Features (MagpieData)',
                     f'Statistics of elemental properties weighted by stoichiometry. Total: {len(element_property_features)} features.',
                     element_property_features),
                    ('Stoichiometry Features',
                     f'Compositional complexity metrics using p-norms. Total: {len(stoichiometry_features)} features.',
                     stoichiometry_features),
                    ('ValenceOrbital Features',
                     f'Statistics of valence electron orbital occupancies. Total: {len(valence_orbital_features)} features.',
                     valence_orbital_features),
                    ('Element Amount Features',
                     f'Direct element amounts from composition. Total: {len(element_amount_features)} features.',
                     element_amount_features)
                ] if features])}
            </div>

            <div class="section">
                <h2>Identified Composition Columns</h2>
                {''.join([f'''<div class="featurizer-card">
                    <h3>{col}</h3>
                    <div class="description">
                        <p><strong>Category:</strong> {info.get('category', 'unknown')}</p>
                        <p><strong>Confidence:</strong> {info.get('confidence', 'unknown')}</p>
                        <p><strong>Description:</strong> {info.get('description', 'N/A')}</p>
                    </div>
                </div>''' for col, info in self.identified_columns.items()])}
            </div>

            <div class="section">
                <h2>Naming Convention Guide</h2>
                <div class="featurizer-card">
                    <h3>Understanding Feature Names</h3>
                    <div class="description">
                        <p><strong>MagpieData [statistic] [property]:</strong> Statistical measures (minimum, maximum, range, mean, avg_dev, mode) of elemental properties across the composition.</p>
                        <p><strong>[N]-norm:</strong> The N-norm of the compositional vector, measuring element distribution complexity.</p>
                        <p><strong>avg/frac [s/p/d/f] valence electrons:</strong> Average number or fraction of valence electrons in specific orbitals.</p>
                        <p><strong>[Element Symbol]:</strong> Direct amount of that element in the chemical formula.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Based on <a href="https://hackingmaterials.lbl.gov/matminer/" target="_blank">Matminer</a> Materials Data Mining Toolkit</p>
        </div>
    </div>
</body>
</html>"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ HTML特征报告已保存到: {report_path}")