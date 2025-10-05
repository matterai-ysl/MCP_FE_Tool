"""
OpenFE特征工程HTML报告生成器
生成美观、交互式的HTML报告，展示特征工程结果
"""

from datetime import datetime
from typing import Dict, List, Any
import os


class OpenFEReportGenerator:
    """OpenFE特征工程报告生成器"""
    
    def __init__(self):
        self.template = self._get_html_template()
    
    def generate_report(
        self,
        output_path: str,
        data_info: Dict[str, Any],
        feature_info: Dict[str, Any],
        feature_descriptions: Dict[str, str]
    ) -> str:
        """
        生成HTML报告
        
        Args:
            output_path: 报告保存路径
            data_info: 数据信息字典
            feature_info: 特征信息字典
            feature_descriptions: 特征描述字典 {特征名: 构造方式}
            
        Returns:
            生成的报告文件路径
        """
        # 准备报告数据
        report_data = {
            'title': 'OpenFE 自动特征工程报告',
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': data_info.get('data_file', 'N/A'),
            'task_type': data_info.get('task_type', 'N/A'),
            'original_shape': data_info.get('original_shape', (0, 0)),
            'final_shape': data_info.get('final_shape', (0, 0)),
            'target_columns': data_info.get('target_columns', []),
            'original_features': feature_info.get('original_features', 0),
            'selected_features': feature_info.get('selected_features', 0),
            'selection_rate': feature_info.get('selection_rate', '0%'),
            'input_features': feature_info.get('input_features', 0),
            'output_features': feature_info.get('output_features', 0),
            'new_features': feature_info.get('new_features', 0),
            'target_new_features': feature_info.get('target_new_features', 0),
            'feature_descriptions': feature_descriptions
        }
        
        # 生成HTML内容
        html_content = self._render_template(report_data)
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _render_template(self, data: Dict[str, Any]) -> str:
        """渲染HTML模板"""
        # 构建特征描述表格
        feature_rows = []
        for idx, (feat_name, feat_desc) in enumerate(data['feature_descriptions'].items(), 1):
            # 确保描述不为空
            if not feat_desc or feat_desc.strip() == '':
                feat_desc = f"自动生成的组合特征 #{idx}"
            
            row = f"""
                <tr>
                    <td class="text-center">{idx}</td>
                    <td><code>{feat_name}</code></td>
                    <td class="feature-desc">{feat_desc}</td>
                </tr>
            """
            feature_rows.append(row)
        
        feature_table = '\n'.join(feature_rows) if feature_rows else '''
            <tr>
                <td colspan="3" class="text-center text-muted">未生成新特征</td>
            </tr>
        '''
        
        # 目标变量显示
        target_display = ', '.join(data['target_columns']) if data['target_columns'] else '未指定'
        
        # 任务类型中文化
        task_type_zh = {
            'regression': '回归',
            'classification': '分类'
        }.get(data['task_type'], data['task_type'])
        
        # 替换模板变量
        html = self.template
        replacements = {
            '{{TITLE}}': data['title'],
            '{{GENERATE_TIME}}': data['generate_time'],
            '{{DATA_FILE}}': data['data_file'],
            '{{TASK_TYPE}}': task_type_zh,
            '{{TARGET_COLUMNS}}': target_display,
            '{{ORIGINAL_SHAPE}}': f"{data['original_shape'][0]} 行 × {data['original_shape'][1]} 列",
            '{{FINAL_SHAPE}}': f"{data['final_shape'][0]} 行 × {data['final_shape'][1]} 列",
            '{{ORIGINAL_FEATURES}}': str(data['original_features']),
            '{{SELECTED_FEATURES}}': str(data['selected_features']),
            '{{SELECTION_RATE}}': data['selection_rate'],
            '{{INPUT_FEATURES}}': str(data['input_features']),
            '{{OUTPUT_FEATURES}}': str(data['output_features']),
            '{{NEW_FEATURES}}': str(data['new_features']),
            '{{TARGET_NEW_FEATURES}}': str(data['target_new_features']),
            '{{FEATURE_TABLE}}': feature_table
        }
        
        for key, value in replacements.items():
            html = html.replace(key, value)
        
        return html
    
    def _get_html_template(self) -> str:
        """获取HTML模板"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .header .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section-title {
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-weight: 600;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .info-card .label {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .info-card .value {
            font-size: 1.3em;
            color: #2d3748;
            font-weight: 600;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.2s;
        }
        
        .stat-box:hover {
            transform: scale(1.05);
        }
        
        .stat-box .number {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .stat-box .label {
            font-size: 1em;
            opacity: 0.9;
        }
        
        .progress-section {
            margin: 30px 0;
        }
        
        .progress-item {
            margin-bottom: 20px;
        }
        
        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 500;
            color: #2d3748;
        }
        
        .progress-bar-container {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .feature-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .feature-table thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .feature-table th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .feature-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .feature-table tbody tr:hover {
            background: #f8f9fa;
        }
        
        .feature-table tbody tr:last-child td {
            border-bottom: none;
        }
        
        .text-center {
            text-align: center;
        }
        
        .text-muted {
            color: #6c757d;
        }
        
        .feature-desc {
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.95em;
        }
        
        code {
            background: #f1f3f5;
            padding: 3px 8px;
            border-radius: 4px;
            color: #667eea;
            font-weight: 600;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .badge-success {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-info {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #fdcb6e;
        }
        
        .highlight-box h3 {
            color: #2d3748;
            margin-bottom: 10px;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            
            .stat-box:hover,
            .info-card:hover {
                transform: none;
            }
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 {{TITLE}}</h1>
            <p class="subtitle">生成时间: {{GENERATE_TIME}}</p>
        </div>
        
        <div class="content">
            <!-- 基本信息 -->
            <div class="section">
                <h2 class="section-title">📋 基本信息</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">数据文件</div>
                        <div class="value">{{DATA_FILE}}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">任务类型</div>
                        <div class="value">{{TASK_TYPE}}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">目标变量</div>
                        <div class="value">{{TARGET_COLUMNS}}</div>
                    </div>
                </div>
                
                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">原始数据维度</div>
                        <div class="value">{{ORIGINAL_SHAPE}}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">最终数据维度</div>
                        <div class="value">{{FINAL_SHAPE}}</div>
                    </div>
                </div>
            </div>
            
            <!-- 特征工程统计 -->
            <div class="section">
                <h2 class="section-title">📊 特征工程统计</h2>
                
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="number">{{ORIGINAL_FEATURES}}</div>
                        <div class="label">原始特征数</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">{{SELECTED_FEATURES}}</div>
                        <div class="label">筛选后特征数</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">{{NEW_FEATURES}}</div>
                        <div class="label">新增特征数</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">{{OUTPUT_FEATURES}}</div>
                        <div class="label">最终特征数</div>
                    </div>
                </div>
                
                <!-- 特征筛选进度 -->
                <div class="progress-section">
                    <div class="progress-item">
                        <div class="progress-label">
                            <span>特征筛选进度</span>
                            <span><strong>{{SELECTION_RATE}}</strong></span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: {{SELECTION_RATE}}">
                                {{ORIGINAL_FEATURES}} → {{SELECTED_FEATURES}}
                            </div>
                        </div>
                    </div>
                    
                    <div class="progress-item">
                        <div class="progress-label">
                            <span>特征增强</span>
                            <span><strong>+{{NEW_FEATURES}} 个新特征</strong></span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: 100%">
                                {{INPUT_FEATURES}} → {{OUTPUT_FEATURES}}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="highlight-box">
                    <h3>✨ 特征工程摘要</h3>
                    <p>
                        从 <strong>{{ORIGINAL_FEATURES}}</strong> 个原始特征中，通过方差筛选和统计显著性检验，
                        选出了 <strong>{{SELECTED_FEATURES}}</strong> 个高质量特征（筛选率: <strong>{{SELECTION_RATE}}</strong>）。
                        OpenFE 在这些特征基础上，通过智能特征组合，生成了 <strong>{{NEW_FEATURES}}</strong> 个新特征，
                        最终得到 <strong>{{OUTPUT_FEATURES}}</strong> 个增强特征。
                    </p>
                </div>
            </div>
            
            <!-- 新特征详情 -->
            <div class="section">
                <h2 class="section-title">🔍 新特征构造详情</h2>
                <p style="color: #6c757d; margin-bottom: 20px;">
                    下表列出了 OpenFE 自动生成的每个新特征及其构造方式。这些特征通过组合原始特征的数学运算得到，
                    可能包含更高层次的信息，有助于提升模型性能。
                </p>
                
                <table class="feature-table">
                    <thead>
                        <tr>
                            <th style="width: 80px;">序号</th>
                            <th style="width: 200px;">特征名称</th>
                            <th>构造方式</th>
                        </tr>
                    </thead>
                    <tbody>
                        {{FEATURE_TABLE}}
                    </tbody>
                </table>
            </div>
            
            <!-- 说明 -->
            <div class="section">
                <h2 class="section-title">📖 说明</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">特征命名</div>
                        <div class="value" style="font-size: 1em;">
                            autoFE_f_N 表示第 N 个生成的特征
                        </div>
                    </div>
                    <div class="info-card">
                        <div class="label">常见运算</div>
                        <div class="value" style="font-size: 1em;">
                            + (加) - (减) * (乘) / (除) ^ (幂)
                        </div>
                    </div>
                    <div class="info-card">
                        <div class="label">特征选择</div>
                        <div class="value" style="font-size: 1em;">
                            基于梯度提升树的特征重要性
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🔬 由 Materials Feature Engineering MCP Tool 自动生成</p>
            <p>OpenFE 自动特征工程 | Powered by AI</p>
        </div>
    </div>
</body>
</html>'''

