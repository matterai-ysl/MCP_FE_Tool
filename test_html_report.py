"""
测试HTML报告生成功能
"""
import pandas as pd
from src.materials_feature_engineering_mcp.feature_generator import MaterialsFeatureGenerator

# 创建示例数据
sample_data = pd.DataFrame({
    'composition': ['Fe2O3', 'NaCl', 'TiO2', 'Al2O3', 'SiO2'],
    'property': [5.24, 2.16, 4.23, 3.95, 2.65]
})

# 保存示例数据
sample_data.to_csv('test_sample.csv', index=False)

print("测试 HTML 报告生成功能")
print("=" * 50)

# 创建特征生成器
generator = MaterialsFeatureGenerator()

# 加载数据
print("\n1. 加载数据...")
generator.load_data('test_sample.csv')

# 分析列
print("\n2. 分析化学组成列...")
analysis = generator.analyze_columns_with_llm()
print(f"   发现化学组成列: {analysis.get('composition_columns', [])}")

# 生成增强数据集（会自动生成文本和HTML报告）
print("\n3. 生成特征并创建报告...")
enhanced_data = generator.create_enhanced_dataset(output_path='test_enhanced.csv')

print("\n" + "=" * 50)
print("✓ 测试完成!")
print(f"  增强数据已保存: test_enhanced.csv")
print(f"  文本报告已保存: test_enhanced_feature_report.txt")
print(f"  HTML报告已保存: test_enhanced_feature_report.html")
print("\n请在浏览器中打开 test_enhanced_feature_report.html 查看详细的特征报告")
