# check.py
import sys, os, subprocess
from pathlib import Path

print("🔍 RAG 项目环境诊断")
print("="*40)

# 1. Python 版本
print(f"✅ Python: {sys.version.split()[0]}")

# 2. 虚拟环境
if hasattr(sys, 'real_prefix') or (sys.base_prefix != sys.prefix):
    print("✅ 虚拟环境: 已激活")
else:
    print("⚠️  虚拟环境: 未激活（可能依赖冲突）")

# 3. 关键依赖
deps = ["streamlit", "langchain_community", "faiss", "PIL"]
for dep in deps:
    try:
        __import__(dep.replace("-", "_"))
        print(f"✅ {dep}: 已安装")
    except:
        print(f"❌ {dep}: 未安装")

# 4. 项目文件
files = ["app.py", "cache_utils.py", "kb_files"]
for f in files:
    path = Path(f)
    if path.exists():
        print(f"✅ {f}: 存在")
    else:
        print(f"❌ {f}: 缺失")

# 5. LM Studio 连接
import requests
try:
    r = requests.get("http://localhost:1234/v1/models", timeout=3)
    print(f"✅ LM Studio: 在线 ({r.status_code})")
except:
    print("❌ LM Studio: 未连接（请启动 Server）")

print("="*40)
print("💡 提示：所有 ✅ 表示环境就绪，可运行 streamlit run app.py")