项目架构--

Knowledge/                          # 项目根目录
├── app.py                          # 主程序（Streamlit 入口）
├── cache_utils.py                  # 缓存模块（精确匹配 + TTL）
├── requirements.txt                # Python 依赖列表
├── .gitignore                      # Git 忽略规则
├── run.bat                         # Windows 一键启动脚本（可选）
├── kb_files/                       # 知识库文件目录（需手动创建）
│   ├── *.txt                       # 文本知识
│   ├── *.md                        # Markdown 笔记
│   └── *.png/*.jpg                 # 图像知识
├── temp_uploads/                   # 临时上传目录（自动创建）
├── .rag_cache/                     # 缓存数据目录（自动创建）
│   └── cache.db                    # SQLite 缓存数据库
└── venv/                           # Python 虚拟环境（可选但推荐）

前置条件--
Python；LM Studio；Git（可选）

验证环境--
# 检查 Python 版本
python --version

# 检查 pip 是否可用
pip --version


快速启动
cd 【项目根目录】

1 创建虚拟环境（只需执行 1 次）
python -m venv venv
# Windows 激活
venv\Scripts\activate

2 激活成功后，命令行前会显示 (venv)

3 创建知识库目录（如果不存在）
mkdir kb_files

4 将你的 .txt/.md/.png 文件放入 kb_files/ 文件夹
# 示例：
# kb_files/
#   ├── 项目笔记.txt
#   ├── 架构图.png
#   └── 会议记录.md

5 启动 LM Studio
打开 LM Studio 应用
搜索并加载模型（如 Qwen-3.5-32B）
切换到 Local Server 标签页
点击 🟢 Start Server
确认地址显示：http://localhost:1234

6 运行
# 确保：①在项目目录 ②虚拟环境已激活 ③LM Studio Server 已启动
streamlit run app.py