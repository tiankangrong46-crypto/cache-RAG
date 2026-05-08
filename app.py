# =============================
# --- 导入模块 ---
# =============================
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader
from cache_utils import SimpleRAGCache
import os
from pathlib import Path
import base64
from PIL import Image
import io
import threading
from langchain_community.chat_models import ChatOpenAI

# =============================
# --- 路径配置（只定义一次）---
# =============================
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "kb_files"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
CACHE_DIR = BASE_DIR / ".rag_cache"

 # 创建文本嵌入
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        

# 确保目录存在
for d in [TEMP_UPLOAD_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

# 验证知识库目录
if not KNOWLEDGE_DIR.exists():
    st.warning(f"⚠️ 知识库目录不存在：{KNOWLEDGE_DIR}\n请创建文件夹并放入 .txt/.md/.png 等文件")

# =============================
# --- 多模态 RAG 工具函数 ---
# =============================

def encode_image_to_base64(image_path):
    """将图像文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_image_file(file_path):
    """检查是否为图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return Path(file_path).suffix.lower() in image_extensions

def is_text_file(file_path):
    """检查是否为文本文件"""
    text_extensions = ['.txt', '.md']
    return Path(file_path).suffix.lower() in text_extensions

def load_and_process_files(directory_path):
    """加载并处理目录下的所有文件（文本和图像）"""
    text_content = ""
    image_files = []
    
    # 要跳过的文件扩展名
    skip_extensions = {'.py', '.db', '.gitignore', '.pyc', '.json'}
    
    for root, dirs, files in os.walk(directory_path):
        # 跳过缓存和临时目录
        if '.rag_cache' in root or 'temp_uploads' in root or '__pycache__' in root:
            continue
            
        for file in files:
            # 跳过代码/缓存文件
            if Path(file).suffix.lower() in skip_extensions:
                continue
                
            file_path = os.path.join(root, file)
            
            if is_text_file(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text_content += f"\n\n文件：{file}\n{content}"
                except Exception as e:
                    print(f"读取文件失败 {file_path}: {e}")
                    
            elif is_image_file(file_path):
                image_files.append(file_path)
    
    return text_content, image_files

def create_multimodal_rag_system(text_directory="./kb_files/"):
    """创建多模态 RAG 系统（修复版）"""
    
    # 加载文本内容和图像文件
    text_content, image_files = load_and_process_files(text_directory)
    
    # 情况1：没有任何文件 → 返回最小配置（含缓存）
    if not text_content and not image_files:
        llm = ChatOpenAI(
            model="local-model",
            base_url="http://localhost:1234/v1/chat/completions",
            api_key="not-needed",
            temperature=0.1,
            streaming=False,
        )
        return {  # ← ✅ 缩进正确：在 if 块内
            "vector_db": None,
            "llm": llm,
            "retriever": None,
            "cache": SimpleRAGCache(cache_dir=str(CACHE_DIR), ttl_hours=24)
        }, []  # ← ✅ 这个 return 只在此 if 成立时执行
    
    # 情况2：有文本内容 → 初始化向量库 + 缓存
    if text_content:
        # 文本分割
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        docs = text_splitter.split_text(text_content)

        embeddings = get_embeddings()
        db = FAISS.from_texts(docs, embeddings)
        
        llm = ChatOpenAI(
            model="local-model",
            base_url="http://localhost:1234/v1/chat/completions",
            api_key="not-needed",
            temperature=0.1,
            streaming=False,
        )
        retriever = db.as_retriever()
        
        return {
            "vector_db": db,
            "llm": llm,
            "retriever": retriever,
            "cache": SimpleRAGCache(cache_dir=str(CACHE_DIR), ttl_hours=24)
        }, image_files
    
    # 情况3：只有图像没有文本 → 返回含缓存的配置
    llm = ChatOpenAI(
        model="local-model",
        base_url="http://localhost:1234/v1/chat/completions",
        api_key="not-needed",
        temperature=0.1,
        streaming=False,
    )
    return {
        "vector_db": None,
        "llm": llm,
        "retriever": None,
        "cache": SimpleRAGCache(cache_dir=str(CACHE_DIR), ttl_hours=24)
    }, image_files

def multimodal_query_with_images(rag_system, query, image_files):
    """多模态查询处理（含缓存支持 + 修复版）"""
    
    # 1️⃣ 空知识库 + 无图像 → 普通对话
    if rag_system is None and not image_files:
        llm = ChatOpenAI(
            model="local-model",
            base_url="http://localhost:1234/v1/chat/completions",  # ✅ 修复
            api_key="not-needed",
            temperature=0.1,
            streaming=False,
        )
        messages = [
            SystemMessage(content="你是一个有用的 AI 助手。"),
            HumanMessage(content=query)
        ]
        return llm.invoke(messages).content

    # 2️⃣ 获取缓存
    cache = rag_system.get("cache") if rag_system else None
    
    # 3️⃣ 缓存检查
    if cache:
        if rag_system.get("retriever"):
            relevant_docs = rag_system["retriever"].get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in relevant_docs])
        else:
            relevant_docs = []
            context = ""  # ✅ 修复缩进
        
        # 显示检索内容（可选）
        if relevant_docs:
            with st.sidebar:
                with st.expander("📚 检索内容", expanded=False):
                    for i, doc in enumerate(relevant_docs[:3]):
                        st.text(doc.page_content[:150] + "...")
        
        cached_response = cache.get(query, context, image_files)
        if cached_response:
            st.toast("🎯 缓存命中！", icon="✅")
            return f"✅ [缓存] {cached_response}"
    
    # 4️⃣ 缓存未命中 → 执行正常流程
    if rag_system is None:
        return handle_image_only_query(query, image_files)
    
    # 确保 context 已定义
    if 'context' not in locals():
        if rag_system.get("retriever"):
            docs = rag_system["retriever"].get_relevant_documents(query)
            context = "\n".join([d.page_content for d in docs])
        else:
            context = ""
    
    # 5️⃣ 构建消息（含图片）
    if image_files:
        messages = [
            SystemMessage(content="你是一个多模态 AI 助手。"),
            HumanMessage(content=[{"type": "text", "text": f"知识库：\n{context}\n\n问题：{query}"}])
        ]
        for img_path in image_files[:3]:
            try:
                if Path(img_path).exists():
                    b64 = encode_image_to_base64(img_path)
                    ext = Path(img_path).suffix.lower().lstrip('.')
                    messages[1].content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"}  # ✅ 修复 URL 格式
                    })
            except Exception as e:
                st.warning(f"⚠️ 图像加载失败：{e}")
    else:
        messages = [
            SystemMessage(content="你是一个 AI 助手。"),
            HumanMessage(content=f"知识库：\n{context}\n\n问题：{query}")
        ]
    
    # 6️⃣ 调用 LLM
    llm = rag_system["llm"]
    result = llm.invoke(messages).content
    
    # 7️⃣ 异步写入缓存
    if cache:
        def _save():
            try: cache.set(query, context, image_files, result)
            except: pass
        threading.Thread(target=_save, daemon=True).start()
    
    return result

def handle_image_only_query(query, image_files):
    """处理只有图像的查询"""
    if not image_files:
        return "没有可用的图像文件来回答您的问题。"
    
    messages = [
        SystemMessage(content="你是一个多模态 AI 助手，专门分析图像内容来回答问题。"),
        HumanMessage(content=[
            {"type": "text", "text": f"请分析以下图像并回答关于'{query}'的问题："}
        ])
    ]
    
    # 添加图像
    for img_path in image_files[:3]:
        try:
            base64_image = encode_image_to_base64(img_path)
            image_ext = Path(img_path).suffix.lower().lstrip('.')
            
            messages[1].content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_ext};base64,{base64_image}"
                }
            })
        except Exception as e:
            st.warning(f"无法处理图像 {img_path}: {str(e)}")
    
    # 创建临时 LLM 实例（用于纯图像分析）
    llm = ChatOpenAI(
        model="local-model",
        base_url="http://localhost:1234/v1/chat/completions",
        api_key="not-needed",
        temperature=0.1,
        streaming=False,
    )
    
    response = llm.invoke(messages)
    return response.content

# =============================
# --- 主程序 ---
# =============================

st.set_page_config(page_title="多模态 RAG 聊天机器人", page_icon="🖼️", layout="wide")

st.title("🖼️ 多模态 RAG 聊天机器人")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("支持文本和图像的知识库问答系统")

# 创建多模态 RAG 系统
@st.cache_resource
def load_rag_system():
    return create_multimodal_rag_system(str(KNOWLEDGE_DIR))

rag_system, image_files = load_rag_system()

# 显示加载的信息
col1, col2 = st.columns(2)
with col1:
    st.info(f"📁 文本知识库已加载")
with col2:
    st.info(f"🖼️ 发现 {len(image_files)} 张图像文件")

# 显示图像预览（如果有的话）
if image_files:
    st.subheader("📷 知识库图像预览")
    cols = st.columns(min(3, len(image_files)))
    for i, img_path in enumerate(image_files[:3]):
        try:
            with cols[i % 3]:
                image = Image.open(img_path)
                st.image(image, caption=Path(img_path).name, width=150)
        except Exception as e:
            st.warning(f"无法显示图像 {img_path}")

# =============================
# --- 🔥 缓存监控侧边栏 ---
# =============================
# =============================
# --- 🔥 缓存监控侧边栏（增强版）---
# =============================
with st.sidebar:
    st.divider()
    st.subheader("缓存状态")
    
    if rag_system and "cache" in rag_system:
        cache = rag_system["cache"]
        stats = cache.stats()
        
        # 指标展示
        col1, col2 = st.columns(2)
        with col1:
            st.metric("缓存条目", stats["total_entries"])
        with col2:
            st.metric("累计命中", stats["total_hits"])
        
        # 命中率进度条
        total_req = stats["total_hits"] + stats["total_entries"]
        if total_req > 0:
            hit_rate = stats["total_hits"] / total_req * 100
            st.progress(min(hit_rate / 100, 1.0))
            st.caption(f"估算命中率：{hit_rate:.1f}%")
    else:
        st.caption("缓存未初始化")
    
    # =============================
    # --- 🗑️ 清除功能区域 ---
    # =============================
    st.divider()
    st.subheader("清除功能")
    
    # 按钮 1：清除对话历史
    if st.button("清除对话历史", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.toast("对话历史已清除", icon="🗑️")
        st.rerun()
    
    # 按钮 2：清空缓存
    if rag_system and "cache" in rag_system:
        if st.button("清空缓存", type="secondary", use_container_width=True):
            rag_system["cache"].clear()
            st.toast("缓存数据已清空", icon="♻️")
            st.rerun()
    
    # 按钮 3：完全重置
    if st.button("完全重置系统", type="primary", use_container_width=True):
        # 清除缓存
        if rag_system and "cache" in rag_system:
            rag_system["cache"].clear()
        # 清除对话
        st.session_state.messages = []
        # 提示刷新
        st.toast("系统已重置，建议刷新页面", icon="🔄")
        st.rerun()
    
    # 使用说明
    with st.expander("❓ 清除功能说明", expanded=False):
        st.markdown("""
        **💬 清除对话历史**
        - 仅清空聊天窗口内容
        - 缓存数据保留
        - 适合开始新话题
        
        **🗑️ 清空缓存**
        - 删除所有缓存的问答记录
        - 对话历史保留
        - 适合知识库更新后
        
        **🔄 完全重置系统**
        - 清除缓存 + 对话历史
        - 重新加载知识库
        - 适合系统异常时
        """)

# =============================
# --- 显示历史消息（折叠AI回答版 - 修复版）---
# =============================
last_user_msg = ""  # 维护最近的用户消息（始终为字符串）

for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        # 🔥 提取用户消息文本（确保是字符串）
        content = message["content"]
        if isinstance(content, dict) and "text" in content:
            last_user_msg = content["text"]
        elif isinstance(content, str):
            last_user_msg = content
        else:
            last_user_msg = str(content)  # 兜底转换
        
        # 用户消息 - 始终显示
        with st.chat_message("user"):
            if isinstance(content, dict) and "text" in content:
                text_preview = content["text"]
                st.markdown(f"**❓ {text_preview[:150]}{'...' if len(text_preview) > 150 else ''}**")
                if "images" in content and content["images"]:
                    st.caption(f"📎 附带 {len(content['images'])} 张图像")
            else:
                st.markdown(f"**❓ {content[:150]}{'...' if len(content) > 150 else ''}**")
    
    else:
        # AI 消息 - 使用折叠面板
        with st.chat_message("assistant"):
            # 安全提取标题文本（确保是字符串再调用 .replace()）
            if isinstance(last_user_msg, dict) and "text" in last_user_msg:
                title = last_user_msg.get("text", str(last_user_msg))
            elif isinstance(last_user_msg, str):
                title = last_user_msg
            else:
                title = str(last_user_msg)
            
            # 清理特殊字符
            title = title.replace("\n", " ").replace("#", "").strip()
            expander_title = f"💬 回答：{title[:60]}{'...' if len(title) > 60 else ''}"
            
            # 提取AI回答文本（类型安全）
            ai_content = message["content"]
            if isinstance(ai_content, dict) and "text" in ai_content:
                text = ai_content["text"]
            elif isinstance(ai_content, str):
                text = ai_content
            else:
                text = str(ai_content)
            
            # 判断是否缓存命中
            is_cached = isinstance(text, str) and text.startswith("✅ [缓存]")
            if is_cached:
                expander_title = "🎯 回答（缓存命中 - 秒级响应）"
            
            # 折叠显示AI回答
            with st.expander(expander_title, expanded=False):
                if is_cached:
                    st.markdown("⚡ **来自缓存**")
                    st.markdown("---")
                    st.markdown(text.replace("✅ [缓存] ", ""))
                else:
                    st.markdown(text)

# 用户输入区域
with st.container():
    # 文件上传（可选）
    uploaded_files = st.file_uploader(
        "上传图像文件进行查询（可选）",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True
    )
    
    # 输入框
    prompt = st.chat_input("请输入您的问题...")
    
if prompt:

    if uploaded_files:
        user_message = {
            "role": "user",
            "content": {
                "text": prompt,
                "images": [f.name for f in uploaded_files]
            }
        }
    else:
        user_message = {
            "role": "user",
            "content": prompt
        }
    
    # =============================
    # --- 🔍 显示用户输入内容 ---
    # =============================
    with st.chat_message("user"):
        # 显示问题文本
        st.markdown(f"**❓ 我的问题：** {prompt}")
        
        # 显示上传的图像
        if uploaded_files:
            st.markdown("**📎 上传的图像：**")
            cols = st.columns(min(3, len(uploaded_files)))
            for i, uploaded_file in enumerate(uploaded_files):
                with cols[i % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, width=150)
    
    st.session_state.messages.append(user_message)

    # =============================
    # --- 🔍 显示处理信息（可选）---
    # =============================
    with st.spinner("正在处理您的查询..."):
        # 显示调试信息（可折叠）
        with st.expander("🔍 查看处理详情", expanded=False):
            st.write(f"**输入问题：** `{prompt}`")
            st.write(f"**上传图像数：** `{len(uploaded_files) if uploaded_files else 0}`")
            st.write(f"**知识库图像数：** `{len(image_files)}`")
            st.write(f"**缓存状态：** `{'✅ 已启用' if rag_system and rag_system.get('cache') else '⚠️ 未启用'}`")
        
        # 处理上传的图像
        temp_uploaded_images = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                temp_path = TEMP_UPLOAD_DIR / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_uploaded_images.append(str(temp_path))
            all_images = image_files + temp_uploaded_images
        else:
            all_images = image_files
        
        # 执行查询
        response = multimodal_query_with_images(rag_system, prompt, all_images)
        
        # 清理临时文件
        for temp_img in temp_uploaded_images:
            try:
                os.remove(temp_img)
            except:
                pass
        
        # 显示助手响应
        with st.chat_message("assistant"):
            st.markdown(response)                
                # 添加到历史记录
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)