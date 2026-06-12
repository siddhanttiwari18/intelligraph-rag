import os
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from rag import RAGPipeline
from rag.document_loader import SUPPORTED_EXTENSIONS

load_dotenv()

st.set_page_config(
    page_title="RAG Co-Pilot Pro (Agentic)",
    page_icon="🤖",
    layout="wide",
)

SUPPORTED_LABEL = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))

# Premium Theme and Typography Injections
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .gradient-text {
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #3B82F6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subgradient-text {
        font-weight: 500;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled Doc Cards */
    .doc-card {
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(229, 231, 235, 1);
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
    }
    
    [data-theme="dark"] .doc-card, .dark-theme-card {
        background-color: #1E293B !important;
        border: 1px solid rgba(51, 65, 85, 1) !important;
    }
    
    /* Styled Citation Cards */
    .citation-box {
        border-left: 4px solid #6366F1;
        background-color: rgba(99, 102, 241, 0.06);
        padding: 0.8rem 1.2rem;
        margin: 0.6rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .citation-header {
        font-weight: 600;
        color: #4F46E5;
        margin-bottom: 0.25rem;
    }
    
    [data-theme="dark"] .citation-header {
        color: #818CF8;
    }
    
    .citation-excerpt {
        font-style: italic;
        color: #4B5563;
        margin-top: 0.4rem;
        border-top: 1px dashed rgba(156, 163, 175, 0.2);
        padding-top: 0.4rem;
    }
    
    [data-theme="dark"] .citation-excerpt {
        color: #94A3B8;
    }
    
    .metric-badge {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.15);
        color: #4F46E5;
        padding: 0.15rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    
    [data-theme="dark"] .metric-badge {
        background-color: rgba(129, 140, 248, 0.2);
        color: #A5B4FC;
    }
    
    .table-badge {
        background-color: rgba(236, 72, 153, 0.15) !important;
        color: #DB2777 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_session_state(pipeline: RAGPipeline) -> None:
    # Sync ingested documents metadata
    st.session_state.ingested_files = pipeline.get_documents_metadata()
    if "reindex_target" not in st.session_state:
        st.session_state.reindex_target = None
    if "active_ingest_tasks" not in st.session_state:
        st.session_state.active_ingest_tasks = []

    # Load session listing
    sessions = pipeline.session_manager.list_sessions()
    
    if "active_session_id" not in st.session_state:
        if sessions:
            st.session_state.active_session_id = sessions[0]["session_id"]
        else:
            new_sess = pipeline.session_manager.create_session("New Chat")
            st.session_state.active_session_id = new_sess["session_id"]

    # Load active session messages
    active_session = pipeline.session_manager.load_session(st.session_state.active_session_id)
    if active_session:
        st.session_state.messages = active_session["messages"]
        st.session_state.active_session_title = active_session["title"]
    else:
        # Fallback if the active session ID was somehow removed
        new_sess = pipeline.session_manager.create_session("New Chat")
        st.session_state.active_session_id = new_sess["session_id"]
        st.session_state.messages = []
        st.session_state.active_session_title = "New Chat"


def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.header("📥 Ingestion Hub")
        st.caption(f"Supported formats: **{SUPPORTED_LABEL.upper()}**")

        uploaded_files = st.file_uploader(
            "Upload text documents",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            key="main_uploader",
        )

        if st.button("Index Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one document first.")
            else:
                from rag.services.background_worker import background_worker
                for uploaded in uploaded_files:
                    task_id = background_worker.submit_task(
                        pipeline.ingest_file,
                        uploaded.name,
                        uploaded.getvalue(),
                        description=f"Ingesting {uploaded.name}"
                    )
                    st.session_state.active_ingest_tasks.append(task_id)
                st.success(f"Submitted {len(uploaded_files)} document(s) for background indexing!")
                st.rerun()

        st.divider()

        # Centralized Agentic RAG Configuration Controls
        st.subheader("⚙️ Agentic Config")
        conf = pipeline.config

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=float(conf.get("confidence_threshold", 0.6)),
            step=0.05,
            help="Minimum evidence confidence before triggering sub-question planning.",
        )
        max_retrieval_iterations = st.slider(
            "Max Retrieval Iterations",
            min_value=1,
            max_value=5,
            value=int(conf.get("max_retrieval_iterations", 2)),
            step=1,
            help="Maximum search loops to execute to find missing information.",
        )
        max_planner_depth = st.slider(
            "Max Sub-Questions",
            min_value=1,
            max_value=5,
            value=int(conf.get("max_planner_depth", 3)),
            step=1,
            help="Maximum sub-questions generated by the planner during one loop.",
        )
        max_retrieved_chunks = st.slider(
            "Max Chunks",
            min_value=5,
            max_value=50,
            value=int(conf.get("max_retrieved_chunks", 20)),
            step=5,
            help="Maximum total context chunks retrieved.",
        )
        agent_trace_visibility = st.toggle(
            "Display Agent Trace",
            value=bool(conf.get("agent_trace_visibility", True)),
            help="Show the step-by-step execution path of the RAG agent in the chat.",
        )
        max_depth = st.slider(
            "Graph Traversal Depth",
            min_value=1,
            max_value=5,
            value=int(conf.get("max_depth", 2)),
            step=1,
            help="Maximum depth limit for BFS traversal starting from query nodes.",
        )
        graph_enabled = st.toggle(
            "Enable Knowledge Graph Layer",
            value=bool(conf.get("graph_enabled", True)),
            help="Toggle to use the Knowledge Graph retrieval layer.",
        )

        pipeline.set_config({
            "confidence_threshold": confidence_threshold,
            "max_retrieval_iterations": max_retrieval_iterations,
            "max_planner_depth": max_planner_depth,
            "max_retrieved_chunks": max_retrieved_chunks,
            "agent_trace_visibility": agent_trace_visibility,
            "max_depth": max_depth,
            "graph_enabled": graph_enabled,
        })

        st.divider()

        # Conversational History Sidebar Section
        st.subheader("💬 Chat Sessions")
        if st.button("➕ New Chat Session", use_container_width=True):
            new_sess = pipeline.session_manager.create_session("New Chat")
            st.session_state.active_session_id = new_sess["session_id"]
            st.session_state.messages = []
            st.session_state.active_session_title = "New Chat"
            st.rerun()

        sessions = pipeline.session_manager.list_sessions()
        if sessions:
            st.caption("Recent Chats:")
            for s in sessions:
                is_active = (s["session_id"] == st.session_state.active_session_id)
                btn_label = f"💬 {s['title']}" if not is_active else f"👉 {s['title']}"
                
                col_title, col_del = st.columns([5, 1])
                with col_title:
                    # Session select button
                    if st.button(btn_label, key=f"sess_{s['session_id']}", use_container_width=True):
                        st.session_state.active_session_id = s["session_id"]
                        sess_data = pipeline.session_manager.load_session(s["session_id"])
                        st.session_state.messages = sess_data["messages"]
                        st.session_state.active_session_title = sess_data["title"]
                        st.rerun()
                with col_del:
                    # Session delete button
                    if st.button("🗑️", key=f"del_sess_{s['session_id']}", use_container_width=True):
                        pipeline.session_manager.delete_session(s["session_id"])
                        if s["session_id"] == st.session_state.active_session_id:
                            st.session_state.pop("active_session_id", None)
                        st.rerun()
        else:
            st.caption("No saved chat history.")

        st.divider()
        col_clear, col_stat = st.columns([1, 1])
        with col_clear:
            if st.button("Clear Active Chat", use_container_width=True):
                pipeline.session_manager.clear_session(st.session_state.active_session_id)
                st.session_state.messages = []
                st.success("Active chat history cleared.")
                st.rerun()
        with col_stat:
            stats = pipeline.get_stats()
            st.metric("Total Indexed Chunks", stats["indexed_chunks"])

        if st.button("Clear Vector Index", use_container_width=True, type="secondary"):
            pipeline.clear_index()
            st.session_state.ingested_files = []
            st.session_state.messages = []
            st.session_state.reindex_target = None
            st.success("All indexes cleared successfully.")
            st.rerun()

        # 1. System Diagnostics Check Panel
        from rag.config import run_diagnostics
        diagnostics = run_diagnostics(pipeline.settings)
        
        st.divider()
        st.subheader("🛠️ Diagnostics Panel")
        with st.expander("Show System Diagnostics", expanded=False):
            if diagnostics["errors"]:
                st.error("🚨 Configuration Errors:")
                for err in diagnostics["errors"]:
                    st.write(f"- {err}")
            elif diagnostics["warnings"]:
                st.warning("⚠️ Configuration Warnings:")
                for warn in diagnostics["warnings"]:
                    st.write(f"- {warn}")
            else:
                st.success("🟢 System Diagnostics Healthy!")
            
            st.markdown(f"**Storage Path:** `{diagnostics['storage_dir_path']}`")
            api_status = "Configured" if diagnostics["deepseek_api_key_configured"] else "Not Configured"
            st.markdown(f"**DeepSeek API:** `{api_status}`")

        # 2. Background Ingestion Tasks List
        from rag.services.background_worker import background_worker
        
        if st.session_state.get("active_ingest_tasks"):
            st.divider()
            st.subheader("⏳ Background Tasks")
            
            has_running = False
            for t_id in list(st.session_state.active_ingest_tasks):
                status = background_worker.get_task_status(t_id)
                if not status:
                    continue
                
                desc = status["description"]
                state = status["status"]
                prog = status["progress"]
                msg = status["message"]
                
                with st.container(border=True):
                    st.markdown(f"**{desc}**")
                    st.caption(f"Status: `{state}` | {msg}")
                    if state in ("PENDING", "RUNNING"):
                        st.progress(prog)
                        has_running = True
                    elif state == "COMPLETED":
                        st.success("Completed successfully!")
                    else:
                        st.error(f"Failed: {status.get('error', 'Unknown error')}")
            
            if st.button("🧹 Clear Task History", use_container_width=True):
                running_tasks = []
                for t_id in st.session_state.active_ingest_tasks:
                    status = background_worker.get_task_status(t_id)
                    if status and status["status"] in ("PENDING", "RUNNING"):
                        running_tasks.append(t_id)
                st.session_state.active_ingest_tasks = running_tasks
                st.session_state.ingested_files = pipeline.get_documents_metadata()
                st.rerun()
                
            if has_running:
                import time
                time.sleep(1.0)
                st.rerun()


def render_doc_manager(pipeline: RAGPipeline) -> None:
    st.subheader("📂 Document Database Management")
    st.markdown("Monitor chunk statistics, manage uploads, and reindex or remove files from the indices.")

    # Show reindex sub-uploader if active
    if st.session_state.reindex_target:
        target = st.session_state.reindex_target
        st.info(f"🔄 Reindexing **{target}**")
        reindex_file = st.file_uploader(
            f"Select new version file for {target}",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            key="reindex_file_uploader",
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Confirm Reindex Upload", type="primary", use_container_width=True):
                if reindex_file:
                    from rag.services.background_worker import background_worker
                    task_id = background_worker.submit_task(
                        pipeline.reindex_document,
                        target,
                        reindex_file.getvalue(),
                        description=f"Reindexing {target}"
                    )
                    st.session_state.active_ingest_tasks.append(task_id)
                    st.session_state.reindex_target = None
                    st.success(f"Submitted reindexing task for {target} in the background!")
                    st.rerun()
                else:
                    st.error("Please upload a file.")
        with col_c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.reindex_target = None
                st.rerun()
        st.divider()

    docs = st.session_state.ingested_files
    if not docs:
        st.info("No documents are currently indexed in the application database.")
        return

    # Document management grid
    for idx, doc in enumerate(docs):
        # We wrap in cols to render data + action buttons inline
        col_meta, col_del, col_reidx = st.columns([6, 1, 1])
        
        with col_meta:
            doc_type_icon = "📄" if doc["document_type"] == ".pdf" else "📝"
            pdf_type_label = doc.get("pdf_type", "Digital Document")
            st.markdown(
                f"""
                <div class="doc-card dark-theme-card">
                    <strong>{doc_type_icon} {doc['filename']}</strong> ({pdf_type_label})<br/>
                    <small>📅 Uploaded: {doc['upload_date']} | 🧩 Chunks: {doc['chunks']} | 🔤 Characters: {doc['characters']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_del:
            # Add unique key to avoid Streamlit conflicts
            if st.button("Delete", key=f"del_{idx}_{doc['filename']}", use_container_width=True, type="secondary"):
                with st.spinner("Removing document chunks..."):
                    pipeline.delete_document(doc["filename"])
                    st.session_state.ingested_files = pipeline.get_documents_metadata()
                    st.success(f"Removed **{doc['filename']}** from database.")
                    st.rerun()

        with col_reidx:
            if st.button("Reindex", key=f"reidx_{idx}_{doc['filename']}", use_container_width=True):
                st.session_state.reindex_target = doc["filename"]
                st.rerun()

    st.divider()
    st.markdown("### 🔍 Document Inspector")
    doc_names = [d["filename"] for d in docs]
    inspect_target = st.selectbox("Select a document to inspect:", ["-- Select Document --"] + doc_names)
    
    if inspect_target != "-- Select Document --":
        doc_meta = next((d for d in docs if d["filename"] == inspect_target), None)
        doc_chunks = [c for c in pipeline.vector_store.chunks if c["filename"] == inspect_target]
        
        # Details tabs
        sub_tab_chunks, sub_tab_ocr, sub_tab_tables = st.tabs(["🧩 Chunk Viewer", "👁️ OCR Results", "📊 Table Extractions"])
        
        with sub_tab_chunks:
            st.markdown(f"Total Chunks: `{len(doc_chunks)}`")
            for c_idx, c in enumerate(doc_chunks):
                with st.expander(f"Chunk #{c_idx+1} (Page {c.get('page_number', 1)}) - Ref: `{c['chunk_id']}`"):
                    st.code(c["text"])
                    st.caption(f"Chars: {len(c['text'])} | Type: {c.get('chunk_type', 'text').upper()}")
                    
        with sub_tab_ocr:
            is_scanned = doc_meta.get("pdf_type") == "Scanned PDF (OCR)" if doc_meta else False
            st.markdown(f"OCR Processed: `{is_scanned}`")
            if is_scanned:
                # Compile text page by page
                text_pages = {}
                for c in doc_chunks:
                    p_num = c.get("page_number", 1)
                    if c.get("chunk_type") != "table":
                        text_pages.setdefault(p_num, []).append(c["text"])
                
                for p_num, texts in sorted(text_pages.items()):
                    st.markdown(f"**Page {p_num} OCR Text:**")
                    st.text_area(label="", value="\n".join(texts), height=150, key=f"ocr_p_{p_num}", disabled=True)
            else:
                st.caption("This document was not processed using the OCR pipeline (digital or text document).")
                
        with sub_tab_tables:
            table_chunks = [c for c in doc_chunks if c.get("chunk_type") == "table"]
            st.markdown(f"Extracted Tables: `{len(table_chunks)}`")
            if table_chunks:
                for t_idx, tc in enumerate(table_chunks):
                    with st.expander(f"Table #{t_idx+1} (Page {tc.get('page_number', 1)})"):
                        st.markdown(tc["text"])
                        if tc.get("table_columns"):
                            st.caption(f"Detected Columns: {', '.join(tc['table_columns'])}")
            else:
                st.caption("No tables were extracted from this document.")


def render_citations(sources: list[dict]) -> None:
    with st.expander("🔍 Citations & Retrieval Evaluation"):
        for idx, source in enumerate(sources, start=1):
            chunk_type = source.get("chunk_type", "text")
            type_class = "metric-badge table-badge" if chunk_type == "table" else "metric-badge"
            
            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="citation-header">
                        [{idx}] Source: {source['source']} (Page {source['page']})
                        <span style="float: right; font-weight: 500; font-size: 0.8rem; color: #4F46E5;">Rank #{source.get('source_rank', 1)}</span>
                    </div>
                    <code>Ref ID: {source['chunk_ref']}</code> | Type: <span class="{type_class}">{chunk_type.upper()}</span>
                    
                    <div style="margin-top: 0.4rem; margin-bottom: 0.4rem; font-size: 0.8rem;">
                        <span class="metric-badge">Confidence Score: {source.get('fused_score', 0.0):.2f}</span>
                        <span class="metric-badge">Semantic (FAISS): {source.get('semantic_score', 0.0):.2f}</span>
                        <span class="metric-badge">Keyword (BM25): {source.get('bm25_score', 0.0):.2f}</span>
                        <span class="metric-badge">Cross-Encoder: {source.get('cross_score', 0.0):.2f}</span>
                    </div>
                    <div class="citation-excerpt">"{source['excerpt']}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_agent_trace(trace: dict) -> None:
    if not trace:
        return

    q_type = trace.get("query_type", "Simple")
    pill_color = "#10B981" if q_type == "Simple" else "#8B5CF6"

    with st.expander("🕵️ Agent Execution Trace"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"Query Type: <span style='background-color: {pill_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;'>{q_type}</span>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(f"Confidence Assessment: **{trace.get('confidence_assessment', 1.0) * 100:.0f}%**")
        with col3:
            st.markdown(f"Retrieval Iterations: **{trace.get('retrieval_iterations', 0)} loops**")

        sub_qs = trace.get("sub_questions", [])
        if sub_qs:
            st.markdown("🎯 **Sub-questions Generated by Planner:**")
            for sq in sub_qs:
                st.markdown(f"- {sq}")

        docs = trace.get("documents_consulted", [])
        if docs:
            st.markdown(f"🗂️ **Documents Consulted:** {', '.join(docs)}")

        logs = trace.get("logs", [])
        if logs:
            st.markdown("📝 **Orchestrator Execution Logs:**")
            logs_html = "".join([f"<div style='margin-bottom: 4px;'>➜ {log}</div>" for log in logs])
            st.markdown(
                f"""
                <div style="background-color: rgba(99, 102, 241, 0.04); padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.8rem; border: 1px solid rgba(99, 102, 241, 0.1); max-height: 200px; overflow-y: auto;">
                    {logs_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat(pipeline: RAGPipeline) -> None:
    st.markdown('<div class="gradient-text">RAG Co-Pilot Pro (Agentic)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subgradient-text">State-Graph Adaptive Retrieval & Query Classification</div>', unsafe_allow_html=True)

    # Chat messages display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_citations(message["sources"])
            # Render trace logs if available and configured visible
            if message.get("agent_trace") and pipeline.config.get("agent_trace_visibility", True):
                render_agent_trace(message["agent_trace"])

    # User input
    if question := st.chat_input("Ask a question about your documents..."):
        if not os.getenv("DEEPSEEK_API_KEY"):
            st.error("Set DEEPSEEK_API_KEY in a `.env` file before asking questions.")
            return

        if len(st.session_state.ingested_files) == 0:
            st.warning("Please upload and index at least one document before querying.")
            return

        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Executing Agentic RAG workflow (Classifier -> Evaluator)..."):
                try:
                    # Pass sliding window history
                    result = pipeline.ask(question, history=st.session_state.messages[:-1])
                    st.markdown(result["answer"])
                    
                    if result["sources"]:
                        render_citations(result["sources"])
                        
                    if result.get("agent_trace") and pipeline.config.get("agent_trace_visibility", True):
                        render_agent_trace(result["agent_trace"])
                except Exception as exc:
                    result = {"answer": f"Error: {exc}", "sources": [], "agent_trace": {}}
                    st.error(result["answer"])

        # Append assistant response to active list
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "agent_trace": result.get("agent_trace", {}),
            }
        )

        # Chat Thread Auto-naming
        new_title = st.session_state.active_session_title
        if new_title == "New Chat" and len(st.session_state.messages) >= 2:
            new_title = question[:25] + ("..." if len(question) > 25 else "")
            st.session_state.active_session_title = new_title

        # Persist session to local disk
        pipeline.session_manager.save_session(
            st.session_state.active_session_id,
            title=new_title,
            messages=st.session_state.messages,
        )


def render_graph_explorer(pipeline: RAGPipeline) -> None:
    import json
    import networkx as nx
    st.subheader("📊 Knowledge Graph Explorer")
    st.markdown("Inspect extracted entities, trace system connections, and view document relationship lineages.")

    # Get nodes and edges
    nodes = pipeline.graph_store.get_nodes()
    edges = pipeline.graph_store.get_edges()

    if not nodes:
        st.info("No entities are currently in the Knowledge Graph. Please upload and index documents containing relationships or click 'Rebuild' below if you have documents indexed.")
        if st.button("🔄 Rebuild Knowledge Graph", use_container_width=True):
            from rag.services.background_worker import background_worker
            task_id = background_worker.submit_task(
                pipeline.rebuild_graph,
                description="Rebuilding Knowledge Graph"
            )
            st.session_state.active_ingest_tasks.append(task_id)
            st.success("Rebuilding task submitted to background!")
            st.rerun()
        return

    # Color definitions matching our theme
    ENTITY_COLORS = {
        "person": "#3B82F6",       # Blue
        "team": "#6366F1",         # Indigo
        "organization": "#06B6D4", # Cyan
        "project": "#8B5CF6",      # Violet
        "service": "#10B981",      # Emerald
        "application": "#14B8A6",  # Teal
        "technology": "#F97316",   # Orange
        "api": "#EC4899",          # Pink
        "database": "#F43F5E",     # Rose
        "protocol": "#F59E0B",     # Amber
        "incident": "#EF4444",     # Red
        "location": "#84CC16",     # Lime
        "unknown": "#6B7280",      # Gray
    }

    # Count node types
    type_counts = {}
    for node in nodes:
        t = node["entity_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Show entity badges and Rebuild button at the top
    col_badges, col_rebuild = st.columns([3, 1])
    with col_badges:
        badge_html = ""
        for etype, count in sorted(type_counts.items()):
            color = ENTITY_COLORS.get(etype.lower(), "#6B7280")
            badge_html += f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; margin-right: 6px; display: inline-block; margin-bottom: 6px;">{etype}: {count}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)
    with col_rebuild:
        if st.button("🔄 Rebuild Knowledge Graph", use_container_width=True, key="rebuild_graph_top", help="Re-extract relationships and rebuild the entire graph from scratch using indexed documents."):
            from rag.services.background_worker import background_worker
            task_id = background_worker.submit_task(
                pipeline.rebuild_graph,
                description="Rebuilding Knowledge Graph"
            )
            st.session_state.active_ingest_tasks.append(task_id)
            st.success("Rebuilding task submitted to background!")
            st.rerun()

    st.divider()

    # Define 3 sub-tabs
    g_tab_visual, g_tab_search, g_tab_topology = st.tabs([
        "🕸️ Visual Connections Graph", 
        "🔍 Relationship Search", 
        "📈 Topology Stats"
    ])

    with g_tab_visual:
        col_filters, col_search = st.columns([1, 1])

        # Filter Graph View
        with col_filters:
            all_types = sorted(list(type_counts.keys()))
            selected_types = st.multiselect(
                "🏷️ Filter by Entity Type",
                options=all_types,
                default=all_types,
                key="visual_type_filters",
                help="Select which entity types to display in the visual graph."
            )
            hide_isolated = st.checkbox(
                "Hide Isolated Nodes", 
                value=True, 
                key="visual_hide_isolated",
                help="Hide nodes that do not have any active relationships in the current filtered graph view."
            )

        # Search Box for Entity Inspecting
        with col_search:
            node_ids = sorted([node["id"] for node in nodes])
            searched_entity = st.selectbox(
                "🔍 Inspect Entity Details",
                options=["-- Select an Entity --"] + node_ids,
                key="visual_entity_inspect",
                help="Select an entity to view its incoming/outgoing edges and source text citations."
            )

        # Filter nodes/edges based on multiselect filter
        filtered_nodes = [n for n in nodes if n["entity_type"] in selected_types]
        filtered_node_ids = {n["id"] for n in filtered_nodes}
        filtered_edges = [
            e for e in edges 
            if e["source"] in filtered_node_ids and e["target"] in filtered_node_ids
        ]

        # Filter out isolated nodes if checkbox is active
        if hide_isolated:
            node_degrees = {n["id"]: 0 for n in filtered_nodes}
            for edge in filtered_edges:
                if edge["source"] in node_degrees:
                    node_degrees[edge["source"]] += 1
                if edge["target"] in node_degrees:
                    node_degrees[edge["target"]] += 1
            filtered_nodes = [n for n in filtered_nodes if node_degrees[n["id"]] > 0]

        # Render inspected entity details
        if searched_entity != "-- Select an Entity --":
            node_obj = next((n for n in nodes if n["id"] == searched_entity), None)
            if node_obj:
                st.markdown(f"### ℹ️ Entity: `{searched_entity}`")
                aliases_str = ", ".join(f"`{a}`" for a in node_obj.get("aliases", [])) if node_obj.get("aliases") else "None"
                st.markdown(
                    f"**Canonical Name**: `{node_obj.get('canonical_name', searched_entity)}` | "
                    f"**Entity Type**: `{node_obj['entity_type']}` | "
                    f"**Confidence**: `{node_obj.get('confidence', 1.0):.2f}`\n\n"
                    f"**Aliases**: {aliases_str}"
                )
                
                # Incoming / Outgoing relationships
                out_rels = [e for e in edges if e["source"] == searched_entity]
                in_rels = [e for e in edges if e["target"] == searched_entity]
                
                col_in, col_out = st.columns(2)
                
                with col_in:
                    st.markdown("**Incoming Connections:**")
                    if in_rels:
                        for r in in_rels:
                            st.markdown(f"- `{r['source']}` ➔ `{r['relation_type']}` ➔ `{searched_entity}` (Confidence: `{r.get('confidence', 1.0):.2f}`)")
                    else:
                        st.caption("No incoming connections.")
                        
                with col_out:
                    st.markdown("**Outgoing Connections:**")
                    if out_rels:
                        for r in out_rels:
                            st.markdown(f"- `{searched_entity}` ➔ `{r['relation_type']}` ➔ `{r['target']}` (Confidence: `{r.get('confidence', 1.0):.2f}`)")
                    else:
                        st.caption("No outgoing connections.")
                
                # Source Citations
                st.markdown("**Source Citations & Excerpts:**")
                sources = []
                seen_source_refs = set()
                
                for s in node_obj.get("sources", []):
                    ref = s.get("chunk_ref")
                    if ref and ref not in seen_source_refs:
                        seen_source_refs.add(ref)
                        sources.append(s)
                
                for r in in_rels + out_rels:
                    for s in r.get("sources", []):
                        ref = s.get("chunk_ref")
                        if ref and ref not in seen_source_refs:
                            seen_source_refs.add(ref)
                            sources.append(s)
                
                if sources:
                    for idx, src in enumerate(sources, start=1):
                        doc_name = src.get("source_document", "unknown")
                        page_num = src.get("page_number", 1)
                        chunk_ref = src.get("chunk_ref", "")
                        
                        excerpt_text = "No excerpt text stored."
                        if chunk_ref:
                            match = next((c for c in pipeline.vector_store.chunks if c["chunk_id"] == chunk_ref), None)
                            if match:
                                excerpt_text = match["text"]
                                
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <div class="citation-header">
                                    [{idx}] Source: {doc_name} (Page {page_num})
                                </div>
                                <code>Ref ID: {chunk_ref}</code>
                                <div class="citation-excerpt">"{excerpt_text}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No source citations found.")
                st.divider()

        # Draw the Vis.js visual network
        if not filtered_nodes:
            st.warning("No nodes match the selected entity types.")
        else:
            vis_nodes = []
            for node in filtered_nodes:
                color = ENTITY_COLORS.get(node["entity_type"].lower(), "#6B7280")
                aliases_tooltip = ", ".join(node.get("aliases", [])) if node.get("aliases") else "None"
                vis_nodes.append({
                    "id": node["id"],
                    "label": node["id"],
                    "title": f"Canonical: {node.get('canonical_name', node['id'])}<br>Type: {node['entity_type']}<br>Confidence: {node.get('confidence', 1.0):.2f}<br>Aliases: {aliases_tooltip}",
                    "shape": "box",
                    "color": {
                        "background": color,
                        "border": color,
                        "highlight": {
                            "background": color,
                            "border": "#22C55E"
                        }
                    },
                    "font": {
                        "color": "#FFFFFF",
                        "face": "Plus Jakarta Sans, sans-serif"
                    }
                })

            vis_edges = []
            for edge in filtered_edges:
                vis_edges.append({
                    "from": edge["source"],
                    "to": edge["target"],
                    "label": edge["relation_type"],
                    "title": f"Relation: {edge['relation_type']}<br>Confidence: {edge.get('confidence', 1.0):.2f}"
                })

            nodes_json = json.dumps(vis_nodes)
            edges_json = json.dumps(vis_edges)

            html_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    html, body {{
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100%;
                        overflow: hidden;
                        background-color: #0F172A; /* Match dark slate background */
                    }}
                    #network {{
                        width: 100%;
                        height: 100%;
                    }}
                </style>
            </head>
            <body>
                <div id="network"></div>
                <script type="text/javascript">
                    var nodes = new vis.DataSet({nodes_json});
                    var edges = new vis.DataSet({edges_json});
                    var container = document.getElementById('network');
                    var data = {{
                        nodes: nodes,
                        edges: edges
                    }};
                    var options = {{
                        nodes: {{
                            margin: 10,
                            font: {{
                                size: 14,
                                face: 'Plus Jakarta Sans, sans-serif'
                            }}
                        }},
                        edges: {{
                            arrows: {{
                                to: {{ enabled: true, scaleFactor: 0.8 }}
                            }},
                            color: {{
                                color: '#475569',
                                highlight: '#818CF8',
                                hover: '#94A3B8'
                            }},
                            font: {{
                                size: 11,
                                align: 'middle',
                                color: '#94A3B8',
                                face: 'Plus Jakarta Sans, sans-serif',
                                strokeWidth: 0
                            }},
                            smooth: {{
                                type: 'cubicBezier',
                                forceDirection: 'none',
                                roundness: 0.4
                            }}
                        }},
                        physics: {{
                            barnesHut: {{
                                gravitationalConstant: -1800,
                                centralGravity: 0.3,
                                springLength: 95,
                                springConstant: 0.04,
                                damping: 0.09,
                                avoidOverlap: 0.1
                            }},
                            maxVelocity: 50,
                            minVelocity: 0.1,
                            solver: 'barnesHut',
                            stabilization: {{
                                enabled: true,
                                iterations: 500,
                                updateInterval: 100,
                                fit: true
                            }}
                        }},
                        interaction: {{
                            hover: true,
                            tooltipDelay: 100,
                            selectable: true
                        }}
                    }};
                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """
            components.html(html_code, height=550)

    with g_tab_search:
        st.markdown("### 🔍 Search Relationships")
        st.markdown("Search relationships by relation type or look up connected entities.")

        # Get unique relationship types
        rel_types = sorted(list({e["relation_type"] for e in edges}))
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            selected_rel_types = st.multiselect(
                "Filter by Relationship Type",
                options=rel_types,
                default=rel_types,
                key="search_rel_types_filter"
            )
        with col_s2:
            search_query = st.text_input(
                "Search Entity Names (Source or Target)",
                value="",
                placeholder="e.g. Kafka, Portal, Service...",
                key="search_rel_text_query"
            )

        # Filter edges based on criteria
        matching_edges = []
        for edge in edges:
            if selected_rel_types and edge["relation_type"] not in selected_rel_types:
                continue
            if search_query:
                q = search_query.lower()
                if q not in edge["source"].lower() and q not in edge["target"].lower():
                    continue
            matching_edges.append(edge)

        if matching_edges:
            # Format results in a dataframe
            records = []
            for edge in matching_edges:
                records.append({
                    "Source Entity": edge["source"],
                    "Relationship": edge["relation_type"],
                    "Target Entity": edge["target"],
                    "Confidence": round(edge.get("confidence", 1.0), 2)
                })
            
            df_matching = pd.DataFrame(records)
            st.dataframe(df_matching, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(matching_edges)} matching relationships.")
        else:
            st.info("No relationships found matching the search criteria.")

    with g_tab_topology:
        st.markdown("### 📈 Network Topology & Metrics")
        
        # Calculate connected components, average degree, and orphan nodes
        G = pipeline.graph_store.graph
        try:
            num_components = nx.number_weakly_connected_components(G)
        except Exception:
            num_components = 0
        orphan_nodes = len([n for n in G.nodes() if G.degree(n) == 0])
        avg_degree = (2.0 * len(edges) / len(nodes)) if len(nodes) > 0 else 0.0
        
        density = 0.0
        if len(nodes) > 1:
            density = len(edges) / (len(nodes) * (len(nodes) - 1))

        # Topology metrics row
        cols_metrics = st.columns(6)
        with cols_metrics[0]:
            st.metric("Total Nodes", len(nodes))
        with cols_metrics[1]:
            st.metric("Total Edges", len(edges))
        with cols_metrics[2]:
            st.metric("Avg Degree", f"{avg_degree:.2f}")
        with cols_metrics[3]:
            st.metric("Density", f"{density:.4f}")
        with cols_metrics[4]:
            st.metric("Orphan Nodes", orphan_nodes)
        with cols_metrics[5]:
            st.metric("Connected Components", num_components)

        st.divider()

        # Display Top Connected Nodes (Most Connected)
        st.markdown("#### 🏆 Most Connected Entities (Highest Degree)")
        
        # Sort nodes by degree
        degrees = list(G.degree())
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        # Get top entities metadata
        top_records = []
        for node_id, degree in degrees[:15]:
            node_obj = next((n for n in nodes if n["id"] == node_id), None)
            entity_type = node_obj["entity_type"] if node_obj else "unknown"
            canonical_name = node_obj.get("canonical_name", node_id) if node_obj else node_id
            top_records.append({
                "Entity ID": node_id,
                "Canonical Name": canonical_name,
                "Entity Type": entity_type,
                "Connections (Degree)": degree
            })
        
        if top_records:
            col_t_table, col_t_chart = st.columns([1, 1])
            df_top = pd.DataFrame(top_records)
            with col_t_table:
                st.dataframe(df_top, use_container_width=True, hide_index=True)
            with col_t_chart:
                # Plot Degree distribution of Top 10
                df_chart = df_top.head(10).set_index("Entity ID")
                st.bar_chart(df_chart["Connections (Degree)"])
        else:
            st.caption("No topology data available.")


def render_analytics_dashboard(pipeline: RAGPipeline) -> None:
    st.subheader("📈 Enterprise Analytics & Observability Dashboard")
    st.markdown("Monitor the operation, cost, accuracy, latency, and performance of the RAG pipeline.")

    # Fetch Cached Metrics
    metrics = pipeline.telemetry.get_metrics(pipeline)
    
    # Process queries dataframe for charts
    query_events = []
    for e in pipeline.telemetry.events:
        if e["event_type"] == "query":
            d = e["data"].copy()
            d["timestamp"] = e["timestamp"]
            d["time_formatted"] = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
            d["date"] = time.strftime("%Y-%m-%d", time.localtime(e["timestamp"]))
            query_events.append(d)
    
    df_queries = pd.DataFrame(query_events)

    # Let's render the sub-tabs
    tab_kpis, tab_perf, tab_agent_graph, tab_resources, tab_obs = st.tabs([
        "📊 Overview & KPIs",
        "⚡ Performance & Latency",
        "⚙️ Agent & Graph Analytics",
        "💾 Resource & Cost Insights",
        "🔍 Observability & Error Logs"
    ])

    with tab_kpis:
        # Metrics row
        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            st.metric("Total Documents", len(st.session_state.ingested_files))
        with kpi_cols[1]:
            st.metric("Total Indexed Chunks", pipeline.get_stats()["indexed_chunks"])
        with kpi_cols[2]:
            st.metric("Total Queries Run", metrics["total_queries"])
        with kpi_cols[3]:
            st.metric("Agent Executions", metrics["total_agent_executions"])
        with kpi_cols[4]:
            st.metric("Graph Queries", metrics["total_graph_queries"])
        with kpi_cols[5]:
            sessions_count = len(pipeline.session_manager.list_sessions())
            st.metric("Conversations", sessions_count)

        st.divider()

        # Query Volume Line Chart & Token Cost card
        col_vol, col_cost = st.columns([2, 1])
        with col_vol:
            st.markdown("#### 📅 Query Volume Over Time")
            if metrics["daily_volume"]:
                df_daily = pd.DataFrame(metrics["daily_volume"], columns=["Date", "Query Count"]).set_index("Date")
                st.line_chart(df_daily, height=220)
            else:
                st.info("No query volume data recorded yet.")
        with col_cost:
            st.markdown("#### 🪙 Cumulative Costs & Projections")
            active_prov = metrics["active_pricing"]["provider_name"]
            
            with st.container(border=True):
                st.markdown(f"🔌 **Active Provider:** `{active_prov}`")
                st.divider()
                
                col_metric_left, col_metric_right = st.columns(2)
                with col_metric_left:
                    st.metric("Input Tokens", f"{metrics['total_input_tokens']:,}")
                    st.metric("Today's Cost", f"${metrics['daily_cost']:.4f}")
                with col_metric_right:
                    st.metric("Output Tokens", f"{metrics['total_output_tokens']:,}")
                    st.metric("Cumulative Cost", f"${metrics['total_ai_cost']:.4f}")
                
                st.divider()
                st.metric("Total Tokens Consumed", f"{metrics['total_tokens_consumed']:,}")
                st.metric("Monthly Cost Projection", f"${metrics['monthly_cost_projection']:.4f}")

        st.divider()

        # Bottom row of top lists
        col_top_ent, col_top_docs, col_top_tech = st.columns(3)
        with col_top_ent:
            st.markdown("#### 🏷️ Top Searched Entities")
            if metrics["top_entities"]:
                df_top_ent = pd.DataFrame(metrics["top_entities"], columns=["Entity ID", "Search Count"])
                st.dataframe(df_top_ent, use_container_width=True, hide_index=True)
            else:
                st.caption("No entity search history recorded.")
        with col_top_docs:
            st.markdown("#### 📄 Most Accessed Documents")
            if metrics["top_documents"]:
                df_top_docs = pd.DataFrame(metrics["top_documents"], columns=["Document", "Access Count"])
                st.dataframe(df_top_docs, use_container_width=True, hide_index=True)
            else:
                st.caption("No document access history recorded.")
        with col_top_tech:
            st.markdown("#### 💻 Top Technology References")
            if metrics["top_technologies"]:
                df_top_tech = pd.DataFrame(metrics["top_technologies"], columns=["Technology", "Mentions"])
                st.dataframe(df_top_tech, use_container_width=True, hide_index=True)
            else:
                st.caption("No technology references recorded.")

    with tab_perf:
        st.markdown("### ⚡ Response Latency & Retrieval Performance")
        
        # Performance KPIs Row
        perf_cols = st.columns(4)
        with perf_cols[0]:
            st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
        with perf_cols[1]:
            st.metric("P95 Latency", f"{metrics['p95_response_time']:.2f}s")
        with perf_cols[2]:
            st.metric("P99 Latency", f"{metrics['p99_response_time']:.2f}s")
        with perf_cols[3]:
            st.metric("Retrieval Success Rate", f"{metrics['retrieval_success_rate']:.1f}%")

        st.divider()

        # Detailed timing timeline (if query logs exist)
        if len(df_queries) > 0:
            st.markdown("#### 📈 Timeline: Response Latency Components")
            df_latencies_timeline = df_queries[["time_formatted", "total_response_time", "retrieval_latency", "graph_retrieval_latency", "re_ranking_time", "embedding_time"]].copy()
            df_latencies_timeline = df_latencies_timeline.rename(columns={
                "total_response_time": "Total Response",
                "retrieval_latency": "Vector Retrieval",
                "graph_retrieval_latency": "Graph Retrieval",
                "re_ranking_time": "Re-ranking",
                "embedding_time": "Embedding"
            })
            df_latencies_timeline = df_latencies_timeline.set_index("time_formatted")
            st.line_chart(df_latencies_timeline, height=250)
            
            st.markdown("#### ⏳ Stage Breakdown (Overall Average Latency)")
            # Average stage bar chart
            avg_df = pd.DataFrame({
                "Stage": ["Embedding", "Vector Retrieval", "Re-ranking", "Graph Retrieval"],
                "Average Latency (s)": [
                    metrics["avg_embedding_time"],
                    metrics["avg_retrieval_latency"],
                    metrics["avg_reranking_time"],
                    metrics["avg_graph_retrieval_latency"]
                ]
            }).set_index("Stage")
            st.bar_chart(avg_df, height=220)
        else:
            st.info("No query logs available to compile latency timelines.")

    with tab_agent_graph:
        st.markdown("### ⚙️ Agent Routing & Graph Retrieval Insights")
        
        # Agent metrics row
        agent_cols = st.columns(5)
        with agent_cols[0]:
            st.metric("Avg Agent Loops", f"{metrics['avg_agent_iterations']:.1f}")
        with agent_cols[1]:
            st.metric("Avg Sub-Questions", f"{metrics['avg_sub_questions']:.1f}")
        with agent_cols[2]:
            st.metric("Graph Nodes", metrics["graph_nodes"])
        with agent_cols[3]:
            st.metric("Graph Relationships", metrics["graph_relationships"])
        with agent_cols[4]:
            st.metric("Graph Density", f"{metrics['relationship_density']:.4f}")

        st.divider()

        col_agent_ratio, col_ent_dist = st.columns(2)
        with col_agent_ratio:
            st.markdown("#### 🎯 Query Classification: Simple vs. Complex")
            # Query type distribution bar chart
            simple_count = metrics["total_queries"] - metrics["total_agent_executions"]
            complex_count = metrics["total_agent_executions"]
            df_class = pd.DataFrame({
                "Query Type": ["Simple (Direct Path)", "Complex (Agentic Path)"],
                "Count": [simple_count, complex_count]
            }).set_index("Query Type")
            st.bar_chart(df_class, height=220)
            
        with col_ent_dist:
            st.markdown("#### 🏷️ Knowledge Graph Entity Type Distribution")
            if metrics["entity_distribution"]:
                df_dist = pd.DataFrame(list(metrics["entity_distribution"].items()), columns=["Entity Type", "Count"]).set_index("Entity Type")
                st.bar_chart(df_dist, height=220)
            else:
                st.caption("No entities in the knowledge graph.")

    with tab_resources:
        st.markdown("### 💾 Storage & Cost Optimization Insights")
        
        # 1. AI Cost Breakdown Row
        st.markdown("#### 🪙 AI Cost Breakdown")
        cost_cols = st.columns(3)
        with cost_cols[0]:
            st.metric("Total LLM Cost", f"${metrics['total_llm_cost']:.4f}")
        with cost_cols[1]:
            st.metric("Total Embedding Cost", f"${metrics['total_embedding_cost']:.4f}")
        with cost_cols[2]:
            st.metric("Total AI Cost", f"${metrics['total_ai_cost']:.4f}")

        st.divider()

        # 2. Database sizes
        st.markdown("#### 💾 Database Storage Sizes")
        res_cols = st.columns(4)
        with res_cols[0]:
            st.metric("Vector DB Size", f"{metrics['resource_vector_store_size_bytes'] / 1024:.2f} KB")
        with res_cols[1]:
            st.metric("Graph DB Size", f"{metrics['resource_graph_store_size_bytes'] / 1024:.2f} KB")
        with res_cols[2]:
            st.metric("Telemetry Log Size", f"{metrics['resource_telemetry_size_bytes'] / 1024:.2f} KB")
        with res_cols[3]:
            st.metric("Extracted Tables", metrics["tables_extracted"])

        st.divider()

        # Ingestion OCR insights row
        st.markdown("#### 👁️ OCR & Table Extraction Quality")
        ocr_cols = st.columns(3)
        with ocr_cols[0]:
            st.metric("OCR Docs Processed", metrics["ocr_documents_processed"])
        with ocr_cols[1]:
            st.metric("OCR Success Rate", f"{metrics['ocr_success_rate']:.1f}%")
        with ocr_cols[2]:
            st.metric("Avg Document Size", f"{metrics['avg_document_size'] / (1024*1024):.2f} MB")

        st.divider()
        
        col_doc_types, col_token_avg = st.columns(2)
        with col_doc_types:
            st.markdown("#### 📝 File Format Distributions")
            if metrics["document_type_distribution"]:
                df_doc_types = pd.DataFrame(list(metrics["document_type_distribution"].items()), columns=["Extension", "Count"]).set_index("Extension")
                st.bar_chart(df_doc_types, height=200)
            else:
                st.caption("No documents ingested.")
        with col_token_avg:
            st.markdown("#### 📏 Input Context & Token Efficiency")
            # Display LLM Context Chunks vs Avg Tokens
            with st.container(border=True):
                st.markdown("**Token Utilization Statistics:**")
                st.markdown(
                    f"- Average input context size: `{metrics['avg_context_size']:.0f} characters`\n"
                    f"- Average candidate chunks retrieved: `{metrics['avg_chunks_retrieved']:.1f}`\n"
                    f"- Average top chunks sent to LLM: `{metrics['avg_chunks_sent_llm']:.1f}`\n"
                    f"- Average estimated tokens per query: `{metrics['avg_tokens_per_query']:.0f} tokens`"
                )
                st.caption("Target 10-15 chunks sent to LLM for balanced cost and relevance context.")

        st.divider()

        # 3. LLM Pricing Configuration Panel
        st.markdown("#### ⚙️ Pricing Rates Configuration")
        presets = metrics["pricing_presets"]
        active_pricing = metrics["active_pricing"]
        
        default_index = list(presets.keys()).index(active_pricing["provider_name"]) if active_pricing["provider_name"] in presets else 0
        
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            preset_choice = st.selectbox(
                "Select LLM Provider Preset",
                options=list(presets.keys()),
                index=default_index,
                key="active_pricing_preset_selectbox"
            )
            
            preset_vals = presets[preset_choice]
            st.info(f"Selected preset default values:\n- Input: **${preset_vals['input_rate']}/1M**\n- Output: **${preset_vals['output_rate']}/1M**\n- Cached: **${preset_vals['cached_input_rate']}/1M**\n- Embedding: **${preset_vals['embedding_rate']}/1M**")
            
        with col_p2:
            in_rate_val = float(preset_vals["input_rate"])
            out_rate_val = float(preset_vals["output_rate"])
            cached_rate_val = float(preset_vals["cached_input_rate"])
            emb_rate_val = float(preset_vals["embedding_rate"])
            
            input_rate = st.number_input("LLM Input Cost per 1M Tokens ($)", value=in_rate_val, min_value=0.0, step=0.01, format="%.4f")
            output_rate = st.number_input("LLM Output Cost per 1M Tokens ($)", value=out_rate_val, min_value=0.0, step=0.01, format="%.4f")
            cached_input_rate = st.number_input("LLM Cached Input Cost per 1M Tokens ($)", value=cached_rate_val, min_value=0.0, step=0.01, format="%.4f")
            embedding_rate = st.number_input("Embedding Cost per 1M Tokens ($)", value=emb_rate_val, min_value=0.0, step=0.001, format="%.4f")

        if st.button("💾 Save Pricing Settings", type="primary", use_container_width=True):
            config_to_save = {
                "input_rate": input_rate,
                "output_rate": output_rate,
                "cached_input_rate": cached_input_rate,
                "embedding_rate": embedding_rate
            }
            pipeline.telemetry.save_pricing_config(preset_choice, config_to_save)
            st.success("Pricing configurations saved! Recalculating dashboard costs...")
            time.sleep(0.5)
            st.rerun()

    with tab_obs:
        st.markdown("### 🔍 System Logs & Error Observability")
        
        # Errors count metrics
        err_cols = st.columns(5)
        with err_cols[0]:
            st.metric("Query Failures", metrics["failed_queries"])
        with err_cols[1]:
            st.metric("Retrieval Failures", metrics["failed_retrievals"])
        with err_cols[2]:
            st.metric("Graph Build Failures", metrics["failed_graph_builds"])
        with err_cols[3]:
            st.metric("OCR Pipeline Failures", metrics["failed_ocr"])
        with err_cols[4]:
            st.metric("Agent Loop Failures", metrics["agent_errors"])

        st.divider()

        # Telemetry Log events list
        st.markdown("#### 📜 Recent System Errors & Exceptions")
        if metrics["error_summaries"]:
            for idx, err in enumerate(reversed(metrics["error_summaries"]), start=1):
                err_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(err.get("timestamp", time.time())))
                with st.expander(f"Error #{idx} - Type: {err.get('error_type', 'unknown').upper()} | {err_time}"):
                    st.error(f"Message: {err.get('error_message')}")
                    if err.get("context"):
                        st.json(err.get("context"))
        else:
            st.success("No system errors or exceptions have been logged. All runs are green!")


def main() -> None:
    pipeline = get_pipeline()
    # Self-healing cache clear check for outdated cached instances in Streamlit
    is_outdated = False
    try:
        if hasattr(pipeline, "telemetry"):
            metrics = pipeline.telemetry.get_metrics(pipeline)
            if "active_pricing" not in metrics:
                is_outdated = True
    except Exception:
        is_outdated = True

    if (not hasattr(pipeline, "session_manager") 
        or not hasattr(pipeline, "config") 
        or not hasattr(pipeline, "graph_store")
        or not hasattr(pipeline, "telemetry")
        or not hasattr(pipeline, "settings")
        or is_outdated):
        if "cache_cleared" not in st.session_state:
            st.session_state.cache_cleared = True
            import sys
            import importlib
            # Reload all rag submodules to pick up updated class definitions
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith("rag"):
                    try:
                        importlib.reload(sys.modules[mod_name])
                    except Exception:
                        pass
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("RAGPipeline class definition is outdated in Python's memory cache. Please restart the Streamlit server to apply changes.")
            st.stop()

    init_session_state(pipeline)
    render_sidebar(pipeline)

    # Main area tabs for clean navigation
    tab_chat, tab_manager, tab_graph, tab_analytics = st.tabs([
        "💬 Chat Room", 
        "📂 Document Database", 
        "📊 Graph Explorer",
        "📈 Analytics & Observability"
    ])

    with tab_chat:
        render_chat(pipeline)

    with tab_manager:
        render_doc_manager(pipeline)

    with tab_graph:
        render_graph_explorer(pipeline)

    with tab_analytics:
        render_analytics_dashboard(pipeline)


if __name__ == "__main__":
    main()
