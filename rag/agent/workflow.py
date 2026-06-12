import json

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer that."

Some parts of the context may represent tables formatted in Markdown. Pay close attention to table headers, columns, and rows, and use them to answer structural questions (such as revenues, sales, counts, regions) accurately.

For any fact or claim you state, you MUST cite the context source(s) using inline numbers like [1], [2], corresponding to the index of the source in the context.
Be concise, accurate, and professional."""


class AgentTrace:
    def __init__(self):
        self.query_type = "Semantic"  # Relationship, Semantic, or Hybrid
        self.retrieval_iterations = 0
        self.sub_questions = []
        self.documents_consulted = set()
        self.confidence_assessment = 1.0
        self.logs = []

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "retrieval_iterations": self.retrieval_iterations,
            "sub_questions": self.sub_questions,
            "documents_consulted": list(self.documents_consulted),
            "confidence_assessment": round(self.confidence_assessment, 2),
            "logs": self.logs,
        }


class WorkflowState:
    def __init__(self, query: str, history: list[dict] = None):
        self.query = query
        self.standalone_query = query
        self.history = history or []
        self.query_classification = "Semantic"  # Relationship, Semantic, Hybrid
        self.query_type = "Simple"  # Simple vs Complex classifier routing
        self.iterations = 0
        self.sub_questions = []
        self.retrieved_chunks = []
        self.graph_context = ""
        self.graph_sources = []
        self.confidence = 1.0
        self.missing_aspects = ""
        self.documents_consulted = set()
        self.answer = ""
        self.agent_trace = AgentTrace()


class RAGWorkflow:
    def __init__(self, pipeline, config: dict):
        self.pipeline = pipeline
        self.config = config

    def run(self, state: WorkflowState) -> WorkflowState:
        # Node 1: Classify Query (Relationship, Semantic, Hybrid) and Type (Simple vs Complex)
        state = self.classify_query_node(state)

        # Node 2: Retrieve Context (Adaptive routing across Vector and Graph layers)
        state = self.retrieve_node(state)

        # Agentic loop for Complex queries
        if state.query_type == "Complex":
            # Node 3: Evaluate Context Sufficiency
            state = self.evaluate_context_node(state)

            threshold = self.config.get("confidence_threshold", 0.6)
            max_iters = self.config.get("max_retrieval_iterations", 2)

            while state.confidence < threshold and state.iterations < max_iters:
                # Node 4: Planner (generate sub-questions & retrieve additional evidence)
                state = self.planning_node(state)
                # Re-evaluate
                state = self.evaluate_context_node(state)

        # Node 5: Generate Final Grounded Answer
        state = self.generate_answer_node(state)
        return state

    def classify_query_node(self, state: WorkflowState) -> WorkflowState:
        # Step A: Classify Relationship vs Semantic vs Hybrid
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Classify the user query into one of the following retrieval strategies for a document index & knowledge graph database:\n"
                        "1. 'Relationship': The question asks strictly about connections, ownership, dependencies, or connections between services, teams, databases, and technologies (e.g. 'Who owns Service X?', 'What databases are connected to App Y?', 'Show all projects using Python').\n"
                        "2. 'Semantic': The question asks for general descriptions, explanations, conceptual architecture, summaries, or textual definitions (e.g. 'Explain Kafka architecture', 'Summarize the refund policies').\n"
                        "3. 'Hybrid': The question requires both conceptual explanation and relationship tracing (e.g. 'Explain Kafka and list the services that use it', 'What is dynamic typing and who works on Project X?').\n"
                        "Respond with exactly one of these words: 'Relationship', 'Semantic', or 'Hybrid'."
                    ),
                },
                {"role": "user", "content": f"Question: {state.query}"},
            ]
            response = self.pipeline.llm.invoke(messages)
            res_text = response.content.strip().lower()
            if "relationship" in res_text:
                state.query_classification = "Relationship"
            elif "hybrid" in res_text:
                state.query_classification = "Hybrid"
            else:
                state.query_classification = "Semantic"
        except Exception as e:
            print(f"Classification failed: {e}. Defaulting to Semantic.")
            state.query_classification = "Semantic"
        
        state.agent_trace.query_type = state.query_classification

        # Step B: Classify Simple vs Complex
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Classify the user question as either 'Simple' or 'Complex' for a document search database.\n"
                        "- 'Simple': Questions asking for a single definition, rule, keyword, or fact located in one spot (e.g. 'What is the refund policy?', 'Explain Kafka').\n"
                        "- 'Complex': Questions requiring comparisons, synthesis across multiple sections/documents, multi-step queries, or summaries (e.g. 'Compare the PTO policies across all departments', 'How is IPv6 implemented differently in service X and Y?').\n"
                        "Respond with exactly one word: 'Simple' or 'Complex'."
                    ),
                },
                {"role": "user", "content": f"Question: {state.query}"},
            ]
            response = self.pipeline.llm.invoke(messages)
            res_text = response.content.strip().lower()
            if "complex" in res_text:
                state.query_type = "Complex"
            else:
                state.query_type = "Simple"
        except Exception as e:
            print(f"Simple/Complex classification failed: {e}. Defaulting to Simple.")
            state.query_type = "Simple"

        state.agent_trace.logs.append(
            f"Query classified as '{state.query_classification}' and routed to '{state.query_type}' path."
        )
        return state

    def retrieve_node(self, state: WorkflowState) -> WorkflowState:
        # Determine active search question
        if state.query_type == "Complex":
            state.standalone_query = self.pipeline.rewrite_query(state.query, state.history)
            question = state.standalone_query
        else:
            question = state.query

        vector_count = 0
        graph_relations_count = 0

        # Adaptive Retrieval Strategy Routing
        graph_enabled = self.config.get("graph_enabled", True)

        if state.query_classification == "Relationship" and graph_enabled:
            # 1. Graph RAG only
            max_depth = self.config.get("max_depth", 2)
            res = self.pipeline.graph_retriever.retrieve(question, max_depth=max_depth)
            state.graph_context = res["context"]
            state.graph_sources = res["sources"]
            state.retrieved_chunks = []
            graph_relations_count = len(res.get("paths", []))
        elif state.query_classification == "Semantic" or not graph_enabled:
            # 2. Vector/BM25 RAG only
            chunks = self.pipeline.retrieve(question)
            max_chunks = self.config.get("max_retrieved_chunks", 20)
            state.retrieved_chunks = chunks[:max_chunks]
            state.graph_context = ""
            state.graph_sources = []
            vector_count = len(state.retrieved_chunks)
        else:
            # 3. Hybrid (Both Graph + Vector)
            # Vector
            chunks = self.pipeline.retrieve(question)
            max_chunks = self.config.get("max_retrieved_chunks", 20)
            state.retrieved_chunks = chunks[:max_chunks]
            vector_count = len(state.retrieved_chunks)
            # Graph
            max_depth = self.config.get("max_depth", 2)
            res = self.pipeline.graph_retriever.retrieve(question, max_depth=max_depth)
            state.graph_context = res["context"]
            state.graph_sources = res["sources"]
            graph_relations_count = len(res.get("paths", []))

        # Record consulted documents
        for chunk in state.retrieved_chunks:
            state.documents_consulted.add(chunk["filename"])
            state.agent_trace.documents_consulted.add(chunk["filename"])
        for src in state.graph_sources:
            doc_name = src.get("source_document", "unknown")
            state.documents_consulted.add(doc_name)
            state.agent_trace.documents_consulted.add(doc_name)

        state.agent_trace.logs.append(
            f"Retrieval complete. Vector chunks = {vector_count}, Graph relationships = {graph_relations_count}."
        )
        return state

    def evaluate_context_node(self, state: WorkflowState) -> WorkflowState:
        # Build composite context
        context_parts = []
        if state.graph_context:
            context_parts.append(state.graph_context)
        if state.retrieved_chunks:
            vector_context = self.pipeline._build_context(state.retrieved_chunks)
            context_parts.append(vector_context)
        context = "\n\n---\n\n".join(context_parts)

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence sufficiency evaluator. Assess if the retrieved context contains sufficient, complete, and accurate information to answer the query.\n"
                        "Respond strictly with a JSON object in this format:\n"
                        "{\n"
                        "  \"sufficient\": true/false,\n"
                        "  \"confidence\": 0.0 to 1.0,\n"
                        "  \"missing_aspects\": \"description of what is missing\"\n"
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {state.query}\n\nContext:\n{context}",
                },
            ]
            response = self.pipeline.llm.invoke(messages)
            text = response.content.strip()
            
            # Clean JSON bounds
            if "{" in text and "}" in text:
                text = text[text.find("{"):text.rfind("}")+1]
            res = json.loads(text)
            
            state.confidence = float(res.get("confidence", 0.5))
            state.missing_aspects = res.get("missing_aspects", "")
            state.agent_trace.confidence_assessment = state.confidence
        except Exception as e:
            print(f"Context evaluation failed: {e}. Defaulting confidence to 0.5.")
            state.confidence = 0.5
            state.missing_aspects = "Failed to run sufficiency evaluation."

        sufficient = state.confidence >= self.config.get("confidence_threshold", 0.6)
        state.agent_trace.logs.append(
            f"Context sufficiency evaluation: Confidence = {state.confidence:.2f} (Sufficient = {sufficient})"
        )
        return state

    def planning_node(self, state: WorkflowState) -> WorkflowState:
        state.iterations += 1
        state.agent_trace.retrieval_iterations = state.iterations
        max_subs = self.config.get("max_planner_depth", 3)

        state.agent_trace.logs.append(f"Confidence below threshold. Triggering Planner loop iteration #{state.iterations}.")

        # Generate sub-questions to gather missing information
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a search planning assistant. Break down the user's question into up to {max_subs} specific search-friendly sub-questions to collect missing facts.\n"
                        "Output ONLY the questions, one per line. Do not include numbers, explanations, or lists."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original Question: {state.query}\nMissing aspects: {state.missing_aspects}",
                },
            ]
            response = self.pipeline.llm.invoke(messages)
            lines = response.content.strip().split("\n")
            sub_qs = []
            for line in lines:
                line_clean = line.strip().lstrip("0123456789.-*• ").strip()
                if line_clean:
                    sub_qs.append(line_clean)
            state.sub_questions = sub_qs[:max_subs]
            state.agent_trace.sub_questions.extend(state.sub_questions)
        except Exception as e:
            print(f"Sub-question planner failed: {e}.")
            state.sub_questions = [state.query]

        state.agent_trace.logs.append(f"Planner generated sub-questions: {state.sub_questions}")

        # Retrieve chunks for each sub-question
        new_chunks = []
        new_graph_sources = []
        new_graph_lines = []

        graph_enabled = self.config.get("graph_enabled", True)
        for sq in state.sub_questions:
            # Query both layers during sub-planning to find relationships + semantic facts
            sq_chunks = self.pipeline.retrieve(sq)
            new_chunks.extend(sq_chunks)
            
            # Graph sub-search
            if graph_enabled:
                max_depth = self.config.get("max_depth", 2)
                res = self.pipeline.graph_retriever.retrieve(sq, max_depth=max_depth)
                if res["context"]:
                    new_graph_lines.append(res["context"])
                    new_graph_sources.extend(res["sources"])

        # Merge, deduplicate, and re-rank with existing chunks
        all_chunks = {c["chunk_id"]: c for c in state.retrieved_chunks}
        for nc in new_chunks:
            c_id = nc["chunk_id"]
            if c_id not in all_chunks:
                all_chunks[c_id] = nc

        merged_list = list(all_chunks.values())
        if merged_list:
            pairs = [[state.query, c["text"]] for c in merged_list]
            scores = self.pipeline.cross_encoder.predict(pairs)
            for c, score in zip(merged_list, scores):
                c["cross_score"] = float(score)
            merged_list.sort(key=lambda x: x["cross_score"], reverse=True)

        max_chunks = self.config.get("max_retrieved_chunks", 20)
        state.retrieved_chunks = merged_list[:max_chunks]

        # Consolidate graph contexts
        if new_graph_lines:
            combined_new_graph = "\n".join(new_graph_lines)
            if state.graph_context:
                state.graph_context += "\n" + combined_new_graph
            else:
                state.graph_context = combined_new_graph
            state.graph_sources.extend(new_graph_sources)

        # Update consulted documents list
        for chunk in state.retrieved_chunks:
            state.documents_consulted.add(chunk["filename"])
            state.agent_trace.documents_consulted.add(chunk["filename"])
        for src in state.graph_sources:
            state.documents_consulted.add(src.get("source_document", "unknown"))
            state.agent_trace.documents_consulted.add(src.get("source_document", "unknown"))

        state.agent_trace.logs.append(
            f"Retrieval iteration completed. Total active vector chunks: {len(state.retrieved_chunks)}."
        )
        return state

    def generate_answer_node(self, state: WorkflowState) -> WorkflowState:
        # Build composite context
        context_parts = []
        if state.graph_context:
            context_parts.append(state.graph_context)
        if state.retrieved_chunks:
            vector_context = self.pipeline._build_context(state.retrieved_chunks)
            context_parts.append(vector_context)
        context = "\n\n---\n\n".join(context_parts)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Append sliding-window conversational history
        for msg in state.history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        instruction = "Answer based only on the context above. Be sure to include the inline citations (e.g. [1], [2]) in your answer."
        
        # Enforce strict evidence validation if confidence remains low
        threshold = self.config.get("confidence_threshold", 0.6)
        if state.query_type == "Complex" and state.confidence < threshold:
            instruction += (
                "\nCRITICAL: The context has been evaluated as insufficient. Answer ONLY with what is fully "
                "supported. If the context lacks details to answer the query, respond with exactly: "
                "'I don't have enough information in the uploaded documents to answer that.' Do NOT fabricate any facts."
            )

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {state.query}\n\n{instruction}",
        })

        try:
            response = self.pipeline.llm.invoke(messages)
            state.answer = response.content
        except Exception as e:
            state.answer = f"Error generating answer: {e}"

        state.agent_trace.logs.append("Final grounded answer generated successfully.")
        return state
