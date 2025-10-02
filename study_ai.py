import streamlit as st
import os
import pickle
from datetime import datetime
from typing import List
import logging
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import docx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Document Processing
# ------------------------------
class SimpleDocProcessor:
    @staticmethod
    def extract_text(file) -> str:
        try:
            file_type = file.name.lower().split('.')[-1]
            if file_type == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
                return text
            elif file_type == 'docx':
                doc = docx.Document(file)
                return "\n".join([p.text for p in doc.paragraphs])
            elif file_type == 'txt':
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                return content
            else:
                return ""
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size) if words[i:i+chunk_size]]
        return chunks

# ------------------------------
# Vector Store with FAISS
# ------------------------------
class SimpleVectorStore:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.model_name = "all-MiniLM-L6-v2"

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def add_documents(self, chunks: List[str]):
        if not chunks:
            return
        self.load_model()
        self.chunks.extend(chunks)
        all_embeddings = self.model.encode(self.chunks)
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings)

    def search(self, query: str, k: int = 5) -> List[str]:
        if not self.chunks or self.index is None:
            return []
        self.load_model()
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        results = [self.chunks[idx] for score, idx in zip(scores[0], indices[0]) if idx < len(self.chunks) and score > 0.05]
        if not results and len(self.chunks) > 0:
            for idx in indices[0][:min(3, len(self.chunks))]:
                if idx < len(self.chunks):
                    results.append(self.chunks[idx])
        return results

# ------------------------------
# Response Formatter
# ------------------------------
class ResponseFormatter:
    @staticmethod
    def aggressive_clean(text: str) -> str:
        text = re.sub(r'MUQuestionPapers?\.com\s*Page\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Q\.?P\.?\s*Code\s*[–-]?\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bQ\.?\s*\d+\s*[a-z]?\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+[a-z]\)', '', text)
        text = re.sub(r'\b\d+M\b', '', text)
        text = re.sub(r'\(\s*\d+\s*marks?\s*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Total\s*:\s*\d+\s*Marks?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Answer|Solution|Ans)\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        return text.strip()

    @staticmethod
    def extract_meaningful_sentences(text: str, max_sentences: int = 6) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 30 and s[0].isupper() and s[-1] in '.!?']
        return meaningful[:max_sentences]

    @staticmethod
    def extract_definition_sentences(sentences: List[str]) -> List[str]:
        definitions = [s for s in sentences if re.search(r'\b(is|acts as|provides|manages|keeps track|refers to)\b', s.lower())]
        return definitions or sentences

    @staticmethod
    def format_as_bullets(sentences: List[str]) -> str:
        return '\n\n'.join([f"• {s.rstrip('.')}" for s in sentences])

    @staticmethod
    def format_as_paragraph(sentences: List[str]) -> str:
        paragraphs = []
        current = []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if len(current) == 2 or i == len(sentences) - 1:
                paragraphs.append(' '.join(current))
                current = []
        return '\n\n'.join(paragraphs)

    @staticmethod
    def create_structured_response(question: str, raw_context: str, question_type: str) -> str:
        cleaned = ResponseFormatter.aggressive_clean(raw_context)
        sentences = ResponseFormatter.extract_meaningful_sentences(cleaned, max_sentences=6)
        if question_type in ["Definition", "Explanation"]:
            sentences = ResponseFormatter.extract_definition_sentences(sentences)
        if not sentences:
            return f"### ℹ️ Limited Information\n\n**Question:** {question}\n\nCould not extract clear information."
        if question_type in ["List/Types", "Process/Steps", "Advantages/Disadvantages"]:
            formatted_content = ResponseFormatter.format_as_bullets(sentences)
        else:
            formatted_content = ResponseFormatter.format_as_paragraph(sentences)
        icon_map = {
            "Definition": "📖",
            "Explanation": "💡",
            "Process/Steps": "🔄",
            "Reasoning": "🤔",
            "Comparison": "⚖️",
            "List/Types": "📋",
            "Advantages/Disadvantages": "✓✗",
            "Answer": "💬"
        }
        icon = icon_map.get(question_type, "💬")
        return f"""### {icon} {question_type}

**Question:** {question}

**Answer:**

{formatted_content}

---
✅ *Information extracted from your study materials*"""

# ------------------------------
# AI Helper
# ------------------------------
class AIHelper:
    @staticmethod
    def determine_question_type(question: str) -> str:
        q = question.lower()
        if any(word in q for word in ['what is', 'define', 'definition', 'meaning of']):
            return "Definition"
        elif any(word in q for word in ['explain', 'describe', 'discuss', 'elaborate']):
            return "Explanation"
        elif any(word in q for word in ['how to', 'steps', 'process', 'method', 'procedure', 'algorithm']):
            return "Process/Steps"
        elif any(word in q for word in ['why', 'reason', 'because', 'cause', 'purpose']):
            return "Reasoning"
        elif any(word in q for word in ['compare', 'difference', 'distinguish', 'contrast', 'vs', 'versus', 'between']):
            return "Comparison"
        elif any(word in q for word in ['list', 'types', 'kinds', 'categories', 'enumerate', 'name']):
            return "List/Types"
        elif any(word in q for word in ['advantage', 'disadvantage', 'benefit', 'drawback', 'pros', 'cons', 'merit', 'demerit']):
            return "Advantages/Disadvantages"
        else:
            return "Answer"

    @staticmethod
    def generate_response(question: str, context: str) -> str:
        if not context.strip():
            return f"### ℹ️ No Information Found\n\n**Question:** {question}\n\nNo relevant info found in uploaded materials."
        question_type = AIHelper.determine_question_type(question)

        # Filter by keywords from question for better accuracy
        key_terms = [word.lower() for word in re.findall(r'\w+', question)]
        filtered_chunks = [chunk for chunk in context.split('. ') if any(k in chunk.lower() for k in key_terms)]
        context = ". ".join(filtered_chunks) if filtered_chunks else context

        return ResponseFormatter.create_structured_response(question, context, question_type)

# ------------------------------
# StudyChatbot Core
# ------------------------------
class StudyChatbot:
    def __init__(self):
        self.subjects = {}
        self.data_dir = "chatbot_data"
        os.makedirs(self.data_dir, exist_ok=True)

    def create_subject(self, name: str):
        if name not in self.subjects:
            self.subjects[name] = {
                'vector_store': SimpleVectorStore(),
                'chat_history': [],
                'documents': [],
                'created_at': datetime.now()
            }
            self.save_data()

    def add_documents(self, subject: str, files):
        if subject not in self.subjects:
            self.create_subject(subject)
        all_chunks = []
        existing_files = [doc['name'] for doc in self.subjects[subject]['documents']]
        new_files_added = 0
        for file in files:
            if file.name in existing_files:
                continue
            text = SimpleDocProcessor.extract_text(file)
            if text:
                chunks = SimpleDocProcessor.chunk_text(text)
                all_chunks.extend(chunks)
                self.subjects[subject]['documents'].append({
                    'name': file.name,
                    'uploaded_at': datetime.now(),
                    'chunks': len(chunks)
                })
                new_files_added += 1
        if all_chunks:
            self.subjects[subject]['vector_store'].add_documents(all_chunks)
            self.save_data()
            return len(all_chunks), new_files_added
        return 0, new_files_added

    def chat(self, subject: str, question: str) -> str:
        if subject not in self.subjects:
            return "Please create a subject first and upload documents."
        vector_store = self.subjects[subject]['vector_store']
        if not vector_store.chunks:
            return f"No documents processed yet for '{subject}'."
        relevant_chunks = vector_store.search(question, k=5)
        cleaned_chunks = [ResponseFormatter.aggressive_clean(c) for c in relevant_chunks if len(c.split()) > 15]
        context = " ".join(cleaned_chunks)
        return AIHelper.generate_response(question, context)

    def save_data(self):
        try:
            metadata = {}
            for name, data in self.subjects.items():
                metadata[name] = {
                    'chat_history': data['chat_history'],
                    'documents': data['documents'],
                    'created_at': data['created_at'],
                    'chunks': data['vector_store'].chunks if data['vector_store'].chunks else []
                }
            with open(os.path.join(self.data_dir, 'subjects.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def load_data(self):
        try:
            filepath = os.path.join(self.data_dir, 'subjects.pkl')
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    metadata = pickle.load(f)
                for name, data in metadata.items():
                    self.subjects[name] = {
                        'vector_store': SimpleVectorStore(),
                        'chat_history': data['chat_history'],
                        'documents': data['documents'],
                        'created_at': data['created_at']
                    }
                    if 'chunks' in data and data['chunks']:
                        self.subjects[name]['vector_store'].add_documents(data['chunks'])
        except Exception as e:
            logger.error(f"Error loading data: {e}")

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="Study Chatbot", page_icon="🤖", layout="wide")
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = StudyChatbot()
        st.session_state.chatbot.load_data()
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None
    chatbot = st.session_state.chatbot

    # Sidebar
    with st.sidebar:
        st.title("📚 Study Subjects")
        st.subheader("➕ Add Subject")
        new_subject = st.text_input("Subject Name", placeholder="e.g., Math, Science", key="new_subject_input")
        if st.button("Create"):
            if new_subject.strip():
                chatbot.create_subject(new_subject.strip())
                st.session_state.current_subject = new_subject.strip()
                if "new_subject_input" in st.session_state:
                    del st.session_state["new_subject_input"]
                st.success(f"Subject '{new_subject.strip()}' created!")
                st.rerun()

        if chatbot.subjects:
            st.subheader("📖 Select Subject")
            subject_names = list(chatbot.subjects.keys())
            current_index = 0
            if st.session_state.current_subject in subject_names:
                current_index = subject_names.index(st.session_state.current_subject)
            selected = st.selectbox("Choose subject:", subject_names, index=current_index)
            if selected != st.session_state.current_subject:
                st.session_state.current_subject = selected
                st.rerun()

            if selected:
                subject_data = chatbot.subjects[selected]
                st.write(f"📄 Documents: {len(subject_data['documents'])}")
                st.write(f"💬 Chats: {len(subject_data['chat_history'])}")

                if subject_data['documents']:
                    with st.expander("View Documents"):
                        for doc in subject_data['documents']:
                            st.write(f"• {doc['name']} ({doc['chunks']} chunks)")
                            st.caption(f"Uploaded: {doc['uploaded_at'].strftime('%Y-%m-%d %H:%M')}")

        st.markdown("---")
        if st.session_state.current_subject:
            st.subheader("📤 Upload Documents")
            uploaded_files = st.file_uploader("Add study materials", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
            if uploaded_files and st.button("Upload"):
                with st.spinner("Processing documents..."):
                    chunks_added, files_added = chatbot.add_documents(st.session_state.current_subject, uploaded_files)
                    if files_added > 0:
                        st.success(f"Added {files_added} new files with {chunks_added} text chunks!")
                        st.rerun()
                    else:
                        st.info("All selected files were already uploaded.")

    # Chat Interface
    st.title("🤖 AI Study Chatbot")
    if st.session_state.current_subject:
        st.subheader(f"Chatting about: {st.session_state.current_subject}")
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("Ask a question:", placeholder="What would you like to know?", key="question_input")
        with col2:
            ask_button = st.button("Ask", type="primary", disabled=not question.strip())

        if ask_button and question.strip():
            with st.spinner("Thinking..."):
                response = chatbot.chat(st.session_state.current_subject, question.strip())
            st.markdown("### 🤖 AI Response:")
            st.markdown(response, unsafe_allow_html=True)

        subject_data = chatbot.subjects[st.session_state.current_subject]
        if subject_data['chat_history']:
            st.markdown("---")
            st.subheader("💬 Chat History")
            recent_chats = subject_data['chat_history'][-5:]
            for chat in reversed(recent_chats):
                with st.expander(f"Q: {chat['question'][:50]}{'...' if len(chat['question']) > 50 else ''}"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown("**Answer:**")
                    st.markdown(chat['response'], unsafe_allow_html=True)
                    st.success("✅ Answer based on your study materials" if chat['has_context'] else "ℹ️ General AI response")
                    st.caption(f"Asked: {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.markdown("## Welcome to AI Study Chatbot! 🎓\n\nCreate a subject, upload materials, and start asking questions!")

if __name__ == "__main__":
    main()
