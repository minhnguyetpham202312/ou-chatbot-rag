# ============================================================
# Streamlit Web App — OU Academic RAG Chatbot using OpenAI API
# ============================================================
# Chức năng:
# 1. Nhận câu hỏi từ sinh viên.
# 2. Tạo embedding cho câu hỏi bằng OpenAI.
# 3. Truy xuất các chunk liên quan từ Chroma DB.
# 4. Rerank nhẹ theo metadata/keyword/ngành học.
# 5. Gửi context + câu hỏi vào OpenAI model để sinh câu trả lời có nguồn.

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# 1. Cấu hình giao diện Streamlit
# ============================================================
# Lưu ý: st.set_page_config nên là lệnh Streamlit đầu tiên.
st.set_page_config(
    page_title="OU Academic RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)

# ============================================================
# 2. Cấu hình hệ thống
# ============================================================
load_dotenv()

CHROMA_DIR = "./chroma_ou_rag_db_openai"
COLLECTION_NAME = "ou_academic_rag_openai"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-5.5")

# ============================================================
# 3. Kiểm tra API key
# ============================================================
if not os.getenv("OPENAI_API_KEY"):
    st.error("Bạn chưa set OPENAI_API_KEY. Hãy kiểm tra file .env hoặc biến môi trường.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# 4. CSS + giao diện tổng quan
# ============================================================
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(255, 80, 80, 0.10), transparent 28%),
                linear-gradient(135deg, #0b1020 0%, #101522 45%, #070b13 100%);
        }
        [data-testid="stHeader"] {background: transparent;}
        .block-container {
            max-width: 1380px;
            padding-top: 2.0rem;
            padding-bottom: 2.0rem;
        }
        .hero-card {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255, 76, 76, 0.18), rgba(255, 209, 102, 0.08));
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
            margin-bottom: 1.2rem;
        }
        .hero-title {
            font-size: 2.45rem;
            font-weight: 850;
            line-height: 1.15;
            margin-bottom: 0.45rem;
            color: #ffffff;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: rgba(255,255,255,0.72);
            margin-bottom: 1rem;
        }
        .badge-row {display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.6rem;}
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.68rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: rgba(255,255,255,0.82);
            font-size: 0.84rem;
            font-weight: 600;
        }
        .panel {
            padding: 1.15rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.055);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 12px 42px rgba(0,0,0,0.18);
        }
        .panel-title {
            font-size: 1.12rem;
            font-weight: 780;
            color: #ffffff;
            margin-bottom: 0.75rem;
        }
        .small-muted {color: rgba(255,255,255,0.62); font-size: 0.92rem;}
        .answer-card {
            padding: 1.15rem 1.25rem;
            border-radius: 22px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.11);
            margin-top: 0.65rem;
        }
        .source-card {
            padding: 0.85rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            margin-top: 0.6rem;
        }
        div.stButton > button {
            border-radius: 14px;
            font-weight: 700;
            min-height: 2.8rem;
        }
        div.stTextArea textarea {
            border-radius: 18px;
            min-height: 150px !important;
        }
        .stExpander {
            border-radius: 18px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cấu hình mặc định cho app. Người dùng có thể chỉnh trong mục Tùy chọn nâng cao.
doc_type_filter = "Tự động"
user_top_k = 5
show_debug = False
# ============================================================
# 6. Load Chroma DB
# ============================================================
@st.cache_resource(show_spinner=False)
def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(name=COLLECTION_NAME)

try:
    collection = load_collection()
except Exception as e:
    st.error("Không mở được Chroma DB. Hãy kiểm tra thư mục chroma_ou_rag_db_openai hoặc chạy Notebook 01 để build DB trước.")
    st.exception(e)
    st.stop()

# ============================================================
# 7. Hàm chuẩn hóa text và nhận diện ngành
# ============================================================
def normalize_text(text: str) -> str:
    """Chuẩn hóa text: chữ thường, bỏ dấu, bỏ ký tự đặc biệt, gom khoảng trắng."""
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    """Chuẩn hóa rồi xóa khoảng trắng để match tên file/slug."""
    return normalize_text(text).replace(" ", "")


MAJOR_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "Công nghệ thông tin",
        "triggers": ["công nghệ thông tin", "cong nghe thong tin", "cntt", "information technology", "information technologies"],
        "keywords": ["công nghệ thông tin", "cong nghe thong tin", "cntt"],
        "filename_keywords": ["congnghethongtin", "cntt", "informationtechnology", "informationtechnologies"],
    },
    {
        "name": "Kế toán",
        "triggers": ["kế toán", "ke toan", "accounting"],
        "keywords": ["kế toán", "ke toan", "accounting"],
        "filename_keywords": ["ketoan", "accounting"],
    },
    {
        "name": "Kiểm toán",
        "triggers": ["kiểm toán", "kiem toan", "auditing", "audit"],
        "keywords": ["kiểm toán", "kiem toan", "auditing", "audit"],
        "filename_keywords": ["kiemtoan", "auditing", "audit"],
    },
    {
        "name": "Tài chính - Ngân hàng",
        "triggers": ["tài chính ngân hàng", "tài chính - ngân hàng", "tai chinh ngan hang", "finance banking", "banking finance"],
        "keywords": ["tài chính ngân hàng", "tai chinh ngan hang", "finance", "banking"],
        "filename_keywords": ["taichinhnganhang", "financebanking", "bankingfinance"],
    },
    {
        "name": "Quản trị kinh doanh",
        "triggers": ["quản trị kinh doanh", "quan tri kinh doanh", "business administration"],
        "keywords": ["quản trị kinh doanh", "quan tri kinh doanh", "business administration"],
        "filename_keywords": ["quantrikinhdoanh", "businessadministration"],
    },
    {
        "name": "Marketing",
        "triggers": ["marketing", "tiếp thị", "tiep thi"],
        "keywords": ["marketing", "tiếp thị", "tiep thi"],
        "filename_keywords": ["marketing", "tiepthi"],
    },
    {
        "name": "Kinh doanh quốc tế",
        "triggers": ["kinh doanh quốc tế", "kinh doanh quoc te", "international business"],
        "keywords": ["kinh doanh quốc tế", "kinh doanh quoc te", "international business"],
        "filename_keywords": ["kinhdoanhquocte", "internationalbusiness"],
    },
    {
        "name": "Quản trị nhân lực",
        "triggers": ["quản trị nhân lực", "quan tri nhan luc", "human resource management", "hrm"],
        "keywords": ["quản trị nhân lực", "quan tri nhan luc", "human resource management"],
        "filename_keywords": ["quantrinhannluc", "quantrinhansu", "quantrinhannhanluc", "quantrinhannluc", "humanresourcemanagement", "hrm"],
    },
    {
        "name": "Luật kinh tế",
        "triggers": ["luật kinh tế", "luat kinh te", "economic law"],
        "keywords": ["luật kinh tế", "luat kinh te", "economic law"],
        "filename_keywords": ["luatkinhte", "economiclaw"],
    },
    {
        "name": "Luật",
        "triggers": ["ngành luật", "luật học", "luat hoc", "law"],
        "keywords": ["ngành luật", "luật", "luat", "law"],
        "filename_keywords": ["luat", "law"],
    },
    {
        "name": "Khoa học máy tính",
        "triggers": ["khoa học máy tính", "khoa hoc may tinh", "computer science"],
        "keywords": ["khoa học máy tính", "khoa hoc may tinh", "computer science"],
        "filename_keywords": ["khoahocmaytinh", "computerscience"],
    },
    {
        "name": "Khoa học dữ liệu",
        "triggers": ["khoa học dữ liệu", "khoa hoc du lieu", "data science"],
        "keywords": ["khoa học dữ liệu", "khoa hoc du lieu", "data science"],
        "filename_keywords": ["khoahocdulieu", "datascience"],
    },
    {
        "name": "Trí tuệ nhân tạo",
        "triggers": ["trí tuệ nhân tạo", "tri tue nhan tao", "artificial intelligence", "ai"],
        "keywords": ["trí tuệ nhân tạo", "tri tue nhan tao", "artificial intelligence"],
        "filename_keywords": ["trituenhantao", "artificialintelligence", "ai"],
    },
    {
        "name": "Hệ thống thông tin quản lý",
        "triggers": ["hệ thống thông tin quản lý", "he thong thong tin quan ly", "management information system", "mis"],
        "keywords": ["hệ thống thông tin quản lý", "he thong thong tin quan ly", "management information system"],
        "filename_keywords": ["hethongthongtinquanly", "managementinformationsystem", "mis"],
    },
    {
        "name": "Kỹ thuật phần mềm",
        "triggers": ["kỹ thuật phần mềm", "ky thuat phan mem", "software engineering"],
        "keywords": ["kỹ thuật phần mềm", "ky thuat phan mem", "software engineering"],
        "filename_keywords": ["kythuatphanmem", "softwareengineering"],
    },
    {
        "name": "An toàn thông tin",
        "triggers": ["an toàn thông tin", "an toan thong tin", "information security"],
        "keywords": ["an toàn thông tin", "an toan thong tin", "information security"],
        "filename_keywords": ["antoanthongtin", "informationsecurity"],
    },
    {
        "name": "Ngôn ngữ Anh",
        "triggers": ["ngôn ngữ anh", "ngon ngu anh", "english language"],
        "keywords": ["ngôn ngữ anh", "ngon ngu anh", "english language"],
        "filename_keywords": ["ngonnguanh", "englishlanguage"],
    },
    {
        "name": "Công nghệ sinh học",
        "triggers": ["công nghệ sinh học", "cong nghe sinh hoc", "biotechnology"],
        "keywords": ["công nghệ sinh học", "cong nghe sinh hoc", "biotechnology"],
        "filename_keywords": ["congnghesinhhoc", "biotechnology"],
    },
    {
        "name": "Công nghệ thực phẩm",
        "triggers": ["công nghệ thực phẩm", "cong nghe thuc pham", "food technology"],
        "keywords": ["công nghệ thực phẩm", "cong nghe thuc pham", "food technology"],
        "filename_keywords": ["congnghethucpham", "foodtechnology"],
    },
    {
        "name": "Quản lý xây dựng",
        "triggers": ["quản lý xây dựng", "quan ly xay dung", "construction management"],
        "keywords": ["quản lý xây dựng", "quan ly xay dung", "construction management"],
        "filename_keywords": ["quanlyxaydung", "constructionmanagement"],
    },
    {
        "name": "Kiến trúc",
        "triggers": ["kiến trúc", "kien truc", "architecture"],
        "keywords": ["kiến trúc", "kien truc", "architecture"],
        "filename_keywords": ["kientruc", "architecture"],
    },
    {
        "name": "Du lịch",
        "triggers": ["du lịch", "du lich", "tourism"],
        "keywords": ["du lịch", "du lich", "tourism"],
        "filename_keywords": ["dulich", "tourism"],
    },
]


def detect_major_from_query(query: str) -> Optional[Dict[str, Any]]:
    """Phát hiện ngành được nhắc trong câu hỏi."""
    q_norm = normalize_text(query)
    q_compact = compact_text(query)

    sorted_catalog = sorted(
        MAJOR_CATALOG,
        key=lambda m: max(len(normalize_text(t)) for t in m["triggers"]),
        reverse=True,
    )

    for major in sorted_catalog:
        trigger_norms = [normalize_text(t) for t in major["triggers"]]
        trigger_compacts = [compact_text(t) for t in major["triggers"]]

        if any(t in q_norm for t in trigger_norms):
            return major
        if any(t in q_compact for t in trigger_compacts):
            return major

    return None

# ============================================================
# 8. Retrieval cơ bản và retrieval cải tiến
# ============================================================
def embed_query(query: str) -> List[float]:
    """Tạo embedding cho câu hỏi bằng cùng model với documents."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def retrieve(query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Truy xuất top_k chunk từ Chroma."""
    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        hits.append(
            {
                "rank": i + 1,
                "id": ids[i],
                "distance": distances[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }
        )
    return hits


def infer_where_filter(query: str) -> Optional[Dict[str, Any]]:
    """Tự suy luận document_type phù hợp với câu hỏi."""
    q = query.lower()

    if any(k in q for k in ["học phí", "mức phí", "đóng phí", "miễn giảm học phí"]):
        return {"document_type": "tuition"}

    if any(
        k in q
        for k in [
            "điều kiện tốt nghiệp",
            "được xét tốt nghiệp",
            "công nhận tốt nghiệp",
            "cấp bằng tốt nghiệp",
            "cảnh báo học tập",
            "bảo lưu kết quả học tập",
            "nghỉ học tạm thời",
            "buộc thôi học",
            "buộc nghỉ học",
            "thi hộ",
            "nhờ người thi hộ",
            "xử lý vi phạm",
        ]
    ):
        return None

    if any(
        k in q
        for k in [
            "chương trình đào tạo",
            "ctđt",
            "chuẩn đầu ra",
            "plo",
            "tín chỉ",
            "học phần",
            "mã môn học",
            "điều kiện tiên quyết",
        ]
    ):
        return {"document_type": "curriculum"}

    if any(
        k in q
        for k in [
            "kế hoạch đào tạo",
            "năm học",
            "học kỳ",
            "mốc thời gian",
            "đợt",
            "thời gian học tập",
            "thời gian công bố",
            "thời gian nhận đơn",
            "kế hoạch xét tốt nghiệp",
        ]
    ):
        return {"document_type": "academic_plan"}

    if any(
        k in q
        for k in [
            "liên hệ",
            "phòng",
            "đơn vị",
            "thủ tục",
            "xác nhận",
            "vay vốn",
            "thời khóa biểu",
            "đăng ký môn học",
            "kết quả học tập cá nhân",
            "hệ thống nào",
        ]
    ):
        return {"document_type": "student_handbook"}

    return None


def keyword_boost_score(query: str, hit: Dict[str, Any]) -> float:
    """Tính điểm boost để rerank nhẹ kết quả retrieval."""
    q = query.lower()
    text = hit.get("document") or ""
    meta = hit.get("metadata") or {}

    text_norm = normalize_text(text)
    doc_name = str(meta.get("document_name", ""))
    doc_name_compact = compact_text(doc_name)
    section = str(meta.get("section", "")).lower()
    chunk_type = str(meta.get("chunk_type", "")).lower()
    doc_type = str(meta.get("document_type", "")).lower()

    boost = 0.0

    # Ưu tiên đúng ngành nếu câu hỏi có nhắc ngành.
    major = detect_major_from_query(query)
    if major is not None:
        filename_keywords = [compact_text(k) for k in major["filename_keywords"]]
        text_keywords = [normalize_text(k) for k in major["keywords"]]

        if any(k in doc_name_compact for k in filename_keywords):
            boost += 14.0

        if any(k in text_norm for k in text_keywords):
            boost += 5.0

        is_same_major = any(k in doc_name_compact for k in filename_keywords) or any(k in text_norm for k in text_keywords)
        if doc_type == "curriculum" and not is_same_major:
            boost -= 5.0

    # PLO / chuẩn đầu ra
    if "plo" in q or "chuẩn đầu ra" in q:
        if chunk_type == "plo":
            boost += 8.0

    # Học phí
    if "học phí" in q or "mức phí" in q:
        if doc_type == "tuition":
            boost += 10.0
        if "chương trình chuẩn" in q and "chuong trinh chuan" in text_norm:
            boost += 3.0
        if "công nghệ thông tin" in q and "cong nghe thong tin" in text_norm:
            boost += 5.0

    # Điều kiện/công nhận tốt nghiệp
    if any(k in q for k in ["điều kiện tốt nghiệp", "được xét tốt nghiệp", "công nhận tốt nghiệp", "cấp bằng tốt nghiệp"]):
        if "dieu 26" in text_norm or "cong nhan tot nghiep va cap bang tot nghiep" in text_norm:
            boost += 8.0
        if doc_type in ["regulation", "student_handbook"]:
            boost += 3.0

    # Buộc thôi học
    if "buộc thôi học" in q:
        if "2. buộc thôi học" in section or "2 buoc thoi hoc" in text_norm:
            boost += 7.0
        elif "buoc thoi hoc" in text_norm:
            boost += 3.0

    # Kế hoạch
    if "kế hoạch" in q or "năm học" in q:
        if doc_type == "academic_plan":
            boost += 4.0
        if "xét tốt nghiệp" in q and "ke hoach xet tot nghiep" in text_norm:
            boost += 4.0

    return boost


def get_adaptive_retrieval_params(query: str) -> Dict[str, int]:
    """Tự điều chỉnh số lượng retrieval theo loại câu hỏi."""
    q = query.lower()

    if "plo" in q or "chuẩn đầu ra" in q:
        return {"top_k": 10, "raw_top_k": 50, "preview_chars": 1200}

    if "học phí" in q or "mức phí" in q:
        return {"top_k": 5, "raw_top_k": 20, "preview_chars": 1500}

    return {"top_k": 5, "raw_top_k": 20, "preview_chars": 900}


def retrieve_routed(
    query: str,
    top_k: int = 5,
    raw_top_k: int = 20,
    document_type_filter: str = "Tự động",
) -> List[Dict[str, Any]]:
    """Retrieval cải tiến: routing theo document_type + rerank keyword/ngành."""
    if document_type_filter == "Tự động":
        where = infer_where_filter(query)
    elif document_type_filter == "Không lọc":
        where = None
    else:
        where = {"document_type": document_type_filter}

    raw_hits = retrieve(query=query, top_k=raw_top_k, where=where)

    reranked = []
    for hit in raw_hits:
        boost = keyword_boost_score(query, hit)
        final_score = float(hit["distance"]) - 0.05 * boost
        hit["auto_where"] = where
        hit["keyword_boost"] = boost
        hit["final_score"] = final_score
        reranked.append(hit)

    reranked.sort(key=lambda x: x["final_score"])
    final_hits = reranked[:top_k]

    for i, hit in enumerate(final_hits, start=1):
        hit["rank"] = i

    return final_hits

# ============================================================
# 9. Nguồn, context và prompt
# ============================================================
def format_source(meta: Dict[str, Any]) -> str:
    """Format metadata thành nguồn ngắn gọn, không hiển thị tên file gốc."""
    document_type = meta.get("document_type", "")
    section = meta.get("section", "")
    article = meta.get("article", "")
    page_start = meta.get("page_start", "")
    page_end = meta.get("page_end", "")

    doc_type_map = {
        "regulation": "Quy chế đào tạo",
        "student_handbook": "Sổ tay sinh viên",
        "curriculum": "Chương trình đào tạo",
        "academic_plan": "Kế hoạch đào tạo năm học",
        "tuition": "Thông tin học phí",
    }
    source_name = doc_type_map.get(document_type, "Tài liệu học vụ")
    parts = [source_name]

    if section not in [None, "", "null"]:
        parts.append(f"Mục: {section}")

    if article not in [None, "", "null"]:
        parts.append(f"Điều: {article}")

    if page_start not in [None, "", "null"] and page_end not in [None, "", "null"]:
        if page_start == page_end:
            parts.append(f"Trang: {page_start}")
        else:
            parts.append(f"Trang: {page_start}-{page_end}")

    return " | ".join(parts)


def build_context(hits: List[Dict[str, Any]]) -> str:
    """Ghép các chunk thành context có đánh số nguồn."""
    blocks = []

    for i, hit in enumerate(hits, start=1):
        source_text = format_source(hit["metadata"])
        document_text = hit["document"]

        block = f"""
[NGUỒN {i}]
{source_text}

Nội dung:
{document_text}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)


RAG_INSTRUCTIONS = """
Bạn là chatbot RAG hỗ trợ sinh viên tra cứu thông tin học vụ, quy chế đào tạo,
chương trình đào tạo, kế hoạch đào tạo và học phí của Trường Đại học Mở TP.HCM.

Nguyên tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong các NGUỒN được cung cấp.
2. Không tự suy đoán, không tự bổ sung thông tin ngoài tài liệu.
3. Nếu nguồn không đủ thông tin, hãy nói rõ: "Mình chưa tìm thấy thông tin đủ chắc chắn trong tài liệu hiện có."
4. Mỗi gạch đầu dòng hoặc mỗi ý quan trọng phải có citation ở cuối dòng, ví dụ: [NGUỒN 1].
5. Cuối câu trả lời phải có mục "Nguồn tham khảo".
6. Không hiển thị tên file gốc như .md, .pdf, docling hoặc đường dẫn file.
7. Nếu câu hỏi đã được trả lời đầy đủ bằng nguồn trực tiếp, không mở rộng sang các quy định liên quan khác.
8. Nếu câu hỏi liên quan đến PLO/chuẩn đầu ra, hãy tổng hợp các PLO tìm được trong nguồn truy xuất và giữ mã PLO nếu có.
9. Nếu câu hỏi liên quan học phí, cần phân biệt chương trình chuẩn/tiên tiến và mức học phí bình quân/dự kiến nếu tài liệu ghi như vậy.

Trả lời bằng tiếng Việt, rõ ràng, thân thiện, phù hợp với sinh viên.
""".strip()


def generate_answer(query: str, context: str) -> str:
    """Gọi OpenAI để sinh câu trả lời dựa trên context."""
    user_prompt = f"""
CÂU HỎI CỦA SINH VIÊN:
{query}

CÁC NGUỒN TÀI LIỆU ĐƯỢC TRUY XUẤT:
{context}

YÊU CẦU TRẢ LỜI:
- Chỉ dùng thông tin trong các nguồn trên.
- Trả lời đúng trọng tâm câu hỏi.
- Mỗi ý quan trọng phải có citation dạng [NGUỒN 1], [NGUỒN 2].
- Nếu nguồn không đủ thông tin, nói rõ chưa tìm thấy thông tin đủ chắc chắn.
- Cuối câu trả lời phải có mục "Nguồn tham khảo".
""".strip()

    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=RAG_INSTRUCTIONS,
        input=user_prompt,
    )
    return response.output_text


def answer_question(query: str, display_top_k: int, document_type_filter: str) -> Dict[str, Any]:
    """Pipeline RAG hoàn chỉnh: retrieve_routed -> build_context -> generate_answer."""
    params = get_adaptive_retrieval_params(query)

    # Ưu tiên số nguồn user chọn, nhưng vẫn giữ adaptive cho PLO/học phí nếu cần.
    top_k = max(display_top_k, params["top_k"] if ("plo" in query.lower() or "chuẩn đầu ra" in query.lower()) else display_top_k)

    hits = retrieve_routed(
        query=query,
        top_k=top_k,
        raw_top_k=params["raw_top_k"],
        document_type_filter=document_type_filter,
    )
    context = build_context(hits)
    answer = generate_answer(query, context)
    return {"answer": answer, "hits": hits, "context": context, "params": params}

# ============================================================
# 10. UI hỏi đáp
# ============================================================
example_questions = [
    "Sinh viên cần điều kiện gì để được xét tốt nghiệp?",
    "Thời gian công bố danh sách đủ điều kiện tốt nghiệp là khi nào?",
    "Chuẩn đầu ra PLO của ngành Khoa học dữ liệu là gì?",
    "Thông tin tổng quát ngành Hệ thống thông tin quản lý",
    "Mức học phí bình quân của nhóm ngành Công nghệ thông tin chương trình chuẩn là bao nhiêu?",
    "Sinh viên kiểm tra lịch thi cá nhân ở đâu?",
    "Thông tin về môn học Xử lý ngôn ngữ tự nhiên ngành Khoa học dữ liệu",
]

if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🎓 OU Academic RAG Chatbot</div>
        <div class="hero-subtitle">
            Hỗ trợ sinh viên tra cứu học vụ, chương trình đào tạo, kế hoạch đào tạo và học phí tại Trường Đại học Mở TP.HCM.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.92, 1.08], gap="large")

with left_col:
    st.markdown('<div class="panel-title">📝 Nhập câu hỏi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Bạn có thể hỏi về quy chế, học phí, CTĐT, PLO, học phần, lịch thi hoặc kế hoạch tốt nghiệp.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("💡 Câu hỏi mẫu", expanded=False):
        for idx, sample in enumerate(example_questions):
            if st.button(sample, key=f"sample_{idx}", use_container_width=True):
                st.session_state["current_question"] = sample
                st.rerun()

    query = st.text_area(
        "Câu hỏi của bạn",
        value=st.session_state.get("current_question", ""),
        placeholder="Ví dụ: Sinh viên cần điều kiện gì để được xét tốt nghiệp?",
        height=150,
    )

    with st.expander("⚙️ Tùy chọn nâng cao", expanded=False):
        doc_type_filter = st.selectbox(
            "Chế độ lọc tài liệu",
            options=[
                "Tự động",
                "Không lọc",
                "regulation",
                "curriculum",
                "student_handbook",
                "academic_plan",
                "tuition",
            ],
            index=0,
            help="Nên để Tự động để chatbot tự chọn nhóm tài liệu phù hợp.",
        )

        user_top_k = st.slider(
            "Số nguồn đưa vào câu trả lời",
            min_value=3,
            max_value=10,
            value=5,
            help="Số nguồn cuối cùng đưa vào model và hiển thị cho người dùng.",
        )

        show_debug = st.checkbox(
            "Hiển thị debug retrieval",
            value=False,
            help="Bật để xem distance, boost, final_score và metadata chi tiết.",
        )
        
    ask_col, clear_col = st.columns([2, 1])
    with ask_col:
        ask_clicked = st.button("🚀 Hỏi chatbot", type="primary", use_container_width=True)
    with clear_col:
        clear_clicked = st.button("Xóa", use_container_width=True)

    if clear_clicked:
        st.session_state["current_question"] = ""
        st.session_state["last_question"] = ""
        st.session_state["last_result"] = None
        st.rerun()

    if ask_clicked:
        if not query.strip():
            st.warning("Bạn hãy nhập câu hỏi trước.")
        else:
            st.session_state["current_question"] = query.strip()
            st.session_state["last_question"] = query.strip()
            with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
                st.session_state["last_result"] = answer_question(
                    query=query.strip(),
                    display_top_k=user_top_k,
                    document_type_filter=doc_type_filter,
                )

with right_col:
    st.markdown('<div class="panel-title">📌 Kết quả trả lời</div>', unsafe_allow_html=True)

    result = st.session_state.get("last_result")
    last_question = st.session_state.get("last_question", "")

    if result is None:
        st.info("Nhập câu hỏi ở khung bên trái rồi bấm **Hỏi chatbot** để xem câu trả lời và nguồn truy xuất.")
    else:
        st.markdown(f"**Câu hỏi:** {last_question}")

        with st.container(border=True):
            st.write(result["answer"])

        st.markdown("### 📚 Nguồn truy xuất")
        for hit in result["hits"]:
            meta = hit["metadata"]
            source_label = format_source(meta)

            if show_debug:
                title = f"Nguồn {hit['rank']} · {source_label} · distance={hit['distance']:.4f}"
            else:
                title = f"Nguồn {hit['rank']} · {source_label}"

            with st.expander(title, expanded=False):
                st.markdown(f"**{source_label}**")
                
                if show_debug:
                    st.json(
                        {
                            "id": hit.get("id"),
                            "distance": hit.get("distance"),
                            "keyword_boost": hit.get("keyword_boost"),
                            "final_score": hit.get("final_score"),
                            "auto_where": hit.get("auto_where"),
                            "metadata": meta,
                        }
                    )

                st.text(hit["document"][:2500])
