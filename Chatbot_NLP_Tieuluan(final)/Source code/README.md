# OU Academic RAG Chatbot

## 1. Giới thiệu đề tài

Đây là project xây dựng hệ thống chatbot hỏi đáp học vụ dành cho sinh viên Trường Đại học Mở TP.HCM. Hệ thống sử dụng phương pháp **Retrieval-Augmented Generation (RAG)** để truy xuất thông tin từ tài liệu học vụ, chương trình đào tạo, kế hoạch đào tạo và học phí, sau đó tạo câu trả lời bằng mô hình ngôn ngữ thông qua OpenAI API.

Repository này được dùng để nộp **source code** cho đồ án môn học **Xử lý ngôn ngữ tự nhiên**. Vì vậy, repo chủ yếu bao gồm mã nguồn, notebook xử lý và hướng dẫn chạy. Dữ liệu gốc, file dữ liệu `.jsonl` đã xử lý và thư mục Chroma vector database không được đưa trực tiếp lên GitHub.

Chatbot được xây dựng nhằm hỗ trợ sinh viên tra cứu nhanh các thông tin như:

- Quy chế đào tạo
- Điều kiện xét tốt nghiệp
- Chương trình đào tạo theo ngành
- Chuẩn đầu ra PLO
- Thông tin học phí
- Kế hoạch đào tạo năm học
- Một số thông tin hướng dẫn sinh viên

## 2. Công nghệ sử dụng

Project sử dụng các công nghệ chính sau:

- **Python**: ngôn ngữ lập trình chính
- **Streamlit**: xây dựng giao diện web chatbot
- **OpenAI API**: tạo embedding và sinh câu trả lời
- **ChromaDB**: lưu trữ và truy xuất vector database
- **JSONL**: định dạng dữ liệu trung gian sau khi tiền xử lý và chia chunk
- **Jupyter Notebook**: thực hiện các bước tiền xử lý, tạo chunk, build vector database và kiểm thử retrieval

## 3. Cấu trúc thư mục source code

```text
OU-Academic-RAG-Chatbot/
│
├── app_openai.py
├── README.md
├── requirements.txt
├── .env
│
└── notebooks/
    ├── DataPreprocessing_Chunking.ipynb
    ├── BuildChromaVectorDB.ipynb
    ├── TestRetrievalOpenAI.ipynb
    ├── RagChatbotOpenAI.ipynb
    └── StreamlitWebAppOpenAI.ipynb
```

Trong đó:

| Thành phần | Mô tả |
|---|---|
| `app_openai.py` | File chính để chạy giao diện web chatbot bằng Streamlit |
| `DataPreprocessing_Chunking.ipynb` | Notebook tiền xử lý dữ liệu và chia chunk |
| `BuildChromaVectorDB.ipynb` | Notebook tạo Chroma vector database từ dữ liệu đã chunk |
| `TestRetrievalOpenAI.ipynb` | Notebook kiểm tra khả năng truy xuất tài liệu |
| `RagChatbotOpenAI.ipynb` | Notebook thử nghiệm pipeline RAG tổng thể |
| `StreamlitWebAppOpenAI.ipynb` | Notebook phát triển giao diện Streamlit |
| `requirements.txt` | Danh sách thư viện cần cài đặt |
| `.env.example` | File mẫu khai báo biến môi trường |

## 4. Lưu ý về dữ liệu

Repository này **không bao gồm**:

```text
data/all_documents_optimized_chunks.jsonl
chroma_ou_rag_db_openai/
file tài liệu gốc PDF/DOCX
```

Lý do không đưa dữ liệu trực tiếp lên GitHub:

- Repository chỉ dùng để nộp source code theo yêu cầu môn học
- Tránh đưa tài liệu gốc hoặc dữ liệu đã trích xuất lên môi trường công khai
- Giảm dung lượng repository
- Tránh rủi ro liên quan đến dữ liệu nội bộ, bản quyền hoặc thông tin nhạy cảm

Nếu muốn chạy đầy đủ hệ thống, cần chuẩn bị file dữ liệu đã xử lý theo định dạng JSONL và đặt tại:

```text
data/all_documents_optimized_chunks.jsonl
```

Sau đó chạy notebook sau để tạo vector database:

```text
notebooks/BuildChromaVectorDB.ipynb
```

Sau khi build thành công, hệ thống sẽ tạo thư mục Chroma DB, ví dụ:

```text
chroma_ou_rag_db_openai/
```

## 5. Cài đặt môi trường

### Bước 1: Clone repository

```bash
git clone https://github.com/your-username/OU-Academic-RAG-Chatbot.git
cd OU-Academic-RAG-Chatbot
```

### Bước 2: Tạo môi trường ảo

Trên Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Trên macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 6. Cấu hình API key

Project sử dụng OpenAI API, vì vậy cần tạo file `.env` trong thư mục gốc của project.

Có thể tạo file `.env` theo mẫu sau:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_GENERATION_MODEL=gpt-5.5
```

Lưu ý:

- Không đưa file `.env` thật lên GitHub.
- Chỉ nên đưa file `.env.example` để hướng dẫn người khác cấu hình.
- Nếu API key bị lộ, cần xóa key cũ và tạo key mới.

## 7. Quy trình chạy project

### Trường hợp 1: Chỉ xem source code

Nếu chỉ cần kiểm tra source code theo yêu cầu nộp bài, có thể xem các file chính:

```text
app_openai.py
notebooks/
README.md
requirements.txt
```

Trong trường hợp này, không bắt buộc phải có file `.jsonl` hoặc Chroma DB.

### Trường hợp 2: Chạy đầy đủ hệ thống chatbot

Để chạy đầy đủ chatbot, cần thực hiện các bước sau:

#### Bước 1: Chuẩn bị dữ liệu đã xử lý

Đặt file dữ liệu đã chunk vào đúng đường dẫn:

```text
data/all_documents_optimized_chunks.jsonl
```

Nếu muốn xử lý lại dữ liệu từ đầu, chạy notebook:

```text
notebooks/DataPreprocessing_Chunking.ipynb
```

#### Bước 2: Build Chroma Vector Database

Chạy notebook:

```text
notebooks/BuildChromaVectorDB.ipynb
```

Sau khi chạy xong, hệ thống sẽ tạo thư mục Chroma DB:

```text
chroma_ou_rag_db_openai/
```

#### Bước 3: Kiểm tra retrieval

Có thể kiểm tra khả năng truy xuất bằng notebook:

```text
notebooks/TestRetrievalOpenAI.ipynb
```

Notebook này dùng để kiểm tra hệ thống có truy xuất đúng các chunk liên quan đến câu hỏi hay không.

#### Bước 4: Chạy giao diện chatbot

Sau khi đã có Chroma DB và cấu hình API key, chạy lệnh:

```bash
streamlit run app_openai.py
```

Sau đó mở đường dẫn localhost được Streamlit hiển thị trên terminal để sử dụng chatbot.

## 8. Một số câu hỏi mẫu

Có thể thử chatbot với các câu hỏi sau:

```text
Sinh viên cần điều kiện gì để được xét tốt nghiệp?
```

```text
Thời gian công bố danh sách đủ điều kiện tốt nghiệp là khi nào?
```

```text
Chuẩn đầu ra PLO của ngành Khoa học dữ liệu là gì?
```

```text
Mức học phí bình quân của nhóm ngành Công nghệ thông tin chương trình chuẩn là bao nhiêu?
```

```text
Sinh viên kiểm tra lịch thi cá nhân ở đâu?
```

## 9. Mô tả pipeline RAG

Pipeline của hệ thống gồm các bước chính:

```text
Tài liệu gốc
→ Trích xuất văn bản
→ Làm sạch dữ liệu
→ Chia chunk
→ Lưu thành file JSONL
→ Tạo embedding bằng OpenAI
→ Lưu vào ChromaDB
→ Truy xuất chunk liên quan theo câu hỏi
→ Đưa context vào mô hình sinh
→ Trả lời kèm nguồn tham khảo
```

Trong quá trình hỏi đáp, chatbot thực hiện:

1. Nhận câu hỏi từ người dùng.
2. Tạo embedding cho câu hỏi.
3. Truy xuất các chunk liên quan từ ChromaDB.
4. Rerank nhẹ theo metadata, keyword và ngành học.
5. Ghép các chunk thành context.
6. Gửi context và câu hỏi vào OpenAI model.
7. Sinh câu trả lời bằng tiếng Việt, có trích dẫn nguồn truy xuất.

## 10. Lưu ý khi chạy project

- Cần có OpenAI API key hợp lệ.
- Cần build Chroma DB trước khi chạy file `app_openai.py`.
- Không nên đổi tên thư mục Chroma DB nếu chưa chỉnh lại biến `CHROMA_DIR` trong code.
- Nếu đổi tên collection trong quá trình build DB, cần chỉnh lại biến `COLLECTION_NAME`.
- Không nên đưa file `.env`, API key, dữ liệu gốc, file `.jsonl` hoặc thư mục Chroma DB lên GitHub nếu repo chỉ dùng để nộp source code.

## 11. Các file không nên đưa lên GitHub

Nên loại trừ các file/thư mục sau bằng `.gitignore`:

```text
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
chroma_ou_rag_db_openai/
*.sqlite3
*.log
```

## 12. Kết quả đạt được

Project đã xây dựng được một chatbot RAG có khả năng:

- Hỏi đáp dựa trên tài liệu học vụ đã cung cấp
- Truy xuất nguồn liên quan từ ChromaDB
- Sinh câu trả lời bằng tiếng Việt
- Hiển thị nguồn tham khảo cho từng câu trả lời
- Hỗ trợ lọc tài liệu theo nhóm như quy chế, chương trình đào tạo, học phí, kế hoạch đào tạo
- Có giao diện web trực quan bằng Streamlit

## 13. Hướng phát triển

Một số hướng có thể cải tiến trong tương lai:

- Bổ sung thêm dữ liệu học vụ mới
- Cải thiện bước reranking kết quả truy xuất
- Thêm chức năng đánh giá độ chính xác câu trả lời
- Thêm giao diện quản trị dữ liệu
- Cho phép upload tài liệu mới và cập nhật vector database tự động
- So sánh kết quả giữa nhiều mô hình embedding khác nhau
- Triển khai chatbot lên nền tảng cloud

## 14. Tác giả

Project được thực hiện trong khuôn khổ môn học **Xử lý ngôn ngữ tự nhiên**.

Sinh viên thực hiện: `Điền tên sinh viên tại đây`

Trường: `Trường Đại học Mở TP.HCM`

## 15. Ghi chú

Repository này chỉ phục vụ mục đích học tập và nghiên cứu trong môn học. Các câu trả lời của chatbot phụ thuộc vào chất lượng dữ liệu đầu vào, quá trình chia chunk, kết quả truy xuất từ vector database và mô hình sinh câu trả lời.
