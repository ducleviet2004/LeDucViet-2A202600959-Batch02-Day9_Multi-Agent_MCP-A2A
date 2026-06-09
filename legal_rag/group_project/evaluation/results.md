# Báo Cáo Kết Quả Đánh Giá RAG Pipeline (Evaluation Report)

Tài liệu này ghi lại kết quả đánh giá chất lượng hệ thống RAG Chatbot bằng công cụ **DeepEval** trên bộ dữ liệu kiểm thử chuẩn **Golden Dataset** (gồm 15 câu hỏi pháp luật ma túy và tin tức showbiz).

---

## 1. Kết Quả Tổng Quan & So Sánh A/B

Chúng tôi đã tiến hành so sánh hai cấu hình của hệ thống:
* **Cấu hình A (Config A):** Hybrid Search (Dense + Sparse) + Cross-Encoder Reranking.
* **Cấu hình B (Config B):** Dense Search đơn thuần (Không sử dụng Reranking).

### Bảng so sánh điểm trung bình:

| Chỉ số đánh giá (Metric) | Cấu hình A (Có Reranking) | Cấu hình B (Không Reranking) | Chênh lệch (Delta) |
| :--- | :---: | :---: | :---: |
| **Faithfulness (Độ trung thực)** | 0.68 | 0.68 | +0.00 |
| **Answer Relevance (Độ liên quan)** | 0.69 | 0.68 | +0.01 |
| **Context Recall (Độ phủ ngữ cảnh)** | 0.85 | 0.83 | +0.02 |
| **Context Precision (Độ chính xác)** | 0.95 | 0.95 | +0.00 |
| **Thời gian phản hồi trung bình (Latency)** | 15.65s | 15.76s | -0.11s |

---

## 2. Phân Tích Chi Tiết Từng Câu Hỏi (Cấu hình A - Hybrid + Reranking)

| STT | Câu hỏi kiểm thử | Độ trung thực | Độ liên quan | Độ phủ | Độ chính xác | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 2... | 0.66 | 0.69 | 0.73 | 0.98 | 16.92s |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai ng... | 0.69 | 0.78 | 0.98 | 0.98 | 14.52s |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luậ... | 0.67 | 0.69 | 0.98 | 0.98 | 16.06s |
| 4 | Chất ma túy tàn phá hệ tim mạch con người như thế nào?... | 0.67 | 0.65 | 0.76 | 0.98 | 15.14s |
| 5 | Vụ đại án ma túy nào ở Việt Nam có tới 30 án tử hình bị tuyê... | 0.69 | 0.65 | 0.75 | 0.91 | 13.93s |
| 6 | Thực trạng sử dụng ma túy trong giới nghệ sĩ, showbiz được p... | 0.69 | 0.65 | 0.74 | 0.98 | 14.20s |
| 7 | Vụ việc hai thanh niên dương tính với ma túy đâm ngã CSGT di... | 0.67 | 0.65 | 0.75 | 0.98 | 14.67s |
| 8 | Nhóm người nước ngoài phê ma túy bị bắt giữ tại khách sạn nà... | 0.67 | 0.65 | 0.90 | 0.91 | 14.85s |
| 9 | Theo Luật Phòng chống ma túy 2021, người từ đủ bao nhiêu tuổ... | 0.69 | 0.77 | 0.98 | 0.98 | 14.40s |
| 10 | Tội tổ chức sử dụng trái phép chất ma túy chịu mức phạt tù t... | 0.68 | 0.71 | 0.85 | 0.98 | 14.50s |
| 11 | Các tiền chất ma túy được phân loại như thế nào theo Nghị đị... | 0.70 | 0.87 | 0.77 | 0.98 | 15.71s |
| 12 | Ai có trách nhiệm hỗ trợ gia đình và cơ quan chức năng trong... | 0.69 | 0.65 | 0.98 | 0.98 | 15.80s |
| 13 | Khối lượng chất ma túy Heroine bao nhiêu thì người tàng trữ ... | 0.66 | 0.65 | 0.80 | 0.91 | 18.52s |
| 14 | Người nghiện ma túy từ đủ bao nhiêu tuổi trở lên mới bị áp d... | 0.68 | 0.65 | 0.89 | 0.98 | 17.34s |
| 15 | Lực lượng CSGT xử lý như thế nào đối với các đối tượng lái x... | 0.69 | 0.65 | 0.84 | 0.72 | 18.12s |

---

## 3. Phân Tích Các Câu Hỏi Cho Kết Quả Thấp Nhất (Worst Performers)

Hệ thống hoạt động khá tốt, có 14 câu hỏi có điểm số dưới kỳ vọng (0.7):

* **Câu hỏi:** "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự?"
  - **Faithfulness:** 0.66 | **Relevance:** 0.69
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào?"
  - **Faithfulness:** 0.69 | **Relevance:** 0.78
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào?"
  - **Faithfulness:** 0.67 | **Relevance:** 0.69
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Chất ma túy tàn phá hệ tim mạch con người như thế nào?"
  - **Faithfulness:** 0.67 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Vụ đại án ma túy nào ở Việt Nam có tới 30 án tử hình bị tuyên?"
  - **Faithfulness:** 0.69 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Thực trạng sử dụng ma túy trong giới nghệ sĩ, showbiz được phản ánh như thế nào?"
  - **Faithfulness:** 0.69 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Vụ việc hai thanh niên dương tính với ma túy đâm ngã CSGT diễn ra như thế nào?"
  - **Faithfulness:** 0.67 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Nhóm người nước ngoài phê ma túy bị bắt giữ tại khách sạn nào ở TP HCM?"
  - **Faithfulness:** 0.67 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Theo Luật Phòng chống ma túy 2021, người từ đủ bao nhiêu tuổi trở lên sử dụng trái phép chất ma túy sẽ bị lập hồ sơ quản lý?"
  - **Faithfulness:** 0.69 | **Relevance:** 0.77
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Tội tổ chức sử dụng trái phép chất ma túy chịu mức phạt tù thấp nhất là bao nhiêu năm theo Bộ luật Hình sự?"
  - **Faithfulness:** 0.68 | **Relevance:** 0.71
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Ai có trách nhiệm hỗ trợ gia đình và cơ quan chức năng trong việc cai nghiện ma túy tự nguyện?"
  - **Faithfulness:** 0.69 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Khối lượng chất ma túy Heroine bao nhiêu thì người tàng trữ trái phép có thể đối diện mức án tù chung thân hoặc tử hình?"
  - **Faithfulness:** 0.66 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Người nghiện ma túy từ đủ bao nhiêu tuổi trở lên mới bị áp dụng biện pháp cai nghiện bắt buộc?"
  - **Faithfulness:** 0.68 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

* **Câu hỏi:** "Lực lượng CSGT xử lý như thế nào đối với các đối tượng lái xe dương tính với ma túy?"
  - **Faithfulness:** 0.69 | **Relevance:** 0.65
  - **Nguyên nhân:** Câu trả lời của LLM trích dẫn các tài liệu liên quan đến luật pháp chung chung hoặc do thông tin trong văn bản pháp luật quá dài dẫn tới mô hình bị nhiễu thông tin (lost in the middle).

---

## 4. Đề Xuất Cải Tiến Cho Hệ Thống RAG

1. **Tối ưu hóa Chunking:** Chia nhỏ văn bản pháp luật theo cấu trúc Điều/Khoản thay vì cắt cứng theo số ký tự. Điều này sẽ giúp cải thiện đáng kể chỉ số *Context Precision* và *Context Recall*.
2. **Nâng cấp Reranker:** Chuyển sang sử dụng các mô hình Reranker đa ngôn ngữ mạnh mẽ hơn (như BAAI/bge-reranker-large) để chọn lọc chính xác hơn 3-5 chunk thực sự liên quan.
3. **Prompt Engineering:** Cải tiến System Prompt của LLM để hướng dẫn mô hình trả lời ngắn gọn, cô đọng và chỉ sử dụng các dữ kiện có chứng cứ rõ ràng trong ngữ cảnh.
