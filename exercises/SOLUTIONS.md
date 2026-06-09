# Đáp Án Chi Tiết Bài Tập & Câu Hỏi Ôn Tập Codelab A2A

Tài liệu này cung cấp lời giải và câu trả lời chi tiết cho toàn bộ bài tập thực hành cũng như các câu hỏi lý thuyết xuất hiện trong [CODELAB.md](file:///c:/Users/ADMIN/Downloads/Day9/LeDucViet-2A202600959-Batch02-Day9_Multi-Agent_MCP-A2A/CODELAB.md).

---

## Phần 1: Direct LLM Calling

### 1. LLM được khởi tạo như thế nào? (Hàm `get_llm()`)
Hàm `get_llm()` trong [llm.py](file:///c:/Users/ADMIN/Downloads/Day9/LeDucViet-2A202600959-Batch02-Day9_Multi-Agent_MCP-A2A/common/llm.py) khởi tạo đối tượng `ChatOpenAI` kết nối qua API của OpenRouter:
```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=1000,
    )
```
*   `model`: Xác định model LLM sử dụng thông qua biến môi trường.
*   `openai_api_base`: Trỏ về endpoint của OpenRouter thay vì OpenAI gốc.
*   `temperature`: Cấu hình độ sáng tạo (0.3 giúp phản hồi ổn định và chính xác hơn).
*   `max_tokens`: Giới hạn độ dài output để kiểm soát chi phí.

### 2. Message gửi đến LLM có cấu trúc gì?
Message gửi đến LLM là một danh sách chứa các đối tượng Message của LangChain, ví dụ:
```python
messages = [
    SystemMessage(content="Bạn là chuyên gia pháp lý..."),
    HumanMessage(content="Thời hiệu khởi kiện...")
]
```

### 3. Tại sao cần có `SystemMessage` và `HumanMessage`?
*   `SystemMessage`: Định hình vai trò (role), phong cách, luật lệ ứng xử và giới hạn phạm vi kiến thức cho LLM (ví dụ: "Bạn là luật sư chuyên nghiệp").
*   `HumanMessage`: Chứa câu hỏi cụ thể, yêu cầu thực tế hoặc ngữ cảnh do người dùng cung cấp cần LLM xử lý.

---

## Phần 2: LLM + RAG & Tools

### 1. Hàm `@tool` decorator được dùng ở đâu?
Decorator `@tool` của LangChain được dùng ngay phía trên các hàm Python thông thường (như `search_legal_knowledge` và `check_statute_of_limitations`) để tự động chuyển đổi chúng thành cấu trúc Schema (JSON schema) định nghĩa tool mà LLM có thể hiểu và gọi được.

### 2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
`LEGAL_KNOWLEDGE` là một danh sách các dictionary đại diện cho một database tri thức dạng đơn giản:
```python
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": "..."
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": "..."
    }
]
```
Mỗi phần tử gồm có định danh (`id`), danh sách từ khóa tìm kiếm (`keywords`) và nội dung chi tiết kiến thức pháp lý (`text`).

### 3. LLM được bind với tools ra sao?
Sử dụng phương thức `.bind_tools(tools)` trên đối tượng LLM để truyền danh sách cấu trúc các tool có sẵn cho LLM:
```python
llm_with_tools = llm.bind_tools(tools)
```
Khi gọi `llm_with_tools.invoke()`, LLM sẽ biết về sự tồn tại của các tool này và có thể đưa ra quyết định gọi chúng dưới dạng `tool_calls`.

---

## Phần 3: Single Agent với ReAct

### 1. Magic function `create_react_agent()` là gì?
`create_react_agent()` là một helper function được cung cấp bởi thư viện LangGraph/LangChain. Nó tự động tạo ra một đồ thị StateGraph thực thi mô hình ReAct (Reasoning + Acting) hoàn chỉnh: nhận message -> gọi LLM quyết định -> chạy tool thích hợp -> quan sát kết quả -> lặp lại cho tới khi có câu trả lời cuối cùng mà lập trình viên không cần viết vòng lặp thủ công.

### 2. So sánh với Stage 2 (LLM + RAG / Tools)
*   **Stage 2:** Vòng lặp kiểm tra và gọi tool phải được code thủ công bằng tay (chỉ thực hiện tối đa 1 lượt gọi tool rồi tổng hợp kết quả).
*   **Stage 3:** Agent tự động điều hướng hoàn toàn. Nó có thể quyết định gọi tool 1, nhận kết quả, sau đó gọi tiếp tool 2 hoặc tự kết thúc vòng lặp dựa trên tư duy logic của mô hình.

---

## Phần 4: Multi-Agent In-Process

### 1. `class State(TypedDict)` trong LangGraph là gì?
`State` định nghĩa cấu trúc lưu trữ trạng thái (dữ liệu chia sẻ) của toàn bộ đồ thị. Tất cả các nodes trong graph đọc dữ liệu từ `State` này và trả về các thay đổi (update) để ghi đè hoặc bổ sung vào `State`.

### 2. Reducer `_last_wins` là gì?
Là một hàm reducer được truyền vào thông qua `Annotated` để chỉ định cách giải quyết xung đột khi có nhiều nodes ghi đè dữ liệu vào cùng một trường thông tin cùng một lúc (ở đây, giá trị được ghi sau cùng sẽ đè lên giá trị trước).

### 3. `Send` API hoạt động như thế nào?
`Send(node_name, state_argument)` cho phép đồ thị phân nhánh động tại thời điểm chạy (dynamic routing). Nó tạo ra một instance chạy song song của node `node_name` với tham số đầu vào được trích xuất từ trạng thái hiện tại. Nhờ đó, `tax_agent`, `compliance_agent`, và `privacy_agent` có thể được thực thi song song hoàn toàn độc lập.

---

## Phần 5: Distributed A2A System

### 1. Phân tích request đi qua bao nhiêu Hops?
Request từ client đi qua các bước (Hops) sau:
1.  `test_client.py` ──(HTTP)──> `Customer Agent` (Port 10100) (Hop 1)
2.  `Customer Agent` ──(HTTP Discover)──> `Registry` (Port 10000) để tìm Law Agent.
3.  `Customer Agent` ──(HTTP A2A)──> `Law Agent` (Port 10101) (Hop 2)
4.  `Law Agent` ──(HTTP Discover)──> `Registry` để tìm Tax & Compliance Agents.
5.  `Law Agent` ──(HTTP A2A - Parallel)──> `Tax Agent` (Port 10102) & `Compliance Agent` (Port 10103) (Hop 3 - song song)
6.  `Tax Agent` & `Compliance Agent` xử lý độc lập và trả về kết quả cho `Law Agent`.
7.  `Law Agent` tổng hợp phân tích, phản hồi lại cho `Customer Agent`.
8.  `Customer Agent` trả kết quả cuối cùng cho `test_client.py`.

### 2. Hệ thống có crash không khi dừng Tax Agent? Tại sao?
*   **Kết quả:** Hệ thống **không bị crash hoàn toàn**.
*   **Giải thích:** Nhờ cơ chế bắt lỗi ngoại lệ (exception handling) và cô lập dịch vụ trong `law_agent/graph.py` (hàm `call_tax` bọc trong khối `try-except`), khi Tax Agent offline, Law Agent sẽ bắt được lỗi kết nối, ghi nhận `[Tax analysis unavailable: ...]` vào state và tiếp tục chạy bình thường để trả về phần phân tích của Law và Compliance Agent mà không làm sập toàn bộ luồng xử lý. Điều này thể hiện tính chất chống chịu lỗi (fault-tolerance) rất tốt của kiến trúc Distributed A2A.

---

## Phần 6: Câu Hỏi Ôn Tập Tổng Kết

### 1. Khi nào nên dùng single agent thay vì multi-agent?
*   **Nên dùng Single Agent:** Cho các tác vụ đơn giản, cùng thuộc một phạm vi kiến thức chuyên môn, các bước xử lý ngắn và không đòi hỏi quá nhiều chỉ dẫn/prompts chuyên biệt. Việc triển khai single agent sẽ tiết kiệm chi phí gọi API, giảm độ trễ (latency) và dễ debug hơn.
*   **Nên dùng Multi-Agent:** Khi bài toán phức tạp, liên quan đến nhiều lĩnh vực chuyên môn sâu khác nhau (như luật tổng quát, thuế, bảo mật), yêu cầu các agent có vai trò và công cụ riêng biệt, hoặc khi cần xử lý song song nhiều nhánh tác vụ để tối ưu hóa thời gian xử lý.

### 2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
A2A không chỉ là giao thức truyền dữ liệu thuần túy (như REST/gRPC) mà nó đóng vai trò là một chuẩn giao tiếp hướng tác vụ (agentic-native protocol) với:
*   **Ngữ cảnh nghiệp vụ chuẩn hóa (Standardized Cards & Messages):** Định nghĩa sẵn cấu trúc Agent Card, Task, Message, và Part để các hệ thống AI tự hiểu khả năng của nhau.
*   **Tích hợp Telemetry & Tracing:** Hỗ trợ lan truyền mã định danh ngữ cảnh (`context_id`, `trace_id`) và kiểm soát độ sâu gọi đệ quy (`delegation_depth`) trực tiếp trong đặc tả giao thức.
*   **Dynamic Discovery tích hợp:** Hỗ trợ các cơ chế đăng ký khả năng nghiệp vụ của Agent (capability-based routing) thông qua Registry.

### 3. Làm thế nào để ngăn chặn vòng lặp ủy quyền vô hạn (infinite delegation loops) trong A2A?
Giao thức A2A ngăn chặn điều này bằng cách sử dụng trường độ sâu ủy quyền **`delegation_depth`** kết hợp với cấu hình **`MAX_DELEGATION_DEPTH`** (trong bài lab đặt là `3`):
*   Mỗi khi một Agent ủy quyền cho một Agent khác qua A2A, chỉ số `depth` sẽ tăng lên 1 đơn vị.
*   Trước khi thực hiện cuộc gọi tiếp theo, Agent kiểm tra nếu `depth >= MAX_DELEGATION_DEPTH` sẽ từ chối gọi tiếp và trả về lỗi hoặc kết quả mặc định ngay lập tức.

### 4. Tại sao cần Registry service? Có thể hardcode URLs không?
*   **Tại sao cần:** Registry đóng vai trò làm Service Discovery trung tâm. Nó cho phép các Agent tự đăng ký địa chỉ mạng (IP, Port) và khả năng của mình một cách linh động khi khởi động.
*   **Hardcode URLs:** Có thể hardcode URLs đối với hệ thống thử nghiệm nhỏ. Tuy nhiên, trong môi trường sản xuất (production), việc hardcode khiến hệ thống không thể scale-up (chạy nhiều bản sao để giảm tải), không thể tự khắc phục khi một máy chủ bị thay đổi địa chỉ IP, và làm tăng sự ràng buộc chặt chẽ (tight coupling) giữa các thành phần.

---

## Bài Tập Cộng Điểm: Đo Đạc & Tối Ưu Latency Stage 5

*   **Latency đo đạc ban đầu:** **33.42 giây** (với mô hình `google/gemini-2.5-flash` chạy tuần tự qua các node phân tích và định tuyến LLM).
*   **Phương án tối ưu đã triển khai:**
    1.  **Song song hóa đồ thị:** Đưa node `analyze_law` và `check_routing` chạy song song tại điểm bắt đầu (`START`), tiết kiệm 1 bước LLM tuần tự.
    2.  **Định tuyến lai bằng từ khóa (Keyword-based routing):** Viết logic kiểm tra từ khóa nhanh trong `check_routing`. Nếu khớp từ khóa thuế/tuân thủ, hệ thống định tuyến thẳng (mất 0ms) thay vì phải gọi LLM phân loại.
*   **Latency sau tối ưu:** **30.95 giây** (Giảm được **2.47 giây**, tốc độ hệ thống tăng **7.4%**).
