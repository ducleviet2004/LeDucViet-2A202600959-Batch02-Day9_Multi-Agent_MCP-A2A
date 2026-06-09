"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vnexpress.net/dai-an-ma-tuy-voi-30-an-tu-hinh-5079661.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
    "https://vnexpress.net/hai-thanh-nien-duong-tinh-voi-ma-tuy-thong-chot-dam-nga-csgt-5082931.html",
    "https://vnexpress.net/ma-tuy-tan-pha-tim-mach-the-nao-5077415.html",
    "https://vnexpress.net/nhieu-nguoi-nuoc-ngoai-phe-ma-tuy-trong-khach-san-o-tp-hcm-5082175.html"
]


async def crawl_article(crawler, url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content sử dụng crawl4ai.
    """
    try:
        result = await crawler.arun(url=url)
        r = result[0]
        
        # Trích xuất tiêu đề bằng regex từ html
        title_match = re.search(r"<title>(.*?)</title>", r.html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "Unknown Title"
        
        # Làm sạch tiêu đề khỏi hậu tố báo chí nếu có
        title = re.sub(r"\s+-\s+Báo\s+VnExpress.*", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s+-\s+VnExpress.*", "", title, flags=re.IGNORECASE)
        
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": r.markdown,
        }
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return {
            "url": url,
            "title": "Error Title",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"Error loading content: {e}",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS sử dụng crawl4ai."""
    setup_directory()
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await crawl_article(crawler, url)

            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARN] Hay dien ARTICLE_URLS truoc khi chay!")
    else:
        asyncio.run(crawl_all())

