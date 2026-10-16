from app.services.webpage_extraction import WebpageExtractionService


class FakeResponse:
    url = "https://example.com/news"
    headers = {"content-type": "text/html; charset=utf-8"}
    encoding = "utf-8"
    apparent_encoding = "utf-8"

    def __init__(self, html: str):
        self.html = html.encode("utf-8")

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=64000):
        yield self.html


class FakeNoCharsetResponse(FakeResponse):
    headers = {"content-type": "text/html"}
    encoding = "ISO-8859-1"
    apparent_encoding = "utf-8"


def test_webpage_extraction_extracts_text_and_representative_image(monkeypatch):
    html = """
    <html>
      <head>
        <title>Fallback title</title>
        <meta property="og:title" content="网页新闻标题" />
        <meta property="og:image" content="/images/main.jpg" />
      </head>
      <body>
        <article>
          <h1>网页新闻标题</h1>
          <p>第一段新闻正文。</p>
          <p>第二段包含图文信息。</p>
          <img src="/images/inline.jpg" />
        </article>
      </body>
    </html>
    """

    monkeypatch.setattr("app.services.webpage_extraction.requests.get", lambda *_, **__: FakeResponse(html))

    result = WebpageExtractionService().extract("example.com/news")

    assert result.url == "https://example.com/news"
    assert result.title == "网页新闻标题"
    assert "第一段新闻正文。" in result.text
    assert result.image_url == "https://example.com/images/main.jpg"
    assert result.images[1] == "https://example.com/images/inline.jpg"


def test_webpage_extraction_truncates_large_html_instead_of_failing(monkeypatch):
    html = """
    <html>
      <head><meta property="og:title" content="超大网页" /></head>
      <body><article><p>可抽取的正文。</p></article></body>
    </html>
    """ + ("x" * 1000)

    monkeypatch.setattr("app.services.webpage_extraction.requests.get", lambda *_, **__: FakeResponse(html))

    result = WebpageExtractionService(max_html_bytes=180).extract("example.com/news")

    assert result.title == "超大网页"
    assert "可抽取的正文。" in result.text
    assert result.metadata["html_truncated"] is True
    assert result.metadata["html_bytes_read"] == 180


def test_webpage_extraction_uses_meta_charset_article_body_and_article_image(monkeypatch):
    html = """
    <html>
      <head><meta charset="UTF-8"><title>中文新闻标题</title></head>
      <body>
        <header>
          <p>首页 导航 财经 体育 娱乐</p>
          <img src="/assets/header.png" width="164" height="50" alt="网站标志">
        </header>
        <div class="content_maincontent_content">
          <p>这是第一段中文新闻正文。</p>
          <p>这是第二段正文，不应混入导航文字。</p>
          <div><img src="/images/article-photo.jpg"></div>
        </div>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "app.services.webpage_extraction.requests.get",
        lambda *_, **__: FakeNoCharsetResponse(html),
    )

    result = WebpageExtractionService().extract("example.com/news")

    assert result.title == "中文新闻标题"
    assert result.text == "这是第一段中文新闻正文。\n这是第二段正文，不应混入导航文字。"
    assert "首页 导航" not in result.text
    assert result.image_url == "https://example.com/images/article-photo.jpg"
    assert "header.png" not in result.images
    assert result.metadata["encoding"].lower() == "utf-8"
    assert result.metadata["content_selector"] == "div.content_maincontent_content"


def test_webpage_extraction_prefers_article_over_larger_main_navigation(monkeypatch):
    html = """
    <html><head><title>Article</title></head><body>
      <main>
        <nav><p>Navigation item one with lots of unrelated text.</p></nav>
        <article><p>The actual article body.</p></article>
        <footer><p>Footer links with even more unrelated text than the article.</p></footer>
      </main>
    </body></html>
    """
    monkeypatch.setattr(
        "app.services.webpage_extraction.requests.get",
        lambda *_, **__: FakeResponse(html),
    )

    result = WebpageExtractionService().extract("example.com/article")

    assert result.text == "The actual article body."
    assert result.metadata["content_selector"] == "article"
