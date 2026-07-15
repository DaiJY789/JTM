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
