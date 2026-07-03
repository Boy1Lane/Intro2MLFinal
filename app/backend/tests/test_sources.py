import httpx

from app.backend.services import sources
from app.backend.services.sources import fetch_comments, resolve, list_sources
from app.backend.services.sources.generic import GenericAdapter
from app.backend.services.sources.tuoitre import TuoiTreAdapter
from app.backend.services.sources.vnexpress import VnExpressAdapter

VNE_URL = "https://vnexpress.net/tuong-quan-truoc-tran-5093229.html"
TT_URL = "https://tuoitre.vn/truc-tuyen-uc-ai-cap-100260703132538628.htm"


def _transport(payload: str, content_type="application/json"):
    def handler(request):
        return httpx.Response(200, headers={"content-type": content_type},
                              text=payload)
    return httpx.MockTransport(handler)


# ---- matching ------------------------------------------------------------
def test_vnexpress_matches_only_its_articles():
    a = VnExpressAdapter()
    assert a.matches(VNE_URL)
    assert not a.matches("https://tuoitre.vn/x-123456.htm")
    assert not a.matches("https://vnexpress.net/the-thao")  # no article id


def test_tuoitre_matches_only_its_articles():
    a = TuoiTreAdapter()
    assert a.matches(TT_URL)
    assert not a.matches(VNE_URL)


def test_generic_matches_everything():
    assert GenericAdapter().matches("https://any-blog.example/post")


def test_resolve_prefers_specific_then_generic():
    assert resolve(VNE_URL).name == "VnExpress"
    assert resolve(TT_URL).name == "TuoiTre"
    assert resolve("https://random.example/thread").name == "Generic"


# ---- fetching (mocked APIs) ---------------------------------------------
def test_vnexpress_parses_api_comments_and_strips_html():
    payload = ('{"data": {"items": ['
               '{"content": "B\\u00ecnh lu\\u1eadn m\\u1ed9t<br/>xu\\u1ed1ng d\\u00f2ng"},'
               '{"content": "b\\u00ecnh lu\\u1eadn hai &amp; ba"},'
               '{"content": "B\\u00ecnh lu\\u1eadn m\\u1ed9t<br/>xu\\u1ed1ng d\\u00f2ng"}'
               ']}}')
    out = VnExpressAdapter().fetch(VNE_URL, max_len=5000, min_len=3,
                                   timeout=5, _transport=_transport(payload))
    assert "Bình luận một xuống dòng" in out       # <br/> -> space
    assert "bình luận hai & ba" in out             # entity unescaped
    assert out.count("Bình luận một xuống dòng") == 1  # deduped
    assert "<br" not in " ".join(out)


def test_tuoitre_parses_data_json_string():
    # Tuổi Trẻ wraps the comment array as a JSON *string* under "Data"
    payload = ('{"Success": true, "Data": '
               '"[{\\"content\\": \\"\\\\u00dac thua ch\\\\u1eafc&nbsp;\\"}]"}')
    out = TuoiTreAdapter().fetch(TT_URL, max_len=5000, min_len=3,
                                 timeout=5, _transport=_transport(payload))
    assert out == ["Úc thua chắc"]


def test_dispatcher_routes_to_adapter():
    payload = '{"data": {"items": [{"content": "một bình luận hợp lệ"}]}}'
    out = fetch_comments(VNE_URL, _transport=_transport(payload))
    assert out == ["một bình luận hợp lệ"]


def test_list_sources_excludes_generic():
    names = {s["name"] for s in list_sources()}
    assert "VnExpress" in names and "TuoiTre" in names
    assert "Generic" not in names
