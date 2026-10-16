import base64
from io import BytesIO

from PIL import Image

from app.core.config import Settings
from app.services.visual.image_io import ImageService


def image_bytes(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "#5a7fa3").save(buffer, format="PNG")
    return buffer.getvalue()


def test_mmd_image_input_is_resized_and_encoded_as_jpeg():
    service = ImageService(Settings(mmd_image_max_side=768))

    part = service.build_mmd_image_content_part(image_bytes((1600, 1200)))

    assert part is not None
    url = part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    encoded = url.split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (768, 576)


def test_mmd_image_input_rejects_model_invalid_tiny_image():
    service = ImageService(Settings(mmd_image_max_side=768))

    assert service.build_mmd_image_content_part(image_bytes((1, 100))) is None


def test_ocr_image_input_keeps_more_detail_than_mmd_input():
    service = ImageService(
        Settings(mmd_image_max_side=768, mmd_ocr_image_max_side=1600)
    )

    part = service.build_ocr_image_content_part(image_bytes((2400, 1800)))

    assert part is not None
    encoded = part["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (1600, 1200)
