from functools import lru_cache

from app.agents.models import ModelFactory
from app.core.config import settings
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.vision.image import ImageDetectionService
from app.services.vision.image_io import ImageService
from app.services.multimodal import MultimodalDetectionService
from app.services.prompt_service import PromptService
from app.services.text.claim_extractor import ClaimExtractor
from app.services.text.evidence_search import EvidenceSearchService
from app.services.text.text_detection import TextDetectionService
from app.services.text.text_verdict import TextVerdictService
from app.services.vision.forensic_pipeline import ImageForensicPipeline


@lru_cache
def get_detection_orchestrator() -> DetectionOrchestrator:
    model_factory = ModelFactory(settings)
    prompt_service = PromptService()
    image_service = ImageService(settings, model_factory, prompt_service)
    claim_extractor = ClaimExtractor(settings, model_factory)
    evidence_search_service = EvidenceSearchService()
    text_verdict_service = TextVerdictService(settings, model_factory)
    text_detection_service = TextDetectionService(text_verdict_service, claim_extractor, evidence_search_service)
    image_forensic_pipeline = ImageForensicPipeline(settings, model_factory)
    image_detection_service = ImageDetectionService(image_service, image_forensic_pipeline)
    multimodal_detection_service = MultimodalDetectionService(
        text_detection_service=text_detection_service,
        image_detection_service=image_detection_service,
        image_service=image_service,
    )
    return DetectionOrchestrator(
        text_detection_service=text_detection_service,
        image_detection_service=image_detection_service,
        multimodal_detection_service=multimodal_detection_service,
    )
