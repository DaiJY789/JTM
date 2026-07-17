from functools import lru_cache

from app.agents.models import ModelFactory
from app.core.config import settings
from app.services.comprehensive_fusion import ComprehensiveFusionService
from app.services.crossmodal import CrossmodalConsistencyPipeline
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.prompt_service import PromptService
from app.services.textual.claim_extractor import ClaimExtractor
from app.services.textual.evidence_search import EvidenceSearchService
from app.services.textual.T_pipeline import TextualDetectionPipeline
from app.services.textual.textual_verdict import TextualVerdictService
from app.services.visual.V_pipeline import VisualForensicPipeline
from app.services.visual.image_io import ImageService
from app.services.visual.visual_verdict import VisualVerdictService


@lru_cache
def get_detection_orchestrator() -> DetectionOrchestrator:
    model_factory = ModelFactory(settings)
    prompt_service = PromptService()
    image_service = ImageService(settings, model_factory, prompt_service)
    claim_extractor = ClaimExtractor(settings, model_factory)
    evidence_search_service = EvidenceSearchService()
    textual_verdict_service = TextualVerdictService(settings, model_factory)
    textual_pipeline = TextualDetectionPipeline(textual_verdict_service, claim_extractor, evidence_search_service)
    visual_pipeline = VisualForensicPipeline(settings, model_factory)
    visual_verdict_service = VisualVerdictService(image_service, visual_pipeline)
    crossmodal_pipeline = CrossmodalConsistencyPipeline(image_service)
    comprehensive_fusion_service = ComprehensiveFusionService(image_service)
    return DetectionOrchestrator(
        textual_pipeline=textual_pipeline,
        visual_verdict_service=visual_verdict_service,
        crossmodal_pipeline=crossmodal_pipeline,
        comprehensive_fusion_service=comprehensive_fusion_service,
    )
