from functools import lru_cache

from app.agents.IFD_Agent import IFDDetectionAgent
from app.agents.MMD_Agent import MMDDetectionAgent
from app.agents.models import ModelFactory
from app.agents.tools.mmd import BoChaSearchTool, ImageOCREntityTool, MMDSearchQueryBuilder
from app.core.config import settings
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.ifd_detection import IFDDetectionService
from app.services.mmd_detection import MMDDetectionService
from app.services.prompt_service import PromptService
from app.services.visual.image_io import ImageService


@lru_cache
def get_detection_orchestrator() -> DetectionOrchestrator:
    model_factory = ModelFactory(settings)
    prompt_service = PromptService()
    image_service = ImageService(settings, model_factory, prompt_service)
    ocr_tool = (
        ImageOCREntityTool(image_service, settings, model_factory)
        if settings.mmd_ocr_enabled
        else None
    )
    search_tool = BoChaSearchTool(settings)
    mmd_detection_service = MMDDetectionService(
        MMDDetectionAgent(
            image_service,
            settings,
            model_factory,
            ocr_tool=ocr_tool,
            search_tool=search_tool,
            query_builder=MMDSearchQueryBuilder(),
        )
    )
    ifd_agent = IFDDetectionAgent(settings, model_factory)
    ifd_detection_service = IFDDetectionService(image_service, ifd_agent)
    return DetectionOrchestrator(
        mmd_detection_service=mmd_detection_service,
        ifd_detection_service=ifd_detection_service,
    )
