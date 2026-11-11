from functools import lru_cache

from app.agents.IFD_Agent import IFDDetectionAgent
from app.agents.MFC_Agent import MFCDetectionAgent
from app.agents.models import ModelFactory
from app.agents.tools.mfc import BoChaSearchTool
from app.core.config import settings
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.ifd_detection import IFDDetectionService
from app.services.mfc_detection import MFCDetectionService
from app.services.prompt_service import PromptService
from app.services.visual.image_io import ImageService


@lru_cache
def get_detection_orchestrator() -> DetectionOrchestrator:
    model_factory = ModelFactory(settings)
    prompt_service = PromptService()
    image_service = ImageService(settings, model_factory, prompt_service)
    search_tool = BoChaSearchTool(settings)
    mfc_detection_service = MFCDetectionService(
        MFCDetectionAgent(
            image_service,
            settings,
            model_factory,
            search_tool=search_tool,
        )
    )
    ifd_agent = IFDDetectionAgent(settings, model_factory)
    ifd_detection_service = IFDDetectionService(image_service, ifd_agent)
    return DetectionOrchestrator(
        mfc_detection_service=mfc_detection_service,
        ifd_detection_service=ifd_detection_service,
    )
