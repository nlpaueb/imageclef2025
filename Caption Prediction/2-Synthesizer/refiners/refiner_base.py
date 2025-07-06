from abc import ABC, abstractmethod

class CaptionRefiner(ABC):
    @abstractmethod
    def refine_caption(self, image, draft_caption, neighbor_image=None, neighbor_caption=None):
        pass