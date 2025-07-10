import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.EdgeFusion.src.utils.response import build_response_fusion
from components.EdgeFusion.src.models.PackageModel import PackageModel


class Fusion(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.fusion_direction = self.request.get_param("FusionDirection")
        self.imageOne = self.request.get_param("inputImageOne")
        self.imageTwo = self.request.get_param("inputImageTwo")


    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def fusion(self, image_one: np.ndarray, image_two: np.ndarray) -> np.ndarray:
        h1, w1 = image_one.shape[:2]
        h2, w2 = image_two.shape[:2]

        if self.fusion_direction == "VerticalFusion":
            if w1 != w2:
                new_w = min(w1, w2)
                image_one = cv2.resize(image_one, (new_w, int(h1 * new_w / w1)))
                image_two = cv2.resize(image_two, (new_w, int(h2 * new_w / w2)))
            return cv2.vconcat([image_one, image_two])
        
        elif self.fusion_direction == "HorizontalFusion":
            if h1 != h2:
                new_h = min(h1, h2)
                image_one = cv2.resize(image_one, (int(w1 * new_h / h1), new_h))
                image_two = cv2.resize(image_two, (int(w2 * new_h / h2), new_h))
            return cv2.hconcat([image_one, image_two])
        
        return image_one

    def run(self):
        img1 = Image.get_frame(img=self.imageOne, redis_db=self.redis_db)
        img2 = Image.get_frame(img=self.imageTwo, redis_db=self.redis_db)
    
        img1.value = self.fusion(img1.value, img2.value)
        self.image = Image.set_frame(img=img1, package_uID=self.uID, redis_db=self.redis_db)
        
        packageModel = build_response_fusion(context=self)
        return packageModel
    
if "__main__" == __name__:
    Executor(sys.argv[1]).run()
        