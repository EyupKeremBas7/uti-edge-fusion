import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.EdgeFusion.src.utils.response import build_response_recognition
from components.EdgeFusion.src.models.PackageModel import PackageModel


class Recognition(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.imageOne = self.request.get_param("inputImageOne")


    @staticmethod
    def bootstrap(config: dict) -> dict:
        sayac = 0
        return {"sayac":sayac}

    def recognition(self, image_one: np.ndarray, image_two: np.ndarray) -> np.ndarray:
        pass

    def run(self):
        img1 = Image.get_frame(img=self.imageOne, redis_db=self.redis_db)
        img1.value = self.recognition(img1.value)
        self.image = Image.set_frame(img=img1, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_recognition(context=self)
        self.bootsrap["sayac"] += 1
        print(self.bootsrap["sayac"])
        return packageModel
    
if "__main__" == __name__:
    Executor(sys.argv[1]).run()
        
