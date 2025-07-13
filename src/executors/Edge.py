import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.EdgeFusion.src.utils.response import build_response_edge
from components.EdgeFusion.src.models.PackageModel import PackageModel


class Edge(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.edge_type = self.request.get_param("edgeType")
        self.image = self.request.get_param("inputImageOne")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def edge(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.edge_type == "LaplacianEdge":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = cv2.convertScaleAbs(edges)
        elif self.edge_type == "SobelEdge":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            edges = cv2.magnitude(sobelx, sobely)
            edges = cv2.convertScaleAbs(edges)
        

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.edge(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_edge(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
