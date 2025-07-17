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

    def edge(self, image):
        image_copy = image.copy()
        
        if self.edge_type == "LaplacianEdge":
            if len(image_copy.shape) == 3:
                gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_copy.copy()
            edges = cv2.Laplacian(gray, ddepth=-1, ksize=3)
            if len(image_copy.shape) == 3:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges
            
        elif self.edge_type == "SobelEdge":
            if len(image_copy.shape) == 3:
                gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_copy.copy()
            sobelx = cv2.Sobel(gray, ddepth=-1, dx=1, dy=0, ksize=3)
            sobely = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1, ksize=3)
            magnitude = cv2.magnitude(sobelx.astype(float), sobely.astype(float))
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = np.uint8(magnitude)
            if len(image_copy.shape) == 3:
                magnitude = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
            return magnitude
            
        else:
            return image_copy


    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.edge(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_edge(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
