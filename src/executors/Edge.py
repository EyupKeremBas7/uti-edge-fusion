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
        try:
            self.edge_type = self.request.get_param("EdgeType")  # Try uppercase first
        except:
            self.edge_type = self.request.get_param("edgeType")  # Fallback to lowercase
        self.image = self.request.get_param("inputImageOne")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def edge(self, image):
        if self.edge_type == "LaplacianEdge":
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            if len(image.shape) == 3:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges
        elif self.edge_type == "SobelEdge":
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges)
            if len(image.shape) == 3:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            edges = cv2.Canny(gray, 100, 200)
            if len(image.shape) == 3:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges


    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.edge(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_edge(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
