from sdks.novavision.src.helper.package import PackageHelper
from components.EdgeFusion.src.models.PackageModel import Edge, EdgeOutputs, EdgeResponse
from components.EdgeFusion.src.models.PackageModel import Fusion, FusionOutputs, FusionResponse
from components.EdgeFusion.src.models.PackageModel import ConfigExecutor, OutputImageOne , OutputImageTwo , PackageModel,PackageConfigs,RecognitionOutputs,RecognitionResponse,Recognition,OutputDetections

def build_response_edge(context):
    outputImageOne = OutputImageOne(value=context.image)
    outputs = EdgeOutputs(outputImageOne=outputImageOne)
    edgeResponse = EdgeResponse(outputs=outputs)
    edgeExecutor = Edge(value=edgeResponse)
    executor = ConfigExecutor(value=edgeExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_fusion(context):
    outputImageOne = OutputImageOne(value=context.image)
    outputImageTwo = OutputImageTwo(value=context.image)
    outputs = FusionOutputs(outputImageOne=outputImageOne, outputImageTwo=outputImageTwo)
    fusionResponse = FusionResponse(outputs=outputs)
    fusionExecutor = Fusion(value=fusionResponse)
    executor = ConfigExecutor(value=fusionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_recognition(context):
    outputDetections = OutputDetections(value=context.detection)
    outputs = RecognitionOutputs(outputDetections=outputDetections)
    fusionResponse = RecognitionResponse(outputs=outputs)
    fusionExecutor = Recognition(value=fusionResponse)
    executor = ConfigExecutor(value=fusionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

