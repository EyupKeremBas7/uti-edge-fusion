from sdks.novavision.src.helper.package import PackageHelper
from components.EdgeFusion.src.models.PackageModel import EdgeConfigs, EdgeExecutor, EdgeOutputs, EdgeResponse, EdgeExecutor
from components.EdgeFusion.src.models.PackageModel import FusionExecutor, FusionOutputs, FusionResponse, FusionExecutor
from components.EdgeFusion.src.models.PackageModel import ConfigExecutor, OutputImageOne , OutputImageTwo , PackageModel,PackageConfigs

def build_response_edge(context):
    outputImage = OutputImageOne(value=context.image)
    outputs = EdgeOutputs(outputImage=outputImage)
    edgeResponse = EdgeResponse(outputs=outputs)
    edgeExecutor = EdgeExecutor(value=edgeResponse)
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
    fusionExecutor = FusionExecutor(value=fusionResponse)
    executor = ConfigExecutor(value=fusionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel