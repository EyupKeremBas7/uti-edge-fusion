from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config,Detection

class InputImageOne(Input):
    name: Literal["inputImageOne"] = "inputImageOne"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class InputImageTwo(Input):
    name: Literal["inputImageTwo"] = "inputImageTwo"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class OutputImageOne(Output):
    name: Literal["outputImageOne"] = "outputImageOne"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"

class OutputImageTwo(Output):
    name: Literal["outputImageTwo"] = "outputImageTwo"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class VerticalFusion(Config):
    name: Literal["VerticalFusion"] = "VerticalFusion"
    value: Literal["VerticalFusion"] = "VerticalFusion"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "Vertical"

class HorizontalFusion(Config):
    name: Literal["HorizontalFusion"] = "HorizontalFusion"
    value: Literal["HorizontalFusion"] = "HorizontalFusion"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "Horizontal"

class ConfigFusionDirection(Config):
    """
        Determines the orientation for fusing the two images.
    """
    name: Literal["FusionDirection"] = "FusionDirection"
    value: Union[VerticalFusion, HorizontalFusion]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title = "Fusion Direction"

class FusionConfigs(Configs):
    fusionDirection: ConfigFusionDirection

class FusionInputs(Inputs):
    inputImageOne: InputImageOne
    inputImageTwo : InputImageTwo

class FusionOutputs(Outputs):
    outputImageOne: OutputImageOne
    outputImageTwo : OutputImageTwo

class FusionRequest(Request):
    inputs: Optional[FusionInputs]
    configs: FusionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class FusionResponse(Response):
    outputs: FusionOutputs


class Fusion(Config):
    name: Literal["Fusion"] = "Fusion"
    value: Union[FusionRequest, FusionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Fusion"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class LaplacianEdge(Config):
    name: Literal["LaplacianEdge"] = "LaplacianEdge"
    value: Literal["LaplacianEdge"] = "LaplacianEdge"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "LaplacianEdge"

class SobelEdge(Config):
    name: Literal["SobelEdge"] = "SobelEdge"
    value: Literal["SobelEdge"] = "SobelEdge"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    class Config:
        title = "SobelEdge"

class ConfigEdgeType(Config):
    """
        Determines the algoritm of edging.
    """
    name: Literal["edgeType"] = "edgeType"
    value: Union[SobelEdge, LaplacianEdge]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    class Config:
        title = "Edging Algorithm"

class EdgeInputs(Inputs):
    inputImageOne: InputImageOne

class EdgeConfigs(Configs):
    edgeType: ConfigEdgeType

class EdgeOutputs(Outputs):
    outputImageOne: OutputImageOne

class EdgeRequest(Request):
    inputs: Optional[EdgeInputs]
    configs: EdgeConfigs
    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class EdgeResponse(Response):
    outputs: EdgeOutputs


class Edge(Config):
    name: Literal["Edge"] = "Edge"
    value: Union[EdgeRequest,EdgeResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Edge"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class CustomDetection(Detection):
    imgUID: str

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[CustomDetection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"

class RecognitionInputs(Inputs):
    inputImageOne: InputImageOne


class RecognitionOutputs(Outputs):
    outputDetections: OutputDetections

class Number(Config):
    name: Literal["Number"] = "Number"
    value: int = Field(ge=5.0, le=0.0,default=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    class Config:
        title = "Number"

class RecognitionConfigs(Configs):
    number : Number

class RecognitionRequest(Request):
    inputs: Optional[RecognitionInputs]
    configs: RecognitionConfigs
    class Config:
        json_schema_extra = {
            "target": "configs"
        }

class RecognitionResponse(Response):
    outputs: RecognitionOutputs

class Recognition(Config):
    name: Literal["Recognition"] = "Recognition"
    value: Union[RecognitionRequest,RecognitionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Recognition"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[Edge,Fusion,Recognition]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Type"

    

class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["EdgeFusion"] = "EdgeFusion"
