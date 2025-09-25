1. include/ 头文件目录
status_code.hpp
定义通用的状态码、错误码等。

data/

tensor.hpp：张量（Tensor）数据结构定义，核心数据存储与操作。
tensor_util.hpp：张量相关的工具函数，如数据变换、初始化等。
tensor_pool.hpp：张量池，负责张量的内存复用与管理。
load_data.hpp：数据加载相关函数。
layer/

abstract/
layer.hpp：Layer 基类，所有算子的抽象接口。
param_layer.hpp、non_param_layer.hpp：有参数/无参数 Layer 的抽象基类。
layer_factory.hpp：Layer 工厂，负责算子的注册与创建。
details/
各种具体算子的实现头文件，如 convolution、relu、softmax、cat、flatten、yolo_detect 等。
parser/

parse_expression.hpp：表达式解析相关，支持 ExpressionLayer 的表达式运算。
runtime/

ir.h：IR（中间表示）相关定义。
runtime_ir.hpp：推理主流程与计算图（RuntimeGraph）定义。
runtime_op.hpp：计算图节点（RuntimeOperator）结构体定义。
runtime_operand.hpp：节点输入输出（Operand）结构体定义。
runtime_attr.hpp：节点权重、属性结构体定义。
runtime_datatype.hpp：数据类型定义。
runtime_parameter.hpp：参数类型定义。
store_zip.hpp：模型文件压缩/解压相关。
utils/

math/fmath.hpp：数学相关工具函数。
time/time_logging.hpp：时间统计与日志工具。
2. source/ 源码目录
ir.cpp、runtime_ir.cpp、runtime_op.cpp、runtime_attr.cpp、runtime_operand.cpp、runtime_parameter.cpp
对应 include/runtime/ 下各头文件的实现，负责模型加载、计算图构建、推理调度、参数和权重管理等。

tensor.cpp、tensor_utils.cpp、tensor_pool.cpp
张量相关实现，包括数据结构、工具函数和张量池。

load_data.cpp
数据加载实现。

layer/abstract/
Layer 抽象基类和工厂的实现。

layer/details/
各种算子的实现，如卷积、激活、池化、拼接、表达式、YOLO 检测等。

parser/parse_expression.cpp
表达式解析实现。

3. test/ 测试目录
test_resnet.cpp、test_yolov5.cpp
针对不同模型的推理测试代码。

image_util.cpp/hpp
图像预处理、加载等工具函数。

4. model_file/ 模型文件目录
存放 pnnx 导出的模型结构和权重文件（.param/.bin），以及测试图片。
5. log/
日志输出目录。

依赖关系简述
runtime/ 依赖 data/（张量）、layer/（算子）、parser/（表达式）、utils/（工具）。
layer/ 依赖 data/（张量）、utils/（数学工具）。
test/ 依赖 runtime/、layer/、data/。
parser/ 主要为 ExpressionLayer 服务。

本项目采用模块化设计，核心为 runtime（推理主流程）、layer（算子实现）、data（张量与内存）、parser（表达式）、utils（工具），各模块通过头文件和实现文件解耦，便于维护和扩展。
7. CMakeLists.txt、main.cpp
CMakeLists.txt：项目构建配置。
main.cpp：程序入口
