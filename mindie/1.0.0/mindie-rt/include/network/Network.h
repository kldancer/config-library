/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDIE_NETWORK_H
#define ASCENDIE_NETWORK_H

#include "common/Common.h"
#include "common/Dims.h"
#include "common/NoCopy.h"
#include "network/layers/ActivationLayer.h"
#include "network/layers/ConstantLayer.h"
#include "network/layers/ConvolutionLayer.h"
#include "network/layers/PoolingLayer.h"
#include "network/layers/ElementWiseLayer.h"
#include "network/layers/MatrixMultiplyLayer.h"
#include "network/layers/ShuffleLayer.h"
#include "network/layers/ReduceLayer.h"
#include "network/layers/ShapeLayer.h"
#include "network/layers/ClipLayer.h"
#include "network/Tensor.h"

namespace AscendIE {
/**
 * Network definition
 */
class NetworkImpl;
class Network : public NoCopy {
public:
    ~Network() override;
    
    /**
     * @brief 设置network的name。
     *
     * 将传入的字符串设置为network的name。
     *
     * @param [in] name: network的name信息，不能为空，不能出现无效字符:
     * "\n", "\f", "\r", "\b", "\u007f"，不能超出长度上限4096。
     */
    void SetName(const char* name) noexcept;

    /**
     * @brief 获取network的name信息。
     *
     * @note 若还未调用SetName接口，则返回默认值"Network"。
     *
     * @return 返回network的name信息。
     */
    const char* GetName() noexcept;

    /**
     * @brief 向network中输入一个tensor。
     *
     * 根据输入的name、DataType和Dims构建传入network的tensor。
     *
     * @note 支持动态shape。
     *
     * @param [in] name: 输入tensor的name信息，不能为空，不能出现无效字符:
     * n, f, r, b, u007f，不能超出长度上限4096。
     * @param [in] type: 输入tensor的dType，支持float、float16、int8、int32、uint8、int16
     * uint16、uint32、int64、uint64、double、bool和string。
     * @param [in] dim: 输入tensor的维度信息。
     * @return 调用成功，返回新的输入tensor对象指针，失败返回nullptr。
     */
    Tensor *AddInput(const char* name, DataType type, Dims dim) noexcept;

    /**
     * @brief 标记network输出。
     *
     * 将传入的tensor标记为network的输出。
     *
     * @param [in] tensor: 要标记为输出的张量，不可以将network的输入标记为输出，不可以将if子图内的张量标记为输出。
     */
    void SetAsOutput(Tensor *tensor) noexcept;

    /**
     * @brief 获取network输入的数量。
     *
     * @return 返回network输入的数量。
     */
    size_t GetInputNum() const noexcept;

    /**
     * @brief 获取network输出的数量。
     *
     * @return 返回network输出的数量。
     */
    size_t GetOutputNum() const noexcept;

    /**
     * @brief 获取network的输出tensor。
     *
     * 根据索引获取network的输出tensor。
     *
     * @param [in] index: 要获取的输出tensor的索引。
     *
     * @return 返回对应索引的tensor。
     */
    Tensor *GetOutput(int32_t index) const noexcept;

    /**
     * @brief 向Network中添加一个constantlayer。
     *
     * 根据输入的dimensions和weights构造返回常量layer。
     *
     * @note 无。
     *
     * @param [in] dimensions: 输出的常量tensor的维度信息，每个元素表示对应维度的信息，最多支持8个元素，
     * 每个元素最大为UINT64上限，支持空shape的常量。
     * @param [in] weights: 输出常量tensor的权重信息。需满足：weights.count = dimensions所有维度的乘积。
     * @return 调用成功，返回新的ConstantLayer对象指针，失败返回nullptr。
     */
    ConstantLayer *AddConstant(Dims dimensions, WeightsBuf weights) noexcept;

    /**
     * @brief 向Network中添加一个convolution层layer。
     *
     * 根据输入的tensor和filter进行卷积运算。
     *
     * @note 支持动态shape。
     *
     * @param [in] input: 4维或5维的输入张量。DataType支持float32，Format支持[NCHW]，四维输入按照[batch, in_channels, in_height,
     * in_width]顺序存储。 in_height取值范围[1,500000]，in_width取值范围[1,500000]。五维输入按照[batch, in_channels, in_depth,
     * in_height, in_width]顺序存储。 in_depth取值范围[1,500000], in_height取值范围[1,500000]，in_width取值范围[1,500000]。
     * @param [in] numOutputMap: 卷积输出的特征图的数量。
     * @param [in] kernel: 四维输入时是卷积核的[H,W]维度。维度为2维，取值范围[1~100000, 1~100000]，五维输入时是卷积核的[D,H,W]维度。
     * 维度为3维，取值范围[1~100000, 1~100000, 1~100000]。
     * @param [in] weights: 卷积核的权重。
     * 四维输入需满足公式：weights.count = numOutputMap * (in_channels / groupNum) * kernel[0] *
     * kernel[1]。groupNum默认值为1。
     * 五维输入需满足公式：weights.count = numOutputMap * (in_channels / groupNum) * kernel[0] *
     * kernel[1] * kernel[2]。groupNum默认值为1。
     * @param [in] bias: 卷积的偏置权重。WeightsBuf{}表示没有偏置。不为空时，需满足bias.count = numOutputMap。
     * @return 调用成功，返回新的ConvolutionLayer对象指针，失败返回nullptr。
     */
    ConvolutionLayer *AddConvolution(Tensor *input, int32_t numOutputMap, Dims kernel, WeightsBuf weights,
        WeightsBuf bias) noexcept;

    /**
     * @brief 向network中添加一个ActivationLayer。
     *
     * 根据输入的tensor和激活函数进行激活计算。
     *
     * @note 支持动态shape。
     *
     * @param [in] input：输入张量。支持dType类型分别如下：
     * RELU：float32、float64、int32、uint8、int16、int8、int64、uint16和float16；
     * SIGMOID：float16、float32和double；
     * LEAKY_RELU：float32、float16和double；
     * HARD_SIGMOID：float16、float32和int32。
     * TANH：float32、float16和double；
     * SWISH：float32和float16；
     * GELU：float32和float16；
     * ELU：float32、float16和double；
     * SELU：float、float16、double、int32和int8；
     * SOFTPLUS：float32和float16；
     * MISH：float32和float16；
     * RELU6：支持所有的RealNumberType，即实数类型。
     * @param [in] kind：激活函数类型，支持RELU、SIGMOID、LEAKY_RELU、HARD_SIGMOID、TANH、SWISH、
     * GELU、ELU、SELU、SOFTPLUS、 MISH 和 RELU6。
     * @return 调用成功，返回新的ActivationLayer对象指针，失败返回nullptr。
     */
    ActivationLayer *AddActivation(Tensor *input, ActivationKind kind) noexcept;

    /**
     * @brief 向Network中添加一个PoolingLayer。
     *
     * 根据输入的input, kind, windowSize和outputSize进行池化运算。
     *
     * @note 支持动态shape。ADAPTIVE_AVERAGE场景windowSize传入{}和outputSize传入2D，其他场景只需传入windowSize。
     *
     * @param [in] input: ADAPTIVE_AVERAGE场景只支持2D输入，其他场景支持2D和3D输入。
     * dType支持float16, float32, double。
     * @param [in] kind: 池化类型，支持MAX, AVERAGE, ADAPTIVE_AVERAGE。
     * @param [in] windowSize: 池化窗口大小。ADAPTIVE_AVERAGE类型输入{}，MAX和AVERAGE类型设置维度为2/3维，
     * windowSize[i]取值范围[1~255]，windowSize[0] * windowSize[1]取值范围[1, 2560)。
     * @param [in] outputSize: 输出特征图大小。默认为Dims(), 池化类型为ADAPTIVE_AVERAGE时需传入具体的outputSize。
     * @return 调用成功，返回新的PoolingLayer对象指针，失败返回nullptr。
     */
    PoolingLayer *AddPooling(Tensor *input, PoolingKind kind, Dims windowSize, Dims outputSize = Dims()) noexcept;

    /**
     * @brief 向Network中添加一个ElementWiselayer。
     *
     * 实现对输入的input0和Input1进行指定op的双目运算。
     *
     * @note 支持动态shape。
     *
     * @param [in] input0: 输入张量，第一个操作数。
     * MAX和MIN运算支持dType：float16, float32, double, int32, int64;
     * BITWISE运算类型dType只支持bool;
     * 其他运算支持dType：float16, float32, double, int8, int16, int32, int64, uint8。
     * @param [in] input1: 输入张量，第二个操作数。
     * @param [in] op：运算类型。包括Max, Min, ADD, SUB等，详见ElementWiseOperation枚举类型。
     * @return 调用成功，返回新的ElementWiseLayer对象指针，失败返回nullptr。
     */
    ElementWiseLayer *AddElementWise(Tensor *input0, Tensor *input1, ElementWiseOperation op) noexcept;

    /**
     * @brief 向Network中添加一个矩阵乘法层，计算两个输入张量的矩阵乘法。
     *
     * @note：
     * 1.支持动态shape；
     * 2.暂不支持一个输入张量维度等于1，另一个张量大于2。
     * 3.需满足第一个张量的最后一维等于第二个张量的倒数第二维（如果有转置操作，需要先转置再校验该条件）
     *
     * @param [in] input0: 第一个输入张量，维度范围[1, 8]。dType支持uint8, int8, int16, int32, int64, float16, float,
     * double。
     * @param [in] type0: 第一个输入张量是否进行转置，NONE/TRANSPOSE。
     * @param [in] input1: 第二个输入张量，维度范围[1, 8]。dType支持uint8, int8, int16, int32, int64, float16, float,
     * double。
     * @param [in] type1: 第二个输入张量是否进行转置，NONE/TRANSPOSE。
     * @return 调用成功，返回新的MatrixMultiplyLayer对象指针，失败返回nullptr。
     */
    MatrixMultiplyLayer *AddMatrixMultiply(Tensor *input0, MatrixOperation type0, Tensor *input1,
        MatrixOperation type1) noexcept;

    /**
     * @brief 创建一个Shufflelayer，需要组合SetFirstTranspose/SetReshapeDimensions/SetSecondTranspose使用实现具体功能。
     * 支持的组合使用方式：
     * 1.搭配SetFirstTranspose是实现张量input转置。
     * 2.搭配SetReshapeDimensions是实现张量input形状变化到指定形状。
     * 3.搭配SetFirstTranspose + SetReshapeDimensions是实现张量input转置后再改变到指定形状。
     * 4.搭配SetReshapeDimensions + SetSecondTranspose是实现张量input改变到指定形状后再转置。
     * 5.搭配SetFirstTranspose + SetReshapeDimensions + SetSecondTranspose是实现张量input转置后再改变到指定形状，再转置。
     * 6.搭配SetFirstTranspose + SetSecondTranspose是实现张量input转置，再进行第二次转置。
     * 这个三个接口分别至多只允许调用一次
     *
     * @note 支持动态shape。
     *
     * @param [in] input: 输入张量。dType支持uint8, int8, int16, int32, int64, float16, float, double。
     * @return 调用成功，返回新的ShuffleLayer对象指针，失败返回nullptr。
     */
    ShuffleLayer *AddShuffle(Tensor *input) noexcept;

    /**
     * @brief 向Network中添加一个Reduce layer。
     *
     * 沿着tensor的指定维度，通过计算元素的最大、最小、平均值等操作，实现输入向量的降维。
     * 当ReduceOperation取值为AVG时，通过计算reduceAxes指定维度的平均值进行降维操作。
     * 当ReduceOperation取值为MAX时，通过计算reduceAxes指定维度的最大值进行降维操作。
     * 当ReduceOperation取值为MIN时，通过计算reduceAxes指定维度的最小值进行降维操作。
     * 当ReduceOperation取值为SUM时，通过计算reduceAxes指定维度的和进行降维操作。
     * 当ReduceOperation取值为STD时，通过计算reduceAxes指定维度的标准差进行降维操作。
     * 当ReduceOperation取值为PROD时，通过计算reduceAxes指定维度的元素乘积进行降维操作。
     * 当reduceAxes为空时，视为对输入tensor的所有维度进行降维操作
     * e.g. input shape = {2, 3, 3, 4}, axes = {} -> axes = {0, 1, 2, 3}
     * 当keepDimensions为true时，保持规约轴并将维度置为1
     * e.g. input shape = {2, 3, 3, 4}, axes = {1, 2} , keepDimensions = true -> outputshape = {2, 1, 1 ,4}
     * 当keepDimensions为false时，不保留规约轴
     * e.g. input shape = {2, 3, 3, 4}, axes = {1, 2} , keepDimensions = false -> outputshape = {2, 4}
     *
     * @note 支持动态shape
     *
     * @param [in] input: 一个ND Tensor作为输入Tensor，dtype支持float16，float32，int8，uint8。
     * @param [in] operation: 对向量内元素的计算方式，支持AVG、MAX、MIN、SUM、STD、PROD。
     * @param [in] reduceAxes: 需要进行reduce操作的维度，当reduceAxes为空时，对所有维度进行reduce操作。
     * @param [in] keepDimensions: 是否保留输出tensor的reduce维度，并设置为1。
     * @return 调用成功，返回新的ReduceLayer对象指针，失败返回nullptr。
     */
    ReduceLayer *AddReduce(Tensor *input, ReduceOperation operation, std::vector<int64_t> reduceAxes,
        bool keepDimensions) noexcept;

    /**
     * @brief 向Network中添加一个Shape layer。
     *
     * shape layer能够根据输入tensor，输出包含其shape信息的tensor。
     *
     * 默认输出类型为int32。
     *
     * @note ShapeLayer支持动态shape
     *
     * @param [in] input: 一个ND Tensor作为输入。
     * @return 调用成功，返回新的ShapeLayer对象指针，失败返回nullptr。
     */
    ShapeLayer *AddShape(Tensor *input) noexcept;

    /**
     * @brief 向network中添加一个ClipLayer。
     *
     * 将张量值剪辑到指定的最大值和最小值之间。
     *
     * @note 支持动态shape。
     *
     * @param [in] input: 输入张量，
     * dType支持float16,float32,double,int8,int16,int32,int64,uint8,uint16,uint32,uint64。
     * @param [in] min: 指定最小值。
     * @param [in] max: 指定最大值。
     * @return 调用成功，返回新的ClipLayer指针，失败返回nullptr。
     */
    ClipLayer *AddClip(Tensor *input, float min, float max) noexcept;

protected:
    std::unique_ptr<NetworkImpl> impl_;
};
}

#endif // ASCENDIE_NETWORK_H

