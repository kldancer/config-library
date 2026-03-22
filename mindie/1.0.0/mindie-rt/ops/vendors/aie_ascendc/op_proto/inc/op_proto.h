#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AllGather)
    .INPUT(sendData, ge::TensorType::ALL())
    .INPUT(commArgs, ge::TensorType::ALL())
    .OUTPUT(recvData, ge::TensorType::ALL())
    .REQUIRED_ATTR(rank, Int)
    .REQUIRED_ATTR(rankSize, Int)
    .REQUIRED_ATTR(magicId, Int)
    .OP_END_FACTORY_REG(AllGather);

REG_OP(Arange)
    .INPUT(startNum, ge::TensorType::ALL())
    .INPUT(endNum, ge::TensorType::ALL())
    .INPUT(stepForward, ge::TensorType::ALL())
    .INPUT(dtypeOutput, ge::TensorType::ALL())
    .OUTPUT(outRange, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(Arange);

REG_OP(FindMax)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(values, ge::TensorType::ALL())
    .OUTPUT(indices, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(FindMax);

REG_OP(FlashAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(seqLen, ge::TensorType::ALL())
    .INPUT(batch, ge::TensorType::ALL())
    .INPUT(spMask, ge::TensorType::ALL())
    .OUTPUT(attnOut, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(FlashAttention);

REG_OP(FlashAttentionSoftmaxFp32)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(qSeqLen, ge::TensorType::ALL())
    .INPUT(kvSeqLen, ge::TensorType::ALL())
    .OPTIONAL_INPUT(mask, ge::TensorType::ALL())
    .OUTPUT(attnOut, ge::TensorType::ALL())
    .ATTR(embedSize, Int, 64)
    .ATTR(tor, Float, 0.125)
    .OP_END_FACTORY_REG(FlashAttentionSoftmaxFp32);

REG_OP(ForwardRasterize)
    .INPUT(face_vertices, ge::TensorType::ALL())
    .INPUT(out_shape, ge::TensorType::ALL())
    .OUTPUT(output_buffer, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(ForwardRasterize);

REG_OP(GenSeqLen)
    .INPUT(attnMask, ge::TensorType::ALL())
    .OUTPUT(seqLenAlign, ge::TensorType::ALL())
    .OUTPUT(seqLenOri, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GenSeqLen);

REG_OP(IndexFill)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(dim, ge::TensorType::ALL())
    .INPUT(index, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(IndexFill);

REG_OP(KvCache)
    .INPUT(newKV, ge::TensorType::ALL())
    .INPUT(layerId, ge::TensorType::ALL())
    .INPUT(CacheIn, ge::TensorType::ALL())
    .INPUT(tokenOffset, ge::TensorType::ALL())
    .INPUT(seqLen, ge::TensorType::ALL())
    .OUTPUT(cacheOut, ge::TensorType::ALL())
    .ATTR(max_seqlen, Int, 32768)
    .OP_END_FACTORY_REG(KvCache);

REG_OP(MatmulAll)
    .INPUT(inputX, ge::TensorType::ALL())
    .INPUT(inputWeight, ge::TensorType::ALL())
    .OPTIONAL_INPUT(inputBias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(inputExtra, ge::TensorType::ALL())
    .OUTPUT(outMatrix, ge::TensorType::ALL())
    .ATTR(needTrans, Int, 0)
    .ATTR(withBias, Int, 0)
    .ATTR(operateType, Int, 0)
    .OP_END_FACTORY_REG(MatmulAll);

REG_OP(MindieRTPlugin)
    .DYNAMIC_INPUT(x, ge::TensorType::ALL())
    .DYNAMIC_OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(name, String)
    .REQUIRED_ATTR(version, String)
    .REQUIRED_ATTR(serialize_info, String)
    .REQUIRED_ATTR(input_number, Int)
    .OP_END_FACTORY_REG(MindieRTPlugin);

REG_OP(PadInput)
    .INPUT(hiddenStates, ge::TensorType::ALL())
    .INPUT(seqLen, ge::TensorType::ALL())
    .INPUT(batch, ge::TensorType::ALL())
    .INPUT(maxSeqLen, ge::TensorType::ALL())
    .INPUT(hiddenSize, ge::TensorType::ALL())
    .OUTPUT(outStates, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(PadInput);

REG_OP(PpMatmulI8)
    .INPUT(a, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .OUTPUT(c, ge::TensorType::ALL())
    .ATTR(transposeA, Bool, false)
    .ATTR(transposeB, Bool, false)
    .ATTR(withBias, Bool, true)
    .ATTR(enDequant, Bool, true)
    .ATTR(oriShape, ListInt, {})
    .ATTR(tilingK, Int, 0)
    .ATTR(tilingN, Int, 0)
    .OP_END_FACTORY_REG(PpMatmulI8);

REG_OP(PpmatmulCube)
    .INPUT(matrixA, ge::TensorType::ALL())
    .INPUT(matrixB, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(matrixC, ge::TensorType::ALL())
    .ATTR(transA, Int, 0)
    .ATTR(transB, Int, 0)
    .OP_END_FACTORY_REG(PpmatmulCube);

REG_OP(Quant)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(z, ge::TensorType::ALL())
    .ATTR(scale, Float, 82.96)
    .ATTR(offset, Float, -36)
    .ATTR(quant_min, Int, -128)
    .OP_END_FACTORY_REG(Quant);

REG_OP(ReshapeAndCache)
    .INPUT(keyInput, ge::TensorType::ALL())
    .INPUT(valueInput, ge::TensorType::ALL())
    .INPUT(keyCache, ge::TensorType::ALL())
    .INPUT(valueCache, ge::TensorType::ALL())
    .INPUT(slotMapping, ge::TensorType::ALL())
    .OUTPUT(keyCache, ge::TensorType::ALL())
    .OUTPUT(valueCache, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(ReshapeAndCache);

REG_OP(RmsNormAie)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(g, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(RmsNormAie);

REG_OP(RmsNormQuant)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(g, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .OPTIONAL_INPUT(scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(offset, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(eps, Float, 1e-05)
    .ATTR(input_scale, Float, 82.96)
    .ATTR(input_offset, Float, -36)
    .ATTR(quant_min, Float, -128)
    .OP_END_FACTORY_REG(RmsNormQuant);

REG_OP(SliceTransGeluMul)
    .INPUT(inputX, ge::TensorType::ALL())
    .OUTPUT(outputX, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(SliceTransGeluMul);

REG_OP(SplitTrans)
    .INPUT(X, ge::TensorType::ALL())
    .OUTPUT(Q, ge::TensorType::ALL())
    .OUTPUT(K, ge::TensorType::ALL())
    .OUTPUT(V, ge::TensorType::ALL())
    .OUTPUT(Shape, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(SplitTrans);

REG_OP(TomeMerged)
    .INPUT(TokenA, ge::TensorType::ALL())
    .INPUT(TokenB, ge::TensorType::ALL())
    .INPUT(TOPK_Indice, ge::TensorType::ALL())
    .INPUT(Arg_Max, ge::TensorType::ALL())
    .OUTPUT(unmergeTokenA, ge::TensorType::ALL())
    .OUTPUT(unReduceTokenB, ge::TensorType::ALL())
    .OUTPUT(unReduceCount, ge::TensorType::ALL())
    .ATTR(topRRate, Float, 0.5)
    .OP_END_FACTORY_REG(TomeMerged);

REG_OP(TomeUnmerged)
    .INPUT(attenOut, ge::TensorType::ALL())
    .INPUT(Ori_IndiceA, ge::TensorType::ALL())
    .INPUT(Ori_IndiceB, ge::TensorType::ALL())
    .INPUT(TOPK_Indice, ge::TensorType::ALL())
    .INPUT(Arg_Max, ge::TensorType::ALL())
    .OUTPUT(unZipToken, ge::TensorType::ALL())
    .ATTR(topRRate, Float, 0.5)
    .OP_END_FACTORY_REG(TomeUnmerged);

REG_OP(UnpadAddLN)
    .INPUT(hiddenStates, ge::TensorType::ALL())
    .INPUT(residual, ge::TensorType::ALL())
    .INPUT(epsilon, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(outStates, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(UnpadAddLN);

REG_OP(UnpadFlashAttentionMix)
    .INPUT(Q, ge::TensorType::ALL())
    .INPUT(Kcache, ge::TensorType::ALL())
    .INPUT(Vcache, ge::TensorType::ALL())
    .OPTIONAL_INPUT(AttentionMask, ge::TensorType::ALL())
    .OUTPUT(outputO, ge::TensorType::ALL())
    .OUTPUT(outputS, ge::TensorType::ALL())
    .OUTPUT(outputP, ge::TensorType::ALL())
    .OUTPUT(oTmp, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(UnpadFlashAttentionMix);

REG_OP(UnpadFlashAttentionMixSd)
    .INPUT(Q, ge::TensorType::ALL())
    .INPUT(Kcache, ge::TensorType::ALL())
    .INPUT(Vcache, ge::TensorType::ALL())
    .INPUT(qSeqlen, ge::TensorType::ALL())
    .INPUT(kvSeqlen, ge::TensorType::ALL())
    .INPUT(kvSeqlenShape, ge::TensorType::ALL())
    .INPUT(layerID, ge::TensorType::ALL())
    .OPTIONAL_INPUT(AttentionMask, ge::TensorType::ALL())
    .OUTPUT(outputO, ge::TensorType::ALL())
    .OUTPUT(outputS, ge::TensorType::ALL())
    .OUTPUT(outputP, ge::TensorType::ALL())
    .OUTPUT(oTmp, ge::TensorType::ALL())
    .ATTR(kv_head, Int, 2)
    .ATTR(max_seqlen, Int, 32768)
    .OP_END_FACTORY_REG(UnpadFlashAttentionMixSd);

REG_OP(UnpadInput)
    .INPUT(hiddenStates, ge::TensorType::ALL())
    .INPUT(seqLen, ge::TensorType::ALL())
    .INPUT(ntokens, ge::TensorType::ALL())
    .OUTPUT(outStates, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(UnpadInput);

REG_OP(UnpadPagedAttentionMixLlm)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(kCache, ge::TensorType::ALL())
    .INPUT(vCache, ge::TensorType::ALL())
    .INPUT(qSeqlen, ge::TensorType::ALL())
    .INPUT(kvSeqlen, ge::TensorType::ALL())
    .INPUT(blockTableGm, ge::TensorType::ALL())
    .INPUT(blockInfo, ge::TensorType::ALL())
    .INPUT(layerId, ge::TensorType::ALL())
    .OPTIONAL_INPUT(attentionMask, ge::TensorType::ALL())
    .OUTPUT(outputO, ge::TensorType::ALL())
    .OUTPUT(outputS, ge::TensorType::ALL())
    .OUTPUT(outputP, ge::TensorType::ALL())
    .OUTPUT(oTmp, ge::TensorType::ALL())
    .ATTR(kvHead, Int, 2)
    .ATTR(maxSeqlen, Int, 32768)
    .OP_END_FACTORY_REG(UnpadPagedAttentionMixLlm);

REG_OP(Yulu)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(y, ge::TensorType::ALL())
    .INPUT(z, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(Yulu);

}

#endif
