import numpy as np


def mrr(gt_item, pred_items):
    if gt_item in pred_items:
        index = np.where(pred_items == gt_item)[0][0]
        return np.reciprocal(float(index + 1))
    else:
        return 0

# 衡量是否存在与topk列表中
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

# NDCG = DCG / IDCG
# gi_item是真实item, pred_item是评分模型排序后的预测item列表, v是user与gt_item对应实际的相关度等级 有交互为1, 无交互为0
# DCG = sumk((2^v - 1) / log2(i + 1)), 这里的v对应评分模型排序后的相关度, i对应未排序的v或者pred_item的索引
# IDCG = sumk((2^v' - 1) / log2(i + 1)), 这里的v’对应按v重新排序的相关度, i对应排序后的v或者pred_item的索引
# 当相关的时候vi=1, 不相关的时候vi=0时, 当gt_item in pred_items, v中仅含一个1, NDCG = DCG = 1 / log2(i + 1)
# 当gt_item not in pred_items ndcg=0
# 衡量排序
def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = np.where(pred_items == gt_item)[0][0]
        return np.reciprocal(np.log2(index + 2))
    return 0
