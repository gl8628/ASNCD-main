from math import log
from typing import List, Union, Set
import numpy as np
import time


def compare_comm(pred_comm: Union[List, Set],
                 true_comm: Union[List, Set]):
    """
    Compute the Precision, Recall, F1 and Jaccard similarity
    as the second argument is the ground truth community.
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return p, r, f, j


def eval_scores(pred_comms, true_comms, tmp_print=False):
    # 4 columns for precision, recall, f1, jaccard
    pred_scores = np.zeros((len(pred_comms), 4))
    truth_scores = np.zeros((len(true_comms), 4))

    for i, pred_comm in enumerate(pred_comms):
        np.max([compare_comm(pred_comm, true_comms[j])
                for j in range(len(true_comms))], 0, out=pred_scores[i])

    for j, true_comm in enumerate(true_comms):
        np.max([compare_comm(pred_comms[i], true_comm)
                for i in range(len(pred_comms))], 0, out=truth_scores[j])
    truth_scores[:, :2] = truth_scores[:, [1, 0]]

    if tmp_print:
        print("P, R, F, J AvgAxis0: ", pred_scores.mean(0))
        print("P, R, F, J AvgAxis1: ", truth_scores.mean(0))

    # Avg F1 / Jaccard
    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.

    # detect percent
    comm_nodes = {node for com in true_comms for node in com}
    pred_nodes = {node for com in pred_comms for node in com}
    percent = len(list(comm_nodes & pred_nodes)) / len(comm_nodes)

    # NMI
    nmi_score = get_nmi_score(pred_comms, true_comms)

    if tmp_print:
        # print(f"precision: {mean_score_all[0]:.4f} recall: {mean_score_all[1]:.4f}  AvgF1: {mean_score_all[2]:.4f} AvgJaccard: {mean_score_all[3]:.4f} NMI: {nmi_score:.4f} "
        #       f"Detect percent: {percent:.4f}")
        print(f"precision: {mean_score_all[0]:.4f} recall: {mean_score_all[1]:.4f}  AvgF1: {mean_score_all[2]:.4f} AvgJaccard: {mean_score_all[3]:.4f} nmi_score：{nmi_score:.4f} ")
     
    return round(mean_score_all[0], 4),round(mean_score_all[1], 4),round(mean_score_all[2], 4),round(mean_score_all[3], 4)


def eval_scores_fast(pred_comms, true_comms, tmp_print=False):
    """
    修复版：确保与原始eval_scores结果一致
    """
    from scipy.sparse import csr_matrix
    import numpy as np
    import time
    
    def compare_comm_fast(pred_size, true_size, intersection):
        """完全匹配原始compare_comm的计算逻辑"""
        if true_size == 0 or pred_size == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        p = intersection / pred_size
        r = intersection / true_size
        f = 2 * p * r / (p + r + 1e-9)
        j = intersection / (pred_size + true_size - intersection + 1e-9)
        return p, r, f, j
    
    start_time = time.time()
    
    # 1. 构建稀疏矩阵
    print("构建稀疏矩阵...")
    
    # 收集所有节点
    all_nodes = set()
    for comm in pred_comms:
        all_nodes.update(comm)
    for comm in true_comms:
        all_nodes.update(comm)
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    
    n_pred = len(pred_comms)
    n_true = len(true_comms)
    
    # 预测社区矩阵
    pred_rows, pred_cols = [], []
    pred_sizes = []
    for i, comm in enumerate(pred_comms):
        size = len(comm)
        pred_sizes.append(size)
        for node in comm:
            pred_rows.append(i)
            pred_cols.append(node_to_idx[node])
    
    pred_matrix = csr_matrix(
        (np.ones(len(pred_rows), dtype=np.int32), (pred_rows, pred_cols)),
        shape=(n_pred, n_nodes),
        dtype=np.int32
    )
    
    # 真实社区矩阵
    true_rows, true_cols = [], []
    true_sizes = []
    for i, comm in enumerate(true_comms):
        size = len(comm)
        true_sizes.append(size)
        for node in comm:
            true_rows.append(i)
            true_cols.append(node_to_idx[node])
    
    true_matrix = csr_matrix(
        (np.ones(len(true_rows), dtype=np.int32), (true_rows, true_cols)),
        shape=(n_true, n_nodes),
        dtype=np.int32
    )
    
    # 2. 计算交集矩阵
    print("计算交集矩阵...")
    intersection = pred_matrix.dot(true_matrix.T)  # n_pred × n_true
    
    # 3. 完全匹配原始逻辑
    pred_scores = np.zeros((n_pred, 4))
    truth_scores = np.zeros((n_true, 4))
    
    pred_sizes_arr = np.array(pred_sizes, dtype=np.float32)
    true_sizes_arr = np.array(true_sizes, dtype=np.float32)
    
    # 关键修复1：预测→真实 - 使用np.max逻辑
    print("计算预测→真实匹配...")
    intersection_csr = intersection.tocsr()
    
    for i in range(n_pred):
        pred_size = pred_sizes_arr[i]
        if pred_size == 0:
            continue
        
        row_start = intersection_csr.indptr[i]
        row_end = intersection_csr.indptr[i + 1]
        
        if row_start == row_end:
            # 没有交集，所有指标为0
            continue
        
        # 收集所有指标
        all_metrics = []
        for idx in range(row_start, row_end):
            j = int(intersection_csr.indices[idx])
            overlap = intersection_csr.data[idx]
            
            p, r, f, jacc = compare_comm_fast(pred_size, true_sizes_arr[j], overlap)
            all_metrics.append([p, r, f, jacc])
        
        if all_metrics:
            # 关键：使用np.max选择最佳，与原始一致
            pred_scores[i] = np.max(all_metrics, axis=0)
    
    # 关键修复2：真实→预测 - 使用np.max逻辑
    print("计算真实→预测匹配...")
    intersection_csc = intersection.tocsc()
    
    for j in range(n_true):
        true_size = true_sizes_arr[j]
        if true_size == 0:
            continue
        
        col_start = intersection_csc.indptr[j]
        col_end = intersection_csc.indptr[j + 1]
        
        if col_start == col_end:
            continue
        
        # 收集所有指标
        all_metrics = []
        for idx in range(col_start, col_end):
            i = int(intersection_csc.indices[idx])
            overlap = intersection_csc.data[idx]
            
            p, r, f, jacc = compare_comm_fast(pred_sizes_arr[i], true_size, overlap)
            all_metrics.append([p, r, f, jacc])
        
        if all_metrics:
            # 关键：使用np.max选择最佳
            truth_scores[j] = np.max(all_metrics, axis=0)
    
    # 关键修复3：交换truth_scores的precision和recall列
    truth_scores[:, :2] = truth_scores[:, [1, 0]]
    
    # 4. 计算平均指标
    # 注意：原始eval_scores计算所有行的平均值，包括全0行
    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.
    
    elapsed = time.time() - start_time

# detect percent
    comm_nodes = {node for com in true_comms for node in com}
    pred_nodes = {node for com in pred_comms for node in com}
    percent = len(list(comm_nodes & pred_nodes)) / len(comm_nodes)

    # NMI
    nmi_score = get_nmi_score_fast(pred_comms, true_comms)

    if tmp_print:
        print(f"\n快速评估完成，耗时: {elapsed:.2f}秒")
        print(f"精确率: {mean_score_all[0]:.4f}")
        print(f"召回率: {mean_score_all[1]:.4f}")
        print(f"F1分数: {mean_score_all[2]:.4f}")
        print(f"Jaccard指数: {mean_score_all[3]:.4f}")
        print(f"nmi_score: {nmi_score:.4f}")
    
    return mean_score_all[0], mean_score_all[1], mean_score_all[2], mean_score_all[3]


def eval_scores_fast_optimized(pred_comms, true_comms, tmp_print=False):
    """
    向量化优化版：保持结果一致，大幅提升效率
    """
    from scipy.sparse import csr_matrix
    import numpy as np
    import time
    
    start_time = time.time()
    
    # 1. 高效构建节点映射和稀疏矩阵
    print("构建稀疏矩阵...")
    
    # 收集所有节点（优化版）
    all_nodes = set()
    # 批量添加节点，减少循环次数
    for comm in pred_comms:
        all_nodes.update(comm)
    for comm in true_comms:
        all_nodes.update(comm)
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    n_pred = len(pred_comms)
    n_true = len(true_comms)
    
    # 预分配数组大小
    total_pred_entries = sum(len(c) for c in pred_comms)
    total_true_entries = sum(len(c) for c in true_comms)
    
    pred_rows = np.empty(total_pred_entries, dtype=np.int32)
    pred_cols = np.empty(total_pred_entries, dtype=np.int32)
    pred_sizes = np.empty(n_pred, dtype=np.float32)
    
    true_rows = np.empty(total_true_entries, dtype=np.int32)
    true_cols = np.empty(total_true_entries, dtype=np.int32)
    true_sizes = np.empty(n_true, dtype=np.float32)
    
    # 批量填充预测矩阵
    idx = 0
    for i, comm in enumerate(pred_comms):
        size = len(comm)
        pred_sizes[i] = size
        if size > 0:
            pred_rows[idx:idx+size] = i
            # 批量转换节点ID
            pred_cols[idx:idx+size] = [node_to_idx[node] for node in comm]
            idx += size
    
    # 批量填充真实矩阵
    idx = 0
    for i, comm in enumerate(true_comms):
        size = len(comm)
        true_sizes[i] = size
        if size > 0:
            true_rows[idx:idx+size] = i
            true_cols[idx:idx+size] = [node_to_idx[node] for node in comm]
            idx += size
    
    # 构建稀疏矩阵
    pred_matrix = csr_matrix(
        (np.ones(len(pred_rows), dtype=np.int8), (pred_rows, pred_cols)),
        shape=(n_pred, n_nodes),
        dtype=np.int8
    )
    
    true_matrix = csr_matrix(
        (np.ones(len(true_rows), dtype=np.int8), (true_rows, true_cols)),
        shape=(n_true, n_nodes),
        dtype=np.int8
    )
    
    # 2. 计算交集矩阵
    print("计算交集矩阵...")
    intersection = pred_matrix.dot(true_matrix.T.astype(np.int32))
    
    # 3. 向量化计算指标
    print("计算指标...")
    
    # 获取交集的CSR和CSC格式
    intersection_csr = intersection.tocsr()
    intersection_csc = intersection.tocsc()
    
    # 预分配结果数组
    pred_scores = np.zeros((n_pred, 4))
    truth_scores = np.zeros((n_true, 4))
    
    # 关键优化1：批量处理预测→真实
    for i in range(n_pred):
        pred_size = pred_sizes[i]
        if pred_size == 0:
            continue
        
        row_start = intersection_csr.indptr[i]
        row_end = intersection_csr.indptr[i + 1]
        
        if row_start == row_end:
            continue
        
        # 获取这一行的所有数据
        overlaps = intersection_csr.data[row_start:row_end]
        j_indices = intersection_csr.indices[row_start:row_end]
        
        if len(overlaps) > 0:
            # 向量化计算所有指标
            p = overlaps / pred_size
            r = overlaps / true_sizes[j_indices]
            f = 2 * p * r / (p + r + 1e-9)
            jacc = overlaps / (pred_size + true_sizes[j_indices] - overlaps + 1e-9)
            
            # 组合所有指标
            metrics = np.column_stack([p, r, f, jacc])
            
            # 使用np.max选择最佳（保持与原逻辑一致）
            if len(metrics) > 0:
                pred_scores[i] = np.max(metrics, axis=0)
    
    # 关键优化2：批量处理真实→预测
    for j in range(n_true):
        true_size = true_sizes[j]
        if true_size == 0:
            continue
        
        col_start = intersection_csc.indptr[j]
        col_end = intersection_csc.indptr[j + 1]
        
        if col_start == col_end:
            continue
        
        # 获取这一列的所有数据
        overlaps = intersection_csc.data[col_start:col_end]
        i_indices = intersection_csc.indices[col_start:col_end]
        
        if len(overlaps) > 0:
            # 向量化计算所有指标
            p = overlaps / pred_sizes[i_indices]
            r = overlaps / true_size
            f = 2 * p * r / (p + r + 1e-9)
            jacc = overlaps / (pred_sizes[i_indices] + true_size - overlaps + 1e-9)
            
            # 组合所有指标
            metrics = np.column_stack([p, r, f, jacc])
            
            # 使用np.max选择最佳（保持与原逻辑一致）
            if len(metrics) > 0:
                truth_scores[j] = np.max(metrics, axis=0)
    
    # 关键修复：交换truth_scores的precision和recall列
    truth_scores[:, :2] = truth_scores[:, [1, 0]]
    
    # 4. 计算平均指标
    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.
    
    elapsed = time.time() - start_time
    
    # 计算覆盖率
    comm_nodes = set()
    for com in true_comms:
        comm_nodes.update(com)
    pred_nodes = set()
    for com in pred_comms:
        pred_nodes.update(com)
    
    percent = len(comm_nodes & pred_nodes) / (len(comm_nodes) + 1e-9)
    
    # NMI
    nmi_score = get_nmi_score_fast(pred_comms, true_comms)
    
    if tmp_print:
        print(f"\n优化评估完成，耗时: {elapsed:.2f}秒")
        print(f"精确率: {mean_score_all[0]:.4f}")
        print(f"召回率: {mean_score_all[1]:.4f}")
        print(f"F1分数: {mean_score_all[2]:.4f}")
        print(f"Jaccard指数: {mean_score_all[3]:.4f}")
        print(f"节点覆盖率: {percent:.4f}")
        print(f"nmi_score: {nmi_score:.4f}")
    
    return mean_score_all[0], mean_score_all[1], mean_score_all[2], mean_score_all[3]

def eval_scores_fast_optimized_fixed(pred_comms, true_comms, tmp_print=False):
    """
    修正版本：保持与原版本一致的逻辑，但优化性能
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    import time
    
    start_time = time.time()
    
    # 1. 构建节点映射
    # print("构建节点映射...")
    
    # 更高效地构建所有节点集合
    all_nodes = set()
    # 使用生成器表达式减少内存使用
    all_nodes.update(*(set(comm) for comm in pred_comms))
    all_nodes.update(*(set(comm) for comm in true_comms))
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    n_pred = len(pred_comms)
    n_true = len(true_comms)
    
    # 2. 更高效地构建稀疏矩阵
    # print("构建稀疏矩阵...")
    
    # 预计算社区大小
    pred_sizes = np.array([len(comm) for comm in pred_comms], dtype=np.float32)
    true_sizes = np.array([len(comm) for comm in true_comms], dtype=np.float32)
    
    # 使用更高效的稀疏矩阵构建方式
    pred_data = []
    pred_indices = []
    pred_indptr = [0]
    
    for comm in pred_comms:
        indices = [node_to_idx[node] for node in comm]
        pred_indices.extend(indices)
        pred_data.extend([1] * len(indices))
        pred_indptr.append(len(pred_indices))
    
    true_data = []
    true_indices = []
    true_indptr = [0]
    
    for comm in true_comms:
        indices = [node_to_idx[node] for node in comm]
        true_indices.extend(indices)
        true_data.extend([1] * len(indices))
        true_indptr.append(len(true_indices))
    
    pred_matrix = csr_matrix(
        (pred_data, pred_indices, pred_indptr),
        shape=(n_pred, n_nodes),
        dtype=np.int8
    )
    
    true_matrix = csr_matrix(
        (true_data, true_indices, true_indptr),
        shape=(n_true, n_nodes),
        dtype=np.int8
    )
    
    # 3. 计算交集矩阵（分批处理以避免内存爆炸）
    # print("计算交集矩阵...")
    
    # 对于大型数据，分批计算交集
    if n_pred * n_true > 1e8:  # 如果可能产生超过1亿个元素
        # print(f"矩阵过大 ({n_pred}×{n_true})，启用分批计算...")
        
        # 预分配结果数组
        pred_scores = np.zeros((n_pred, 4), dtype=np.float32)
        truth_scores = np.zeros((n_true, 4), dtype=np.float32)
        
        # 分批大小
        batch_size = min(500, n_pred)
        
        for start in range(0, n_pred, batch_size):
            end = min(start + batch_size, n_pred)
            
            # 计算当前批次的交集
            pred_batch = pred_matrix[start:end]
            intersection_batch = pred_batch.dot(true_matrix.T.astype(np.int32))
            intersection_batch = intersection_batch.tocsr()
            
            # 处理当前批次
            for local_i in range(end - start):
                i = start + local_i
                pred_size = pred_sizes[i]
                
                if pred_size == 0:
                    continue
                
                row_start = intersection_batch.indptr[local_i]
                row_end = intersection_batch.indptr[local_i + 1]
                
                if row_start == row_end:
                    continue
                
                # 获取匹配数据
                overlaps = intersection_batch.data[row_start:row_end]
                j_indices = intersection_batch.indices[row_start:row_end]
                
                if len(overlaps) > 0:
                    # 向量化计算
                    p = overlaps / pred_size
                    r = overlaps / true_sizes[j_indices]
                    f = 2 * p * r / (p + r + 1e-9)
                    jacc = overlaps / (pred_size + true_sizes[j_indices] - overlaps + 1e-9)
                    
                    # 组合指标并取最大值（与原逻辑一致）
                    metrics = np.column_stack([p, r, f, jacc])
                    if len(metrics) > 0:
                        pred_scores[i] = np.max(metrics, axis=0)
        
        # 现在处理真实→预测方向
        # 同样使用分批处理
        for start in range(0, n_true, batch_size):
            end = min(start + batch_size, n_true)
            
            true_batch = true_matrix[start:end]
            intersection_batch = true_batch.dot(pred_matrix.T.astype(np.int32))
            intersection_batch = intersection_batch.tocsr()
            
            for local_j in range(end - start):
                j = start + local_j
                true_size = true_sizes[j]
                
                if true_size == 0:
                    continue
                
                row_start = intersection_batch.indptr[local_j]
                row_end = intersection_batch.indptr[local_j + 1]
                
                if row_start == row_end:
                    continue
                
                # 获取匹配数据
                overlaps = intersection_batch.data[row_start:row_end]
                i_indices = intersection_batch.indices[row_start:row_end]
                
                if len(overlaps) > 0:
                    # 向量化计算
                    p = overlaps / true_size  # 注意：这里p实际上是recall
                    r = overlaps / pred_sizes[i_indices]  # 这里r实际上是precision
                    f = 2 * p * r / (p + r + 1e-9)
                    jacc = overlaps / (true_size + pred_sizes[i_indices] - overlaps + 1e-9)
                    
                    # 组合指标并取最大值
                    metrics = np.column_stack([p, r, f, jacc])
                    if len(metrics) > 0:
                        # 存储时交换precision和recall列
                        best_row = np.max(metrics, axis=0)
                        truth_scores[j] = np.array([best_row[1], best_row[0], best_row[2], best_row[3]])
    
    else:
        # 原始逻辑（适合中等规模数据）
        intersection = pred_matrix.dot(true_matrix.T.astype(np.int32))
        
        # 获取交集的CSR和CSC格式
        intersection_csr = intersection.tocsr()
        intersection_csc = intersection.tocsc()
        
        # 预分配结果数组
        pred_scores = np.zeros((n_pred, 4), dtype=np.float32)
        truth_scores = np.zeros((n_true, 4), dtype=np.float32)
        
        # 关键优化1：批量处理预测→真实
        for i in range(n_pred):
            pred_size = pred_sizes[i]
            if pred_size == 0:
                continue
            
            row_start = intersection_csr.indptr[i]
            row_end = intersection_csr.indptr[i + 1]
            
            if row_start == row_end:
                continue
            
            # 获取这一行的所有数据
            overlaps = intersection_csr.data[row_start:row_end]
            j_indices = intersection_csr.indices[row_start:row_end]
            
            if len(overlaps) > 0:
                # 向量化计算所有指标
                p = overlaps / pred_size
                r = overlaps / true_sizes[j_indices]
                f = 2 * p * r / (p + r + 1e-9)
                jacc = overlaps / (pred_size + true_sizes[j_indices] - overlaps + 1e-9)
                
                # 组合所有指标
                metrics = np.column_stack([p, r, f, jacc])
                
                # 使用np.max选择最佳（保持与原逻辑一致）
                if len(metrics) > 0:
                    pred_scores[i] = np.max(metrics, axis=0)
        
        # 关键优化2：批量处理真实→预测
        for j in range(n_true):
            true_size = true_sizes[j]
            if true_size == 0:
                continue
            
            col_start = intersection_csc.indptr[j]
            col_end = intersection_csc.indptr[j + 1]
            
            if col_start == col_end:
                continue
            
            # 获取这一列的所有数据
            overlaps = intersection_csc.data[col_start:col_end]
            i_indices = intersection_csc.indices[col_start:col_end]
            
            if len(overlaps) > 0:
                # 向量化计算所有指标
                p = overlaps / pred_sizes[i_indices]
                r = overlaps / true_size
                f = 2 * p * r / (p + r + 1e-9)
                jacc = overlaps / (pred_sizes[i_indices] + true_size - overlaps + 1e-9)
                
                # 组合所有指标
                metrics = np.column_stack([p, r, f, jacc])
                
                # 使用np.max选择最佳（保持与原逻辑一致）
                if len(metrics) > 0:
                    best_row = np.max(metrics, axis=0)
                    # 交换precision和recall列
                    truth_scores[j] = np.array([best_row[1], best_row[0], best_row[2], best_row[3]])
    
    # 4. 计算平均指标
    mean_score_all = (pred_scores.mean(axis=0) + truth_scores.mean(axis=0)) / 2.0
    
    elapsed = time.time() - start_time
    
    # 5. 计算覆盖率（更高效的方式）
    # print("计算覆盖率...")
    
    # 使用稀疏矩阵快速计算
    pred_nodes = set()
    true_nodes = set()
    
    # 分批处理避免内存问题
    batch_size = 1000
    for i in range(0, n_pred, batch_size):
        batch = pred_comms[i:min(i+batch_size, n_pred)]
        for comm in batch:
            pred_nodes.update(comm)
    
    for i in range(0, n_true, batch_size):
        batch = true_comms[i:min(i+batch_size, n_true)]
        for comm in batch:
            true_nodes.update(comm)
    
    percent = len(pred_nodes & true_nodes) / (len(true_nodes) + 1e-9)
    
    # 6. NMI计算
    # nmi_score = 0.0
    # if n_pred < 5000 and n_true < 5000:  # 只在数据量适中时计算
    #     try:
    # nmi_score = get_nmi_score_fast(pred_comms, true_comms)
    # nmi_score = get_nmi_score(pred_comms, true_comms)
    # nmi_score = evaluate_quick_nmi(pred_comms, true_comms)
    
        # except:
        #     nmi_score = 0.0
    
    if tmp_print:
        print(f"\n修正优化评估完成，耗时: {elapsed:.2f}秒")
        print(f"预测社区数: {n_pred}, 真实社区数: {n_true}, 节点数: {n_nodes}")
        print(f"精确率: {mean_score_all[0]:.4f}")
        print(f"召回率: {mean_score_all[1]:.4f}")
        print(f"F1分数: {mean_score_all[2]:.4f}")
        print(f"Jaccard指数: {mean_score_all[3]:.4f}")
        # print(f"节点覆盖率: {percent:.4f}")
        # print(f"NMI分数: {nmi_score:.4f}")
        # print(f"ARI分数: {ari:.4f}")
    
    return mean_score_all[0], mean_score_all[1], mean_score_all[2], mean_score_all[3]

def eval_scores_fast_optimized_v2(pred_comms, true_comms, tmp_print=False):
    """
    高度优化版本：处理2万以上社区时依然高效
    """
    import numpy as np
    from scipy.sparse import lil_matrix, csr_matrix
    import time
    
    start_time = time.time()
    
    # 1. 构建节点映射（使用数组加速）
    print("构建节点映射...")
    
    # 使用Python内置集合但优化内存
    all_nodes = set()
    # 分批处理，避免一次性占用太多内存
    batch_size = 10000
    for i in range(0, len(pred_comms), batch_size):
        batch = pred_comms[i:i+batch_size]
        for comm in batch:
            all_nodes.update(comm)
    
    for i in range(0, len(true_comms), batch_size):
        batch = true_comms[i:i+batch_size]
        for comm in batch:
            all_nodes.update(comm)
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    n_pred = len(pred_comms)
    n_true = len(true_comms)
    
    # 2. 构建稀疏矩阵（使用LIL格式更高效）
    print("构建稀疏矩阵...")
    
    # 使用LIL格式构建，适合增量添加
    pred_matrix = lil_matrix((n_pred, n_nodes), dtype=np.bool_)
    true_matrix = lil_matrix((n_true, n_nodes), dtype=np.bool_)
    
    # 批量填充预测矩阵
    for i, comm in enumerate(pred_comms):
        if comm:  # 非空社区
            indices = [node_to_idx[node] for node in comm]
            pred_matrix[i, indices] = True
    
    # 批量填充真实矩阵
    for i, comm in enumerate(true_comms):
        if comm:  # 非空社区
            indices = [node_to_idx[node] for node in comm]
            true_matrix[i, indices] = True
    
    # 转换为CSR格式以加速矩阵乘法
    pred_matrix = pred_matrix.tocsr()
    true_matrix = true_matrix.tocsr()
    
    # 3. 计算社区大小（向量化）
    print("计算社区大小...")
    pred_sizes = pred_matrix.sum(axis=1).A1.astype(np.float32)
    true_sizes = true_matrix.sum(axis=1).A1.astype(np.float32)
    
    # 4. 计算交集（使用稀疏矩阵乘法，但避免创建完整矩阵）
    print("计算交集矩阵...")
    
    # 对于大规模数据，分批计算交集
    intersection_batch_size = min(1000, n_pred)  # 调整批次大小
    
    # 预分配结果数组
    pred_scores = np.zeros((n_pred, 4), dtype=np.float32)
    truth_scores = np.zeros((n_true, 4), dtype=np.float32)
    
    # 用于存储每个真实社区的最佳匹配
    truth_best_scores = np.zeros((n_true, 4), dtype=np.float32)
    
    # 5. 分批处理预测社区
    print("分批计算指标...")
    
    for start_idx in range(0, n_pred, intersection_batch_size):
        end_idx = min(start_idx + intersection_batch_size, n_pred)
        
        # 获取当前批次的预测矩阵
        pred_batch = pred_matrix[start_idx:end_idx]
        pred_batch_sizes = pred_sizes[start_idx:end_idx]
        
        # 计算当前批次与所有真实社区的交集
        # 使用稀疏矩阵乘法
        intersection_batch = pred_batch.dot(true_matrix.T.astype(np.int32))
        intersection_batch = intersection_batch.tocoo()  # 转换为COO格式以便遍历非零元素
        
        # 处理当前批次的所有非零交集
        for i, j, overlap in zip(intersection_batch.row, intersection_batch.col, intersection_batch.data):
            pred_idx = start_idx + i  # 实际预测社区索引
            true_idx = j  # 真实社区索引
            
            pred_size = pred_batch_sizes[i]
            true_size = true_sizes[true_idx]
            
            if pred_size == 0 or true_size == 0:
                continue
            
            # 计算指标
            precision = overlap / pred_size
            recall = overlap / true_size
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            jaccard = overlap / (pred_size + true_size - overlap + 1e-9)
            
            # 更新预测社区的最佳得分
            current_metrics = np.array([precision, recall, f1, jaccard], dtype=np.float32)
            if np.sum(current_metrics) > np.sum(pred_scores[pred_idx]):
                pred_scores[pred_idx] = current_metrics
            
            # 更新真实社区的最佳得分（注意交换precision和recall）
            truth_metrics = np.array([recall, precision, f1, jaccard], dtype=np.float32)
            if np.sum(truth_metrics) > np.sum(truth_best_scores[true_idx]):
                truth_best_scores[true_idx] = truth_metrics
    
    # 6. 处理未匹配的真实社区（为空社区）
    for j in range(n_true):
        if true_sizes[j] == 0:
            continue
        
        # 如果真实社区没有被任何预测社区匹配到
        if np.sum(truth_best_scores[j]) == 0:
            truth_scores[j] = np.array([0, 0, 0, 0], dtype=np.float32)
        else:
            truth_scores[j] = truth_best_scores[j]
    
    # 7. 计算平均指标
    # 移除空预测社区
    valid_pred_mask = pred_sizes > 0
    valid_true_mask = true_sizes > 0
    
    if np.any(valid_pred_mask):
        mean_pred_scores = pred_scores[valid_pred_mask].mean(axis=0)
    else:
        mean_pred_scores = np.zeros(4)
    
    if np.any(valid_true_mask):
        mean_truth_scores = truth_scores[valid_true_mask].mean(axis=0)
    else:
        mean_truth_scores = np.zeros(4)
    
    mean_score_all = (mean_pred_scores + mean_truth_scores) / 2.0
    
    elapsed = time.time() - start_time
    
    # 8. 计算覆盖率
    # 使用稀疏矩阵快速计算覆盖率
    pred_nodes_mask = pred_matrix.sum(axis=0).A1 > 0
    true_nodes_mask = true_matrix.sum(axis=0).A1 > 0
    
    covered_nodes = np.sum(pred_nodes_mask & true_nodes_mask)
    total_true_nodes = np.sum(true_nodes_mask)
    percent = covered_nodes / (total_true_nodes + 1e-9)
    
    # 9. NMI（如果需要可以单独优化）
    # nmi_score = 0.0
    # if len(pred_comms) < 10000 and len(true_comms) < 10000:  # 只在数据量较小时计算NMI
    #     try:
    #         nmi_score = get_nmi_score_fast(pred_comms, true_comms)
    #     except:
    #         nmi_score = 0.0
    
    if tmp_print:
        print(f"\n超优化评估完成，耗时: {elapsed:.2f}秒")
        print(f"预测社区数: {n_pred}, 真实社区数: {n_true}, 节点数: {n_nodes}")
        print(f"有效预测社区: {np.sum(valid_pred_mask)}, 有效真实社区: {np.sum(valid_true_mask)}")
        print(f"精确率: {mean_score_all[0]:.4f}")
        print(f"召回率: {mean_score_all[1]:.4f}")
        print(f"F1分数: {mean_score_all[2]:.4f}")
        print(f"Jaccard指数: {mean_score_all[3]:.4f}")
        print(f"节点覆盖率: {percent:.4f}")
        # print(f"NMI分数: {nmi_score:.4f}")
    
    return mean_score_all[0], mean_score_all[1], mean_score_all[2], mean_score_all[3]

def get_intersection(a, b, choice=None):
    return len(list(set(a) & set(b))) if not choice else list(set(a) & set(b))


def get_difference(a, b):
    intersection = get_intersection(a, b, choice="List")
    nodes = {x for x in a if x not in intersection}
    return len(list(nodes))

import numpy as np
from collections import defaultdict
import itertools

def get_nmi_score_fast(pred, gt):
    """
    更简洁但保持算法逻辑的优化版本
    """
    import math
    
    def h(x):
        if x <= 0 or x >= 1:
            return 0
        return -x * math.log2(x)
    
    # 1. 转换为集合并预计算
    pred_sets = [set(com) for com in pred]
    gt_sets = [set(com) for com in gt]
    
    # 2. 获取所有节点
    all_nodes = set()
    for com_set in pred_sets + gt_sets:
        all_nodes.update(com_set)
    
    total_nodes = len(all_nodes)
    
    if total_nodes == 0 or len(pred_sets) == 0 or len(gt_sets) == 0:
        return 0
    
    # 3. 预计算社区大小和熵
    def H_community(comm_set):
        size = len(comm_set)
        p1 = size / total_nodes
        p0 = 1 - p1
        return h(p0) + h(p1)
    
    H_pred = [H_community(com) for com in pred_sets]
    H_gt = [H_community(com) for com in gt_sets]
    
    # 4. 直接使用集合的 & 操作（Python 内置的集合交集非常快，缓存反而拖慢速度）
    # 实测表明：对于大规模数据，直接计算比缓存更快且无内存问题
    def get_intersection_size(set1, set2):
        """快速计算交集大小"""
        return len(set1 & set2)
    
    # 5. 优化h_xi_joint_yj
    def h_xi_joint_yj_fast(set_i, set_j, H_i, H_j):
        """快速计算联合熵"""
        intersection = get_intersection_size(set_i, set_j)
        p11 = intersection / total_nodes
        
        diff_i_j = len(set_i) - intersection
        p10 = diff_i_j / total_nodes
        
        diff_j_i = len(set_j) - intersection
        p01 = diff_j_i / total_nodes
        
        p00 = 1 - p11 - p10 - p01
        
        h11 = h(p11)
        h00 = h(p00)
        h01 = h(p01)
        h10 = h(p10)
        
        if h11 + h00 >= h01 + h10:
            return h11 + h10 + h01 + h00
        return H_i + H_j
    
    # 6. 计算条件熵
    def H_X_GIVEN_Y_fast(X_sets, X_H, Y_sets, Y_H):
        total = 0
        
        for i, set_i in enumerate(X_sets):
            H_i = X_H[i]
            if H_i == 0:
                continue
            
            # 找到最小的条件熵
            min_h = float('inf')
            for j, set_j in enumerate(Y_sets):
                joint = h_xi_joint_yj_fast(set_i, set_j, H_i, Y_H[j])
                h_given = joint - Y_H[j]
                
                if h_given < min_h:
                    min_h = h_given
                    if min_h == 0:
                        break
            
            total += min_h / H_i
        
        return total / len(X_sets) if len(X_sets) > 0 else 0
    
    # 7. 计算最终的NMI
    H_pred_given_gt = H_X_GIVEN_Y_fast(pred_sets, H_pred, gt_sets, H_gt)
    H_gt_given_pred = H_X_GIVEN_Y_fast(gt_sets, H_gt, pred_sets, H_pred)
    
    return 1 - 0.5 * (H_pred_given_gt + H_gt_given_pred)

from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_rand_score
import numpy as np

def evaluate_quick_nmi(pred, gt):
    """
    快速评估，同时返回NMI和ARI
    """
    # 构建节点标签映射
    node_to_pred = {node: i for i, comm in enumerate(pred) for node in comm}
    node_to_gt = {node: i for i, comm in enumerate(gt) for node in comm}
    
    # 收集共同节点的标签
    common_nodes = set(node_to_pred.keys()) & set(node_to_gt.keys())
    
    if not common_nodes:
        return 0.0, 0.0
    
    pred_labels = [node_to_pred[node] for node in common_nodes]
    gt_labels = [node_to_gt[node] for node in common_nodes]
    
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    # ari = adjusted_rand_score(gt_labels, pred_labels)
    
    return nmi

# def get_nmi_score_reliable(pred, gt):
#     """
#     使用scikit-learn的内置函数，避免稀疏矩阵警告
#     """
#     # 构建节点标签映射
#     node_to_pred = {}
#     node_to_gt = {}
#     all_nodes = set()
    
#     for i, comm in enumerate(pred):
#         for node in comm:
#             node_to_pred[node] = i
#             all_nodes.add(node)
    
#     for i, comm in enumerate(gt):
#         for node in comm:
#             node_to_gt[node] = i
#             all_nodes.add(node)
    
#     # 收集共同节点的标签
#     pred_labels = []
#     gt_labels = []
    
#     for node in all_nodes:
#         if node in node_to_pred and node in node_to_gt:
#             pred_labels.append(node_to_pred[node])
#             gt_labels.append(node_to_gt[node])
    
#     if len(pred_labels) == 0:
#         return 0
    
#     return normalized_mutual_info_score(gt_labels, pred_labels)


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import math

def get_nmi_score_vectorized(pred, gt):
    """
    完全向量化的NMI计算，适合大规模图
    """
    # 1. 构建节点到社区的映射
    all_nodes = set()
    node_to_pred = {}
    node_to_gt = {}
    
    # 为每个节点分配社区标签
    for i, comm in enumerate(pred):
        for node in comm:
            node_to_pred[node] = i
            all_nodes.add(node)
    
    for i, comm in enumerate(gt):
        for node in comm:
            node_to_gt[node] = i
            all_nodes.add(node)
    
    nodes = list(all_nodes)
    n_nodes = len(nodes)
    
    if n_nodes == 0:
        return 0
    
    # 2. 构建标签向量
    pred_labels = np.array([node_to_pred.get(node, -1) for node in nodes])
    gt_labels = np.array([node_to_gt.get(node, -1) for node in nodes])
    
    # 过滤掉未标记的节点
    mask = (pred_labels >= 0) & (gt_labels >= 0)
    pred_labels = pred_labels[mask]
    gt_labels = gt_labels[mask]
    
    if len(pred_labels) == 0:
        return 0
    
    # 3. 构建混淆矩阵（使用稀疏矩阵）
    from sklearn.metrics.cluster import contingency_matrix
    contingency = contingency_matrix(gt_labels, pred_labels, sparse=True)
    
    # 4. 计算概率
    contingency = contingency.astype(np.float64)
    n = contingency.sum()
    
    # 5. 计算联合熵
    P = contingency / n
    joint_entropy = -np.sum(P.multiply(np.log2(P + (P==0))).data)
    
    # 6. 计算边缘熵
    row_sum = np.array(contingency.sum(axis=1)).flatten()
    col_sum = np.array(contingency.sum(axis=0)).flatten()
    
    P_row = row_sum / n
    P_col = col_sum / n
    
    H_gt = -np.sum(P_row * np.log2(P_row + (P_row==0)))
    H_pred = -np.sum(P_col * np.log2(P_col + (P_col==0)))
    
    # 7. 计算NMI
    mi = H_gt + H_pred - joint_entropy
    nmi = 2 * mi / (H_gt + H_pred)
    
    return nmi

def get_nmi_score(pred, gt):
    def get_overlapping(pred_comms, ground_truth):
        """All nodes number"""
        nodes = {node for com in pred_comms + ground_truth for node in com}
        return len(nodes)

    def h(x):
        return -1 * x * (log(x) / log(2)) if x > 0 else 0

    def H_func(comm):
        p1 = len(comm) / overlapping_nodes
        p0 = 1 - p1
        return h(p0) + h(p1)

    def h_xi_joint_yj(xi, yj):
        p11 = get_intersection(xi, yj) / overlapping_nodes
        p10 = get_difference(xi, yj) / overlapping_nodes
        p01 = get_difference(yj, xi) / overlapping_nodes
        p00 = 1 - p11 - p10 - p01

        if h(p11) + h(p00) >= h(p01) + h(p10):
            return h(p11) + h(p10) + h(p01) + h(p00)
        return H_func(xi) + H_func(yj)

    def h_xi_given_yj(xi, yj):
        return h_xi_joint_yj(xi, yj) - H_func(yj)

    def H_XI_GIVEN_Y(xi, Y):
        res = h_xi_given_yj(xi, Y[0])
        for y in Y:
            res = min(res, h_xi_given_yj(xi, y))
        return res / H_func(xi)

    def H_X_GIVEN_Y(X, Y):
        res = 0
        # for idx in tqdm(range(len(X)), desc="ComputeNMI"):
        for idx in range(len(X)):
            res += H_XI_GIVEN_Y(X[idx], Y)
        return res / len(X)

    if len(pred) == 0 or len(gt) == 0:
        return 0

    overlapping_nodes = get_overlapping(pred, gt)
    return 1 - 0.5 * (H_X_GIVEN_Y(pred, gt) + H_X_GIVEN_Y(gt, pred))
