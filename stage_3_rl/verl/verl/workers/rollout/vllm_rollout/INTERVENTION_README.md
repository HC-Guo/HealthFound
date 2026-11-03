# Intervention-based Two-Stage Rollout for GRPO

## 概述

这是一个改进版的GRPO（Group Relative Policy Optimization）算法实现，通过引入intervention机制来提高模型思考行为的多样性。该实现基于verl和vLLM框架，支持两阶段并行rollout生成。

## 核心设计

### 两阶段Rollout机制

对于每个输入样本，系统执行两个阶段的rollout：

1. **Stage 1 - 基础轨迹生成**（4次并行rollout）
   - 使用较高的温度（默认1.2）生成4个多样化的基础轨迹
   - 这些轨迹作为后续intervention的基础

2. **Stage 2 - Intervention轨迹生成**（12次并行rollout）
   - 对每个基础轨迹，生成3个intervention变体
   - 在轨迹的中间位置（30%-80%）插入intervention trigger短语
   - 使用正常温度（默认1.0）继续生成

总计：每个样本生成16个轨迹（4个基础 + 12个intervention）

### Intervention机制

Intervention通过在生成过程中插入触发短语来改变模型的思考路径。这些短语基于实际模型行为分析得出，分为8个类别：

- **触发短语类别**：
  - **问题拆解类**：如 "Let me break this down step by step: "
  - **换思路类**：如 "Let me try to think differently: "
  - **验证/纠错类**：如 "Wait, no, that's not correct: "
  - **深入思考类**：如 "Let me think about how to approach this: "
  - **总结类**：如 "So, putting it all together: "
  - **记录/表达类**：如 "Let me write that down: "
  - **不确定/质疑类**：如 "Wait, but I'm not sure: "
  - **回溯/重新考虑类**：如 "Let me reconsider the: "

- **插入位置**：在response的30%-80%之间随机选择
- **效果**：促使模型探索不同的推理路径，增加输出多样性

## 文件结构

```
verl/workers/rollout/vllm_rollout/
├── intervention_sampler.py          # Intervention采样逻辑
├── vllm_intervention_rollout.py     # 两阶段rollout主实现
├── example_intervention_config.yaml # 配置示例
├── example_intervention_usage.py    # 使用示例
└── INTERVENTION_README.md           # 本文档
```

## 使用方法

### 1. 配置

在配置文件中指定使用intervention rollout：

```yaml
rollout:
  name: "vllm_intervention"  # 使用intervention版本
  mode: "sync"

  # 启用intervention
  enable_intervention: true

  # Intervention配置
  intervention:
    n_base_trajectories: 4           # 基础轨迹数量
    n_interventions_per_base: 3      # 每个基础轨迹的intervention数
    base_temperature: 1.2             # 基础轨迹温度（较高）
    intervention_temperature: 1.0     # Intervention温度
    min_intervention_position: 0.3    # 最小插入位置（30%）
    max_intervention_position: 0.8    # 最大插入位置（80%）
```

### 2. 代码集成

```python
from verl.workers.rollout.vllm_rollout import vLLMInterventionRollout

# 初始化rollout
rollout = vLLMInterventionRollout(
    config=rollout_config,
    model_config=model_config,
    device_mesh=device_mesh
)

# 生成sequences
outputs = rollout.generate_sequences(prompts)

# outputs包含所有16个轨迹
# responses shape: [batch_size * 16, response_length]
```

### 3. 处理输出

输出包含所有轨迹，按以下顺序排列：

```python
# 对于每个样本i：
# - 轨迹 [i*16 : i*16+4]：基础轨迹
# - 轨迹 [i*16+4 : i*16+16]：intervention轨迹

metadata = outputs.non_tensor_batch["intervention_metadata"]
n_base = metadata['n_base_trajectories']
n_interventions = metadata['n_interventions_per_base']
```

## 关键参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `n_base_trajectories` | 4 | 基础轨迹数量 |
| `n_interventions_per_base` | 3 | 每个基础轨迹的intervention数 |
| `base_temperature` | 1.2 | 基础轨迹采样温度 |
| `intervention_temperature` | 1.0 | Intervention采样温度 |
| `min_intervention_position` | 0.3 | 最小插入位置（占response长度比例） |
| `max_intervention_position` | 0.8 | 最大插入位置（占response长度比例） |

## 优势

1. **提高多样性**：通过intervention机制探索不同的推理路径
2. **可控性**：可以自定义intervention短语和插入位置
3. **效率**：两阶段并行生成，充分利用GPU资源
4. **兼容性**：完全兼容现有verl/vLLM框架

## 注意事项

1. **内存使用**：生成16倍的轨迹会增加内存消耗，请适当调整batch_size
2. **计算成本**：两阶段rollout会增加计算时间，但通过并行化减少了延迟
3. **日志概率**：Intervention轨迹的log_probs只包含continuation部分

## 扩展性

系统设计支持以下扩展：

- 自定义intervention短语
- 动态调整intervention位置策略
- 不同的温度调度策略
- 支持多轮对话中的intervention

## 故障排除

如果遇到问题：

1. 确保vLLM和verl正确安装
2. 检查GPU内存是否充足
3. 验证模型路径和tokenizer配置
4. 查看日志中的详细错误信息

## 相关链接

- [verl文档](https://verl.readthedocs.io/)
- [vLLM文档](https://docs.vllm.ai/)