# OpenClaw 上复现 LAR（简化两组对比）执行指南

## 当前阶段范围锁定（2026-03）

为保证本周可交付，当前代码实现和实验执行统一收敛到以下范围：

1. 仅做 `TriviaQA`（不包含 KodCode）。
2. 基座模型固定为 `Qwen/Qwen3-8B`。
3. 运行环境以 Ubuntu 上的真实 OpenClaw 部署为准，Windows 仅用于脚本开发与预处理。
4. 对比组仅保留：
   - `Vanilla OpenClaw`
   - `OpenClaw + Prompt Replacement + LoRA/KL Distillation`
5. 统一评测约束：
   - `validation` 子集 5000 条
   - `seed=42`
   - `temperature=0`
   - 两组使用同一子集与同一评测入口

## 0. 项目目标与实验边界

### 目标
在真实复杂 Agent（OpenClaw）场景中，验证 LAR 思路是否成立：  
将高频、低熵、结构性重复的长 prompt（System / Tool Usage Prompt）替换为 special token，并通过 LoRA/蒸馏训练恢复语义能力，在保持任务效果的前提下降低推理冗余。

### 本轮实验边界
只做两组对比：
1. `Vanilla OpenClaw`
2. `OpenClaw + Prompt Replacement + LoRA/蒸馏`

不展开复杂 ablation，不讨论训练超参数网格搜索。

---

## 1. 学习并对齐两个 Benchmark（TriviaQA / KodCode）

### 这一步的意义
先统一任务接口和评测逻辑，避免后续“能跑但不对题”的问题。

### 需要做什么
1. 梳理 TriviaQA 与 KodCode 的输入输出格式、评测方式、交互轮次特征。
2. 明确在 OpenClaw 中每个 benchmark 的调用入口、数据读取方式、结果落盘方式。
3. 形成统一运行手册，确保别人可复现同样流程。

### 本步产出
1. `benchmark_brief.md`：每个 benchmark 的任务格式、交互方式、评测口径。
2. `runbook.md`：在 OpenClaw 中如何运行 Vanilla 评测与采样。

### 验收标准
1. 能用一条清晰流程跑通两个 benchmark 的最小样本。
2. 输出可被自动评测脚本读取。

---

## 2. 用 OpenClaw + Llama3-8B 跑 Vanilla，采样轨迹

### 这一步的意义
构建教师轨迹数据，用于后续 special token 训练（LoRA/蒸馏）。

### 需要做什么
1. 选定基础模型（Llama3-8B 系列）并接入 OpenClaw。
2. 以 Vanilla 模式运行 TriviaQA / KodCode，记录完整轨迹。
3. 轨迹中保留：系统提示、工具提示、模型输出、工具调用、环境返回、最终答案与评测结果。

### 资源说明
采样阶段可优先使用 API 平台，不强依赖本地显卡。  
核心要求是轨迹可追溯、格式统一、可用于后续训练。

### 本步产出
1. `data/vanilla_traces/triviaqa/*.jsonl`
2. `data/vanilla_traces/kodcode/*.jsonl`
3. `vanilla_eval_summary.md`：Vanilla 基线结果与数据规模统计。

### 验收标准
1. 两个 benchmark 都有可训练的轨迹样本。
2. 轨迹字段完整且可解析。

---

上述两个任务在本周日3月22号完成






## 3. 将 LAR 的 special token 替换逻辑接到 OpenClaw

### 这一步的意义
把论文方法落到真实 Agent 框架：不改任务本身，只改动作表示方式。

### 需要做什么
1. 基于现有 LAR 代码，迁移 special token 替换逻辑到 OpenClaw。
2. 在 OpenClaw 中选定 1-2 个高频且长的 prompt 片段（System / Tool Usage Prompt）作为替换对象。
3. 定义替换映射表（`原 prompt -> special token`），并保证替换过程可回溯。
4. 保留高熵参数内容为显式文本，不做压缩替换（例如 query、实体、数字、tool 参数）。

### 实现建议（指导级）
1. 在 prompt 组装层加入“替换前/替换后”开关，便于回放对比。
2. 将替换字典独立配置文件化，避免硬编码在业务逻辑中。
3. 所有替换行为记录日志，方便排查工具调用异常。

### 本步产出
1. `configs/lar_prompt_mapping.yaml`（或等价配置）
2. `openclaw_lar_integration.md`：接入点说明与代码说明
3. 一份最小样本的替换前后轨迹对照

### 验收标准
1. OpenClaw 能稳定输出替换后的输入序列。
2. 替换过程不破坏基础执行链路（能跑通最小样本）。

---

## 4. 基于替换后轨迹进行 LoRA/蒸馏训练

### 这一步的意义
让模型学会 special token 与原长 prompt 的等价语义。

### 需要做什么
1. 从 Vanilla 轨迹构造训练样本（输入为替换后表示，目标为正确决策/输出）。
2. 使用 LoRA 或蒸馏方案训练模型适配器，保持底座模型主体不变。
3. 训练期间关注可执行性：工具调用格式、任务完成链路、最终答案一致性。

### 资源说明
该步骤通常需要 GPU 和可控模型权重（本地或云训练环境）。

### 本步产出
1. 训练后的适配器权重（LoRA/蒸馏产物）
2. `training_note.md`：数据版本、代码版本、训练输入来源说明

### 验收标准
1. 产物可被 OpenClaw 正常加载并推理。
2. 训练过程可复现（版本与数据可追踪）。

---

## 5. 评测与结论输出（Vanilla vs Replace+LoRA）

### 这一步的意义
用统一口径检验方法在真实 Agent 中是否成立。

### 需要做什么
1. 在同一评测配置下分别跑两组：
   - Vanilla
   - Replace+LoRA
2. 汇总两个 benchmark 的结果对比，重点看：
   - 任务效果（准确率/成功率）
   - 推理代价（token 与时延）
   - 工具调用是否稳定
3. 记录失败样例并分析是否由 prompt 压缩导致。

### 本步产出
1. `final_comparison.md`
2. `results/*.csv`（统一结构化结果）
3. 结论段落：是否支持“LAR 可迁移到真实复杂 Agent”

### 验收标准
1. 两组结果可直接横向对比。
2. 结论可回溯到具体样本与日志。



该任务需要在3月29号前完成

---

## 6. 分工建议

1. 任务 A：Benchmark 对齐与运行手册 （2天 3.17-3.18）
2. 任务 B：Vanilla 采样与轨迹清洗 （4天 3.19-3.22）
3. 任务 C：OpenClaw special token 替换接入 （2天 3.23-3.24）
4. 任务 D：LoRA/蒸馏训练执行 （4天 3.25-3.28）
5. 任务 E：最终评测与报告汇总 （1天 3.28-3.29）


---

## 7. 最终交付清单

1. 一套可运行脚本：Vanilla 采样、Replace+LoRA 推理、统一评测
2. 一份 prompt 替换映射配置
3. 一份训练产物与加载说明
4. 一份最终对比报告（两个 benchmark）
5. 一份结论说明：方法是否在 OpenClaw 场景成立

