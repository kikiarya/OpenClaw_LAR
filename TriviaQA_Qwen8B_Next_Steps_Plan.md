# TriviaQA + Qwen3-8B + OpenClaw LAR 后续工作计划

## 1. 目标与范围

本计划用于收束当前实验工作，并把后续任务组织成一条可执行、可复现、可交付的主线。

当前实验目标固定为：

- 任务：TriviaQA
- 模型：Qwen3-8B
- 环境：OpenClaw 真实运行环境
- 方法：system prompt replacement + distillation
- 对比对象：
  - Vanilla OpenClaw
  - OpenClaw + prompt replacement + LoRA/distillation

当前明确不做：

- 不扩展到 KodCode 或其他 benchmark
- 不扩展到其他模型
- 不继续做碎片化多 segment 挖掘
- 不把本轮工作扩展成完整多轮工具轨迹蒸馏

本轮实验以单长段 `<seg_0>` 替换方案为准。

---

## 2. 当前已完成的工作

你已经完成的核心工作包括：

### 2.1 真实数据采集

以下两个文件是本轮实验的真实数据基础：

- `system_prompt_log.jsonl`
- `triviaqa_5000.jsonl`

其中：

- `triviaqa_5000.jsonl` 是真实跑完的 5000 条 TriviaQA Vanilla 结果
- `system_prompt_log.jsonl` 是同一次真实 OpenClaw 运行过程中记录下来的 prompt 日志

### 2.2 数据加工与训练集构建

你已经完成：

- 从 `system_prompt_log.jsonl` 中提取 `llm_input`
- 对 system prompt 重复性做分析
- 将 system prompt 处理为可用于替换的候选文本
- 最终确定采用单长段 `<seg_0>` 的保守替换方案
- 将真实 system prompt 与 TriviaQA 结果对齐，构造成 `train.py` 可直接使用的 dual-view 数据

### 2.3 当前产物状态

已经具备：

- 可用于训练的 `train.jsonl` / `validation.jsonl`
- 固定的 mapping 文件
- 用于检查数据构造是否正确的 `examples.json`
- 基础统计文件 `build_stats.json`

因此，当前阶段不再需要继续围绕数据脚本做大规模改动，主线应切换到训练、评测和总结。

---

## 3. 当前所处阶段

按实验主线划分，你现在大致处于：

1. Vanilla 真实运行完成
2. Prompt replacement 方案已确定
3. 蒸馏训练集已构建完成
4. 尚未完成正式训练、合并部署和最终对比评测

也就是说，后续重点已经不是“做更多 preprocessing”，而是“把实验真正跑完并形成结论”。

---

## 4. 后续总计划

后续工作建议分为 10 个阶段推进。

---

## 5. 阶段 1：冻结实验范围

### 目标

彻底锁定实验边界，避免后续范围漂移。

### 要做的事

- 只保留 `TriviaQA + Qwen3-8B + OpenClaw + system prompt replacement`
- 不再引入新的 benchmark
- 不再引入新的模型
- 不再引入新的 segment 设计路线
- 默认 mapping 固定为当前单段 `<seg_0>` 版本

### 阶段结果

- 后续所有训练、评测、日志和文档都围绕同一方案展开
- 实验命名规则固定，例如 `triviaqa_openclaw_seg0_*`

### 完成标准

- 后续不再修改实验题目定义
- 不再新增“顺手试一下”的分支实验

---

## 6. 阶段 2：训练前 sanity check

### 目标

确认当前训练数据、实验口径和方法定义都清楚无误。

### 要做的事

- 检查 `system_prompt_log.jsonl` 与 `triviaqa_5000.jsonl` 的关联逻辑是否稳定
- 确认 `build_stats.json` 中的样本数统计符合预期
- 检查 `examples.json` 中 `original_messages` 和 `compressed_messages` 是否正确
- 明确当前 supervision 的真实含义

### 这里必须明确的实验定义

当前训练目标更准确地说是：

- 在压缩后的 system prompt 条件下
- 让模型继续输出 TriviaQA 的正确 final answer

这不等价于：

- 完整多轮工具使用轨迹蒸馏
- 完整 ReAct 行为蒸馏

### 阶段结果

- 你能准确描述你当前在训练什么
- 文档口径和真实实现保持一致

### 完成标准

- 能用一段话清楚说明当前数据集的输入、输出、替换逻辑、训练目标

---

## 7. 阶段 3：修训练侧关键风险点

### 目标

在正式训练前处理掉最容易导致训练失败或实验失真的实现风险。

### 重点检查项

- `train.py` 中 special token 的加入逻辑是否正常
- teacher / student vocab 处理是否符合设计预期
- Qwen chat template 在训练和评测阶段是否一致
- `train_new_embeddings_only` 是否真的按预期生效
- 显存、`max_length`、`batch_size` 是否与实际机器匹配

### 建议策略

- 第一版正式实验优先保证稳定可跑通
- 不要一开始就追求最复杂的训练开关
- 若实现存在明显不确定性，先修训练代码，再进入正式训练

### 阶段结果

- 训练脚本可稳定运行
- 不会因 vocab、token、mask、模板问题导致训练无效

### 完成标准

- smoke training 可以稳定启动并跑完

---

## 8. 阶段 4：smoke training

### 目标

先验证整条链路可跑通，再投入正式训练资源。

### 要做的事

- 跑一个小规模训练
- 检查 loss 是否正常
- 检查 special token 是否被正确加入 tokenizer 和模型
- 检查 checkpoint 是否能正常保存
- 检查 merge 是否可执行
- 检查 merge 后模型能否被评测脚本加载

### 不要追求的事情

- 最终精度
- 最优超参
- 一步到位的大规模评测

### 阶段结果

- 证明“数据 -> 训练 -> merge -> eval”全链路可用

### 完成标准

- 能产出一个 smoke checkpoint
- 能对 smoke 模型做最小评测

---

## 9. 阶段 5：正式训练

### 目标

产出本轮实验的主模型。

### 要做的事

- 固定当前 dataset 版本
- 固定当前 mapping 版本
- 固定一组训练超参
- 正式跑 LoRA + distillation
- 完整保留训练日志和输出目录

### 必须记录的信息

- 数据文件版本
- mapping 文件版本
- base model 名称
- teacher model 名称
- epoch 数
- batch size
- learning rate
- max length
- kl weight
- 是否开启 `train_new_embeddings_only`
- 最终 checkpoint 路径

### 阶段结果

- 得到主实验 LoRA 权重

### 完成标准

- 正式训练完成
- 最终 checkpoint 可用

---

## 10. 阶段 6：模型合并与部署验证

### 目标

把训练产物变成可直接推理和评测的模型版本。

### 要做的事

- 执行 LoRA + embedding merge
- 检查 merged model 是否可加载
- 检查 special token 是否在 merged model 中存在
- 用最小样例测试 `<seg_0>` 输入是否可以正常推理

### 如果后续要接回 OpenClaw

还需要：

- 确认 OpenClaw 侧能加载该模型
- 确认替换后的 system prompt 能正常送入模型
- 确认不会因 tokenizer 不一致导致推理失败

### 阶段结果

- 产出一个可评测的 merged model

### 完成标准

- merge 成功
- merged model 可独立完成至少一次推理

---

## 11. 阶段 7：小样本对比评测

### 目标

先用小规模评测看实验方向是否成立。

### 要做的事

- 先跑 100 到 200 条样本
- 比较 Vanilla 与 Replace+LoRA
- 重点观察：
  - accuracy
  - avg prompt tokens
  - 输出格式稳定性
  - 是否出现异常空答或重复

### 这一阶段要回答的问题

- token 是否真的下降
- 性能是否还能接受
- 模型输出是否稳定
- 当前方案是否值得做全量评测

### 阶段结果

- 得到一版方向性判断

### 完成标准

- 拿到小样本对比结果
- 做出“继续全量评测”或“先回去调参”的决定

---

## 12. 阶段 8：全量评测

### 目标

生成最终实验结论所需的核心结果。

### 要做的事

- 在统一设置下跑 Vanilla
- 在统一设置下跑 Replace+LoRA
- 保证比较口径完全一致
- 保存完整日志和汇总结果

### 核心指标

- 准确率
- 平均 prompt token 数
- 总体 token 消耗
- 错误样本数
- 输出格式异常比例
- 如果可记录，也可保留延迟变化

### 阶段结果

- 得到最终对比表格和核心指标

### 完成标准

- 能直接支撑最后的实验结论

---

## 13. 阶段 9：失败样例分析

### 目标

解释实验结果，而不只是报一个分数。

### 要做的事

- 抽取 Replace+LoRA 失败而 Vanilla 成功的样例
- 抽取两者都失败的样例
- 分析失败是否与 prompt replacement 相关
- 分析失败是否来源于：
  - system 压缩后语义恢复不足
  - 压缩幅度不够但模型仍受扰动
  - 训练目标过弱
  - 评测格式或输出稳定性问题

### 建议重点关注的失败类型

- 明显答非所问
- 标签格式错误
- 原本能答对但压缩后答错
- 输出异常缩短、异常重复或空答

### 阶段结果

- 你能够解释结果变化原因

### 完成标准

- 至少形成一小节失败样例分析总结

---

## 14. 阶段 10：最终文档与交付

### 目标

将本轮实验整理成完整、可复现、可汇报的成果。

### 建议最终文档至少包含以下内容

#### 1. 实验定义

- 任务范围
- 模型范围
- 数据来源
- 方法边界

#### 2. 数据链路

- 如何从真实 OpenClaw 运行结果构造训练数据
- `system_prompt_log.jsonl` 与 `triviaqa_5000.jsonl` 分别承担什么角色

#### 3. 方法实现

- 为什么采用单长段 `<seg_0>`
- 为什么不继续做多段碎片化替换
- dual-view 数据如何组织
- 训练目标是什么

#### 4. 实验结果

- Vanilla vs Replace+LoRA 的核心指标
- token 节省与性能变化
- 小样本与全量评测结果

#### 5. 失败案例分析

- 主要失败类型
- 原因判断

#### 6. 最终结论

- 在 TriviaQA + OpenClaw + Qwen3-8B 的真实场景下，system prompt replacement 是否成立
- 当前方案的优势与局限
- 若继续推进，下一步最值得扩展的方向是什么

### 阶段结果

- 一份别人可以阅读、复现、评估的实验总结

### 完成标准

- 实验闭环形成
- 结论可以回溯到日志、模型和样例

---

## 15. 推荐推进顺序

建议实际执行顺序如下：

1. 冻结实验范围
2. 做训练前 sanity check
3. 修训练脚本关键风险
4. 跑 smoke training
5. 跑正式训练
6. merge 模型
7. 做小样本评测
8. 做全量评测
9. 做失败样例分析
10. 写最终总结文档

---

## 16. 每个阶段的完成判断

- 阶段 1 完成：实验边界不再改动
- 阶段 2 完成：能准确说清当前训练目标和数据定义
- 阶段 3 完成：训练脚本关键风险已排除
- 阶段 4 完成：全链路 smoke test 通过
- 阶段 5 完成：主实验模型训练完成
- 阶段 6 完成：merged model 可正常推理
- 阶段 7 完成：小样本评测结果出来
- 阶段 8 完成：全量对比结果出来
- 阶段 9 完成：失败模式分析完成
- 阶段 10 完成：最终报告可交付

---

## 17. 你现在最应该做的下一件事

现在不要继续花时间在新数据脚本或新 segment 设计上。

最优先事项是：

- 完成训练前 sanity check
- 处理训练脚本关键风险
- 然后立刻开始 smoke training

因为你当前真正缺的，不是更多 preprocessing，而是验证这套基于真实数据构建出的 system replacement + distillation 方案，能否真正训练出一个可评测的模型。

