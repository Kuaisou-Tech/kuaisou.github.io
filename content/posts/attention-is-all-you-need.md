---
title: "论文解读：Attention Is All You Need"
date: 2024-01-15
description: "Transformer 架构原论文深度解析，从自注意力机制到多头注意力，彻底理解现代大语言模型的基石。"
tags: ["Transformer", "注意力机制", "NLP", "论文解读"]
categories: ["论文解读"]
math: true
showtoc: true
tocopen: true
---

## 简介

2017 年，Google 发布了论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，提出了 **Transformer** 架构。这篇论文彻底改变了 NLP 领域，也成为了 GPT、BERT、LLaMA 等现代大模型的基石。

本文将深度解析 Transformer 的核心设计思想。

---

## 核心问题：为什么抛弃 RNN？

在 Transformer 之前，序列建模的主流方案是 RNN/LSTM。但 RNN 有两个根本性的缺陷：

1. **无法并行**：必须按时间步逐步计算，训练极慢
2. **长程依赖问题**：距离较远的词之间的关联难以捕捉

Transformer 用 **自注意力机制（Self-Attention）** 完全替代了循环结构。

---

## 自注意力机制

自注意力的核心思想：**让序列中每个位置都能直接关注到其他所有位置**。

### 计算公式

给定输入序列，首先将每个词向量线性投影为三个向量：

- $Q$（Query）：当前词"想找什么"
- $K$（Key）：每个词"提供什么"
- $V$（Value）：每个词"实际携带的信息"

注意力权重的计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $d_k$ 是 Key 的维度，除以 $\sqrt{d_k}$ 是为了防止点积过大导致 softmax 梯度消失。

### 直觉理解

| 符号 | 类比 |
|------|------|
| $Q$ | 搜索查询词 |
| $K$ | 文档的索引关键词 |
| $V$ | 文档的实际内容 |
| Attention | 根据查询词检索最相关的内容 |

---

## 多头注意力

单头注意力只能关注一种模式。**多头注意力（Multi-Head Attention）** 将 $Q, K, V$ 分别投影到 $h$ 个不同的子空间：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

不同的"头"可以并行关注不同的语言特征（语法、语义、指代关系等）。

---

## 位置编码

自注意力本身没有位置感知，Transformer 通过 **位置编码（Positional Encoding）** 注入位置信息：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

使用正弦/余弦函数的好处：可以外推到训练时未见过的序列长度。

---

## 整体架构

Transformer 由 **编码器（Encoder）** 和 **解码器（Decoder）** 两部分组成：

```
输入序列
    ↓
[词嵌入 + 位置编码]
    ↓
[编码器层 × N]
  ├─ 多头自注意力
  ├─ Add & Norm
  ├─ 前馈网络
  └─ Add & Norm
    ↓
[解码器层 × N]
  ├─ 掩码多头自注意力
  ├─ Add & Norm
  ├─ 交叉注意力（关注编码器输出）
  ├─ Add & Norm
  ├─ 前馈网络
  └─ Add & Norm
    ↓
输出序列
```

---

## 复杂度对比

| 模型 | 每层复杂度 | 最大路径长度 | 可并行 |
|------|-----------|------------|--------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | ✅ |
| RNN | $O(n \cdot d^2)$ | $O(n)$ | ❌ |
| CNN | $O(k \cdot n \cdot d^2)$ | $O(\log_k n)$ | ✅ |

自注意力的最大路径长度为 $O(1)$，这意味着任意两个词之间的信息只需一步就能交互，从根本上解决了长程依赖问题。

---

## 总结

Transformer 的三大核心贡献：

1. **自注意力替代循环**：实现完全并行，大幅提升训练效率
2. **多头注意力**：同时捕捉多种语言模式
3. **简洁的 Encoder-Decoder 架构**：成为后续所有大模型的设计范式

这篇论文发表至今已超过 7 年，但其核心思想仍然主宰着整个深度学习领域。
