# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00392v1">Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Processing long contexts has become a critical capability for modern large language models (LLMs). However, serving long-context LLMs comes with significant inference costs due to the high memory overhead of the key-value (KV) cache. Existing work leverages dynamic sparse attention algorithms (DSAes) to mitigate the KV cache overhead, but these algorithms rely on top-$k$ KV cache selection, which results in a trade-off between accuracy and efficiency. A larger $k$ improves accuracy but decreases efficiency, while a smaller $k$ boosts efficiency but compromises accuracy. To overcome this trade-off, this paper presents PSA, a $\underline{P}$rogressive $\underline{S}$parse $\underline{A}$ttention mechanism that integrates algorithmic innovations with system co-design to achieve both high inference accuracy and improved efficiency in LLM serving. The PSA algorithm adaptively adjusts the KV cache budget of different tokens and layers according to their real attention weight distributions, rather than relying on a fixed budget $k$. This enables high accuracy while minimizing KV cache usage. To further enhance execution efficiency, we introduce a pipelined iteration scheme that reduces CPU-GPU interleaving and synchronization overhead during PSA computation. Additionally, we implement unified GPU memory management that optimizes PSA's memory utilization by accounting for uneven memory requirements across different model layers. Extensive experimental results demonstrate that PSA reduces KV cache usage for attention computation by up to 2.4$\times$ and 8.8$\times$, and increases end-to-end serving throughput by up to 1.4$\times$ and 2.0$\times$, compared to state-of-the-art DSAes and systems without sparse attention, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00353v1">U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have expanded their context windows to unprecedented lengths, sparking debates about the necessity of Retrieval-Augmented Generation (RAG). To address the fragmented evaluation paradigms and limited cases in existing Needle-in-a-Haystack (NIAH), this paper introduces U-NIAH, a unified framework that systematically compares LLMs and RAG methods in controlled long context settings. Our framework extends beyond traditional NIAH by incorporating multi-needle, long-needle, and needle-in-needle configurations, along with different retrieval settings, while leveraging the synthetic Starlight Academy dataset-a fictional magical universe-to eliminate biases from pre-trained knowledge. Through extensive experiments, we investigate three research questions: (1) performance trade-offs between LLMs and RAG, (2) error patterns in RAG, and (3) RAG's limitations in complex settings. Our findings show that RAG significantly enhances smaller LLMs by mitigating the "lost-in-the-middle" effect and improving robustness, achieving an 82.58% win-rate over LLMs. However, we observe that retrieval noise and reverse chunk ordering degrade performance, while surprisingly, advanced reasoning LLMs exhibit reduced RAG compatibility due to sensitivity to semantic distractors. We identify typical error patterns including omission due to noise, hallucination under high noise critical condition, and self-doubt behaviors. Our work not only highlights the complementary roles of RAG and LLMs, but also provides actionable insights for optimizing deployments. Code: https://github.com/Tongji-KGLLM/U-NIAH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00330v1">How Deep is Love in LLMs' Hearts? Exploring Semantic Size in Human-like Cognition</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      How human cognitive abilities are formed has long captivated researchers. However, a significant challenge lies in developing meaningful methods to measure these complex processes. With the advent of large language models (LLMs), which now rival human capabilities in various domains, we are presented with a unique testbed to investigate human cognition through a new lens. Among the many facets of cognition, one particularly crucial aspect is the concept of semantic size, the perceived magnitude of both abstract and concrete words or concepts. This study seeks to investigate whether LLMs exhibit similar tendencies in understanding semantic size, thereby providing insights into the underlying mechanisms of human cognition. We begin by exploring metaphorical reasoning, comparing how LLMs and humans associate abstract words with concrete objects of varying sizes. Next, we examine LLMs' internal representations to evaluate their alignment with human cognitive processes. Our findings reveal that multi-modal training is crucial for LLMs to achieve more human-like understanding, suggesting that real-world, multi-modal experiences are similarly vital for human cognitive development. Lastly, we examine whether LLMs are influenced by attention-grabbing headlines with larger semantic sizes in a real-world web shopping scenario. The results show that multi-modal LLMs are more emotionally engaged in decision-making, but this also introduces potential biases, such as the risk of manipulation through clickbait headlines. Ultimately, this study offers a novel perspective on how LLMs interpret and internalize language, from the smallest concrete objects to the most profound abstract concepts like love. The insights gained not only improve our understanding of LLMs but also provide new avenues for exploring the cognitive abilities that define human intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00309v1">Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized natural language processing. However, these models face challenges in retrieving precise information from vast datasets. Retrieval-Augmented Generation (RAG) was developed to combining LLMs with external information retrieval systems to enhance the accuracy and context of responses. Despite improvements, RAG still struggles with comprehensive retrieval in high-volume, low-information-density databases and lacks relational awareness, leading to fragmented answers. To address this, this paper introduces the Pseudo-Knowledge Graph (PKG) framework, designed to overcome these limitations by integrating Meta-path Retrieval, In-graph Text and Vector Retrieval into LLMs. By preserving natural language text and leveraging various retrieval techniques, the PKG offers a richer knowledge representation and improves accuracy in information retrieval. Extensive evaluations using Open Compass and MultiHop-RAG benchmarks demonstrate the framework's effectiveness in managing large volumes of data and complex relationships.
    </details>
</div>
