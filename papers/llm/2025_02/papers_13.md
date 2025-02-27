# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- Part 13

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00339v1">Challenges and Innovations in LLM-Powered Fake News Detection: A Synthesis of Approaches and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      The pervasiveness of the dissemination of fake news through social media platforms poses critical risks to the trust of the general public, societal stability, and democratic institutions. This challenge calls for novel methodologies in detection, which can keep pace with the dynamic and multi-modal nature of misinformation. Recent works include powering the detection using large language model advances in multimodal frameworks, methodologies using graphs, and adversarial training in the literature of fake news. Based on the different approaches which can bring success, some key highlights will be underlined: enhanced LLM-improves accuracy through more advanced semantics and cross-modality fusion for robust detections. The review further identifies critical gaps in adaptability to dynamic social media trends, real-time, and cross-platform detection capabilities, as well as the ethical challenges thrown up by the misuse of LLMs. Future directions underline the development of style-agnostic models, cross-lingual detection frameworks, and robust policies with a view to mitigating LLM-driven misinformation. This synthesis thus lays a concrete foundation for those researchers and practitioners committed to reinforcing fake news detection systems with complications that keep on growing in the digital landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v1">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      To reduce memory costs in long-context inference with Large Language Models (LLMs), many recent works focus on compressing the key-value (KV) cache of different tokens. However, we identify that the previous KV cache compression methods measure token importance individually, neglecting the dependency between different tokens in the real-world language characterics. In light of this, we introduce ChunkKV, grouping the tokens in a chunk as a basic compressing unit, and retaining the most informative semantic chunks while discarding the less important ones. Furthermore, observing that ChunkKV exhibits higher similarity in the preserved indices across different layers, we propose layer-wise index reuse to further reduce computational overhead. We evaluated ChunkKV on cutting-edge long-context benchmarks including LongBench and Needle-In-A-HayStack, as well as the GSM8K and JailbreakV in-context learning benchmark. Our experiments with instruction tuning and multi-step reasoning (O1 and R1) LLMs, achieve up to 10\% performance improvement under aggressive compression ratios compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v1">Estimating LLM Uncertainty with Logits</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have seen remarkable advancements and have been extensively integrated across various fields. Despite their progress, LLMs are prone to hallucinations, producing responses that may not be dependable if the models lack sufficient grounding knowledge. To mitigate this issue, methods for estimating uncertainty have been adopted, with a focus on critical tokens as indicators of reliability. Nevertheless, probability-based approaches have shown limitations in assessing token-level reliability due to the erosion of evidence strength information acquired during training. In this paper, we introduce Logits-induced Token Uncertainty (LogU), a novel framework designed to estimate token-specific uncertainty in LLMs in real time, without the need for multiple sampling rounds. By leveraging evidence modeling for the implementation of LogU, we utilize the derived uncertainty measures to steer downstream tasks. Our experimental findings highlight the substantial effectiveness and potential of LogU, marking a significant advancement in addressing the challenge of model hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00258v1">ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance in natural language processing tasks, yet their massive size makes serving them inefficient and costly. Semi-structured pruning has emerged as an effective method for model acceleration, but existing approaches are suboptimal because they focus on local, layer-wise optimizations using heuristic rules, failing to leverage global feedback. We present ProxSparse, a learning-based framework for mask selection enabled by regularized optimization. ProxSparse transforms the rigid, non-differentiable mask selection process into a smoother optimization procedure, allowing gradual mask exploration with flexibility. ProxSparse does not involve additional weight updates once the mask is determined. Our extensive evaluations on 7 widely used models show that ProxSparse consistently outperforms previously proposed semi-structured mask selection methods with significant improvement, demonstrating the effectiveness of our learned approach towards semi-structured pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14654v3">Comparative Analysis of Pooling Mechanisms in LLMs: A Sentiment Analysis Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Accepted to ISMSI'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP) by delivering state-of-the-art performance across a variety of tasks. Among these, Transformer-based models like BERT and GPT rely on pooling layers to aggregate token-level embeddings into sentence-level representations. Common pooling mechanisms such as Mean, Max, and Weighted Sum play a pivotal role in this aggregation process. Despite their widespread use, the comparative performance of these strategies on different LLM architectures remains underexplored. To address this gap, this paper investigates the effects of these pooling mechanisms on two prominent LLM families -- BERT and GPT, in the context of sentence-level sentiment analysis. Comprehensive experiments reveal that each pooling mechanism exhibits unique strengths and weaknesses depending on the task's specific requirements. Our findings underline the importance of selecting pooling methods tailored to the demands of particular applications, prompting a re-evaluation of common assumptions regarding pooling operations. By offering actionable insights, this study contributes to the optimization of LLM-based models for downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02625v2">HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Quantized training of Large Language Models (LLMs) remains an open challenge, as maintaining accuracy while performing all matrix multiplications in low precision has proven difficult. This is particularly the case when fine-tuning pre-trained models, which can have large weight and activation outlier values that make lower-precision optimization difficult. To address this, we present HALO, a novel quantization-aware training approach for Transformers that enables accurate and efficient low-precision training by combining 1) strategic placement of Hadamard rotations in both forward and backward passes, which mitigate outliers, 2) high-performance kernel support, and 3) FSDP integration for low-precision communication. Our approach ensures that all large matrix multiplications during the forward and backward passes are executed in lower precision. Applied to LLAMA-family models, HALO achieves near-full-precision-equivalent results during fine-tuning on various tasks, while delivering up to 1.41x end-to-end speedup for full fine-tuning on RTX 4090 GPUs. HALO efficiently supports both standard and parameterefficient fine-tuning (PEFT). Our results demonstrate the first practical approach to fully quantized LLM fine-tuning that maintains accuracy in 8-bit precision, while delivering performance benefits. Code is available at \url{https://github.com/IST-DASLab/HALO}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v2">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 16 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
