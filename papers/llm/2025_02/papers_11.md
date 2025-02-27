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
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04008v1">Automating a Complete Software Test Process Using LLMs: An Automotive Case Study</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted by International Conference on Software Engineering (ICSE) 2025
    </div>
    <details class="paper-abstract">
      Vehicle API testing verifies whether the interactions between a vehicle's internal systems and external applications meet expectations, ensuring that users can access and control various vehicle functions and data. However, this task is inherently complex, requiring the alignment and coordination of API systems, communication protocols, and even vehicle simulation systems to develop valid test cases. In practical industrial scenarios, inconsistencies, ambiguities, and interdependencies across various documents and system specifications pose significant challenges. This paper presents a system designed for the automated testing of in-vehicle APIs. By clearly defining and segmenting the testing process, we enable Large Language Models (LLMs) to focus on specific tasks, ensuring a stable and controlled testing workflow. Experiments conducted on over 100 APIs demonstrate that our system effectively automates vehicle API testing. The results also confirm that LLMs can efficiently handle mundane tasks requiring human judgment, making them suitable for complete automation in similar industrial contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09243v3">SPRec: Self-Play to Debias LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted by WWW 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have attracted significant attention in recommendation systems. Current work primarily applies supervised fine-tuning (SFT) to adapt the model for recommendation tasks. However, SFT on positive examples only limits the model's ability to align with user preference. To address this, researchers recently introduced Direct Preference Optimization (DPO), which explicitly aligns LLMs with user preferences using offline preference ranking data. However, we found that DPO inherently biases the model towards a few items, exacerbating the filter bubble issue and ultimately degrading user experience. In this paper, we propose SPRec, a novel self-play framework designed to mitigate over-recommendation and improve fairness without requiring additional data or manual intervention. In each self-play iteration, the model undergoes an SFT step followed by a DPO step, treating offline interaction data as positive samples and the predicted outputs from the previous iteration as negative samples. This effectively re-weights the DPO loss function using the model's logits, adaptively suppressing biased items. Extensive experiments on multiple real-world datasets demonstrate SPRec's effectiveness in enhancing recommendation accuracy and fairness. The implementation is available via https://github.com/RegionCh/SPRec
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01866v2">House of Cards: Massive Weights in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Massive activations, which manifest in specific feature dimensions of hidden states, introduce a significant bias in large language models (LLMs), leading to an overemphasis on the corresponding token. In this paper, we identify that massive activations originate not from the hidden state but from the intermediate state of a feed-forward network module in an early layer. Expanding on the previous observation that massive activations occur only in specific feature dimensions, we dive deep into the weights that cause massive activations. Specifically, we define top-$k$ massive weights as the weights that contribute to the dimensions with the top-$k$ magnitudes in the intermediate state. When these massive weights are set to zero, the functionality of LLMs is entirely disrupted. However, when all weights except for massive weights are set to zero, it results in a relatively minor performance drop, even though a much larger number of weights are set to zero. This implies that during the pre-training process, learning is dominantly focused on massive weights. Building on this observation, we propose a simple plug-and-play method called MacDrop (massive weights curriculum dropout), to rely less on massive weights during parameter-efficient fine-tuning. This method applies dropout to the pre-trained massive weights, starting with a high dropout probability and gradually decreasing it as fine-tuning progresses. Through various experiments, including zero-shot downstream tasks, long-context tasks, and ablation studies, we demonstrate that \texttt{MacDrop} generally improves performance and strengthens robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01269v4">CPRM: A LLM-based Continual Pre-training Framework for Relevance Modeling in Commercial Search</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Relevance modeling between queries and items stands as a pivotal component in commercial search engines, directly affecting the user experience. Given the remarkable achievements of large language models (LLMs) in various natural language processing (NLP) tasks, LLM-based relevance modeling is gradually being adopted within industrial search systems. Nevertheless, foundational LLMs lack domain-specific knowledge and do not fully exploit the potential of in-context learning. Furthermore, structured item text remains underutilized, and there is a shortage in the supply of corresponding queries and background knowledge. We thereby propose CPRM (Continual Pre-training for Relevance Modeling), a framework designed for the continual pre-training of LLMs to address these issues. Our CPRM framework includes three modules: 1) employing both queries and multi-field item to jointly pre-train for enhancing domain knowledge, 2) applying in-context pre-training, a novel approach where LLMs are pre-trained on a sequence of related queries or items, and 3) conducting reading comprehension on items to produce associated domain knowledge and background information (e.g., generating summaries and corresponding queries) to further strengthen LLMs. Results on offline experiments and online A/B testing demonstrate that our model achieves convincing performance compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03964v1">"It Warned Me Just at the Right Moment": Exploring LLM-based Real-time Detection of Phone Scams</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Despite living in the era of the internet, phone-based scams remain one of the most prevalent forms of scams. These scams aim to exploit victims for financial gain, causing both monetary losses and psychological distress. While governments, industries, and academia have actively introduced various countermeasures, scammers also continue to evolve their tactics, making phone scams a persistent threat. To combat these increasingly sophisticated scams, detection technologies must also advance. In this work, we propose a framework for modeling scam calls and introduce an LLM-based real-time detection approach, which assesses fraudulent intent in conversations, further providing immediate warnings to users to mitigate harm. Through experiments, we evaluate the method's performance and analyze key factors influencing its effectiveness. This analysis enables us to refine the method to improve precision while exploring the trade-off between recall and timeliness, paving the way for future directions in this critical area of research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18652v6">$C^2$: Scalable Auto-Feedback for LLM-based Chart Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 NAACL 2025 Main (Long)
    </div>
    <details class="paper-abstract">
      Generating high-quality charts with Large Language Models (LLMs) presents significant challenges due to limited data and the high cost of scaling through human curation. $\langle \text{instruction}, \text{data}, \text{code} \rangle$ triplets are scarce and expensive to manually curate as their creation demands technical expertise. To address this scalability challenge, we introduce a reference-free automatic feedback generator, which eliminates the need for costly human intervention. Our novel framework, C$^2$, consists of (1) an automatic feedback provider (ChartAF) and (2) a diverse, reference-free dataset (ChartUIE-8K). The results are compelling: in our first experiment, 74% of respondents strongly preferred, and 10% preferred, the results after feedback. The second post-feedback experiment demonstrates that ChartAF outperform nine baselines. Moreover, ChartUIE-8K significantly improves data diversity by increasing queries, datasets, and chart types by 5982%, 1936%, and 91%, respectively, over benchmarks. Finally, a study of LLM users revealed that 94% of participants preferred ChartUIE-8K's queries, with 93% deeming them aligned with real-world use cases. Core contributions are available as open-source at chartsquared.github.io, with ample qualitative examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v1">InfinitePOD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 takes a middle-ground approach by leveraging Optical Circuit Switches, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfinitePOD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfinitePOD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt into variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh) based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfinitePOD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03884v1">Rank Also Matters: Hierarchical Configuration for Mixture of Adapter Experts in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across various tasks, accompanied by a continuous increase in their parameter size. Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), address the challenges of fine-tuning LLMs by significantly reducing the number of trainable parameters. Recent studies have integrated LoRA with Mixture of Experts (MoE) architectures, leveraging multiple adapter experts and gating mechanisms to further improve fine-tuning performance. However, existing approaches primarily focus on adjusting the allocations of adapter experts per layer to optimize the introduced trainable parameter size, while neglecting a critical factor of adapters' rank. To this end, we propose a hierarchical scheme for expert allocation and rank configuration, HILO, which dynamically adjusts the number and rank of adapter experts across layers, matching the varying representational complexity of model layers in adapter-granularity. Extensive experiments on multiple benchmark tasks demonstrate that HILO outperforms existing methods in accuracy while introducing fewer trainable parameters, providing an efficient and practical solution for fine-tuning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03843v1">Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      High-quality, large-scale instructions are crucial for aligning large language models (LLMs), however, there is a severe shortage of instruction in the field of natural language understanding (NLU). Previous works on constructing NLU instructions mainly focus on information extraction (IE), neglecting tasks such as machine reading comprehension, question answering, and text classification. Furthermore, the lack of diversity in the data has led to a decreased generalization ability of trained LLMs in other NLU tasks and a noticeable decline in the fundamental model's general capabilities. To address this issue, we propose Hum, a large-scale, high-quality synthetic instruction corpus for NLU tasks, designed to enhance the NLU capabilities of LLMs. Specifically, Hum includes IE (either close IE or open IE), machine reading comprehension, text classification, and instruction generalist tasks, thereby enriching task diversity. Additionally, we introduce a human-LLMs collaborative mechanism to synthesize instructions, which enriches instruction diversity by incorporating guidelines, preference rules, and format variants. We conduct extensive experiments on 5 NLU tasks and 28 general capability evaluation datasets for LLMs. Experimental results show that Hum enhances the NLU capabilities of six LLMs by an average of 3.1\%, with no significant decline observed in other general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03824v1">Syntriever: How to Train Your Retriever with Synthetic Data from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted
    </div>
    <details class="paper-abstract">
      LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06387v4">Self-Training Meets Consistency: Improving LLMs' Reasoning with Consistency-Driven Rationale Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Self-training approach for large language models (LLMs) improves reasoning abilities by training the models on their self-generated rationales. Previous approaches have labeled rationales that produce correct answers for a given question as appropriate for training. However, a single measure risks misjudging rationale quality, leading the models to learn flawed reasoning patterns. To address this issue, we propose CREST (Consistency-driven Rationale Evaluation for Self-Training), a self-training framework that further evaluates each rationale through follow-up questions and leverages this evaluation to guide its training. Specifically, we introduce two methods: (1) filtering out rationales that frequently result in incorrect answers on follow-up questions and (2) preference learning based on mixed preferences from rationale evaluation results of both original and follow-up questions. Experiments on three question-answering datasets using open LLMs show that CREST not only improves the logical robustness and correctness of rationales but also improves reasoning abilities compared to previous self-training approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01720v3">Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Synthetic data has become a pivotal resource in post-training tasks for large language models (LLMs) due to the scarcity of high-quality, specific data. While various methods have been developed to generate synthetic data, there remains a discernible gap between the practical effects of synthetic data and our theoretical comprehension. To address this challenge, we commence by presenting a detailed modeling of the prevalent synthetic data generation process. Building upon this modeling, we demonstrate that the generalization capability of the post-trained model is critically determined by the information gain derived from the generative model, as analyzed from a novel reverse-bottleneck perspective. Moreover, we introduce the concept of Generalization Gain via Mutual Information (GGMI) and elucidate the relationship between generalization gain and information gain. This analysis serves as a theoretical foundation for synthetic data generation and further highlights its connection with the generalization capability of post-trained models, offering an understanding about the design of synthetic data generation techniques and the optimization of the post-training process. We open-source our code at https://github.com/ZyGan1999/Towards-a-Theoretical-Understanding-of-Synthetic-Data-in-LLM-Post-Training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14259v2">Beyond Binary: Towards Fine-Grained LLM-Generated Text Detection via Role Recognition and Involvement Measurement</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Social Media, Large Language Models, LLM-generated Text Detection, AI-assisted News Detection; Accepted by WWW2025
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs), like ChatGPT, has resulted in the widespread presence of LLM-generated content on social media platforms, raising concerns about misinformation, data biases, and privacy violations, which can undermine trust in online discourse. While detecting LLM-generated content is crucial for mitigating these risks, current methods often focus on binary classification, failing to address the complexities of real-world scenarios like human-LLM collaboration. To move beyond binary classification and address these challenges, we propose a new paradigm for detecting LLM-generated content. This approach introduces two novel tasks: LLM Role Recognition (LLM-RR), a multi-class classification task that identifies specific roles of LLM in content generation, and LLM Influence Measurement (LLM-IM), a regression task that quantifies the extent of LLM involvement in content creation. To support these tasks, we propose LLMDetect, a benchmark designed to evaluate detectors' performance on these new tasks. LLMDetect includes the Hybrid News Detection Corpus (HNDC) for training detectors, as well as DetectEval, a comprehensive evaluation suite that considers five distinct cross-context variations and two multi-intensity variations within the same LLM role. This allows for a thorough assessment of detectors' generalization and robustness across diverse contexts. Our empirical validation of 10 baseline detection methods demonstrates that fine-tuned PLM-based models consistently outperform others on both tasks, while advanced LLMs face challenges in accurately detecting their own generated content. Our experimental results and analysis offer insights for developing more effective detection models for LLM-generated content. This research enhances the understanding of LLM-generated content and establishes a foundation for more nuanced detection methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v3">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 32 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving simple scientific problems but, even with domain-specific fine-tuning, often produce hallucinations for complex ones. While integrating LLMs with tools can mitigate this reliability issue, models finetuned on tool usage only often over-rely on them, incurring unnecessary costs from resource-intensive scientific tools even for simpler problems. Inspired by how human experts assess the complexity of the problem before choosing the solutions, we propose a novel two-component fine-tuning method, Adapting While Learning (AWL). In the first component, World Knowledge Learning (WKL), LLMs internalize scientific knowledge by learning from tools-generated solutions. In the second component, Tool Usage Adaptation (TUA), we classify questions as easy or hard based on the WKL-trained model's accuracy, and train it to maintain direct reasoning for simple problems while switching to tools for challenging ones. We validate our method on 6 scientific benchmark datasets in climate science, epidemiology, and mathematics. Compared to the base 8B model, our trained models achieve 28.27% higher answer accuracy and 13.76% better tool usage accuracy, even surpassing state-of-the-art models including GPT-4 and Claude-3.5 on 4 custom-created datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01070v2">An Investigation of FP8 Across Accelerators for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      The introduction of 8-bit floating-point (FP8) computation units in modern AI accelerators has generated significant interest in FP8-based large language model (LLM) inference. Unlike 16-bit floating-point formats, FP8 in deep learning requires a shared scaling factor. Additionally, while E4M3 and E5M2 are well-defined at the individual value level, their scaling and accumulation methods remain unspecified and vary across hardware and software implementations. As a result, FP8 behaves more like a quantization format than a standard numeric representation. In this work, we provide the first comprehensive analysis of FP8 computation and acceleration on two AI accelerators: the NVIDIA H100 and Intel Gaudi 2. Our findings highlight that the Gaudi 2, by leveraging FP8, achieves higher throughput-to-power efficiency during LLM inference, offering valuable insights into the practical implications of FP8 adoption for datacenter-scale LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03723v1">Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Credit assignment, the process of attributing credit or blame to individual agents for their contributions to a team's success or failure, remains a fundamental challenge in multi-agent reinforcement learning (MARL), particularly in environments with sparse rewards. Commonly-used approaches such as value decomposition often lead to suboptimal policies in these settings, and designing dense reward functions that align with human intuition can be complex and labor-intensive. In this work, we propose a novel framework where a large language model (LLM) generates dense, agent-specific rewards based on a natural language description of the task and the overall team goal. By learning a potential-based reward function over multiple queries, our method reduces the impact of ranking errors while allowing the LLM to evaluate each agent's contribution to the overall task. Through extensive experiments, we demonstrate that our approach achieves faster convergence and higher policy returns compared to state-of-the-art MARL baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03708v1">Aggregate and conquer: detecting and steering LLM concepts by combining nonlinear predictors over multiple layers</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      A trained Large Language Model (LLM) contains much of human knowledge. Yet, it is difficult to gauge the extent or accuracy of that knowledge, as LLMs do not always ``know what they know'' and may even be actively misleading. In this work, we give a general method for detecting semantic concepts in the internal activations of LLMs. Furthermore, we show that our methodology can be easily adapted to steer LLMs toward desirable outputs. Our innovations are the following: (1) we use a nonlinear feature learning method to identify important linear directions for predicting concepts from each layer; (2) we aggregate features across layers to build powerful concept detectors and steering mechanisms. We showcase the power of our approach by attaining state-of-the-art results for detecting hallucinations, harmfulness, toxicity, and untruthful content on seven benchmarks. We highlight the generality of our approach by steering LLMs towards new concepts that, to the best of our knowledge, have not been previously considered in the literature, including: semantic disambiguation, human languages, programming languages, hallucinated responses, science subjects, poetic/Shakespearean English, and even multiple concepts simultaneously. Moreover, our method can steer concepts with numerical attributes such as product reviews. We provide our code (including a simple API for our methods) at https://github.com/dmbeaglehole/neural_controllers .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03699v1">LLM Alignment as Retriever Optimization: An Information Retrieval Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03688v1">A Comparison of DeepSeek and Other LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 21 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Recently, DeepSeek has been the focus of attention in and beyond the AI community. An interesting problem is how DeepSeek compares to other large language models (LLMs). There are many tasks an LLM can do, and in this paper, we use the task of predicting an outcome using a short text for comparison. We consider two settings, an authorship classification setting and a citation classification setting. In the first one, the goal is to determine whether a short text is written by human or AI. In the second one, the goal is to classify a citation to one of four types using the textual content. For each experiment, we compare DeepSeek with $4$ popular LLMs: Claude, Gemini, GPT, and Llama. We find that, in terms of classification accuracy, DeepSeek outperforms Gemini, GPT, and Llama in most cases, but underperforms Claude. We also find that DeepSeek is comparably slower than others but with a low cost to use, while Claude is much more expensive than all the others. Finally, we find that in terms of similarity, the output of DeepSeek is most similar to those of Gemini and Claude (and among all $5$ LLMs, Claude and Gemini have the most similar outputs). In this paper, we also present a fully-labeled dataset collected by ourselves, and propose a recipe where we can use the LLMs and a recent data set, MADStat, to generate new data sets. The datasets in our paper can be used as benchmarks for future study on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v2">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16594v6">From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 v6: add new citations; 36 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Assessment and evaluation have long been critical challenges in artificial intelligence (AI) and natural language processing (NLP). However, traditional methods, whether matching-based or embedding-based, often fall short of judging subtle attributes and delivering satisfactory results. Recent advancements in Large Language Models (LLMs) inspire the "LLM-as-a-judge" paradigm, where LLMs are leveraged to perform scoring, ranking, or selection across various tasks and applications. This paper provides a comprehensive survey of LLM-based judgment and assessment, offering an in-depth overview to advance this emerging field. We begin by giving detailed definitions from both input and output perspectives. Then we introduce a comprehensive taxonomy to explore LLM-as-a-judge from three dimensions: what to judge, how to judge and where to judge. Finally, we compile benchmarks for evaluating LLM-as-a-judge and highlight key challenges and promising directions, aiming to provide valuable insights and inspire future research in this promising research area. Paper list and more resources about LLM-as-a-judge can be found at https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge and https://llm-as-a-judge.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03685v1">Controlled LLM Decoding via Discrete Auto-regressive Biasing</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Controlled text generation allows for enforcing user-defined constraints on large language model outputs, an increasingly important field as LLMs become more prevalent in everyday life. One common approach uses energy-based decoding, which defines a target distribution through an energy function that combines multiple constraints into a weighted average. However, these methods often struggle to balance fluency with constraint satisfaction, even with extensive tuning of the energy function's coefficients. In this paper, we identify that this suboptimal balance arises from sampling in continuous space rather than the natural discrete space of text tokens. To address this, we propose Discrete Auto-regressive Biasing, a controlled decoding algorithm that leverages gradients while operating entirely in the discrete text domain. Specifically, we introduce a new formulation for controlled text generation by defining a joint distribution over the generated sequence and an auxiliary bias sequence. To efficiently sample from this joint distribution, we propose a Langevin-within-Gibbs sampling algorithm using gradient-based discrete MCMC. Our method significantly improves constraint satisfaction while maintaining comparable or better fluency, all with even lower computational costs. We demonstrate the advantages of our controlled decoding method on sentiment control, language detoxification, and keyword-guided generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04564v1">My LLM might Mimic AAE -- But When Should it?</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      We examine the representation of African American English (AAE) in large language models (LLMs), exploring (a) the perceptions Black Americans have of how effective these technologies are at producing authentic AAE, and (b) in what contexts Black Americans find this desirable. Through both a survey of Black Americans ($n=$ 104) and annotation of LLM-produced AAE by Black Americans ($n=$ 228), we find that Black Americans favor choice and autonomy in determining when AAE is appropriate in LLM output. They tend to prefer that LLMs default to communicating in Mainstream U.S. English in formal settings, with greater interest in AAE production in less formal settings. When LLMs were appropriately prompted and provided in context examples, our participants found their outputs to have a level of AAE authenticity on par with transcripts of Black American speech. Select code and data for our project can be found here: https://github.com/smelliecat/AAEMime.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04563v1">WaferLLM: A Wafer-Scale LLM Inference System</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Emerging AI accelerators increasingly adopt wafer-scale manufacturing technologies, integrating hundreds of thousands of AI cores in a mesh-based architecture with large distributed on-chip memory (tens of GB in total) and ultra-high on-chip memory bandwidth (tens of PB/s). However, current LLM inference systems, optimized for shared memory architectures like GPUs, fail to fully exploit these accelerators. We introduce WaferLLM, the first wafer-scale LLM inference system. WaferLLM is guided by a novel PLMR device model that captures the unique hardware characteristics of wafer-scale architectures. Leveraging this model, WaferLLM pioneers wafer-scale LLM parallelism, optimizing the utilization of hundreds of thousands of on-chip cores. It also introduces MeshGEMM and MeshGEMV, the first GEMM and GEMV implementations designed to scale effectively on wafer-scale accelerators. Evaluations show that WaferLLM achieves 200$\times$ better wafer-scale accelerator utilization than state-of-the-art systems. On a commodity wafer-scale accelerator, WaferLLM delivers 606$\times$ faster and 22$\times$ more energy-efficient GEMV compared to an advanced GPU. For LLMs, WaferLLM enables 39$\times$ faster decoding with 1.7$\times$ better energy efficiency. We anticipate these numbers will grow significantly as wafer-scale AI models, software, and hardware continue to mature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04556v1">TruthFlow: Truthful LLM Generation via Representation Flow Correction</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to struggle with consistently generating truthful responses. While various representation intervention techniques have been proposed, these methods typically apply a universal representation correction vector to all input queries, limiting their effectiveness against diverse queries in practice. In this study, we introduce TruthFlow, a novel method that leverages the Flow Matching technique for query-specific truthful representation correction. Specifically, TruthFlow first uses a flow model to learn query-specific correction vectors that transition representations from hallucinated to truthful states. Then, during inference, the trained flow model generates these correction vectors to enhance the truthfulness of LLM outputs. Experimental results demonstrate that TruthFlow significantly improves performance on open-ended generation tasks across various advanced LLMs evaluated on TruthfulQA. Moreover, the trained TruthFlow model exhibits strong transferability, performing effectively on other unseen hallucination benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14442v2">A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to NAACL2025 main conference
    </div>
    <details class="paper-abstract">
      Recently, sharing key-value (KV) cache across layers has been found effective in efficient inference of large language models (LLMs). To systematically investigate different techniques of cross-layer KV sharing, we propose a unified framework that covers several recent methods and their novel variants. We conduct comprehensive experiments on all the configurations of the framework, evaluating their generation throughput and performance in language modeling and downstream tasks. We find that when reducing the size of the KV cache by 2$\times$, most configurations can achieve higher throughput than standard transformers while maintaining competitive performance. When further reducing the size of the KV cache, however, pairing queries of all layers with KVs of upper layers performs better, at the expense of additional training cost and prefilling latency. We hope that this work will help users make more informed choices of cross-layer KV sharing approaches and facilitate future research on efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10020v3">Lost in Overlap: Exploring Logit-based Watermark Collision in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Long Paper, 9 pages, accepted at NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13331v2">Qrazor: Reliable and Effortless 4-bit LLM Quantization by Significant Data Razoring</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) excel in language processing tasks but face deployment challenges due to high memory and computational demands. While low-bit quantization, such as 4-bit techniques, offers a potential solution, these methods often suffer from significant accuracy loss or require considerable effort for implementation such as reordering, rotation, etc. To address these challenges, we propose QRazor, a simple yet effective quantization scheme that enables 4-bit quantization of weights, activations, and KV cache in transformer-based LLMs. QRazor operates in two stages: first, quantizing data using 8 or 16-bit integers as a basis with absolute max scaling to preserve accuracy close to full-precision models, and second, compressing the quantized data to 4-bit using our significant data razoring (SDR) technique, which retains only the four most salient bits. Without any additional requirment of fine-tuning or additional training, QRazor achieves performance similar or better compared to state-of-the-art in 4-bit quantization method, surpassing Smoothquant and QLLM by over 12 points and Quarot(RTN) by more than 2.9 points in zero-shot reasoning task accuracy on the LLaMA2-7B model. Additionally, we introduce an integer-based arithmetic unit optimized for QRazor, allowing direct low-precision operations on SDR data without decompression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02966v1">FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      We propose FACTER, a fairness-aware framework for LLM-based recommendation systems that integrates conformal prediction with dynamic prompt engineering. By introducing an adaptive semantic variance threshold and a violation-triggered mechanism, FACTER automatically tightens fairness constraints whenever biased patterns emerge. We further develop an adversarial prompt generator that leverages historical violations to reduce repeated demographic biases without retraining the LLM. Empirical results on MovieLens and Amazon show that FACTER substantially reduces fairness violations (up to 95.5%) while maintaining strong recommendation accuracy, revealing semantic variance as a potent proxy of bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14202v3">Rationale Behind Essay Scores: Enhancing S-LLM's Multi-Trait Essay Scoring with Rationale Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Existing automated essay scoring (AES) has solely relied on essay text without using explanatory rationales for the scores, thereby forgoing an opportunity to capture the specific aspects evaluated by rubric indicators in a fine-grained manner. This paper introduces Rationale-based Multiple Trait Scoring (RMTS), a novel approach for multi-trait essay scoring that integrates prompt-engineering-based large language models (LLMs) with a fine-tuning-based essay scoring model using a smaller large language model (S-LLM). RMTS uses an LLM-based trait-wise rationale generation system where a separate LLM agent generates trait-specific rationales based on rubric guidelines, which the scoring model uses to accurately predict multi-trait scores. Extensive experiments on benchmark datasets, including ASAP, ASAP++, and Feedback Prize, show that RMTS significantly outperforms state-of-the-art models and vanilla S-LLMs in trait-specific scoring. By assisting quantitative assessment with fine-grained qualitative rationales, RMTS enhances the trait-wise reliability, providing partial explanations about essays. The code is available at https://github.com/BBeeChu/RMTS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00165v3">TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to WWW 2025 Research Track
    </div>
    <details class="paper-abstract">
      Hierarchical text classification aims to categorize each document into a set of classes in a label taxonomy, which is a fundamental web text mining task with broad applications such as web content analysis and semantic indexing. Most earlier works focus on fully or semi-supervised methods that require a large amount of human annotated data which is costly and time-consuming to acquire. To alleviate human efforts, in this paper, we work on hierarchical text classification with a minimal amount of supervision: using the sole class name of each node as the only supervision. Recently, large language models (LLM) have shown competitive performance on various tasks through zero-shot prompting, but this method performs poorly in the hierarchical setting because it is ineffective to include the large and structured label space in a prompt. On the other hand, previous weakly-supervised hierarchical text classification methods only utilize the raw taxonomy skeleton and ignore the rich information hidden in the text corpus that can serve as additional class-indicative features. To tackle the above challenges, we propose TELEClass, Taxonomy Enrichment and LLM-Enhanced weakly-supervised hierarchical text Classification, which combines the general knowledge of LLMs and task-specific features mined from an unlabeled corpus. TELEClass automatically enriches the raw taxonomy with class-indicative features for better label space understanding and utilizes novel LLM-based data annotation and generation methods specifically tailored for the hierarchical setting. Experiments show that TELEClass can significantly outperform previous baselines while achieving comparable performance to zero-shot prompting of LLMs with drastically less inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11629v6">Can Many-Shot In-Context Learning Help LLMs as Evaluators? A Preliminary Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      Utilizing Large Language Models (LLMs) as evaluators to assess the performance of LLMs has garnered attention. However, this kind of evaluation approach is affected by potential biases within LLMs, raising concerns about the accuracy and reliability of the evaluation results of LLMs. To address this problem, we propose and study two many-shot In-Context Learning (ICL) prompt templates to help LLM evaluators mitigate potential biases: Many-Shot with Reference (MSwR) and Many-Shot without Reference (MSoR). Specifically, the former utilizes in-context examples with model-generated evaluation rationales as references, while the latter does not include these references. Using these prompt designs, we investigate the impact of increasing the number of in-context examples on the consistency and quality of the evaluation results. Experimental results show that advanced LLMs, such as GPT-4o, perform better in the many-shot regime than in the zero-shot and few-shot regimes. Furthermore, when using GPT-4o as an evaluator in the many-shot regime, adopting MSwR as the prompt template performs better than MSoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02909v1">SPARC: Subspace-Aware Prompt Adaptation for Robust Continual Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      We propose SPARC, a lightweight continual learning framework for large language models (LLMs) that enables efficient task adaptation through prompt tuning in a lower-dimensional space. By leveraging principal component analysis (PCA), we identify a compact subspace of the training data. Optimizing prompts in this lower-dimensional space enhances training efficiency, as it focuses updates on the most relevant features while reducing computational overhead. Furthermore, since the model's internal structure remains unaltered, the extensive knowledge gained from pretraining is fully preserved, ensuring that previously learned information is not compromised during adaptation. Our method achieves high knowledge retention in both task-incremental and domain-incremental continual learning setups while fine-tuning only 0.04% of the model's parameters. Additionally, by integrating LoRA, we enhance adaptability to computational constraints, allowing for a tradeoff between accuracy and training cost. Experiments on the SuperGLUE benchmark demonstrate that our PCA-based prompt tuning combined with LoRA maintains full knowledge retention while improving accuracy, utilizing only 1% of the model's parameters. These results establish our approach as a scalable and resource-efficient solution for continual learning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02896v1">A Benchmark for the Detection of Metalinguistic Disagreements between LLMs and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 6 pages, 2 tables, to appear in Reham Alharbi, Jacopo de Berardinis, Paul Groth, Albert Mero\~no-Pe\~nuela, Elena Simperl, Valentina Tamma (eds.), ISWC 2024 Special Session on Harmonising Generative AI and Semantic Web Technologies. CEUR-WS.org (forthcoming), for associated code and data see https://github.com/bradleypallen/trex-metalinguistic-disagreement
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) for tasks like fact extraction in support of knowledge graph construction frequently involves computing accuracy metrics using a ground truth benchmark based on a knowledge graph (KG). These evaluations assume that errors represent factual disagreements. However, human discourse frequently features metalinguistic disagreement, where agents differ not on facts but on the meaning of the language used to express them. Given the complexity of natural language processing and generation using LLMs, we ask: do metalinguistic disagreements occur between LLMs and KGs? Based on an investigation using the T-REx knowledge alignment dataset, we hypothesize that metalinguistic disagreement does in fact occur between LLMs and KGs, with potential relevance for the practice of knowledge graph engineering. We propose a benchmark for evaluating the detection of factual and metalinguistic disagreements between LLMs and KGs. An initial proof of concept of such a benchmark is available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02893v1">Lowering the Barrier of Machine Learning: Achieving Zero Manual Labeling in Review Classification Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to 2025 11th International Conference on Computing and Artificial Intelligence (ICCAI 2025)
    </div>
    <details class="paper-abstract">
      With the internet's evolution, consumers increasingly rely on online reviews for service or product choices, necessitating that businesses analyze extensive customer feedback to enhance their offerings. While machine learning-based sentiment classification shows promise in this realm, its technical complexity often bars small businesses and individuals from leveraging such advancements, which may end up making the competitive gap between small and large businesses even bigger in terms of improving customer satisfaction. This paper introduces an approach that integrates large language models (LLMs), specifically Generative Pre-trained Transformer (GPT) and Bidirectional Encoder Representations from Transformers (BERT)-based models, making it accessible to a wider audience. Our experiments across various datasets confirm that our approach retains high classification accuracy without the need for manual labeling, expert knowledge in tuning and data annotation, or substantial computational power. By significantly lowering the barriers to applying sentiment classification techniques, our methodology enhances competitiveness and paves the way for making machine learning technology accessible to a broader audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06540v4">Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Scaling laws for large language models (LLMs) predict model performance based on parameters like size and training data. However, differences in training configurations and data processing across model families lead to significant variations in benchmark performance, making it difficult for a single scaling law to generalize across all LLMs. On the other hand, training family-specific scaling laws requires training models of varying sizes for every family. In this work, we propose Skills Scaling Laws (SSLaws, pronounced as Sloth), a novel scaling law that leverages publicly available benchmark data and assumes LLM performance is driven by low-dimensional latent skills, such as reasoning and instruction following. These latent skills are influenced by computational resources like model size and training tokens but with varying efficiencies across model families. Sloth exploits correlations across benchmarks to provide more accurate and interpretable predictions while alleviating the need to train multiple LLMs per family. We present both theoretical results on parameter identification and empirical evaluations on 12 prominent benchmarks, from Open LLM Leaderboard v1/v2, demonstrating that Sloth predicts LLM performance efficiently and offers insights into scaling behaviors for complex downstream tasks and increased test-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01991v2">Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)
    </div>
    <details class="paper-abstract">
      Nowadays, social media is pivotal in shaping public discourse, especially on polarizing issues like vaccination, where diverse moral perspectives influence individual opinions. In NLP, data scarcity and complexity of psycholinguistic tasks, such as identifying morality frames, make relying solely on human annotators costly, time-consuming, and prone to inconsistency due to cognitive load. To address these issues, we leverage large language models (LLMs), which are adept at adapting new tasks through few-shot learning, utilizing a handful of in-context examples coupled with explanations that connect examples to task principles. Our research explores LLMs' potential to assist human annotators in identifying morality frames within vaccination debates on social media. We employ a two-step process: generating concepts and explanations with LLMs, followed by human evaluation using a "think-aloud" tool. Our study shows that integrating LLMs into the annotation process enhances accuracy, reduces task difficulty, lowers cognitive load, suggesting a promising avenue for human-AI collaboration in complex psycholinguistic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02818v1">Accessible and Portable LLM Inference by Compiling Computational Graphs into SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) often demands specialized hardware, dedicated frameworks, and substantial development efforts, which restrict their accessibility, especially for edge devices and organizations with limited technical resources. We propose a novel compiler that translates LLM inference graphs into SQL queries, enabling relational databases, one of the most widely used and mature software systems globally, to serve as the runtime. By mapping neural operators such as matrix multiplication and attention into relational primitives like joins and aggregations, our approach leverages database capabilities, including disk-based data management and native caching. Supporting key transformer components, such as attention mechanisms and key-value caching, our system generates SQL pipelines for end-to-end LLM inference. Using the Llama3 family as a case study, we demonstrate up to 30x speedup in token generation for memory-constrained scenarios comparable to competitive CPU-based frameworks. Our work offers an accessible, portable, and efficient solution, facilitating the serving of LLMs across diverse deployment environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02810v1">Mol-LLM: Generalist Molecular LLM with Improved Graph Utilization</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have motivated the development of general LLMs for molecular tasks. While several studies have demonstrated that fine-tuned LLMs can achieve impressive benchmark performances, they are far from genuine generalist molecular LLMs due to a lack of fundamental understanding of molecular structure. Specifically, when given molecular task instructions, LLMs trained with naive next-token prediction training assign similar likelihood scores to both original and negatively corrupted molecules, revealing their lack of molecular structure understanding that is crucial for reliable and general molecular LLMs. To overcome this limitation and obtain a true generalist molecular LLM, we introduce a novel multi-modal training method based on a thorough multi-modal instruction tuning as well as a molecular structure preference optimization between chosen and rejected graphs. On various molecular benchmarks, the proposed generalist molecular LLM, called Mol-LLM, achieves state-of-the-art performances among generalist LLMs on most tasks, at the same time, surpassing or comparable to state-of-the-art specialist LLMs. Moreover, Mol-LLM also shows superior generalization performances in reaction prediction tasks, demonstrating the effect of the molecular structure understanding for generalization perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02794v1">METAMON: Finding Inconsistencies between Program Documentation and Behavior using Metamorphic LLM Queries</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 8 pages and 7 figures, accepted to LLM4Code 2025
    </div>
    <details class="paper-abstract">
      Code documentation can, if written precisely, help developers better understand the code they accompany. However, unlike code, code documentation cannot be automatically verified via execution, potentially leading to inconsistencies between documentation and the actual behavior. While such inconsistencies can be harmful for the developer's understanding of the code, checking and finding them remains a costly task due to the involvement of human engineers. This paper proposes METAMON, which uses an existing search-based test generation technique to capture the current program behavior in the form of test cases, and subsequently uses LLM-based code reasoning to identify the generated regression test oracles that are not consistent with the program specifications in the documentation. METAMON is supported in this task by metamorphic testing and self-consistency. An empirical evaluation against 9,482 pairs of code documentation and code snippets, generated using five open-source projects from Defects4J v2.0.1, shows that METAMON can classify the code-and-documentation inconsistencies with a precision of 0.72 and a recall of 0.48.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02790v1">Leveraging the true depth of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate remarkable capabilities at the cost of high compute requirements. While recent research has shown that intermediate layers can be removed or have their order shuffled without impacting performance significantly, these findings have not been employed to reduce the computational cost of inference. We investigate several potential ways to reduce the depth of pre-trained LLMs without significantly affecting performance. Leveraging our insights, we present a novel approach that exploits this decoupling between layers by grouping some of them into pairs that can be evaluated in parallel. This modification of the computational graph -- through better parallelism -- results in an average improvement of around 1.20x on the number of tokens generated per second, without re-training nor fine-tuning, while retaining 95%-99% of the original accuracy. Empirical evaluation demonstrates that this approach significantly improves serving efficiency while maintaining model performance, offering a practical improvement for large-scale LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15483v3">Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Post-training of pre-trained LLMs, which typically consists of the supervised fine-tuning (SFT) stage and the preference learning (RLHF or DPO) stage, is crucial to effective and safe LLM applications. The widely adopted approach in post-training popular open-source LLMs is to sequentially perform SFT and RLHF/DPO. However, sequential training is sub-optimal in terms of SFT and RLHF/DPO trade-off: the LLM gradually forgets about the first stage's training when undergoing the second stage's training. We theoretically prove the sub-optimality of sequential post-training. Furthermore, we propose a practical joint post-training framework with theoretical convergence guarantees and empirically outperforms sequential post-training framework, while having similar computational cost. Our code is available at https://github.com/heshandevaka/XRIGHT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03647v1">Looking for the Inner Music: Probing LLMs' Understanding of Literary Style</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated that language models can be trained to identify the author of much shorter literary passages than has been thought feasible for traditional stylometry. We replicate these results for authorship and extend them to a new dataset measuring novel genre. We find that LLMs are able to distinguish authorship and genre, but they do so in different ways. Some models seem to rely more on memorization, while others benefit more from training to learn author/genre characteristics. We then use three methods to probe one high-performing LLM for features that define style. These include direct syntactic ablations to input text as well as two methods that look at model internals. We find that authorial style is easier to define than genre-level style and is more impacted by minor syntactic decisions and contextual word usage. However, some traits like pronoun usage and word order prove significant for defining both kinds of literary style.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15676v3">Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been a widely adopted prompting method, eliciting impressive reasoning abilities of Large Language Models (LLMs). Inspired by the sequential thought structure of CoT, a number of Chain-of-X (CoX) methods have been developed to address various challenges across diverse domains and tasks involving LLMs. In this paper, we provide a comprehensive survey of Chain-of-X methods for LLMs in different contexts. Specifically, we categorize them by taxonomies of nodes, i.e., the X in CoX, and application tasks. We also discuss the findings and implications of existing CoX methods, as well as potential future directions. Our survey aims to serve as a detailed and up-to-date resource for researchers seeking to apply the idea of CoT to broader scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17406v2">ProveRAG: Provenance-Driven Vulnerability Analysis with Automated Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      In cybersecurity, security analysts face the challenge of mitigating newly discovered vulnerabilities in real-time, with over 300,000 Common Vulnerabilities and Exposures (CVEs) identified since 1999. The sheer volume of known vulnerabilities complicates the detection of patterns for unknown threats. While LLMs can assist, they often hallucinate and lack alignment with recent threats. Over 25,000 vulnerabilities have been identified so far in 2024, which are introduced after popular LLMs' (e.g., GPT-4) training data cutoff. This raises a major challenge of leveraging LLMs in cybersecurity, where accuracy and up-to-date information are paramount. In this work, we aim to improve the adaptation of LLMs in vulnerability analysis by mimicking how analysts perform such tasks. We propose ProveRAG, an LLM-powered system designed to assist in rapidly analyzing CVEs with automated retrieval augmentation of web data while self-evaluating its responses with verifiable evidence. ProveRAG incorporates a self-critique mechanism to help alleviate omission and hallucination common in the output of LLMs applied in cybersecurity applications. The system cross-references data from verifiable sources (NVD and CWE), giving analysts confidence in the actionable insights provided. Our results indicate that ProveRAG excels in delivering verifiable evidence to the user with over 99% and 97% accuracy in exploitation and mitigation strategies, respectively. This system outperforms direct prompting and chunking retrieval in vulnerability analysis by overcoming temporal and context-window limitations. ProveRAG guides analysts to secure their systems more effectively while documenting the process for future audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03604v1">Bilevel ZOFO: Bridging Parameter-Efficient and Zeroth-Order Techniques for Efficient LLM Fine-Tuning and Meta-Training</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained Large Language Models (LLMs) for downstream tasks using First-Order (FO) optimizers presents significant computational challenges. Parameter-Efficient Fine-Tuning(PEFT) methods have been proposed to address these challenges by freezing most model parameters and training only a small subset. While PEFT is efficient, it may not outperform full fine-tuning when high task-specific performance is required. Zeroth-Order (ZO) methods offer an alternative for fine-tuning the entire pre-trained model by approximating gradients using only the forward pass, thus eliminating the computational burden of back-propagation in first-order methods. However, when implementing ZO methods, a hard prompt is crucial, and relying on simple, fixed hard prompts may not be optimal. In this paper, we propose a bilevel optimization framework that complements ZO methods with PEFT to mitigate sensitivity to hard prompts while efficiently and effectively fine-tuning LLMs. Our Bilevel ZOFO (Zeroth-Order-First-Order) method employs a double-loop optimization strategy, where only the gradient of the PEFT model and the forward pass of the base model are required. We provide convergence guarantees for Bilevel ZOFO. Empirically, we demonstrate that Bilevel ZOFO outperforms both PEFT and ZO methods in single-task settings while maintaining similar memory efficiency. Additionally, we show its strong potential for multitask learning. Compared to current first-order meta-training algorithms for multitask learning, our method has significantly lower computational demands while maintaining or improving performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03589v1">HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Disaggregated Large Language Model (LLM) inference has gained popularity as it separates the computation-intensive prefill stage from the memory-intensive decode stage, avoiding the prefill-decode interference and improving resource utilization. However, transmitting Key-Value (KV) data between the two stages can be a bottleneck, especially for long prompts. Additionally, the computation time overhead for prefill and decode is key for optimizing Job Completion Time (JCT), and KV data size can become prohibitive for long prompts and sequences. Existing KV quantization methods can alleviate the transmission bottleneck and reduce memory requirements, but they introduce significant dequantization overhead, exacerbating the computation time. We propose Homomorphic Acceleration via Compression of the KV cache (HACK) for disaggregated LLM inference. HACK eliminates the heavy KV dequantization step, and directly performs computations on quantized KV data to approximate and reduce the cost of the expensive matrix-multiplication step. Extensive trace-driven experiments show that HACK reduces JCT by up to 70.9% compared to disaggregated LLM inference baseline and by up to 52.3% compared to state-of-the-art KV quantization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03579v1">A Mixed-Methods Evaluation of LLM-Based Chatbots for Menopause</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into healthcare settings has gained significant attention, particularly for question-answering tasks. Given the high-stakes nature of healthcare, it is essential to ensure that LLM-generated content is accurate and reliable to prevent adverse outcomes. However, the development of robust evaluation metrics and methodologies remains a matter of much debate. We examine the performance of publicly available LLM-based chatbots for menopause-related queries, using a mixed-methods approach to evaluate safety, consensus, objectivity, reproducibility, and explainability. Our findings highlight the promise and limitations of traditional evaluation metrics for sensitive health topics. We propose the need for customized and ethically grounded evaluation frameworks to assess LLMs to advance safe and effective use in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04390v1">In Praise of Stubbornness: The Case for Cognitive-Dissonance-Aware Knowledge Updates in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Despite remarkable capabilities, large language models (LLMs) struggle to continually update their knowledge without catastrophic forgetting. In contrast, humans effortlessly integrate new information, detect conflicts with existing beliefs, and selectively update their mental models. This paper introduces a cognitive-inspired investigation paradigm to study continual knowledge updating in LLMs. We implement two key components inspired by human cognition: (1) Dissonance and Familiarity Awareness, analyzing model behavior to classify information as novel, familiar, or dissonant; and (2) Targeted Network Updates, which track neural activity to identify frequently used (stubborn) and rarely used (plastic) neurons. Through carefully designed experiments in controlled settings, we uncover a number of empirical findings demonstrating the potential of this approach. First, dissonance detection is feasible using simple activation and gradient features, suggesting potential for cognitive-inspired training. Second, we find that non-dissonant updates largely preserve prior knowledge regardless of targeting strategy, revealing inherent robustness in LLM knowledge integration. Most critically, we discover that dissonant updates prove catastrophically destructive to the model's knowledge base, indiscriminately affecting even information unrelated to the current updates. This suggests fundamental limitations in how neural networks handle contradictions and motivates the need for new approaches to knowledge updating that better mirror human cognitive mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04387v1">FedP$^2$EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Federated learning (FL) has enabled the training of multilingual large language models (LLMs) on diverse and decentralized multilingual data, especially on low-resource languages. To improve client-specific performance, personalization via the use of parameter-efficient fine-tuning (PEFT) modules such as LoRA is common. This involves a personalization strategy (PS), such as the design of the PEFT adapter structures (e.g., in which layers to add LoRAs and what ranks) and choice of hyperparameters (e.g., learning rates) for fine-tuning. Instead of manual PS configuration, we propose FedP$^2$EFT, a federated learning-to-personalize method for multilingual LLMs in cross-device FL settings. Unlike most existing PEFT structure selection methods, which are prone to overfitting low-data regimes, FedP$^2$EFT collaboratively learns the optimal personalized PEFT structure for each client via Bayesian sparse rank selection. Evaluations on both simulated and real-world multilingual FL benchmarks demonstrate that FedP$^2$EFT largely outperforms existing personalized fine-tuning methods, while complementing a range of existing FL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04380v1">Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 26 pages, 15 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this paper, we study the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations for both inter- and intra-diversity. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-development for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04376v1">MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      In contemporary workplaces, meetings are essential for exchanging ideas and ensuring team alignment but often face challenges such as time consumption, scheduling conflicts, and inefficient participation. Recent advancements in Large Language Models (LLMs) have demonstrated their strong capabilities in natural language generation and reasoning, prompting the question: can LLMs effectively delegate participants in meetings? To explore this, we develop a prototype LLM-powered meeting delegate system and create a comprehensive benchmark using real meeting transcripts. Our evaluation reveals that GPT-4/4o maintain balanced performance between active and cautious engagement strategies. In contrast, Gemini 1.5 Pro tends to be more cautious, while Gemini 1.5 Flash and Llama3-8B/70B display more active tendencies. Overall, about 60\% of responses address at least one key point from the ground-truth. However, improvements are needed to reduce irrelevant or repetitive content and enhance tolerance for transcription errors commonly found in real-world settings. Additionally, we implement the system in practical settings and collect real-world feedback from demos. Our findings underscore the potential and challenges of utilizing LLMs as meeting delegates, offering valuable insights into their practical application for alleviating the burden of meetings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04362v1">LLMs can be easily Confused by Instructional Distractions</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Despite the fact that large language models (LLMs) show exceptional skill in instruction following tasks, this strength can turn into a vulnerability when the models are required to disregard certain instructions. Instruction-following tasks typically involve a clear task description and input text containing the target data to be processed. However, when the input itself resembles an instruction, confusion may arise, even if there is explicit prompting to distinguish between the task instruction and the input. We refer to this phenomenon as instructional distraction. In this paper, we introduce a novel benchmark, named DIM-Bench, specifically designed to assess LLMs' performance under instructional distraction. The benchmark categorizes real-world instances of instructional distraction and evaluates LLMs across four instruction tasks: rewriting, proofreading, translation, and style transfer -- alongside five input tasks: reasoning, code generation, mathematical reasoning, bias detection, and question answering. Our experimental results reveal that even the most advanced LLMs are susceptible to instructional distraction, often failing to accurately follow user intent in such cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05223v1">KDA: A Knowledge-Distilled Attacker for Generating Diverse Prompts to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Jailbreak attacks exploit specific prompts to bypass LLM safeguards, causing the LLM to generate harmful, inappropriate, and misaligned content. Current jailbreaking methods rely heavily on carefully designed system prompts and numerous queries to achieve a single successful attack, which is costly and impractical for large-scale red-teaming. To address this challenge, we propose to distill the knowledge of an ensemble of SOTA attackers into a single open-source model, called Knowledge-Distilled Attacker (KDA), which is finetuned to automatically generate coherent and diverse attack prompts without the need for meticulous system prompt engineering. Compared to existing attackers, KDA achieves higher attack success rates and greater cost-time efficiency when targeting multiple SOTA open-source and commercial black-box LLMs. Furthermore, we conducted a quantitative diversity analysis of prompts generated by baseline methods and KDA, identifying diverse and ensemble attacks as key factors behind KDA's effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03450v1">A Schema-Guided Reason-while-Retrieve framework for Reasoning on Scene Graphs with Large-Language-Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Scene graphs have emerged as a structured and serializable environment representation for grounded spatial reasoning with Large Language Models (LLMs). In this work, we propose SG-RwR, a Schema-Guided Retrieve-while-Reason framework for reasoning and planning with scene graphs. Our approach employs two cooperative, code-writing LLM agents: a (1) Reasoner for task planning and information queries generation, and a (2) Retriever for extracting corresponding graph information following the queries. Two agents collaborate iteratively, enabling sequential reasoning and adaptive attention to graph information. Unlike prior works, both agents are prompted only with the scene graph schema rather than the full graph data, which reduces the hallucination by limiting input tokens, and drives the Reasoner to generate reasoning trace abstractly.Following the trace, the Retriever programmatically query the scene graph data based on the schema understanding, allowing dynamic and global attention on the graph that enhances alignment between reasoning and retrieval. Through experiments in multiple simulation environments, we show that our framework surpasses existing LLM-based approaches in numerical Q\&A and planning tasks, and can benefit from task-level few-shot examples, even in the absence of agent-level demonstrations. Project code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03447v1">Designing LLM-simulated Immersive Spaces to Enhance Autistic Children's Social Affordances Understanding</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 iui2025
    </div>
    <details class="paper-abstract">
      One of the key challenges faced by autistic children is understanding social affordances in complex environments, which further impacts their ability to respond appropriately to social signals. In traffic scenarios, this impairment can even lead to safety concerns. In this paper, we introduce an LLM-simulated immersive projection environment designed to improve this ability in autistic children while ensuring their safety. We first propose 17 design considerations across four major categories, derived from a comprehensive review of previous research. Next, we developed a system called AIroad, which leverages LLMs to simulate drivers with varying social intents, expressed through explicit multimodal social signals. AIroad helps autistic children bridge the gap in recognizing the intentions behind behaviors and learning appropriate responses through various stimuli. A user study involving 14 participants demonstrated that this technology effectively engages autistic children and leads to significant improvements in their comprehension of social affordances in traffic scenarios. Additionally, parents reported high perceived usability of the system. These findings highlight the potential of combining LLM technology with immersive environments for the functional rehabilitation of autistic children in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v1">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating proof search spaces. While the existing approaches primarily rely on value functions and Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Search (BFS) remains underexplored. This paper investigates whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present \texttt{BFS-Prover}, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. \texttt{BFS-Prover} achieves a score of $71.31$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02542v2">OverThink: Slowdown Attacks on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 18x slowdown on FreshQA dataset and 46x slowdown on SQuAD dataset. The attack also shows high transferability across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third-party applications operating reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03373v1">Demystifying Long Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Scaling inference compute enhances reasoning in large language models (LLMs), with long chains-of-thought (CoTs) enabling strategies like backtracking and error correction. Reinforcement learning (RL) has emerged as a crucial method for developing these capabilities, yet the conditions under which long CoTs emerge remain unclear, and RL training requires careful design choices. In this study, we systematically investigate the mechanics of long CoT reasoning, identifying the key factors that enable models to generate long CoT trajectories. Through extensive supervised fine-tuning (SFT) and RL experiments, we present four main findings: (1) While SFT is not strictly necessary, it simplifies training and improves efficiency; (2) Reasoning capabilities tend to emerge with increased training compute, but their development is not guaranteed, making reward shaping crucial for stabilizing CoT length growth; (3) Scaling verifiable reward signals is critical for RL. We find that leveraging noisy, web-extracted solutions with filtering mechanisms shows strong potential, particularly for out-of-distribution (OOD) tasks such as STEM reasoning; and (4) Core abilities like error correction are inherently present in base models, but incentivizing these skills effectively for complex tasks via RL demands significant compute, and measuring their emergence requires a nuanced approach. These insights provide practical guidance for optimizing training strategies to enhance long CoT reasoning in LLMs. Our code is available at: https://github.com/eddycmu/demystify-long-cot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00326v8">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 19 pages, 12 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v1">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose \textbf{Di}vergence-driven \textbf{Z}eroth-\textbf{O}rder (\textbf{DiZO}) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v2">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      The "LLM-as-a-judge" paradigm employs Large Language Models (LLMs) as annotators and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure -- the Alternative Annotator Test (alt-test) -- that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming open-source LLMs, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15230v3">PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 32 pages, 7 figures, 24 tables
    </div>
    <details class="paper-abstract">
      Neural Networks can be effectively compressed through pruning, significantly reducing storage and compute demands while maintaining predictive performance. Simple yet effective methods like magnitude pruning remove less important parameters and typically require a costly retraining procedure to restore performance. However, with the rise of LLMs, full retraining has become infeasible due to memory and compute constraints. This study challenges the practice of retraining all parameters by showing that updating a small subset of highly expressive parameters can suffice to recover or even enhance performance after pruning. Surprisingly, retraining just 0.01%-0.05% of the parameters in GPT-architectures can match the performance of full retraining across various sparsity levels, significantly reducing compute and memory requirements, and enabling retraining of models with up to 30 billion parameters on a single GPU in minutes. To bridge the gap to full retraining in the high sparsity regime, we introduce two novel LoRA variants that, unlike standard LoRA, allow merging adapters back without compromising sparsity. Going a step further, we show that these methods can be applied for memory-efficient layer-wise reconstruction, significantly enhancing state-of-the-art retraining-free methods like Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023). Our findings present a promising alternative to avoiding retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03159v1">PICBench: Benchmarking LLMs for Photonic Integrated Circuits Design</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable potential in automating various tasks in digital chip design, the field of Photonic Integrated Circuits (PICs)-a promising solution to advanced chip designs-remains relatively unexplored in this context. The design of PICs is time-consuming and prone to errors due to the extensive and repetitive nature of code involved in photonic chip design. In this paper, we introduce PICBench, the first benchmarking and evaluation framework specifically designed to automate PIC design generation using LLMs, where the generated output takes the form of a netlist. Our benchmark consists of dozens of meticulously crafted PIC design problems, spanning from fundamental device designs to more complex circuit-level designs. It automatically evaluates both the syntax and functionality of generated PIC designs by comparing simulation outputs with expert-written solutions, leveraging an open-source simulator. We evaluate a range of existing LLMs, while also conducting comparative tests on various prompt engineering techniques to enhance LLM performance in automated PIC design. The results reveal the challenges and potential of LLMs in the PIC design domain, offering insights into the key areas that require further research and development to optimize automation in this field. Our benchmark and evaluation code is available at https://github.com/PICDA/PICBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03080v1">IAO Prompting: Making Knowledge Flow Explicit in LLMs through Structured Reasoning Templates</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted as Oral at KnowFM @ AAAI 2025
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, understanding and validating their knowledge utilization remains challenging. Chain-of-thought (CoT) prompting partially addresses this by revealing intermediate reasoning steps, but the knowledge flow and application remain implicit. We introduce IAO (Input-Action-Output) prompting, a structured template-based method that explicitly models how LLMs access and apply their knowledge during complex reasoning tasks. IAO decomposes problems into sequential steps, each clearly identifying the input knowledge being used, the action being performed, and the resulting output. This structured decomposition enables us to trace knowledge flow, verify factual consistency, and identify potential knowledge gaps or misapplications. Through experiments across diverse reasoning tasks, we demonstrate that IAO not only improves zero-shot performance but also provides transparency in how LLMs leverage their stored knowledge. Human evaluation confirms that this structured approach enhances our ability to verify knowledge utilization and detect potential hallucinations or reasoning errors. Our findings provide insights into both knowledge representation within LLMs and methods for more reliable knowledge application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05498v3">SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted by USENIX Security Symposium 2025. Please cite the conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner. In Proc. USENIX Security, 2025."
    </div>
    <details class="paper-abstract">
      Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02988v1">Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 accepted at WWW'25 (Industrial Track), extended version
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for their adoption as evaluative judges. This paper introduces Themis, a fine-tuned LLM judge that delivers sophisticated context-aware evaluations. We provide a comprehensive overview of the development pipeline for Themis, highlighting its scenario-dependent evaluation prompts and two novel methods for controlled instruction generation. These designs enable Themis to effectively distill evaluative skills from teacher models, while retaining flexibility for continuous development. We introduce two human-labeled benchmarks for meta-evaluation, demonstrating that Themis can achieve high alignment with human preferences in an economical manner. Additionally, we explore insights into the LLM-as-a-judge paradigm, revealing nuances in performance and the varied effects of reference answers. Notably, we observe that pure knowledge distillation from strong LLMs, though common, does not guarantee performance improvement through scaling. We propose a mitigation strategy based on instruction-following difficulty. Furthermore, we provide practical guidelines covering data balancing, prompt customization, multi-objective training, and metric aggregation. We aim for our method and findings, along with the fine-tuning data, benchmarks, and model checkpoints, to support future research and development in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v2">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14304v2">MASTER: A Multi-Agent System with LLM Specialized MCTS</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted by main NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) are increasingly being explored for problem-solving tasks. However, their strategic planning capability is often viewed with skepticism. Recent studies have incorporated the Monte Carlo Tree Search (MCTS) algorithm to augment the planning capacity of LLM. Despite its potential, MCTS relies on extensive sampling simulations to approximate the true reward distribution, which leads to two primary issues. Firstly, MCTS is effective for tasks like the Game of Go, where simulation results can yield objective rewards (e.g., 1 for a win and 0 for a loss). However, for tasks such as question answering, the result of a simulation is the answer to the question, which cannot yield an objective reward without the ground truth. Secondly, obtaining statistically significant reward estimations typically requires a sample size exceeding 30 simulations, resulting in excessive token usage and time consumption. To address these challenges, we present the Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS (MASTER), a novel framework that coordinates agent recruitment and communication through LLM specialized MCTS. This system autonomously adjusts the number of agents based on task complexity and ensures focused communication among them. Comprehensive experiments across various tasks demonstrate the effectiveness of our proposed framework. It achieves 76% accuracy on HotpotQA and 80% on WebShop, setting new state-of-the-art performance on these datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v2">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 32 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving simple scientific problems but, even with domain-specific fine-tuning, often produce hallucinations for complex ones. While integrating LLMs with tools can mitigate this reliability issue, models finetuned on tool usage only often over-rely on them, incurring unnecessary costs from resource-intensive scientific tools even for simpler problems. Inspired by how human experts assess the complexity of the problem before choosing the solutions, we propose a novel two-component fine-tuning method, Adapting While Learning (AWL). In the first component, World Knowledge Learning (WKL), LLMs internalize scientific knowledge by learning from tools-generated solutions. In the second component, Tool Usage Adaptation (TUA), we classify questions as easy or hard based on the WKL-trained model's accuracy, and train it to maintain direct reasoning for simple problems while switching to tools for challenging ones. We validate our method on 6 scientific benchmark datasets in climate science, epidemiology, and mathematics. Compared to the base 8B model, our trained models achieve 28.27% higher answer accuracy and 13.76% better tool usage accuracy, even surpassing state-of-the-art models including GPT-4 and Claude-3.5 on 4 custom-created datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02009v1">LLMSecConfig: An LLM-Based Approach for Fixing Software Container Misconfigurations</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Security misconfigurations in Container Orchestrators (COs) can pose serious threats to software systems. While Static Analysis Tools (SATs) can effectively detect these security vulnerabilities, the industry currently lacks automated solutions capable of fixing these misconfigurations. The emergence of Large Language Models (LLMs), with their proven capabilities in code understanding and generation, presents an opportunity to address this limitation. This study introduces LLMSecConfig, an innovative framework that bridges this gap by combining SATs with LLMs. Our approach leverages advanced prompting techniques and Retrieval-Augmented Generation (RAG) to automatically repair security misconfigurations while preserving operational functionality. Evaluation of 1,000 real-world Kubernetes configurations achieved a 94\% success rate while maintaining a low rate of introducing new misconfigurations. Our work makes a promising step towards automated container security management, reducing the manual effort required for configuration maintenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01992v1">FinRLlama: A Solution to LLM-Engineered Signals Challenge at FinRL Contest 2024</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Competition Track FinRL, ICAIF 2024
    </div>
    <details class="paper-abstract">
      In response to Task II of the FinRL Challenge at ACM ICAIF 2024, this study proposes a novel prompt framework for fine-tuning large language models (LLM) with Reinforcement Learning from Market Feedback (RLMF). Our framework incorporates market-specific features and short-term price dynamics to generate more precise trading signals. Traditional LLMs, while competent in sentiment analysis, lack contextual alignment for financial market applications. To bridge this gap, we fine-tune the LLaMA-3.2-3B-Instruct model using a custom RLMF prompt design that integrates historical market data and reward-based feedback. Our evaluation shows that this RLMF-tuned framework outperforms baseline methods in signal consistency and achieving tighter trading outcomes; awarded as winner of Task II. You can find the code for this project on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01991v1">Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)
    </div>
    <details class="paper-abstract">
      Nowadays, social media is pivotal in shaping public discourse, especially on polarizing issues like vaccination, where diverse moral perspectives influence individual opinions. In NLP, data scarcity and complexity of psycholinguistic tasks such as identifying morality frames makes relying solely on human annotators costly, time-consuming, and prone to inconsistency due to cognitive load. To address these issues, we leverage large language models (LLMs), which are adept at adapting new tasks through few-shot learning, utilizing a handful of in-context examples coupled with explanations that connect examples to task principles. Our research explores LLMs' potential to assist human annotators in identifying morality frames within vaccination debates on social media. We employ a two-step process: generating concepts and explanations with LLMs, followed by human evaluation using a "think-aloud" tool. Our study shows that integrating LLMs into the annotation process enhances accuracy, reduces task difficulty, lowers cognitive load, suggesting a promising avenue for human-AI collaboration in complex psycholinguistic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17017v3">Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Self-Consistency mitigates hallucinations in Large Language Models (LLMs) by sampling multiple reasoning paths,but it lacks a systematic approach to determine the optimal number of samples or select the most faithful rationale. To address this limitation, we introduce Reasoning-Aware Self-Consistency (RASC), a novel framework that enhances sampling efficiency and reasoning faithfulness by dynamically evaluating both outputs and rationales. RASC assesses the quality of reasoning and the consistency of answers for each generated sample, using these assessments to guide early stopping decisions and rationale selection. The framework employs criteria-based stopping and weighted majority voting, enabling more informed choices on when to halt sampling and which rationale to select. Our comprehensive experiments across diverse question-answering datasets demonstrate that RASC outperforms existing methods, reducing sample usage by approximately 70% while maintaining accuracy. Moreover, RASC facilitates the selection of high-fidelity rationales, thereby improving the faithfulness of LLM outputs. Our approach effectively addresses the efficiency-accuracy trade-off in LLM reasoning tasks, offering a new perspective for more nuanced, faithful, and effective utilization of LLMs in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01066v2">From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 15 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      LLMs when used with Retrieval Augmented Generation (RAG), are greatly improving the SOTA of translating natural language queries to structured and correct SQL. Unlike previous reviews, this survey provides a comprehensive study of the evolution of LLM-based text-to-SQL systems, from early rule-based models to advanced LLM approaches that use (RAG) systems. We discuss benchmarks, evaluation methods, and evaluation metrics. Also, we uniquely study the use of Graph RAGs for better contextual accuracy and schema linking in these systems. Finally, we highlight key challenges such as computational efficiency, model robustness, and data privacy toward improvements of LLM-based text-to-SQL systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01977v1">AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      User interface understanding with vision-language models has received much attention due to its potential for enabling next-generation software automation. However, existing UI datasets either only provide large-scale context-free element annotations or contextualized functional descriptions for elements at a much smaller scale. In this work, we propose the \methodname{} pipeline for automatically annotating UI elements with detailed functionality descriptions at scale. Specifically, we leverage large language models (LLMs) to infer element functionality by comparing the UI content changes before and after simulated interactions with specific UI elements. To improve annotation quality, we propose LLM-aided rejection and verification, eliminating invalid and incorrect annotations without human labor. We construct an \methodname{}-704k dataset using the proposed pipeline, featuring multi-resolution, multi-device screenshots, diverse data domains, and detailed functionality annotations that have never been provided by previous datasets. Human evaluation shows that the AutoGUI pipeline achieves annotation correctness comparable to trained human annotators. Extensive experimental results show that our \methodname{}-704k dataset remarkably enhances VLM's UI grounding capabilities, exhibits significant scaling effects, and outperforms existing web pre-training data types. We envision AutoGUI as a scalable pipeline for generating massive data to build GUI-oriented VLMs. AutoGUI dataset can be viewed at this anonymous URL: https://autogui-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01968v1">Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant or uninformative. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves performance across multiple downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.00948v3">Large Linguistic Models: Investigating LLMs' metalinguistic abilities</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) has recently improved to the point where the models can perform well on many language tasks. We show here that -- for the first time -- the models can also generate valid metalinguistic analyses of language data. We outline a research program where the behavioral interpretability of LLMs on these tasks is tested via prompting. LLMs are trained primarily on text -- as such, evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. We show that OpenAI's o1 vastly outperforms other models on tasks involving drawing syntactic trees and phonological generalization. We speculate that OpenAI o1's unique advantage over other models may result from the model's chain-of-thought mechanism, which mimics the structure of human reasoning used in complex cognitive tasks, such as linguistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01941v1">Can LLMs Maintain Fundamental Abilities under KV Cache Compression?</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      This paper investigates an under-explored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. While existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive empirical study evaluating prominent KV cache compression methods across diverse tasks, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation.Our analysis reveals that KV cache compression methods exhibit task-specific performance degradation. Arithmetic reasoning tasks prove particularly sensitive to aggressive compression, with different methods showing performance drops of $17.4\%$-$43.3\%$. Notably, the DeepSeek R1 Distill model exhibits more robust compression tolerance compared to instruction-tuned models, showing only $9.67\%$-$25.53\%$ performance degradation. Based on our analysis of attention patterns and cross-task compression performance, we propose ShotKV, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09615v2">SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Conventional model compression techniques for LLMs address high memory consumption and slow inference challenges but typically require computationally expensive retraining to preserve accuracy. In contrast, one-shot compression methods eliminate retraining cost, but struggle to achieve accuracy comparable to dense models. This paper presents SLIM, a new one-shot compression framework that holistically integrates hardware-friendly quantization, sparsity, and low-rank approximation into a unified process. First, we formulate the quantization process using a probabilistic approach (SLIM-Quant) that enables us to apply uniform quantization. Then, we use an existing one-shot pruning method to apply semi-structured sparsity on top of the quantized weights. Finally, to compensate for the introduced aggregated quantization and sparsity error, we use a novel saliency function with unique invertible and additive features that enables us to mathematically compute the value of low-rank adapters. SLIM improves model accuracy by up to 5.66% (LLaMA-2-7B) for 2:4 sparsity with 4-bit weight quantization, outperforming prior methods. Models compressed with SLIM achieve up to 3.78x and 3.75x layer-wise speedup on Nvidia RTX3060 and A100 GPUs, respectively. We also propose an optional PEFT recipe that further improves accuracy by up to 1.66% (LLaMA-2-13B) compared to SLIM without fine-tuning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02773v1">SD++: Enhancing Standard Definition Maps by Incorporating Road Knowledge using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      High-definition maps (HD maps) are detailed and informative maps capturing lane centerlines and road elements. Although very useful for autonomous driving, HD maps are costly to build and maintain. Furthermore, access to these high-quality maps is usually limited to the firms that build them. On the other hand, standard definition (SD) maps provide road centerlines with an accuracy of a few meters. In this paper, we explore the possibility of enhancing SD maps by incorporating information from road manuals using LLMs. We develop SD++, an end-to-end pipeline to enhance SD maps with location-dependent road information obtained from a road manual. We suggest and compare several ways of using LLMs for such a task. Furthermore, we show the generalization ability of SD++ by showing results from both California and Japan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05534v2">Can LLMs Replace Manual Annotation of Software Engineering Artifacts?</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Experimental evaluations of software engineering innovations, e.g., tools and processes, often include human-subject studies as a component of a multi-pronged strategy to obtain greater generalizability of the findings. However, human-subject studies in our field are challenging, due to the cost and difficulty of finding and employing suitable subjects, ideally, professional programmers with varying degrees of experience. Meanwhile, large language models (LLMs) have recently started to demonstrate human-level performance in several areas. This paper explores the possibility of substituting costly human subjects with much cheaper LLM queries in evaluations of code and code-related artifacts. We study this idea by applying six state-of-the-art LLMs to ten annotation tasks from five datasets created by prior work, such as judging the accuracy of a natural language summary of a method or deciding whether a code change fixes a static analysis warning. Our results show that replacing some human annotation effort with LLMs can produce inter-rater agreements equal or close to human-rater agreement. To help decide when and how to use LLMs in human-subject studies, we propose model-model agreement as a predictor of whether a given task is suitable for LLMs at all, and model confidence as a means to select specific samples where LLMs can safely replace human annotators. Overall, our work is the first step toward mixed human-LLM evaluations in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02743v1">LLM Bandit: Cost-Efficient LLM Generation via Preference-Conditioned Dynamic Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The rapid advancement in large language models (LLMs) has brought forth a diverse range of models with varying capabilities that excel in different tasks and domains. However, selecting the optimal LLM for user queries often involves a challenging trade-off between accuracy and cost, a problem exacerbated by the diverse demands of individual queries. In this work, we present a novel framework that formulates the LLM selection process as a multi-armed bandit problem, enabling dynamic and intelligent routing of queries to the most appropriate model. Our approach incorporates a preference-conditioned dynamic routing mechanism, allowing users to specify their preferences at inference time, thereby offering a customizable balance between performance and cost. Additionally, our selection policy is designed to generalize to unseen LLMs, ensuring adaptability to new models as they emerge. Experimental results demonstrate that our method achieves significant improvements in both accuracy and cost-effectiveness across various LLM platforms, showcasing the potential of our framework to adaptively optimize LLM selection in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13284v2">Learning to Route LLMs with Confidence Tokens</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance on several tasks and are increasingly deployed in real-world applications. However, especially in high-stakes settings, it becomes vital to know when the output of an LLM may be unreliable. Depending on whether an answer is trustworthy, a system can then choose to route the question to another expert, or otherwise fall back on a safe default behavior. In this work, we study the extent to which LLMs can reliably indicate confidence in their answers, and how this notion of confidence can translate into downstream accuracy gains. We propose Self-REF, a lightweight training strategy to teach LLMs to express confidence in whether their answers are correct in a reliable manner. Self-REF introduces confidence tokens into the LLM, from which a confidence score can be extracted. Compared to conventional approaches such as verbalizing confidence and examining token probabilities, we demonstrate empirically that confidence tokens show significant improvements in downstream routing and rejection learning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02723v1">Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      We provide a new LLM-compression solution via SVD, unlocking new possibilities for LLM compression beyond quantization and pruning. We point out that the optimal use of SVD lies in truncating activations, rather than merely using activations as an optimization distance. Building on this principle, we address three critical challenges in SVD-based LLM compression: including (1) How can we determine the optimal activation truncation position for each weight matrix in LLMs? (2) How can we efficiently reconstruct the weight matrices based on truncated activations? (3) How can we address the inherent "injection" nature that results in the information loss of the SVD? We propose Dobi-SVD, which establishes a new, principled approach to SVD-based LLM compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02715v1">An Analysis of LLM Fine-Tuning and Few-Shot Learning for Flaky Test Detection and Classification</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Flaky tests exhibit non-deterministic behavior during execution and they may pass or fail without any changes to the program under test. Detecting and classifying these flaky tests is crucial for maintaining the robustness of automated test suites and ensuring the overall reliability and confidence in the testing. However, flaky test detection and classification is challenging due to the variability in test behavior, which can depend on environmental conditions and subtle code interactions. Large Language Models (LLMs) offer promising approaches to address this challenge, with fine-tuning and few-shot learning (FSL) emerging as viable techniques. With enough data fine-tuning a pre-trained LLM can achieve high accuracy, making it suitable for organizations with more resources. Alternatively, we introduce FlakyXbert, an FSL approach that employs a Siamese network architecture to train efficiently with limited data. To understand the performance and cost differences between these two methods, we compare fine-tuning on larger datasets with FSL in scenarios restricted by smaller datasets. Our evaluation involves two existing flaky test datasets, FlakyCat and IDoFT. Our results suggest that while fine-tuning can achieve high accuracy, FSL provides a cost-effective approach with competitive accuracy, which is especially beneficial for organizations or projects with limited historical data available for training. These findings underscore the viability of both fine-tuning and FSL in flaky test detection and classification with each suited to different organizational needs and resource availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02705v4">Certifying LLM Safety against Adversarial Prompting</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08854v3">Hybrid LLM-DDQN based Joint Optimization of V2I Communication and Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted by IEEE Wireless Communications Letters
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have received considerable interest recently due to their outstanding reasoning and comprehension capabilities. This work explores applying LLMs to vehicular networks, aiming to jointly optimize vehicle-to-infrastructure (V2I) communications and autonomous driving (AD) policies. We deploy LLMs for AD decision-making to maximize traffic flow and avoid collisions for road safety, and a double deep Q-learning algorithm (DDQN) is used for V2I optimization to maximize the received data rate and reduce frequent handovers. In particular, for LLM-enabled AD, we employ the Euclidean distance to identify previously explored AD experiences, and then LLMs can learn from past good and bad decisions for further improvement. Then, LLM-based AD decisions will become part of states in V2I problems, and DDQN will optimize the V2I decisions accordingly. After that, the AD and V2I decisions are iteratively optimized until convergence. Such an iterative optimization approach can better explore the interactions between LLMs and conventional reinforcement learning techniques, revealing the potential of using LLMs for network optimization and management. Finally, the simulations demonstrate that our proposed hybrid LLM-DDQN approach outperforms the conventional DDQN algorithm, showing faster convergence and higher average rewards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02675v1">Exploring LLMs Impact on Student-Created User Stories and Acceptance Testing in Software Development</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 3m pages
    </div>
    <details class="paper-abstract">
      In Agile software development methodology, a user story describes a new feature or functionality from an end user's perspective. The user story details may also incorporate acceptance testing criteria, which can be developed through negotiation with users. When creating stories from user feedback, the software engineer may maximize their usefulness by considering story attributes, including scope, independence, negotiability, and testability. This study investigates how LLMs (large language models), with guided instructions, affect undergraduate software engineering students' ability to transform user feedback into user stories. Students, working individually, were asked to analyze user feedback comments, appropriately group related items, and create user stories following the principles of INVEST, a framework for assessing user stories. We found that LLMs help students develop valuable stories with well-defined acceptance criteria. However, students tend to perform better without LLMs when creating user stories with an appropriate scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02659v1">A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation (GALI)</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 9 pages, under review in the conference
    </div>
    <details class="paper-abstract">
      Transformer-based Large Language Models (LLMs) struggle to process inputs exceeding their training context window, with performance degrading due to positional out-of-distribution (O.O.D.) that disrupt attention computations. Existing solutions, fine-tuning and training-free methods, are limited by computational inefficiency, attention logit outliers or loss of local positional information. To address this, we propose Greedy Attention Logit Interpolation (GALI), a training-free length extrapolation method that maximizes the utilization of pretrained positional intervals while avoiding attention logit outliers through attention logit interpolation. The result demonstrates that GALI consistently outperforms state-of-the-art training-free methods. Our findings reveal that LLMs interpret positional intervals unevenly within their training context window, suggesting that extrapolating within a smaller positional interval range yields superior results-even for short-context tasks. GALI represents a significant step toward resolving the positional O.O.D. challenge, enabling more reliable long-text understanding in LLMs. Our implementation of GALI, along with the experiments from our paper, is open-sourced at https://github.com/AcademyCityL/GALI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04358v1">Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 12 pages including references
    </div>
    <details class="paper-abstract">
      Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04351v1">NER4all or Context is All You Need: Using LLMs for low-effort, high-performance NER on historical texts. A humanities informed approach</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Named entity recognition (NER) is a core task for historical research in automatically establishing all references to people, places, events and the like. Yet, do to the high linguistic and genre diversity of sources, only limited canonisation of spellings, the level of required historical domain knowledge, and the scarcity of annotated training data, established approaches to natural language processing (NLP) have been both extremely expensive and yielded only unsatisfactory results in terms of recall and precision. Our paper introduces a new approach. We demonstrate how readily-available, state-of-the-art LLMs significantly outperform two leading NLP frameworks, spaCy and flair, for NER in historical documents by seven to twentytwo percent higher F1-Scores. Our ablation study shows how providing historical context to the task and a bit of persona modelling that turns focus away from a purely linguistic approach are core to a successful prompting strategy. We also demonstrate that, contrary to our expectations, providing increasing numbers of examples in few-shot approaches does not improve recall or precision below a threshold of 16-shot. In consequence, our approach democratises access to NER for all historians by removing the barrier of scripting languages and computational skills required for established NLP tools and instead leveraging natural language prompts and consumer-grade tools and frontends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04349v1">Dynamic benchmarking framework for LLM-based conversational data capture</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has transformed conversational agents, enabling complex human-machine interactions. However, evaluation frameworks often focus on single tasks, failing to capture the dynamic nature of multi-turn dialogues. This paper introduces a dynamic benchmarking framework to assess LLM-based conversational agents through interactions with synthetic users. The framework integrates generative agent simulation to evaluate performance on key dimensions: information extraction, context awareness, and adaptive engagement. By simulating various aspects of user behavior, our work provides a scalable, automated, and flexible benchmarking approach. Experimental evaluation - within a loan application use case - demonstrates the framework's effectiveness under one-shot and few-shot extraction conditions. Results show that adaptive strategies improve data extraction accuracy, especially when handling ambiguous responses. Future work will extend its applicability to broader domains and incorporate additional metrics (e.g., conversational coherence, user engagement). This study contributes a structured, scalable approach to evaluating LLM-based conversational agents, facilitating real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04347v1">SCALM: Detecting Bad Practices in Smart Contracts Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      As the Ethereum platform continues to mature and gain widespread usage, it is crucial to maintain high standards of smart contract writing practices. While bad practices in smart contracts may not directly lead to security issues, they do elevate the risk of encountering problems. Therefore, to understand and avoid these bad practices, this paper introduces the first systematic study of bad practices in smart contracts, delving into over 35 specific issues. Specifically, we propose a large language models (LLMs)-based framework, SCALM. It combines Step-Back Prompting and Retrieval-Augmented Generation (RAG) to identify and address various bad practices effectively. Our extensive experiments using multiple LLMs and datasets have shown that SCALM outperforms existing tools in detecting bad practices in smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04346v1">Multi-Lingual Cyber Threat Detection in Tweets/X Using ML, DL, and LLM: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Cyber threat detection has become an important area of focus in today's digital age due to the growing spread of fake information and harmful content on social media platforms such as Twitter (now 'X'). These cyber threats, often disguised within tweets, pose significant risks to individuals, communities, and even nations, emphasizing the need for effective detection systems. While previous research has explored tweet-based threats, much of the work is limited to specific languages, domains, or locations, or relies on single-model approaches, reducing their applicability to diverse real-world scenarios. To address these gaps, our study focuses on multi-lingual tweet cyber threat detection using a variety of advanced models. The research was conducted in three stages: (1) We collected and labeled tweet datasets in four languages English, Chinese, Russian, and Arabic employing both manual and polarity-based labeling methods to ensure high-quality annotations. (2) Each dataset was analyzed individually using machine learning (ML) and deep learning (DL) models to assess their performance on distinct languages. (3) Finally, we combined all four datasets into a single multi-lingual dataset and applied DL and large language model (LLM) architectures to evaluate their efficacy in identifying cyber threats across various languages. Our results show that among machine learning models, Random Forest (RF) attained the highest performance; however, the Bi-LSTM architecture consistently surpassed other DL and LLM architectures across all datasets. These findings underline the effectiveness of Bi-LSTM in multilingual cyber threat detection. The code for this paper can be found at this link: https://github.com/Mmurrad/Tweet-Data-Classification.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02542v1">OVERTHINKING: Slowdown Attacks on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 46x slowdown and high transferability of the attack across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third party applications operating reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02539v1">LLMs for Generation of Architectural Components: An Exploratory Empirical Study in the Serverless World</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to IEEE International Conference on Software Architecture (ICSA) 2025 Main Track (https://conf.researchr.org/home/icsa-2025)
    </div>
    <details class="paper-abstract">
      Recently, the exponential growth in capability and pervasiveness of Large Language Models (LLMs) has led to significant work done in the field of code generation. However, this generation has been limited to code snippets. Going one step further, our desideratum is to automatically generate architectural components. This would not only speed up development time, but would also enable us to eventually completely skip the development phase, moving directly from design decisions to deployment. To this end, we conduct an exploratory study on the capability of LLMs to generate architectural components for Functions as a Service (FaaS), commonly known as serverless functions. The small size of their architectural components make this architectural style amenable for generation using current LLMs compared to other styles like monoliths and microservices. We perform the study by systematically selecting open source serverless repositories, masking a serverless function and utilizing state of the art LLMs provided with varying levels of context information about the overall system to generate the masked function. We evaluate correctness through existing tests present in the repositories and use metrics from the Software Engineering (SE) and Natural Language Processing (NLP) domains to evaluate code quality and the degree of similarity between human and LLM generated code respectively. Along with our findings, we also present a discussion on the path forward for using GenAI in architectural component generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02534v1">Adaptive Self-improvement LLM Agentic System for ML Library Development</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      ML libraries, often written in architecture-specific programming languages (ASPLs) that target domain-specific architectures, are key to efficient ML systems. However, writing these high-performance ML libraries is challenging because it requires expert knowledge of ML algorithms and the ASPL. Large language models (LLMs), on the other hand, have shown general coding capabilities. However, challenges remain when using LLMs for generating ML libraries using ASPLs because 1) this task is complicated even for experienced human programmers and 2) there are limited code examples because of the esoteric and evolving nature of ASPLs. Therefore, LLMs need complex reasoning with limited data in order to complete this task. To address these challenges, we introduce an adaptive self-improvement agentic system. In order to evaluate the effectiveness of our system, we construct a benchmark of a typical ML library and generate ASPL code with both open and closed-source LLMs on this benchmark. Our results show improvements of up to $3.9\times$ over a baseline single LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02508v1">Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models will be fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18626v3">The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies. Warning: this paper contains examples of unethical inquiries used solely for research purposes.
    </details>
</div>
