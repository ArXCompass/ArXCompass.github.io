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
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08661v2">Few-shot LLM Synthetic Data with Distribution Matching</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 10 pages, 5 figures, accepted at www 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) advance, their ability to perform in-context learning and few-shot language generation has improved significantly. This has spurred using LLMs to produce high-quality synthetic data to enhance the performance of smaller models like online retrievers or weak LLMs. However, LLM-generated synthetic data often differs from the real data in key language attributes (e.g., styles, tones, content proportions, etc.). As a result, mixing these synthetic data directly with real data may distort the original data distribution, potentially hindering performance improvements. To solve this, we introduce SynAlign: a synthetic data generation and filtering framework based on key attribute distribution matching. Before generation, SynAlign employs an uncertainty tracker surrogated by the Gaussian Process model to iteratively select data clusters distinct from selected ones as demonstrations for new data synthesis, facilitating the efficient exploration diversity of the real data. Then, a latent attribute reasoning method is employed: the LLM summarizes linguistic attributes of demonstrations and then synthesizes new data based on them. This approach facilitates synthesizing diverse data with linguistic attributes that appear in real data.After generation, the Maximum Mean Discrepancy is used as the objective function to learn the sampling weight of each synthetic data, ensuring distribution matching with the real data. Our experiments on multiple text prediction tasks show significant performance improvements. We also conducted an online A/B test on an online retriever to demonstrate SynAlign's effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06326v4">Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle to provide up-to-date information due to their one-time training and the constantly evolving nature of the world. To keep LLMs current, existing approaches typically involve continued pre-training on new documents. However, they frequently face difficulties in extracting stored knowledge. Motivated by the remarkable success of the Feynman Technique in efficient human learning, we introduce Self-Tuning, a learning framework aimed at improving an LLM's ability to effectively acquire new knowledge from unseen raw documents through self-teaching. Specifically, we develop a Self-Teaching strategy that augments the documents with a set of knowledge-intensive tasks created in a self-supervised manner, focusing on three crucial aspects: memorization, comprehension, and self-reflection. Additionally, we introduce three Wiki-Newpages-2023-QA datasets to facilitate an in-depth analysis of an LLM's knowledge acquisition ability concerning memorization, extraction, and reasoning. Extensive experimental results on various models, e.g., Llama2-7B reveal that Self-Tuning consistently exhibits superior performance across all knowledge acquisition tasks and excels in preserving previous knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10638v1">Script&Shift: A Layered Interface Paradigm for Integrating Content Development and Rhetorical Strategy with LLM Writing Assistants</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Good writing is a dynamic process of knowledge transformation, where writers refine and evolve ideas through planning, translating, and reviewing. Generative AI-powered writing tools can enhance this process but may also disrupt the natural flow of writing, such as when using LLMs for complex tasks like restructuring content across different sections or creating smooth transitions. We introduce Script&Shift, a layered interface paradigm designed to minimize these disruptions by aligning writing intents with LLM capabilities to support diverse content development and rhetorical strategies. By bridging envisioning, semantic, and articulatory distances, Script&Shift's interactions allow writers to leverage LLMs for various content development tasks (scripting) and experiment with diverse organization strategies while tailoring their writing for different audiences (shifting). This approach preserves creative control while encouraging divergent and iterative writing. Our evaluation shows that Script&Shift enables writers to creatively and efficiently incorporate LLMs while preserving a natural flow of composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06394v5">GameArena: Evaluating LLM Reasoning through Live Computer Games</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Evaluating the reasoning abilities of large language models (LLMs) is challenging. Existing benchmarks often depend on static datasets, which are vulnerable to data contamination and may get saturated over time, or on binary live human feedback that conflates reasoning with other abilities. As the most prominent dynamic benchmark, Chatbot Arena evaluates open-ended questions in real-world settings, but lacks the granularity in assessing specific reasoning capabilities. We introduce GameArena, a dynamic benchmark designed to evaluate LLM reasoning capabilities through interactive gameplay with humans. GameArena consists of three games designed to test specific reasoning capabilities (e.g., deductive and inductive reasoning), while keeping participants entertained and engaged. We analyze the gaming data retrospectively to uncover the underlying reasoning processes of LLMs and measure their fine-grained reasoning capabilities. We collect over 2000 game sessions and provide detailed assessments of various reasoning capabilities for five state-of-the-art LLMs. Our user study with 100 participants suggests that GameArena improves user engagement compared to Chatbot Arena. For the first time, GameArena enables the collection of step-by-step LLM reasoning data in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20299v2">EACO-RAG: Towards Distributed Tiered LLM Deployment using Edge-Assisted and Collaborative RAG with Adaptive Knowledge Update</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in language tasks, but they require high computing power and rely on static knowledge. To overcome these limitations, Retrieval-Augmented Generation (RAG) incorporates up-to-date external information into LLMs without extensive fine-tuning. Meanwhile, small language models (SLMs) deployed on edge devices offer efficiency and low latency but often struggle with complex reasoning tasks. Unfortunately, current RAG approaches are predominantly based on centralized databases and have not been adapted to address the distinct constraints associated with deploying SLMs in edge environments. To bridge this gap, we propose Edge-Assisted and Collaborative RAG (EACO-RAG), a lightweight framework that leverages distributed edge nodes for adaptive knowledge updates and retrieval. EACO-RAG also employs a hierarchical collaborative gating mechanism to dynamically select among local, edge-assisted, and cloud-based strategies, with a carefully designed algorithm based on Safe Online Bayesian Optimization to maximize the potential performance enhancements. Experimental results demonstrate that EACO-RAG matches the accuracy of cloud-based knowledge graph RAG systems while reducing total costs by up to 84.6% under relaxed delay constraints and by 65.3% under stricter delay requirements. This work represents our initial effort toward achieving a distributed and scalable tiered LLM deployments, with EACO-RAG serving as a promising first step in unlocking the full potential of hybrid edge-cloud intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20856v2">Strada-LLM: Graph LLM for traffic prediction</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 The reviewers decided to reject it. After getting the reviews, we wanted to study more.
    </div>
    <details class="paper-abstract">
      Traffic prediction is a vital component of intelligent transportation systems. By reasoning about traffic patterns in both the spatial and temporal dimensions, accurate and interpretable predictions can be provided. A considerable challenge in traffic prediction lies in handling the diverse data distributions caused by vastly different traffic conditions occurring at different locations. LLMs have been a dominant solution due to their remarkable capacity to adapt to new datasets with very few labeled data samples, i.e., few-shot adaptability. However, existing forecasting techniques mainly focus on extracting local graph information and forming a text-like prompt, leaving LLM- based traffic prediction an open problem. This work presents a probabilistic LLM for traffic forecasting with three highlights. We propose a graph-aware LLM for traffic prediction that considers proximal traffic information. Specifically, by considering the traffic of neighboring nodes as covariates, our model outperforms the corresponding time-series LLM. Furthermore, we adopt a lightweight approach for efficient domain adaptation when facing new data distributions in few-shot fashion. The comparative experiment demonstrates the proposed method outperforms the state-of-the-art LLM-based methods and the traditional GNN- based supervised approaches. Furthermore, Strada-LLM can be easily adapted to different LLM backbones without a noticeable performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14050v3">Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. We reduce biases by finetuning on curated non-harmful text, but find only direct preference optimization to be effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10201v1">Prediction hubs are context-informed frequent tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Hubness, the tendency for few points to be among the nearest neighbours of a disproportionate number of other points, commonly arises when applying standard distance measures to high-dimensional data, often negatively impacting distance-based analysis. As autoregressive large language models (LLMs) operate on high-dimensional representations, we ask whether they are also affected by hubness. We first show, theoretically, that the only representation comparison operation performed by LLMs, namely that between context and unembedding vectors to determine continuation probabilities, is not characterized by the concentration of distances phenomenon that typically causes the appeareance of nuisance hubness. We then empirically show that this comparison still leads to a high degree of hubness, but the hubs in this case do not constitute a disturbance. They are rather the result of context-modulated frequent tokens often appearing in the pool of likely candidates for next token prediction. On the other hand, when other distance computations involving LLM representations are performed, we do not have the same theoretical guarantees, and, indeed, we see nuisance hubs appear. In summary, our work highlights, on the one hand, how hubness, while omnipresent in high-dimensional spaces, is not always a negative property that needs to be mitigated, and, on the other hand, it shows that various widely-used LLMs have developed a guessing strategy that consists in constantly assigning a high probability to frequent tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01237v2">Self-Refinement Strategies for LLM-based Product Attribute Value Extraction</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Structured product data, in the form of attribute-value pairs, is essential for e-commerce platforms to support features such as faceted product search and attribute-based product comparison. However, vendors often provide unstructured product descriptions, making attribute value extraction necessary to ensure data consistency and usability. Large language models (LLMs) have demonstrated their potential for product attribute value extraction in few-shot scenarios. Recent research has shown that self-refinement techniques can improve the performance of LLMs on tasks such as code generation and text-to-SQL translation. For other tasks, the application of these techniques has resulted in increased costs due to processing additional tokens, without achieving any improvement in performance. This paper investigates applying two self-refinement techniques (error-based prompt rewriting and self-correction) to the product attribute value extraction task. The self-refinement techniques are evaluated across zero-shot, few-shot in-context learning, and fine-tuning scenarios using GPT-4o. The experiments show that both self-refinement techniques fail to significantly improve the extraction performance while substantially increasing processing costs. For scenarios with development data, fine-tuning yields the highest performance, while the ramp-up costs of fine-tuning are balanced out as the amount of product descriptions increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10197v1">MathConstruct: Challenging LLM Reasoning with Constructive Proofs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate impressive performance in mathematics, existing math benchmarks come with significant limitations. Many focus on problems with fixed ground-truth answers, and are often saturated due to problem simplicity or the viability of guessing or memorization. Crucially, they capture only a narrow subset of relevant math problems. To address this research gap, we introduce \mc, a new benchmark of 126 challenging problems sourced from various math competitions, which targets constructive proofs, a widely encountered problem type requiring the construction of mathematical objects with specific properties. These proofs are particularly suitable for LLM evaluation, as solution correctness can be easily verified. Our automated verifiers also enable MathConstruct to generate problem variations, used to evaluate robustness. State-of-the-art LLMs solve only 54% of MathConstruct problems, highlighting its complexity and importance for LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v3">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 Code will be available at https://github.com/iamhankai/Forest-of-Thought
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency.Code will be available at https://github.com/iamhankai/Forest-of-Thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14391v2">Context-Aware or Context-Insensitive? Assessing LLMs' Performance in Document-Level Translation</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 9 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly strong contenders in machine translation. In this work, we focus on document-level translation, where some words cannot be translated without context from outside the sentence. Specifically, we investigate the ability of prominent LLMs to utilize the document context during translation through a perturbation analysis (analyzing models' robustness to perturbed and randomized document context) and an attribution analysis (examining the contribution of relevant context to the translation). We conduct an extensive evaluation across nine LLMs from diverse model families and training paradigms, including translation-specialized LLMs, alongside two encoder-decoder transformer baselines. We find that LLMs' improved document-translation performance compared to encoder-decoder models is not reflected in pronoun translation performance. Our analysis highlight the need for context-aware finetuning of LLMs with a focus on relevant parts of the context to improve their reliability for document-level translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10347v2">A Unified Approach to Routing and Cascading for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      The availability of a wide range of large language models (LLMs) embedded in various agentic systems has significantly increased the potential of model selection strategies to improve the cost-performance tradeoff. Existing strategies involve either routing, where a single model is chosen per query, or cascading, which sequentially runs increasingly larger models until a satisfactory answer is found. However, current approaches face three key limitations: they (1) lack formal proofs of optimality, (2) fail to identify the conditions under which these strategies are most effective to improve the cost-performance tradeoff, and (3) are unable to combine both paradigms for further improvements. To address these issues, we first derive a novel optimal strategy for cascading and prove the optimality of an existing routing strategy. Further, we propose cascade routing, a unified framework that integrates routing and cascading into a theoretically optimal strategy. Through our analysis, we identify good quality estimators as the critical factor for the success of model selection paradigms. Finally, in our experiments, we show that cascade routing consistently outperforms the individual approaches by a large margin and we analyze quality estimators to determine when routing and/or cascading are useful paradigms for model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v3">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07016v3">Delving into LLM-assisted writing in biomedical publications through excess vocabulary</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 v3: Updating the manuscript to include all PubMed abstracts until the end of 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like ChatGPT can generate and revise text with human-level performance. These models come with clear limitations: they can produce inaccurate information, reinforce existing biases, and be easily misused. Yet, many scientists use them for their scholarly writing. But how wide-spread is such LLM usage in the academic literature? To answer this question for the field of biomedical research, we present an unbiased, large-scale approach: we study vocabulary changes in over 15 million biomedical abstracts from 2010--2024 indexed by PubMed, and show how the appearance of LLMs led to an abrupt increase in the frequency of certain style words. This excess word analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs. This lower bound differed across disciplines, countries, and journals, reaching 40% for some subcorpora. We show that LLMs have had an unprecedented impact on scientific writing in biomedical research, surpassing the effect of major world events such as the Covid pandemic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10050v1">A Survey on LLM-powered Agents for Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Recommender systems are essential components of many online platforms, yet traditional approaches still struggle with understanding complex user preferences and providing explainable recommendations. The emergence of Large Language Model (LLM)-powered agents offers a promising approach by enabling natural language interactions and interpretable reasoning, potentially transforming research in recommender systems. This survey provides a systematic review of the emerging applications of LLM-powered agents in recommender systems. We identify and analyze three key paradigms in current research: (1) Recommender-oriented approaches, which leverage intelligent agents to enhance the fundamental recommendation mechanisms; (2) Interaction-oriented approaches, which facilitate dynamic user engagement through natural dialogue and interpretable suggestions; and (3) Simulation-oriented approaches, which employ multi-agent frameworks to model complex user-item interactions and system dynamics. Beyond paradigm categorization, we analyze the architectural foundations of LLM-powered recommendation agents, examining their essential components: profile construction, memory management, strategic planning, and action execution. Our investigation extends to a comprehensive analysis of benchmark datasets and evaluation frameworks in this domain. This systematic examination not only illuminates the current state of LLM-powered agent recommender systems but also charts critical challenges and promising research directions in this transformative field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10038v1">POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      POI representation learning plays a crucial role in handling tasks related to user mobility data. Recent studies have shown that enriching POI representations with multimodal information can significantly enhance their task performance. Previously, the textual information incorporated into POI representations typically involved only POI categories or check-in content, leading to relatively weak textual features in existing methods. In contrast, large language models (LLMs) trained on extensive text data have been found to possess rich textual knowledge. However leveraging such knowledge to enhance POI representation learning presents two key challenges: first, how to extract POI-related knowledge from LLMs effectively, and second, how to integrate the extracted information to enhance POI representations. To address these challenges, we propose POI-Enhancer, a portable framework that leverages LLMs to improve POI representations produced by classic POI learning models. We first design three specialized prompts to extract semantic information from LLMs efficiently. Then, the Dual Feature Alignment module enhances the quality of the extracted information, while the Semantic Feature Fusion module preserves its integrity. The Cross Attention Fusion module then fully adaptively integrates such high-quality information into POI representations and Multi-View Contrastive Learning further injects human-understandable semantic information into these representations. Extensive experiments on three real-world datasets demonstrate the effectiveness of our framework, showing significant improvements across all baseline representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12475v2">RareAgents: Advancing Rare Disease Care through LLM-Empowered Multi-disciplinary Team</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Rare diseases, despite their low individual incidence, collectively impact around 300 million people worldwide due to the vast number of diseases. The involvement of multiple organs and systems, and the shortage of specialized doctors with relevant experience make diagnosing and treating rare diseases more challenging than common diseases. Recently, agents powered by large language models (LLMs) have demonstrated notable applications across various domains. In the medical field, some agent methods have outperformed direct prompts in question-answering tasks from medical examinations. However, current agent frameworks are not well-adapted to real-world clinical scenarios, especially those involving the complex demands of rare diseases. To bridge this gap, we introduce RareAgents, the first LLM-driven multi-disciplinary team framework designed specifically for the complex clinical context of rare diseases. RareAgents integrates advanced Multidisciplinary Team (MDT) coordination, memory mechanisms, and medical tools utilization, leveraging Llama-3.1-8B/70B as the base model. Experimental results show that RareAgents outperforms state-of-the-art domain-specific models, GPT-4o, and current agent frameworks in differential diagnosis and medication recommendation for rare diseases. Furthermore, we contribute a novel rare disease dataset, MIMIC-IV-Ext-Rare, to support further advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09990v1">X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Despite the rapid development of safety alignment techniques for LLMs, defending against multi-turn jailbreaks is still a challenging task. In this paper, we conduct a comprehensive comparison, revealing that some existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability, i.e., reducing general capabilities or causing the over-refusal problem. From the perspective of mechanism interpretability of LLMs, we discover that these methods fail to establish a boundary that exactly distinguishes safe and harmful feature representations. Therefore, boundary-safe representations close to harmful representations are inevitably disrupted, leading to a decline in usability. To address this issue, we propose X-Boundary to push harmful representations away from boundary-safe representations and obtain an exact distinction boundary. In this way, harmful representations can be precisely erased without disrupting safe ones. Experimental results show that X-Boundary achieves state-of-the-art defense performance against multi-turn jailbreaks, while reducing the over-refusal rate by about 20% and maintaining nearly complete general capability. Furthermore, we theoretically prove and empirically verify that X-Boundary can accelerate the convergence process during training. Please see our code at: https://github.com/AI45Lab/X-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09977v1">LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet for LC or RAG Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Effectively incorporating external knowledge into Large Language Models (LLMs) is crucial for enhancing their capabilities and addressing real-world needs. Retrieval-Augmented Generation (RAG) offers an effective method for achieving this by retrieving the most relevant fragments into LLMs. However, the advancements in context window size for LLMs offer an alternative approach, raising the question of whether RAG remains necessary for effectively handling external knowledge. Several existing studies provide inconclusive comparisons between RAG and long-context (LC) LLMs, largely due to limitations in the benchmark designs. In this paper, we present LaRA, a novel benchmark specifically designed to rigorously compare RAG and LC LLMs. LaRA encompasses 2,326 test cases across four practical QA task categories and three types of naturally occurring long texts. Through systematic evaluation of seven open-source and four proprietary LLMs, we find that the optimal choice between RAG and LC depends on a complex interplay of factors, including the model's parameter size, long-text capabilities, context length, task type, and the characteristics of the retrieved chunks. Our findings provide actionable guidelines for practitioners to effectively leverage both RAG and LC approaches in developing and deploying LLM applications. Our code and dataset is provided at: \href{https://github.com/likuanppd/LaRA}{\textbf{https://github.com/likuanppd/LaRA}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19324v2">Reward-Guided Speculative Decoding for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draft model with a more powerful target model, incorporating a controlled bias to prioritize high-reward outputs, in contrast to existing speculative decoding methods that enforce strict unbiasedness. RSD employs a process reward model to evaluate intermediate decoding steps and dynamically decide whether to invoke the target model, optimizing the trade-off between computational cost and output quality. We theoretically demonstrate that a threshold-based mixture strategy achieves an optimal balance between resource utilization and performance. Extensive evaluations on challenging reasoning benchmarks, including Olympiad-level tasks, show that RSD delivers significant efficiency gains against decoding with the target model only (up to 4.4x fewer FLOPs), while achieving significant better accuracy than parallel decoding method on average (up to +3.5). These results highlight RSD as a robust and cost-effective approach for deploying LLMs in resource-intensive scenarios. The code is available at https://github.com/BaohaoLiao/RSD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v1">MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 32 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03824v3">Syntriever: How to Train Your Retriever with Synthetic Data from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted
    </div>
    <details class="paper-abstract">
      LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v2">AlphaPO - Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce \textbf{AlphaPO}, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15\% to 50\% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10596v1">Post-training an LLM for RAG? Train on Self-Generated Demonstrations</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with knowledge intensive NLP tasks, such as answering "Who won the latest World Cup?" because the knowledge they learn during training may be insufficient or outdated. Conditioning generation on retrieved documents -- a technique known as retrieval augmented generation (RAG) -- mitigates these shortcomings by allowing the model to leverage in-context information. Practitioners can improve LLM RAG performance by fine-tuning on retrieval-augmented instructions, but must beware that this can cause undesirable model behaviors like hallucinations. We attribute this degradation to the fact that the training data is likely to be out-of-distribution for the model and may suffer from quality issues, such as misalignment between retrievals and target responses (since retrievals are frequently added post-hoc). We propose a recipe for training RAG-enabled LLMs using self-generated demonstrations, thereby avoiding training on out-of-distribution text and integrating retrievals into the LLM responses. We evaluate our method on knowledge intensive question answering (QA) tasks and show that our method teaches LLMs to properly handle in-context retrievals and abstain from questions it will likely get wrong. Compared to conventional RA-IT methods, our method prevents model degradation in non-RAG settings while exhibiting superior QA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10577v1">Man Made Language Models? Evaluating LLMs' Perpetuation of Masculine Generics Bias</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to propagate and even amplify gender bias, in English and other languages, in specific or constrained contexts. However, no studies so far have focused on gender biases conveyed by LLMs' responses to generic instructions, especially with regard to masculine generics (MG). MG are a linguistic feature found in many gender-marked languages, denoting the use of the masculine gender as a "default" or supposedly neutral gender to refer to mixed group of men and women, or of a person whose gender is irrelevant or unknown. Numerous psycholinguistics studies have shown that MG are not neutral and induce gender bias. This work aims to analyze the use of MG by both proprietary and local LLMs in responses to generic instructions and evaluate their MG bias rate. We focus on French and create a human noun database from existing lexical resources. We filter existing French instruction datasets to retrieve generic instructions and analyze the responses of 6 different LLMs. Overall, we find that $\approx$39.5\% of LLMs' responses to generic instructions are MG-biased ($\approx$73.1\% across responses with human nouns). Our findings also reveal that LLMs are reluctant to using gender-fair language spontaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10563v1">Accelerating Unbiased LLM Evaluation via Synthetic Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      When developing new large language models (LLMs), a key step is evaluating their final performance, often by computing the win-rate against a reference model based on external feedback. Human feedback is the gold standard, particularly for capturing nuanced qualities like coherence, readability, and alignment with human expectations. However, human evaluations are costly -- even for large tech companies -- and when conducted with active users, they may negatively impact user experience. A promising alternative is synthetic feedback, where evaluations are conducted by other large language models, including reward models. While this eliminates the need for costly human annotations, it introduces biases that may distort the evaluation process. In this work, we propose a statistically principled framework that integrates human and synthetic feedback to reduce reliance on human annotations while maintaining unbiased win-rate calculations. Our experiments demonstrate a reduction in human annotations by up to 12.2% with an off-the-shelf synthetic evaluator and up to 24.8% with a finetuned variant. Apart from being generalizable, scalable, and free of hyper-parameter tuning, our method offers predictable annotation savings, which can be estimated based on data-dependent characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14482v3">ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      In this work, we introduce ChatQA 2, an Llama 3.0-based model with a 128K context window, designed to bridge the gap between open-source LLMs and leading proprietary models (e.g., GPT-4-Turbo-2024-04-09) in long context understanding and retrieval-augmented generation (RAG) capabilities. These two capabilities are complementary to each other and essential for LLMs to process large volumes of information that cannot fit into a single prompt. We present a detailed continued training recipe to extend the context window of Llama3-70B-base from 8K to 128K tokens, along with a three-stage instruction tuning process to enhance the model's instruction-following, RAG performance, and long-context understanding capabilities. Our results demonstrate that the Llama3-ChatQA-2-70B model outperforms most existing state-of-the-art models, including GPT-4-Turbo-2024-04-09, Qwen2-72B-Instruct, and Llama3.1-70B-Instruct, on ultra-long tasks beyond 100K tokens, as well as on the RAG benchmark using only a 4K context window, showing the strong long context capability across varying sequence lengths. We further provide extensive comparisons between direct long-context and RAG solutions using the same state-of-the-art long-context LLMs. Interestingly, we find that the performance of strong long-context LLMs using RAG improves when retrieving a larger number of chunks. With a large set of top-k chunks, RAG consistently outperforms direct long-context solution using the same state-of-the-art long-context models (e.g., Llama3-ChatQA-2-70B and Qwen2-72B-Instruct) on both 32K and 128K benchmarks. We open-source the model weights, training data, and the evaluation setup for the for the community: https://chatqa2-project.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15948v3">Teaching LLMs to Abstain across Languages via Multilingual Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Multilingual LLMs often have knowledge disparities across languages, with larger gaps in under-resourced languages. Teaching LLMs to abstain in the face of knowledge gaps is thus a promising strategy to mitigate hallucinations in multilingual settings. However, previous studies on LLM abstention primarily focus on English; we find that directly applying existing solutions beyond English results in up to 20.5% performance gaps between high and low-resource languages, potentially due to LLMs' drop in calibration and reasoning beyond a few resource-rich languages. To this end, we propose strategies to enhance LLM abstention by learning from multilingual feedback, where LLMs self-reflect on proposed answers in one language by generating multiple feedback items in related languages: we show that this helps identifying the knowledge gaps across diverse languages, cultures, and communities. Extensive experiments demonstrate that our multilingual feedback approach outperforms various strong baselines, achieving up to 9.2% improvement for low-resource languages across three black-box and open models on three datasets, featuring open-book, closed-book, and commonsense QA. Further analysis reveals that multilingual feedback is both an effective and a more equitable abstain strategy to serve diverse language speakers, and cultural factors have great impact on language selection and LLM abstention behavior, highlighting future directions for multilingual and multi-cultural reliable language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10525v1">Towards Watermarking of Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      While watermarks for closed LLMs have matured and have been included in large-scale deployments, these methods are not applicable to open-source models, which allow users full control over the decoding process. This setting is understudied yet critical, given the rising performance of open-source models. In this work, we lay the foundation for systematic study of open-source LLM watermarking. For the first time, we explicitly formulate key requirements, including durability against common model modifications such as model merging, quantization, or finetuning, and propose a concrete evaluation setup. Given the prevalence of these modifications, durability is crucial for an open-source watermark to be effective. We survey and evaluate existing methods, showing that they are not durable. We also discuss potential ways to improve their durability and highlight remaining challenges. We hope our work enables future progress on this important problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10522v1">GraphiT: Efficient Node Classification on Text-Attributed Graphs with Prompt Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) to graph data has attracted a lot of attention recently. LLMs allow us to use deep contextual embeddings from pretrained models in text-attributed graphs, where shallow embeddings are often used for the text at- tributes of nodes. However, it is still challenging to efficiently en- code the graph structure and features into a sequential form for use by LLMs. In addition, the performance of an LLM alone, is highly dependent on the structure of the input prompt, which limits their effectiveness as a reliable approach and often requires iterative man- ual adjustments that could be slow, tedious and difficult to replicate programmatically. In this paper, we propose GraphiT (Graphs in Text), a framework for encoding graphs into a textual format and optimizing LLM prompts for graph prediction tasks. Here we focus on node classification for text-attributed graphs. We encode the graph data for every node and its neighborhood into a concise text to enable LLMs to better utilize the information in the graph. We then further programmatically optimize the LLM prompts us- ing the DSPy framework to automate this step and make it more efficient and reproducible. GraphiT outperforms our LLM-based baselines on three datasets and we show how the optimization step in GraphiT leads to measurably better results without manual prompt tweaking. We also demonstrated that our graph encoding approach is competitive to other graph encoding methods while being less expensive because it uses significantly less tokens for the same task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10487v1">Fast Proxies for LLM Robustness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Evaluating the robustness of LLMs to adversarial attacks is crucial for safe deployment, yet current red-teaming methods are often prohibitively expensive. We compare the ability of fast proxy metrics to predict the real-world robustness of an LLM against a simulated attacker ensemble. This allows us to estimate a model's robustness to computationally expensive attacks without requiring runs of the attacks themselves. Specifically, we consider gradient-descent-based embedding-space attacks, prefilling attacks, and direct prompting. Even though direct prompting in particular does not achieve high ASR, we find that it and embedding-space attacks can predict attack success rates well, achieving $r_p=0.87$ (linear) and $r_s=0.94$ (Spearman rank) correlations with the full attack ensemble while reducing computational cost by three orders of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10361v1">Enhancing Multilingual LLM Pretraining with Model-Based Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Dataset curation has become a basis for strong large language model (LLM) performance. While various rule-based filtering heuristics exist for English and multilingual datasets, model-based filtering techniques have primarily focused on English. To address the disparity stemming from limited research on non-English languages, we propose a model-based filtering framework for multilingual datasets that aims to identify a diverse set of structured and knowledge-rich samples. Our approach emphasizes transparency, simplicity, and efficiency, leveraging Transformer- and FastText-based classifiers to ensure the broad accessibility of our technique and data. We conduct comprehensive ablation studies on the FineWeb-2 web crawl dataset across diverse language families, scripts, and resource availability to demonstrate the effectiveness of our method. Training a 1B-parameter Llama model for 70B and 119B tokens, our approach can match the baseline MMLU score with as little as 15% of the training tokens, while also improving across other benchmarks. These findings provide strong evidence for the generalizability of our approach to other languages. As a result, we extend our framework to 20 languages for which we release the refined pretraining datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00843v2">The Graph's Apprentice: Teaching an LLM Low Level Knowledge for Circuit Quality Estimation</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Logic synthesis is a crucial phase in the circuit design process, responsible for transforming hardware description language (HDL) designs into optimized netlists. However, traditional logic synthesis methods are computationally intensive, restricting their iterative use in refining chip designs. Recent advancements in large language models (LLMs), particularly those fine-tuned on programming languages, present a promising alternative. This work proposes augmenting LLMs with predictor networks trained to estimate circuit quality directly from HDL code. To enhance performance, the model is regularized using embeddings from graph neural networks (GNNs) trained on Look-Up Table (LUT) graphs, thereby incorporating lower-level circuit insights. The proposed method demonstrates superior performance compared to existing graph-based RTL-level estimation techniques on the established benchmark OpenABCD, while providing instant feedback on HDL code quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10325v1">Process Reward Models for LLM Agents: Practical Framework and Directions</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We introduce Agent Process Reward Models (AgentPRM), a simple and scalable framework for training LLM agents to continually improve through interactions. AgentPRM follows a lightweight actor-critic paradigm, using Monte Carlo rollouts to compute reward targets and optimize policies. It requires minimal modifications to existing RLHF pipelines, making it easy to integrate at scale. Beyond AgentPRM, we propose InversePRM, which learns process rewards directly from demonstrations without explicit outcome supervision. We also explore key challenges and opportunities, including exploration, process reward shaping, and model-predictive reasoning. We evaluate on ALFWorld benchmark, show that small 3B models trained with AgentPRM and InversePRM outperform strong GPT-4o baselines, and analyze test-time scaling, reward hacking, and more. Our code is available at: https://github.com/sanjibanc/agent_prm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10308v1">LLM-Powered Preference Elicitation in Combinatorial Assignment</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      We study the potential of large language models (LLMs) as proxies for humans to simplify preference elicitation (PE) in combinatorial assignment. While traditional PE methods rely on iterative queries to capture preferences, LLMs offer a one-shot alternative with reduced human effort. We propose a framework for LLM proxies that can work in tandem with SOTA ML-powered preference elicitation schemes. Our framework handles the novel challenges introduced by LLMs, such as response variability and increased computational costs. We experimentally evaluate the efficiency of LLM proxies against human queries in the well-studied course allocation domain, and we investigate the model capabilities required for success. We find that our approach improves allocative efficiency by up to 20%, and these results are robust across different LLMs and to differences in quality and accuracy of reporting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13610v2">MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 NAACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Integrating tools into Large Language Models (LLMs) has facilitated the widespread application. Despite this, in specialized downstream task contexts, reliance solely on tools is insufficient to fully address the complexities of the real world. This particularly restricts the effective deployment of LLMs in fields such as medicine. In this paper, we focus on the downstream tasks of medical calculators, which use standardized tests to assess an individual's health status. We introduce MeNTi, a universal agent architecture for LLMs. MeNTi integrates a specialized medical toolkit and employs meta-tool and nested calling mechanisms to enhance LLM tool utilization. Specifically, it achieves flexible tool selection and nested tool calling to address practical issues faced in intricate medical scenarios, including calculator selection, slot filling, and unit conversion. To assess the capabilities of LLMs for quantitative assessment throughout the clinical process of calculator scenarios, we introduce CalcQA. This benchmark requires LLMs to use medical calculators to perform calculations and assess patient health status. CalcQA is constructed by professional physicians and includes 100 case-calculator pairs, complemented by a toolkit of 281 medical tools. The experimental results demonstrate significant performance improvements with our framework. This research paves new directions for applying LLMs in demanding scenarios of medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v3">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18649v2">LeDex: Training LLMs to Better Self-Debug and Explain Code</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 This paper is accepted by The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)
    </div>
    <details class="paper-abstract">
      In the domain of code generation, self-debugging is crucial. It allows LLMs to refine their generated code based on execution feedback. This is particularly important because generating correct solutions in one attempt proves challenging for complex tasks. Prior works on self-debugging mostly focus on prompting methods by providing LLMs with few-shot examples, which work poorly on small open-sourced LLMs. In this work, we propose LeDex, a training framework that significantly improves the self-debugging capability of LLMs. Intuitively, we observe that a chain of explanations on the wrong code followed by code refinement helps LLMs better analyze the wrong code and do refinement. We thus propose an automated pipeline to collect a high-quality dataset for code explanation and refinement by generating a number of explanations and refinement trajectories from the LLM itself or a larger teacher model and filtering via execution verification. We perform supervised fine-tuning (SFT) and further reinforcement learning (RL) on both success and failure trajectories with a novel reward design considering code explanation and refinement quality. SFT improves the pass@1 by up to 15.92% and pass@10 by 9.30% over four benchmarks. RL training brings additional up to 3.54% improvement on pass@1 and 2.55% improvement on pass@10. The trained LLMs show iterative refinement ability and can keep refining code continuously. Lastly, our human evaluation shows that the LLMs trained with our framework generate more useful code explanations and help developers better understand bugs in source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09819v1">A Solver-Aided Hierarchical Language for LLM-Driven CAD Design</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been enormously successful in solving a wide variety of structured and unstructured generative tasks, but they struggle to generate procedural geometry in Computer Aided Design (CAD). These difficulties arise from an inability to do spatial reasoning and the necessity to guide a model through complex, long range planning to generate complex geometry. We enable generative CAD Design with LLMs through the introduction of a solver-aided, hierarchical domain specific language (DSL) called AIDL, which offloads the spatial reasoning requirements to a geometric constraint solver. Additionally, we show that in the few-shot regime, AIDL outperforms even a language with in-training data (OpenSCAD), both in terms of generating visual results closer to the prompt and creating objects that are easier to post-process and reason about.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09801v1">Unit Testing Past vs. Present: Examining LLMs' Impact on Defect Detection and Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs), such as ChatGPT and GitHub Copilot, into software engineering workflows has shown potential to enhance productivity, particularly in software testing. This paper investigates whether LLM support improves defect detection effectiveness during unit testing. Building on prior studies comparing manual and tool-supported testing, we replicated and extended an experiment where participants wrote unit tests for a Java-based system with seeded defects within a time-boxed session, supported by LLMs. Comparing LLM supported and manual testing, results show that LLM support significantly increases the number of unit tests generated, defect detection rates, and overall testing efficiency. These findings highlight the potential of LLMs to improve testing and defect detection outcomes, providing empirical insights into their practical application in software testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09797v1">A Survey on LLM-based News Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      News recommender systems play a critical role in mitigating the information overload problem. In recent years, due to the successful applications of large language model technologies, researchers have utilized Discriminative Large Language Models (DLLMs) or Generative Large Language Models (GLLMs) to improve the performance of news recommender systems. Although several recent surveys review significant challenges for deep learning-based news recommender systems, such as fairness, privacy-preserving, and responsibility, there is a lack of a systematic survey on Large Language Model (LLM)-based news recommender systems. In order to review different core methodologies and explore potential issues systematically, we categorize DLLM-based and GLLM-based news recommender systems under the umbrella of LLM-based news recommender systems. In this survey, we first overview the development of deep learning-based news recommender systems. Then, we review LLM-based news recommender systems based on three aspects: news-oriented modeling, user-oriented modeling, and prediction-oriented modeling. Next, we examine the challenges from various perspectives, including datasets, benchmarking tools, and methodologies. Furthermore, we conduct extensive experiments to analyze how large language model technologies affect the performance of different news recommender systems. Finally, we comprehensively explore the future directions for LLM-based news recommendations in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09778v1">Prompt and circumstance: A word-by-word LLM prompting approach to interlinear glossing for low-resource languages</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Partly automated creation of interlinear glossed text (IGT) has the potential to assist in linguistic documentation. We argue that LLMs can make this process more accessible to linguists because of their capacity to follow natural-language instructions. We investigate the effectiveness of a retrieval-based LLM prompting approach to glossing, applied to the seven languages from the SIGMORPHON 2023 shared task. Our system beats the BERT-based shared task baseline for every language in the morpheme-level score category, and we show that a simple 3-best oracle has higher word-level scores than the challenge winner (a tuned sequence model) in five languages. In a case study on Tsez, we ask the LLM to automatically create and follow linguistic instructions, reducing errors on a confusing grammatical feature. Our results thus demonstrate the potential contributions which LLMs can make in interactive systems for glossing, both in making suggestions to human annotators and following directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09766v1">LLM-Generated Microservice Implementations from RESTful API Definitions</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The growing need for scalable, maintainable, and fast-deploying systems has made microservice architecture widely popular in software development. This paper presents a system that uses Large Language Models (LLMs) to automate the API-first development of RESTful microservices. This system assists in creating OpenAPI specification, generating server code from it, and refining the code through a feedback loop that analyzes execution logs and error messages. By focusing on the API-first methodology, this system ensures that microservices are designed with well-defined interfaces, promoting consistency and reliability across the development life-cycle. The integration of log analysis enables the LLM to detect and address issues efficiently, reducing the number of iterations required to produce functional and robust services. This process automates the generation of microservices and also simplifies the debugging and refinement phases, allowing developers to focus on higher-level design and integration tasks. This system has the potential to benefit software developers, architects, and organizations to speed up software development cycles and reducing manual effort. To assess the potential of the system, we conducted surveys with six industry practitioners. After surveying practitioners, the system demonstrated notable advantages in enhancing development speed, automating repetitive tasks, and simplifying the prototyping process. While experienced developers appreciated its efficiency for specific tasks, some expressed concerns about its limitations in handling advanced customizations and larger scale projects. The code is publicly available at https://github.com/sirbh/code-gen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00735v2">`Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09749v1">Vote-Tree-Planner: Optimizing Execution Order in LLM-based Task Planning Pipeline via Voting</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted to RSS24-W: TaskSpec
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into closed-loop robotic task planning has become increasingly popular within embodied artificial intelligence. Previous efforts mainly focused on leveraging the strong reasoning abilities of LLMs to enhance task planning performance while often overlooking task planning efficiency and executability due to repetitive queries to LLMs. This paper addresses the synergy between LLMs and task planning systems, aiming to minimize redundancy while enhancing planning effectiveness. Specifically, building upon Prog-Prompt and the high-level concept of Tree-Planner, we propose Vote-Tree-Planner. This sampling strategy utilizes votes to guide plan traversal during the decision-making process. Our approach is motivated by a straightforward observation: assigning weights to agents during decision-making enables the evaluation of critical paths before execution. With this simple vote-tree construction, our method further improves the success rate and reduces the number of queries to LLMs. The experimental results highlight that our Vote-Tree-Planner demonstrates greater stability and shows a higher average success rate and goal condition recall on the unseen dataset compared with previous baseline methods. These findings underscore the potential of the Vote-Tree-Planner to enhance planning accuracy, reliability, and efficiency in LLM-based planning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09720v1">NestQuant: Nested Lattice Quantization for Matrix Products and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent work have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Meta's SpinQuant (perplexity 7.3). Comparisons on various LLM evaluation benchmarks also show a reduction in performance degradation induced by quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09673v1">Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various NLP benchmarks. However, excelling in complex tasks that require nuanced reasoning and precise decision-making demands more than raw language proficiency--LLMs must reason, i.e., think logically, draw from past experiences, and synthesize information to reach conclusions and take action. To enhance reasoning abilities, approaches such as prompting and fine-tuning have been widely explored. While these methods have led to clear improvements in reasoning, their impact on LLM safety remains less understood. In this work, we investigate the interplay between reasoning and safety in LLMs. We highlight the latent safety risks that arise as reasoning capabilities improve, shedding light on previously overlooked vulnerabilities. At the same time, we explore how reasoning itself can be leveraged to enhance safety, uncovering potential mitigation strategies. By examining both the risks and opportunities in reasoning-driven LLM safety, our study provides valuable insights for developing models that are not only more capable but also more trustworthy in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09606v1">Human-LLM Coevolution: Evidence from Academic Writing</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      With a statistical analysis of arXiv paper abstracts, we report a marked drop in the frequency of several words previously identified as overused by ChatGPT, such as "delve", starting soon after they were pointed out in early 2024. The frequency of certain other words favored by ChatGPT, such as "significant", has instead kept increasing. These phenomena suggest that some authors of academic papers have adapted their use of large language models (LLMs), for example, by selecting outputs or applying modifications to the LLM-generated content. Such coevolution and cooperation of humans and LLMs thus introduce additional challenges to the detection of machine-generated text in real-world scenarios. Estimating the impact of LLMs on academic writing by examining word frequency remains feasible, and more attention should be paid to words that were already frequently employed, including those that have decreased in frequency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09597v1">Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted at ICLR 2025 as oral presentation. Code and data at: https://prefeval.github.io/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as chatbots, yet their ability to personalize responses to user preferences remains limited. We introduce PrefEval, a benchmark for evaluating LLMs' ability to infer, memorize and adhere to user preferences in a long-context conversational setting. PrefEval comprises 3,000 manually curated user preference and query pairs spanning 20 topics. PrefEval contains user personalization or preference information in both explicit and implicit forms, and evaluates LLM performance using a generation and a classification task. With PrefEval, we evaluated the aforementioned preference following capabilities of 10 open-source and proprietary LLMs in multi-session conversations with varying context lengths up to 100k tokens. We benchmark with various prompting, iterative feedback, and retrieval-augmented generation methods. Our benchmarking effort reveals that state-of-the-art LLMs face significant challenges in proactively following users' preferences during conversations. In particular, in zero-shot settings, preference following accuracy falls below 10% at merely 10 turns (~3k tokens) across most evaluated models. Even with advanced prompting and retrieval methods, preference following still deteriorates in long-context conversations. Furthermore, we show that fine-tuning on PrefEval significantly improves performance. We believe PrefEval serves as a valuable resource for measuring, understanding, and enhancing LLMs' preference following abilities, paving the way for personalized conversational agents. Our code and dataset are available at https://prefeval.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05925v2">Hello Again! LLM-powered Personalized Agent for Long-term Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Open-domain dialogue systems have seen remarkable advancements with the development of large language models (LLMs). Nonetheless, most existing dialogue systems predominantly focus on brief single-session interactions, neglecting the real-world demands for long-term companionship and personalized interactions with chatbots. Crucial to addressing this real-world need are event summary and persona management, which enable reasoning for appropriate long-term dialogue responses. Recent progress in the human-like cognitive and reasoning capabilities of LLMs suggests that LLM-based agents could significantly enhance automated perception, decision-making, and problem-solving. In response to this potential, we introduce a model-agnostic framework, the Long-term Dialogue Agent (LD-Agent), which incorporates three independently tunable modules dedicated to event perception, persona extraction, and response generation. For the event memory module, long and short-term memory banks are employed to separately focus on historical and ongoing sessions, while a topic-based retrieval mechanism is introduced to enhance the accuracy of memory retrieval. Furthermore, the persona module conducts dynamic persona modeling for both users and agents. The integration of retrieved memories and extracted personas is subsequently fed into the generator to induce appropriate responses. The effectiveness, generality, and cross-domain capabilities of LD-Agent are empirically demonstrated across various illustrative benchmarks, models, and tasks. The code is released at https://github.com/leolee99/LD-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06773v2">Evaluating Zero-Shot Long-Context LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of zero-shot compression techniques on large language models (LLMs) under long-context. We identify the tendency for computational errors to increase under long-context when employing certain compression methods. We propose a hypothesis to explain the varied behavior of different LLM compression techniques and explore remedies to mitigate the performance decline observed in some techniques under long-context. This is a course report for COS 598D Machine Learning and Systems by Prof. Kai Li at Princeton University. Due to limited computational resources, our experiments were conducted only on LLaMA-2-7B-32K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09532v1">Mind the Gap! Choice Independence in Using Multilingual LLMs for Persuasive Co-Writing Tasks in Different Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Recent advances in generative AI have precipitated a proliferation of novel writing assistants. These systems typically rely on multilingual large language models (LLMs), providing globalized workers the ability to revise or create diverse forms of content in different languages. However, there is substantial evidence indicating that the performance of multilingual LLMs varies between languages. Users who employ writing assistance for multiple languages are therefore susceptible to disparate output quality. Importantly, recent research has shown that people tend to generalize algorithmic errors across independent tasks, violating the behavioral axiom of choice independence. In this paper, we analyze whether user utilization of novel writing assistants in a charity advertisement writing task is affected by the AI's performance in a second language. Furthermore, we quantify the extent to which these patterns translate into the persuasiveness of generated charity advertisements, as well as the role of peoples' beliefs about LLM utilization in their donation choices. Our results provide evidence that writers who engage with an LLM-based writing assistant violate choice independence, as prior exposure to a Spanish LLM reduces subsequent utilization of an English LLM. While these patterns do not affect the aggregate persuasiveness of the generated advertisements, people's beliefs about the source of an advertisement (human versus AI) do. In particular, Spanish-speaking female participants who believed that they read an AI-generated advertisement strongly adjusted their donation behavior downwards. Furthermore, people are generally not able to adequately differentiate between human-generated and LLM-generated ads. Our work has important implications for the design, development, integration, and adoption of multilingual LLMs as assistive agents -- particularly in writing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05331v2">Fine-Tuned LLMs are "Time Capsules" for Tracking Societal Bias Through Books</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages (excluding references), accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Books, while often rich in cultural insights, can also mirror societal biases of their eras - biases that Large Language Models (LLMs) may learn and perpetuate during training. We introduce a novel method to trace and quantify these biases using fine-tuned LLMs. We develop BookPAGE, a corpus comprising 593 fictional books across seven decades (1950-2019), to track bias evolution. By fine-tuning LLMs on books from each decade and using targeted prompts, we examine shifts in biases related to gender, sexual orientation, race, and religion. Our findings indicate that LLMs trained on decade-specific books manifest biases reflective of their times, with both gradual trends and notable shifts. For example, model responses showed a progressive increase in the portrayal of women in leadership roles (from 8% to 22%) from the 1950s to 2010s, with a significant uptick in the 1990s (from 4% to 12%), possibly aligning with third-wave feminism. Same-sex relationship references increased markedly from the 1980s to 2000s (from 0% to 10%), mirroring growing LGBTQ+ visibility. Concerningly, negative portrayals of Islam rose sharply in the 2000s (26% to 38%), likely reflecting post-9/11 sentiments. Importantly, we demonstrate that these biases stem mainly from the books' content and not the models' architecture or initial training. Our study offers a new perspective on societal bias trends by bridging AI, literary studies, and social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09497v1">Improve LLM-based Automatic Essay Scoring with Linguistic Features</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 To be published in the workshop Innovation and Responsibility in AI-Supported Education (iRaise) at the 2025 Conference on Artificial Intelligence (AAAI)
    </div>
    <details class="paper-abstract">
      Automatic Essay Scoring (AES) assigns scores to student essays, reducing the grading workload for instructors. Developing a scoring system capable of handling essays across diverse prompts is challenging due to the flexibility and diverse nature of the writing task. Existing methods typically fall into two categories: supervised feature-based approaches and large language model (LLM)-based methods. Supervised feature-based approaches often achieve higher performance but require resource-intensive training. In contrast, LLM-based methods are computationally efficient during inference but tend to suffer from lower performance. This paper combines these approaches by incorporating linguistic features into LLM-based scoring. Experimental results show that this hybrid method outperforms baseline models for both in-domain and out-of-domain writing prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00326v9">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 19 pages, 12 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09419v1">On multi-token prediction for efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      We systematically investigate multi-token prediction (MTP) capabilities within LLMs pre-trained for next-token prediction (NTP). We first show that such models inherently possess MTP capabilities via numerical marginalization over intermediate token probabilities, though performance is data-dependent and improves with model scale. Furthermore, we explore the challenges of integrating MTP heads into frozen LLMs and find that their hidden layers are strongly specialized for NTP, making adaptation non-trivial. Finally, we show that while joint training of MTP heads with the backbone improves performance, it cannot fully overcome this barrier, prompting further research in this direction. Our findings provide a deeper understanding of MTP applied to pretrained LLMs, informing strategies for accelerating inference through parallel token prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02280v2">The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable capabilities on not just language tasks, but also various tasks that are not linguistic in nature, such as logical reasoning and social inference. In the human brain, neuroscience has identified a core language system that selectively and causally supports language processing. We here ask whether similar specialization for language emerges in LLMs. We identify language-selective units within 18 popular LLMs, using the same localization approach that is used in neuroscience. We then establish the causal role of these units by demonstrating that ablating LLM language-selective units -- but not random units -- leads to drastic deficits in language tasks. Correspondingly, language-selective LLM units are more aligned to brain recordings from the human language system than random units. Finally, we investigate whether our localization method extends to other cognitive domains: while we find specialized networks in some LLMs for reasoning and social capabilities, there are substantial differences among models. These findings provide functional and causal evidence for specialization in large language models, and highlight parallels with the functional organization in the brain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04708v2">Exploring Hierarchical Molecular Graph Representation in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages, 4 tables, 1 figure, paper under review
    </div>
    <details class="paper-abstract">
      Following the milestones in large language models (LLMs) and multimodal models, we have seen a surge in applying LLMs to biochemical tasks. Leveraging graph features and molecular text representations, LLMs can tackle various tasks, such as predicting chemical reaction outcomes and describing molecular properties. However, most current work overlooks the *multi-level nature* of the graph modality, even though different chemistry tasks may benefit from different feature levels. In this work, we first study the effect of feature granularity and reveal that even reducing all GNN-generated feature tokens to a single one does not significantly impact model performance. We then investigate the effect of various graph feature levels and demonstrate that both the quality of LLM-generated molecules and model performance across different tasks depend on different graph feature levels. Therefore, we conclude with two key insights: (1) current molecular-related multimodal LLMs lack a comprehensive understanding of graph features, and (2) static processing is not sufficient for hierarchical graph feature. We share our findings in detail, with the hope of paving the way for the community to develop more advanced multimodal LLMs for incorporating molecular graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09334v1">ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 MLSys 2025
    </div>
    <details class="paper-abstract">
      Recent developments in large language models (LLMs) have demonstrated their remarkable proficiency in a range of tasks. Compared to in-house homogeneous GPU clusters, deploying LLMs in cloud environments with diverse types of GPUs is crucial for addressing the GPU shortage problem and being more cost-effective. However, the diversity of network environments and various GPU types on the cloud bring difficulties to achieving high-performance serving. In this work, we propose ThunderServe, a high-performance and cost-efficient LLM serving system for heterogeneous cloud environments. We introduce a novel scheduling algorithm, which optimizes the deployment plan of LLM serving to accommodate the heterogeneous resource and network bandwidth conditions in cloud environments. Furthermore, we propose a lightweight re-scheduling mechanism, designed to adapt to fluctuating online conditions (e.g., node failures, workload shifts) without the need for costly restarts of ongoing services. Empirical results in both heterogeneous cloud and homogeneous in-house environments reveal that ThunderServe delivers up to a 2.1$\times$ and on average a $1.7\times$ increase in throughput and achieves up to a 2.5$\times$ and on average a $1.5\times$ reduction in latency deadlines compared with state-of-the-art systems given the same price budget, suggesting opting for cloud services provides a more cost-efficient solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09331v1">Beyond English: The Impact of Prompt Translation Strategies across Languages and Tasks in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted for NAACL findings 2025
    </div>
    <details class="paper-abstract">
      Despite advances in the multilingual capabilities of Large Language Models (LLMs) across diverse tasks, English remains the dominant language for LLM research and development. So, when working with a different language, this has led to the widespread practice of pre-translation, i.e., translating the task prompt into English before inference. Selective pre-translation, a more surgical approach, focuses on translating specific prompt components. However, its current use is sporagic and lacks a systematic research foundation. Consequently, the optimal pre-translation strategy for various multilingual settings and tasks remains unclear. In this work, we aim to uncover the optimal setup for pre-translation by systematically assessing its use. Specifically, we view the prompt as a modular entity, composed of four functional parts: instruction, context, examples, and output, either of which could be translated or not. We evaluate pre-translation strategies across 35 languages covering both low and high-resource languages, on various tasks including Question Answering (QA), Natural Language Inference (NLI), Named Entity Recognition (NER), and Abstractive Summarization. Our experiments show the impact of factors as similarity to English, translation quality and the size of pre-trained data, on the model performance with pre-translation. We suggest practical guidelines for choosing optimal strategies in various multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09328v1">Copilot Arena: A Platform for Code LLM Evaluation in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Evaluating in-the-wild coding capabilities of large language models (LLMs) is a challenging endeavor with no clear solution. We introduce Copilot Arena, a platform to collect user preferences for code generation through native integration into a developer's working environment. Copilot Arena comprises a novel interface for comparing pairs of model outputs, a sampling strategy optimized to reduce latency, and a prompting scheme to enable code completion functionality. Copilot Arena has served over 4.5 million suggestions from 10 models and collected over 11k pairwise judgements. Our results highlight the importance of model evaluations in integrated settings. We find that model rankings from Copilot Arena differ from those of existing evaluations, which we attribute to the more realistic distribution of data and tasks contained in Copilot Arena. We also identify novel insights into human preferences on code such as an observed consistency in user preference across programming languages yet significant variation in preference due to task category. We open-source Copilot Arena and release data to enable human-centric evaluations and improve understanding of coding assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09316v1">A Judge-free LLM Open-ended Generation Benchmark Based on the Distributional Hypothesis</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Evaluating the open-ended text generation of large language models (LLMs) is challenging because of the lack of a clear ground truth and the high cost of human or LLM-based assessments. We propose a novel benchmark that evaluates LLMs using n-gram statistics and rules, without relying on human judgement or LLM-as-a-judge approaches. Using 50 question and reference answer sets, we introduce three new metrics based on n-grams and rules: Fluency, Truthfulness, and Helpfulness. Our benchmark strongly correlates with GPT-4o-based evaluations while requiring significantly fewer computational resources, demonstrating its effectiveness as a scalable alternative for assessing LLMs' open-ended generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09284v1">SparQLe: Speech Queries to Text Translation Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that leverages self-supervised speech representations in combination with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English-language data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising solution for various speech understanding applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07115v2">Online Scheduling for LLM Inference with KV Cache Constraints</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference, where a trained model generates text one word at a time in response to user prompts, is a computationally intensive process requiring efficient scheduling to optimize latency and resource utilization. A key challenge in LLM inference is the management of the Key-Value (KV) cache, which reduces redundant computations but introduces memory constraints. In this work, we model LLM inference with KV cache constraints theoretically and propose novel batching and scheduling algorithms that minimize inference latency while effectively managing the KV cache's memory. We analyze both semi-online and fully online scheduling models, and our results are threefold. First, we provide a polynomial-time algorithm that achieves exact optimality in terms of average latency in the semi-online prompt arrival model. Second, in the fully online case with a stochastic prompt arrival, we introduce an efficient online scheduling algorithm with constant regret. Third, we prove that no algorithm (deterministic or randomized) can achieve a constant competitive ratio in fully online adversarial settings. Our empirical evaluations on a public LLM inference dataset, using the Llama-70B model on A100 GPUs, show that our approach significantly outperforms benchmark algorithms used currently in practice, achieving lower latency while reducing energy consumption. Overall, our results offer a path toward more sustainable and cost-effective LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09211v1">Visual Graph Question Answering with ASP and LLMs for Language Parsing</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 In Proceedings ICLP 2024, arXiv:2502.08453. This work was partially funded from the Bosch Center for AI
    </div>
    <details class="paper-abstract">
      Visual Question Answering (VQA) is a challenging problem that requires to process multimodal input. Answer-Set Programming (ASP) has shown great potential in this regard to add interpretability and explainability to modular VQA architectures. In this work, we address the problem of how to integrate ASP with modules for vision and natural language processing to solve a new and demanding VQA variant that is concerned with images of graphs (not graphs in symbolic form). Images containing graph-based structures are an ubiquitous and popular form of visualisation. Here, we deal with the particular problem of graphs inspired by transit networks, and we introduce a novel dataset that amends an existing one by adding images of graphs that resemble metro lines. Our modular neuro-symbolic approach combines optical graph recognition for graph parsing, a pretrained optical character recognition neural network for parsing labels, Large Language Models (LLMs) for language processing, and ASP for reasoning. This method serves as a first baseline and achieves an overall average accuracy of 73% on the dataset. Our evaluation provides further evidence of the potential of modular neuro-symbolic systems, in particular with pretrained models that do not involve any further training and logic programming for reasoning, to solve complex VQA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09209v1">On LLM-generated Logic Programs and their Inference Execution Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 In Proceedings ICLP 2024, arXiv:2502.08453
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on petabytes of data are highly compressed repositories of a significant proportion of the knowledge accumulated and distilled so far. In this paper we study techniques to elicit this knowledge in the form of several classes of logic programs, including propositional Horn clauses, Dual Horn clauses, relational triplets and Definite Clause Grammars. Exposing this knowledge as logic programs enables sound reasoning methods that can verify alignment of LLM outputs to their intended uses and extend their inference capabilities. We study new execution methods for the generated programs, including soft-unification of abducible facts against LLM-generated content stored in a vector database as well as GPU-based acceleration of minimal model computation that supports inference with large LLM-generated programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09204v1">Logical Lease Litigation: Prolog and LLMs for Rental Law Compliance in New York</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 In Proceedings ICLP 2024, arXiv:2502.08453
    </div>
    <details class="paper-abstract">
      Legal cases require careful logical reasoning following the laws, whereas interactions with non- technical users must be in natural language. As an application combining logical reasoning using Prolog and natural language processing using large language models (LLMs), this paper presents a novel approach and system, LogicLease, to automate the analysis of landlord-tenant legal cases in the state of New York. LogicLease determines compliance with relevant legal requirements by analyzing case descriptions and citing all relevant laws. It leverages LLMs for information extraction and Prolog for legal reasoning. By separating information extraction from legal reasoning, LogicLease achieves greater transparency and control over the legal logic applied to each case. We evaluate the accuracy, efficiency, and robustness of LogicLease through a series of tests, achieving 100% accuracy and an average processing time of 2.57 seconds. LogicLease presents advantages over state-of-the-art LLM- based legal analysis systems by providing clear, step-by-step reasoning, citing specific laws, and distinguishing itself by its ability to avoid hallucinations - a common issue in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09192v1">Thinking beyond the anthropomorphic paradigm benefits LLM research</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of computer science research articles from the past decade and present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). This terminology reflects deeper anthropomorphic conceptualizations which shape how we think about and conduct LLM research. We argue these conceptualizations may be limiting, and that challenging them opens up new pathways for understanding and improving LLMs beyond human analogies. To illustrate this, we identify and analyze five core anthropomorphic assumptions shaping prominent methodologies across the LLM development lifecycle, from the assumption that models must use natural language for reasoning tasks to the assumption that model capabilities should be evaluated through human-centric benchmarks. For each assumption, we demonstrate how non-anthropomorphic alternatives can open new directions for research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09834v3">LLMs Meet Library Evolution: Evaluating Deprecated API Usage in LLM-based Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted by ICSE'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), pre-trained or fine-tuned on large code corpora, have shown effectiveness in generating code completions. However, in LLM-based code completion, LLMs may struggle to use correct and up-to-date Application Programming Interfaces (APIs) due to the rapid and continuous evolution of libraries. While existing studies have highlighted issues with predicting incorrect APIs, the specific problem of deprecated API usage in LLM-based code completion has not been thoroughly investigated. To address this gap, we conducted the first evaluation study on deprecated API usage in LLM-based code completion. This study involved seven advanced LLMs, 145 API mappings from eight popular Python libraries, and 28,125 completion prompts. The study results reveal the status quo (i.e., API usage plausibility and deprecated usage rate) of deprecated API and replacing API usage in LLM-based code completion from the perspectives of model, prompt, and library, and indicate the root causes behind. Based on these findings, we propose two lightweight fixing approaches, REPLACEAPI and INSERTPROMPT, which can serve as baseline approaches for future research on mitigating deprecated API usage in LLM-based completion. Additionally, we provide implications for future research on integrating library evolution with LLM-driven software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09175v1">FLAME: Flexible LLM-Assisted Moderation Engine</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09156v1">Improving TCM Question Answering through Tree-Organized Self-Reflective Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Objectives: Large language models (LLMs) can harness medical knowledge for intelligent question answering (Q&A), promising support for auxiliary diagnosis and medical talent cultivation. However, there is a deficiency of highly efficient retrieval-augmented generation (RAG) frameworks within the domain of Traditional Chinese Medicine (TCM). Our purpose is to observe the effect of the Tree-Organized Self-Reflective Retrieval (TOSRR) framework on LLMs in TCM Q&A tasks. Materials and Methods: We introduce the novel approach of knowledge organization, constructing a tree structure knowledge base with hierarchy. At inference time, our self-reflection framework retrieves from this knowledge base, integrating information across chapters. Questions from the TCM Medical Licensing Examination (MLE) and the college Classics Course Exam (CCE) were randomly selected as benchmark datasets. Results: By coupling with GPT-4, the framework can improve the best performance on the TCM MLE benchmark by 19.85% in absolute accuracy, and improve recall accuracy from 27% to 38% on CCE datasets. In manual evaluation, the framework improves a total of 18.52 points across dimensions of safety, consistency, explainability, compliance, and coherence. Conclusion: The TOSRR framework can effectively improve LLM's capability in Q&A tasks of TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09142v1">LLM-Driven Augmented Reality Puppeteer: Controller-Free Voice-Commanded Robot Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted as conference proceeding in International Conference on Human-Computer Interaction 2025 (HCI International 2025)
    </div>
    <details class="paper-abstract">
      The integration of robotics and augmented reality (AR) presents transformative opportunities for advancing human-robot interaction (HRI) by improving usability, intuitiveness, and accessibility. This work introduces a controller-free, LLM-driven voice-commanded AR puppeteering system, enabling users to teleoperate a robot by manipulating its virtual counterpart in real time. By leveraging natural language processing (NLP) and AR technologies, our system -- prototyped using Meta Quest 3 -- eliminates the need for physical controllers, enhancing ease of use while minimizing potential safety risks associated with direct robot operation. A preliminary user demonstration successfully validated the system's functionality, demonstrating its potential for safer, more intuitive, and immersive robotic control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09101v1">Bridging the Gap Between LLMs and Human Intentions: Progresses and Challenges in Instruction Understanding, Intention Reasoning, and Reliable Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities in understanding and generation. However, when interacting with human instructions in real-world scenarios, LLMs still face significant challenges, particularly in accurately capturing and comprehending human instructions and intentions. This paper focuses on three challenges in LLM-based text generation tasks: instruction understanding, intention reasoning, and reliable generation. Regarding human complex instruction, LLMs have deficiencies in understanding long contexts and instructions in multi-round conversations. For intention reasoning, LLMs may have inconsistent command reasoning, difficulty reasoning about commands containing incorrect information, difficulty understanding user ambiguous language commands, and a weak understanding of user intention in commands. Besides, In terms of reliable generation, LLMs may have unstable generated content and unethical generation. To this end, we classify and analyze the performance of LLMs in challenging scenarios and conduct a comprehensive evaluation of existing solutions. Furthermore, we introduce benchmarks and categorize them based on the aforementioned three core challenges. Finally, we explore potential directions for future research to enhance the reliability and adaptability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06931v2">Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The ability to wield tools was once considered exclusive to human intelligence, but it's now known that many other animals, like crows, possess this capability. Yet, robotic systems still fall short of matching biological dexterity. In this paper, we investigate the use of Large Language Models (LLMs), tool affordances, and object manoeuvrability for non-prehensile tool-based manipulation tasks. Our novel method leverages LLMs based on scene information and natural language instructions to enable symbolic task planning for tool-object manipulation. This approach allows the system to convert the human language sentence into a sequence of feasible motion functions. We have developed a novel manoeuvrability-driven controller using a new tool affordance model derived from visual feedback. This controller helps guide the robot's tool utilization and manipulation actions, even within confined areas, using a stepping incremental approach. The proposed methodology is evaluated with experiments to prove its effectiveness under various manipulation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09082v1">CoSER: Coordinating LLM-Based Persona Simulation of Established Roles</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10913v2">ASHABot: An LLM-Powered Chatbot to Support the Informational Needs of Community Health Workers</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Community health workers (CHWs) provide last-mile healthcare services but face challenges due to limited medical knowledge and training. This paper describes the design, deployment, and evaluation of ASHABot, an LLM-powered, experts-in-the-loop, WhatsApp-based chatbot to address the information needs of CHWs in India. Through interviews with CHWs and their supervisors and log analysis, we examine factors affecting their engagement with ASHABot, and ASHABot's role in addressing CHWs' informational needs. We found that ASHABot provided a private channel for CHWs to ask rudimentary and sensitive questions they hesitated to ask supervisors. CHWs trusted the information they received on ASHABot and treated it as an authoritative resource. CHWs' supervisors expanded their knowledge by contributing answers to questions ASHABot failed to answer, but were concerned about demands on their workload and increased accountability. We emphasize positioning LLMs as supplemental fallible resources within the community healthcare ecosystem, instead of as replacements for supervisor support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v1">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09056v1">An Open Recipe: Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper investigates data selection and model merging methodologies aimed at incorporating advanced reasoning capabilities such as those of DeepSeek R1 into language-specific large language models (LLMs), with a particular focus on the Thai LLM. Our goal is to enhance the reasoning capabilities of language-specific LLMs while maintaining their target language abilities. DeepSeek R1 excels in reasoning but primarily benefits high-resource languages such as English and Chinese. However, low-resource languages remain underserved due to the dominance of English-centric training data and model optimizations, which limit performance in these languages. This limitation results in unreliable code-switching and diminished effectiveness on tasks in low-resource languages. Meanwhile, local and regional LLM initiatives have attempted to bridge this gap by developing language-specific LLMs that focus on improving local linguistic fidelity. We demonstrate that, with only publicly available datasets and a computational budget of $120, it is possible to enhance the reasoning capabilities of language-specific LLMs to match the level of DeepSeek R1, without compromising their performance on target language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09054v1">Cost-Saving LLM Cascades with Early Abstention</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      LLM cascades are based on the idea that processing all queries with the largest and most expensive LLMs is inefficient. Instead, cascades deploy small LLMs to answer the majority of queries, limiting the use of large and expensive LLMs to only the most difficult queries. This approach can significantly reduce costs without impacting performance. However, risk-sensitive domains such as finance or medicine place an additional premium on avoiding model errors. Recognizing that even the most expensive models may make mistakes, applications in these domains benefit from allowing LLM systems to completely abstain from answering a query when the chance of making a mistake is significant. However, giving a cascade the ability to abstain poses an immediate design question for LLM cascades: should abstention only be allowed at the final model or also at earlier models? Since the error patterns of small and large models are correlated, the latter strategy may further reduce inference costs by letting inexpensive models anticipate abstention decisions by expensive models, thereby obviating the need to run the expensive models. We investigate the benefits of "early abstention" in LLM cascades and find that it reduces the overall test loss by 2.2% on average across six benchmarks (GSM8K, MedMCQA, MMLU, TriviaQA, TruthfulQA, and XSum). These gains result from a more effective use of abstention, which trades a 4.1% average increase in the overall abstention rate for a 13.0% reduction in cost and a 5.0% reduction in error rate. Our findings demonstrate that it is possible to leverage correlations between the error patterns of different language models to drive performance improvements for LLM systems with abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01694v2">Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      This paper tackles the problem of video question answering (VideoQA), a task that often requires multi-step reasoning and a profound understanding of spatial-temporal dynamics. While large video-language models perform well on benchmarks, they often lack explainability and spatial-temporal grounding. In this paper, we propose Agent-of-Thoughts Distillation (AoTD), a method that enhances models by incorporating automatically generated Chain-of-Thoughts (CoTs) into the instruction-tuning process. Specifically, we leverage an agent-based system to decompose complex questions into sub-tasks, and address them with specialized vision models, the intermediate results are then treated as reasoning chains. We also introduce a verification mechanism using a large language model (LLM) to ensure the reliability of generated CoTs. Extensive experiments demonstrate that AoTD improves the performance on multiple-choice and open-ended benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00511v2">Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Preliminary work
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, single-shot inference often yields unreliable results for complex reasoning tasks, leading researchers to explore multiple reasoning paths through methods such as perplexity and self-consistency. In this paper, we present the first theoretical error decomposition analysis of these techniques, breaking down their error into estimation error and model error. Our analysis reveals a fundamental trade-off: perplexity methods suffer from substantial model error due to the absence of a proper consistency function, while self-consistency exhibits high estimation error due to a slow error convergence rate. To overcome these limitations, we propose Reasoning-Pruning Perplexity Consistency (RPC). This approach combines Perplexity Consistency, which seamlessly integrates LLM perplexity with self-consistency, and Reasoning Pruning, which eliminates low-probability reasoning paths to effectively prevent the degeneration of estimation error reduction. Theoretical analysis demonstrates that RPC not only accelerates the convergence rate of estimation error to an exponential level but also holds strong potential for further reducing model error. Extensive empirical evaluations on seven benchmark datasets confirm that RPC can significantly improve reasoning performance, sample efficiency, and confidence reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06635v2">Steel-LLM:From Scratch to Open Source -- A Personal Journey in Building a Chinese-Centric LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Steel-LLM is a Chinese-centric language model developed from scratch with the goal of creating a high-quality, open-source model despite limited computational resources. Launched in March 2024, the project aimed to train a 1-billion-parameter model on a large-scale dataset, prioritizing transparency and the sharing of practical insights to assist others in the community. The training process primarily focused on Chinese data, with a small proportion of English data included, addressing gaps in existing open-source LLMs by providing a more detailed and practical account of the model-building journey. Steel-LLM has demonstrated competitive performance on benchmarks such as CEVAL and CMMLU, outperforming early models from larger institutions. This paper provides a comprehensive summary of the project's key contributions, including data collection, model design, training methodologies, and the challenges encountered along the way, offering a valuable resource for researchers and practitioners looking to develop their own LLMs. The model checkpoints and training script are available at https://github.com/zhanshijinwat/Steel-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06572v2">LawGPT: Knowledge-Guided Data Generation and Its Application to Legal LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), both proprietary and open-source, have demonstrated remarkable capabilities across various natural language processing tasks. However, they face significant limitations in legal reasoning tasks. Proprietary models introduce data privacy risks and high inference costs, while open-source models underperform due to insufficient legal domain training data. To address these limitations, we study data generation for legal reasoning to improve the legal reasoning performance of open-source LLMs with the help of proprietary LLMs. This is challenging due to the lack of legal knowledge in proprietary LLMs and the difficulty in verifying the generated data. We propose KgDG, a knowledge-guided data generation framework for legal reasoning. Our framework enables leveraging legal knowledge to enhance generation diversity and introduces a refinement and verification process to ensure the quality of generated data. Moreover, we expand the generated dataset to further enhance the LLM reasoning capabilities. Using KgDG, we create a synthetic legal reasoning dataset containing 50K high-quality examples. Our trained model LawGPT outperforms existing legal-specific LLMs and achieves performance comparable to proprietary LLMs, demonstrating the effectiveness of KgDG and LawGPT. Our code and resources is publicly available at https://github.com/LAMDASZ-ML/Knowledge-Guide-Data-Generation .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09017v1">Diversity Enhances an LLM's Performance in RAG and Long-context Task</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have highlighted the challenge of context window limitations, primarily due to the quadratic time complexity of the self-attention mechanism (\(O(N^2)\), where \(N\) denotes the context window length). This constraint impacts tasks such as retrieval-augmented generation (RAG) in question answering (Q\&A) and long context summarization. A common approach involves selecting content with the highest similarity to the query; however, this often leads to redundancy and the exclusion of diverse yet relevant information. Building on principles from Maximal Marginal Relevance (MMR) and Farthest Point Sampling (FPS), we integrate diversity into the content selection process. Our findings reveal that incorporating diversity substantially increases the recall of selecting relevant sentences or chunks before LLM-based Q\&A and summarization. These results highlight the importance of maintaining diversity in future LLM applications to further improve summarization and Q\&A outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07987v2">Universal Adversarial Attack on Aligned Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Added an affiliation
    </div>
    <details class="paper-abstract">
      We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07058v2">Using Contextually Aligned Online Reviews to Measure LLMs' Performance Disparities Across Language Varieties</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted by 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), theme track
    </div>
    <details class="paper-abstract">
      A language can have different varieties. These varieties can affect the performance of natural language processing (NLP) models, including large language models (LLMs), which are often trained on data from widely spoken varieties. This paper introduces a novel and cost-effective approach to benchmark model performance across language varieties. We argue that international online review platforms, such as Booking.com, can serve as effective data sources for constructing datasets that capture comments in different language varieties from similar real-world scenarios, like reviews for the same hotel with the same rating using the same language (e.g., Mandarin Chinese) but different language varieties (e.g., Taiwan Mandarin, Mainland Mandarin). To prove this concept, we constructed a contextually aligned dataset comprising reviews in Taiwan Mandarin and Mainland Mandarin and tested six LLMs in a sentiment analysis task. Our results show that LLMs consistently underperform in Taiwan Mandarin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08954v1">Medicine on the Edge: Comparative Performance Analysis of On-Device LLMs for Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLM) on mobile devices offers significant potential for medical applications, enhancing privacy, security, and cost-efficiency by eliminating reliance on cloud-based services and keeping sensitive health data local. However, the performance and accuracy of on-device LLMs in real-world medical contexts remain underexplored. In this study, we benchmark publicly available on-device LLMs using the AMEGA dataset, evaluating accuracy, computational efficiency, and thermal limitation across various mobile devices. Our results indicate that compact general-purpose models like Phi-3 Mini achieve a strong balance between speed and accuracy, while medically fine-tuned models such as Med42 and Aloe attain the highest accuracy. Notably, deploying LLMs on older devices remains feasible, with memory constraints posing a greater challenge than raw processing power. Our study underscores the potential of on-device LLMs for healthcare while emphasizing the need for more efficient inference and models tailored to real-world clinical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08946v1">The Stochastic Parrot on LLM's Shoulder: A Summative Assessment of Physical Concept Understanding</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 NAACL 2025 Main Conference. First 5 authors contributed equally. Project page: https://physico-benchmark.github.io/
    </div>
    <details class="paper-abstract">
      In a systematic way, we investigate a widely asked question: Do LLMs really understand what they say?, which relates to the more familiar term Stochastic Parrot. To this end, we propose a summative assessment over a carefully designed physical concept understanding task, PhysiCo. Our task alleviates the memorization issue via the usage of grid-format inputs that abstractly describe physical phenomena. The grids represents varying levels of understanding, from the core phenomenon, application examples to analogies to other abstract patterns in the grid world. A comprehensive study on our task demonstrates: (1) state-of-the-art LLMs, including GPT-4o, o1 and Gemini 2.0 flash thinking, lag behind humans by ~40%; (2) the stochastic parrot phenomenon is present in LLMs, as they fail on our grid task but can describe and recognize the same concepts well in natural language; (3) our task challenges the LLMs due to intrinsic difficulties rather than the unfamiliar grid format, as in-context learning and fine-tuning on same formatted data added little to their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08923v1">CopySpec: Accelerating LLMs with Speculative Copy-and-Paste Without Compromising Quality</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 33 pages, 18 figures, 19 tables
    </div>
    <details class="paper-abstract">
      We introduce CopySpec, an innovative technique designed to tackle the inefficiencies LLMs face when generating responses that closely resemble previous outputs. CopySpec identifies repeated sequences in the model's chat history and speculates that the same tokens will follow, enabling seamless copying without compromising output quality or requiring additional GPU memory. To evaluate the effectiveness of our approach, we conducted experiments using five LLMs and five datasets: MT-Bench, CNN/DM, GSM-8K, HumanEval, and our newly created dataset, MT-Redundant. MT-Redundant, introduced in this paper, transforms the second turn of MT-Bench into a request for variations of the first turn's answer, simulating real-world scenarios where users request modifications to prior responses. Our results demonstrate significant speed-ups: up to 2.35x on CNN/DM, 3.08x on the second turn of select MT-Redundant categories, and 2.66x on the third turn of GSM-8K's self-correction tasks. Moreover, we show that CopySpec integrates seamlessly with speculative decoding, yielding an average 49% additional speed-up over speculative decoding for the second turn of MT-Redundant across all eight categories. While LLMs, even with speculative decoding, suffer from slower inference as context sizes grow, CopySpec leverages the expanded context to accelerate inference, making it faster as the context size increases. Our code and dataset are publicly available at https://github.com/RazvanDu/CopySpec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08920v1">Exploring Emotion-Sensitive LLM-Based Conversational AI</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 7 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      Conversational AI chatbots have become increasingly common within the customer service industry. Despite improvements in their emotional development, they often lack the authenticity of real customer service interactions or the competence of service providers. By comparing emotion-sensitive and emotion-insensitive LLM-based chatbots across 30 participants, we aim to explore how emotional sensitivity in chatbots influences perceived competence and overall customer satisfaction in service interactions. Additionally, we employ sentiment analysis techniques to analyze and interpret the emotional content of user inputs. We highlight that perceptions of chatbot trustworthiness and competence were higher in the case of the emotion-sensitive chatbot, even if issue resolution rates were not affected. We discuss implications of improved user satisfaction from emotion-sensitive chatbots and potential applications in support services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08909v1">Towards Automated Fact-Checking of Real-World Claims: Exploring Task Formulation and Assessment with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Fact-checking is necessary to address the increasing volume of misinformation. Traditional fact-checking relies on manual analysis to verify claims, but it is slow and resource-intensive. This study establishes baseline comparisons for Automated Fact-Checking (AFC) using Large Language Models (LLMs) across multiple labeling schemes (binary, three-class, five-class) and extends traditional claim verification by incorporating analysis, verdict classification, and explanation in a structured setup to provide comprehensive justifications for real-world claims. We evaluate Llama-3 models of varying sizes (3B, 8B, 70B) on 17,856 claims collected from PolitiFact (2007-2024) using evidence retrieved via restricted web searches. We utilize TIGERScore as a reference-free evaluation metric to score the justifications. Our results show that larger LLMs consistently outperform smaller LLMs in classification accuracy and justification quality without fine-tuning. We find that smaller LLMs in a one-shot scenario provide comparable task performance to fine-tuned Small Language Models (SLMs) with large context sizes, while larger LLMs consistently surpass them. Evidence integration improves performance across all models, with larger LLMs benefiting most. Distinguishing between nuanced labels remains challenging, emphasizing the need for further exploration of labeling schemes and alignment with evidences. Our findings demonstrate the potential of retrieval-augmented AFC with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09213v2">FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the advanced reasoning required for complex clinical scenarios, such as differential diagnosis or personalized treatment suggestions. We proposed FineMedLM-o1, which leverages high-quality synthetic medical data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduced Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also proposed a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08904v1">MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Recent methodologies utilizing synthetic datasets have aimed to address inconsistent hallucinations in large language models (LLMs); however,these approaches are primarily tailored to specific tasks, limiting their generalizability. Inspired by the strong performance of code-trained models in logic-intensive domains, we propose a novel framework that leverages event-based text to generate corresponding code and employs cyclic training to transfer the logical consistency of code to natural language effectively. Our method significantly reduces inconsistent hallucinations across three leading LLMs and two categories of natural language tasks while maintaining overall performance. This framework effectively alleviates hallucinations without necessitating adaptation to downstream tasks, demonstrating generality and providing new perspectives to tackle the challenge of inconsistent hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08896v1">Communication is All You Need: Persuasion Dataset Construction via Multi-LLM Communication</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted to NAACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown proficiency in generating persuasive dialogue, yet concerns about the fluency and sophistication of their outputs persist. This paper presents a multi-LLM communication framework designed to enhance the generation of persuasive data automatically. This framework facilitates the efficient production of high-quality, diverse linguistic content with minimal human oversight. Through extensive evaluations, we demonstrate that the generated data excels in naturalness, linguistic diversity, and the strategic use of persuasion, even in complex scenarios involving social taboos. The framework also proves adept at generalizing across novel contexts. Our results highlight the framework's potential to significantly advance research in both computational and social science domains concerning persuasive communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08888v1">LLM-Enhanced Multiple Instance Learning for Joint Rumor and Stance Detection with Social Context Information</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted by ACM TIST
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation, such as rumors on social media, has drawn significant attention, prompting various expressions of stance among users. Although rumor detection and stance detection are distinct tasks, they can complement each other. Rumors can be identified by cross-referencing stances in related posts, and stances are influenced by the nature of the rumor. However, existing stance detection methods often require post-level stance annotations, which are costly to obtain. We propose a novel LLM-enhanced MIL approach to jointly predict post stance and claim class labels, supervised solely by claim labels, using an undirected microblog propagation model. Our weakly supervised approach relies only on bag-level labels of claim veracity, aligning with multi-instance learning (MIL) principles. To achieve this, we transform the multi-class problem into multiple MIL-based binary classification problems. We then employ a discriminative attention layer to aggregate the outputs from these classifiers into finer-grained classes. Experiments conducted on three rumor datasets and two stance datasets demonstrate the effectiveness of our approach, highlighting strong connections between rumor veracity and expressed stances in responding posts. Our method shows promising performance in joint rumor and stance detection compared to the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05006v3">COAST: Enhancing the Code Debugging Ability of LLMs through Communicative Agent Based Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Code debugging is a vital stage of software development, essential for ensuring the reliability and performance of Large Language Models (LLMs) in the code generation task. Human debugging typically follows a multi-stage process, which includes Bug Localization, Bug Identification, Code Repair, and Code Recognition. However, existing code debugging benchmarks predominantly focus on the Code Repair stage, which offers only a limited perspective on evaluating the debugging capabilities of LLMs. In this paper, we introduce DEBUGEVAL, a comprehensive benchmark for evaluating the debugging abilities of LLMs by emulating the multi-stage human debugging process. Through evaluating on DEBUGEVAL, we observe that 7B-scale models consistently underperform compared to their larger counterparts, highlighting their limitations in comprehending code semantics. In this case, we propose the COmmunicative Agent-based data SynThesis (COAST) framework, which employs a multi-agent system to generate high-quality training data for supervised fine-tuning (SFT). Experimental results demonstrate that COAST-generated data outperform human-curated and GPT-4-generated data, enabling 7B-scale LLMs to achieve debugging performance comparable to GPT-3.5. All data and codes are available at https://github.com/NEUIR/COAST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v2">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15512v2">Reverse Question Answering: Can an LLM Write a Question so Hard (or Bad) that it Can't Answer?</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Question answering (QA), giving correct answers to questions, is a popular task, but we test reverse question answering (RQA): for an input answer, give a question with that answer. Past work tests QA and RQA separately, but we test them jointly, comparing their difficulty, aiding benchmark design, and checking reasoning consistency. We run 16 LLMs on QA and RQA with trivia questions/answers, revealing: 1) Versus QA, LLMs are much less accurate in RQA for numerical answers, but slightly more accurate in RQA for textual answers; 2) LLMs often answer their own invalid questions from RQA accurately in QA, so RQA errors are not from knowledge gaps alone; 3) RQA errors correlate with question difficulty and inversely correlate with answer frequencies in the Dolma corpus; and 4) LLMs struggle to provide valid multi-hop questions. By finding question and answer types that lead to RQA errors, we suggest improvements for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09113v2">Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which has been shown to successfully jailbreak multiple open-source LLMs. Drawing inspiration from the difficulties of discrete token optimization, our method relaxes the discrete jailbreak optimization into a continuous optimization process while gradually increasing the sparsity of the optimizing vectors. This technique effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than state-of-the-art token-level methods. On Harmbench, our approach achieves the highest attack success rate on seven out of eight LLMs compared to the latest jailbreak methods. Trigger Warning: This paper contains model behavior that can be offensive in nature.
    </details>
</div>
