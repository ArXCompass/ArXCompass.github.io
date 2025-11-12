# llm - 2024_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03158v1">CSS: Contrastive Semantic Similarity for Uncertainty Quantification of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 The paper is accepted by The Conference on Uncertainty in Artificial Intelligence (UAI), 2024
    </div>
    <details class="paper-abstract">
      Despite the impressive capability of large language models (LLMs), knowing when to trust their generations remains an open challenge. The recent literature on uncertainty quantification of natural language generation (NLG) utilises a conventional natural language inference (NLI) classifier to measure the semantic dispersion of LLMs responses. These studies employ logits of NLI classifier for semantic clustering to estimate uncertainty. However, logits represent the probability of the predicted class and barely contain feature information for potential clustering. Alternatively, CLIP (Contrastive Language-Image Pre-training) performs impressively in extracting image-text pair features and measuring their similarity. To extend its usability, we propose Contrastive Semantic Similarity, the CLIP-based feature extraction module to obtain similarity features for measuring uncertainty for text pairs. We apply this method to selective NLG, which detects and rejects unreliable generations for better trustworthiness of LLMs. We conduct extensive experiments with three LLMs on several benchmark question-answering datasets with comprehensive evaluation metrics. Results show that our proposed method performs better in estimating reliable responses of LLMs than comparable baselines. Results show that our proposed method performs better in estimating reliable responses of LLMs than comparable baselines. The code are available at \url{https://github.com/AoShuang92/css_uq_llms}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14883v3">Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      To support various applications, a prevalent and efficient approach for business owners is leveraging their valuable datasets to fine-tune a pre-trained LLM through the API provided by LLM owners or cloud servers. However, this process carries a substantial risk of model misuse, potentially resulting in severe economic consequences for business owners. Thus, safeguarding the copyright of these customized models during LLM fine-tuning has become an urgent practical requirement, but there are limited existing solutions to provide such protection. To tackle this pressing issue, we propose a novel watermarking approach named ``Double-I watermark''. Specifically, based on the instruct-tuning data, two types of backdoor data paradigms are introduced with trigger in the instruction and the input, respectively. By leveraging LLM's learning capability to incorporate customized backdoor samples into the dataset, the proposed approach effectively injects specific watermarking information into the customized model during fine-tuning, which makes it easy to inject and verify watermarks in commercial scenarios. We evaluate the proposed "Double-I watermark" under various fine-tuning methods, demonstrating its harmlessness, robustness, uniqueness, imperceptibility, and validity through both quantitative and qualitative analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15337v2">Ranking Entities along Conceptual Space Dimensions with LLMs: An Analysis of Fine-Tuning Strategies</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted in ACL 2024 (Findings)
    </div>
    <details class="paper-abstract">
      Conceptual spaces represent entities in terms of their primitive semantic features. Such representations are highly valuable but they are notoriously difficult to learn, especially when it comes to modelling perceptual and subjective features. Distilling conceptual spaces from Large Language Models (LLMs) has recently emerged as a promising strategy, but existing work has been limited to probing pre-trained LLMs using relatively simple zero-shot strategies. We focus in particular on the task of ranking entities according to a given conceptual space dimension. Unfortunately, we cannot directly fine-tune LLMs on this task, because ground truth rankings for conceptual space dimensions are rare. We therefore use more readily available features as training data and analyse whether the ranking capabilities of the resulting models transfer to perceptual and subjective features. We find that this is indeed the case, to some extent, but having at least some perceptual and subjective features in the training data seems essential for achieving the best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20835v3">Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Post-Training Quantization (PTQ) enhances the efficiency of Large Language Models (LLMs) by enabling faster operation and compatibility with more accessible hardware through reduced memory usage, at the cost of small performance drops. We explore the role of calibration sets in PTQ, specifically their effect on hidden activations in various notable open-source LLMs. Calibration sets are crucial for evaluating activation magnitudes and identifying outliers, which can distort the quantization range and negatively impact performance. Our analysis reveals a marked contrast in quantization effectiveness across models. The older OPT model, upon which much of the quantization literature is based, shows significant performance deterioration and high susceptibility to outliers with varying calibration sets. In contrast, newer models like Llama-2 7B, Llama-3 8B, Command-R 35B, and Mistral 7B demonstrate strong robustness, with Mistral 7B showing near-immunity to outliers and stable activations. These findings suggest a shift in PTQ strategies might be needed. As advancements in pre-training methods reduce the relevance of outliers, there is an emerging need to reassess the fundamentals of current quantization literature. The emphasis should pivot towards optimizing inference speed, rather than primarily focusing on outlier preservation, to align with the evolving characteristics of state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09836v2">Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      The powerful reasoning capabilities of large language models (LLMs) have brought revolutionary changes to many fields, but their performance in human behaviour generation has not yet been extensively explored. This gap likely emerges because the internal processes governing behavioral intentions cannot be solely explained by abstract reasoning. Instead, they are also influenced by a multitude of factors, including social norms and personal preference. Inspired by the Theory of Planned Behaviour (TPB), we develop a LLM workflow named Chain-of-Planned Behaviour (CoPB) for mobility behaviour generation, which reflects the important spatio-temporal dynamics of human activities. Through exploiting the cognitive structures of attitude, subjective norms, and perceived behaviour control in TPB, CoPB significantly enhance the ability of LLMs to reason the intention of next movement. Specifically, CoPB substantially reduces the error rate of mobility intention generation from 57.8% to 19.4%. To improve the scalability of the proposed CoPB workflow, we further explore the synergy between LLMs and mechanistic models. We find mechanistic mobility models, such as gravity model, can effectively map mobility intentions to physical mobility behaviours. The strategy of integrating CoPB with gravity model can reduce the token cost by 97.7% and achieve better performance simultaneously. Besides, the proposed CoPB workflow can facilitate GPT-4-turbo to automatically generate high quality labels for mobility behavior reasoning. We show such labels can be leveraged to fine-tune the smaller-scale, open source LLaMA 3-8B, which significantly reduces usage costs without sacrificing the quality of the generated behaviours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12532v2">PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted by ACL 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable comprehension abilities but face challenges in GPU memory usage during inference, hindering their scalability for real-time applications like chatbots. To accelerate inference, we store computed keys and values (KV cache) in the GPU memory. Existing methods study the KV cache compression to reduce memory by pruning the pre-computed KV cache. However, they neglect the inter-layer dependency between layers and huge memory consumption in pre-computation. To explore these deficiencies, we find that the number of crucial keys and values that influence future generations decreases layer by layer and we can extract them by the consistency in attention weights. Based on the findings, we propose PyramidInfer, a method that compresses the KV cache by layer-wise retaining crucial context. PyramidInfer saves significant memory by computing fewer keys and values without sacrificing performance. Experimental results show PyramidInfer improves 2.2x throughput compared to Accelerate with over 54% GPU memory reduction in KV cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03075v1">Towards Detecting LLMs Hallucination via Markov Chain-based Multi-agent Debate Framework</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 18 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has facilitated the development of natural language text generation. It also poses unprecedented challenges, with content hallucination emerging as a significant concern. Existing solutions often involve expensive and complex interventions during the training process. Moreover, some approaches emphasize problem disassembly while neglecting the crucial validation process, leading to performance degradation or limited applications. To overcome these limitations, we propose a Markov Chain-based multi-agent debate verification framework to enhance hallucination detection accuracy in concise claims. Our method integrates the fact-checking process, including claim detection, evidence retrieval, and multi-agent verification. In the verification stage, we deploy multiple agents through flexible Markov Chain-based debates to validate individual claims, ensuring meticulous verification outcomes. Experimental results across three generative tasks demonstrate that our approach achieves significant improvements over baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06587v1">Exploring Human-AI Perception Alignment in Sensory Experiences: Do LLMs Understand Textile Hand?</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) behaviour with human intent is critical for future AI. An important yet often overlooked aspect of this alignment is the perceptual alignment. Perceptual modalities like touch are more multifaceted and nuanced compared to other sensory modalities such as vision. This work investigates how well LLMs align with human touch experiences using the "textile hand" task. We created a "Guess What Textile" interaction in which participants were given two textile samples -- a target and a reference -- to handle. Without seeing them, participants described the differences between them to the LLM. Using these descriptions, the LLM attempted to identify the target textile by assessing similarity within its high-dimensional embedding space. Our results suggest that a degree of perceptual alignment exists, however varies significantly among different textile samples. For example, LLM predictions are well aligned for silk satin, but not for cotton denim. Moreover, participants didn't perceive their textile experiences closely matched by the LLM predictions. This is only the first exploration into perceptual alignment around touch, exemplified through textile hand. We discuss possible sources of this alignment variance, and how better human-AI perceptual alignment can benefit future everyday tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03030v1">From Tarzan to Tolkien: Controlling the Language Proficiency Level of LLMs for Content Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      We study the problem of controlling the difficulty level of text generated by Large Language Models (LLMs) for contexts where end-users are not fully proficient, such as language learners. Using a novel framework, we evaluate the effectiveness of several key approaches for this task, including few-shot prompting, supervised finetuning, and reinforcement learning (RL), utilising both GPT-4 and open source alternatives like LLama2-7B and Mistral-7B. Our findings reveal a large performance gap between GPT-4 and the open source models when using prompt-based strategies. However, we show how to bridge this gap with a careful combination of finetuning and RL alignment. Our best model, CALM (CEFR-Aligned Language Model), surpasses the performance of GPT-4 and other strategies, at only a fraction of the cost. We further validate the quality of our results through a small-scale human study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03007v1">BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted by ACL 2024
    </div>
    <details class="paper-abstract">
      With the prosperity of large language models (LLMs), powerful LLM-based intelligent agents have been developed to provide customized services with a set of user-defined tools. State-of-the-art methods for constructing LLM agents adopt trained LLMs and further fine-tune them on data for the agent task. However, we show that such methods are vulnerable to our proposed backdoor attacks named BadAgent on various agent tasks, where a backdoor can be embedded by fine-tuning on the backdoor data. At test time, the attacker can manipulate the deployed LLM agents to execute harmful operations by showing the trigger in the agent input or environment. To our surprise, our proposed attack methods are extremely robust even after fine-tuning on trustworthy data. Though backdoor attacks have been studied extensively in natural language processing, to the best of our knowledge, we could be the first to study them on LLM agents that are more dangerous due to the permission to use external tools. Our work demonstrates the clear risk of constructing LLM agents based on untrusted LLMs or data. Our code is public at https://github.com/DPamK/BadAgent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03003v1">Verified Code Transpilation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Domain-specific languages (DSLs) are integral to various software workflows. Such languages offer domain-specific optimizations and abstractions that improve code readability and maintainability. However, leveraging these languages requires developers to rewrite existing code using the specific DSL's API. While large language models (LLMs) have shown some success in automatic code transpilation, none of them provide any functional correctness guarantees on the transpiled code. Another approach for automating this task is verified lifting, which relies on program synthesis to find programs in the target language that are functionally equivalent to the source language program. While several verified lifting tools have been developed for various application domains, they are specialized for specific source-target languages or require significant expertise in domain knowledge to make the search efficient. In this paper, leveraging recent advances in LLMs, we propose an LLM-based approach (LLMLift) to building verified lifting tools. We use the LLM's capabilities to reason about programs to translate a given program into its corresponding equivalent in the target language. Additionally, we use LLMs to generate proofs for functional equivalence. We develop lifting-based compilers for {\em four different} DSLs targeting different application domains. Our approach not only outperforms previous symbolic-based tools in both the number of benchmarks transpiled and transpilation time, but also requires significantly less effort to build.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04247v4">Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Intelligent agents powered by large language models (LLMs) have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, these agents, called scientific LLM agents, also introduce novel vulnerabilities that demand careful consideration for safety. However, there exists a notable gap in the literature, as there has been no comprehensive exploration of these vulnerabilities. This perspective paper fills this gap by conducting a thorough examination of vulnerabilities in LLM-based agents within scientific domains, shedding light on potential risks associated with their misuse and emphasizing the need for safety measures. We begin by providing a comprehensive overview of the potential risks inherent to scientific LLM agents, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we delve into the origins of these vulnerabilities and provide a scoping review of the limited existing works. Based on our analysis, we propose a triadic framework involving human regulation, agent alignment, and an understanding of environmental feedback (agent regulation) to mitigate these identified risks. Furthermore, we highlight the limitations and challenges associated with safeguarding scientific agents and advocate for the development of improved models, robust benchmarks, and comprehensive regulations to address these issues effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02919v1">MultifacetEval: Multifaceted Evaluation to Probe LLMs in Mastering Medical Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted by IJCAI 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have excelled across domains, also delivering notable performance on the medical evaluation benchmarks, such as MedQA. However, there still exists a significant gap between the reported performance and the practical effectiveness in real-world medical scenarios. In this paper, we aim to explore the causes of this gap by employing a multifaceted examination schema to systematically probe the actual mastery of medical knowledge by current LLMs. Specifically, we develop a novel evaluation framework MultifacetEval to examine the degree and coverage of LLMs in encoding and mastering medical knowledge at multiple facets (comparison, rectification, discrimination, and verification) concurrently. Based on the MultifacetEval framework, we construct two multifaceted evaluation datasets: MultiDiseK (by producing questions from a clinical disease knowledge base) and MultiMedQA (by rephrasing each question from a medical benchmark MedQA into multifaceted questions). The experimental results on these multifaceted datasets demonstrate that the extent of current LLMs in mastering medical knowledge is far below their performance on existing medical benchmarks, suggesting that they lack depth, precision, and comprehensiveness in mastering medical knowledge. Consequently, current LLMs are not yet ready for application in real-world medical tasks. The codes and datasets are available at https://github.com/THUMLP/MultifacetEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02913v1">Zeroth-Order Fine-Tuning of LLMs with Extreme Sparsity</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Zeroth-order optimization (ZO) is a memory-efficient strategy for fine-tuning Large Language Models using only forward passes. However, the application of ZO fine-tuning in memory-constrained settings such as mobile phones and laptops is still challenging since full precision forward passes are infeasible. In this study, we address this limitation by integrating sparsity and quantization into ZO fine-tuning of LLMs. Specifically, we investigate the feasibility of fine-tuning an extremely small subset of LLM parameters using ZO. This approach allows the majority of un-tuned parameters to be quantized to accommodate the constraint of limited device memory. Our findings reveal that the pre-training process can identify a set of "sensitive parameters" that can guide the ZO fine-tuning of LLMs on downstream tasks. Our results demonstrate that fine-tuning 0.1% sensitive parameters in the LLM with ZO can outperform the full ZO fine-tuning performance, while offering wall-clock time speedup. Additionally, we show that ZO fine-tuning targeting these 0.1% sensitive parameters, combined with 4 bit quantization, enables efficient ZO fine-tuning of an Llama2-7B model on a GPU device with less than 8 GiB of memory and notably reduced latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04511v3">An LLM Compiler for Parallel Function Calling</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 ICML 2024
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of the recent LLMs enable them to execute external function calls to overcome their inherent limitations, such as knowledge cutoffs, poor arithmetic skills, or lack of access to private data. This development has allowed LLMs to select and coordinate multiple functions based on the context to tackle more complex problems. However, current methods for function calling often require sequential reasoning and acting for each function which can result in high latency, cost, and sometimes inaccurate behavior. To address this, we introduce LLMCompiler, which executes functions in parallel to efficiently orchestrate multiple function calls. Drawing inspiration from the principles of classical compilers, LLMCompiler enables parallel function calling with three components: (i) a Function Calling Planner, formulating execution plans for function calling; (ii) a Task Fetching Unit, dispatching function calling tasks; and (iii) an Executor, executing these tasks in parallel. LLMCompiler automatically generates an optimized orchestration for the function calls and can be used with both open-source and closed-source models. We have benchmarked LLMCompiler on a range of tasks with different patterns of function calling. We observe consistent latency speedup of up to 3.7x, cost savings of up to 6.7x, and accuracy improvement of up to ~9% compared to ReAct. Our code is available at https://github.com/SqueezeAILab/LLMCompiler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02863v1">LLM as a Scorer: The Impact of Output Order on Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Presented in AAAI 2024 Spring Symposium. The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      This research investigates the effect of prompt design on dialogue evaluation using large language models (LLMs). While LLMs are increasingly used for scoring various inputs, creating effective prompts for dialogue evaluation remains challenging due to model sensitivity and subjectivity in dialogue assessments. Our study experimented with different prompt structures, altering the sequence of output instructions and including explanatory reasons. We found that the order of presenting reasons and scores significantly influences LLMs' scoring, with a "reason-first" approach yielding more comprehensive evaluations. This insight is crucial for enhancing the accuracy and consistency of LLM-based evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.01943v1">Enhancing Trust in LLMs: Algorithms for Comparing and Interpreting LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 An extensive survey of the literature specifying algorithms and techniques enhancing the trustworthiness and understanding of Large Language Models (LLMs)
    </div>
    <details class="paper-abstract">
      This paper surveys evaluation techniques to enhance the trustworthiness and understanding of Large Language Models (LLMs). As reliance on LLMs grows, ensuring their reliability, fairness, and transparency is crucial. We explore algorithmic methods and metrics to assess LLM performance, identify weaknesses, and guide development towards more trustworthy applications. Key evaluation metrics include Perplexity Measurement, NLP metrics (BLEU, ROUGE, METEOR, BERTScore, GLEU, Word Error Rate, Character Error Rate), Zero-Shot and Few-Shot Learning Performance, Transfer Learning Evaluation, Adversarial Testing, and Fairness and Bias Evaluation. We introduce innovative approaches like LLMMaps for stratified evaluation, Benchmarking and Leaderboards for competitive assessment, Stratified Analysis for in-depth understanding, Visualization of Blooms Taxonomy for cognitive level accuracy distribution, Hallucination Score for quantifying inaccuracies, Knowledge Stratification Strategy for hierarchical analysis, and Machine Learning Models for Hierarchy Generation. Human Evaluation is highlighted for capturing nuances that automated metrics may miss. These techniques form a framework for evaluating LLMs, aiming to enhance transparency, guide development, and establish user trust. Future papers will describe metric visualization and demonstrate each approach on practical examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2211.04325v2">Will we run out of data? Limits of LLM scaling based on human-generated data</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      We investigate the potential constraints on LLM scaling posed by the availability of public human-generated text data. We forecast the growing demand for training data based on current trends and estimate the total stock of public human text data. Our findings indicate that if current LLM development trends continue, models will be trained on datasets roughly equal in size to the available stock of public human text data between 2026 and 2032, or slightly earlier if models are overtrained. We explore how progress in language modeling can continue when human-generated text datasets cannot be scaled any further. We argue that synthetic data generation, transfer learning from data-rich domains, and data efficiency improvements might support further progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05766v3">FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 NAACL 2024 (Camera Ready) (fixed minor typos)
    </div>
    <details class="paper-abstract">
      Planning is a crucial task for agents in task oriented dialogs (TODs). Human agents typically resolve user issues by following predefined workflows, decomposing workflow steps into actionable items, and performing actions by executing APIs in order; all of which require reasoning and planning. With the recent advances in LLMs, there have been increasing attempts to use them for task planning and API usage. However, the faithfulness of the plans to predefined workflows and API dependencies, is not guaranteed with LLMs. Moreover, workflows in real life are often custom-defined and prone to changes; hence, adaptation is desirable. To study this, we propose the problem of faithful planning in TODs that needs to resolve user intents by following predefined flows and preserving API dependencies. To solve this problem, we propose FLAP, a Flow-Adhering Planning algorithm based on constrained decoding with lookahead heuristic for LLMs. Our algorithm alleviates the need for finetuning LLMs using domain specific (plan/dependency) data, enables quick adaptation to predefined flows, and outperforms other decoding and prompting-based baselines. Further, our algorithm empowers smaller LLMs (7B) to perform at par larger LLMs (30B-40B).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03505v1">Dynamic and Adaptive Feature Generation with LLM</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      The representation of feature space is a crucial environment where data points get vectorized and embedded for upcoming modeling. Thus the efficacy of machine learning (ML) algorithms is closely related to the quality of feature engineering. As one of the most important techniques, feature generation transforms raw data into an optimized feature space conducive to model training and further refines the space. Despite the advancements in automated feature engineering and feature generation, current methodologies often suffer from three fundamental issues: lack of explainability, limited applicability, and inflexible strategy. These shortcomings frequently hinder and limit the deployment of ML models across varied scenarios. Our research introduces a novel approach adopting large language models (LLMs) and feature-generating prompts to address these challenges. We propose a dynamic and adaptive feature generation method that enhances the interpretability of the feature generation process. Our approach broadens the applicability across various data types and tasks and draws advantages over strategic flexibility. A broad range of experiments showcases that our approach is significantly superior to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12097v3">An Empirical Study of In-context Learning in LLMs for Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted to ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Recent interest has surged in employing Large Language Models (LLMs) for machine translation (MT) via in-context learning (ICL) (Vilar et al., 2023). Most prior studies primarily focus on optimizing translation quality, with limited attention to understanding the specific aspects of ICL that influence the said quality. To this end, we perform the first of its kind, an exhaustive study of in-context learning for machine translation. We first establish that ICL is primarily example-driven and not instruction-driven. Following this, we conduct an extensive exploration of various aspects of the examples to understand their influence on downstream performance. Our analysis includes factors such as quality and quantity of demonstrations, spatial proximity, and source versus target originality. Further, we also investigate challenging scenarios involving indirectness and misalignment of examples to understand the limits of ICL. While we establish the significance of the quality of the target distribution over the source distribution of demonstrations, we further observe that perturbations sometimes act as regularizers, resulting in performance improvements. Surprisingly, ICL does not necessitate examples from the same task, and a related task with the same target distribution proves sufficient. We hope that our study acts as a guiding resource for considerations in utilizing ICL for MT. Our code is available on https://github.com/PranjalChitale/in-context-mt-analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04887v2">AutoChip: Automating HDL Generation Using LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      Traditionally, designs are written in Verilog hardware description language (HDL) and debugged by hardware engineers. While this approach is effective, it is time-consuming and error-prone for complex designs. Large language models (LLMs) are promising in automating HDL code generation. LLMs are trained on massive datasets of text and code, and they can learn to generate code that compiles and is functionally accurate. We aim to evaluate the ability of LLMs to generate functionally correct HDL models. We build AutoChip by combining the interactive capabilities of LLMs and the output from Verilog simulations to generate Verilog modules. We start with a design prompt for a module and the context from compilation errors and debugging messages, which highlight differences between the expected and actual outputs. This ensures that accurate Verilog code can be generated without human intervention. We evaluate AutoChip using problem sets from HDLBits. We conduct a comprehensive analysis of the AutoChip using several LLMs and problem categories. The results show that incorporating context from compiler tools, such as Icarus Verilog, improves the effectiveness, yielding 24.20% more accurate Verilog. We release our evaluation scripts and datasets as open-source contributions at the following link https://github.com/shailja-thakur/AutoChip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00059v2">Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The complexity of large language model (LLM) serving workloads has substantially increased due to the integration with external tool invocations, such as ChatGPT plugins. In this paper, we identify a new opportunity for efficient LLM serving for requests that trigger tools: tool partial execution alongside LLM decoding. To this end, we design Conveyor, an efficient LLM serving system optimized for handling requests involving external tools. We introduce a novel interface for tool developers to expose partial execution opportunities to the LLM serving system and a request scheduler that facilitates partial tool execution. Our results demonstrate that tool partial execution can improve request completion latency by up to 38.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17197v2">Data-efficient Fine-tuning for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted by SIGIR 2024
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) for recommendation has recently garnered considerable attention, where fine-tuning plays a key role in LLMs' adaptation. However, the cost of fine-tuning LLMs on rapidly expanding recommendation data limits their practical application. To address this challenge, few-shot fine-tuning offers a promising approach to quickly adapt LLMs to new recommendation data. We propose the task of data pruning for efficient LLM-based recommendation, aimed at identifying representative samples tailored for LLMs' few-shot fine-tuning. While coreset selection is closely related to the proposed task, existing coreset selection methods often rely on suboptimal heuristic metrics or entail costly optimization on large-scale recommendation data. To tackle these issues, we introduce two objectives for the data pruning task in the context of LLM-based recommendation: 1) high accuracy aims to identify the influential samples that can lead to high overall performance; and 2) high efficiency underlines the low costs of the data pruning process. To pursue the two objectives, we propose a novel data pruning method based on two scores, i.e., influence score and effort score, to efficiently identify the influential samples. Particularly, the influence score is introduced to accurately estimate the influence of sample removal on the overall performance. To achieve low costs of the data pruning process, we use a small-sized surrogate model to replace LLMs to obtain the influence score. Considering the potential gap between the surrogate model and LLMs, we further propose an effort score to prioritize some hard samples specifically for LLMs. Empirical results on three real-world datasets validate the effectiveness of our proposed method. In particular, the proposed method uses only 2% samples to surpass the full data fine-tuning, reducing time costs by 97%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10693v3">Exploring Precision and Recall to assess the quality and diversity of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 21 pages, 15 figures, ACL 2024 Main
    </div>
    <details class="paper-abstract">
      We introduce a novel evaluation framework for Large Language Models (LLMs) such as \textsc{Llama-2} and \textsc{Mistral}, focusing on importing Precision and Recall metrics from image generation to text generation. This approach allows for a nuanced assessment of the quality and diversity of generated text without the need for aligned corpora. By conducting a comprehensive evaluation of state-of-the-art language models, the study reveals new insights into their performance on open-ended generation tasks, which are not adequately captured by traditional benchmarks. The findings highlight a trade-off between the quality and diversity of generated samples, particularly when models are fine-tuned on instruction dataset or with human feedback. This work extends the toolkit for distribution-based NLP evaluation, offering insights into the practical capabilities and challenges that current LLMs face in generating diverse and high-quality text. We release our code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08516v3">LLMs cannot find reasoning errors, but can correct them given the error location</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      While self-correction has shown promise in improving LLM outputs in terms of style and quality (e.g. Chen et al., 2023b; Madaan et al., 2023), recent attempts to self-correct logical or reasoning errors often cause correct answers to become incorrect, resulting in worse performances overall (Huang et al., 2023). In this paper, we show that poor self-correction performance stems from LLMs' inability to find logical mistakes, rather than their ability to correct a known mistake. Firstly, we benchmark several state-of-the-art LLMs on their mistake-finding ability and demonstrate that they generally struggle with the task, even in highly objective, unambiguous cases. Secondly, we test the correction abilities of LLMs -- separately from mistake finding -- using a backtracking setup that feeds ground truth mistake location information to the model. We show that this boosts downstream task performance across our 5 reasoning tasks, indicating that LLMs' correction abilities are robust. Finally, we show that it is possible to obtain mistake location information without ground truth labels or in-domain training data. We train a small classifier with out-of-domain data, which exhibits stronger mistake-finding performance than prompting a large model. We release our dataset of LLM-generated logical mistakes, BIG-Bench Mistake, to enable further research into locating LLM reasoning mistakes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15813v3">Measuring Bargaining Abilities of LLMs: A Benchmark and A Buyer-Enhancement Method</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted by ACL 2024 Findings. The dataset AmazonHistoryPrice and our code are available at https://github.com/TianXiaSJTU/AmazonPriceHistory
    </div>
    <details class="paper-abstract">
      Bargaining is an important and unique part of negotiation between humans. As LLM-driven agents learn to negotiate and act like real humans, how to evaluate agents' bargaining abilities remains an open problem. For the first time, we formally described the Bargaining task as an asymmetric incomplete information game, defining the gains of the Buyer and Seller in multiple bargaining processes. It allows us to quantitatively assess an agent's performance in the Bargain task. We collected a real product price dataset, AmazonHistoryPrice, and conducted evaluations of various LLM agents' bargaining abilities. We find that playing a Buyer is much harder than a Seller, and increasing model size can not effectively improve the Buyer's performance. To address the challenge, we propose a novel approach called OG-Narrator that integrates a deterministic Offer Generator to control the price range of Buyer's offers, and an LLM Narrator to create natural language sentences for generated offers. Experimental results show that OG-Narrator improves the buyer's deal rates from 26.67% to 88.88% and brings a ten times multiplication of profits on all baselines, even a model that has not been aligned.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19749v2">Generating Query Recommendations via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Generating Query Recommendations via LLMs
    </div>
    <details class="paper-abstract">
      Query recommendation systems are ubiquitous in modern search engines, assisting users in producing effective queries to meet their information needs. However, these systems require a large amount of data to produce good recommendations, such as a large collection of documents to index and query logs. In particular, query logs and user data are not available in cold start scenarios. Query logs are expensive to collect and maintain and require complex and time-consuming cascading pipelines for creating, combining, and ranking recommendations. To address these issues, we frame the query recommendation problem as a generative task, proposing a novel approach called Generative Query Recommendation (GQR). GQR uses an LLM as its foundation and does not require to be trained or fine-tuned to tackle the query recommendation problem. We design a prompt that enables the LLM to understand the specific recommendation task, even using a single example. We then improved our system by proposing a version that exploits query logs called Retriever-Augmented GQR (RA-GQR). RA-GQr dynamically composes its prompt by retrieving similar queries from query logs. GQR approaches reuses a pre-existing neural architecture resulting in a simpler and more ready-to-market approach, even in a cold start scenario. Our proposed GQR obtains state-of-the-art performance in terms of NDCG@10 and clarity score against two commercial search engines and the previous state-of-the-art approach on the Robust04 and ClueWeb09B collections, improving on average the NDCG@10 performance up to ~4% on Robust04 and ClueWeb09B w.r.t the previous best competitor. RA-GQR further improve the NDCG@10 obtaining an increase of ~11%, ~6\% on Robust04 and ClueWeb09B w.r.t the best competitor. Furthermore, our system obtained ~59% of user preferences in a blind user study, proving that our method produces the most engaging queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02060v1">I've got the "Answer"! Interpretation of LLMs Hidden States in Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted for NLDB-2024 conference
    </div>
    <details class="paper-abstract">
      Interpretability and explainability of AI are becoming increasingly important in light of the rapid development of large language models (LLMs). This paper investigates the interpretation of LLMs in the context of the knowledge-based question answering. The main hypothesis of the study is that correct and incorrect model behavior can be distinguished at the level of hidden states. The quantized models LLaMA-2-7B-Chat, Mistral-7B, Vicuna-7B and the MuSeRC question-answering dataset are used to test this hypothesis. The results of the analysis support the proposed hypothesis. We also identify the layers which have a negative effect on the model's behavior. As a prospect of practical application of the hypothesis, we propose to train such "weak" layers additionally in order to improve the quality of the task solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07677v2">ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 LLM+KG
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) and knowledge graphs (KGs) has achieved remarkable success in various natural language processing tasks. However, existing methodologies that integrate LLMs and KGs often navigate the task-solving process solely based on the LLM's analysis of the question, overlooking the rich cognitive potential inherent in the vast knowledge encapsulated in KGs. To address this, we introduce Observation-Driven Agent (ODA), a novel AI agent framework tailored for tasks involving KGs. ODA incorporates KG reasoning abilities via global observation, which enhances reasoning capabilities through a cyclical paradigm of observation, action, and reflection. Confronting the exponential explosion of knowledge during observation, we innovatively design a recursive observation mechanism. Subsequently, we integrate the observed knowledge into the action and reflection modules. Through extensive experiments, ODA demonstrates state-of-the-art performance on several datasets, notably achieving accuracy improvements of 12.87% and 8.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.16617v3">Efficient Detection of LLM-generated Texts with a Bayesian Surrogate Model</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      The detection of machine-generated text, especially from large language models (LLMs), is crucial in preventing serious social problems resulting from their misuse. Some methods train dedicated detectors on specific datasets but fall short in generalizing to unseen test data, while other zero-shot ones often yield suboptimal performance. Although the recent DetectGPT has shown promising detection performance, it suffers from significant inefficiency issues, as detecting a single candidate requires querying the source LLM with hundreds of its perturbations. This paper aims to bridge this gap. Concretely, we propose to incorporate a Bayesian surrogate model, which allows us to select typical samples based on Bayesian uncertainty and interpolate scores from typical samples to other samples, to improve query efficiency. Empirical results demonstrate that our method significantly outperforms existing approaches under a low query budget. Notably, when detecting the text generated by LLaMA family models, our method with just 2 or 3 queries can outperform DetectGPT with 200 queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02143v3">Competition-Level Problems are Effective LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities, yet there is ongoing debate about these abilities and the potential data contamination problem recently. This paper aims to evaluate the reasoning capacities of LLMs, specifically in solving recent competition-level programming problems in Codeforces, which are expert-crafted and unique, requiring deep understanding and robust reasoning skills. We first provide a comprehensive evaluation of GPT-4's peiceived zero-shot performance on this task, considering various aspects such as problems' release time, difficulties, and types of errors encountered. Surprisingly, the peiceived performance of GPT-4 has experienced a cliff like decline in problems after September 2021 consistently across all the difficulties and types of problems, which shows the potential data contamination, as well as the challenges for any existing LLM to solve unseen complex reasoning problems. We further explore various approaches such as fine-tuning, Chain-of-Thought prompting and problem description simplification, unfortunately none of them is able to consistently mitigate the challenges. Through our work, we emphasis the importance of this excellent data source for assessing the genuine reasoning capabilities of LLMs, and foster the development of LLMs with stronger reasoning abilities and better generalization in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15447v3">Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted to ICML'24
    </div>
    <details class="paper-abstract">
      Compressing high-capability Large Language Models (LLMs) has emerged as a favored strategy for resource-efficient inferences. While state-of-the-art (SoTA) compression methods boast impressive advancements in preserving benign task performance, the potential risks of compression in terms of safety and trustworthiness have been largely neglected. This study conducts the first, thorough evaluation of three (3) leading LLMs using five (5) SoTA compression techniques across eight (8) trustworthiness dimensions. Our experiments highlight the intricate interplay between compression and trustworthiness, revealing some interesting patterns. We find that quantization is currently a more effective approach than pruning in achieving efficiency and trustworthiness simultaneously. For instance, a 4-bit quantized model retains the trustworthiness of its original counterpart, but model pruning significantly degrades trustworthiness, even at 50% sparsity. Moreover, employing quantization within a moderate bit range could unexpectedly improve certain trustworthiness dimensions such as ethics and fairness. Conversely, extreme quantization to very low bit levels (3 bits) tends to reduce trustworthiness significantly. This increased risk cannot be uncovered by looking at benign performance alone, in turn, mandating comprehensive trustworthiness evaluation in practice. These findings culminate in practical recommendations for simultaneously achieving high utility, efficiency, and trustworthiness in LLMs. Code and models are available at https://decoding-comp-trust.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04396v2">QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 ICML 2024
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) reduces the memory footprint of LLMs by quantizing their weights to low-precision. In this work, we introduce QuIP#, a weight-only PTQ method that achieves state-of-the-art results in extreme compression regimes ($\le$ 4 bits per weight) using three novel techniques. First, QuIP# improves QuIP's (Chee et al., 2023) incoherence processing by using the randomized Hadamard transform, which is faster and has better theoretical properties. Second, QuIP# uses vector quantization to take advantage of the ball-shaped sub-Gaussian distribution that incoherent weights possess: specifically, we introduce a set of hardware-efficient codebooks based on the highly symmetric $E_8$ lattice, which achieves the optimal 8-dimension unit ball packing. Third, QuIP# uses fine-tuning to improve fidelity to the original model. Our experiments show that QuIP# outperforms existing PTQ methods, enables new behaviors in PTQ scaling, and supports fast inference. Our code can be found at https://github.com/Cornell-RelaxML/quip-sharp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01943v1">Enhancing Trust in LLMs: Algorithms for Comparing and Interpreting LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 An extensive survey of the literature specifying algorithms and techniques enhancing the trustworthiness and understanding of Large Language Models (LLMs)
    </div>
    <details class="paper-abstract">
      This paper surveys evaluation techniques to enhance the trustworthiness and understanding of Large Language Models (LLMs). As reliance on LLMs grows, ensuring their reliability, fairness, and transparency is crucial. We explore algorithmic methods and metrics to assess LLM performance, identify weaknesses, and guide development towards more trustworthy applications. Key evaluation metrics include Perplexity Measurement, NLP metrics (BLEU, ROUGE, METEOR, BERTScore, GLEU, Word Error Rate, Character Error Rate), Zero-Shot and Few-Shot Learning Performance, Transfer Learning Evaluation, Adversarial Testing, and Fairness and Bias Evaluation. We introduce innovative approaches like LLMMaps for stratified evaluation, Benchmarking and Leaderboards for competitive assessment, Stratified Analysis for in-depth understanding, Visualization of Blooms Taxonomy for cognitive level accuracy distribution, Hallucination Score for quantifying inaccuracies, Knowledge Stratification Strategy for hierarchical analysis, and Machine Learning Models for Hierarchy Generation. Human Evaluation is highlighted for capturing nuances that automated metrics may miss. These techniques form a framework for evaluating LLMs, aiming to enhance transparency, guide development, and establish user trust. Future papers will describe metric visualization and demonstrate each approach on practical examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16006v2">ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07092v3">Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Conversational search utilizes muli-turn natural language contexts to retrieve relevant passages. Existing conversational dense retrieval models mostly view a conversation as a fixed sequence of questions and responses, overlooking the severe data sparsity problem -- that is, users can perform a conversation in various ways, and these alternate conversations are unrecorded. Consequently, they often struggle to generalize to diverse conversations in real-world scenarios. In this work, we propose a framework for generalizing Conversational dense retrieval via LLM-cognition data Augmentation (ConvAug). ConvAug first generates multi-level augmented conversations to capture the diverse nature of conversational contexts. Inspired by human cognition, we devise a cognition-aware process to mitigate the generation of false positives, false negatives, and hallucinations. Moreover, we develop a difficulty-adaptive sample filter that selects challenging samples for complex conversations, thereby giving the model a larger learning space. A contrastive learning objective is then employed to train a better conversational context encoder. Extensive experiments conducted on four public datasets, under both normal and zero-shot settings, demonstrate the effectiveness, generalizability, and applicability of ConvAug. The code is released at https://github.com/haon-chen/ConvAug.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18721v2">PhysicsAssistant: An LLM-Powered Interactive Learning Robot for Physics Lab Investigations</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted to IEEE RO-MAN Special Session
    </div>
    <details class="paper-abstract">
      Robot systems in education can leverage Large language models' (LLMs) natural language understanding capabilities to provide assistance and facilitate learning. This paper proposes a multimodal interactive robot (PhysicsAssistant) built on YOLOv8 object detection, cameras, speech recognition, and chatbot using LLM to provide assistance to students' physics labs. We conduct a user study on ten 8th-grade students to empirically evaluate the performance of PhysicsAssistant with a human expert. The Expert rates the assistants' responses to student queries on a 0-4 scale based on Bloom's taxonomy to provide educational support. We have compared the performance of PhysicsAssistant (YOLOv8+GPT-3.5-turbo) with GPT-4 and found that the human expert rating of both systems for factual understanding is the same. However, the rating of GPT-4 for conceptual and procedural knowledge (3 and 3.2 vs 2.2 and 2.6, respectively) is significantly higher than PhysicsAssistant (p < 0.05). However, the response time of GPT-4 is significantly higher than PhysicsAssistant (3.54 vs 1.64 sec, p < 0.05). Hence, despite the relatively lower response quality of PhysicsAssistant than GPT-4, it has shown potential for being used as a real-time lab assistant to provide timely responses and can offload teachers' labor to assist with repetitive tasks. To the best of our knowledge, this is the first attempt to build such an interactive multimodal robotic assistant for K-12 science (physics) education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14928v2">Graph Machine Learning in the Era of Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      Graphs play an important role in representing complex relationships in various domains like social networks, knowledge graphs, and molecular discovery. With the advent of deep learning, Graph Neural Networks (GNNs) have emerged as a cornerstone in Graph Machine Learning (Graph ML), facilitating the representation and processing of graph structures. Recently, LLMs have demonstrated unprecedented capabilities in language tasks and are widely adopted in a variety of applications such as computer vision and recommender systems. This remarkable success has also attracted interest in applying LLMs to the graph domain. Increasing efforts have been made to explore the potential of LLMs in advancing Graph ML's generalization, transferability, and few-shot learning ability. Meanwhile, graphs, especially knowledge graphs, are rich in reliable factual knowledge, which can be utilized to enhance the reasoning capabilities of LLMs and potentially alleviate their limitations such as hallucinations and the lack of explainability. Given the rapid progress of this research direction, a systematic review summarizing the latest advancements for Graph ML in the era of LLMs is necessary to provide an in-depth understanding to researchers and practitioners. Therefore, in this survey, we first review the recent developments in Graph ML. We then explore how LLMs can be utilized to enhance the quality of graph features, alleviate the reliance on labeled data, and address challenges such as graph heterogeneity and out-of-distribution (OOD) generalization. Afterward, we delve into how graphs can enhance LLMs, highlighting their abilities to enhance LLM pre-training and inference. Furthermore, we investigate various applications and discuss the potential future directions in this promising field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01855v1">TruthEval: A Dataset to Evaluate LLM Truthfulness and Reliability</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) evaluation is currently one of the most important areas of research, with existing benchmarks proving to be insufficient and not completely representative of LLMs' various capabilities. We present a curated collection of challenging statements on sensitive topics for LLM benchmarking called TruthEval. These statements were curated by hand and contain known truth values. The categories were chosen to distinguish LLMs' abilities from their stochastic nature. We perform some initial analyses using this dataset and find several instances of LLMs failing in simple tasks showing their inability to understand simple questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.01789v2">The Political Preferences of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      I report here a comprehensive analysis about the political preferences embedded in Large Language Models (LLMs). Namely, I administer 11 political orientation tests, designed to identify the political preferences of the test taker, to 24 state-of-the-art conversational LLMs, both closed and open source. When probed with questions/statements with political connotations, most conversational LLMs tend to generate responses that are diagnosed by most political test instruments as manifesting preferences for left-of-center viewpoints. This does not appear to be the case for five additional base (i.e. foundation) models upon which LLMs optimized for conversation with humans are built. However, the weak performance of the base models at coherently answering the tests' questions makes this subset of results inconclusive. Finally, I demonstrate that LLMs can be steered towards specific locations in the political spectrum through Supervised Fine-Tuning (SFT) with only modest amounts of politically aligned data, suggesting SFT's potential to embed political orientation in LLMs. With LLMs beginning to partially displace traditional information sources like search engines and Wikipedia, the societal implications of political biases embedded in LLMs are substantial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17027v3">Player-Driven Emergence in LLM-Driven Game Narrative</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 Accepted at IEEE Conference on Games 2024
    </div>
    <details class="paper-abstract">
      We explore how interaction with large language models (LLMs) can give rise to emergent behaviors, empowering players to participate in the evolution of game narratives. Our testbed is a text-adventure game in which players attempt to solve a mystery under a fixed narrative premise, but can freely interact with non-player characters generated by GPT-4, a large language model. We recruit 28 gamers to play the game and use GPT-4 to automatically convert the game logs into a node-graph representing the narrative in the player's gameplay. We find that through their interactions with the non-deterministic behavior of the LLM, players are able to discover interesting new emergent nodes that were not a part of the original narrative but have potential for being fun and engaging. Players that created the most emergent nodes tended to be those that often enjoy games that facilitate discovery, exploration and experimentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20956v2">A Robot Walks into a Bar: Can Language Models Serve as Creativity Support Tools for Comedy? An Evaluation of LLMs' Humour Alignment with Comedians</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 15 pages, 1 figure, published at ACM FAccT 2024
    </div>
    <details class="paper-abstract">
      We interviewed twenty professional comedians who perform live shows in front of audiences and who use artificial intelligence in their artistic process as part of 3-hour workshops on ``AI x Comedy'' conducted at the Edinburgh Festival Fringe in August 2023 and online. The workshop consisted of a comedy writing session with large language models (LLMs), a human-computer interaction questionnaire to assess the Creativity Support Index of AI as a writing tool, and a focus group interrogating the comedians' motivations for and processes of using AI, as well as their ethical concerns about bias, censorship and copyright. Participants noted that existing moderation strategies used in safety filtering and instruction-tuned LLMs reinforced hegemonic viewpoints by erasing minority groups and their perspectives, and qualified this as a form of censorship. At the same time, most participants felt the LLMs did not succeed as a creativity support tool, by producing bland and biased comedy tropes, akin to ``cruise ship comedy material from the 1950s, but a bit less racist''. Our work extends scholarship about the subtle difference between, one the one hand, harmful speech, and on the other hand, ``offensive'' language as a practice of resistance, satire and ``punching up''. We also interrogate the global value alignment behind such language models, and discuss the importance of community-based value alignment and data ownership to build AI tools that better suit artists' needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01771v1">LLMs Beyond English: Scaling the Multilingual Capability of LLMs with Cross-Lingual Feedback</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 Accepted to Findings of ACL 2024. The code, datasets, and models are publicly available at https://github.com/boschresearch/ACL24-MLLM
    </div>
    <details class="paper-abstract">
      To democratize large language models (LLMs) to most natural languages, it is imperative to make these models capable of understanding and generating texts in many languages, in particular low-resource ones. While recent multilingual LLMs demonstrate remarkable performance in such capabilities, these LLMs still support a limited number of human languages due to the lack of training data for low-resource languages. Moreover, these LLMs are not yet aligned with human preference for downstream tasks, which is crucial for the success of LLMs in English. In this paper, we introduce xLLaMA-100 and xBLOOM-100 (collectively xLLMs-100), which scale the multilingual capabilities of LLaMA and BLOOM to 100 languages. To do so, we construct two datasets: a multilingual instruction dataset including 100 languages, which represents the largest language coverage to date, and a cross-lingual human feedback dataset encompassing 30 languages. We perform multilingual instruction tuning on the constructed instruction data and further align the LLMs with human feedback using the DPO algorithm on our cross-lingual human feedback dataset. We evaluate the multilingual understanding and generating capabilities of xLLMs-100 on five multilingual benchmarks. Experimental results show that xLLMs-100 consistently outperforms its peers across the benchmarks by considerable margins, defining a new state-of-the-art multilingual LLM that supports 100 languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01768v1">TSpec-LLM: An Open-source Dataset for LLM Understanding of 3GPP Specifications</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Understanding telecom standards involves sorting through numerous technical documents, such as those produced by the 3rd Generation Partnership Project (3GPP), which is time-consuming and labor-intensive. While large language models (LLMs) can assist with the extensive 3GPP knowledge base, an inclusive dataset is crucial for their effective pre-training and fine-tuning. In this paper, we introduce \textit{TSpec-LLM}, an open-source comprehensive dataset covering all 3GPP documents from Release 8 to Release 19 (1999--2023). To evaluate its efficacy, we first select a representative sample of 3GPP documents, create corresponding technical questions, and assess the baseline performance of various LLMs. We then incorporate a retrieval-augmented generation (RAG) framework to enhance LLM capabilities by retrieving relevant context from the \textit{TSpec-LLM} dataset. Our evaluation shows that using a naive-RAG framework on \textit{TSpec-LLM} improves the accuracy of GPT-3.5, Gemini 1.0 Pro, and GPT-4 from 44\%, 46\%, and 51\% to 71\%, 75\%, and 72\%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06575v1">Ask-EDA: A Design Assistant Empowered by LLM, Hybrid RAG and Abbreviation De-hallucination</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 Accepted paper at The First IEEE International Workshop on LLM-Aided Design, 2024 (LAD 24)
    </div>
    <details class="paper-abstract">
      Electronic design engineers are challenged to find relevant information efficiently for a myriad of tasks within design construction, verification and technology development. Large language models (LLM) have the potential to help improve productivity by serving as conversational agents that effectively function as subject-matter experts. In this paper we demonstrate Ask-EDA, a chat agent designed to serve as a 24x7 expert available to provide guidance to design engineers. Ask-EDA leverages LLM, hybrid retrieval augmented generation (RAG) and abbreviation de-hallucination (ADH) techniques to deliver more relevant and accurate responses. We curated three evaluation datasets, namely q2a-100, cmds-100 and abbr-100. Each dataset is tailored to assess a distinct aspect: general design question answering, design command handling and abbreviation resolution. We demonstrated that hybrid RAG offers over a 40% improvement in Recall on the q2a-100 dataset and over a 60% improvement on the cmds-100 dataset compared to not using RAG, while ADH yields over a 70% enhancement in Recall on the abbr-100 dataset. The evaluation results show that Ask-EDA can effectively respond to design-related inquiries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06574v1">Towards Transparency: Exploring LLM Trainings Datasets through Visual Topic Modeling and Semantic Frame</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      LLMs are now responsible for making many decisions on behalf of humans: from answering questions to classifying things, they have become an important part of everyday life. While computation and model architecture have been rapidly expanding in recent years, the efforts towards curating training datasets are still in their beginnings. This underappreciation of training datasets has led LLMs to create biased and low-quality content. In order to solve that issue, we present Bunka, a software that leverages AI and Cognitive Science to improve the refinement of textual datasets. We show how Topic Modeling coupled with 2-dimensional Cartography can increase the transparency of datasets. We then show how the same Topic Modeling techniques can be applied to Preferences datasets to accelerate the fine-tuning process and increase the capacities of the model on different benchmarks. Lastly, we show how using Frame Analysis can give insights into existing biases in the training corpus. Overall, we argue that we need better tools to explore and increase the quality and transparency of LLMs training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01698v1">Demystifying Platform Requirements for Diverse LLM Inference Use Cases</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 12 Pages, https://github.com/abhibambhaniya/GenZ-LLM-Analyzer
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across a wide range of applications, often outperforming human experts. However, deploying these parameter-heavy models efficiently for diverse inference use cases requires carefully designed hardware platforms with ample computing, memory, and network resources. With LLM deployment scenarios and models evolving at breakneck speed, the hardware requirements to meet SLOs remains an open research question. In this work, we present an analytical tool, GenZ, to study the relationship between LLM inference performance and various platform design parameters. Our analysis provides insights into configuring platforms for different LLM workloads and use cases. We quantify the platform requirements to support SOTA LLMs models like LLaMA and GPT-4 under diverse serving settings. Furthermore, we project the hardware capabilities needed to enable future LLMs potentially exceeding hundreds of trillions of parameters. The trends and insights derived from GenZ can guide AI engineers deploying LLMs as well as computer architects designing next-generation hardware accelerators and platforms. Ultimately, this work sheds light on the platform design considerations for unlocking the full potential of large language models across a spectrum of applications. The source code is available at https://github.com/abhibambhaniya/GenZ-LLM-Analyzer .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.04854v3">Are Language Models More Like Libraries or Like Librarians? Bibliotechnism, the Novel Reference Problem, and the Attitudes of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Are LLMs cultural technologies like photocopiers or printing presses, which transmit information but cannot create new content? A challenge for this idea, which we call bibliotechnism, is that LLMs generate novel text. We begin with a defense of bibliotechnism, showing how even novel text may inherit its meaning from original human-generated text. We then argue that bibliotechnism faces an independent challenge from examples in which LLMs generate novel reference, using new names to refer to new entities. Such examples could be explained if LLMs were not cultural technologies but had beliefs, desires, and intentions. According to interpretationism in the philosophy of mind, a system has such attitudes if and only if its behavior is well explained by the hypothesis that it does. Interpretationists may hold that LLMs have attitudes, and thus have a simple solution to the novel reference problem. We emphasize, however, that interpretationism is compatible with very simple creatures having attitudes and differs sharply from views that presuppose these attitudes require consciousness, sentience, or intelligence (topics about which we make no claims).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08277v5">Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Advances towards more faithful and traceable answers of Large Language Models (LLMs) are crucial for various research and practical endeavors. One avenue in reaching this goal is basing the answers on reliable sources. However, this Evidence-Based QA has proven to work insufficiently with LLMs in terms of citing the correct sources (source quality) and truthfully representing the information within sources (answer attributability). In this work, we systematically investigate how to robustly fine-tune LLMs for better source quality and answer attributability. Specifically, we introduce a data generation pipeline with automated data quality filters, which can synthesize diversified high-quality training and testing data at scale. We further introduce four test sets to benchmark the robustness of fine-tuned specialist models. Extensive evaluation shows that fine-tuning on synthetic data improves performance on both in- and out-of-distribution. Furthermore, we show that data quality, which can be drastically improved by proposed quality filters, matters more than quantity in improving Evidence-Based QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06270v3">XAI4LLM. Let Machine Learning Models and LLMs Collaborate for Enhanced In-Context Learning in Healthcare</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into healthcare diagnostics offers a promising avenue for clinical decision-making. This study outlines the development of a novel method for zero-shot/few-shot in-context learning (ICL) by integrating medical domain knowledge using a multi-layered structured prompt. We also explore the efficacy of two communication styles between the user and LLMs: the Numerical Conversational (NC) style, which processes data incrementally, and the Natural Language Single-Turn (NL-ST) style, which employs long narrative prompts. Our study systematically evaluates the diagnostic accuracy and risk factors, including gender bias and false negative rates, using a dataset of 920 patient records in various few-shot scenarios. Results indicate that traditional clinical machine learning (ML) models generally outperform LLMs in zero-shot and few-shot settings. However, the performance gap narrows significantly when employing few-shot examples alongside effective explainable AI (XAI) methods as sources of domain knowledge. Moreover, with sufficient time and an increased number of examples, the conversational style (NC) nearly matches the performance of ML models. Most notably, LLMs demonstrate comparable or superior cost-sensitive accuracy relative to ML models. This research confirms that, with appropriate domain knowledge and tailored communication strategies, LLMs can significantly enhance diagnostic processes. The findings highlight the importance of optimizing the number of training examples and communication styles to improve accuracy and reduce biases in LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01364v1">BELLS: A Framework Towards Future Proof Benchmarks for the Evaluation of LLM Safeguards</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Input-output safeguards are used to detect anomalies in the traces produced by Large Language Models (LLMs) systems. These detectors are at the core of diverse safety-critical applications such as real-time monitoring, offline evaluation of traces, and content moderation. However, there is no widely recognized methodology to evaluate them. To fill this gap, we introduce the Benchmarks for the Evaluation of LLM Safeguards (BELLS), a structured collection of tests, organized into three categories: (1) established failure tests, based on already-existing benchmarks for well-defined failure modes, aiming to compare the performance of current input-output safeguards; (2) emerging failure tests, to measure generalization to never-seen-before failure modes and encourage the development of more general safeguards; (3) next-gen architecture tests, for more complex scaffolding (such as LLM-agents and multi-agent systems), aiming to foster the development of safeguards that could adapt to future applications for which no safeguard currently exists. Furthermore, we implement and share the first next-gen architecture test, using the MACHIAVELLI environment, along with an interactive visualization of the dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01363v1">Privacy in LLM-based Recommendation: Recent Advances and Future Directions</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Nowadays, large language models (LLMs) have been integrated with conventional recommendation models to improve recommendation performance. However, while most of the existing works have focused on improving the model performance, the privacy issue has only received comparatively less attention. In this paper, we review recent advancements in privacy within LLM-based recommendation, categorizing them into privacy attacks and protection mechanisms. Additionally, we highlight several challenges and propose future directions for the community to address these critical problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06833v2">Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 GitHub: https://github.com/egozverev/Shold-It-Be-Executed-Or-Processed. 10 pages main text, 30 pages in total
    </div>
    <details class="paper-abstract">
      Instruction-tuned Large Language Models (LLMs) show impressive results in numerous practical applications, but they lack essential safety features that are common in other areas of computer science, particularly an explicit separation of instructions and data. This makes them vulnerable to manipulations such as indirect prompt injections and generally unsuitable for safety-critical tasks. Surprisingly, there is currently no established definition or benchmark to quantify this phenomenon. In this work, we close this gap by introducing a formal measure for instruction-data separation and an empirical variant that is calculable from a model's outputs. We also present a new dataset, SEP, that allows estimating the measure for real-world models. Our results on various LLMs show that the problem of instruction-data separation is real: all models fail to achieve high separation, and canonical mitigation techniques, such as prompt engineering and fine-tuning, either fail to substantially improve separation or reduce model utility. The source code and SEP dataset are openly accessible at https://github.com/egozverev/Shold-It-Be-Executed-Or-Processed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08045v4">Adversarial Preference Optimization: Enhancing Your Alignment via RM-LLM Game</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 Accepted by ACL2024 findings
    </div>
    <details class="paper-abstract">
      Human preference alignment is essential to improve the interaction quality of large language models (LLMs). Existing alignment methods depend on manually annotated preference data to guide the LLM optimization directions. However, continuously updating LLMs for alignment raises a distribution gap between model-generated samples and human-annotated responses, hindering training effectiveness. To mitigate this issue, previous methods require additional preference annotation on newly generated samples to adapt to the shifted distribution, which consumes a large amount of annotation resources. Targeting more efficient human preference optimization, we propose an Adversarial Preference Optimization (APO) framework, in which the LLM and the reward model update alternatively via a min-max game. Through adversarial training, the reward model can adapt to the shifted generation distribution of the LLM without any additional annotation. With comprehensive experiments, we find the proposed adversarial training framework further enhances existing alignment baselines in terms of LLM helpfulness and harmlessness. The code is at https://github.com/Linear95/APO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17167v3">Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 Accepted by ACL2024 Findings
    </div>
    <details class="paper-abstract">
      The recent trend of using Large Language Models (LLMs) as tool agents in real-world applications underscores the necessity for comprehensive evaluations of their capabilities, particularly in complex scenarios involving planning, creating, and using tools. However, existing benchmarks typically focus on simple synthesized queries that do not reflect real-world complexity, thereby offering limited perspectives in evaluating tool utilization. To address this issue, we present UltraTool, a novel benchmark designed to improve and evaluate LLMs' ability in tool utilization within real-world scenarios. UltraTool focuses on the entire process of using tools - from planning and creating to applying them in complex tasks. It emphasizes real-world complexities, demanding accurate, multi-step planning for effective problem-solving. A key feature of UltraTool is its independent evaluation of planning with natural language, which happens before tool usage and simplifies the task solving by mapping out the intermediate steps. Thus, unlike previous work, it eliminates the restriction of pre-defined toolset. Through extensive experiments on various LLMs, we offer novel insights into the evaluation of capabilities of LLMs in tool utilization, thereby contributing a fresh perspective to this rapidly evolving field. The benchmark is publicly available at https://github.com/JoeYing1019/UltraTool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01145v1">Explore then Determine: A GNN-LLM Synergy Framework for Reasoning over Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      The task of reasoning over Knowledge Graphs (KGs) poses a significant challenge for Large Language Models (LLMs) due to the complex structure and large amounts of irrelevant information. Existing LLM reasoning methods overlook the importance of compositional learning on KG to supply with precise knowledge. Besides, the fine-tuning and frequent interaction with LLMs incur substantial time and resource costs. This paper focuses on the Question Answering over Knowledge Graph (KGQA) task and proposes an Explore-then-Determine (EtD) framework that synergizes LLMs with graph neural networks (GNNs) for reasoning over KGs. The Explore stage employs a lightweight GNN to explore promising candidates and relevant fine-grained knowledge to the questions, while the Determine stage utilizes the explored information to construct a knowledge-enhanced multiple-choice prompt, guiding a frozen LLM to determine the final answer. Extensive experiments on three benchmark KGQA datasets demonstrate that EtD achieves state-of-the-art performance and generates faithful reasoning results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02613v1">ACCO: Accumulate while you Communicate, Hiding Communications in Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) relies heavily on distributed implementations, employing multiple GPUs to compute stochastic gradients on model replicas in parallel. However, synchronizing gradients in data parallel settings induces a communication overhead increasing with the number of distributed workers, which can impede the efficiency gains of parallelization. To address this challenge, optimization algorithms reducing inter-worker communication have emerged, such as local optimization methods used in Federated Learning. While effective in minimizing communication overhead, these methods incur significant memory costs, hindering scalability: in addition to extra momentum variables, if communications are only allowed between multiple local optimization steps, then the optimizer's states cannot be sharded among workers. In response, we propose $\textbf{AC}$cumulate while $\textbf{CO}$mmunicate ($\texttt{ACCO}$), a memory-efficient optimization algorithm tailored for distributed training of LLMs. $\texttt{ACCO}$ allows to shard optimizer states across workers, overlaps gradient computations and communications to conceal communication costs, and accommodates heterogeneous hardware. Our method relies on a novel technique to mitigate the one-step delay inherent in parallel execution of gradient computations and communications, eliminating the need for warmup steps and aligning with the training dynamics of standard distributed optimization while converging faster in terms of wall-clock time. We demonstrate the effectiveness of $\texttt{ACCO}$ on several LLMs training and fine-tuning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07688v2">CyberMetric: A Benchmark Dataset based on Retrieval-Augmented Generation for Evaluating LLMs in Cybersecurity Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used across various domains, from software development to cyber threat intelligence. Understanding all the different fields of cybersecurity, which includes topics such as cryptography, reverse engineering, and risk assessment, poses a challenge even for human experts. To accurately test the general knowledge of LLMs in cybersecurity, the research community needs a diverse, accurate, and up-to-date dataset. To address this gap, we present CyberMetric-80, CyberMetric-500, CyberMetric-2000, and CyberMetric-10000, which are multiple-choice Q&A benchmark datasets comprising 80, 500, 2000, and 10,000 questions respectively. By utilizing GPT-3.5 and Retrieval-Augmented Generation (RAG), we collected documents, including NIST standards, research papers, publicly accessible books, RFCs, and other publications in the cybersecurity domain, to generate questions, each with four possible answers. The results underwent several rounds of error checking and refinement. Human experts invested over 200 hours validating the questions and solutions to ensure their accuracy and relevance, and to filter out any questions unrelated to cybersecurity. We have evaluated and compared 25 state-of-the-art LLM models on the CyberMetric datasets. In addition to our primary goal of evaluating LLMs, we involved 30 human participants to solve CyberMetric-80 in a closed-book scenario. The results can serve as a reference for comparing the general cybersecurity knowledge of humans and LLMs. The findings revealed that GPT-4o, GPT-4-turbo, Mixtral-8x7B-Instruct, Falcon-180B-Chat, and GEMINI-pro 1.0 were the best-performing LLMs. Additionally, the top LLMs were more accurate than humans on CyberMetric-80, although highly experienced human experts still outperformed small models such as Llama-3-8B, Phi-2 or Gemma-7b.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04373v1">VerilogReader: LLM-Aided Hardware Test Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Test generation has been a critical and labor-intensive process in hardware design verification. Recently, the emergence of Large Language Model (LLM) with their advanced understanding and inference capabilities, has introduced a novel approach. In this work, we investigate the integration of LLM into the Coverage Directed Test Generation (CDG) process, where the LLM functions as a Verilog Reader. It accurately grasps the code logic, thereby generating stimuli that can reach unexplored code branches. We compare our framework with random testing, using our self-designed Verilog benchmark suite. Experiments demonstrate that our framework outperforms random testing on designs within the LLM's comprehension scope. Our work also proposes prompt engineering optimizations to augment LLM's understanding scope and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01045v1">Decompose, Enrich, and Extract! Schema-aware Event Extraction using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate significant capabilities in processing natural language data, promising efficient knowledge extraction from diverse textual sources to enhance situational awareness and support decision-making. However, concerns arise due to their susceptibility to hallucination, resulting in contextually inaccurate content. This work focuses on harnessing LLMs for automated Event Extraction, introducing a new method to address hallucination by decomposing the task into Event Detection and Event Argument Extraction. Moreover, the proposed method integrates dynamic schema-aware augmented retrieval examples into prompts tailored for each specific inquiry, thereby extending and adapting advanced prompting techniques such as Retrieval-Augmented Generation. Evaluation findings on prominent event extraction benchmarks and results from a synthesized benchmark illustrate the method's superior performance compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01032v1">LLM and GNN are Complementary: Distilling LLM for Multimodal Graph Learning</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Recent progress in Graph Neural Networks (GNNs) has greatly enhanced the ability to model complex molecular structures for predicting properties. Nevertheless, molecular data encompasses more than just graph structures, including textual and visual information that GNNs do not handle well. To bridge this gap, we present an innovative framework that utilizes multimodal molecular data to extract insights from Large Language Models (LLMs). We introduce GALLON (Graph Learning from Large Language Model Distillation), a framework that synergizes the capabilities of LLMs and GNNs by distilling multimodal knowledge into a unified Multilayer Perceptron (MLP). This method integrates the rich textual and visual data of molecules with the structural analysis power of GNNs. Extensive experiments reveal that our distilled MLP model notably improves the accuracy and efficiency of molecular property predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11930v2">Data Contamination Calibration for Black-box LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      The rapid advancements of Large Language Models (LLMs) tightly associate with the expansion of the training data size. However, the unchecked ultra-large-scale training sets introduce a series of potential risks like data contamination, i.e. the benchmark data is used for training. In this work, we propose a holistic method named Polarized Augment Calibration (PAC) along with a new to-be-released dataset to detect the contaminated data and diminish the contamination effect. PAC extends the popular MIA (Membership Inference Attack) -- from machine learning community -- by forming a more global target at detecting training data to Clarify invisible training data. As a pioneering work, PAC is very much plug-and-play that can be integrated with most (if not all) current white- and black-box LLMs. By extensive experiments, PAC outperforms existing methods by at least 4.5%, towards data contamination detection on more 4 dataset formats, with more than 10 base LLMs. Besides, our application in real-world scenarios highlights the prominent presence of contamination and related issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00476v3">TempCompass: Do Video LLMs Really Understand Videos?</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Recently, there is a surge in interest surrounding video large language models (Video LLMs). However, existing benchmarks fail to provide a comprehensive feedback on the temporal perception ability of Video LLMs. On the one hand, most of them are unable to distinguish between different temporal aspects (e.g., speed, direction) and thus cannot reflect the nuanced performance on these specific aspects. On the other hand, they are limited in the diversity of task formats (e.g., only multi-choice QA), which hinders the understanding of how temporal perception performance may vary across different types of tasks. Motivated by these two problems, we propose the \textbf{TempCompass} benchmark, which introduces a diversity of temporal aspects and task formats. To collect high-quality test data, we devise two novel strategies: (1) In video collection, we construct conflicting videos that share the same static content but differ in a specific temporal aspect, which prevents Video LLMs from leveraging single-frame bias or language priors. (2) To collect the task instructions, we propose a paradigm where humans first annotate meta-information for a video and then an LLM generates the instruction. We also design an LLM-based approach to automatically and accurately evaluate the responses from Video LLMs. Based on TempCompass, we comprehensively evaluate 8 state-of-the-art (SOTA) Video LLMs and 3 Image LLMs, and reveal the discerning fact that these models exhibit notably poor temporal perception ability. Our data will be available at https://github.com/llyx97/TempCompass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00969v1">Using RL to Identify Divisive Perspectives Improves LLMs Abilities to Identify Communities on Social Media</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      The large scale usage of social media, combined with its significant impact, has made it increasingly important to understand it. In particular, identifying user communities, can be helpful for many downstream tasks. However, particularly when models are trained on past data and tested on future, doing this is difficult. In this paper, we hypothesize to take advantage of Large Language Models (LLMs), to better identify user communities. Due to the fact that many LLMs, such as ChatGPT, are fixed and must be treated as black-boxes, we propose an approach to better prompt them, by training a smaller LLM to do this. We devise strategies to train this smaller model, showing how it can improve the larger LLMs ability to detect communities. Experimental results show improvements on Reddit and Twitter data, on the tasks of community detection, bot detection, and news media profiling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17233v2">CLAQ: Pushing the Limits of Low-Bit Post-Training Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      Parameter quantization for Large Language Models (LLMs) has attracted increasing attentions recently in reducing memory costs and improving computational efficiency. Early approaches have been widely adopted. However, the existing methods suffer from poor performance in low-bit (such as 2 to 3 bits) scenarios. In this paper, we present a novel and effective Column-Level Adaptive weight Quantization (CLAQ) framework by introducing three different types of adaptive strategies for LLM quantization. Firstly, a K-Means clustering based algorithm is proposed that allows dynamic generation of quantization centroids for each column of a parameter matrix. Secondly, we design an outlier-guided adaptive precision search strategy which can dynamically assign varying bit-widths to different columns. Finally, a dynamic outlier reservation scheme is developed to retain some parameters in their original float point precision, in trade off of boosted model performance. Experiments on various mainstream open source LLMs including LLaMA-1, LLaMA-2 and Yi demonstrate that our methods achieve the state-of-the-art results across different bit settings, especially in extremely low-bit scenarios. Code is available at https://github.com/fayuge/CLAQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00936v1">A Survey of Useful LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-06-03
    </div>
    <details class="paper-abstract">
      LLMs have gotten attention across various research domains due to their exceptional performance on a wide range of complex tasks. Therefore, refined methods to evaluate the capabilities of LLMs are needed to determine the tasks and responsibility they should undertake. Our study mainly discussed how LLMs, as useful tools, should be effectively assessed. We proposed the two-stage framework: from ``core ability'' to ``agent'', clearly explaining how LLMs can be applied based on their specific capabilities, along with the evaluation methods in each stage. Core ability refers to the capabilities that LLMs need in order to generate high-quality natural language texts. After confirming LLMs possess core ability, they can solve real-world and complex tasks as agent. In the "core ability" stage, we discussed the reasoning ability, societal impact, and domain knowledge of LLMs. In the ``agent'' stage, we demonstrated embodied action, planning, and tool learning of LLMs agent applications. Finally, we examined the challenges currently confronting the evaluation methods for LLMs, as well as the directions for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08417v4">Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 Accepted at ICML 2024
    </div>
    <details class="paper-abstract">
      Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation models, like ALMA, does not match the performance of state-of-the-art conventional encoder-decoder translation models or larger-scale LLMs such as GPT-4. In this study, we bridge this performance gap. We first assess the shortcomings of supervised fine-tuning for LLMs in the MT task, emphasizing the quality issues present in the reference data, despite being human-generated. Then, in contrast to SFT which mimics reference translations, we introduce Contrastive Preference Optimization (CPO), a novel approach that trains models to avoid generating adequate but not perfect translations. Applying CPO to ALMA models with only 22K parallel sentences and 12M parameters yields significant improvements. The resulting model, called ALMA-R, can match or exceed the performance of the WMT competition winners and GPT-4 on WMT'21, WMT'22 and WMT'23 test datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18672v2">LLM-based Hierarchical Concept Decomposition for Interpretable Fine-Grained Image Classification</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      (Renyi Qu's Master's Thesis) Recent advancements in interpretable models for vision-language tasks have achieved competitive performance; however, their interpretability often suffers due to the reliance on unstructured text outputs from large language models (LLMs). This introduces randomness and compromises both transparency and reliability, which are essential for addressing safety issues in AI systems. We introduce \texttt{Hi-CoDe} (Hierarchical Concept Decomposition), a novel framework designed to enhance model interpretability through structured concept analysis. Our approach consists of two main components: (1) We use GPT-4 to decompose an input image into a structured hierarchy of visual concepts, thereby forming a visual concept tree. (2) We then employ an ensemble of simple linear classifiers that operate on concept-specific features derived from CLIP to perform classification. Our approach not only aligns with the performance of state-of-the-art models but also advances transparency by providing clear insights into the decision-making process and highlighting the importance of various concepts. This allows for a detailed analysis of potential failure modes and improves model compactness, therefore setting a new benchmark in interpretability without compromising the accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02479v1">Applying Fine-Tuned LLMs for Reducing Data Needs in Load Profile Analysis</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      This paper presents a novel method for utilizing fine-tuned Large Language Models (LLMs) to minimize data requirements in load profile analysis, demonstrated through the restoration of missing data in power system load profiles. A two-stage fine-tuning strategy is proposed to adapt a pre-trained LLMs, i.e., GPT-3.5, for missing data restoration tasks. Through empirical evaluation, we demonstrate the effectiveness of the fine-tuned model in accurately restoring missing data, achieving comparable performance to state-of-the-art specifically designed models such as BERT-PIN. Key findings include the importance of prompt engineering and the optimal utilization of fine-tuning samples, highlighting the efficiency of few-shot learning in transferring knowledge from general user cases to specific target users. Furthermore, the proposed approach demonstrates notable cost-effectiveness and time efficiency compared to training models from scratch, making it a practical solution for scenarios with limited data availability and computing resources. This research has significant potential for application to other power system load profile analysis tasks. Consequently, it advances the use of LLMs in power system analytics, offering promising implications for enhancing the resilience and efficiency of power distribution systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14367v3">Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data</a></div>
    <div class="paper-meta">
      📅 2024-06-02
      | 💬 International Conference on Machine Learning (ICML), 2024
    </div>
    <details class="paper-abstract">
      Learning from preference labels plays a crucial role in fine-tuning large language models. There are several distinct approaches for preference fine-tuning, including supervised learning, on-policy reinforcement learning (RL), and contrastive learning. Different methods come with different implementation tradeoffs and performance differences, and existing empirical findings present different conclusions, for instance, some results show that online RL is quite important to attain good fine-tuning results, while others find (offline) contrastive or even purely supervised methods sufficient. This raises a natural question: what kind of approaches are important for fine-tuning with preference data and why? In this paper, we answer this question by performing a rigorous analysis of a number of fine-tuning techniques on didactic and full-scale LLM problems. Our main finding is that, in general, approaches that use on-policy sampling or attempt to push down the likelihood on certain responses (i.e., employ a "negative gradient") outperform offline and maximum likelihood objectives. We conceptualize our insights and unify methods that use on-policy sampling or negative gradient under a notion of mode-seeking objectives for categorical distributions. Mode-seeking objectives are able to alter probability mass on specific bins of a categorical distribution at a fast rate compared to maximum likelihood, allowing them to relocate masses across bins more effectively. Our analysis prescribes actionable insights for preference fine-tuning of LLMs and informs how data should be collected for maximal improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03507v2">GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection</a></div>
    <div class="paper-meta">
      📅 2024-06-02
      | 💬 ICML 2024 (Oral)
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) presents significant memory challenges, predominantly due to the growing size of weights and optimizer states. Common memory-reduction approaches, such as low-rank adaptation (LoRA), add a trainable low-rank matrix to the frozen pre-trained weight in each layer, reducing trainable parameters and optimizer states. However, such approaches typically underperform training with full-rank weights in both pre-training and fine-tuning stages since they limit the parameter search to a low-rank subspace and alter the training dynamics, and further, may require full-rank warm start. In this work, we propose Gradient Low-Rank Projection (GaLore), a training strategy that allows full-parameter learning but is more memory-efficient than common low-rank adaptation methods such as LoRA. Our approach reduces memory usage by up to 65.5% in optimizer states while maintaining both efficiency and performance for pre-training on LLaMA 1B and 7B architectures with C4 dataset with up to 19.7B tokens, and on fine-tuning RoBERTa on GLUE tasks. Our 8-bit GaLore further reduces optimizer memory by up to 82.5% and total training memory by 63.3%, compared to a BF16 baseline. Notably, we demonstrate, for the first time, the feasibility of pre-training a 7B model on consumer GPUs with 24GB memory (e.g., NVIDIA RTX 4090) without model parallel, checkpointing, or offloading strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12933v2">Skin-in-the-Game: Decision Making via Multi-Stakeholder Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-02
      | 💬 ACL 2024, long paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in tasks such as summarization, arithmetic reasoning, and question answering. However, they encounter significant challenges in the domain of moral reasoning and ethical decision-making, especially in complex scenarios with multiple stakeholders. This paper introduces the Skin-in-the-Game (SKIG) framework, aimed at enhancing moral reasoning in LLMs by exploring decisions' consequences from multiple stakeholder perspectives. Central to SKIG's mechanism is simulating accountability for actions, which, alongside empathy exercises and risk assessment, is pivotal to its effectiveness. We validate SKIG's performance across various moral reasoning benchmarks with proprietary and opensource LLMs, and investigate its crucial components through extensive ablation analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11073v3">AFaCTA: Assisting the Annotation of Factual Claim Detection with Reliable LLM Annotators</a></div>
    <div class="paper-meta">
      📅 2024-06-02
      | 💬 ACL2024 Main Conference
    </div>
    <details class="paper-abstract">
      With the rise of generative AI, automated fact-checking methods to combat misinformation are becoming more and more important. However, factual claim detection, the first step in a fact-checking pipeline, suffers from two key issues that limit its scalability and generalizability: (1) inconsistency in definitions of the task and what a claim is, and (2) the high cost of manual annotation. To address (1), we review the definitions in related work and propose a unifying definition of factual claims that focuses on verifiability. To address (2), we introduce AFaCTA (Automatic Factual Claim deTection Annotator), a novel framework that assists in the annotation of factual claims with the help of large language models (LLMs). AFaCTA calibrates its annotation confidence with consistency along three predefined reasoning paths. Extensive evaluation and experiments in the domain of political speech reveal that AFaCTA can efficiently assist experts in annotating factual claims and training high-quality classifiers, and can work with or without expert supervision. Our analyses also result in PoliClaim, a comprehensive claim detection dataset spanning diverse political topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14259v3">Let's Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Multi-modal Text Recognition</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      We introduce "Generative Fusion Decoding" (GFD), a novel shallow fusion framework, utilized to integrate Large Language Models (LLMs) into multi-modal text recognition systems such as automatic speech recognition (ASR) and optical character recognition (OCR). We derive the formulas necessary to enable GFD to operate across mismatched token spaces of different models by mapping text token space to byte token space, enabling seamless fusion during the decoding process. The framework is plug-and-play, compatible with various auto-regressive models, and does not require re-training for feature alignment, thus overcoming limitations of previous fusion techniques. We highlight three main advantages of GFD: First, by simplifying the complexity of aligning different model sample spaces, GFD allows LLMs to correct errors in tandem with the recognition model, reducing computation latencies. Second, the in-context learning ability of LLMs is fully capitalized by GFD, increasing robustness in long-form speech recognition and instruction aware speech recognition. Third, GFD enables fusing recognition models deficient in Chinese text recognition with LLMs extensively trained on Chinese. Our evaluation demonstrates that GFD significantly improves performance in ASR and OCR tasks, with ASR reaching state-of-the-art in the NTUML2021 benchmark. GFD provides a significant step forward in model integration, offering a unified solution that could be widely applicable to leveraging existing pre-trained models through step by step fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01637v1">Teams of LLM Agents can Exploit Zero-Day Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      LLM agents have become increasingly sophisticated, especially in the realm of cybersecurity. Researchers have shown that LLM agents can exploit real-world vulnerabilities when given a description of the vulnerability and toy capture-the-flag problems. However, these agents still perform poorly on real-world vulnerabilities that are unknown to the agent ahead of time (zero-day vulnerabilities). In this work, we show that teams of LLM agents can exploit real-world, zero-day vulnerabilities. Prior agents struggle with exploring many different vulnerabilities and long-range planning when used alone. To resolve this, we introduce HPTSA, a system of agents with a planning agent that can launch subagents. The planning agent explores the system and determines which subagents to call, resolving long-term planning issues when trying different vulnerabilities. We construct a benchmark of 15 real-world vulnerabilities and show that our team of agents improve over prior work by up to 4.5$\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.06782v2">PentestGPT: An LLM-empowered Automatic Penetration Testing Tool</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      Penetration testing, a crucial industrial practice for ensuring system security, has traditionally resisted automation due to the extensive expertise required by human professionals. Large Language Models (LLMs) have shown significant advancements in various domains, and their emergent abilities suggest their potential to revolutionize industries. In this research, we evaluate the performance of LLMs on real-world penetration testing tasks using a robust benchmark created from test machines with platforms. Our findings reveal that while LLMs demonstrate proficiency in specific sub-tasks within the penetration testing process, such as using testing tools, interpreting outputs, and proposing subsequent actions, they also encounter difficulties maintaining an integrated understanding of the overall testing scenario. In response to these insights, we introduce PentestGPT, an LLM-empowered automatic penetration testing tool that leverages the abundant domain knowledge inherent in LLMs. PentestGPT is meticulously designed with three self-interacting modules, each addressing individual sub-tasks of penetration testing, to mitigate the challenges related to context loss. Our evaluation shows that PentestGPT not only outperforms LLMs with a task-completion increase of 228.6\% compared to the \gptthree model among the benchmark targets but also proves effective in tackling real-world penetration testing challenges. Having been open-sourced on GitHub, PentestGPT has garnered over 4,700 stars and fostered active community engagement, attesting to its value and impact in both the academic and industrial spheres.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18628v2">Hardware-Aware Parallel Prompt Decoding for Memory-Efficient Acceleration of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-06-02
      | 💬 The code for this implementation is available at https://github.com/hmarkc/parallel-prompt-decoding
    </div>
    <details class="paper-abstract">
      The auto-regressive decoding of Large Language Models (LLMs) results in significant overheads in their hardware performance. While recent research has investigated various speculative decoding techniques for multi-token generation, these efforts have primarily focused on improving processing speed such as throughput. Crucially, they often neglect other metrics essential for real-life deployments, such as memory consumption and training cost. To overcome these limitations, we propose a novel parallel prompt decoding that requires only $0.0002$% trainable parameters, enabling efficient training on a single A100-40GB GPU in just 16 hours. Inspired by the human natural language generation process, $PPD$ approximates outputs generated at future timesteps in parallel by using multiple prompt tokens. This approach partially recovers the missing conditional dependency information necessary for multi-token generation, resulting in up to a 28% higher acceptance rate for long-range predictions. Furthermore, we present a hardware-aware dynamic sparse tree technique that adaptively optimizes this decoding scheme to fully leverage the computational capacities on different GPUs. Through extensive experiments across LLMs ranging from MobileLlama to Vicuna-13B on a wide range of benchmarks, our approach demonstrates up to 2.49$\times$ speedup and maintains a minimal runtime memory overhead of just $0.0004$%. More importantly, our parallel prompt decoding can serve as an orthogonal optimization for synergistic integration with existing speculative decoding, showing up to $1.22\times$ further speed improvement. Our code is available at https://github.com/hmarkc/parallel-prompt-decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00765v1">The Embodied World Model Based on LLM with Visual Information and Prediction-Oriented Prompts</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      In recent years, as machine learning, particularly for vision and language understanding, has been improved, research in embedded AI has also evolved. VOYAGER is a well-known LLM-based embodied AI that enables autonomous exploration in the Minecraft world, but it has issues such as underutilization of visual data and insufficient functionality as a world model. In this research, the possibility of utilizing visual data and the function of LLM as a world model were investigated with the aim of improving the performance of embodied AI. The experimental results revealed that LLM can extract necessary information from visual data, and the utilization of the information improves its performance as a world model. It was also suggested that devised prompts could bring out the LLM's function as a world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01789v2">The Political Preferences of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      I report here a comprehensive analysis about the political preferences embedded in Large Language Models (LLMs). Namely, I administer 11 political orientation tests, designed to identify the political preferences of the test taker, to 24 state-of-the-art conversational LLMs, both closed and open source. When probed with questions/statements with political connotations, most conversational LLMs tend to generate responses that are diagnosed by most political test instruments as manifesting preferences for left-of-center viewpoints. This does not appear to be the case for five additional base (i.e. foundation) models upon which LLMs optimized for conversation with humans are built. However, the weak performance of the base models at coherently answering the tests' questions makes this subset of results inconclusive. Finally, I demonstrate that LLMs can be steered towards specific locations in the political spectrum through Supervised Fine-Tuning (SFT) with only modest amounts of politically aligned data, suggesting SFT's potential to embed political orientation in LLMs. With LLMs beginning to partially displace traditional information sources like search engines and Wikipedia, the societal implications of political biases embedded in LLMs are substantial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08017v2">Lumos : Empowering Multimodal LLMs with Scene Text Recognition</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 Accepted to KDD 2024 (ADS Track)
    </div>
    <details class="paper-abstract">
      We introduce Lumos, the first end-to-end multimodal question-answering system with text understanding capabilities. At the core of Lumos is a Scene Text Recognition (STR) component that extracts text from first person point-of-view images, the output of which is used to augment input to a Multimodal Large Language Model (MM-LLM). While building Lumos, we encountered numerous challenges related to STR quality, overall latency, and model inference. In this paper, we delve into those challenges, and discuss the system architecture, design choices, and modeling techniques employed to overcome these obstacles. We also provide a comprehensive evaluation for each component, showcasing high quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00522v1">Wav2Prompt: End-to-End Speech Prompt Generation and Tuning For LLM in Zero and Few-shot Learning</a></div>
    <div class="paper-meta">
      📅 2024-06-01
    </div>
    <details class="paper-abstract">
      Wav2Prompt is proposed which allows straightforward integration between spoken input and a text-based large language model (LLM). Wav2Prompt uses a simple training process with only the same data used to train an automatic speech recognition (ASR) model. After training, Wav2Prompt learns continuous representations from speech and uses them as LLM prompts. To avoid task over-fitting issues found in prior work and preserve the emergent abilities of LLMs, Wav2Prompt takes LLM token embeddings as the training targets and utilises a continuous integrate-and-fire mechanism for explicit speech-text alignment. Therefore, a Wav2Prompt-LLM combination can be applied to zero-shot spoken language tasks such as speech translation (ST), speech understanding (SLU), speech question answering (SQA) and spoken-query-based QA (SQQA). It is shown that for these zero-shot tasks, Wav2Prompt performs similarly to an ASR-LLM cascade and better than recent prior work. If relatively small amounts of task-specific paired data are available in few-shot scenarios, the Wav2Prompt-LLM combination can be end-to-end (E2E) fine-tuned. The Wav2Prompt-LLM combination then yields greatly improved results relative to an ASR-LLM cascade for the above tasks. For instance, for English-French ST with the BLOOMZ-7B1 LLM, a Wav2Prompt-LLM combination gave a 8.5 BLEU point increase over an ASR-LLM cascade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02456v2">tnGPS: Discovering Unknown Tensor Network Structure Search Algorithms via Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 Accepted by ICML2024, pre-printed version
    </div>
    <details class="paper-abstract">
      Tensor networks are efficient for extremely high-dimensional representation, but their model selection, known as tensor network structure search (TN-SS), is a challenging problem. Although several works have targeted TN-SS, most existing algorithms are manually crafted heuristics with poor performance, suffering from the curse of dimensionality and local convergence. In this work, we jump out of the box, studying how to harness large language models (LLMs) to automatically discover new TN-SS algorithms, replacing the involvement of human experts. By observing how human experts innovate in research, we model their common workflow and propose an automatic algorithm discovery framework called tnGPS. The proposed framework is an elaborate prompting pipeline that instruct LLMs to generate new TN-SS algorithms through iterative refinement and enhancement. The experimental results demonstrate that the algorithms discovered by tnGPS exhibit superior performance in benchmarks compared to the current state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01633v1">On Overcoming Miscalibrated Conversational Priors in LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 Preprint of UAI'24 conference publication
    </div>
    <details class="paper-abstract">
      We explore the use of Large Language Model (LLM-based) chatbots to power recommender systems. We observe that the chatbots respond poorly when they encounter under-specified requests (e.g., they make incorrect assumptions, hedge with a long response, or refuse to answer). We conjecture that such miscalibrated response tendencies (i.e., conversational priors) can be attributed to LLM fine-tuning using annotators -- single-turn annotations may not capture multi-turn conversation utility, and the annotators' preferences may not even be representative of users interacting with a recommender system. We first analyze public LLM chat logs to conclude that query under-specification is common. Next, we study synthetic recommendation problems with configurable latent item utilities and frame them as Partially Observed Decision Processes (PODP). We find that pre-trained LLMs can be sub-optimal for PODPs and derive better policies that clarify under-specified queries when appropriate. Then, we re-calibrate LLMs by prompting them with learned control messages to approximate the improved policy. Finally, we show empirically that our lightweight learning approach effectively uses logged conversation data to re-calibrate the response strategies of LLM-based chatbots for recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07120v2">Position: Key Claims in LLM Research Have a Long Tail of Footnotes</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 ICML 2024 camera-ready (https://openreview.net/forum?id=M2cwkGleRL)
    </div>
    <details class="paper-abstract">
      Much of the recent discourse within the ML community has been centered around Large Language Models (LLMs), their functionality and potential -- yet not only do we not have a working definition of LLMs, but much of this discourse relies on claims and assumptions that are worth re-examining. We contribute a definition of LLMs, critically examine five common claims regarding their properties (including 'emergent properties'), and conclude with suggestions for future research directions and their framing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07572v1">Domain-specific ReAct for physics-integrated iterative modeling: A case study of LLM agents for gas path analysis of gas turbines</a></div>
    <div class="paper-meta">
      📅 2024-06-01
    </div>
    <details class="paper-abstract">
      This study explores the application of large language models (LLMs) with callable tools in energy and power engineering domain, focusing on gas path analysis of gas turbines. We developed a dual-agent tool-calling process to integrate expert knowledge, predefined tools, and LLM reasoning. We evaluated various LLMs, including LLama3, Qwen1.5 and GPT. Smaller models struggled with tool usage and parameter extraction, while larger models demonstrated favorable capabilities. All models faced challenges with complex, multi-component problems. Based on the test results, we infer that LLMs with nearly 100 billion parameters could meet professional scenario requirements with fine-tuning and advanced prompt design. Continued development are likely to enhance their accuracy and effectiveness, paving the way for more robust AI-driven solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00430v1">Evaluating Uncertainty-based Failure Detection for Closed-Loop LLM Planners</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 Accepted at ICRA 2024 Workshop on Back to the Future: Robot Learning Going Probabilistic. Website: https://sites.google.com/view/konwloop/home
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have witnessed remarkable performance as zero-shot task planners for robotic manipulation tasks. However, the open-loop nature of previous works makes LLM-based planning error-prone and fragile. On the other hand, failure detection approaches for closed-loop planning are often limited by task-specific heuristics or following an unrealistic assumption that the prediction is trustworthy all the time. As a general-purpose reasoning machine, LLMs or Multimodal Large Language Models (MLLMs) are promising for detecting failures. However, However, the appropriateness of the aforementioned assumption diminishes due to the notorious hullucination problem. In this work, we attempt to mitigate these issues by introducing a framework for closed-loop LLM-based planning called KnowLoop, backed by an uncertainty-based MLLMs failure detector, which is agnostic to any used MLLMs or LLMs. Specifically, we evaluate three different ways for quantifying the uncertainty of MLLMs, namely token probability, entropy, and self-explained confidence as primary metrics based on three carefully designed representative prompting strategies. With a self-collected dataset including various manipulation tasks and an LLM-based robot system, our experiments demonstrate that token probability and entropy are more reflective compared to self-explained confidence. By setting an appropriate threshold to filter out uncertain predictions and seek human help actively, the accuracy of failure detection can be significantly enhanced. This improvement boosts the effectiveness of closed-loop planning and the overall success rate of tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06556v1">Enhancing Presentation Slide Generation by LLMs with a Multi-Staged End-to-End Approach</a></div>
    <div class="paper-meta">
      📅 2024-06-01
    </div>
    <details class="paper-abstract">
      Generating presentation slides from a long document with multimodal elements such as text and images is an important task. This is time consuming and needs domain expertise if done manually. Existing approaches for generating a rich presentation from a document are often semi-automatic or only put a flat summary into the slides ignoring the importance of a good narrative. In this paper, we address this research gap by proposing a multi-staged end-to-end model which uses a combination of LLM and VLM. We have experimentally shown that compared to applying LLMs directly with state-of-the-art prompting, our proposed multi-staged solution is better in terms of automated metrics and human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14809v4">CriticBench: Benchmarking LLMs for Critique-Correct Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      The ability of Large Language Models (LLMs) to critique and refine their reasoning is crucial for their application in evaluation, feedback provision, and self-improvement. This paper introduces CriticBench, a comprehensive benchmark designed to assess LLMs' abilities to critique and rectify their reasoning across a variety of tasks. CriticBench encompasses five reasoning domains: mathematical, commonsense, symbolic, coding, and algorithmic. It compiles 15 datasets and incorporates responses from three LLM families. Utilizing CriticBench, we evaluate and dissect the performance of 17 LLMs in generation, critique, and correction reasoning, i.e., GQC reasoning. Our findings reveal: (1) a linear relationship in GQC capabilities, with critique-focused training markedly enhancing performance; (2) a task-dependent variation in correction effectiveness, with logic-oriented tasks being more amenable to correction; (3) GQC knowledge inconsistencies that decrease as model size increases; and (4) an intriguing inter-model critiquing dynamic, where stronger models are better at critiquing weaker ones, while weaker models can surprisingly surpass stronger ones in their self-critique. We hope these insights into the nuanced critique-correct reasoning of LLMs will foster further research in LLM critique and self-improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11343v2">Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 KDD 2024
    </div>
    <details class="paper-abstract">
      Collaborative filtering recommender systems (CF-RecSys) have shown successive results in enhancing the user experience on social media and e-commerce platforms. However, as CF-RecSys struggles under cold scenarios with sparse user-item interactions, recent strategies have focused on leveraging modality information of user/items (e.g., text or images) based on pre-trained modality encoders and Large Language Models (LLMs). Despite their effectiveness under cold scenarios, we observe that they underperform simple traditional collaborative filtering models under warm scenarios due to the lack of collaborative knowledge. In this work, we propose an efficient All-round LLM-based Recommender system, called A-LLMRec, that excels not only in the cold scenario but also in the warm scenario. Our main idea is to enable an LLM to directly leverage the collaborative knowledge contained in a pre-trained state-of-the-art CF-RecSys so that the emergent ability of the LLM as well as the high-quality user/item embeddings that are already trained by the state-of-the-art CF-RecSys can be jointly exploited. This approach yields two advantages: (1) model-agnostic, allowing for integration with various existing CF-RecSys, and (2) efficiency, eliminating the extensive fine-tuning typically required for LLM-based recommenders. Our extensive experiments on various real-world datasets demonstrate the superiority of A-LLMRec in various scenarios, including cold/warm, few-shot, cold user, and cross-domain scenarios. Beyond the recommendation task, we also show the potential of A-LLMRec in generating natural language outputs based on the understanding of the collaborative knowledge by performing a favorite genre prediction task. Our code is available at https://github.com/ghdtjr/A-LLMRec .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19616v2">Easy Problems That LLMs Get Wrong</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 AutogenAI Ltd. GitHub Repo: https://github.com/autogenai/easy-problems-that-llms-get-wrong
    </div>
    <details class="paper-abstract">
      We introduce a comprehensive Linguistic Benchmark designed to evaluate the limitations of Large Language Models (LLMs) in domains such as logical reasoning, spatial intelligence, and linguistic understanding, among others. Through a series of straightforward questions, it uncovers the significant limitations of well-regarded models to perform tasks that humans manage with ease. It also highlights the potential of prompt engineering to mitigate some errors and underscores the necessity for better training methodologies. Our findings stress the importance of grounding LLMs with human reasoning and common sense, emphasising the need for human-in-the-loop for enterprise applications. We hope this work paves the way for future research to enhance the usefulness and reliability of new models.
    </details>
</div>
