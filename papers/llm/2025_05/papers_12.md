# llm - 2025_05

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
- Part 12
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07372v1">Synthetic Code Surgery: Repairing Bugs and Vulnerabilities with LLMs and Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for enhancing Automated Program Repair (APR) through synthetic data generation utilizing Large Language Models (LLMs). Current APR systems are constrained by the limited availability of high-quality training data encompassing diverse bug types across multiple programming languages. The proposed approach addresses this limitation through a two-phase process: a synthetic sample generation followed by a rigorous quality assessment. Multiple state-of-the-art LLMs were employed to generate approximately 30,000 paired examples of buggy and fixed code across 12 programming languages and 13 bug categories. Subsequently, these samples underwent cross-model evaluation against five criteria: correctness, code quality, security, performance, and completeness. Experimental evaluation on the VulRepair test set dataset showed statistically significant improvements in Perfect Prediction rates, with the quality-filtered synthetic dataset outperforming both baseline and real-world commit data configurations in certain scenarios. The methodology was validated through rigorous statistical testing, including ANOVA and post-hoc Tukey's Honest Significant Difference analysis. Furthermore, the best-performing configurations surpassed existing systems despite using a less computationally intensive decoding strategy. This research establishes a self-bootstrapping paradigm in which LLMs generate and evaluate their own training data, potentially transforming approaches to data scarcity across software engineering tasks and advancing the development of robust, adaptable tools for automated code maintenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12275v2">Integrating Expert Knowledge into Logical Programs via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models' capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma3, Codestral and QwenCoder. The results reveal that most models generate nearly perfect syntactically correct code and exhibit strong performance in translating expert knowledge into correct code. At the same time, while most LLMs produce nearly flawless syntactic output, their ability to correctly implement logical rules varies, as does their capacity for self-improvement. Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07329v1">Private LoRA Fine-tuning of Open-Source LLMs with Homomorphic Encryption</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Preserving data confidentiality during the fine-tuning of open-source Large Language Models (LLMs) is crucial for sensitive applications. This work introduces an interactive protocol adapting the Low-Rank Adaptation (LoRA) technique for private fine-tuning. Homomorphic Encryption (HE) protects the confidentiality of training data and gradients handled by remote worker nodes performing the bulk of computations involving the base model weights. The data owner orchestrates training, requiring minimal local computing power and memory, thus alleviating the need for expensive client-side GPUs. We demonstrate feasibility by fine-tuning a Llama-3.2-1B model, presenting convergence results using HE-compatible quantization and performance benchmarks for HE computations on GPU hardware. This approach enables applications such as confidential knowledge base question answering, private codebase fine-tuning for AI code assistants, AI agents for drafting emails based on a company's email archive, and adapting models to analyze sensitive legal or healthcare documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05758v2">APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07309v1">Uncertainty Profiles for LLMs: Uncertainty Source Decomposition and Adaptive Model-Metric Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate fluent but factually incorrect outputs, known as hallucinations, which undermine their reliability in real-world applications. While uncertainty estimation has emerged as a promising strategy for detecting such errors, current metrics offer limited interpretability and lack clarity about the types of uncertainty they capture. In this paper, we present a systematic framework for decomposing LLM uncertainty into four distinct sources, inspired by previous research. We develop a source-specific estimation pipeline to quantify these uncertainty types and evaluate how existing metrics relate to each source across tasks and models. Our results show that metrics, task, and model exhibit systematic variation in uncertainty characteristic. Building on this, we propose a method for task specific metric/model selection guided by the alignment or divergence between their uncertainty characteristics and that of a given task. Our experiments across datasets and models demonstrate that our uncertainty-aware selection strategy consistently outperforms baseline strategies, helping us select appropriate models or uncertainty metrics, and contributing to more reliable and efficient deployment in uncertainty estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07289v1">Semantic Retention and Extreme Compression in LLMs: Can We Have Both?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Accepted for publication in the Proceedings of the 2025 International Joint Conference on Neural Networks (IJCNN); this arXiv version includes an appendix with 6 result tables; 10 pages, 15 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The exponential growth in Large Language Model (LLM) deployment has intensified the need for efficient model compression techniques to reduce computational and memory costs. While pruning and quantization have shown promise, their combined potential remains largely unexplored. In this paper, we examine joint compression and how strategically combining pruning and quantization could yield superior performance-to-compression ratios compared to single-method approaches. Recognizing the challenges in accurately assessing LLM performance, we address key limitations of previous evaluation frameworks and introduce the Semantic Retention Compression Rate (SrCr), a novel metric that quantifies the trade-off between model compression and semantic preservation, facilitating the optimization of pruning-quantization configurations. Experiments demonstrate that our recommended combination achieves, on average, a 20% performance increase compared to an equivalent quantization-only model at the same theoretical compression rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21098v2">Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Generative retrieval (GR) has revolutionized document retrieval with the advent of large language models (LLMs), and LLM-based GR is gradually being adopted by the industry. Despite its remarkable advantages and potential, LLM-based GR suffers from hallucination and generates documents that are irrelevant to the query in some instances, severely challenging its credibility in practical applications. We thereby propose an optimized GR framework designed to alleviate retrieval hallucination, which integrates knowledge distillation reasoning in model training and incorporate decision agent to further improve retrieval precision. Specifically, we employ LLMs to assess and reason GR retrieved query-document (q-d) pairs, and then distill the reasoning data as transferred knowledge to the GR model. Moreover, we utilize a decision agent as post-processing to extend the GR retrieved documents through retrieval model and select the most relevant ones from multi perspectives as the final generative retrieval result. Extensive offline experiments on real-world datasets and online A/B tests on Fund Search and Insurance Search in Alipay demonstrate our framework's superiority and effectiveness in improving search quality and conversion gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07274v1">Cache-Efficient Posterior Sampling for Reinforcement Learning with LLM-Derived Priors Across Discrete and Continuous Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) as priors in reinforcement learning (RL) offers significant advantages but comes with substantial computational costs. We present a principled cache-efficient framework for posterior sampling with LLM-derived priors that dramatically reduces these costs while maintaining high performance. At the core of our approach is an adaptive caching mechanism, where cache parameters are meta-optimized using surrogate gradients derived from policy performance. This design enables efficient inference across both discrete text environments (e.g., TextWorld, ALFWorld) and continuous control domains (e.g., MuJoCo), achieving a 3.8--4.7$\times$ reduction in LLM queries and 4.0--12.0$\times$ lower median latencies (85--93\,ms on a consumer GPU) while retaining 96--98\% of uncached performance. Our theoretical analysis provides KL divergence bounds on approximation quality, validated empirically. The framework extends to offline RL, where our CQL-Prior variant improves performance by 14--29\% and reduces training time by 38--40\%. Extensive evaluations across a diverse suite of eight tasks demonstrate the generalizability and practical viability of LLM-guided RL in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v6">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 40 pages, 38 figures An earlier revision of this paper was accepted at ICML 2025. Since then, it has been updated to include new results on training dynamics (4.7) and base models (4.8)
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding. It asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05712v2">LLM-Text Watermarking based on Lagrange Interpolation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs (Large Language Models) has established them as a foundational technology for many AI and ML-powered human computer interactions. A critical challenge in this context is the attribution of LLM-generated text -- either to the specific language model that produced it or to the individual user who embedded their identity via a so-called multi-bit watermark. This capability is essential for combating misinformation, fake news, misinterpretation, and plagiarism. One of the key techniques for addressing this challenge is digital watermarking. This work presents a watermarking scheme for LLM-generated text based on Lagrange interpolation, enabling the recovery of a multi-bit author identity even when the text has been heavily redacted by an adversary. The core idea is to embed a continuous sequence of points $(x, f(x))$ that lie on a single straight line. The $x$-coordinates are computed pseudorandomly using a cryptographic hash function $H$ applied to the concatenation of the previous token's identity and a secret key $s_k$. Crucially, the $x$-coordinates do not need to be embedded into the text -- only the corresponding $f(x)$ values are embedded. During extraction, the algorithm recovers the original points along with many spurious ones, forming an instance of the Maximum Collinear Points (MCP) problem, which can be solved efficiently. Experimental results demonstrate that the proposed method is highly effective, allowing the recovery of the author identity even when as few as three genuine points remain after adversarial manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19657v4">LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size and deployment scale, quantization has become an essential technique for reducing memory footprint and improving inference efficiency. However, existing quantization toolkits often lack transparency, flexibility, and system-level scalability across GPUs and distributed environments. We present \textbf{LLMEasyQuant}, a modular, system-aware quantization framework designed for efficient, low-bit inference of LLMs on single-node multi-GPU, multi-node, and edge hardware. LLMEasyQuant supports a wide range of quantization methods -- including Symmetric Quantization, ZeroQuant, SmoothQuant, and SimQuant -- with unified interfaces for per-layer calibration, bitwidth assignment, and runtime adaptation. It integrates fused CUDA kernels with NCCL-based distributed synchronization and supports both static and online quantization. Empirical results show that LLMEasyQuant can achieve substantial speedups in GEMM execution, HBM load time, and near-linear multi-GPU scaling. Ablation studies further validate its ability to balance latency, memory, and accuracy under diverse deployment conditions. LLMEasyQuant offers a practical quantization serving system for scalable, hardware-optimized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08418v3">Can LLMs advance democratic values?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      LLMs are among the most advanced tools ever devised for understanding and generating natural language. Democratic deliberation and decision-making involve, at several distinct stages, the production and comprehension of language. So it is natural to ask whether our best linguistic tools might prove instrumental to one of our most important tasks involving language. Researchers and practitioners have recently asked whether LLMs can support democratic deliberation by leveraging abilities to summarise content, to aggregate opinion over summarised content, and to represent voters by predicting their preferences over unseen choices. In this paper, we assess whether using LLMs to perform these and related functions really advances the democratic values behind these experiments. We suggest that the record is mixed. In the presence of background inequality of power and resources, as well as deep moral and political disagreement, we should not use LLMs to automate non-instrumentally valuable components of the democratic process, nor be tempted to supplant fair and transparent decision-making procedures that are practically necessary to reconcile competing interests and values. However, while LLMs should be kept well clear of formal democratic decision-making processes, we think they can instead strengthen the informal public sphere--the arena that mediates between democratic governments and the polities that they serve, in which political communities seek information, form civic publics, and hold their leaders to account.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02216v2">LLM-Guided Probabilistic Program Induction for POMDP Model Estimation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Partially Observable Markov Decision Processes (POMDPs) model decision making under uncertainty. While there are many approaches to approximately solving POMDPs, we aim to address the problem of learning such models. In particular, we are interested in a subclass of POMDPs wherein the components of the model, including the observation function, reward function, transition function, and initial state distribution function, can be modeled as low-complexity probabilistic graphical models in the form of a short probabilistic program. Our strategy to learn these programs uses an LLM as a prior, generating candidate probabilistic programs that are then tested against the empirical distribution and adjusted through feedback. We experiment on a number of classical toy POMDP problems, simulated MiniGrid domains, and two real mobile-base robotics search domains involving partial observability. Our results show that using an LLM to guide in the construction of a low-complexity POMDP model can be more effective than tabular POMDP learning, behavior cloning, or direct LLM planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07205v1">Benchmarking Ethical and Safety Risks of Healthcare LLMs in China-Toward Systemic Governance under Healthy China 2030</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are poised to transform healthcare under China's Healthy China 2030 initiative, yet they introduce new ethical and patient-safety challenges. We present a novel 12,000-item Q&A benchmark covering 11 ethics and 9 safety dimensions in medical contexts, to quantitatively evaluate these risks. Using this dataset, we assess state-of-the-art Chinese medical LLMs (e.g., Qwen 2.5-32B, DeepSeek), revealing moderate baseline performance (accuracy 42.7% for Qwen 2.5-32B) and significant improvements after fine-tuning on our data (up to 50.8% accuracy). Results show notable gaps in LLM decision-making on ethics and safety scenarios, reflecting insufficient institutional oversight. We then identify systemic governance shortfalls-including the lack of fine-grained ethical audit protocols, slow adaptation by hospital IRBs, and insufficient evaluation tools-that currently hinder safe LLM deployment. Finally, we propose a practical governance framework for healthcare institutions (embedding LLM auditing teams, enacting data ethics guidelines, and implementing safety simulation pipelines) to proactively manage LLM risks. Our study highlights the urgent need for robust LLM governance in Chinese healthcare, aligning AI innovation with patient safety and ethical standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13471v3">From Large to Super-Tiny: End-to-End Optimization for Cost-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced artificial intelligence by optimizing traditional Natural Language Processing (NLP) workflows, facilitating their integration into various systems. Many such NLP systems, including ours, directly incorporate LLMs. However, this approach either results in expensive costs or yields suboptimal performance after fine-tuning. In this paper, we introduce a three-stage cost-efficient end-to-end LLM deployment pipeline, comprising prototyping, knowledge transfer, and model compression, to effectively tackle the cost-performance dilemma in LLM-based frameworks. Its high cost-efficiency is manifested not only in simplifying system complexity and producing super-tiny online models with enhanced performance and reduced costs in the results, but also in addressing development cycle constraints, the lack of extensive high-quality data, and limited computational resources during the project development process. In the first stage, we construct an optimal performance prototype system by transforming complex tasks into a function call-based LLM-driven pipeline, which serves as a teacher model to generate high-quality data. In the second stage, we combine techniques like rejection sampling fine-tuning, reinforcement learning, and knowledge distillation to transfer knowledge to 0.5B student models, delivering effective performance at minimal cost. In the final stage, we further compress models to 0.4B via quantization and pruning, achieving ultra-low latency and cost. Extensive experimental results and the framework's modular design suggest cross-domain capabilities and potential applicability in other NLP areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07184v1">Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved unprecedented performance by leveraging vast pretraining corpora, yet their performance remains suboptimal in knowledge-intensive domains such as medicine and scientific research, where high factual precision is required. While synthetic data provides a promising avenue for augmenting domain knowledge, existing methods frequently generate redundant samples that do not align with the model's true knowledge gaps. To overcome this limitation, we propose a novel Structural Entropy-guided Knowledge Navigator (SENATOR) framework that addresses the intrinsic knowledge deficiencies of LLMs. Our approach employs the Structure Entropy (SE) metric to quantify uncertainty along knowledge graph paths and leverages Monte Carlo Tree Search (MCTS) to selectively explore regions where the model lacks domain-specific knowledge. Guided by these insights, the framework generates targeted synthetic data for supervised fine-tuning, enabling continuous self-improvement. Experimental results on LLaMA-3 and Qwen2 across multiple domain-specific benchmarks show that SENATOR effectively detects and repairs knowledge deficiencies, achieving notable performance improvements. The code and data for our methods and experiments are available at https://github.com/weiyifan1023/senator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01930v2">Efficient LLM Context Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate proficiency across diverse tasks but often require targeted adaptations for specific applications. Various methods have been proposed to facilitate this adaptation, including fewshot fine-tuning, in-context learning, and context distillation. This paper specifically investigates context distillation a method that extends the utility of task-specific examples by internalizing them, thus augmenting the example set accessible for model inference. We conduct a comparative analysis of context distillation with in-context learning (ICL) and few-shot fine-tuning (FT), aiming to ascertain the efficacy of context distillation in adapting models using minimal in-context examples. Employing matched datasets from Mobach, our experiments leverage OPT models of various sizes. The results indicate that context distillation effectively adapts models, with student models attaining comparable in-domain and out-of-domain accuracies to in-context learning. Although context distillation surpasses ICL in out-of-domain generalization, it does not achieve the performance levels of FT. However, the reduced dataset size and computational demands position context distillation as a viable alternative, especially for smaller datasets. Overall, this study presents context distillation as an efficient and potent method for customizing LLMs to specific tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16732v4">Perfectly to a Tee: Understanding User Perceptions of Personalized LLM-Enhanced Narrative Interventions</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Stories about overcoming personal struggles can effectively illustrate the application of psychological theories in real life, yet they may fail to resonate with individuals' experiences. In this work, we employ large language models (LLMs) to create tailored narratives that acknowledge and address unique challenging thoughts and situations faced by individuals. Our study, involving 346 young adults across two settings, demonstrates that personalized LLM-enhanced stories were perceived to be better than human-written ones in conveying key takeaways, promoting reflection, and reducing belief in negative thoughts. These stories were not only seen as more relatable but also similarly authentic to human-written ones, highlighting the potential of LLMs in helping young adults manage their struggles. The findings of this work provide crucial design considerations for future narrative-based digital mental health interventions, such as the need to maintain relatability without veering into implausibility and refining the wording and tone of AI-enhanced content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08106v1">Are LLMs complicated ethical dilemma analyzers?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 CS194-280 Advanced LLM Agents project. Project page: https://github.com/ALT-JS/ethicaLLM
    </div>
    <details class="paper-abstract">
      One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08083v1">LLMs to Support K-12 Teachers in Culturally Relevant Pedagogy: An AI Literacy Example</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Culturally Relevant Pedagogy (CRP) is vital in K-12 education, yet teachers struggle to implement CRP into practice due to time, training, and resource gaps. This study explores how Large Language Models (LLMs) can address these barriers by introducing CulturAIEd, an LLM tool that assists teachers in adapting AI literacy curricula to students' cultural contexts. Through an exploratory pilot with four K-12 teachers, we examined CulturAIEd's impact on CRP integration. Results showed CulturAIEd enhanced teachers' confidence in identifying opportunities for cultural responsiveness in learning activities and making culturally responsive modifications to existing activities. They valued CulturAIEd's streamlined integration of student demographic information, immediate actionable feedback, which could result in high implementation efficiency. This exploration of teacher-AI collaboration highlights how LLM can help teachers include CRP components into their instructional practices efficiently, especially in global priorities for future-ready education, such as AI literacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08063v1">Who's the Leader? Analyzing Novice Workflows in LLM-Assisted Debugging of Machine Learning Code</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Tools for Thought Workshop at CHI 2025
    </div>
    <details class="paper-abstract">
      While LLMs are often touted as tools for democratizing specialized knowledge to beginners, their actual effectiveness for improving task performance and learning is still an open question. It is known that novices engage with LLMs differently from experts, with prior studies reporting meta-cognitive pitfalls that affect novices' ability to verify outputs and prompt effectively. We focus on a task domain, machine learning (ML), which embodies both high complexity and low verifiability to understand the impact of LLM assistance on novices. Provided a buggy ML script and open access to ChatGPT, we conduct a formative study with eight novice ML engineers to understand their reliance on, interactions with, and perceptions of the LLM. We find that user actions can be roughly categorized into leading the LLM and led-by the LLM, and further investigate how they affect reliance outcomes like over- and under-reliance. These results have implications on novices' cognitive engagement in LLM-assisted tasks and potential negative effects on downstream learning. Lastly, we pose potential augmentations to the novice-LLM interaction paradigm to promote cognitive engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08054v1">FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04168v2">From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. The performance of LLM judges is typically evaluated by measuring the correlation with human judgments on generative tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that easy samples are easy to judge, and difficult samples are difficult to judge. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance, indicating that judges tend to favor higher-quality models even if their answer is incorrect. As a consequence, we test whether we can predict the behavior of LLM judges using simple features such as part-of-speech tags and find that we can correctly predict 70%-75% of judgments. We conclude this study by analyzing practical use cases, showing that LLM judges consistently detect the on-average better model but largely fail if we use them to improve task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01052v2">ALinFiK: Learning to Approximate Linearized Future Influence Kernel for Scalable Third-Party LLM Data Valuation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Proceedings of the NAACL 2025. Keywords: Influence Function, Data Valuation, Influence Estimation. https://aclanthology.org/2025.naacl-long.589/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) heavily rely on high-quality training data, making data valuation crucial for optimizing model performance, especially when working within a limited budget. In this work, we aim to offer a third-party data valuation approach that benefits both data providers and model developers. We introduce a linearized future influence kernel (LinFiK), which assesses the value of individual data samples in improving LLM performance during training. We further propose ALinFiK, a learning strategy to approximate LinFiK, enabling scalable data valuation. Our comprehensive evaluations demonstrate that this approach surpasses existing baselines in effectiveness and efficiency, demonstrating significant scalability advantages as LLM parameters increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04021v2">Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads. This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07128v2">DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 142 pages, pre-print
    </div>
    <details class="paper-abstract">
      Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs approach complex problems. Instead of directly producing an answer for a given input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly "thinking" about a problem before providing an answer. This reasoning process is publicly available to the user, creating endless opportunities for studying the reasoning behaviour of the model and opening up the field of Thoughtology. Starting from a taxonomy of DeepSeek-R1's basic building blocks of reasoning, our analyses on DeepSeek-R1 investigate the impact and controllability of thought length, management of long or confusing contexts, cultural and safety concerns, and the status of DeepSeek-R1 vis-\`a-vis cognitive phenomena, such as human-like language processing and world modelling. Our findings paint a nuanced picture. Notably, we show DeepSeek-R1 has a 'sweet spot' of reasoning, where extra inference time can impair model performance. Furthermore, we find a tendency for DeepSeek-R1 to persistently ruminate on previously explored problem formulations, obstructing further exploration. We also note strong safety vulnerabilities of DeepSeek-R1 compared to its non-reasoning counterpart, which can also compromise safety-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07897v1">LongCodeBench: Evaluating Coding LLMs at 1M Context Windows</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07793v1">Overflow Prevention Enhances Long-Context Recurrent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      A recent trend in LLMs is developing recurrent sub-quadratic models that improve long-context processing efficiency. We investigate leading large long-context models, focusing on how their fixed-size recurrent memory affects their performance. Our experiments reveal that, even when these models are trained for extended contexts, their use of long contexts remains underutilized. Specifically, we demonstrate that a chunk-based inference procedure, which identifies and processes only the most relevant portion of the input can mitigate recurrent memory failures and be effective for many long-context tasks: On LongBench, our method improves the overall performance of Falcon3-Mamba-Inst-7B by 14%, Falcon-Mamba-Inst-7B by 28%, RecurrentGemma-IT-9B by 50%, and RWKV6-Finch-7B by 51%. Surprisingly, this simple approach also leads to state-of-the-art results in the challenging LongBench v2 benchmark, showing competitive performance with equivalent size Transformers. Furthermore, our findings raise questions about whether recurrent models genuinely exploit long-range dependencies, as our single-chunk strategy delivers stronger performance - even in tasks that presumably require cross-context relations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v6">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07078v1">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07049v1">DialogueReason: Rule-Based RL Sparks Dialogue Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      We propose DialogueReason, a reasoning paradigm that uncovers the lost roles in monologue-style reasoning models, aiming to boost diversity and coherency of the reasoning process. Recent advances in RL-based large reasoning models have led to impressive long CoT capabilities and high performance on math and science benchmarks. However, these reasoning models rely mainly on monologue-style reasoning, which often limits reasoning diversity and coherency, frequently recycling fixed strategies or exhibiting unnecessary shifts in attention. Our work consists of an analysis of monologue reasoning patterns and the development of a dialogue-based reasoning approach. We first introduce the Compound-QA task, which concatenates multiple problems into a single prompt to assess both diversity and coherency of reasoning. Our analysis shows that Compound-QA exposes weaknesses in monologue reasoning, evidenced by both quantitative metrics and qualitative reasoning traces. Building on the analysis, we propose a dialogue-based reasoning, named DialogueReason, structured around agents, environment, and interactions. Using PPO with rule-based rewards, we train open-source LLMs (Qwen-QWQ and Qwen-Base) to adopt dialogue reasoning. We evaluate trained models on MATH, AIME, and GPQA datasets, showing that the dialogue reasoning model outperforms monologue models under more complex compound questions. Additionally, we discuss how dialogue-based reasoning helps enhance interpretability, facilitate more intuitive human interaction, and inspire advances in multi-agent system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07027v1">LLM-Augmented Chemical Synthesis and Design Decision Programs</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06972v1">Web Page Classification using LLMs for Crawling Support</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      A web crawler is a system designed to collect web pages, and efficient crawling of new pages requires appropriate algorithms. While website features such as XML sitemaps and the frequency of past page updates provide important clues for accessing new pages, their universal application across diverse conditions is challenging. In this study, we propose a method to efficiently collect new pages by classifying web pages into two types, "Index Pages" and "Content Pages," using a large language model (LLM), and leveraging the classification results to select index pages as starting points for accessing new pages. We construct a dataset with automatically annotated web page types and evaluate our approach from two perspectives: the page type classification performance and coverage of new pages. Experimental results demonstrate that the LLM-based method outperformed baseline methods in both evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06964v1">From Knowledge to Reasoning: Evaluating LLMs for Ionic Liquids Research in Chemical and Biological Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have achieved remarkable performance in diverse general knowledge and reasoning tasks, their utility in the scientific domain of Chemical and Biological Engineering (CBE) is unclear. Hence, it necessitates challenging evaluation benchmarks that can measure LLM performance in knowledge- and reasoning-based tasks, which is lacking. As a foundational step, we empirically measure the reasoning capabilities of LLMs in CBE. We construct and share an expert-curated dataset of 5,920 examples for benchmarking LLMs' reasoning capabilities in the niche domain of Ionic Liquids (ILs) for carbon sequestration, an emergent solution to reducing global warming. The dataset presents different difficulty levels by varying along the dimensions of linguistic and domain-specific knowledge. Benchmarking three less than 10B parameter open-source LLMs on the dataset suggests that while smaller general-purpose LLMs are knowledgeable about ILs, they lack domain-specific reasoning capabilities. Based on our results, we further discuss considerations for leveraging LLMs for carbon capture research using ILs. Since LLMs have a high carbon footprint, gearing them for IL research can symbiotically benefit both fields and help reach the ambitious carbon neutrality target by 2050. Dataset link: https://github.com/sougata-ub/llms_for_ionic_liquids
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06912v1">Building a Human-Verified Clinical Reasoning Dataset via a Human LLM Hybrid Pipeline for Trustworthy Medical AI</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      Despite strong performance in medical question-answering, the clinical adoption of Large Language Models (LLMs) is critically hampered by their opaque 'black-box' reasoning, limiting clinician trust. This challenge is compounded by the predominant reliance of current medical LLMs on corpora from scientific literature or synthetic data, which often lack the granular expert validation and high clinical relevance essential for advancing their specialized medical capabilities. To address these critical gaps, we introduce a highly clinically relevant dataset with 31,247 medical question-answer pairs, each accompanied by expert-validated chain-of-thought (CoT) explanations. This resource, spanning multiple clinical domains, was curated via a scalable human-LLM hybrid pipeline: LLM-generated rationales were iteratively reviewed, scored, and refined by medical experts against a structured rubric, with substandard outputs revised through human effort or guided LLM regeneration until expert consensus. This publicly available dataset provides a vital source for the development of medical LLMs that capable of transparent and verifiable reasoning, thereby advancing safer and more interpretable AI in medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06901v1">Ecco: Improving Memory Bandwidth and Capacity for LLMs via Entropy-aware Cache Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 ISCA 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated transformative capabilities across diverse artificial intelligence applications, yet their deployment is hindered by substantial memory and computational demands, especially in resource-constrained environments. Quantization techniques have emerged as a critical solution, reducing data precision to enhance memory and computational efficiency. However, existing methods often suffer from high runtime overheads and potential accuracy degradation. To address these challenges, we propose Ecco, an entropy-based cache compression technique tailored for LLMs. Ecco combines group-wise and non-uniform quantization with pre-defined shared k-means patterns and Huffman coding to exploit the inherent entropy characteristics of LLM cache data. Recognizing the inefficiencies of traditional Huffman coding in terms of parallelism and latency, we introduce a novel parallel Huffman-based decoding process with a multi-stage pipeline design, reducing latency by two orders of magnitude and achieving throughput comparable to GPU L2 caches. Comprehensive evaluations demonstrate that Ecco achieves an up to 2.9$\times$ and 1.9$\times$ speedup over the state-of-the-art AWQ and SmoothQuant framework, 2.4$\times$ over the Olive accelerator, all while increasing memory capacity by nearly 4$\times$ and maintaining state-of-the-art LLM accuracy. These results underscore the effectiveness of our entropy-based cache compression in enhancing LLM performance and efficiency, paving the way for more deployable large-scale AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10424v5">CodeV: Empowering LLMs with HDL Generation through Multi-Level Summarization</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 13 pages, 10 figures, journal
    </div>
    <details class="paper-abstract">
      The design flow of processors, particularly in hardware description languages (HDL) like Verilog and Chisel, is complex and costly. While recent advances in large language models (LLMs) have significantly improved coding tasks in software languages such as Python, their application in HDL generation remains limited due to the scarcity of high-quality HDL data. Traditional methods of adapting LLMs for hardware design rely on synthetic HDL datasets, which often suffer from low quality because even advanced LLMs like GPT perform poorly in the HDL domain. Moreover, these methods focus solely on chat tasks and the Verilog language, limiting their application scenarios. In this paper, we observe that: (1) HDL code collected from the real world is of higher quality than code generated by LLMs. (2) LLMs like GPT-3.5 excel in summarizing HDL code rather than generating it. (3) An explicit language tag can help LLMs better adapt to the target language when there is insufficient data. Based on these observations, we propose an efficient LLM fine-tuning pipeline for HDL generation that integrates a multi-level summarization data synthesis process with a novel Chat-FIM-Tag supervised fine-tuning method. The pipeline enhances the generation of HDL code from natural language descriptions and enables the handling of various tasks such as chat and infilling incomplete code. Utilizing this pipeline, we introduce CodeV, a series of HDL generation LLMs. Among them, CodeV-All not only possesses a more diverse range of language abilities, i.e. Verilog and Chisel, and a broader scope of tasks, i.e. Chat and fill-in-middle (FIM), but it also achieves performance on VerilogEval that is comparable to or even surpasses that of CodeV-Verilog fine-tuned on Verilog only, making them the first series of open-source LLMs designed for multi-scenario HDL generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06821v1">ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 This paper has been presented at IEEE VLSI Test Symposium (VTS) 2025
    </div>
    <details class="paper-abstract">
      Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07888v1">Implementing Long Text Style Transfer with LLMs through Dual-Layered Sentence and Paragraph Structure Extraction and Mapping</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge in long-text style transfer using zero-shot learning of large language models (LLMs), proposing a hierarchical framework that combines sentence-level stylistic adaptation with paragraph-level structural coherence. We argue that in the process of effective paragraph-style transfer, to preserve the consistency of original syntactic and semantic information, it is essential to perform style transfer not only at the sentence level but also to incorporate paragraph-level semantic considerations, while ensuring structural coherence across inter-sentential relationships. Our proposed framework, ZeroStylus, operates through two systematic phases: hierarchical template acquisition from reference texts and template-guided generation with multi-granular matching. The framework dynamically constructs sentence and paragraph template repositories, enabling context-aware transformations while preserving inter-sentence logical relationships. Experimental evaluations demonstrate significant improvements over baseline methods, with structured rewriting achieving 6.90 average score compared to 6.70 for direct prompting approaches in tri-axial metrics assessing style consistency, content preservation, and expression quality. Ablation studies validate the necessity of both template hierarchies during style transfer, showing higher content preservation win rate against sentence-only approaches through paragraph-level structural encoding, as well as direct prompting method through sentence-level pattern extraction and matching. The results establish new capabilities for coherent long-text style transfer without requiring parallel corpora or LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08067v6">Reward-Augmented Data Enhances Direct Preference Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Preference alignment in Large Language Models (LLMs) has significantly improved their ability to adhere to human instructions and intentions. However, existing direct alignment algorithms primarily focus on relative preferences and often overlook the qualitative aspects of responses, despite having access to preference data that includes reward scores from judge models during AI feedback. Striving to maximize the implicit reward gap between the chosen and the slightly inferior rejected responses can cause overfitting and unnecessary unlearning of the high-quality rejected responses. The unawareness of the reward scores also drives the LLM to indiscriminately favor the low-quality chosen responses and fail to generalize to optimal responses that are sparse in data. To overcome these shortcomings, our study introduces reward-conditioned LLM policies that discern and learn from the entire spectrum of response quality within the dataset, helping extrapolate to more optimal regions. We propose an effective yet simple data relabeling method that conditions the preference pairs on quality scores to construct a reward-augmented dataset. The experiments across various benchmarks and diverse models demonstrate that our approach consistently boosts DPO by a considerable margin. Through comprehensive ablation studies, we demonstrate that our method not only maximizes the utility of preference data but also mitigates the issue of unlearning, demonstrating its broad effectiveness beyond mere data expansion. Our code is available at https://github.com/shenao-zhang/reward-augmented-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07118v1">KOKKAI DOC: An LLM-driven framework for scaling parliamentary representatives</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      This paper introduces an LLM-driven framework designed to accurately scale the political issue stances of parliamentary representatives. By leveraging advanced natural language processing techniques and large language models, the proposed methodology refines and enhances previous approaches by addressing key challenges such as noisy speech data, manual bias in selecting political axes, and the lack of dynamic, diachronic analysis. The framework incorporates three major innovations: (1) de-noising parliamentary speeches via summarization to produce cleaner, more consistent opinion embeddings; (2) automatic extraction of axes of political controversy from legislators' speech summaries; and (3) a diachronic analysis that tracks the evolution of party positions over time. We conduct quantitative and qualitative evaluations to verify our methodology. Quantitative evaluations demonstrate high correlation with expert predictions across various political topics, while qualitative analyses reveal meaningful associations between language patterns and political ideologies. This research aims to have an impact beyond the field of academia by making the results accessible by the public on teh web application: kokkaidoc.com. We are hoping that through our application, Japanese voters can gain a data-driven insight into the political landscape which aids them to make more nuanced voting decisions. Overall, this work contributes to the growing body of research that applies LLMs in political science, offering a flexible and reliable framework for scaling political positions from parliamentary speeches. But also explores the practical applications of the research in the real world to have real world impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16971v4">AIOS: LLM Agent Operating System</a></div>
    <div class="paper-meta">
      📅 2025-05-11
    </div>
    <details class="paper-abstract">
      LLM-based intelligent agents face significant deployment challenges, particularly related to resource management. Allowing unrestricted access to LLM or tool resources can lead to inefficient or even potentially harmful resource allocation and utilization for agents. Furthermore, the absence of proper scheduling and resource management mechanisms in current agent designs hinders concurrent processing and limits overall system efficiency. As the diversity and complexity of agents continue to grow, addressing these resource management issues becomes increasingly critical to LLM-based agent systems. To address these challenges, this paper proposes the architecture of AIOS (LLM-based AI Agent Operating System) under the context of managing LLM-based agents. It introduces a novel architecture for serving LLM-based agents by isolating resources and LLM-specific services from agent applications into an AIOS kernel. This AIOS kernel provides fundamental services (e.g., scheduling, context management, memory management, storage management, access control) and efficient management of resources (e.g., LLM and external tools) for runtime agents. To enhance usability, AIOS also includes an AIOS-Agent SDK, a comprehensive suite of APIs designed for utilizing functionalities provided by the AIOS kernel. Experimental results demonstrate that using AIOS can achieve up to 2.1x faster execution for serving agents built by various agent frameworks. The source code is available at https://github.com/agiresearch/AIOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v4">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07840v2">Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Accepted for AIED 2025, the 26th International Conference on Artificial Intelligence in Education, July 22 - 26, 2025, Palermo, Italy
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01639v4">Moral Alignment for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Published at the 13th International Conference on Learning Representations (ICLR'25), Singapore, Apr 2025. https://openreview.net/forum?id=MeGDmZjUXy
    </div>
    <details class="paper-abstract">
      Decision-making agents based on pre-trained Large Language Models (LLMs) are increasingly being deployed across various domains of human activity. While their applications are currently rather specialized, several research efforts are underway to develop more generalist agents. As LLM-based systems become more agentic, their influence on human activity will grow and their transparency will decrease. Consequently, developing effective methods for aligning them to human values is vital. The prevailing practice in alignment often relies on human preference data (e.g., in RLHF or DPO), in which values are implicit, opaque and are essentially deduced from relative preferences over different model outputs. In this work, instead of relying on human feedback, we introduce the design of reward functions that explicitly and transparently encode core human values for Reinforcement Learning-based fine-tuning of foundation agent models. Specifically, we use intrinsic rewards for the moral alignment of LLM agents. We evaluate our approach using the traditional philosophical frameworks of Deontological Ethics and Utilitarianism, quantifying moral rewards for agents in terms of actions and consequences on the Iterated Prisoner's Dilemma (IPD) environment. We also show how moral fine-tuning can be deployed to enable an agent to unlearn a previously developed selfish strategy. Finally, we find that certain moral strategies learned on the IPD game generalize to several other matrix game environments. In summary, we demonstrate that fine-tuning with intrinsic rewards is a promising general solution for aligning LLM agents to human values, and it might represent a more transparent and cost-effective alternative to currently predominant alignment techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06782v1">Utilizing LLMs to Investigate the Disputed Role of Evidence in Electronic Cigarette Health Policy Formation in Australia and the UK</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Australia and the UK have developed contrasting approaches to the regulation of electronic cigarettes, with - broadly speaking - Australia adopting a relatively restrictive approach and the UK adopting a more permissive approach. Notably, these divergent policies were developed from the same broad evidence base. In this paper, to investigate differences in how the two jurisdictions manage and present evidence, we developed and evaluated a Large Language Model-based sentence classifier to perform automated analyses of electronic cigarette-related policy documents drawn from official Australian and UK legislative processes (109 documents in total). Specifically, we utilized GPT-4 to automatically classify sentences based on whether they contained claims that e-cigarettes were broadly helpful or harmful for public health. Our LLM-based classifier achieved an F-score of 0.9. Further, when applying the classifier to our entire sentence-level corpus, we found that Australian legislative documents show a much higher proportion of harmful statements, and a lower proportion of helpful statements compared to the expected values, with the opposite holding for the UK. In conclusion, this work utilized an LLM-based approach to provide evidence to support the contention that - drawing on the same evidence base - Australian ENDS-related policy documents emphasize the harms associated with ENDS products and UK policy documents emphasize the benefits. Further, our approach provides a starting point for using LLM-based methods to investigate the complex relationship between evidence and health policy formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19044v2">Calibrating Translation Decoding with Quality Estimation on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Neural machine translation (NMT) systems typically employ maximum a posteriori (MAP) decoding to select the highest-scoring translation from the distribution mass. However, recent evidence highlights the inadequacy of MAP decoding, often resulting in low-quality or even pathological hypotheses -- the decoding objective is not aligned with real-world translation quality. This paper proposes calibrating hypothesis likelihoods with translation quality from a distribution view by directly optimizing their Pearson correlation -- thereby enhancing the effectiveness of translation decoding. With our method, translation on large language models (LLMs) improves substantially after limited training (2K instances per direction). This improvement is orthogonal to those achieved through supervised fine-tuning, leading to substantial gains across a broad range of metrics and human evaluations -- even when applied to top-performing translation-specialized LLMs fine-tuned on high-quality translation data, such as Tower, or when compared to recent preference optimization methods, like CPO. Moreover, the calibrated translation likelihood can directly serve as a strong proxy for translation quality, closely approximating or even surpassing some state-of-the-art translation quality estimation models, like CometKiwi. Lastly, our in-depth analysis demonstrates that calibration enhances the effectiveness of MAP decoding, thereby enabling greater efficiency in real-world deployment. The resulting state-of-the-art translation model, which covers 10 languages, along with the accompanying code and human evaluation data, has been released to the community: https://github.com/moore3930/calibrating-llm-mt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v2">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06653v1">Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demand extensive memory capacity during both fine-tuning and inference. To enable memory-efficient fine-tuning, existing methods apply block-wise quantization techniques, such as NF4 and AF4, to the network weights. We show that these quantization techniques incur suboptimal quantization errors. Therefore, as a first novelty, we propose an optimization approach for block-wise quantization. Using this method, we design a family of quantizers named 4-bit block-wise optimal float (BOF4), which consistently reduces the quantization error compared to both baseline methods. We provide both a theoretical and a data-driven solution for the optimization process and prove their practical equivalence. Secondly, we propose a modification to the employed normalization method based on the signed absolute block maximum (BOF4-S), enabling further reduction of the quantization error and empirically achieving less degradation in language modeling performance. Thirdly, we explore additional variations of block-wise quantization methods applied to LLMs through an experimental study on the importance of accurately representing zero and large-amplitude weights on the one hand, and optimization towards various error metrics on the other hand. Lastly, we introduce a mixed-precision quantization strategy dubbed outlier-preserving quantization (OPQ) to address the distributional mismatch induced by outlier weights in block-wise quantization. By storing outlier weights in 16-bit precision (OPQ) while applying BOF4-S, we achieve top performance among 4-bit block-wise quantization techniques w.r.t. perplexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06591v1">Evaluating LLM-Generated Q&A Test: a Student-Centered Study</a></div>
    <div class="paper-meta">
      📅 2025-05-10
      | 💬 accepted to AIED 2025
    </div>
    <details class="paper-abstract">
      This research prepares an automatic pipeline for generating reliable question-answer (Q&A) tests using AI chatbots. We automatically generated a GPT-4o-mini-based Q&A test for a Natural Language Processing course and evaluated its psychometric and perceived-quality metrics with students and experts. A mixed-format IRT analysis showed that the generated items exhibit strong discrimination and appropriate difficulty, while student and expert star ratings reflect high overall quality. A uniform DIF check identified two items for review. These findings demonstrate that LLM-generated assessments can match human-authored tests in psychometric performance and user satisfaction, illustrating a scalable approach to AI-assisted assessment development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09798v2">Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06481v1">QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      The deployment of mixture-of-experts (MoE) large language models (LLMs) presents significant challenges due to their high memory demands. These challenges become even more pronounced in multi-tenant environments, where shared resources must accommodate multiple models, limiting the effectiveness of conventional virtualization techniques. This paper addresses the problem of efficiently serving multiple fine-tuned MoE-LLMs on a single-GPU. We propose a serving system that employs \textit{similarity-based expert consolidation} to reduce the overall memory footprint by sharing similar experts across models. To ensure output quality, we introduce \textit{runtime partial reconfiguration}, dynamically replacing non-expert layers when processing requests from different models. As a result, our approach achieves a competitive output quality while maintaining throughput comparable to serving a single model while incurring a negligible increase in time-to-first-token (TTFT). Experiments on a server with a single NVIDIA A100 GPU (80GB) using Mixtral-8x7B models demonstrate an 85\% average reduction in turnaround time compared to NVIDIA's multi-instance GPU (MIG). Furthermore, experiments on Google's Switch Transformer Base-8 model with up to four variants demonstrate the scalability and resilience of our approach in maintaining output quality compared to other model merging baselines, highlighting its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07877v1">Efficient Telecom Specific LLM: TSLAM-Mini with QLoRA and Digital Twin Data</a></div>
    <div class="paper-meta">
      📅 2025-05-10
      | 💬 Introducing TSLAM-Mini, a specialized language model for telecommunications, demonstrating the efficacy of QLoRA fine-tuning and digital twin-synthesized data for enhanced network intelligence. Model available on: https://huggingface.co/NetoAISolutions/TSLAM-Mini-2B
    </div>
    <details class="paper-abstract">
      General-purpose large language models (LLMs), despite their broad capabilities accrued from open-world data, frequently exhibit suboptimal performance when confronted with the nuanced and specialized demands inherent in real-time telecommunications applications. This investigation addresses this critical limitation through the meticulous fine-tuning of TSLAM-Mini developed by NetoAI, a compact (3.8-billion parameter) causal language model architecturally derived from Phi-4 Mini Instruct 4B. The fine-tuning regimen leverages a bespoke dataset comprising 100,000 samples, strategically engineered to address 20 pivotal telecommunications use-cases, encompassing domains such as Network Fundamentals, IP Routing, MPLS, Network Security, Automation, OSS/BSS, RAN, Mobile Core, Satellite Communications, and Ethical AI. This dataset was curated utilizing NetoAI's DigiTwin platform, enriched with granular insights from venerated network Subject Matter Experts (SMEs) and authoritative RFC documents, thereby capturing high-fidelity representations of real-world network dynamics through simulations inspired by digital twin paradigms. Employing Quantized Low-Rank Adaptation (QLoRA), a state-of-the-art Parameter Efficient Fine-Tuning (PEFT) technique, we achieved substantial training efficiency and enabled prospective deployment on resource-constrained hardware. A novel evaluation framework, predicated on a high-capacity LLM (Qwen3-235B-A22B) functioning as an automated adjudicator, was instituted to rigorously assess instruction-following fidelity and response quality across the specified telecom use-cases. Empirical results unequivocally demonstrate TSLAM-Mini's superior aptitude in telecom-centric applications, underscoring the profound efficacy of domain-specific datasets and PEFT methodologies for advancing intelligent network management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v2">LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Update WIP
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LiTransProQA, a novel, reference-free, LLM-based question-answering framework designed specifically for literary translation evaluation. LiTransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LiTransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LiTransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04134v2">The Order Effect: Investigating Prompt Sensitivity to Input Order in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 The first 3 authors have contributed equally
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become integral to diverse applications, ensuring their reliability under varying input conditions is crucial. One key issue affecting this reliability is order sensitivity, wherein slight variations in the input arrangement can lead to inconsistent or biased outputs. Although recent advances have reduced this sensitivity, the problem remains unresolved. This paper investigates the extent of order sensitivity in LLMs whose internal components are hidden from users (such as closed-source models or those accessed via API calls). We conduct experiments across multiple tasks, including paraphrasing, relevance judgment, and multiple-choice questions. Our results show that input order significantly affects performance across tasks, with shuffled inputs leading to measurable declines in output accuracy. Few-shot prompting demonstrates mixed effectiveness and offers partial mitigation; however, fails to fully resolve the problem. These findings highlight persistent risks, particularly in high-stakes applications, and point to the need for more robust LLMs or improved input-handling techniques in future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06184v1">From Millions of Tweets to Actionable Insights: Leveraging LLMs for User Profiling</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted at MisD @ AAAI ICWSM 2025
    </div>
    <details class="paper-abstract">
      Social media user profiling through content analysis is crucial for tasks like misinformation detection, engagement prediction, hate speech monitoring, and user behavior modeling. However, existing profiling techniques, including tweet summarization, attribute-based profiling, and latent representation learning, face significant limitations: they often lack transferability, produce non-interpretable features, require large labeled datasets, or rely on rigid predefined categories that limit adaptability. We introduce a novel large language model (LLM)-based approach that leverages domain-defining statements, which serve as key characteristics outlining the important pillars of a domain as foundations for profiling. Our two-stage method first employs semi-supervised filtering with a domain-specific knowledge base, then generates both abstractive (synthesized descriptions) and extractive (representative tweet selections) user profiles. By harnessing LLMs' inherent knowledge with minimal human validation, our approach is adaptable across domains while reducing the need for large labeled datasets. Our method generates interpretable natural language user profiles, condensing extensive user data into a scale that unlocks LLMs' reasoning and knowledge capabilities for downstream social network tasks. We contribute a Persian political Twitter (X) dataset and an LLM-based evaluation framework with human validation. Experimental results show our method significantly outperforms state-of-the-art LLM-based and traditional methods by 9.8%, demonstrating its effectiveness in creating flexible, adaptable, and interpretable user profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06150v1">A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06149v1">Can Prompting LLMs Unlock Hate Speech Detection across Languages? A Zero-shot and Few-shot Study</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Despite growing interest in automated hate speech detection, most existing approaches overlook the linguistic diversity of online content. Multilingual instruction-tuned large language models such as LLaMA, Aya, Qwen, and BloomZ offer promising capabilities across languages, but their effectiveness in identifying hate speech through zero-shot and few-shot prompting remains underexplored. This work evaluates LLM prompting-based detection across eight non-English languages, utilizing several prompting techniques and comparing them to fine-tuned encoder models. We show that while zero-shot and few-shot prompting lag behind fine-tuned encoder models on most of the real-world evaluation sets, they achieve better generalization on functional tests for hate speech detection. Our study also reveals that prompt design plays a critical role, with each language often requiring customized prompting techniques to maximize performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09667v2">k-LLMmeans: Scalable, Stable, and Interpretable Text Clustering via LLM-based Centroids</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      We introduce k-LLMmeans, a novel modification of the k-means algorithm for text clustering that leverages LLM-generated summaries as cluster centroids, capturing semantic nuances often missed by purely numerical averages. This design preserves the core optimization properties of k-means while enhancing semantic interpretability and avoiding the scalability and instability issues typical of modern LLM-based clustering. Unlike existing methods, our approach does not increase LLM usage with dataset size and produces transparent intermediate outputs. We further extend it with a mini-batch variant for efficient, real-time clustering of streaming text. Extensive experiments across multiple datasets, embeddings, and LLMs show that k-LLMmeans consistently outperforms k-means and other traditional baselines and achieves results comparable to state-of-the-art LLM-based clustering, with a fraction of the LLM calls. Finally, we present a case study on sequential text streams and introduce a new benchmark dataset constructed from StackExchange to evaluate text-stream clustering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06120v1">LLMs Get Lost In Multi-Turn Conversation</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are conversational interfaces. As such, LLMs have the potential to assist their users not only when they can fully specify the task at hand, but also to help them define, explore, and refine what they need through multi-turn conversational exchange. Although analysis of LLM conversation logs has confirmed that underspecification occurs frequently in user instructions, LLM evaluation has predominantly focused on the single-turn, fully-specified instruction setting. In this work, we perform large-scale simulation experiments to compare LLM performance in single- and multi-turn settings. Our experiments confirm that all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks. Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability. We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely. In simpler terms, we discover that *when LLMs take a wrong turn in a conversation, they get lost and do not recover*.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06096v1">Free and Fair Hardware: A Pathway to Copyright Infringement-Free Verilog Generation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted at DAC 2025
    </div>
    <details class="paper-abstract">
      Limitations in Large Language Model (LLM) capabilities for hardware design tasks, such as generating functional Verilog codes, have motivated various fine-tuning optimizations utilizing curated hardware datasets from open-source repositories. However, these datasets remain limited in size and contain minimal checks on licensing for reuse, resulting in potential copyright violations by fine-tuned LLMs. Therefore, we propose an evaluation benchmark to estimate the risk of Verilog-trained LLMs to generate copyright-protected codes. To minimize this risk, we present an open-source Verilog dataset, FreeSet, containing over 220k files, along with the automated dataset curation framework utilized to provide additional guarantees of fair-use Verilog data. We then execute an LLM fine-tuning framework consisting of continual pre-training, resulting in a fine-tuned Llama model for Verilog, FreeV. Our results indicate that FreeV demonstrates the smallest risk of copyright-infringement among prior works, with only a 3% violation rate. Furthermore, experimental results demonstrate improvements in Verilog generation functionality over its baseline model, improving VerilogEval pass@10 rates by over 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06046v1">Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 24 pages, 10 pages main text
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries, created via an automated pipeline. We also release a new dataset of the extracted UK Government public health guidance documents used as source text for PubHealthBench. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, whilst there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses on public health topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11963v2">NeedleBench: Can LLMs Do Retrieval and Reasoning in Information-Dense Context?</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 v2: updated with tested models and Multi-Needle Reasoning implementation
    </div>
    <details class="paper-abstract">
      The capability of large language models to handle long-context information is crucial across various real-world applications. Existing evaluation methods often rely either on real-world long texts, making it difficult to exclude the influence of models' inherent knowledge, or introduce irrelevant filler content to artificially achieve target lengths, reducing assessment effectiveness. To address these limitations, we introduce NeedleBench, a synthetic framework for assessing retrieval and reasoning performance in bilingual long-context tasks with adaptive context lengths. NeedleBench systematically embeds key data points at varying depths to rigorously test model capabilities. Tasks are categorized into two scenarios: information-sparse, featuring minimal relevant details within extensive irrelevant text to simulate simple retrieval tasks; and information-dense (the Ancestral Trace Challenge), where relevant information is continuously distributed throughout the context to simulate complex reasoning tasks. Our experiments reveal that although recent reasoning models like Deepseek-R1 and OpenAI's o3 excel in mathematical reasoning, they struggle with continuous retrieval and reasoning in information-dense scenarios, even at shorter context lengths. We also characterize a phenomenon termed 'under-thinking', where models prematurely conclude reasoning despite available information. NeedleBench thus provides critical insights and targeted tools essential for evaluating and improving LLMs' long-context capabilities. All resources are available at OpenCompass: https://github.com/open-compass/opencompass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05832v1">Augmented Body Communicator: Enhancing daily body expression for people with upper limb limitations through LLM and a robotic arm</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Individuals with upper limb movement limitations face challenges in interacting with others. Although robotic arms are currently used primarily for functional tasks, there is considerable potential to explore ways to enhance users' body language capabilities during social interactions. This paper introduces an Augmented Body Communicator system that integrates robotic arms and a large language model. Through the incorporation of kinetic memory, disabled users and their supporters can collaboratively design actions for the robot arm. The LLM system then provides suggestions on the most suitable action based on contextual cues during interactions. The system underwent thorough user testing with six participants who have conditions affecting upper limb mobility. Results indicate that the system improves users' ability to express themselves. Based on our findings, we offer recommendations for developing robotic arms that support disabled individuals with body language capabilities and functional tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v5">Estimating LLM Uncertainty with Evidence</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Over the past few years, Large Language Models (LLMs) have developed rapidly and are widely applied in various domains. However, LLMs face the issue of hallucinations, generating responses that may be unreliable when the models lack relevant knowledge. To be aware of potential hallucinations, uncertainty estimation methods have been introduced, and most of them have confirmed that reliability lies in critical tokens. However, probability-based methods perform poorly in identifying token reliability, limiting their practical utility. In this paper, we reveal that the probability-based method fails to estimate token reliability due to the loss of evidence strength information which is accumulated in the training stage. Therefore, we present Logits-induced token uncertainty (LogTokU), a framework for estimating decoupled token uncertainty in LLMs, enabling real-time uncertainty estimation without requiring multiple sampling processes. We employ evidence modeling to implement LogTokU and use the estimated uncertainty to guide downstream tasks. The experimental results demonstrate that LogTokU has significant effectiveness and promise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05794v1">What Is Next for LLMs? Next-Generation AI Computing Hardware Using Photonic Chips</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 36 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly pushing the limits of contemporary computing hardware. For example, training GPT-3 has been estimated to consume around 1300 MWh of electricity, and projections suggest future models may require city-scale (gigawatt) power budgets. These demands motivate exploration of computing paradigms beyond conventional von Neumann architectures. This review surveys emerging photonic hardware optimized for next-generation generative AI computing. We discuss integrated photonic neural network architectures (e.g., Mach-Zehnder interferometer meshes, lasers, wavelength-multiplexed microring resonators) that perform ultrafast matrix operations. We also examine promising alternative neuromorphic devices, including spiking neural network circuits and hybrid spintronic-photonic synapses, which combine memory and processing. The integration of two-dimensional materials (graphene, TMDCs) into silicon photonic platforms is reviewed for tunable modulators and on-chip synaptic elements. Transformer-based LLM architectures (self-attention and feed-forward layers) are analyzed in this context, identifying strategies and challenges for mapping dynamic matrix multiplications onto these novel hardware substrates. We then dissect the mechanisms of mainstream LLMs, such as ChatGPT, DeepSeek, and LLaMA, highlighting their architectural similarities and differences. We synthesize state-of-the-art components, algorithms, and integration methods, highlighting key advances and open issues in scaling such systems to mega-sized LLM models. We find that photonic computing systems could potentially surpass electronic processors by orders of magnitude in throughput and energy efficiency, but require breakthroughs in memory, especially for long-context windows and long token sequences, and in storage of ultra-large datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05786v1">A Day in Their Shoes: Using LLM-Based Perspective-Taking Interactive Fiction to Reduce Stigma Toward Dirty Work</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Conference paper for FAccT '25
    </div>
    <details class="paper-abstract">
      Occupations referred to as "dirty work" often face entrenched social stigma, which adversely affects the mental health of workers in these fields and impedes occupational equity. In this study, we propose a novel Interactive Fiction (IF) framework powered by Large Language Models (LLMs) to encourage perspective-taking and reduce biases against these stigmatized yet essential roles. Through an experiment with participants (n = 100) across four such occupations, we observed a significant increase in participants' understanding of these occupations, as well as a high level of empathy and a strong sense of connection to individuals in these roles. Additionally, qualitative interviews with participants (n = 15) revealed that the LLM-based perspective-taking IF enhanced immersion, deepened emotional resonance and empathy toward "dirty work," and allowed participants to experience a sense of professional fulfillment in these occupations. However, participants also highlighted ongoing challenges, such as limited contextual details generated by the LLM and the unintentional reinforcement of existing stereotypes. Overall, our findings underscore that an LLM-based perspective-taking IF framework offers a promising and scalable strategy for mitigating stigma and promoting social equity in marginalized professions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21223v3">Rethinking Graph Structure Learning in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 29 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recently, the emergence of LLMs has prompted researchers to integrate language descriptions into graphs, aiming to enhance model encoding capabilities from a data-centric perspective. This graph representation is called text-attributed graphs (TAGs). A review of prior advancements highlights that graph structure learning (GSL) is a pivotal technique for improving data utility, making it highly relevant to efficient TAG learning. However, most GSL methods are tailored for traditional graphs without textual information, underscoring the necessity of developing a new GSL paradigm. Despite clear motivations, it remains challenging: (1) How can we define a reasonable optimization objective for GSL in the era of LLMs, considering the massive parameters in LLM? (2) How can we design an efficient model architecture that enables seamless integration of LLM for this optimization objective? For Question 1, we reformulate existing GSL optimization objectives as a tree optimization framework, shifting the focus from obtaining a well-trained edge predictor to a language-aware tree sampler. For Question 2, we propose decoupled and training-free model design principles for LLM integration, shifting the focus from computation-intensive fine-tuning to more efficient inference. Based on this, we propose Large Language and Tree Assistant (LLaTA), which leverages tree-based LLM in-context learning to enhance the understanding of topology and text, enabling reliable inference and generating improved graph structure. Extensive experiments on 10 datasets demonstrate that LLaTA enjoys flexibility-incorporated with any backbone; scalability-outperforms other LLM-enhanced graph learning methods; effectiveness-achieves SOTA predictive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05772v1">Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Transformer-based models are the foundation of modern machine learning, but their execution, particularly during autoregressive decoding in large language models (LLMs), places significant pressure on memory systems due to frequent memory accesses and growing key-value (KV) caches. This creates a bottleneck in memory bandwidth, especially as context lengths increase. Processing-in-memory (PIM) architectures are a promising solution, offering high internal bandwidth and compute parallelism near memory. However, current PIM designs are primarily optimized for dense attention and struggle with the dynamic, irregular access patterns introduced by modern KV cache sparsity techniques. Consequently, they suffer from workload imbalance, reducing throughput and resource utilization. In this work, we propose STARC, a novel sparsity-optimized data mapping scheme tailored specifically for efficient LLM decoding on PIM architectures. STARC clusters KV pairs by semantic similarity and maps them to contiguous memory regions aligned with PIM bank structures. During decoding, queries retrieve relevant tokens at cluster granularity by matching against precomputed centroids, enabling selective attention and parallel processing without frequent reclustering or data movement overhead. Experiments on the HBM-PIM system show that, compared to common token-wise sparsity methods, STARC reduces attention-layer latency by 19%--31% and energy consumption by 19%--27%. Under a KV cache budget of 1024, it achieves up to 54%--74% latency reduction and 45%--67% energy reduction compared to full KV cache retrieval. Meanwhile, STARC maintains model accuracy comparable to state-of-the-art sparse attention methods, demonstrating its effectiveness in enabling efficient and hardware-friendly long-context LLM inference on PIM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05762v1">Multi-Agent Systems for Robotic Autonomy with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 11 pages, 2 figures, 5 tables, submitted for publication
    </div>
    <details class="paper-abstract">
      Since the advent of Large Language Models (LLMs), various research based on such models have maintained significant academic attention and impact, especially in AI and robotics. In this paper, we propose a multi-agent framework with LLMs to construct an integrated system for robotic task analysis, mechanical design, and path generation. The framework includes three core agents: Task Analyst, Robot Designer, and Reinforcement Learning Designer. Outputs are formatted as multimodal results, such as code files or technical reports, for stronger understandability and usability. To evaluate generalizability comparatively, we conducted experiments with models from both GPT and DeepSeek. Results demonstrate that the proposed system can design feasible robots with control strategies when appropriate task inputs are provided, exhibiting substantial potential for enhancing the efficiency and accessibility of robotic system development in research and industrial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05758v1">APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05744v1">Harnessing LLMs Explanations to Boost Surrogate Models in Tabular Data Classification</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable ability in solving complex tasks, making them a promising tool for enhancing tabular learning. However, existing LLM-based methods suffer from high resource requirements, suboptimal demonstration selection, and limited interpretability, which largely hinder their prediction performance and application in the real world. To overcome these problems, we propose a novel in-context learning framework for tabular prediction. The core idea is to leverage the explanations generated by LLMs to guide a smaller, locally deployable Surrogate Language Model (SLM) to make interpretable tabular predictions. Specifically, our framework mainly involves three stages: (i) Post Hoc Explanation Generation, where LLMs are utilized to generate explanations for question-answer pairs in candidate demonstrations, providing insights into the reasoning behind the answer. (ii) Post Hoc Explanation-Guided Demonstrations Selection, which utilizes explanations generated by LLMs to guide the process of demonstration selection from candidate demonstrations. (iii) Post Hoc Explanation-Guided Interpretable SLM Prediction, which utilizes the demonstrations obtained in step (ii) as in-context and merges corresponding explanations as rationales to improve the performance of SLM and guide the model to generate interpretable outputs. Experimental results highlight the framework's effectiveness, with an average accuracy improvement of 5.31% across various tabular datasets in diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05712v1">LLM-Text Watermarking based on Lagrange Interpolation</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs (Large Language Models) has established them as a foundational technology for many AI and ML powered human computer interactions. A critical challenge in this context is the attribution of LLM-generated text, either to the specific language model used or to the individual user who generated it. This is essential for combating misinformation, fake news, misinterpretation, and plagiarism. One of the key techniques for addressing this issue is watermarking. This work presents a watermarking scheme for LLM-generated text based on Lagrange interpolation, which enables the recovery of a secret author identity even when the text has been heavily redacted by an adversary. The core idea is to embed a continuous sequence of points (x, f(x)) that lie on a single straight line. The x-coordinates are generated pseudorandomly using either an LFSR (when security is not a priority) or a cryptographically secure NFSR for high-security applications. The scheme efficiency and resilience to adversarial modifications are analysed. Experimental results show that the proposed method is highly effective, allowing the recovery of the author identity when as few as three points survive adversarial manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17264v3">Medha: Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) handle increasingly longer contexts, serving long inference requests of millions of tokens presents unique challenges. We show that existing work for long context inference is largely based on techniques from long context training, and does not handle the high variability in input lengths during inference. This leads to inefficient resource utilization, server fragmentation, and head-of-line (HOL) blocking. We present Medha, an end-to-end system for efficient long-context LLM inference that addresses these challenges through fine-grained time sharing. Medha introduces three key innovations: (1) the mechanism of adaptive prefill chunking to help mitigate HOL blocking with preemption; (2) two new parallelism strategies: Sequence Pipeline Parallelism (SPP) to reduce time-to-first-token by pipelining prefill chunks, and KV-Cache Parallelism (KVP) to lower time-peroutput-token by distributing decoding across servers; and (3) a novel input-length aware least remaining slack scheduling to meet Service Level Objectives (SLOs). Medha enables exact inference scaling beyond 10 million tokens, maintaining high throughput and low latency across mixed-length workloads. Compared to state-of-the-art systems, Medha reduces server fragmentation, cuts median latency by up to 30x, and improves throughput by over 5x, delivering production-scale long-context inference without compromising performance on shorter requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20834v3">Token-Efficient RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Title updated to "Token-Efficient RL for LLM Reasoning" to better reflect algorithmic focus. Revised abstract, intro, and conclusion. Paper shortened and typos fixed
    </div>
    <details class="paper-abstract">
      We propose reinforcement learning (RL) strategies tailored for reasoning in large language models (LLMs) under strict memory and compute limits, with a particular focus on compatibility with LoRA fine-tuning. Building on early policy gradient methods with baseline subtraction, we design critic-free methods that operate on a small, informative subset of output tokens to reduce memory usage and stabilize training. We introduce S-GRPO, a stochastic variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching approach for fine-grained credit assignment. Applied to Qwen2-1.5B, our methods raise accuracy on the SVAMP benchmark from 46% to over 70% and show strong performance on multi-digit multiplication. Surprisingly, full-token GRPO under LoRA fails to improve over the base model, suggesting that selective token-level optimization may act as an implicit regularizer in low-parameter training regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06469v1">KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted to the Educational Data Mining (EDM) 2025 conference
    </div>
    <details class="paper-abstract">
      Educators evaluate student knowledge using knowledge component (KC) models that map assessment questions to KCs. Still, designing KC models for large question banks remains an insurmountable challenge for instructors who need to analyze each question by hand. The growing use of Generative AI in education is expected only to aggravate this chronic deficiency of expert-designed KC models, as course engineers designing KCs struggle to keep up with the pace at which questions are generated. In this work, we propose KCluster, a novel KC discovery algorithm based on identifying clusters of congruent questions according to a new similarity metric induced by a large language model (LLM). We demonstrate in three datasets that an LLM can create an effective metric of question similarity, which a clustering algorithm can use to create KC models from questions with minimal human effort. Combining the strengths of LLM and clustering, KCluster generates descriptive KC labels and discovers KC models that predict student performance better than the best expert-designed models available. In anticipation of future work, we illustrate how KCluster can reveal insights into difficult KCs and suggest improvements to instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18966v3">Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 This paper is accepted by NAACL 2025 findings. Link to the paper presentation: https://youtu.be/IhaxwbZOcaU
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06461v1">Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      The common assumption in on-device AI is that GPUs, with their superior parallel processing, always provide the best performance for large language model (LLM) inference. In this work, we challenge this notion by empirically demonstrating that, under certain conditions, CPUs can outperform GPUs for LLM inference on mobile devices. Using a 1-billion-parameter LLM deployed via llama.cpp on the iPhone 15 Pro, we show that a CPU-only configuration (two threads, F16 precision) achieves 17 tokens per second, surpassing the 12.8 tokens per second obtained with GPU acceleration. We analyze the architectural factors driving this counterintuitive result, revealing that GPU memory transfer overhead and CPU thread optimization play a critical role. Furthermore, we explore the impact of thread oversubscription, quantization strategies, and hardware constraints, providing new insights into efficient on-device AI execution. Our findings challenge conventional GPU-first thinking, highlighting the untapped potential of optimized CPU inference and paving the way for smarter deployment strategies in mobile AI. However, fully explaining the observed CPU advantage remains difficult due to limited access to low-level profiling tools on iOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06438v1">Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct result for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts -- unreliable, and malicious knowledge is easy to inject. With the help of logic programming tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more efficient and secure. We proposed an Administrator-Assistant Dual-Agent paradigm, where the two ASP-driven bots share the same knowledge base and complete their tasks independently, while the information can be passed by a Collaborative Rule Set (CRS). The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. We have constructed AutoManager, a dual-agent system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the assistant bot takes the customer's order while the administrator bot manages the menu and food supply. We evaluated our AutoManager and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06416v1">ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and the introduction of the Model Context Protocol (MCP) have significantly expanded LLM agents' capability to interact dynamically with external tools and APIs. However, existing tool selection frameworks do not integrate MCP servers, instead relying heavily on error-prone manual updates to monolithic local tool repositories, leading to duplication, inconsistencies, and inefficiencies. Additionally, current approaches abstract tool selection before the LLM agent is invoked, limiting its autonomy and hindering dynamic re-querying capabilities during multi-turn interactions. To address these issues, we introduce ScaleMCP, a novel tool selection approach that dynamically equips LLM agents with a MCP tool retriever, giving agents the autonomy to add tools into their memory, as well as an auto-synchronizing tool storage system pipeline through CRUD (create, read, update, delete) operations with MCP servers as the single source of truth. We also propose a novel embedding strategy, Tool Document Weighted Average (TDWA), designed to selectively emphasize critical components of tool documents (e.g. tool name or synthetic questions) during the embedding process. Comprehensive evaluations conducted on a created dataset of 5,000 financial metric MCP servers, across 10 LLM models, 5 embedding models, and 5 retriever types, demonstrate substantial improvements in tool retrieval and agent invocation performance, emphasizing ScaleMCP's effectiveness in scalable, dynamic tool selection and invocation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06003v2">LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Conference Version (ISCA'25)
    </div>
    <details class="paper-abstract">
      As large language model (LLM) inference continues to demand increasing computational resources, there is a rapidly growing trend toward using low-bit weights to reduce memory footprint and improve inference efficiency. However, low-bit LLMs introduce the need for mixed-precision general matrix multiplication (mpGEMM), which involves multiplying low-precision weights with higher-precision activations - a critical yet under-explored operation. Current hardware lacks native support for mpGEMM, leading to inefficient dequantization-based implementations. To address this, we explore a lookup table (LUT)-based approach to accelerate mpGEMM. While conventional LUT implementations fall short in performance and flexibility, we propose LUT Tensor Core, a software-hardware co-designed solution optimized for low-bit LLM inference. On the software side, we introduce operator fusion and table symmetrization techniques to optimize LUT generation and storage. On the hardware side, LUT Tensor Core adopts an elongated tiling shape to maximize table reuse and employs a bit-serial architecture to flexibly support a variety of precision combinations. Additionally, we design an end-to-end compilation stack with custom instructions to enable efficient code generation and optimization for LUT-based mpGEMM. Experimental results on low-bit LLMs such as BitNet and LLaMA demonstrate that LUT Tensor Core delivers over an order-of-magnitude improvement in both compute density and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09701v2">Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted and to appear in IJCNN 2025
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) for program code generation has gained substantial attention, but their biases and limitations with non-English prompts challenge global inclusivity. This paper investigates the complexities of multilingual prompt-based code generation. Our evaluations of LLMs, including CODELLAMA and CODEGEMMA, reveal significant disparities in code quality for non-English prompts; we also demonstrate the inadequacy of simple approaches like prompt translation, bootstrapped data augmentation, and fine-tuning. To address this, we propose a zero-shot cross-lingual approach using a neural projection technique, integrating a cross-lingual encoder like LASER to map multilingual embeddings from it into the LLM's token space. This method requires training only on English data and scales effectively to other languages. Results on a translated and quality-checked MBPP dataset show substantial improvements in code quality. This research promotes a more inclusive code generation landscape by empowering LLMs with multilingual capabilities to support the diverse linguistic spectrum in programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06364v1">LATENT: LLM-Augmented Trojan Insertion and Evaluation Framework for Analog Netlist Topologies</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted for presentation at IEEE International Conference on LLM-Aided Design (ICLAD), 2025
    </div>
    <details class="paper-abstract">
      Analog and mixed-signal (A/MS) integrated circuits (ICs) are integral to safety-critical applications. However, the globalization and outsourcing of A/MS ICs to untrusted third-party foundries expose them to security threats, particularly analog Trojans. Unlike digital Trojans which have been extensively studied, analog Trojans remain largely unexplored. There has been only limited research on their diversity and stealth in analog designs, where a Trojan is activated only during a narrow input voltage range. Effective defense techniques require a clear understanding of the attack vectors; however, the lack of diverse analog Trojan instances limits robust advances in detection strategies. To address this gap, we present LATENT, the first large language model (LLM)-driven framework for crafting stealthy, circuit-specific analog Trojans. LATENT incorporates LLM as an autonomous agent to intelligently insert and refine Trojan components within analog designs based on iterative feedback from a detection model. This feedback loop ensures that the inserted Trojans remain stealthy while successfully evading detection. Experimental results demonstrate that our generated Trojan designs exhibit an average Trojan-activation range of 15.74%, ensuring they remain inactive under most operating voltages, while causing a significant performance degradation of 11.3% upon activation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06321v1">Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-09
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt. Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design. Code can be found in https://github.com/zch65458525/L2T.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05237v1">Latte: Transfering LLMs` Latent-level Knowledge for Few-shot Tabular Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Few-shot tabular learning, in which machine learning models are trained with a limited amount of labeled data, provides a cost-effective approach to addressing real-world challenges. The advent of Large Language Models (LLMs) has sparked interest in leveraging their pre-trained knowledge for few-shot tabular learning. Despite promising results, existing approaches either rely on test-time knowledge extraction, which introduces undesirable latency, or text-level knowledge, which leads to unreliable feature engineering. To overcome these limitations, we propose Latte, a training-time knowledge extraction framework that transfers the latent prior knowledge within LLMs to optimize a more generalized downstream model. Latte enables general knowledge-guided downstream tabular learning, facilitating the weighted fusion of information across different feature values while reducing the risk of overfitting to limited labeled data. Furthermore, Latte is compatible with existing unsupervised pre-training paradigms and effectively utilizes available unlabeled samples to overcome the performance limitations imposed by an extremely small labeled dataset. Extensive experiments on various few-shot tabular learning benchmarks demonstrate the superior performance of Latte, establishing it as a state-of-the-art approach in this domain
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05225v1">QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      The rapid advancement of Chinese large language models (LLMs) underscores the need for domain-specific evaluations to ensure reliable applications. However, existing benchmarks often lack coverage in vertical domains and offer limited insights into the Chinese working context. Leveraging qualification exams as a unified framework for human expertise evaluation, we introduce QualBench, the first multi-domain Chinese QA benchmark dedicated to localized assessment of Chinese LLMs. The dataset includes over 17,000 questions across six vertical domains, with data selections grounded in 24 Chinese qualifications to closely align with national policies and working standards. Through comprehensive evaluation, the Qwen2.5 model outperformed the more advanced GPT-4o, with Chinese LLMs consistently surpassing non-Chinese models, highlighting the importance of localized domain knowledge in meeting qualification requirements. The best performance of 75.26% reveals the current gaps in domain coverage within model capabilities. Furthermore, we present the failure of LLM collaboration with crowdsourcing mechanisms and suggest the opportunities for multi-domain RAG knowledge enhancement and vertical domain LLM training with Federated Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v3">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05196v1">Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based Retrieval-Augmented Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      We present a systematic study of provider-side data poisoning in retrieval-augmented recommender systems (RAG-based). By modifying only a small fraction of tokens within item descriptions -- for instance, adding emotional keywords or borrowing phrases from semantically related items -- an attacker can significantly promote or demote targeted items. We formalize these attacks under token-edit and semantic-similarity constraints, and we examine their effectiveness in both promotion (long-tail items) and demotion (short-head items) scenarios. Our experiments on MovieLens, using two large language model (LLM) retrieval modules, show that even subtle attacks shift final rankings and item exposures while eluding naive detection. The results underscore the vulnerability of RAG-based pipelines to small-scale metadata rewrites and emphasize the need for robust textual consistency checks and provenance tracking to thwart stealthy provider-side poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18635v2">Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      While Retrieval Augmented Generation (RAG) has emerged as a popular technique for improving Large Language Model (LLM) systems, it introduces a large number of choices, parameters and hyperparameters that must be made or tuned. This includes the LLM, embedding, and ranker models themselves, as well as hyperparameters governing individual RAG components. Yet, collectively optimizing the entire configuration in a RAG or LLM system remains under-explored - especially in multi-objective settings - due to intractably large solution spaces, noisy objective evaluations, and the high cost of evaluations. In this work, we introduce the first approach for multi-objective parameter optimization of cost, latency, safety and alignment over entire LLM and RAG systems. We find that Bayesian optimization methods significantly outperform baseline approaches, obtaining a superior Pareto front on two new RAG benchmark tasks. We conclude our work with important considerations for practitioners who are designing multi-objective RAG systems, highlighting nuances such as how optimal configurations may not generalize across tasks and objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05057v1">Towards Mitigating API Hallucination in Code Generated by LLMs with Hierarchical Dependency Aware</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted by FSE 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Application Programming Interfaces (APIs) are crucial in modern software development. Large Language Models (LLMs) assist in automated code generation but often struggle with API hallucination, including invoking non-existent APIs and misusing existing ones in practical development scenarios. Existing studies resort to Retrieval-Augmented Generation (RAG) methods for mitigating the hallucination issue, but tend to fail since they generally ignore the structural dependencies in practical projects and do not indeed validate whether the generated APIs are available or not. To address these limitations, we propose MARIN, a framework for mitigating API hallucination in code generated by LLMs with hierarchical dependency aware. MARIN consists of two phases: Hierarchical Dependency Mining, which analyzes local and global dependencies of the current function, aiming to supplement comprehensive project context in LLMs input, and Dependency Constrained Decoding, which utilizes mined dependencies to adaptively constrain the generation process, aiming to ensure the generated APIs align with the projects specifications. To facilitate the evaluation of the degree of API hallucination, we introduce a new benchmark APIHulBench and two new metrics including Micro Hallucination Number (MiHN) and Macro Hallucination Rate (MaHR). Experiments on six state-of-the-art LLMs demonstrate that MARIN effectively reduces API hallucinations, achieving an average decrease of 67.52% in MiHN and 73.56% in MaHR compared to the RAG approach. Applied to Huaweis internal projects and two proprietary LLMs, MARIN achieves average decreases of 57.33% in MiHN and 59.41% in MaHR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19981v2">Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4% absolute on SAT MATH). Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03961v2">The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 16 pages, 8 figures. Code available at https://github.com/storyagents25/story-agents
    </div>
    <details class="paper-abstract">
      According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00762v4">Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scales Test-Time Compute</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05016v1">The Pitfalls of Growing Group Complexity: LLMs and Social Choice-Based Aggregation for Group Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 To be published in: Adjunct Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization (UMAP Adjunct '25), June 16--19, 2025, New York City, NY, USA Accepted at the 4th Workshop on Group Modeling, Adaptation and Personalization (GMAP), co-located at UMAP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied in recommender systems aimed at both individuals and groups. Previously, Group Recommender Systems (GRS) often used social choice-based aggregation strategies to derive a single recommendation based on the preferences of multiple people. In this paper, we investigate under which conditions language models can perform these strategies correctly based on zero-shot learning and analyse whether the formatting of the group scenario in the prompt affects accuracy. We specifically focused on the impact of group complexity (number of users and items), different LLMs, different prompting conditions, including In-Context learning or generating explanations, and the formatting of group preferences. Our results show that performance starts to deteriorate when considering more than 100 ratings. However, not all language models were equally sensitive to growing group complexity. Additionally, we showed that In-Context Learning (ICL) can significantly increase the performance at higher degrees of group complexity, while adding other prompt modifications, specifying domain cues or prompting for explanations, did not impact accuracy. We conclude that future research should include group complexity as a factor in GRS evaluation due to its effect on LLM performance. Furthermore, we showed that formatting the group scenarios differently, such as rating lists per user or per item, affected accuracy. All in all, our study implies that smaller LLMs are capable of generating group recommendations under the right conditions, making the case for using smaller models that require less computing power and costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14401v2">LLM-Driven Usefulness Judgment for Web Search Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Evaluation is fundamental in optimizing search experiences and supporting diverse user intents in Information Retrieval (IR). Traditional search evaluation methods primarily rely on relevance labels, which assess how well retrieved documents match a user's query. However, relevance alone fails to capture a search system's effectiveness in helping users achieve their search goals, making usefulness a critical evaluation criterion. In this paper, we explore an alternative approach: LLM-generated usefulness labels, which incorporate both implicit and explicit user behavior signals to evaluate document usefulness. We propose Task-aware Rubric-based Usefulness Evaluation (TRUE), a rubric-driven evaluation method that employs iterative sampling and reasoning to model complex search behavior patterns. Our findings show that (i) LLMs can generate moderate usefulness labels by leveraging comprehensive search session history incorporating personalization and contextual understanding, and (ii) fine-tuned LLMs improve usefulness judgments when provided with structured search session contexts. Additionally, we examine whether LLMs can distinguish between relevance and usefulness, particularly in cases where this divergence impacts search success. We also conduct an ablation study to identify key metrics for accurate usefulness label generation, optimizing for token efficiency and cost-effectiveness in real-world applications. This study advances LLM-based usefulness evaluation by refining key user metrics, exploring LLM-generated label reliability, and ensuring feasibility for large-scale search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06877v3">LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 11 pages. Accepted at SIGIR 2025 (48th International ACM SIGIR Conference on Research and Development in Information Retrieval)
    </div>
    <details class="paper-abstract">
      Test collections are information retrieval tools that allow researchers to quickly and easily evaluate ranking algorithms. While test collections have become an integral part of IR research, the process of data creation involves significant effort in manual annotations, which often makes it very expensive and time-consuming. Thus, test collections could become too small when the budget is limited, which may lead to unstable evaluations. As a cheaper alternative, recent studies have proposed the use of large language models (LLMs) to completely replace human assessors. However, while LLMs seem to somewhat correlate with human judgments, their predictions are not perfect and often show bias. Thus a complete replacement with LLMs is argued to be too risky and not fully reliable. Thus, in this paper, we propose LLM-Assisted Relevance Assessments (LARA), an effective method to balance manual annotations with LLM annotations, which helps to build a rich and reliable test collection even under a low budget. We use the LLM's predicted relevance probabilities to select the most profitable documents to manually annotate under a budget constraint. With theoretical reasoning, LARA effectively guides the human annotation process by actively learning to calibrate the LLM's predicted relevance probabilities. Then, using the calibration model learned from the limited manual annotations, LARA debiases the LLM predictions to annotate the remaining non-assessed data. Empirical evaluations on TREC-7 Ad Hoc, TREC-8 Ad Hoc, TREC Robust 2004, and TREC-COVID datasets show that LARA outperforms alternative solutions under almost any budget constraint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01259v2">Facilitating Instructors-LLM Collaboration for Problem Design in Introductory Programming Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted at CHI 2025 Workshop on Augmented Educators and AI: Shaping the Future of Human and AI Cooperation in Learning
    </div>
    <details class="paper-abstract">
      Advancements in Large Language Models (LLMs), such as ChatGPT, offer significant opportunities to enhance instructional support in introductory programming courses. While extensive research has explored the effectiveness of LLMs in supporting student learning, limited studies have examined how these models can assist instructors in designing instructional activities. This work investigates how instructors' expertise in effective activity design can be integrated with LLMs' ability to generate novel and targeted programming problems, facilitating more effective activity creation for programming classrooms. To achieve this, we employ a participatory design approach to develop an instructor-authoring tool that incorporates LLM support, fostering collaboration between instructors and AI in generating programming exercises. This tool also allows instructors to specify common student mistakes and misconceptions, which informs the adaptive feedback generation process. We conduct case studies with three instructors, analyzing how they use our system to design programming problems for their introductory courses. Through these case studies, we assess instructors' perceptions of the usefulness and limitations of LLMs in authoring problem statements for instructional purposes. Additionally, we compare the efficiency, quality, effectiveness, and coverage of designed activities when instructors create problems with and without structured LLM prompting guidelines. Our findings provide insights into the potential of LLMs in enhancing instructor workflows and improving programming education and provide guidelines for designing effective AI-assisted problem-authoring interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09798v3">ReadMe.LLM: A Framework to Help LLMs Understand Your Library</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 15 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with code generation tasks involving niche software libraries. Existing code generation techniques with only human-oriented documentation can fail -- even when the LLM has access to web search and the library is documented online. To address this challenge, we propose ReadMe$.$LLM, LLM-oriented documentation for software libraries. By attaching the contents of ReadMe$.$LLM to a query, performance consistently improves to near-perfect accuracy, with one case study demonstrating up to 100% success across all tested models. We propose a software development lifecycle where LLM-specific documentation is maintained alongside traditional software updates. In this study, we present two practical applications of the ReadMe$.$LLM idea with diverse software libraries, highlighting that our proposed approach could generalize across programming domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02309v2">Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted to IEEE COMPSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02665v2">A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems.
    </details>
</div>
