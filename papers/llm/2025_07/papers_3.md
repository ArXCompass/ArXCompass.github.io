# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08267v1">A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Presented at ICML 2025 Workshop on The second AI for MATH
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning of Large Language Models (LLMs) is a pivotal challenge in advancing AI capabilities. While Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are the dominant training paradigms, a systematic methodology for combining them to maximize both accuracy and efficiency remains largely unexplored. This paper introduces a practical and effective training recipe that strategically integrates extended SFT with RL from online inference (GRPO). We posit that these methods play complementary, not competing, roles: a prolonged SFT phase first pushes the model's accuracy to its limits, after which a GRPO phase dramatically improves token efficiency while preserving this peak performance. Our experiments reveal that extending SFT for as many as 10 epochs is crucial for performance breakthroughs, and that the primary role of GRPO in this framework is to optimize solution length. The efficacy of our recipe is rigorously validated through top-tier performance on challenging benchmarks, including a high rank among over 2,200 teams in the strictly leak-free AI Mathematical Olympiad (AIMO). This work provides the community with a battle-tested blueprint for developing state-of-the-art mathematical reasoners that are both exceptionally accurate and practically efficient. To ensure full reproducibility and empower future research, we will open-source our entire framework, including all code, model checkpoints, and training configurations at https://github.com/analokmaus/kaggle-aimo2-fast-math-r1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17621v3">Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a pivotal method for improving the reasoning capabilities of Large Language Models (LLMs). However, prevalent RL approaches such as Proximal Policy Optimization (PPO) and Group-Regularized Policy Optimization (GRPO) face critical limitations due to their reliance on sparse outcome-based rewards and inadequate mechanisms for incentivizing exploration. These limitations result in inefficient guidance for multi-step reasoning processes. Specifically, sparse reward signals fail to deliver effective or sufficient feedback, particularly for challenging problems. Furthermore, such reward structures induce systematic biases that prioritize exploitation of familiar trajectories over novel solution discovery. These shortcomings critically hinder performance in complex reasoning tasks, which inherently demand iterative refinement across ipntermediate steps. To address these challenges, we propose an Intrinsic Motivation guidEd exploratioN meThOd foR LLM Reasoning (i-MENTOR), a novel method designed to both deliver dense rewards and amplify explorations in the RL-based training paradigm. i-MENTOR introduces three key innovations: trajectory-aware exploration rewards that mitigate bias in token-level strategies while maintaining computational efficiency; dynamic reward scaling to stabilize exploration and exploitation in large action spaces; and advantage-preserving reward implementation that maintains advantage distribution integrity while incorporating exploratory guidance. Experiments across three public datasets demonstrate i-MENTOR's effectiveness with a 22.39% improvement on the difficult dataset Countdown-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08255v1">Quantum-Accelerated Neural Imputation with Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Missing data presents a critical challenge in real-world datasets, significantly degrading the performance of machine learning models. While Large Language Models (LLMs) have recently demonstrated remarkable capabilities in tabular data imputation, exemplified by frameworks like UnIMP, their reliance on classical embedding methods often limits their ability to capture complex, non-linear correlations, particularly in mixed-type data scenarios encompassing numerical, categorical, and textual features. This paper introduces Quantum-UnIMP, a novel framework that integrates shallow quantum circuits into an LLM-based imputation architecture. Our core innovation lies in replacing conventional classical input embeddings with quantum feature maps generated by an Instantaneous Quantum Polynomial (IQP) circuit. This approach enables the model to leverage quantum phenomena such as superposition and entanglement, thereby learning richer, more expressive representations of data and enhancing the recovery of intricate missingness patterns. Our experiments on benchmark mixed-type datasets demonstrate that Quantum-UnIMP reduces imputation error by up to 15.2% for numerical features (RMSE) and improves classification accuracy by 8.7% for categorical features (F1-Score) compared to state-of-the-art classical and LLM-based methods. These compelling results underscore the profound potential of quantum-enhanced representations for complex data imputation tasks, even with near-term quantum hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08235v1">InsightBuild: LLM-Powered Causal Reasoning in Smart Building Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Smart buildings generate vast streams of sensor and control data, but facility managers often lack clear explanations for anomalous energy usage. We propose InsightBuild, a two-stage framework that integrates causality analysis with a fine-tuned large language model (LLM) to provide human-readable, causal explanations of energy consumption patterns. First, a lightweight causal inference module applies Granger causality tests and structural causal discovery on building telemetry (e.g., temperature, HVAC settings, occupancy) drawn from Google Smart Buildings and Berkeley Office datasets. Next, an LLM, fine-tuned on aligned pairs of sensor-level causes and textual explanations, receives as input the detected causal relations and generates concise, actionable explanations. We evaluate InsightBuild on two real-world datasets (Google: 2017-2022; Berkeley: 2018-2020), using expert-annotated ground-truth causes for a held-out set of anomalies. Our results demonstrate that combining explicit causal discovery with LLM-based natural language generation yields clear, precise explanations that assist facility managers in diagnosing and mitigating energy inefficiencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08232v1">Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Accepted to the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), co-located with ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxy students in the development of Intelligent Tutoring Systems (ITSs) and in piloting test questions. However, to what extent these proxy students accurately emulate the behavior and characteristics of real students remains an open question. To investigate this, we collected a dataset of 489 items from the National Assessment of Educational Progress (NAEP), covering mathematics and reading comprehension in grades 4, 8, and 12. We then apply an Item Response Theory (IRT) model to position 11 diverse and state-of-the-art LLMs on the same ability scale as real student populations. Our findings reveal that, without guidance, strong general-purpose models consistently outperform the average student at every grade, while weaker or domain-mismatched models may align incidentally. Using grade-enforcement prompts changes models' performance, but whether they align with the average grade-level student remains highly model- and prompt-specific: no evaluated model-prompt pair fits the bill across subjects and grades, underscoring the need for new training and evaluation strategies. We conclude by providing guidelines for the selection of viable proxies based on our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09037v1">ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 10 pages total (including appendix), ICML 2025 Workshop on Reliable and Responsible Foundation Models
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used as decision aids. However, users have diverse values and preferences that can affect their decision-making, which requires novel methods for LLM alignment and personalization. Existing LLM comparison tools largely focus on benchmarking tasks, such as knowledge-based question answering. In contrast, our proposed ALIGN system focuses on dynamic personalization of LLM-based decision-makers through prompt-based alignment to a set of fine-grained attributes. Key features of our system include robust configuration management, structured output generation with reasoning, and several algorithm implementations with swappable LLM backbones, enabling different types of analyses. Our user interface enables a qualitative, side-by-side comparison of LLMs and their alignment to various attributes, with a modular backend for easy algorithm integration. Additionally, we perform a quantitative analysis comparing alignment approaches in two different domains: demographic alignment for public opinion surveys and value alignment for medical triage decision-making. The entire ALIGN framework is open source and will enable new research on reliable, responsible, and personalized LLM-based decision-makers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09019v1">On Evaluating Performance of LLM Inference Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Model (LLM) inference systems has yielded significant efficiency improvements. However, our systematic analysis reveals that current evaluation methodologies frequently exhibit fundamental flaws, often manifesting as common evaluation anti-patterns that obscure true performance characteristics and impede scientific progress. Through a comprehensive examination of recent systems, we identify recurring anti-patterns across three key dimensions: Baseline Fairness, Evaluation Setup, and Metric Design. These anti-patterns are uniquely problematic for LLM inference due to its dual-phase nature combining distinct prefill and decode operations, its handling of highly heterogeneous workloads, and its strict temporal requirements for interactive use. We demonstrate how common anti-patterns -- such as inadequate baseline comparisons that conflate engineering effort with algorithmic novelty, workload selections that fail to represent production scenarios, and metric normalizations that hide substantial performance variability like generation stalls-lead to misleading conclusions. To address these challenges, we provide a comprehensive checklist derived from our analysis, establishing a framework for recognizing and avoiding these anti-patterns in favor of robust LLM inference evaluation. To demonstrate the practical application of our framework, we present a case study analyzing speculative decoding, a technique whose bursty, non-uniform token generation is easily misinterpreted when evaluated using approaches characteristic of these anti-patterns. Our work establishes a rigorous foundation for evaluation methodology, enabling meaningful comparisons, ensuring reproducible results, and ultimately accelerating genuine progress in LLM inference systems by moving beyond common anti-patterns to align evaluation with real-world requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08979v1">PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      We introduce Projection-based Reduction of Implicit Spurious bias in vision-language Models (PRISM), a new data-free and task-agnostic solution for bias mitigation in VLMs like CLIP. VLMs often inherit and amplify biases in their training data, leading to skewed predictions. PRISM is designed to debias VLMs without relying on predefined bias categories or additional external data. It operates in two stages: first, an LLM is prompted with simple class prompts to generate scene descriptions that contain spurious correlations. Next, PRISM uses our novel contrastive-style debiasing loss to learn a projection that maps the embeddings onto a latent space that minimizes spurious correlations while preserving the alignment between image and text embeddings.Extensive experiments demonstrate that PRISM outperforms current debiasing methods on the commonly used Waterbirds and CelebA datasets We make our code public at: https://github.com/MahdiyarMM/PRISM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08960v1">How to Train a Leader: Hierarchical Reasoning in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved strong performance on a wide range of complex reasoning tasks, yet further gains are often possible by leveraging the complementary strengths of multiple models. While multi-agent frameworks can improve solution quality by leveraging multiple LLMs, existing methods are often computationally expensive, both at training and inference time. In this work, we introduce a hierarchical multi-agent framework that addresses these challenges by training only a single leader LLM to coordinate a team of untrained peer agents. To this end, we propose Multi-agent guided Leader Policy \textbf{O}ptimization (MLPO), a novel approach which trains the leader to evaluate and synthesize agent responses without auxiliary value networks or explicit agent feedback. Leaders trained with MLPO exhibit improved performance not only when interacting with the agent team at inference time, but also enjoy improved performance when deployed in single-agent settings without the team. Empirical results on Big-Bench Hard (BBH), MATH, and MMLU demonstrate that our framework achieves substantial performance improvements over both single-agent and multi-agent baselines. Our results highlight the effectiveness and efficiency of training a single, flexible leader for collaborative reasoning in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v2">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08944v1">Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 ICML 2025 Workshop on MAS
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems have demonstrated remarkable promise for tackling complex tasks by breaking them down into subtasks that are iteratively planned, executed, observed, and refined. Despite their effectiveness, these systems often incur high latency because real-world problems frequently demand multiple iterative cycles of reasoning steps. To address this challenge, we propose M1-Parallel, a framework that concurrently runs multiple multi-agent teams in parallel to uncover distinct solution paths. By leveraging an event-driven communication model with asynchronous messaging, M1-Parallel efficiently capitalizes on the inherent diversity of valid plans to either reduce end-to-end latency or boost task completion rates. Our experiments on complex tasks show that M1-Parallel with early termination achieves up to $2.2\times$ speedup while preserving accuracy, and that M1-Parallel with aggregation yields higher task completion rates. We further investigate strategies aimed at encouraging diverse execution plans but observe no additional performance gains over repeated sampling. Overall, these findings underscore the potential of parallel plan execution for optimizing multi-agent systems for real-world, high-complexity reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08924v1">From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) requires robust benchmarks that encompass not only academic domains but also industrial fields to effectively evaluate their applicability in real-world scenarios. In this paper, we introduce two Korean expert-level benchmarks. KMMLU-Redux, reconstructed from the existing KMMLU, consists of questions from the Korean National Technical Qualification exams, with critical errors removed to enhance reliability. KMMLU-Pro is based on Korean National Professional Licensure exams to reflect professional knowledge in Korea. Our experiments demonstrate that these benchmarks comprehensively represent industrial knowledge in Korea. We release our dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08916v1">Evaluating LLMs in Medicine: A Call for Rigor, Transparency</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Objectives: To evaluate the current limitations of large language models (LLMs) in medical question answering, focusing on the quality of datasets used for their evaluation. Materials and Methods: Widely-used benchmark datasets, including MedQA, MedMCQA, PubMedQA, and MMLU, were reviewed for their rigor, transparency, and relevance to clinical scenarios. Alternatives, such as challenge questions in medical journals, were also analyzed to identify their potential as unbiased evaluation tools. Results: Most existing datasets lack clinical realism, transparency, and robust validation processes. Publicly available challenge questions offer some benefits but are limited by their small size, narrow scope, and exposure to LLM training. These gaps highlight the need for secure, comprehensive, and representative datasets. Conclusion: A standardized framework is critical for evaluating LLMs in medicine. Collaborative efforts among institutions and policymakers are needed to ensure datasets and methodologies are rigorous, unbiased, and reflective of clinical complexities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04219v2">Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their training objectives. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method - Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from the model. Our core idea is to leverage this collapse for unlearning by triggering collapse partially on the sensitive data. We theoretically analyze that our approach converges to the desired outcome, i.e. the LLM unlearns the information in the forget set. We empirically demonstrate that PMC overcomes two key limitations of existing unlearning approaches that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at https://www.cs.cit.tum.de/daml/partial-model-collapse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v1">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Under Review at Information and Software Technology
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10593v1">ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) applications are increasingly relying on external tools to extend their capabilities beyond text generation. However, current tool integration approaches suffer from fragmentation, protocol limitations, and implementation complexity, leading to substantial development overhead. This paper presents Toolregistry, a protocol-agnostic tool management library that simplifies tool registration, representation, execution, and lifecycle management via a unified interface. Our evaluation demonstrates that \toolregistry achieves 60-80% reduction in tool integration code, up to 3.1x performance improvements through concurrent execution, and 100% compatibility with OpenAI function calling standards. Real-world case studies show significant improvements in development efficiency and code maintainability across diverse integration scenarios. \toolregistry is open-source and available at https://github.com/Oaklight/ToolRegistry, with comprehensive documentation at https://toolregistry.readthedocs.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04426v2">Decoding AI Judgment: How LLMs Assess News Credibility and Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in workflows that involve evaluative processes. This raises the need to examine how such evaluations are built, what assumptions they rely on, and how their strategies diverge from those of humans. We benchmark six LLMs against expert ratings--NewsGuard and Media Bias/Fact Check (MBFC)--and against human judgments collected through a controlled experiment. To enable direct comparison, we implement a structured agentic framework in which both models and non-expert participants follow the same evaluation procedure: selecting criteria, retrieving content, and producing justifications. Despite output alignment, LLMs rely on different mechanisms: lexical associations and statistical priors replace contextual reasoning. This reliance produces systematic effects: political asymmetries, opaque justifications, and a tendency to confuse linguistic form with epistemic validity. Delegating judgment to such systems does not merely automate evaluation--it redefines it, shifting from normative reasoning to pattern-based approximation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11984v2">Understanding Chain-of-Thought in LLMs through Information Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance in complex reasoning tasks through the use of Chain-of-Thought (CoT) reasoning, allowing models to break down problems into manageable sub-tasks. However, existing CoT evaluation techniques either require annotated CoT data or fall short in accurately assessing intermediate reasoning steps, leading to high rates of false positives. In this paper, we formalize CoT reasoning in LLMs through an information-theoretic lens. Specifically, our framework quantifies the `information-gain' at each reasoning step, enabling the identification of failure modes in LLMs without the need for expensive annotated datasets. We demonstrate the efficacy of our approach through extensive experiments on toy arithmetic, GSM8K and PRM800k datasets, where it significantly outperforms existing outcome-based methods by providing more accurate insights into model performance on individual subtasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07745v1">On the capabilities of LLMs for classifying and segmenting time series of fruit picking motions into primitive actions</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 This paper is a Late Breaking Results report and it will be presented through a poster at the 34th IEEE International Conference on Robot and Human Interactive Communication (ROMAN), 2025 at Eindhoven, the Netherlands
    </div>
    <details class="paper-abstract">
      Despite their recent introduction to human society, Large Language Models (LLMs) have significantly affected the way we tackle mental challenges in our everyday lives. From optimizing our linguistic communication to assisting us in making important decisions, LLMs, such as ChatGPT, are notably reducing our cognitive load by gradually taking on an increasing share of our mental activities. In the context of Learning by Demonstration (LbD), classifying and segmenting complex motions into primitive actions, such as pushing, pulling, twisting etc, is considered to be a key-step towards encoding a task. In this work, we investigate the capabilities of LLMs to undertake this task, considering a finite set of predefined primitive actions found in fruit picking operations. By utilizing LLMs instead of simple supervised learning or analytic methods, we aim at making the method easily applicable and deployable in a real-life scenario. Three different fine-tuning approaches are investigated, compared on datasets captured kinesthetically, using a UR10e robot, during a fruit-picking scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07695v1">KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 21 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07644v1">PlanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 25 pages, 18 figures. Diagnostic benchmark for spatial reasoning in LLMs. Project page: https://OldDelorean.github.io/PlanQA/
    </div>
    <details class="paper-abstract">
      We introduce PlanQA, a diagnostic benchmark for evaluating geometric and spatial reasoning in large-language models (LLMs). PlanQA is grounded in structured representations of indoor scenes, such as kitchens, living rooms, and bedrooms, encoded in a symbolic format (e.g., JSON, XML layouts). The benchmark includes diverse question types that test not only metric and topological reasoning (e.g., distance, visibility, shortest paths) but also interior design constraints such as affordance, clearance, balance, and usability. Our results across a variety of frontier open-source and commercial LLMs show that while models may succeed in shallow queries, they often fail to simulate physical constraints, preserve spatial coherence, or generalize under layout perturbation. PlanQA uncovers a clear blind spot in today's LLMs: they do not consistently reason about real-world layouts. We hope that this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07630v1">Exploring the Limits of Model Compression in LLMs: A Knowledge Distillation Study on QA Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Accepted four publication at the 26th Meeting of the Special Interest on Discourse and Dialogue
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated outstanding performance across a range of NLP tasks, however, their computational demands hinder their deployment in real-world, resource-constrained environments. This work investigates the extent to which LLMs can be compressed using Knowledge Distillation (KD) while maintaining strong performance on Question Answering (QA) tasks. We evaluate student models distilled from the Pythia and Qwen2.5 families on two QA benchmarks, SQuAD and MLQA, under zero-shot and one-shot prompting conditions. Results show that student models retain over 90% of their teacher models' performance while reducing parameter counts by up to 57.1%. Furthermore, one-shot prompting yields additional performance gains over zero-shot setups for both model families. These findings underscore the trade-off between model efficiency and task performance, demonstrating that KD, combined with minimal prompting, can yield compact yet capable QA systems suitable for resource-constrained applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14382v2">Good/Evil Reputation Judgment of Celebrities by LLMs via Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The purpose of this paper is to examine whether large language models (LLMs) can understand what is good and evil with respect to judging good/evil reputation of celebrities. Specifically, we first apply a large language model (namely, ChatGPT) to the task of collecting sentences that mention the target celebrity from articles about celebrities on Web pages. Next, the collected sentences are categorized based on their contents by ChatGPT, where ChatGPT assigns a category name to each of those categories. Those assigned category names are referred to as "aspects" of each celebrity. Then, by applying the framework of retrieval augmented generation (RAG), we show that the large language model is quite effective in the task of judging good/evil reputation of aspects and descriptions of each celebrity. Finally, also in terms of proving the advantages of the proposed method over existing services incorporating RAG functions, we show that the proposed method of judging good/evil of aspects/descriptions of each celebrity significantly outperform an existing service incorporating RAG functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07548v1">From Requirements to Code: Understanding Developer Practices in LLM-Assisted Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 This paper has been accepted for publication at the 33rd IEEE International Requirements Engineering (RE) conference
    </div>
    <details class="paper-abstract">
      With the advent of generative LLMs and their advanced code generation capabilities, some people already envision the end of traditional software engineering, as LLMs may be able to produce high-quality code based solely on the requirements a domain expert feeds into the system. The feasibility of this vision can be assessed by understanding how developers currently incorporate requirements when using LLMs for code generation-a topic that remains largely unexplored. We interviewed 18 practitioners from 14 companies to understand how they (re)use information from requirements and other design artifacts to feed LLMs when generating code. Based on our findings, we propose a theory that explains the processes developers employ and the artifacts they rely on. Our theory suggests that requirements, as typically documented, are too abstract for direct input into LLMs. Instead, they must first be manually decomposed into programming tasks, which are then enriched with design decisions and architectural constraints before being used in prompts. Our study highlights that fundamental RE work is still necessary when LLMs are used to generate code. Our theory is important for contextualizing scientific approaches to automating requirements-centric SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11329v2">TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 14 pages, 16 figures. For source code, see https://github.com/microsoft/tokenweave
    </div>
    <details class="paper-abstract">
      Distributed inference of large language models (LLMs) can introduce overheads of up to 20% even over GPUs connected via high-speed interconnects such as NVLink. Multiple techniques have been proposed to mitigate these overheads by decomposing computations into finer-grained tasks and overlapping communication with sub-tasks as they complete. However, fine-grained decomposition of a large computation into many smaller computations on GPUs results in overheads. Furthermore, the communication itself uses many streaming multiprocessors (SMs), adding to the overhead. We present TokenWeave to address these challenges. TokenWeave proposes a Token-Splitting technique that divides the tokens in the inference batch into two approximately equal subsets in a wave-aware manner. The communication of one subset is then overlapped with the computation of the other. In addition, TokenWeave optimizes the order of the layer normalization computation with respect to communication operations and implements a novel fused AllReduce--RMSNorm kernel that carefully leverages Multimem instruction support available on NVIDIA Hopper GPUs. These optimizations allow TokenWeave to perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel enables the memory-bound RMSNorm to be overlapped with the other batch's computation, providing additional gains. Our evaluations demonstrate up to 1.29x speedup in latency and 1.26x higher throughput across multiple models and workloads. In several settings, TokenWeave results in better performance compared to an equivalent model with all communication removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05085v4">Multi-Head RAG: Solving Multi-Aspect Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) enhances the abilities of Large Language Models (LLMs) by enabling the retrieval of documents into the LLM context to provide more accurate and relevant responses. Existing RAG solutions do not focus on queries that may require fetching multiple documents with substantially different contents. Such queries occur frequently, but are challenging because the embeddings of these documents may be distant in the embedding space, making it hard to retrieve them all. This paper introduces Multi-Head RAG (MRAG), a novel scheme designed to address this gap with a simple yet powerful idea: leveraging activations of Transformer's multi-head attention layer, instead of the decoder layer, as keys for fetching multi-aspect documents. The driving observation is that different attention heads learn to capture different data aspects. Harnessing the corresponding activations results in embeddings that represent various facets of data items and queries, improving the retrieval accuracy for complex queries. We provide an evaluation methodology and metrics, multi-aspect datasets, and real-world use cases to demonstrate MRAG's effectiveness. We show MRAG's design advantages over 18 RAG baselines, empirical improvements of up to 20% in retrieval success ratios, and benefits for downstream LLM generation. MRAG can be seamlessly integrated with existing RAG frameworks and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07539v1">CEA-LIST at CheckThat! 2025: Evaluating LLMs as Detectors of Bias and Opinion in Text</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Notebook for the CheckThat! Lab at CLEF 2025
    </div>
    <details class="paper-abstract">
      This paper presents a competitive approach to multilingual subjectivity detection using large language models (LLMs) with few-shot prompting. We participated in Task 1: Subjectivity of the CheckThat! 2025 evaluation campaign. We show that LLMs, when paired with carefully designed prompts, can match or outperform fine-tuned smaller language models (SLMs), particularly in noisy or low-quality data settings. Despite experimenting with advanced prompt engineering techniques, such as debating LLMs and various example selection strategies, we found limited benefit beyond well-crafted standard few-shot prompts. Our system achieved top rankings across multiple languages in the CheckThat! 2025 subjectivity detection task, including first place in Arabic and Polish, and top-four finishes in Italian, English, German, and multilingual tracks. Notably, our method proved especially robust on the Arabic dataset, likely due to its resilience to annotation inconsistencies. These findings highlight the effectiveness and adaptability of LLM-based few-shot learning for multilingual sentiment tasks, offering a strong alternative to traditional fine-tuning, particularly when labeled data is scarce or inconsistent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02524v5">CheckEmbed: Effective Verification of LLM Solutions to Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming a wide range of domains, yet verifying their outputs remains a significant challenge, especially for complex open-ended tasks such as consolidation, summarization, and knowledge extraction. To address this, we introduce CheckEmbed (CE): a simple, scalable, and accurate verification method. CE reduces each LLM answer to a single embedding vector using powerful modern embedding LLM models like SFR-Embedding-Mistral. Prior methods such as BERTScore and SelfCheckGPT relied on weaker encoders like BERT, forcing them to operate at token or sentence granularity. In contrast, CE performs fast, semantically rich comparisons directly at the whole-answer level, overcoming key limitations in both accuracy and scalability. We conduct a comprehensive design and time complexity analysis across 13 verification baselines, including classical text scorers (e.g., BLEU), stability-based methods (e.g., SelfCheckGPT), and generative evaluators (e.g., LLM-as-a-Judge), which highlights the effectiveness, efficiency, versatility, and simplicity of CE. Empirical results show that CE reliably detects hallucinations in both closed and open-ended tasks. We further present evidence that CE generalizes beyond text to other modalities such as vision, establishing it as a practical and versatile verification framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07498v1">Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07441v1">SAND: Boosting LLM Agents with Self-Taught Action Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are commonly tuned with supervised finetuning on ReAct-style expert trajectories or preference optimization over pairwise rollouts. Most of these methods focus on imitating specific expert behaviors or promoting chosen reasoning thoughts and actions over rejected ones. However, without reasoning and comparing over alternatives actions, LLM agents finetuned with these methods may over-commit towards seemingly plausible but suboptimal actions due to limited action space exploration. To address this, in this paper we propose Self-taught ActioN Deliberation (SAND) framework, enabling LLM agents to explicitly deliberate over candidate actions before committing to one. To tackle the challenges of when and what to deliberate given large action space and step-level action evaluation, we incorporate self-consistency action sampling and execution-guided action critique to help synthesize step-wise action deliberation thoughts using the base model of the LLM agent. In an iterative manner, the deliberation trajectories are then used to finetune the LLM agent itself. Evaluating on two representative interactive agent tasks, SAND achieves an average 20% improvement over initial supervised finetuning and also outperforms state-of-the-art agent tuning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07421v1">SynthEHR-Eviction: Enhancing Eviction SDoH Detection with LLM-Augmented Synthetic EHR Data</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Equal contribution for the first two authors
    </div>
    <details class="paper-abstract">
      Eviction is a significant yet understudied social determinants of health (SDoH), linked to housing instability, unemployment, and mental health. While eviction appears in unstructured electronic health records (EHRs), it is rarely coded in structured fields, limiting downstream applications. We introduce SynthEHR-Eviction, a scalable pipeline combining LLMs, human-in-the-loop annotation, and automated prompt optimization (APO) to extract eviction statuses from clinical notes. Using this pipeline, we created the largest public eviction-related SDoH dataset to date, comprising 14 fine-grained categories. Fine-tuned LLMs (e.g., Qwen2.5, LLaMA3) trained on SynthEHR-Eviction achieved Macro-F1 scores of 88.8% (eviction) and 90.3% (other SDoH) on human validated data, outperforming GPT-4o-APO (87.8%, 87.3%), GPT-4o-mini-APO (69.1%, 78.1%), and BioBERT (60.7%, 68.3%), while enabling cost-effective deployment across various model sizes. The pipeline reduces annotation effort by over 80%, accelerates dataset creation, enables scalable eviction detection, and generalizes to other information extraction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07413v1">Hybrid LLM-Enhanced Intrusion Detection for Zero-Day Threats in IoT Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 6 pages, IEEE conference
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to intrusion detection by integrating traditional signature-based methods with the contextual understanding capabilities of the GPT-2 Large Language Model (LLM). As cyber threats become increasingly sophisticated, particularly in distributed, heterogeneous, and resource-constrained environments such as those enabled by the Internet of Things (IoT), the need for dynamic and adaptive Intrusion Detection Systems (IDSs) becomes increasingly urgent. While traditional methods remain effective for detecting known threats, they often fail to recognize new and evolving attack patterns. In contrast, GPT-2 excels at processing unstructured data and identifying complex semantic relationships, making it well-suited to uncovering subtle, zero-day attack vectors. We propose a hybrid IDS framework that merges the robustness of signature-based techniques with the adaptability of GPT-2-driven semantic analysis. Experimental evaluations on a representative intrusion dataset demonstrate that our model enhances detection accuracy by 6.3%, reduces false positives by 9.0%, and maintains near real-time responsiveness. These results affirm the potential of language model integration to build intelligent, scalable, and resilient cybersecurity defences suited for modern connected environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07406v1">Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 8 Pages, IEEE Conference
    </div>
    <details class="paper-abstract">
      Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07400v1">KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based agentic workflows have become a popular paradigm for coordinating multiple specialized agents to solve complex tasks. To improve serving efficiency, existing LLM systems employ prefix caching to reuse key-value (KV) tensors corresponding to agents' fixed prompts, thereby avoiding redundant computation across repeated invocations. However, current systems typically evict KV caches using a Least Recently Used (LRU) policy, which fails to anticipate future agent usage and often discards KV caches shortly before their reuse. This leads to frequent cache misses and substantial recomputation or swapping overhead. We present KVFlow, a workflow-aware KV cache management framework tailored for agentic workloads. KVFlow abstracts the agent execution schedule as an Agent Step Graph and assigns each agent a steps-to-execution value that estimates its temporal proximity to future activation. These values guide a fine-grained eviction policy at the KV node level, allowing KVFlow to preserve entries likely to be reused and efficiently manage shared prefixes in tree-structured caches. Moreover, KVFlow introduces a fully overlapped KV prefetching mechanism, which proactively loads required tensors from CPU to GPU in background threads for agents scheduled in the next step, thereby avoiding cache miss stalls during generation. Compared to SGLang with hierarchical radix cache, KVFlow achieves up to 1.83$\times$ speedup for single workflows with large prompts, and up to 2.19$\times$ speedup for scenarios with many concurrent workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06920v2">Rethinking Verification for LLM Code Generation: From Generation to Testing</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved notable success in code-generation benchmarks such as HumanEval and LiveCodeBench. However, a detailed examination reveals that these evaluation suites often comprise only a limited number of homogeneous test cases, resulting in subtle faults going undetected. This not only artificially inflates measured performance but also compromises accurate reward estimation in reinforcement learning frameworks utilizing verifiable rewards (RLVR). To address these critical shortcomings, we systematically investigate the test-case generation (TCG) task by proposing multi-dimensional metrics designed to rigorously quantify test-suite thoroughness. Furthermore, we introduce a human-LLM collaborative method (SAGA), leveraging human programming expertise with LLM reasoning capability, aimed at significantly enhancing both the coverage and the quality of generated test cases. In addition, we develop a TCGBench to facilitate the study of the TCG task. Experiments show that SAGA achieves a detection rate of 90.62% and a verifier accuracy of 32.58% on TCGBench. The Verifier Accuracy (Verifier Acc) of the code generation evaluation benchmark synthesized by SAGA is 10.78% higher than that of LiveCodeBench-v6. These results demonstrate the effectiveness of our proposed method. We hope this work contributes to building a scalable foundation for reliable LLM code evaluation, further advancing RLVR in code generation, and paving the way for automated adversarial test synthesis and adaptive benchmark integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05401v3">Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Facebook advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group, achieving an overall accuracy of 88.55%. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. In addition to evaluating the effectiveness of LLMs in detecting microtargeted messaging, we conduct a comprehensive fairness analysis to identify potential biases in model predictions. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of senior citizens and male audiences. By showcasing the efficacy of LLMs in dissecting and explaining targeted communication strategies and by highlighting fairness concerns, this study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08208v1">Reasoning and Behavioral Equilibria in LLM-Nash Games: From Mindsets to Actions</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      We introduce the LLM-Nash framework, a game-theoretic model where agents select reasoning prompts to guide decision-making via Large Language Models (LLMs). Unlike classical games that assume utility-maximizing agents with full rationality, this framework captures bounded rationality by modeling the reasoning process explicitly. Equilibrium is defined over the prompt space, with actions emerging as the behavioral output of LLM inference. This approach enables the study of cognitive constraints, mindset expressiveness, and epistemic learning. Through illustrative examples, we show how reasoning equilibria can diverge from classical Nash outcomes, offering a new foundation for strategic interaction in LLM-enabled systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08207v1">A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08203v1">TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs)inevitably produce untruthful responses. Accurately predicting the truthfulness of these outputs is critical, especially in high-stakes settings. To accelerate research in this domain and make truthfulness prediction methods more accessible, we introduce TruthTorchLM an open-source, comprehensive Python library featuring over 30 truthfulness prediction methods, which we refer to as Truth Methods. Unlike existing toolkits such as Guardrails, which focus solely on document-grounded verification, or LM-Polygraph, which is limited to uncertainty-based methods, TruthTorchLM offers a broad and extensible collection of techniques. These methods span diverse tradeoffs in computational cost, access level (e.g., black-box vs white-box), grounding document requirements, and supervision type (self-supervised or supervised). TruthTorchLM is seamlessly compatible with both HuggingFace and LiteLLM, enabling support for locally hosted and API-based models. It also provides a unified interface for generation, evaluation, calibration, and long-form truthfulness prediction, along with a flexible framework for extending the library with new methods. We conduct an evaluation of representative truth methods on three datasets, TriviaQA, GSM8K, and FactScore-Bio. The code is available at https://github.com/Ybakman/TruthTorchLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06565v2">The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 27 pages, 3 figures, 4 tables, 1 algorithm, 48 references
    </div>
    <details class="paper-abstract">
      Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08881v1">The Consistency-Acceptability Divergence of LLMs in Judicial Decision-Making: Task and Stakeholder Dimensions</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 12 pages,2 figures
    </div>
    <details class="paper-abstract">
      The integration of large language model (LLM) technology into judicial systems is fundamentally transforming legal practice worldwide. However, this global transformation has revealed an urgent paradox requiring immediate attention. This study introduces the concept of ``consistency-acceptability divergence'' for the first time, referring to the gap between technical consistency and social acceptance. While LLMs achieve high consistency at the technical level, this consistency demonstrates both positive and negative effects. Through comprehensive analysis of recent data on LLM judicial applications from 2023--2025, this study finds that addressing this challenge requires understanding both task and stakeholder dimensions. This study proposes the Dual-Track Deliberative Multi-Role LLM Judicial Governance Framework (DTDMR-LJGF), which enables intelligent task classification and meaningful interaction among diverse stakeholders. This framework offers both theoretical insights and practical guidance for building an LLM judicial ecosystem that balances technical efficiency with social legitimacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08877v1">ODIA: Oriented Distillation for Inline Acceleration of LLM-based Function Calling</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Function Calling is a crucial technique that enables Large Language Models (LLMs) to interact with external systems through APIs. However, the high latency associated with LLM-based Function Calling significantly impacts user experience. This paper presents a novel approach called Oriented Distillation for Inline Acceleration (ODIA) that leverages online user interaction data to accelerate Function Calling. By automatically identifying "simple queries" from production traffic and distilling knowledge from larger models to smaller ones, our method reduces response latency by 45% (expected) and 78% (median) while maintaining accuracy. We demonstrate the effectiveness of our approach through real-world deployment in a music application, where the smaller model successfully handles 60% of traffic with negligible accuracy loss. Our method requires minimal human intervention and continuously improves through automated data collection and model updating, making it a practical solution for production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07996v1">Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Can a pretrained neural network adapt its architecture to different inputs without any finetuning? Do we need all layers for simple tasks, and are they adequate for challenging tasks? We found that the layers of a pretrained large language model (LLM) can be manipulated as separate modules to build a better and even shallower model customized for each test sample. In particular, each layer from the pretrained model can be skipped/pruned or repeated multiple times as recurrent neural networks (RNN), and stacked with others in arbitrary orders, yielding a chain-of-layers (CoLa) per sample. This compositional space greatly expands the scope of existing works on looped/recurrent pretrained modules, layer pruning, or early-exit networks. We develop a Monte Carlo Tree Search (MCTS) protocol to explore and identify the optimal CoLa for each sample from math and commonsense reasoning benchmarks. Compared to a static model of a fixed depth, CoLa allows shortcut paths (fast thinking), recurrence of the same layer(s) (slow thinking), and combining both, offering more flexible, dynamic architectures for different inputs. We conduct an extensive analysis of the MCTS-optimized CoLa, which leads to two key findings: (1) For >75% of samples with correct predictions by the original LLM, we can find shorter CoLa, suggesting a large space for improving inference efficiency; (2) For >60% of samples with originally incorrect predictions, we can identify CoLa achieving correct predictions, suggesting a large space of performance enhancement. Our results highlight the shortcomings of using a fixed architecture of pre-trained LLMs for inference on different samples and pave the way to unlock the generalization power of test-time depth adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07990v1">Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Accepted at ICCV2025; Project page: https://www.jshyun.me/projects/sttm
    </div>
    <details class="paper-abstract">
      Video large language models (LLMs) achieve strong video understanding by leveraging a large number of spatio-temporal tokens, but suffer from quadratic computational scaling with token count. To address this, we propose a training-free spatio-temporal token merging method, named STTM. Our key insight is to exploit local spatial and temporal redundancy in video data which has been overlooked in prior work. STTM first transforms each frame into multi-granular spatial tokens using a coarse-to-fine search over a quadtree structure, then performs directed pairwise merging across the temporal dimension. This decomposed merging approach outperforms existing token reduction methods across six video QA benchmarks. Notably, STTM achieves a 2$\times$ speed-up with only a 0.5% accuracy drop under a 50% token budget, and a 3$\times$ speed-up with just a 2% drop under a 30% budget. Moreover, STTM is query-agnostic, allowing KV cache reuse across different questions for the same video. The project page is available at https://www.jshyun.me/projects/sttm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14937v2">Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Transactions of Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07957v1">MIRIX: Multi-Agent Memory System for LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Although memory capabilities of AI agents are gaining increasing attention, existing solutions remain fundamentally limited. Most rely on flat, narrowly scoped memory components, constraining their ability to personalize, abstract, and reliably recall user-specific information over time. To this end, we introduce MIRIX, a modular, multi-agent memory system that redefines the future of AI memory by solving the field's most critical challenge: enabling language models to truly remember. Unlike prior approaches, MIRIX transcends text to embrace rich visual and multimodal experiences, making memory genuinely useful in real-world scenarios. MIRIX consists of six distinct, carefully structured memory types: Core, Episodic, Semantic, Procedural, Resource Memory, and Knowledge Vault, coupled with a multi-agent framework that dynamically controls and coordinates updates and retrieval. This design enables agents to persist, reason over, and accurately retrieve diverse, long-term user data at scale. We validate MIRIX in two demanding settings. First, on ScreenshotVQA, a challenging multimodal benchmark comprising nearly 20,000 high-resolution computer screenshots per sequence, requiring deep contextual understanding and where no existing memory systems can be applied, MIRIX achieves 35% higher accuracy than the RAG baseline while reducing storage requirements by 99.9%. Second, on LOCOMO, a long-form conversation benchmark with single-modal textual input, MIRIX attains state-of-the-art performance of 85.4%, far surpassing existing baselines. These results show that MIRIX sets a new performance standard for memory-augmented LLM agents. To allow users to experience our memory system, we provide a packaged application powered by MIRIX. It monitors the screen in real time, builds a personalized memory base, and offers intuitive visualization and secure local storage to ensure privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14993v2">Multi-modal Generative AI: Multi-modal LLMs, Diffusions and the Unification</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 20 pages, 11 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03296v3">Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads. We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like VLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 49% (T4) and 37% (A10) higher throughput in long-output settings. APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07870v1">DocCHA: Towards LLM-Augmented Interactive Online diagnosis System</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Despite the impressive capabilities of Large Language Models (LLMs), existing Conversational Health Agents (CHAs) remain static and brittle, incapable of adaptive multi-turn reasoning, symptom clarification, or transparent decision-making. This hinders their real-world applicability in clinical diagnosis, where iterative and structured dialogue is essential. We propose DocCHA, a confidence-aware, modular framework that emulates clinical reasoning by decomposing the diagnostic process into three stages: (1) symptom elicitation, (2) history acquisition, and (3) causal graph construction. Each module uses interpretable confidence scores to guide adaptive questioning, prioritize informative clarifications, and refine weak reasoning links. Evaluated on two real-world Chinese consultation datasets (IMCS21, DX), DocCHA consistently outperforms strong prompting-based LLM baselines (GPT-3.5, GPT-4o, LLaMA-3), achieving up to 5.18 percent higher diagnostic accuracy and over 30 percent improvement in symptom recall, with only modest increase in dialogue turns. These results demonstrate the effectiveness of DocCHA in enabling structured, transparent, and efficient diagnostic conversations -- paving the way for trustworthy LLM-powered clinical assistants in multilingual and resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v2">The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v5">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02357v2">Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint. This work has been submitted to the Technical AI Governance Workshop at ICML 2025 for review
    </div>
    <details class="paper-abstract">
      Credible safety plans for advanced AI development require methods to verify agent behavior and detect potential control deficiencies early. A fundamental aspect is ensuring agents adhere to safety-critical principles, especially when these conflict with operational goals. This paper introduces a lightweight, interpretable benchmark to evaluate an LLM agent's ability to uphold a high-level safety principle when faced with conflicting task instructions. Our evaluation of six LLMs reveals two primary findings: (1) a quantifiable "cost of compliance" where safety constraints degrade task performance even when compliant solutions exist, and (2) an "illusion of compliance" where high adherence often masks task incompetence rather than principled choice. These findings provide initial evidence that while LLMs can be influenced by hierarchical directives, current approaches lack the consistency required for reliable safety governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01936v2">The Thin Line Between Comprehension and Persuasion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16903v2">GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/
    </div>
    <details class="paper-abstract">
      Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06774v1">Checklist Engineering Empowers Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04446v2">Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06762v1">Leveraging LLMs for Semantic Conflict Detection via Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Comments: 11 pages, in Portuguese language. 3 figures. Submitted to SAST 2025 (X Simp\'osio Brasileiro de Teste de Software Sistem\'atico e Automatizado)
    </div>
    <details class="paper-abstract">
      Semantic conflicts arise when a developer introduces changes to a codebase that unintentionally affect the behavior of changes integrated in parallel by other developers. Traditional merge tools are unable to detect such conflicts, so complementary tools like SMAT have been proposed. SMAT relies on generating and executing unit tests: if a test fails on the base version, passes on a developer's modified version, but fails again after merging with another developer's changes, a semantic conflict is indicated. While SMAT is effective at detecting conflicts, it suffers from a high rate of false negatives, partly due to the limitations of unit test generation tools such as Randoop and Evosuite. To investigate whether large language models (LLMs) can overcome these limitations, we propose and integrate a new test generation tool based on Code Llama 70B into SMAT. We explore the model's ability to generate tests using different interaction strategies, prompt contents, and parameter configurations. Our evaluation uses two samples: a benchmark with simpler systems from related work, and a more significant sample based on complex, real-world systems. We assess the effectiveness of the new SMAT extension in detecting conflicts. Results indicate that, although LLM-based test generation remains challenging and computationally expensive in complex scenarios, there is promising potential for improving semantic conflict detection. -- Conflitos sem^anticos surgem quando um desenvolvedor introduz mudan\c{c}as em uma base de c\'odigo que afetam, de forma n~ao intencional, o comportamento de altera\c{c}~oes integradas em paralelo por outros desenvolvedores. Ferramentas tradicionais de merge n~ao conseguem detectar esse tipo de conflito, por isso ferramentas complementares como o SMAT foram propostas. O SMAT depende da gera\c{c}~ao e execu\c{c}~ao de testes de unidade: se um teste falha na vers~ao base, passa na vers~ao modificada por um desenvolvedor, mas volta a falhar ap\'os o merge com as mudan\c{c}as de outro desenvolvedor, um conflito sem^antico \'e identificado. Embora o SMAT seja eficaz na detec\c{c}~ao de conflitos, apresenta alta taxa de falsos negativos, em parte devido \`as limita\c{c}~oes das ferramentas de gera\c{c}~ao de testes como Randoop e Evosuite. Para investigar se modelos de linguagem de grande porte (LLMs) podem superar essas limita\c{c}~oes, propomos e integramos ao SMAT uma nova ferramenta de gera\c{c}~ao de testes baseada no Code Llama 70B. Exploramos a capacidade do modelo de gerar testes utilizando diferentes estrat\'egias de intera\c{c}~ao, conte\'udos de prompts e configura\c{c}~oes de par^ametros. Nossa avalia\c{c}~ao utiliza duas amostras: um benchmark com sistemas mais simples, usados em trabalhos relacionados, e uma amostra mais significativa baseada em sistemas complexos e reais. Avaliamos a efic\'acia da nova extens~ao do SMAT na detec\c{c}~ao de conflitos. Os resultados indicam que, embora a gera\c{c}~ao de testes por LLM em cen\'arios complexos ainda seja desafiadora e custosa computacionalmente, h\'a potencial promissor para aprimorar a detec\c{c}~ao de conflitos sem^anticos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03785v3">Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted to GEM @ ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14541v2">LLM-based User Profile Management for Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted GENNEXT@SIGIR'25 Workshop
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06719v1">A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06715v1">CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), including zero-shot and few-shot paradigms, have shown promising capabilities in clinical text generation. However, real-world applications face two key challenges: (1) patient data is highly unstructured, heterogeneous, and scattered across multiple note types and (2) clinical notes are often long and semantically dense, making naive prompting infeasible due to context length constraints and the risk of omitting clinically relevant information. We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a domain-specific framework for structured and clinically grounded text generation using LLMs. It incorporates a novel hierarchical chunking strategy that respects clinical document structure and introduces a task-specific dual-stage retrieval mechanism. The global stage identifies relevant note types using evidence-based queries, while the local stage extracts high-value content within those notes creating relevance at both document and section levels. We apply the system to generate structured progress notes for individual hospital visits using 15 clinical note types from the MIMIC-III dataset. Experiments show that it preserves temporal and semantic alignment across visits, achieving an average alignment score of 87.7%, surpassing the 80.7% baseline from real clinician-authored notes. The generated outputs also demonstrate high consistency across LLMs, reinforcing deterministic behavior essential for reproducibility, reliability, and clinical trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07675v2">QUITE: A Query Rewrite System Beyond Rules with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18951v2">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 26 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06623v1">Expediting data extraction using a large language model (LLM) and scoping review protocol: a methodological study within a complex scoping review</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 44 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The data extraction stages of reviews are resource-intensive, and researchers may seek to expediate data extraction using online (large language models) LLMs and review protocols. Claude 3.5 Sonnet was used to trial two approaches that used a review protocol to prompt data extraction from 10 evidence sources included in a case study scoping review. A protocol-based approach was also used to review extracted data. Limited performance evaluation was undertaken which found high accuracy for the two extraction approaches (83.3% and 100%) when extracting simple, well-defined citation details; accuracy was lower (9.6% and 15.8%) when extracting more complex, subjective data items. Considering all data items, both approaches had precision >90% but low recall (<25%) and F1 scores (<40%). The context of a complex scoping review, open response types and methodological approach likely impacted performance due to missed and misattributed data. LLM feedback considered the baseline extraction accurate and suggested minor amendments: four of 15 (26.7%) to citation details and 8 of 38 (21.1%) to key findings data items were considered to potentially add value. However, when repeating the process with a dataset featuring deliberate errors, only 2 of 39 (5%) errors were detected. Review-protocol-based methods used for expediency require more robust performance evaluation across a range of LLMs and review contexts with comparison to conventional prompt engineering approaches. We recommend researchers evaluate and report LLM performance if using them similarly to conduct data extraction or review extracted data. LLM feedback contributed to protocol adaptation and may assist future review protocol drafting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23132v2">LAURA: LLM-Assisted UAV Routing for AoI Minimization</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      With the rapid growth of the low-altitude economy, there is increasing demand for real-time data collection using UAV-assisted wireless sensor networks. This paper investigates the problem of minimizing the age of information (AoI) in UAV-assisted wireless sensor networks by optimizing the UAV flight routing. We formulate the AoI minimization task and propose a large language model (LLM)-assisted UAV routing algorithm (LAURA). LAURA employs an LLM as intelligent crossover operators within an evolutionary optimization framework to efficiently explore the solution space. Simulation results show that LAURA outperforms benchmark methods in reducing the maximum AoI, especially in scenarios with a large number of sensor nodes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06573v1">From Data-Centric to Sample-Centric: Enhancing LLM Reasoning via Progressive Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has recently advanced the reasoning capabilities of large language models (LLMs). While prior work has emphasized algorithmic design, data curation, and reward shaping, we investigate RLVR from a sample-centric perspective and introduce LPPO (Learning-Progress and Prefix-guided Optimization), a framework of progressive optimization techniques. Our work addresses a critical question: how to best leverage a small set of trusted, high-quality demonstrations, rather than simply scaling up data volume. First, motivated by how hints aid human problem-solving, we propose prefix-guided sampling, an online data augmentation method that incorporates partial solution prefixes from expert demonstrations to guide the policy, particularly for challenging instances. Second, inspired by how humans focus on important questions aligned with their current capabilities, we introduce learning-progress weighting, a dynamic strategy that adjusts each training sample's influence based on model progression. We estimate sample-level learning progress via an exponential moving average of per-sample pass rates, promoting samples that foster learning and de-emphasizing stagnant ones. Experiments on mathematical-reasoning benchmarks demonstrate that our methods outperform strong baselines, yielding faster convergence and a higher performance ceiling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12538v2">PersonaFlow: Designing LLM-Simulated Expert Perspectives for Enhanced Research Ideation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted to DIS2025
    </div>
    <details class="paper-abstract">
      Generating interdisciplinary research ideas requires diverse domain expertise, but access to timely feedback is often limited by the availability of experts. In this paper, we introduce PersonaFlow, a novel system designed to provide multiple perspectives by using LLMs to simulate domain-specific experts. Our user studies showed that the new design 1) increased the perceived relevance and creativity of ideated research directions, and 2) promoted users' critical thinking activities (e.g., interpretation, analysis, evaluation, inference, and self-regulation), without increasing their perceived cognitive load. Moreover, users' ability to customize expert profiles significantly improved their sense of agency, which can potentially mitigate their over-reliance on AI. This work contributes to the design of intelligent systems that augment creativity and collaboration, and provides design implications of using customizable AI-simulated personas in domains within and beyond research ideation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09073v3">CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 full draft v2: 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various NLP tasks but struggle with code-mixed (or code-switched) language understanding. For example, prior work benchmarking the performance of multilingual LLMs on code-mixed translation tasks has demonstrated that current state-of-the-art multilingual LLMs are ineffective in dealing with code-mixed languages. However, the question of how to improve the capability of multilingual LLMs to handle code-mixed language has not received any attention to date. In this paper, we tackle this research gap by proposing CHAI, a novel general-purpose framework for improving the ability of multilingual LLMs to handle code-mixed languages. CHAI relies on three novel contributions made in this paper. First, we explore the ability of LLMs to provide accurate annotations for code-mixed translation tasks. Second, we leverage this ability of LLMs as annotators to generate preference data for code-mixed translation tasks at scale, which are then used within a reinforcement learning from AI feedback (RLAIF) procedure to improve LLMs' capability on code-mixed tasks. Third, we conduct a rigorous experimental evaluation across various real-world datasets and settings. Our analysis shows that CHAI-powered LLMs outperform state-of-the-art open-source LLMs by 25.66% (in terms of win rate adjudicated by human annotators) in code-mixed translation tasks. This work represents a first step towards developing more inclusive code-mixed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06565v1">The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 27 pages, 3 figures, 4 tables, 1 algorithm, 28 references
    </div>
    <details class="paper-abstract">
      Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19030v2">WiLLM: an Open Framework for LLM Services over Wireless Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) services fundamentally differ from traditional Deep Neural Network (DNN) applications in wireless networks. We identify three critical distinctions: (1) unlike traditional DNNs with unidirectional data flows, LLM's multimodal interactions create bidirectional heavy loads with contrasting bottlenecks, requiring direction-aware resource scheduling; (2) while traditional DNNs exhibit fixed computational patterns, LLM's highly variable inference times interact complexly with network slicing, causing dynamic bottleneck migration; and (3) in contrast to predictable DNN traffic, LLM's token streams demonstrate unprecedented burstiness and state dependencies. These insights motivate WiLLM, the first open-source framework, implemented as a wireless platform, for LLM service research. Built on OpenAirInterface, WiLLM introduces several technical innovations: dynamic slice compatibility, universal UE compatibility through application-layer tunneling, multi-UE multi-slice scheduling, dual-mode resource allocation, and cross-layer APIs. In addition, WiLLM eliminates the need for specialized wireless expertise, enabling researchers and developers to experiment with LLM services over realistic cellular networks. We demonstrate the platform's capabilities through a smart glasses case study and provide a comprehensive dataset of \~1.6 million synchronized measurements. The complete system, dataset, and appendix are available at https://openwillm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12399v2">FinSphere, a Real-Time Stock Analysis Agent Powered by Instruction-Tuned LLMs and Domain Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Current financial large language models (FinLLMs) struggle with two critical limitations: the absence of objective evaluation metrics to assess the quality of stock analysis reports and a lack of depth in stock analysis, which impedes their ability to generate professional-grade insights. To address these challenges, this paper introduces FinSphere, a stock analysis agent, along with three major contributions: (1) AnalyScore, a systematic evaluation framework for assessing stock analysis quality, (2) Stocksis, a dataset curated by industry experts to enhance LLMs' stock analysis capabilities, and (3) FinSphere, an AI agent that can generate high-quality stock analysis reports in response to user queries. Experiments demonstrate that FinSphere achieves superior performance compared to both general and domain-specific LLMs, as well as existing agent-based systems, even when they are enhanced with real-time data access and few-shot guidance. The integrated framework, which combines real-time data feeds, quantitative tools, and an instruction-tuned LLM, yields substantial improvements in both analytical quality and practical applicability for real-world stock analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12022v3">Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06520v1">Gradientsys: A Multi-Agent LLM Scheduler with ReAct Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      We present Gradientsys, a next-generation multi-agent scheduling framework that coordinates diverse specialized AI agents using a typed Model-Context Protocol (MCP) and a ReAct-based dynamic planning loop. At its core, Gradientsys employs an LLM-powered scheduler for intelligent one-to-many task dispatch, enabling parallel execution of heterogeneous agents such as PDF parsers, web search modules, GUI controllers, and web builders. The framework supports hybrid synchronous/asynchronous execution, respects agent capacity constraints, and incorporates a robust retry-and-replan mechanism to handle failures gracefully. To promote transparency and trust, Gradientsys includes an observability layer streaming real-time agent activity and intermediate reasoning via Server-Sent Events (SSE). We offer an architectural overview and evaluate Gradientsys against existing frameworks in terms of extensibility, scheduling topology, tool reusability, parallelism, and observability. Experiments on the GAIA general-assistant benchmark show that Gradientsys achieves higher task success rates with reduced latency and lower API costs compared to a MinionS-style baseline, demonstrating the strength of its LLM-driven multi-agent orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21285v2">Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. Our codes and data are available at https://github.com/XinXU-USTC/DoubleChecker
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17946v4">Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11951v3">SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 13 pages, 10 tables, 5 figures
    </div>
    <details class="paper-abstract">
      This paper introduces SagaLLM, a structured multi-agent architecture designed to address four foundational limitations of current LLM-based planning systems: unreliable self-validation, context loss, lack of transactional safeguards, and insufficient inter-agent coordination. While recent frameworks leverage LLMs for task decomposition and multi-agent communication, they often fail to ensure consistency, rollback, or constraint satisfaction across distributed workflows. SagaLLM bridges this gap by integrating the Saga transactional pattern with persistent memory, automated compensation, and independent validation agents. It leverages LLMs' generative reasoning to automate key tasks traditionally requiring hand-coded coordination logic, including state tracking, dependency analysis, log schema generation, and recovery orchestration. Although SagaLLM relaxes strict ACID guarantees, it ensures workflow-wide consistency and recovery through modular checkpointing and compensable execution. Empirical evaluations across planning domains demonstrate that standalone LLMs frequently violate interdependent constraints or fail to recover from disruptions. In contrast, SagaLLM achieves significant improvements in consistency, validation accuracy, and adaptive coordination under uncertainty, establishing a robust foundation for real-world, scalable LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06512v1">Towards LLM-based Root Cause Analysis of Hardware Design Failures</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 6 pages. Accepted for publication in IEEE COINS 2025 Special Session on LLMs for EDA and Security
    </div>
    <details class="paper-abstract">
      With advances in large language models (LLMs), new opportunities have emerged to develop tools that support the digital hardware design process. In this work, we explore how LLMs can assist with explaining the root cause of design issues and bugs that are revealed during synthesis and simulation, a necessary milestone on the pathway towards widespread use of LLMs in the hardware design process and for hardware security analysis. We find promising results: for our corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model reached a correct determination 100% of the time under pass@5 scoring, with other state of the art models and configurations usually achieving more than 80% performance and more than 90% when assisted with retrieval-augmented generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06489v1">On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v3">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06463v1">Evaluating Efficiency and Novelty of LLM-Generated Code for Graph Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate software development, yet most prior evaluations focus on functional correctness or high-level languages such as Python. We present the first systematic study of LLMs' ability to generate efficient C implementations of graph-analysis routines--code that must satisfy the stringent runtime and memory constraints. Eight state-of-the-art models (OpenAI ChatGPT o3 and o4-mini-high, Anthropic Claude 4 Sonnet and Sonnet Extended, Google Gemini 2.5 Flash and Pro, xAI Grok 3-Think, and DeepSeek DeepThink R1) are benchmarked by two distinct approaches. The first approach checks the ability of LLMs in generating an algorithm outperforming other present algorithms in the benchmark. The second approach evaluates the ability of LLMs to generate graph algorithms for integration into the benchmark. Results show that Claude Sonnet 4 Extended achieves the best result in the case of ready-to-use code generation and efficiency, outperforming human-written baselines in triangle counting. The study confirms that contemporary LLMs excel at optimizing and integrating established algorithms but not inventing novel techniques. We provide prompts, the first approach's generated code, and measurement scripts to foster reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14879v3">Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-In-The-Loop LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Passive tracking methods, such as phone and wearable sensing, have become dominant in monitoring human behaviors in modern ubiquitous computing studies. While there have been significant advances in machine-learning approaches to translate periods of raw sensor data to model momentary behaviors, (e.g., physical activity recognition), there still remains a significant gap in the translation of these sensing streams into meaningful, high-level, context-aware insights that are required for various applications (e.g., summarizing an individual's daily routine). To bridge this gap, experts often need to employ a context-driven sensemaking process in real-world studies to derive insights. This process often requires manual effort and can be challenging even for experienced researchers due to the complexity of human behaviors. We conducted three rounds of user studies with 21 experts to explore solutions to address challenges with sensemaking. We follow a human-centered design process to identify needs and design, iterate, build, and evaluate Vital Insight (VI), a novel, LLM-assisted, prototype system to enable human-in-the-loop inference (sensemaking) and visualizations of multi-modal passive sensing data from smartphones and wearables. Using the prototype as a technology probe, we observe experts' interactions with it and develop an expert sensemaking model that explains how experts move between direct data representations and AI-supported inferences to explore, question, and validate insights. Through this iterative process, we also synthesize and discuss a list of design implications for the design of future AI-augmented visualization systems to better assist experts' sensemaking processes in multi-modal health sensing data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07328v1">Bridging the Plausibility-Validity Gap by Fine-Tuning a Reasoning-Enhanced LLM for Chemical Synthesis and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 42 pages, 8 figures, 1 equation, 2 algorithms, 31 tables, to be published in ISPACS Conference 2025, unabridged version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate scientifically plausible but factually invalid information, a challenge we term the "plausibility-validity gap," particularly in specialized domains like chemistry. This paper presents a systematic methodology to bridge this gap by developing a specialized scientific assistant. We utilized the Magistral Small model, noted for its integrated reasoning capabilities, and fine-tuned it using Low-Rank Adaptation (LoRA). A key component of our approach was the creation of a "dual-domain dataset," a comprehensive corpus curated from various sources encompassing both molecular properties and chemical reactions, which was standardized to ensure quality. Our evaluation demonstrates that the fine-tuned model achieves significant improvements over the baseline model in format adherence, chemical validity of generated molecules, and the feasibility of proposed synthesis routes. The results indicate a hierarchical learning pattern, where syntactic correctness is learned more readily than chemical possibility and synthesis feasibility. While a comparative analysis with human experts revealed competitive performance in areas like chemical creativity and reasoning, it also highlighted key limitations, including persistent errors in stereochemistry, a static knowledge cutoff, and occasional reference hallucination. This work establishes a viable framework for adapting generalist LLMs into reliable, specialized tools for chemical research, while also delineating critical areas for future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07313v1">Frontier LLMs Still Struggle with Simple Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 53 pages
    </div>
    <details class="paper-abstract">
      While state-of-the-art large language models (LLMs) demonstrate advanced reasoning capabilities-achieving remarkable performance on challenging competitive math and coding benchmarks-they also frequently fail on tasks that are easy for humans. This work studies the performance of frontier LLMs on a broad set of such "easy" reasoning problems. By extending previous work in the literature, we create a suite of procedurally generated simple reasoning tasks, including counting, first-order logic, proof trees, and travel planning, with changeable parameters (such as document length. or the number of variables in a math problem) that can arbitrarily increase the amount of computation required to produce the answer while preserving the fundamental difficulty. While previous work showed that traditional, non-thinking models can be made to fail on such problems, we demonstrate that even state-of-the-art thinking models consistently fail on such problems and for similar reasons (e.g. statistical shortcuts, errors in intermediate steps, and difficulties in processing long contexts). To further understand the behavior of the models, we introduce the unpuzzles dataset, a different "easy" benchmark consisting of trivialized versions of well-known math and logic puzzles. Interestingly, while modern LLMs excel at solving the original puzzles, they tend to fail on the trivialized versions, exhibiting several systematic failure patterns related to memorizing the originals. We show that this happens even if the models are otherwise able to solve problems with different descriptions but requiring the same logic. Our results highlight that out-of-distribution generalization is still problematic for frontier language models and the new generation of thinking models, even for simple reasoning tasks, and making tasks easier does not necessarily imply improved performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19092v2">Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integral to information retrieval (IR), powering ranking, evaluation, and AI-assisted content creation. This widespread adoption necessitates a critical examination of potential biases arising from the interplay between these LLM-based components. This paper synthesizes existing research and presents novel experiment designs that explore how LLM-based rankers and assistants influence LLM-based judges. We provide the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers. Furthermore, we observe limitations in LLM judges' ability to discern subtle system performance differences. Contrary to some previous findings, our preliminary study does not find evidence of bias against AI-generated content. These results highlight the need for a more holistic view of the LLM-driven information ecosystem. To this end, we offer initial guidelines and a research agenda to ensure the reliable use of LLMs in IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07251v1">A Language-Driven Framework for Improving Personalized Recommendations: Merging LLMs with Traditional Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Traditional recommendation algorithms are not designed to provide personalized recommendations based on user preferences provided through text, e.g., "I enjoy light-hearted comedies with a lot of humor". Large Language Models (LLMs) have emerged as one of the most promising tools for natural language processing in recent years. This research proposes a novel framework that mimics how a close friend would recommend items based on their knowledge of an individual's tastes. We leverage LLMs to enhance movie recommendation systems by refining traditional algorithm outputs and integrating them with language-based user preference inputs. We employ Singular Value Decomposition (SVD) or SVD++ algorithms to generate initial movie recommendations, implemented using the Surprise Python library and trained on the MovieLens-Latest-Small dataset. We compare the performance of the base algorithms with our LLM-enhanced versions using leave-one-out validation hit rates and cumulative hit rates. Additionally, to compare the performance of our framework against the current state-of-the-art recommendation systems, we use rating and ranking metrics with an item-based stratified 0.75 train, 0.25 test split. Our framework can generate preference profiles automatically based on users' favorite movies or allow manual preference specification for more personalized results. Using an automated approach, our framework overwhelmingly surpassed SVD and SVD++ on every evaluation metric used (e.g., improvements of up to ~6x in cumulative hit rate, ~3.7x in NDCG, etc.), albeit at the cost of a slight increase in computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11005v4">A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 ACL 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in autonomous decision-making, where they sample options from vast action spaces. However, the heuristics that guide this sampling process remain under explored. We study this sampling behavior and show that this underlying heuristics resembles that of human decision-making: comprising a descriptive component (reflecting statistical norm) and a prescriptive component (implicit ideal encoded in the LLM) of a concept. We show that this deviation of a sample from the statistical norm towards a prescriptive component consistently appears in concepts across diverse real-world domains like public health, and economic trends. To further illustrate the theory, we demonstrate that concept prototypes in LLMs are affected by prescriptive norms, similar to the concept of normality in humans. Through case studies and comparison with human studies, we illustrate that in real-world applications, the shift of samples toward an ideal value in LLMs' outputs can result in significantly biased decision-making, raising ethical concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07236v1">An Information-Theoretic Perspective on Multi-LLM Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and naive ensemble baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v3">Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14641v2">HLSTester: Efficient Testing of Behavioral Discrepancies with LLMs for High-Level Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 arXiv admin note: text overlap with arXiv:2407.03889
    </div>
    <details class="paper-abstract">
      In high-level synthesis (HLS), C/C++ programs with synthesis directives are used to generate circuits for FPGA implementations. However, hardware-specific and platform-dependent characteristics in these implementations can introduce behavioral discrepancies between the original C/C++ programs and the circuits after high-level synthesis. Existing methods for testing behavioral discrepancies in HLS are still immature, and the testing workflow requires significant human efforts. To address this challenge, we propose HLSTester, a large language model (LLM) aided testing framework that efficiently detects behavioral discrepancies in HLS. To mitigate hallucinations in LLMs and enhance prompt quality, the testbenches for original C/C++ programs are leveraged to guide LLMs in generating HLS-compatible testbenches, effectively eliminating certain traditional C/C++ constructs that are incompatible with HLS tools. Key variables are pinpointed through a backward slicing technique in both C/C++ and HLS programs to monitor their runtime spectra, enabling an in-depth analysis of the discrepancy symptoms. To reduce test time, a testing input generation mechanism is introduced to integrate dynamic mutation with insights from an LLM-based progressive reasoning chain. In addition, repetitive hardware testing is skipped by a redundancy-aware filtering technique for the generated test inputs. Experimental results demonstrate that the proposed LLM-aided testing framework significantly accelerates the testing workflow while achieving higher testbench simulation pass rates compared with the traditional method and the direct use of LLMs on the same HLS programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07188v1">Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 18 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts -- we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also reveal that all tested models exhibit a consistent \textit{recency bias} varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07186v1">Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 CoLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit cognitive biases -- systematic tendencies of irrational decision-making, similar to those seen in humans. Prior work has found that these biases vary across models and can be amplified by instruction tuning. However, it remains unclear if these differences in biases stem from pretraining, finetuning, or even random noise due to training stochasticity. We propose a two-step causal experimental approach to disentangle these factors. First, we finetune models multiple times using different random seeds to study how training randomness affects over $30$ cognitive biases. Second, we introduce \emph{cross-tuning} -- swapping instruction datasets between models to isolate bias sources. This swap uses datasets that led to different bias patterns, directly testing whether biases are dataset-dependent. Our findings reveal that while training randomness introduces some variability, biases are mainly shaped by pretraining: models with the same pretrained backbone exhibit more similar bias patterns than those sharing only finetuning data. These insights suggest that understanding biases in finetuned models requires considering their pretraining origins beyond finetuning effects. This perspective can guide future efforts to develop principled strategies for evaluating and mitigating bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07146v1">An attention-aware GNN-based input defender against multi-turn jailbreak on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained widespread popularity and are increasingly integrated into various applications. However, their capabilities can be exploited for both benign and harmful purposes. Despite rigorous training and fine-tuning for safety, LLMs remain vulnerable to jailbreak attacks. Recently, multi-turn attacks have emerged, exacerbating the issue. Unlike single-turn attacks, multi-turn attacks gradually escalate the dialogue, making them more difficult to detect and mitigate, even after they are identified. In this study, we propose G-Guard, an innovative attention-aware GNN-based input classifier designed to defend against multi-turn jailbreak attacks on LLMs. G-Guard constructs an entity graph for multi-turn queries, explicitly capturing relationships between harmful keywords and queries even when those keywords appear only in previous queries. Additionally, we introduce an attention-aware augmentation mechanism that retrieves the most similar single-turn query based on the multi-turn conversation. This retrieved query is treated as a labeled node in the graph, enhancing the ability of GNN to classify whether the current query is harmful. Evaluation results demonstrate that G-Guard outperforms all baselines across all datasets and evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07145v1">CCQ: Convolutional Code for Extreme Low-bit Quantization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The rapid scaling of Large Language Models (LLMs) elevates inference costs and compounds substantial deployment barriers. While quantization to 8 or 4 bits mitigates this, sub-3-bit methods face severe accuracy, scalability, and efficiency degradation. We propose Convolutional Code Quantization (CCQ), an inference-optimized quantization approach compressing LLMs to 2.0-2.75 bits with minimal accuracy loss. Departing from error-prone scalar quantization or slow vector quantization, CCQ integrates a hardware-aware bit-shift encoding and decoding solution with Convolutional Code, Hybrid Encoding, and Code Cluster, jointly overcoming accuracy-speed bottlenecks. We construct a lookup-free encoding space, enabling a linear mapping between the codebook and weight vectors, thereby optimizing inference performance. Meanwhile, by drawing on the concept of data mapping from vector quantization, we minimize the performance degradation of the model under extremely low-bit conditions. Experiments demonstrate that CCQ achieves outstanding performance on LLMs across various benchmarks. We compress DeepSeek-V3 (671B total parameters) to 184GB and ERNIE-4.5-300B-A47B to 89GB, enabling single-GPU deployment of ERNIE 4.5 and eliminating inter-card communication. The 2-bit ERNIE-4.5-300B-A47B model and inference engine have been open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v3">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of \textbf{Trust, Risk, and Security Management (TRiSM)} in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around four key pillars: Explainability, ModelOps, Security, Privacy and Governance, each contextualized to the challenges of multi-agent LLM systems. A novel risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI , as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, outlining critical directions to align emerging systems with TRiSM principles for safe, transparent, and accountable operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07064v1">Boosting Parameter Efficiency in LLM-Based Recommendation through Sophisticated Pruning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      LLM-based recommender systems have made significant progress; however, the deployment cost associated with the large parameter volume of LLMs still hinders their real-world applications. This work explores parameter pruning to improve parameter efficiency while maintaining recommendation quality, thereby enabling easier deployment. Unlike existing approaches that focus primarily on inter-layer redundancy, we uncover intra-layer redundancy within components such as self-attention and MLP modules. Building on this analysis, we propose a more fine-grained pruning approach that integrates both intra-layer and layer-wise pruning. Specifically, we introduce a three-stage pruning strategy that progressively prunes parameters at different levels and parts of the model, moving from intra-layer to layer-wise pruning, or from width to depth. Each stage also includes a performance restoration step using distillation techniques, helping to strike a balance between performance and parameter efficiency. Empirical results demonstrate the effectiveness of our approach: across three datasets, our models achieve an average of 88% of the original model's performance while pruning more than 95% of the non-embedding parameters. This underscores the potential of our method to significantly reduce resource requirements without greatly compromising recommendation quality. Our code will be available at: https://github.com/zheng-sl/PruneRec
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07045v1">5C Prompt Contracts: A Minimalist, Creative-Friendly, Token-Efficient Design Framework for Individual and SME LLM Usage</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 5 pages, 5 tables. Includes comparative experimental results across OpenAI, Anthropic, DeepSeek, and Gemini LLMs
    </div>
    <details class="paper-abstract">
      The progression from traditional prompt engineering to a more rigorous discipline of prompt design marks a pivotal shift in human-LLM interaction. As Large Language Models (LLMs) become increasingly embedded in mission-critical applications, there emerges a pressing need for frameworks that are not only explicit and systematic but also minimal enough to remain practical and broadly accessible. While many existing approaches address prompt structuring through elaborate Domain-Specific Languages (DSLs) or multi-layered templates, such methods can impose significant token and cognitive overhead, potentially constraining the model's creative capacity. In this context, we propose the 5C Prompt Contract, a framework that distills prompt design into five intuitive components: Character, Cause, Constraint, Contingency, and Calibration. This minimal cognitive schema explicitly integrates fallback and output optimization directives, fostering reliable, interpretable, and creatively flexible AI interactions. Experimental results demonstrate that the 5C framework consistently achieves superior input token efficiency while maintaining rich and consistent outputs across diverse LLM architectures (OpenAI, Anthropic, DeepSeek, and Gemini), making it particularly suited for individuals and Small-to-Medium Enterprises (SMEs) with limited AI engineering resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06999v1">Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reasoning is a key capability for large language models (LLMs), particularly when applied to complex tasks such as mathematical problem solving. However, multimodal reasoning research still requires further exploration of modality alignment and training costs. Many of these approaches rely on additional data annotation and relevant rule-based rewards to enhance the understanding and reasoning ability, which significantly increases training costs and limits scalability. To address these challenges, we propose the Deliberate-to-Intuitive reasoning framework (D2I) that improves the understanding and reasoning ability of multimodal LLMs (MLLMs) without extra annotations and complex rewards. Specifically, our method sets deliberate reasoning strategies to enhance modality alignment only through the rule-based format reward during training. While evaluating, the reasoning style shifts to intuitive, which removes deliberate reasoning strategies during training and implicitly reflects the model's acquired abilities in the response. D2I outperforms baselines across both in-domain and out-of-domain benchmarks. Our findings highlight the role of format reward in fostering transferable reasoning skills in MLLMs, and inspire directions for decoupling training-time reasoning depth from test-time response flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06992v1">MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12112v3">Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 57 pages, 25 figures, 15 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have recently demonstrated strong potential in solving planning problems, there is a trade-off between flexibility and complexity. LLMs, as zero-shot planners themselves, are still not capable of directly generating valid plans for complex planning problems such as multi-constraint or long-horizon tasks. On the other hand, many frameworks aiming to solve complex planning problems often rely on task-specific preparatory efforts, such as task-specific in-context examples and pre-defined critics/verifiers, which limits their cross-task generalization capability. In this paper, we tackle these challenges by observing that the core of many planning problems lies in optimization problems: searching for the optimal solution (best plan) with goals subject to constraints (preconditions and effects of decisions). With LLMs' commonsense, reasoning, and programming capabilities, this opens up the possibilities of a universal LLM-based approach to planning problems. Inspired by this observation, we propose LLMFP, a general-purpose framework that leverages LLMs to capture key information from planning problems and formally formulate and solve them as optimization problems from scratch, with no task-specific examples needed. We apply LLMFP to 9 planning problems, ranging from multi-constraint decision making to multi-step planning problems, and demonstrate that LLMFP achieves on average 83.7% and 86.8% optimal rate across 9 tasks for GPT-4o and Claude 3.5 Sonnet, significantly outperforming the best baseline (direct planning with OpenAI o1-preview) with 37.6% and 40.7% improvements. We also validate components of LLMFP with ablation experiments and analyzed the underlying success and failure reasons. Project page: https://sites.google.com/view/llmfp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06980v1">Are They All Good? Evaluating the Quality of CoTs in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance in code generation, particularly when augmented with chain-of-thought (CoT) prompting techniques. They break down requirements into intermediate reasoning steps, which act as design rationales to guide LLMs in writing code like human programmers. Thus, the quality of these steps is crucial for ensuring the correctness and reliability of the generated code. However, little is known about the quality of CoT generated by LLMs. To what extent can we trust the thoughts generated by LLMs? How good are they? This paper empirically explores the external and internal factors of why LLMs generate unsatisfactory CoTs by analyzing 1,023 failed code samples on two widely used code generation benchmarks. We also evaluate their impact on code generation performance by analyzing 210 CoT-code pairs and refining the unsatisfied CoTs by prompting LLMs. Our study reveals three key findings: (1) External factors (53.60%), such as unclear requirements and lack of context, mainly affect CoT quality, while internal factors (40.10%) stem from LLMs' misunderstanding prompts. (2) Even when CoTs are correct, 18.5% of the generated code contains errors due to instruction-following issues; conversely, 11.90% of correct code is paired with flawed CoTs. (3) Refining low-quality CoTs is feasible, i.e., LLMs improve when given detailed problem descriptions. These findings highlight key challenges in CoT-based code generation and suggest directions for improving LLM reasoning and reliability.
    </details>
</div>
