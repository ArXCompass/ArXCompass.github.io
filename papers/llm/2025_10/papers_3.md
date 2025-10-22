# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13982v2">Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11218v2">The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Code: https://github.com/WorldHellow/SLAQ/tree/main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09986v3">From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Motivated by the remarkable success of artificial intelligence (AI) across diverse fields, the application of AI to solve scientific problems, often formulated as partial differential equations (PDEs), has garnered increasing attention. While most existing research concentrates on theoretical properties (such as well-posedness, regularity, and continuity) of the solutions, alongside direct AI-driven methods for solving PDEs, the challenge of uncovering symbolic relationships within these equations remains largely unexplored. In this paper, we propose leveraging large language models (LLMs) to learn such symbolic relationships. Our results demonstrate that LLMs can effectively predict the operators involved in PDE solutions by utilizing the symbolic information in the PDEs both theoretically and numerically. Furthermore, we show that discovering these symbolic relationships can substantially improve both the efficiency and accuracy of symbolic machine learning for finding analytical approximation of PDE solutions, delivering a fully interpretable solution pipeline. This work opens new avenues for understanding the symbolic structure of scientific problems and advancing their solution processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16590v1">Atom-anchored LLMs speak Chemistry: A Retrosynthesis Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Alan Kai Hassen and Andrius Bernatavicius contributed equally to this work
    </div>
    <details class="paper-abstract">
      Applications of machine learning in chemistry are often limited by the scarcity and expense of labeled data, restricting traditional supervised methods. In this work, we introduce a framework for molecular reasoning using general-purpose Large Language Models (LLMs) that operates without requiring labeled training data. Our method anchors chain-of-thought reasoning to the molecular structure by using unique atomic identifiers. First, the LLM performs a one-shot task to identify relevant fragments and their associated chemical labels or transformation classes. In an optional second step, this position-aware information is used in a few-shot task with provided class examples to predict the chemical transformation. We apply our framework to single-step retrosynthesis, a task where LLMs have previously underperformed. Across academic benchmarks and expert-validated drug discovery molecules, our work enables LLMs to achieve high success rates in identifying chemically plausible reaction sites ($\geq90\%$), named reaction classes ($\geq40\%$), and final reactants ($\geq74\%$). Beyond solving complex chemical tasks, our work also provides a method to generate theoretically grounded synthetic datasets by mapping chemical knowledge onto the molecular structure and thereby addressing data scarcity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v4">Unlocking LLM Repair Capabilities Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06767v2">Beyond Surface Similarity: Evaluating LLM-Based Test Refactorings with Structural and Semantic Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to refactor unit tests, improving readability and structure while preserving behavior. Evaluating such refactorings, however, remains difficult: metrics like CodeBLEU penalize beneficial renamings and edits, while semantic similarities overlook readability and modularity. We propose CTSES, a first step toward human-aligned evaluation of refactored tests. CTSES combines CodeBLEU, METEOR, and ROUGE-L into a composite score that balances semantics, lexical clarity, and structural alignment. Evaluated on 5,000+ refactorings from Defects4J and SF110 (GPT-4o and Mistral-Large), CTSES reduces false negatives and provides more interpretable signals than individual metrics. Our emerging results illustrate that CTSES offers a proof-of-concept for composite approaches, showing their promise in bridging automated metrics and developer judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22961v2">ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16559v1">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16552v1">LANPO: Bootstrapping Language and Numerical Feedback for Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning in large language models (LLMs) often relies on scalar rewards, a practice that discards valuable textual rationale buried in the rollouts, forcing the model to explore \textit{de novo} with each attempt and hindering sample efficiency. While LLMs can uniquely learn from language feedback provided in-context, naively integrating on-line experiences into RL training presents a paradox: feedback from the same problem risks information leakage and memorization, while feedback from different problems often leads to behavior collapse due to irrelevant context. To resolve this tension, we propose \textbf{Language-And-Numerical Policy Optimization (LANPO)}, a framework that cleanly separates the roles of feedback: language guides exploration, while numerical rewards drive optimization. LANPO builds a dynamic experience pool from past trials and introduces two principles to ensure feedback is effective: \emph{Reward-Agnostic Reflection} for safe intra-sample self-correction and \emph{Relevant Abstraction} to distill generalizable lessons from inter-sample experiences. Across mathematical reasoning benchmarks, LANPO enables 7B and 14B models to significantly outperform strong baselines trained with GRPO in test accuracy. Our work provides a robust method for integrating historical experiences into the LLM RL loop, creating more effective and data-efficient learning agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16551v1">From Reviews to Actionable Insights: An LLM-Based Approach for Attribute and Feature Extraction</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      This research proposes a systematic, large language model (LLM) approach for extracting product and service attributes, features, and associated sentiments from customer reviews. Grounded in marketing theory, the framework distinguishes perceptual attributes from actionable features, producing interpretable and managerially actionable insights. We apply the methodology to 20,000 Yelp reviews of Starbucks stores and evaluate eight prompt variants on a random subset of reviews. Model performance is assessed through agreement with human annotations and predictive validity for customer ratings. Results show high consistency between LLMs and human coders and strong predictive validity, confirming the reliability of the approach. Human coders required a median of six minutes per review, whereas the LLM processed each in two seconds, delivering comparable insights at a scale unattainable through manual coding. Managerially, the analysis identifies attributes and features that most strongly influence customer satisfaction and their associated sentiments, enabling firms to pinpoint "joy points," address "pain points," and design targeted interventions. We demonstrate how structured review data can power an actionable marketing dashboard that tracks sentiment over time and across stores, benchmarks performance, and highlights high-leverage features for improvement. Simulations indicate that enhancing sentiment for key service features could yield 1-2% average revenue gains per store.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16549v1">ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. Recent work has focused on using LLMs to improve review efficiency or generate insightful review content. However, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine the peer review ecosystem and compromise academic integrity. To address this critical issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews. ReviewGuard employs a comprehensive four-stage LLM-driven framework that: (1) collects ICLR and NeurIPS papers with their corresponding reviews from OpenReview; (2) annotates review types using GPT-4.1 with human validation; (3) addresses class imbalance and data scarcity through LLM-driven synthetic data augmentation, producing a final corpus of 6,634 papers, 24,657 real reviews, and 46,438 synthetic reviews; and (4) fine-tunes both encoder-based models and open source LLMs. We perform comprehensive feature analysis of the structure and quality of the review text. Compared to sufficient reviews, deficient reviews demonstrate lower rating scores, higher self-reported confidence, reduced structural complexity, and a higher proportion of negative sentiment. AI-generated text detection reveals that, since ChatGPT's emergence, AI-generated reviews have increased dramatically. In the evaluation of deficient review detection models, mixed training with synthetic and real review data provides substantial enhancements to recall and F1 scores on the binary task. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review while offering valuable insights into human-AI collaboration to maintain academic integrity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16530v1">Realizing LLMs' Causal Potential Requires Science-Grounded, Novel Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent claims of strong performance by Large Language Models (LLMs) on causal discovery are undermined by a key flaw: many evaluations rely on benchmarks likely included in pretraining corpora. Thus, apparent success suggests that LLM-only methods, which ignore observational data, outperform classical statistical approaches. We challenge this narrative by asking: Do LLMs truly reason about causal structure, and how can we measure it without memorization concerns? Can they be trusted for real-world scientific discovery? We argue that realizing LLMs' potential for causal analysis requires two shifts: (P.1) developing robust evaluation protocols based on recent scientific studies to guard against dataset leakage, and (P.2) designing hybrid methods that combine LLM-derived knowledge with data-driven statistics. To address P.1, we encourage evaluating discovery methods on novel, real-world scientific studies. We outline a practical recipe for extracting causal graphs from recent publications released after an LLM's training cutoff, ensuring relevance and preventing memorization while capturing both established and novel relations. Compared to benchmarks like BNLearn, where LLMs achieve near-perfect accuracy, they perform far worse on our curated graphs, underscoring the need for statistical grounding. Supporting P.2, we show that using LLM predictions as priors for the classical PC algorithm significantly improves accuracy over both LLM-only and purely statistical methods. We call on the community to adopt science-grounded, leakage-resistant benchmarks and invest in hybrid causal discovery methods suited to real-world inquiry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05909v2">Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Open-source code available at https://github.com/flowersteam/llm_persuasion
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16967v2">Hard Negatives, Hard Lessons: Revisiting Training Data Quality for Robust Information Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection, reduces the training set size by 2.35$\times$, surprisingly increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We utilize LLMs as a simple, cost-effective approach to identify and relabel false negatives in training datasets. Experimental results show that relabeling false negatives as true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7$\unicode{x2013}$1.4 points on BEIR and by 1.7$\unicode{x2013}$1.8 points at nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of LLMs to identify false negatives is supported by human annotation results. Our training dataset and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17891v1">TritonRL: Training LLMs to Think and Code Triton Without Cheating</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      With the rapid evolution of large language models (LLMs), the demand for automated, high-performance system kernels has emerged as a key enabler for accelerating development and deployment. We introduce TritonRL, a domain-specialized LLM for Triton kernel generation, trained with a novel training framework that enables robust and automated kernel synthesis. Unlike general-purpose programming languages, Triton kernel generation faces unique challenges due to data scarcity and incomplete evaluation criteria, vulnerable to reward hacking. Our approach addresses these challenges end-to-end by distilling Triton-specific knowledge through supervised fine-tuning on curated datasets, and further improving code quality via reinforcement learning (RL) with robust, verifiable rewards and hierarchical reward assignment. Our RL framework robustly detects reward hacking and guides both reasoning traces and code tokens through fine-grained verification and hierarchical reward decomposition, enabling the model to generate high-quality Triton kernels that can truly replace existing modules. With robust and fine-grained evaluation, our experiments on KernelBench demonstrate that TritonRL achieves state-of-the-art correctness and speedup, surpassing all other Triton-specific models and underscoring the effectiveness of our RL-based training paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15870v1">OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Technical Report. Code: https://github.com/NVlabs/OmniVinci
    </div>
    <details class="paper-abstract">
      Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15859v1">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 17 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown substantial advances through reinforcement learning (RL), particularly in domains where rewards can be programmatically verified, such as mathematics and code. In these areas, models benefit from a well-defined operational base guided by explicit rule-based objectives. However, this progress reveals a significant limitation: in open-ended domains where rewards are ambiguous, subjective, or context-dependent, such as creative writing, scientific reasoning, and notably medical consultation, robust reward functions are lacking, making these areas challenging for current RL strategies. To bridge this gap, we introduce ORBIT, an open-ended rubric-based incremental training framework specifically designed for high-stakes medical dialogue. ORBIT integrates syn- thetic dialogue generation with the dynamic creation of rubrics, employing these rubrics to direct an incremental RL process. In particular, this approach does not depend on external medical knowledge or manual rules, instead utilizing rubric-guided feedback to shape learning. When implemented on the Qwen3-4B-Instruct model, our method can greatly enhance its performance on the HealthBench-Hard benchmark from 7.0 to 27.2 using only 2k samples, thus achieving state-of-the-art results for models of this scale. Our analysis confirms that rubric-driven RL fos-ters consistent performance gains across diverse consultation scenarios, going beyond simple numerical improvements. These findings underscore rubric-based feedback as a scalable strategy for advancing LLMs in intricate, open-ended tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15746v1">LLMs Judge Themselves: A Game-Theoretic Framework for Human-Aligned Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Ideal or real - that is the question.In this work, we explore whether principles from game theory can be effectively applied to the evaluation of large language models (LLMs). This inquiry is motivated by the growing inadequacy of conventional evaluation practices, which often rely on fixed-format tasks with reference answers and struggle to capture the nuanced, subjective, and open-ended nature of modern LLM behavior. To address these challenges, we propose a novel alternative: automatic mutual evaluation, where LLMs assess each other's output through self-play and peer review. These peer assessments are then systematically compared with human voting behavior to evaluate their alignment with human judgment. Our framework incorporates game-theoretic voting algorithms to aggregate peer reviews, enabling a principled investigation into whether model-generated rankings reflect human preferences. Empirical results reveal both convergences and divergences between theoretical predictions and human evaluations, offering valuable insights into the promises and limitations of mutual evaluation. To the best of our knowledge, this is the first work to jointly integrate mutual evaluation, game-theoretic aggregation, and human-grounded validation for evaluating the capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15729v1">FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been explored for integration with collaborative filtering (CF)-based recommendation systems, which are crucial for personalizing user experiences. However, a key challenge is that LLMs struggle to interpret the latent, non-semantic embeddings produced by CF approaches, limiting recommendation effectiveness and further applications. To address this, we propose FACE, a general interpretable framework that maps CF embeddings into pre-trained LLM tokens. Specifically, we introduce a disentangled projection module to decompose CF embeddings into concept-specific vectors, followed by a quantized autoencoder to convert continuous embeddings into LLM tokens (descriptors). Then, we design a contrastive alignment objective to ensure that the tokens align with corresponding textual signals. Hence, the model-agnostic FACE framework achieves semantic alignment without fine-tuning LLMs and enhances recommendation performance by leveraging their pre-trained capabilities. Empirical results on three real-world recommendation datasets demonstrate performance improvements in benchmark models, with interpretability studies confirming the interpretability of the descriptors. Code is available in https://github.com/YixinRoll/FACE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v4">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We introduce time-to-unsafe-sampling, a novel safety measure for generative models, defined as the number of generations required by a large language model (LLM) to trigger an unsafe (e.g., toxic) response. While providing a new dimension for prompt-adaptive safety evaluation, quantifying time-to-unsafe-sampling is challenging: unsafe outputs are often rare in well-aligned models and thus may not be observed under any feasible sampling budget. To address this challenge, we frame this estimation problem as one of survival analysis. We build on recent developments in conformal prediction and propose a novel calibration technique to construct a lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt with rigorous coverage guarantees. Our key technical innovation is an optimized sampling-budget allocation scheme that improves sample efficiency while maintaining distribution-free guarantees. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15690v1">MirrorFuzz: Leveraging LLM and Shared Bugs for Deep Learning Framework APIs Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted for publication in IEEE Transactions on Software Engineering (TSE), 2025
    </div>
    <details class="paper-abstract">
      Deep learning (DL) frameworks serve as the backbone for a wide range of artificial intelligence applications. However, bugs within DL frameworks can cascade into critical issues in higher-level applications, jeopardizing reliability and security. While numerous techniques have been proposed to detect bugs in DL frameworks, research exploring common API patterns across frameworks and the potential risks they entail remains limited. Notably, many DL frameworks expose similar APIs with overlapping input parameters and functionalities, rendering them vulnerable to shared bugs, where a flaw in one API may extend to analogous APIs in other frameworks. To address this challenge, we propose MirrorFuzz, an automated API fuzzing solution to discover shared bugs in DL frameworks. MirrorFuzz operates in three stages: First, MirrorFuzz collects historical bug data for each API within a DL framework to identify potentially buggy APIs. Second, it matches each buggy API in a specific framework with similar APIs within and across other DL frameworks. Third, it employs large language models (LLMs) to synthesize code for the API under test, leveraging the historical bug data of similar APIs to trigger analogous bugs across APIs. We implement MirrorFuzz and evaluate it on four popular DL frameworks (TensorFlow, PyTorch, OneFlow, and Jittor). Extensive evaluation demonstrates that MirrorFuzz improves code coverage by 39.92\% and 98.20\% compared to state-of-the-art methods on TensorFlow and PyTorch, respectively. Moreover, MirrorFuzz discovers 315 bugs, 262 of which are newly found, and 80 bugs are fixed, with 52 of these bugs assigned CNVD IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15685v1">Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 8 pages, 9 figures, submitted to LREC 2026
    </div>
    <details class="paper-abstract">
      This research introduces a novel approach to textual and multimodal Hate Speech Detection (HSD), using Large Language Models (LLMs) as dynamic knowledge bases to generate background context and incorporate it into the input of HSD classifiers. Two context generation strategies are examined: one focused on named entities and the other on full-text prompting. Four methods of incorporating context into the classifier input are compared: text concatenation, embedding concatenation, a hierarchical transformer-based fusion, and LLM-driven text enhancement. Experiments are conducted on the textual Latent Hatred dataset of implicit hate speech and applied in a multimodal setting on the MAMI dataset of misogynous memes. Results suggest that both the contextual information and the method by which it is incorporated are key, with gains of up to 3 and 6 F1 points on textual and multimodal setups respectively, from a zero-context baseline to the highest-performing system, based on embedding concatenation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14846v2">Where to Search: Measure the Prior-Structured Search Space of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 10 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      The generate-filter-refine (iterative) paradigm based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07581v2">Expanding the Action Space of LLMs to Reason Beyond Language</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful reasoners in natural language, but their actions are typically confined to outputting vocabulary tokens. As a result, interactions with external environments -- such as symbolic operators or simulators -- must be expressed through text in predefined formats, parsed, and routed to external interfaces. This overloads the model's language with both reasoning and control duties, and requires a hand-crafted parser, external to the LLM. To address this, we decouple environment interactions from language by internalizing them in an Expanded Action space (ExpA), beyond the vocabulary. The model starts reasoning in the default language environment, but may trigger routing actions and switch to an external environment at any time. From there, the model can only invoke environment-specific actions, receive feedback from the environment, and potentially route back to language as a result. To promote effective exploration of the expanded action space and new environments, we introduce ExpA Reinforcement Learning (EARL) with counterfactual policy optimization. On tasks requiring multi-turn interactions and contingent planning, EARL outperforms strong baselines with vocabulary-constrained actions. It performs robustly across calculator-based multi-task learning and, in the partially observed sorting problem, achieves perfect Sort-4 accuracy while self-discovering an efficient algorithm competitive with classical designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26184v4">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation (RG) are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recently proposed ARGUE framework for RG evaluation. We present analysis of Auto-ARGUE on the RG pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15614v1">HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations-not just a single correct answer-becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe-rather than a leaderboard-for methods that explicitly explore and cover admissible explanation spaces. Code is available at: https://github.com/CTT-Pavilion/_HypoSpace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15561v1">Finetuning LLMs for EvaCun 2025 token prediction shared task</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      In this paper, we present our submission for the token prediction task of EvaCun 2025. Our sys-tems are based on LLMs (Command-R, Mistral, and Aya Expanse) fine-tuned on the task data provided by the organizers. As we only pos-sess a very superficial knowledge of the subject field and the languages of the task, we simply used the training data without any task-specific adjustments, preprocessing, or filtering. We compare 3 different approaches (based on 3 different prompts) of obtaining the predictions, and we evaluate them on a held-out part of the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15522v1">Latent Reasoning in LLMs as a Vocabulary-Space Superposition</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong reasoning abilities with chain-of-thought prompting, but explicit reasoning introduces substantial computational overhead. Recent work on latent reasoning reduces this cost by reasoning in latent space without explicit supervision, but performance drops significantly. Our preliminary experiments suggest that this degradation stems from the unstructured latent space, which makes fitting latent tokens difficult. To address this, we restrict the latent space to the column space of the LLM vocabulary, treating latent reasoning as a superposition over vocabulary probabilities. Once latent reasoning concludes, it collapses into an eigenstate of explicit reasoning to yield the final answer. Based on this idea, we propose Latent-SFT, a two-stage learning framework. In the first stage, we design two specialized attention masks to guide the Latent Token Encoder in generating latent tokens, allowing the LLM to produce the correct answer conditioned on them. In the second stage, the Latent Token Encoder is discarded, and the LLM is directly trained to generate these latent tokens autonomously for latent reasoning, optimized with KL and CE losses. Latent-SFT sets a new state of the art on GSM8k, matching explicit SFT performance while cutting reasoning chains by up to 4 times and outperforming prior latent methods. On Math500 and AIME24, lexical probability-based latent reasoning also clearly surpasses hidden-state-based approaches. Our metrics of effective compression rate and effective global parallelism further show that latent reasoning is both the compression of a single path and the superposition of multiple paths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15513v1">Temporal Referential Consistency: Do LLMs Favor Sequences Over Absolute Time References?</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 EMNLP Main Long Paper 2025
    </div>
    <details class="paper-abstract">
      The increasing acceptance of large language models (LLMs) as an alternative to knowledge sources marks a significant paradigm shift across various domains, including time-sensitive fields such as law, healthcare, and finance. To fulfill this expanded role, LLMs must not only be factually accurate but also demonstrate consistency across temporal dimensions, necessitating robust temporal reasoning capabilities. Despite this critical requirement, efforts to ensure temporal consistency in LLMs remain scarce including noticeable absence of endeavors aimed at evaluating or augmenting LLMs across temporal references in time-sensitive inquiries. In this paper, we seek to address this gap by introducing a novel benchmark entitled temporal referential consistency, accompanied by a resource TEMP-ReCon designed to benchmark a wide range of both open-source and closed-source LLMs with various linguistic contexts characterized by differing resource richness (including English, French, and Romanian). The findings emphasis that LLMs do exhibit insufficient temporal referent consistency. To address this, we propose \newmodel, a reasoning path alignment-based model that aims to enhance the temporal referential consistency of LLMs. Our empirical experiments substantiate the efficacy of UnTRaP compared to several baseline models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15502v1">The Road Less Traveled: Enhancing Exploration in LLMs via Sequential Sampling</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has been pivotal in enhancing the reasoning capabilities of large language models (LLMs), but it often suffers from limited exploration and entropy collapse, where models exploit a narrow set of solutions, leading to a loss of sampling diversity and subsequently preventing RL from further improving performance. This issue is exacerbated in parallel sampling methods, where multiple outputs are drawn from the same distribution, potentially causing the model to converge to similar solutions. We propose SESA, a novel SEquential SAmpling framework that mitigates this challenge by generating diverse solution sketches sequentially before expanding them into full reasoning paths. This approach ensures broader exploration by conditioning each new output on previous ones, promoting diversity throughout the process and preventing policy collapse. Our experiments on a synthetic task show that sequential sampling consistently outperforms traditional RL methods in terms of path diversity and recovery from collapse. Further evaluations on real-world tasks demonstrate that SESA improves both the exploration of valid strategies and the overall performance of LLMs. On three agent benchmarks, SESA lifts success rates by $+0.25$, $+0.42$, and $+0.07$ absolute over the base model (up to an additional $211\%$ relative improvement over baseline RL), underscoring its exploration advantage. This work introduces a structured approach to exploration, paving the way for more effective and diverse reasoning in RL-trained LLMs. Our code is released at https://github.com/MuLabPKU/sesa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15494v1">An Experimental Study of Real-Life LLM-Proposed Performance Improvements</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate code, but can they generate fast code? In this paper, we study this question using a dataset of 65 real-world tasks mined from open-source Java programs. We specifically select tasks where developers achieved significant speedups, and employ an automated pipeline to generate patches for these issues using two leading LLMs under four prompt variations. By rigorously benchmarking the results against the baseline and human-authored solutions, we demonstrate that LLM-generated code indeed improves performance over the baseline in most cases. However, patches proposed by human developers outperform LLM fixes by a statistically significant margin, indicating that LLMs often fall short of finding truly optimal solutions. We further find that LLM solutions are semantically identical or similar to the developer optimization idea in approximately two-thirds of cases, whereas they propose a more original idea in the remaining one-third. However, these original ideas only occasionally yield substantial performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15455v1">CORE: Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Mobile agents rely on Large Language Models (LLMs) to plan and execute tasks on smartphone user interfaces (UIs). While cloud-based LLMs achieve high task accuracy, they require uploading the full UI state at every step, exposing unnecessary and often irrelevant information. In contrast, local LLMs avoid UI uploads but suffer from limited capacity, resulting in lower task success rates. We propose $\textbf{CORE}$, a $\textbf{CO}$llaborative framework that combines the strengths of cloud and local LLMs to $\textbf{R}$educe UI $\textbf{E}$xposure, while maintaining task accuracy for mobile agents. CORE comprises three key components: (1) $\textbf{Layout-aware block partitioning}$, which groups semantically related UI elements based on the XML screen hierarchy; (2) $\textbf{Co-planning}$, where local and cloud LLMs collaboratively identify the current sub-task; and (3) $\textbf{Co-decision-making}$, where the local LLM ranks relevant UI blocks, and the cloud LLM selects specific UI elements within the top-ranked block. CORE further introduces a multi-round accumulation mechanism to mitigate local misjudgment or limited context. Experiments across diverse mobile apps and tasks show that CORE reduces UI exposure by up to 55.6% while maintaining task success rates slightly below cloud-only agents, effectively mitigating unnecessary privacy exposure to the cloud. The code is available at https://github.com/Entropy-Fighter/CORE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15444v1">A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Test-time scaling seeks to improve the reasoning performance of large language models (LLMs) by adding computational resources. A prevalent approach within the field is sampling-based test-time scaling methods, which enhance reasoning by generating multiple reasoning paths for a given input during inference. However, despite its practical success, the theoretical foundations remain underexplored. In this paper, we provide the first theoretical framework for analyzing sampling-based test-time scaling methods, grounded in the perspective of confidence estimation. Based on the framework, we analyze two dominant paradigms: self-consistency and perplexity, and reveal key limitations: self-consistency suffers from high estimation error while perplexity exhibits substantial modeling error and possible degradation of the estimation error convergence. To address these limitations, we introduce RPC, a hybrid method that leverages our theoretical insights through two key components: Perplexity Consistency and Reasoning Pruning. Perplexity Consistency combines the strengths of self-consistency and perplexity, boosting the convergence rate of estimation error from linear to exponential while preserving model error. Reasoning Pruning prevents degradation by eliminating low-probability reasoning paths. Both theoretical analysis and empirical results across seven benchmark datasets demonstrate that RPC has a strong potential for reducing reasoning error. Notably, RPC achieves reasoning performance comparable to self-consistency while not only enhancing confidence reliability but also reducing sampling costs by 50%. The code and resources are available at https://wnjxyk.github.io/RPC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14365v2">On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce UCC-Inj, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and implicit versus explicit denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15428v1">Fault Cause Identification across Manufacturing Lines through Ontology-Guided and Process-Aware FMEA Graph Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Fault cause identification in automated manufacturing lines is challenging due to the system's complexity, frequent reconfigurations, and the limited reusability of existing Failure Mode and Effects Analysis (FMEA) knowledge. Although FMEA worksheets contain valuable expert insights, their reuse across heterogeneous lines is hindered by natural language variability, inconsistent terminology, and process differences. To address these limitations, this study proposes a process-aware framework that enhances FMEA reusability by combining manufacturing-domain conceptualization with graph neural network (GNN) reasoning. First, FMEA worksheets from multiple manufacturing lines are transformed into a unified knowledge graph through ontology-guided large language model (LLM) extraction, capturing domain concepts such as actions, states, components, and parameters. Second, a Relational Graph Convolutional Network (RGCN) with the process-aware scoring function learns embeddings that respect both semantic relationships and sequential process flows. Finally, link prediction is employed to infer and rank candidate fault causes consistent with the target line's process flow. A case study on automotive pressure sensor assembly lines demonstrates that the proposed method outperforms a state-of-the-art retrieval-augmented generation (RAG) baseline (F1@20 = 0.267) and an RGCN approach (0.400), achieving the best performance (0.523) in fault cause identification. Ablation studies confirm the contributions of both LLM-driven domain conceptualization and process-aware learning. These results indicate that the proposed framework significantly improves the transferability of FMEA knowledge across heterogeneous lines, thereby supporting operators in diagnosing failures more reliably and paving the way for future domain-adaptive LLM applications in smart manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15414v1">MARS: Reinforcing Multi-Agent Reasoning of LLMs through Self-Play in Strategic Games</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Developing Large Language Models (LLMs) to cooperate and compete effectively within multi-agent systems is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARS, an end-to-end RL framework that incentivizes Multi-Agent Reasoning of LLMs through Self-play in both cooperative and competitive games. MARS features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, the MARS agent trained from Qwen3-4B develops strong strategic abilities that generalize to held-out games with up to 28.7% performance improvements. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of multi-agent systems in reasoning benchmarks. When integrated into leading multi-agent systems, our MARS agent achieves significant performance gains of 10.0% on AIME and 12.5% on GPQA-Diamond. These results establish end-to-end RL training with self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs. Our code and models are publicly available at https://github.com/thu-nics/MARS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26490v2">VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 The code, dataset, and leaderboard are available at https://vitabench.github.io/
    </div>
    <details class="paper-abstract">
      As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at https://vitabench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15406v1">VocalBench-DF: A Benchmark for Evaluating Speech LLM Robustness to Disfluency</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While Speech Large Language Models (Speech-LLMs) show strong performance in many applications, their robustness is critically under-tested, especially to speech disfluency. Existing evaluations often rely on idealized inputs, overlooking common disfluencies, particularly those associated with conditions like Parkinson's disease. This work investigates whether current Speech-LLMs can maintain performance when interacting with users who have speech impairments. To facilitate this inquiry, we introduce VocalBench-DF, a framework for the systematic evaluation of disfluency across a multi-dimensional taxonomy. Our evaluation of 22 mainstream Speech-LLMs reveals substantial performance degradation, indicating that their real-world readiness is limited. Further analysis identifies phoneme-level processing and long-context modeling as primary bottlenecks responsible for these failures. Strengthening recognition and reasoning capability from components and pipelines can substantially improve robustness. These findings highlight the urgent need for new methods to improve disfluency handling and build truly inclusive Speech-LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18492v2">GuardReasoner: Towards Reasoning-based LLM Safeguards</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 22 pages, 18 figures
    </div>
    <details class="paper-abstract">
      As LLMs increasingly impact safety-critical applications, ensuring their safety using guardrails remains a key challenge. This paper proposes GuardReasoner, a new safeguard for LLMs, by guiding the guard model to learn to reason. Concretely, we first create the GuardReasonerTrain dataset, which consists of 127K samples with 460K detailed reasoning steps. Then, we introduce reasoning SFT to unlock the reasoning capability of guard models. In addition, we present hard sample DPO to further strengthen their reasoning ability. In this manner, GuardReasoner achieves better performance, explainability, and generalizability. Extensive experiments and analyses on 13 benchmarks of 3 guardrail tasks demonstrate its superiority. Remarkably, GuardReasoner 8B surpasses GPT-4o+CoT by 5.74% and LLaMA Guard 3 8B by 20.84% F1 score on average. We release the training data, code, and models with different scales (1B, 3B, 8B) of GuardReasoner : https://github.com/yueliu1999/GuardReasoner/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01077v2">Zero-Shot Document-Level Biomedical Relation Extraction via Scenario-based Prompt Design in Two-Stage with LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      With the advent of artificial intelligence (AI), many researchers are attempting to extract structured information from document-level biomedical literature by fine-tuning large language models (LLMs). However, they face significant challenges such as the need for expensive hardware, like high-performance GPUs and the high labor costs associated with annotating training datasets, especially in biomedical realm. Recent research on LLMs, such as GPT-4 and Llama3, has shown promising performance in zero-shot settings, inspiring us to explore a novel approach to achieve the same results from unannotated full documents using general LLMs with lower hardware and labor costs. Our approach combines two major stages: named entity recognition (NER) and relation extraction (RE). NER identifies chemical, disease and gene entities from the document with synonym and hypernym extraction using an LLM with a crafted prompt. RE extracts relations between entities based on predefined relation schemas and prompts. To enhance the effectiveness of prompt, we propose a five-part template structure and a scenario-based prompt design principles, along with evaluation method to systematically assess the prompts. Finally, we evaluated our approach against fine-tuning and pre-trained models on two biomedical datasets: ChemDisGene and CDR. The experimental results indicate that our proposed method can achieve comparable accuracy levels to fine-tuning and pre-trained models but with reduced human and hardware expenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03345v2">Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted for publication in the Journal of Systems and Software (Special Issue on Reliable and Secure Large Language Models for Software Engineering)
    </div>
    <details class="paper-abstract">
      The swift spread of fake news and disinformation campaigns poses a significant threat to public trust, political stability, and cybersecurity. Traditional Cyber Threat Intelligence (CTI) approaches, which rely on low-level indicators such as domain names and social media handles, are easily evaded by adversaries who frequently modify their online infrastructure. To address these limitations, we introduce a novel CTI framework that focuses on high-level, semantic indicators derived from recurrent narratives and relationships of disinformation campaigns. Our approach extracts structured CTI indicators from unstructured disinformation content, capturing key entities and their contextual dependencies within fake news using Large Language Models (LLMs). We further introduce FakeCTI, the first dataset that systematically links fake news to disinformation campaigns and threat actors. To evaluate the effectiveness of our CTI framework, we analyze multiple fake news attribution techniques, spanning from traditional Natural Language Processing (NLP) to fine-tuned LLMs. This work shifts the focus from low-level artifacts to persistent conceptual structures, establishing a scalable and adaptive approach to tracking and countering disinformation campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05846v2">EMCee: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 under review, 21pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive progress across a wide range of tasks, yet their heavy reliance on English-centric training data leads to significant performance degradation in non-English languages. While existing multilingual prompting methods emphasize reformulating queries into English or enhancing reasoning capabilities, they often fail to incorporate the language- and culture-specific grounding that is essential for some queries. To address this limitation, we propose EMCee (Extracting synthetic Multilingual Context and merging), a simple yet effective framework that enhances the multilingual capabilities of LLMs by explicitly extracting and utilizing query-relevant knowledge from the LLM itself. In particular, EMCee first extracts synthetic context to uncover latent, language-specific knowledge encoded within the LLM, and then dynamically merges this contextual insight with reasoning-oriented outputs through a judgment-based selection mechanism. Extensive experiments on four multilingual benchmarks covering diverse languages and tasks demonstrate that EMCee consistently outperforms prior approaches, achieving an average relative improvement of 16.4% overall and 31.7% in low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00615v2">ACON: Optimizing Context Compression for Long-horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. Our code is available at https://github.com/microsoft/acon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15346v1">When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Ensembling Large Language Models (LLMs) has gained attention as a promising approach to surpass the performance of individual models by leveraging their complementary strengths. In particular, aggregating models' next-token probability distributions to select the next token has been shown to be effective in various tasks. However, while successful for short-form answers, its application to long-form generation remains underexplored. In this paper, we show that using existing ensemble methods in long-form generation requires a careful choice of ensembling positions, since the standard practice of ensembling at every token often degrades performance. We identify two key factors for determining these positions: tokenization mismatch across models and consensus in their next-token probability distributions. Based on this, we propose SAFE, (Stable And Fast LLM Ensembling), a framework that selectively ensembles by jointly considering these factors. To further improve stability, we introduce a probability sharpening strategy that consolidates probabilities spread across multiple sub-word tokens representing the same word into a single representative token. Our experiments on diverse benchmarks, including MATH500 and BBH, demonstrate that SAFE outperforms existing methods in both accuracy and efficiency, with gains achieved even when ensembling fewer than 1% of tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15330v1">BeLLMan: Controlling LLM Congestion</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 To be presented at FAISYS 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) applications are blindfolded to the infrastructure underneath and generate tokens autoregressively, indifferent to the system load, thus risking inferencing latency inflation and poor user experience. Our first-cut controller, named beLLMan, enables the LLM infrastructure to actively and progressively signal the first-party LLM application to adjust the output length in response to changing system load. On a real testbed with H100 GPUs, beLLMan helps keep inferencing latency under control (upto 8X lower end-to-end latency) and reduces energy consumption by 25% (while serving 19% more requests) during periods of congestion for a summarization workload.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15306v1">WebGen-V Bench: Structured Representation for Enhancing Visual Design in LLM-based Web Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Witnessed by the recent advancements on leveraging LLM for coding and multimodal understanding, we present WebGen-V, a new benchmark and framework for instruction-to-HTML generation that enhances both data quality and evaluation granularity. WebGen-V contributes three key innovations: (1) an unbounded and extensible agentic crawling framework that continuously collects real-world webpages and can leveraged to augment existing benchmarks; (2) a structured, section-wise data representation that integrates metadata, localized UI screenshots, and JSON-formatted text and image assets, explicit alignment between content, layout, and visual components for detailed multimodal supervision; and (3) a section-level multimodal evaluation protocol aligning text, layout, and visuals for high-granularity assessment. Experiments with state-of-the-art LLMs and ablation studies validate the effectiveness of our structured data and section-wise evaluation, as well as the contribution of each component. To the best of our knowledge, WebGen-V is the first work to enable high-granularity agentic crawling and evaluation for instruction-to-HTML generation, providing a unified pipeline from real-world data acquisition and webpage generation to structured multimodal assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15283v1">Exemplar-Guided Planing: Enhanced LLM Agent for KGQA</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as interactive agents show significant promise in Knowledge Graph Question Answering (KGQA) but often struggle with the semantic gap between natural language queries and structured knowledge graph (KG) representations. This leads to suboptimal planning and inefficient exploration on KG, while training-free approaches often underutilize valuable reasoning patterns in training data. To address these limitations, we propose a novel framework, Exemplar-Guided Planning (EGP), which enhances the planning capabilities of LLM agents for KGQA. EGP first preprocesses the training set questions via entity templating to normalize semantic variations. It then retrieves highly similar exemplary questions and their successful reasoning paths from this preprocessed set using semantic embeddings and an efficient FAISS index. These retrieved exemplars dynamically guide the LLM's planning process in two key phases: (1) Task Decomposition, by aligning generated sub-objectives with proven reasoning steps, and (2) Relation Exploration, by providing high-quality auxiliary information to improve relation pruning accuracy. Additionally, we introduce a Smart Lookahead mechanism during relation exploration to improve efficiency by preemptively exploring promising paths and potentially terminating exploration earlier. We apply EGP to the Plan-on-Graph (PoG) framework, termed PoG-EGP. Extensive experiments on two real-world KGQA datasets, WebQSP and CWQ, demonstrate that PoG-EGP significantly improves over the baseline PoG system and other compared methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14005v2">PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 The code is available at https://github.com/weizou52/PIShield
    </div>
    <details class="paper-abstract">
      LLM-integrated applications are vulnerable to prompt injection attacks, where an attacker contaminates the input to inject malicious prompts, causing the LLM to follow the attacker's intent instead of the original user's. Existing prompt injection detection methods often have sub-optimal performance and/or high computational overhead. In this work, we propose PIShield, a detection method that is both effective and efficient. Our key observation is that the internal representation of the final token in a prompt-extracted from a specific layer of the LLM, which we term the injection-critical layer-captures distinguishing features between clean and contaminated prompts. Leveraging this insight, we train a simple linear classifier on these internal representations using a labeled set of clean and contaminated prompts. We compare PIShield against 11 baselines across 5 diverse benchmark datasets and 8 prompt injection attacks. The results demonstrate that PIShield is both highly effective and efficient, substantially outperforming existing methods. Additionally, we show that PIShield resists strong adaptive attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15261v1">AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 LAW 2025 Workshop at NeurIPS 2025. Work done from late 2023 to early 2024
    </div>
    <details class="paper-abstract">
      Riding on the success of LLMs with retrieval-augmented generation (RAG), there has been a growing interest in augmenting agent systems with external memory databases. However, the existing systems focus on storing text information in their memory, ignoring the importance of multimodal signals. Motivated by the multimodal nature of human memory, we present AUGUSTUS, a multimodal agent system aligned with the ideas of human memory in cognitive science. Technically, our system consists of 4 stages connected in a loop: (i) encode: understanding the inputs; (ii) store in memory: saving important information; (iii) retrieve: searching for relevant context from memory; and (iv) act: perform the task. Unlike existing systems that use vector databases, we propose conceptualizing information into semantic tags and associating the tags with their context to store them in a graph-structured multimodal contextual memory for efficient concept-driven retrieval. Our system outperforms the traditional multimodal RAG approach while being 3.5 times faster for ImageNet classification and outperforming MemGPT on the MSC benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15258v1">Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 14 pages, 7 figures, 40 references
    </div>
    <details class="paper-abstract">
      In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15216v1">Soundness-Aware Level: A Microscopic Signature that Predicts LLM Reasoning Potential</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) can elicit strong reasoning in large language models (LLMs), while their performance after RLVR varies dramatically across different base models. This raises a fundamental question: what microscopic property of pre-trained models leads to this variation? To investigate, we formalize reasoning as chains of Horn clauses ("if-then" rules) built from features extracted from the LLM's latent space via cross-layer sparse autoencoders (SAEs). We estimate the transition probabilities between its features, and further categorize each rule by its semantic soundness level (e.g., strict, plausible, noisy) with an LLM. Our key discovery is that high-potential models are inherently soundness-aware: their internal probability distributions systematically shift across rules' soundness levels, becoming highly distinct for "strict" versus "noisy" rules. In contrast, weaker models are soundness-agnostic, collapsing to one distribution regardless of soundness levels. To quantify this, we introduce the Soundness-Aware Level (SAL), a microscopic metric using the Jensen-Shannon Divergence to measure the separation between these distributions. We show that SAL's predictions of post-RLVR reasoning performance follow a precise empirical law (R^2=0.87) across diverse model families (Qwen, Mistral, Llama, DeepSeek) and scales (0.5B-14B). This reveals that a model's reasoning potential is tied to its intrinsic, pre-trained ability to distinguish sound knowledge from unsound ones. These findings underscore the critical role of model pre-training in shaping reasoning and offer a practical metric grounded in the model's internal mechanisms for selecting/designing stronger base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09585v3">Elevating Visual Perception in Multimodal LLMs with Visual Embedding Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project Page: https://praeclarumjj3.github.io/visper_lm/
    </div>
    <details class="paper-abstract">
      In recent times, the standard practice for developing MLLMs is to feed features from vision encoder(s) into the LLM and train with natural language supervision. This approach often causes models to lean towards language comprehension and undermine the rich visual perception signals present in the data, which are critical for tasks involving spatial reasoning in the domain of embodied AI and robotics. Is it possible to optimize both at the same time? In this work, we propose VisPer-LM, the first approach that infuses visual perception knowledge from expert vision encoders into the LLM's (of an MLLM) hidden representations. We start by investigating MLLMs trained solely with natural language supervision and identify a positive correlation between the quality of visual representations within these models and their downstream performance. Given this insight, we formulate the objective during the pretraining stage in MLLMs as a coupled optimization of predictive visual embedding and next (text) token prediction. Moreover, through extensive probing, we observe improved visual representation quality due to embedding optimization, underscoring the effectiveness of our probing setup. We demonstrate that our VisPer-LM outperforms the single and multi-encoder baselines, proving our approach's superiority over explicitly feeding the corresponding features to the LLM. In particular, VisPer-LM boosts performance by an average margin of up to 2.5% on various benchmarks, with a notable improvement of 8.7% on the Depth task in CV-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12535v2">CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 42 pages, 9 tables
    </div>
    <details class="paper-abstract">
      Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby reducing spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma-2 2B and LLaMA-3.1 8B, notably achieving a +3.3% improvement in MMLU performance with 4000 samples and a +27.2% improvement in HarmBench with only 108 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlation-based selection as an effective and scalable approach for automated SAE steering across language model applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17792v3">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/sftekin/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12997v2">Max It or Miss It: Benchmarking LLM On Solving Extremal Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Our benchmark dataset is available at https://huggingface.co/datasets/binxingao/extrem-bench
    </div>
    <details class="paper-abstract">
      Test-time scaling has enabled Large Language Models (LLMs) with remarkable reasoning capabilities, particularly in mathematical domains, through intermediate chain-of-thought (CoT) reasoning before generating final answers. However, the specific sources and mechanisms underlying these reasoning capabilities remain insufficiently understood. Optimization reasoning, i.e. finding extrema under constraints, represents a fundamental abstraction that underpins critical applications in planning, control, resource allocation, and prompt search. To systematically evaluate this capability, we introduce ExtremBench, a benchmark dataset for solving mathematical extremal problems, curated from inequality exercises used for Chinese Mathematical Olympiad and transformed into $93$ standardized extrema-finding problems. We conduct extensive evaluations across various state-of-the-art open-source model families, including the Qwen3, GPT-OSS, and DeepSeek. Our results reveal that LLMs' extremal-solving reasoning capabilities do not always align with those of current mathematical benchmarks such as AIME25 and MATH-500, with some models showing strong general mathematical reasoning but poor extremal-solving skills, and vice versa. This discrepancy highlights a critical gap in current evaluation practices and suggests that existing benchmarks may not comprehensively capture the full spectrum of mathematical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10343v2">Code Execution as Grounded Supervision for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19131v2">ZeST: an LLM-based Zero-Shot Traversability Navigation for Unknown Environments</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      The advancement of robotics and autonomous navigation systems hinges on the ability to accurately predict terrain traversability. Traditional methods for generating datasets to train these prediction models often involve putting robots into potentially hazardous environments, posing risks to equipment and safety. To solve this problem, we present ZeST, a novel approach leveraging visual reasoning capabilities of Large Language Models (LLMs) to create a traversability map in real-time without exposing robots to danger. Our approach not only performs zero-shot traversability and mitigates the risks associated with real-world data collection but also accelerates the development of advanced navigation systems, offering a cost-effective and scalable solution. To support our findings, we present navigation results, in both controlled indoor and unstructured outdoor environments. As shown in the experiments, our method provides safer navigation when compared to other state-of-the-art methods, constantly reaching the final goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16846v2">BASIL: Bayesian Assessment of Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Sycophancy (overly agreeable or flattering behavior) is critical to understand in the context of human-AI collaboration, especially in decision-making settings like health, law, and education. Existing methods for studying sycophancy in LLMs are either descriptive (study behavior change when sycophancy is elicited) or normative (provide values-based judgment on behavior change). Together, these approaches help us understand the extent, and impacts, of sycophancy. However, existing normative approaches only apply for objective tasks where ground-truth data exists, ignoring the natural subjectivity in many NLP tasks. Drawing from behavioral economics and rational decision theory, we introduce an Bayesian framework to study the normative effects of sycophancy on rationality in LLMs, without requiring labeled ground-truth. Using this interdisciplinary framework, we study sycophantic behavior in multiple LLM baselines across three different tasks, experimenting with various methods for eliciting sycophancy and obtaining probability judgments from LLMs. We find significant evidence of sycophancy in our experiments (7 of 8 baselines for one of our probing techniques), and observe that sycophancy is more likely to reduce rationality than it is to increase rationality in LLMs' decisions when they are directly probed for probabilities (2 out of 4 baselines show significant increases overall).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04652v2">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04303v2">Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 13 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Multi-agent deployments of large language models (LLMs) are increasingly embedded in market, allocation, and governance workflows, yet covert coordination among agents can silently erode trust and social welfare. Existing audits are dominated by heuristics that lack theoretical guarantees, struggle to transfer across tasks, and seldom ship with the infrastructure needed for independent replication. We introduce Audit the Whisper, a conference-grade research artifact that spans theory, benchmark design, detection, and reproducibility. Our contributions are: (i) a channel-capacity analysis showing how interventions such as paraphrase, rate limiting, and role permutation impose quantifiable capacity penalties-operationalised via paired-run Kullback--Leibler diagnostics-that tighten mutual-information thresholds with finite-sample guarantees and full proofs; (ii) ColludeBench-v0, covering pricing, first-price auctions, peer review, and hosted Gemini/Groq APIs with configurable covert schemes, deterministic manifests, and reward instrumentation; and (iii) a calibrated auditing pipeline that fuses cross-run mutual information, permutation invariance, watermark variance, and fairness-aware acceptance bias, each tuned to a $10^{-3}$ false-positive budget and validated by 10k honest runs plus an e-value martingale. Across ColludeBench and external suites including Secret Collusion, CASE, Perfect Collusion Benchmark, and SentinelAgent, the union meta-test attains state-of-the-art power at fixed FPR while ablations surface price-of-auditing trade-offs and fairness-driven colluders invisible to MI alone. We release regeneration scripts, anonymized manifests, and documentation so that external auditors can reproduce every figure, satisfy double-blind requirements, and extend the framework with minimal effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07163v4">Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16156v1">AsyncVoice Agent: Real-Time Explanation for LLM Planning and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted to the IEEE ASRU 2025 Demo Track
    </div>
    <details class="paper-abstract">
      Effective human-AI collaboration on complex reasoning tasks requires that users understand and interact with the model's process, not just receive an output. However, the monolithic text from methods like Chain-of-Thought (CoT) prevents this, as current interfaces lack real-time verbalization and robust user barge-in. We present AsyncVoice Agent, a system whose asynchronous architecture decouples a streaming LLM backend from a conversational voice frontend. This design allows narration and inference to run in parallel, empowering users to interrupt, query, and steer the model's reasoning process at any time. Objective benchmarks show this approach reduces interaction latency by more than 600x compared to monolithic baselines while ensuring high fidelity and competitive task accuracy. By enabling a two-way dialogue with a model's thought process, AsyncVoice Agent offers a new paradigm for building more effective, steerable, and trustworthy human-AI systems for high-stakes tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16147v1">Procedural Scene Programs for Open-Universe Scene Generation: LLM-Free Error Correction via Program Search</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 To appear in SIGGRAPH Asia 2025
    </div>
    <details class="paper-abstract">
      Synthesizing 3D scenes from open-vocabulary text descriptions is a challenging, important, and recently-popular application. One of its critical subproblems is layout generation: given a set of objects, lay them out to produce a scene matching the input description. Nearly all recent work adopts a declarative paradigm for this problem: using an LLM to generate a specification of constraints between objects, then solving those constraints to produce the final layout. In contrast, we explore an alternative imperative paradigm, in which an LLM iteratively places objects, with each object's position and orientation computed as a function of previously-placed objects. The imperative approach allows for a simpler scene specification language while also handling a wider variety and larger complexity of scenes. We further improve the robustness of our imperative scheme by developing an error correction mechanism that iteratively improves the scene's validity while staying as close as possible to the original layout generated by the LLM. In forced-choice perceptual studies, participants preferred layouts generated by our imperative approach 82% and 94% of the time when compared against two declarative layout generation methods. We also present a simple, automated evaluation metric for 3D scene layout generation that aligns well with human preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16085v1">MoPHES:Leveraging on-device LLMs as Agent for Mobile Psychological Health Evaluation and Support</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      The 2022 World Mental Health Report calls for global mental health care reform, amid rising prevalence of issues like anxiety and depression that affect nearly one billion people worldwide. Traditional in-person therapy fails to meet this demand, and the situation is worsened by stigma. While general-purpose large language models (LLMs) offer efficiency for AI-driven mental health solutions, they underperform because they lack specialized fine-tuning. Existing LLM-based mental health chatbots can engage in empathetic conversations, but they overlook real-time user mental state assessment which is critical for professional counseling. This paper proposes MoPHES, a framework that integrates mental state evaluation, conversational support, and professional treatment recommendations. The agent developed under this framework uses two fine-tuned MiniCPM4-0.5B LLMs: one is fine-tuned on mental health conditions datasets to assess users' mental states and predict the severity of anxiety and depression; the other is fine-tuned on multi-turn dialogues to handle conversations with users. By leveraging insights into users' mental states, our agent provides more tailored support and professional treatment recommendations. Both models are also deployed directly on mobile devices to enhance user convenience and protect user privacy. Additionally, to evaluate the performance of MoPHES with other LLMs, we develop a benchmark for the automatic evaluation of mental state prediction and multi-turn counseling dialogues, which includes comprehensive evaluation metrics, datasets, and methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16082v1">Interpretable RNA-Seq Clustering with an LLM-Based Agentic Evidence-Grounded Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We propose CITE V.1, an agentic, evidence-grounded framework that leverages Large Language Models (LLMs) to provide transparent and reproducible interpretations of RNA-seq clusters. Unlike existing enrichment-based approaches that reduce results to broad statistical associations and LLM-only models that risk unsupported claims or fabricated citations, CITE V.1 transforms cluster interpretation by producing biologically coherent explanations explicitly anchored in the biomedical literature. The framework orchestrates three specialized agents: a Retriever that gathers domain knowledge from PubMed and UniProt, an Interpreter that formulates functional hypotheses, and Critics that evaluate claims, enforce evidence grounding, and qualify uncertainty through confidence and reliability indicators. Applied to Salmonella enterica RNA-seq data, CITE V.1 generated biologically meaningful insights supported by the literature, while an LLM-only Gemini baseline frequently produced speculative results with false citations. By moving RNA-seq analysis from surface-level enrichment to auditable, interpretable, and evidence-based hypothesis generation, CITE V.1 advances the transparency and reliability of AI in biomedicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16081v1">SARHAchat: An LLM-Based Chatbot for Sexual and Reproductive Health Counseling</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      While Artificial Intelligence (AI) shows promise in healthcare applications, existing conversational systems often falter in complex and sensitive medical domains such as Sexual and Reproductive Health (SRH). These systems frequently struggle with hallucination and lack the specialized knowledge required, particularly for sensitive SRH topics. Furthermore, current AI approaches in healthcare tend to prioritize diagnostic capabilities over comprehensive patient care and education. Addressing these gaps, this work at the UNC School of Nursing introduces SARHAchat, a proof-of-concept Large Language Model (LLM)-based chatbot. SARHAchat is designed as a reliable, user-centered system integrating medical expertise with empathetic communication to enhance SRH care delivery. Our evaluation demonstrates SARHAchat's ability to provide accurate and contextually appropriate contraceptive counseling while maintaining a natural conversational flow. The demo is available at https://sarhachat.com/}{https://sarhachat.com/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16080v1">TriAgent: Automated Biomarker Discovery with Deep Research Grounding for Triage in Acute Care by LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Emergency departments worldwide face rising patient volumes, workforce shortages, and variability in triage decisions that threaten the delivery of timely and accurate care. Current triage methods rely primarily on vital signs, routine laboratory values, and clinicians' judgment, which, while effective, often miss emerging biological signals that could improve risk prediction for infection typing or antibiotic administration in acute conditions. To address this challenge, we introduce TriAgent, a large language model (LLM)-based multi-agent framework that couples automated biomarker discovery with deep research for literature-grounded validation and novelty assessment. TriAgent employs a supervisor research agent to generate research topics and delegate targeted queries to specialized sub-agents for evidence retrieval from various data sources. Findings are synthesized to classify biomarkers as either grounded in existing knowledge or flagged as novel candidates, offering transparent justification and highlighting unexplored pathways in acute care risk stratification. Unlike prior frameworks limited to existing routine clinical biomarkers, TriAgent aims to deliver an end-to-end framework from data analysis to literature grounding to improve transparency, explainability and expand the frontier of potentially actionable clinical biomarkers. Given a user's clinical query and quantitative triage data, TriAgent achieved a topic adherence F1 score of 55.7 +/- 5.0%, surpassing the CoT-ReAct agent by over 10%, and a faithfulness score of 0.42 +/- 0.39, exceeding all baselines by more than 50%. Across experiments, TriAgent consistently outperformed state-of-the-art LLM-based agentic frameworks in biomarker justification and literature-grounded novelty assessment. We share our repo: https://github.com/CellFace/TriAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16079v1">EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at https://github.com/Edaizi/EvolveR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16062v1">Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 38 pages, 25 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Self-correction of large language models (LLMs) emerges as a critical component for enhancing their reasoning performance. Although various self-correction methods have been proposed, a comprehensive evaluation of these methods remains largely unexplored, and the question of whether LLMs can truly correct themselves is a matter of significant interest and concern. In this study, we introduce CorrectBench, a benchmark developed to evaluate the effectiveness of self-correction strategies, including intrinsic, external, and fine-tuned approaches, across three tasks: commonsense reasoning, mathematical reasoning, and code generation. Our findings reveal that: 1) Self-correction methods can improve accuracy, especially for complex reasoning tasks; 2) Mixing different self-correction strategies yields further improvements, though it reduces efficiency; 3) Reasoning LLMs (e.g., DeepSeek-R1) have limited optimization under additional self-correction methods and have high time costs. Interestingly, a comparatively simple chain-of-thought (CoT) baseline demonstrates competitive accuracy and efficiency. These results underscore the potential of self-correction to enhance LLM's reasoning performance while highlighting the ongoing challenge of improving their efficiency. Consequently, we advocate for further research focused on optimizing the balance between reasoning capabilities and operational efficiency. Project Page: https://correctbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14205v1">DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v2">How Far Have LLMs Come Toward Automated SATD Taxonomy Construction?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. In this study, we investigate to what extent large language models (LLMs) can generate SATD taxonomies. We designed a structured, LLM-driven pipeline that mirrors the taxonomy construction steps researchers typically follow. We evaluated it on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovered domain-specific categories reported in prior work, such as Layer Configuration in machine learning. It also completed taxonomy generation in under two hours and for less than $1, even on the largest dataset. These results suggest that, while full automation remains challenging, LLMs can support semi-automated SATD taxonomy construction. Furthermore, our work opens up avenues for future work, such as automated taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15191v1">Structure-R1: Dynamically Leveraging Structural Knowledge in LLM Reasoning through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable advances in reasoning capabilities. However, their performance remains constrained by limited access to explicit and structured domain knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external information as context to augment reasoning. Nevertheless, traditional RAG systems typically operate over unstructured and fragmented text, resulting in low information density and suboptimal reasoning. To overcome these limitations, we propose \textsc{Structure-R1}, a novel framework that transforms retrieved content into structured representations optimized for reasoning. Leveraging reinforcement learning, \textsc{Structure-R1} learns a content representation policy that dynamically generates and adapts structural formats based on the demands of multi-step reasoning. Unlike prior methods that rely on fixed schemas, our approach adopts a generative paradigm capable of producing task-specific structures tailored to individual queries. To ensure the quality and reliability of these representations, we introduce a self-reward structural verification mechanism that checks whether the generated structures are both correct and self-contained. Extensive experiments on seven knowledge-intensive benchmarks show that \textsc{Structure-R1} consistently achieves competitive performance with a 7B-scale backbone model and matches the performance of much larger models. Additionally, our theoretical analysis demonstrates how structured representations enhance reasoning by improving information density and contextual clarity. Our code and data are available at: https://github.com/jlwu002/sr1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15188v1">OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15152v1">Tail-Optimized Caching for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Prompt caching is critical for reducing latency and cost in LLM inference: OpenAI and Anthropic report up to 50-90% cost savings through prompt reuse. Despite its widespread success, little is known about what constitutes an optimal prompt caching policy, particularly when optimizing tail latency, a metric of central importance to practitioners. The widely used Least Recently Used (LRU) policy can perform arbitrarily poor on this metric, as it is oblivious to the heterogeneity of conversation lengths. To address this gap, we propose Tail-Optimized LRU, a simple two-line modification that reallocates KV cache capacity to prioritize high-latency conversations by evicting cache entries that are unlikely to affect future turns. Though the implementation is simple, we prove its optimality under a natural stochastic model of conversation dynamics, providing the first theoretical justification for LRU in this setting, a result that may be of independent interest to the caching community. Experimentally, on real conversation data WildChat, Tail-Optimized LRU achieves up to 27.5% reduction in P90 tail Time to First Token latency and 23.9% in P95 tail latency compared to LRU, along with up to 38.9% decrease in SLO violations of 200ms. We believe this provides a practical and theoretically grounded option for practitioners seeking to optimize tail latency in real-world LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15144v1">HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
    </div>
    <details class="paper-abstract">
      Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15125v1">Latent Topic Synthesis: Leveraging LLMs for Electoral Ad Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Under-submission
    </div>
    <details class="paper-abstract">
      Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically generating an interpretable topic taxonomy from an unlabeled corpus. By combining unsupervised clustering with prompt-based labeling, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets or domain expertise. We apply this framework to a large corpus of Meta (previously known as Facebook) political ads from the month ahead of the 2024 U.S. Presidential election. Our approach uncovers latent discourse structures, synthesizes semantically rich topic labels, and annotates topics with moral framing dimensions. We show quantitative and qualitative analyses to demonstrate the effectiveness of our framework. Our findings reveal that voting and immigration ads dominate overall spending and impressions, while abortion and election-integrity achieve disproportionate reach. Funding patterns are equally polarized: economic appeals are driven mainly by conservative PACs, abortion messaging splits between pro- and anti-rights coalitions, and crime-and-justice campaigns are fragmented across local committees. The framing of these appeals also diverges--abortion ads emphasize liberty/oppression rhetoric, while economic messaging blends care/harm, fairness/cheating, and liberty/oppression narratives. Topic salience further reveals strong correlations between moral foundations and issues. Demographic targeting also emerges. This work supports scalable, interpretable analysis of political messaging on social media, enabling researchers, policymakers, and the public to better understand emerging narratives, polarization dynamics, and the moral underpinnings of digital political communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15112v1">AndroByte: LLM-Driven Privacy Analysis through Bytecode Summarization and Dynamic Dataflow Call Graph Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted at the Annual Computer Security Applications Conference (ACSAC) 2025
    </div>
    <details class="paper-abstract">
      With the exponential growth in mobile applications, protecting user privacy has become even more crucial. Android applications are often known for collecting, storing, and sharing sensitive user information such as contacts, location, camera, and microphone data often without the user's clear consent or awareness raising significant privacy risks and exposure. In the context of privacy assessment, dataflow analysis is particularly valuable for identifying data usage and potential leaks. Traditionally, this type of analysis has relied on formal methods, heuristics, and rule-based matching. However, these techniques are often complex to implement and prone to errors, such as taint explosion for large programs. Moreover, most existing Android dataflow analysis methods depend heavily on predefined list of sinks, limiting their flexibility and scalability. To address the limitations of these existing techniques, we propose AndroByte, an AI-driven privacy analysis tool that leverages LLM reasoning on bytecode summarization to dynamically generate accurate and explainable dataflow call graphs from static code analysis. AndroByte achieves a significant F\b{eta}-Score of 89% in generating dynamic dataflow call graphs on the fly, outperforming the effectiveness of traditional tools like FlowDroid and Amandroid in leak detection without relying on predefined propagation rules or sink lists. Moreover, AndroByte's iterative bytecode summarization provides comprehensive and explainable insights into dataflow and leak detection, achieving high, quantifiable scores based on the G-Eval metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15081v1">A Generalizable Rhetorical Strategy Annotation Model Using LLM-based Debate Simulation and Labelling</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Rhetorical strategies are central to persuasive communication, from political discourse and marketing to legal argumentation. However, analysis of rhetorical strategies has been limited by reliance on human annotation, which is costly, inconsistent, difficult to scale. Their associated datasets are often limited to specific topics and strategies, posing challenges for robust model development. We propose a novel framework that leverages large language models (LLMs) to automatically generate and label synthetic debate data based on a four-part rhetorical typology (causal, empirical, emotional, moral). We fine-tune transformer-based classifiers on this LLM-labeled dataset and validate its performance against human-labeled data on this dataset and on multiple external corpora. Our model achieves high performance and strong generalization across topical domains. We illustrate two applications with the fine-tuned model: (1) the improvement in persuasiveness prediction from incorporating rhetorical strategy labels, and (2) analyzing temporal and partisan shifts in rhetorical strategies in U.S. Presidential debates (1960-2020), revealing increased use of affective over cognitive argument in U.S. Presidential debates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v3">To Err Is Human; To Annotate, SILICON? Reducing Measurement Error in LLM Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation is foundational to management research and Large Language Models (LLMs) promise a cost-effective and scalable alternative to human annotation. The validity of insights drawn from LLM annotated data critically depends on minimizing the discrepancy between LLM assigned labels and the unobserved ground truth, as well as ensuring long-term reproducibility of results. We address the gap in the literature on LLM annotation by decomposing measurement error in LLM-based text annotation into four distinct sources: (1) guideline-induced error from inconsistent annotation criteria, (2) baseline-induced error from unreliable human reference standards, (3) prompt-induced error from suboptimal meta-instruction formatting, and (4) model-induced error from architectural differences across LLMs. We develop the SILICON methodology to systematically reduce measurement error from LLM annotation in all four sources above. Empirical validation across seven management research cases shows iteratively refined guidelines substantially increases the LLM-human agreement compared to one-shot guidelines; expert-generated baselines exhibit higher inter-annotator agreement as well as are less prone to producing misleading LLM-human agreement estimates compared to crowdsourced baselines; placing content in the system prompt reduces prompt-induced error; and model performance varies substantially across tasks. To further reduce error, we introduce a cost-effective multi-LLM labeling method, where only low-confidence items receive additional labels from alternative models. Finally, in addressing closed source model retirement cycles, we introduce an intuitive regression-based methodology to establish robust reproducibility protocols. Our evidence indicates that reducing each error source is necessary, and that SILICON supports reproducible, rigorous annotation in management research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v4">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted at ACM International Conference on AI in Finance (ICAIF)
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts arising from discrepancies between their pre-trained parametric knowledge and real-time market data. These conflicts are especially problematic in real-world investment services, where a model's inherent biases can misalign with institutional objectives, leading to unreliable recommendations. Despite this risk, the intrinsic investment biases of LLMs remain underexplored. We propose an experimental framework to investigate emergent behaviors in such conflict scenarios, offering a quantitative analysis of bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent biases of models and measure their persistence. Our analysis, centered on sector, size, and momentum, reveals distinct, model-specific biases. Across most models, a tendency to prefer technology stocks, large-cap stocks, and contrarian strategies is observed. These foundational biases often escalate into confirmation bias, causing models to cling to initial judgments even when faced with increasing counter-evidence. A public leaderboard benchmarking bias across a broader set of models is available at https://linqalpha.com/leaderboard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15017v1">Active Honeypot Guardrail System: Probing and Confirming Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 6pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly vulnerable to multi-turn jailbreak attacks, where adversaries iteratively elicit harmful behaviors that bypass single-turn safety filters. Existing defenses predominantly rely on passive rejection, which either fails against adaptive attackers or overly restricts benign users. We propose a honeypot-based proactive guardrail system that transforms risk avoidance into risk utilization. Our framework fine-tunes a bait model to generate ambiguous, non-actionable but semantically relevant responses, which serve as lures to probe user intent. Combined with the protected LLM's safe reply, the system inserts proactive bait questions that gradually expose malicious intent through multi-turn interactions. We further introduce the Honeypot Utility Score (HUS), measuring both the attractiveness and feasibility of bait responses, and use a Defense Efficacy Rate (DER) for balancing safety and usability. Initial experiment on MHJ Datasets with recent attack method across GPT-4o show that our system significantly disrupts jailbreak success while preserving benign user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14973v1">Attention Is All You Need for KV Cache in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 https://vila-lab.github.io/elastic-cache-webpage/
    </div>
    <details class="paper-abstract">
      This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14972v1">TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14967v1">Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14966v1">Identity-Link IRT for Label-Free LLM Evaluation: Preserving Additivity in TVD-MI Scores</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Pairwise comparisons of large language models using total variation distance mutual information (TVD-MI) produce binary critic decisions per pair. We show that averaging TVD-MI's binary trials yields centered-probability scores with additive structure suitable for item-response theory (IRT) without nonlinear link functions. Maximum-likelihood approaches to IRT use logistic links, but we find empirically that these transformations introduce curvature that breaks additivity: across three domains, the identity link yields median curl on raw data of 0.080-0.150 (P95 = [0.474, 0.580]), whereas probit/logit introduce substantially higher violations (median [0.245, 0.588], P95 [0.825, 2.252]). We derive this clipped-linear model from Gini entropy maximization, yielding a box-constrained least-squares formulation that handles boundary saturation. At 33% coverage, we achieve holdout RMSE $0.117 \pm 0.008$ while preserving agent rankings (Spearman $\rho = 0.972 \pm 0.015$), three times fewer evaluations than full dense. Judge robustness analysis (GPT-4o-mini vs. Llama3-70b) shows strong agreement in agent rankings ($\rho = 0.872$) and consistent identity-link advantage. TVD-MI's geometry is best preserved by identity mapping for efficient LLM evaluation, applicable to other bounded-response domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14944v1">MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 22 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23339v2">VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 6 pages, 1 figure, 1 algorithm, 5 tables, to be published in ISPACS 2025, unabridged version exists as arXiv:2506.23339v1
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate substantial promise for advancing scientific discovery, yet their deployment in disciplines demanding factual precision and specialized domain constraints presents significant challenges. Within molecular design for pharmaceutical development, these models can propose innovative molecular modifications but frequently generate chemically infeasible structures. We introduce VALID-Mol, a comprehensive framework that integrates chemical validation with LLM-driven molecular design, achieving an improvement in valid chemical structure generation from 3% to 83%. Our methodology synthesizes systematic prompt optimization, automated chemical verification, and domain-adapted fine-tuning to ensure dependable generation of synthesizable molecules with enhanced properties. Our contribution extends beyond implementation details to provide a transferable methodology for scientifically-constrained LLM applications with measurable reliability enhancements. Computational analyses indicate our framework generates promising synthesis candidates with up to 17-fold predicted improvements in target binding affinity while preserving synthetic feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13091v2">Unmasking Hiring Bias: Platform Data Analysis and Controlled Experiments on Bias in Online Freelance Marketplaces via RAG-LLM Generated Contents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Online freelance marketplaces, a rapidly growing part of the global labor market, are creating a fair environment where professional skills are the main factor for hiring. While these platforms can reduce bias from traditional hiring, the personal information in user profiles raises concerns about ongoing discrimination. Past studies on this topic have mostly used existing data, which makes it hard to control for other factors and clearly see the effect of things like gender or race. To solve these problems, this paper presents a new method that uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to create realistic, artificial freelancer profiles for controlled experiments. This approach effectively separates individual factors, enabling a clearer statistical analysis of how different variables influence the freelancer project process. In addition to analyzing extracted data with traditional statistical methods for post-project stage analysis, our research utilizes a dataset with highly controlled variables, generated by an RAG-LLM, to conduct a simulated hiring experiment for pre-project stage analysis. The results of our experiments show that, regarding gender, while no significant preference emerged in initial hiring decisions, female freelancers are substantially more likely to receive imperfect ratings post-project stage. Regarding regional bias, a strong and consistent preference favoring US-based freelancers shows that people are more likely to be selected in the simulated experiments, perceived as more leader-like, and receive higher ratings on the live platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06948v2">Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach suffers from catastrophic forgetting: second-stage RL gradually loses SFT-acquired behaviors and inefficiently explores new patterns. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v3">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14846v1">Where to Search: Measure the Prior-Structured Search Space of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 10 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14824v1">Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      In information retrieval, training reranking models mainly focuses on two types of objectives: metric learning (e.g. contrastive loss to increase the predicted scores on relevant query-document pairs) and classification (binary label prediction of relevance vs. irrelevance). For BERT-style encoders, various studies have shown that contrastive learning (CL) can be more effective than discriminative (classification) learning. However, for large language models (LLMs), classification via supervised fine-tuning (SFT), which predicts ''yes'' (resp. ''no'') token for relevant (resp. irrelevant) pairs, appears more promising as it aligns well with the generative nature of LLMs. This divergence raises a central question: which objective is intrinsically better suited to LLM-based reranking, and what mechanism underlies the difference? In this work, we conduct a comprehensive comparison and analysis between CL and SFT for reranking, taking the universal multimodal retrieval (UMR) as the experimental playground. We first decompose the objectives into two components: weight, which controls the magnitude of those updates, and direction, which guides the model updates, then present a unified framework for understanding their interactions. Through probing experiments, we find that SFT provides a substantially stronger weighting scheme than CL, whereas the preferred scoring direction shows no clear winner. Taken together, these results point to a consistent advantage of SFT over CL for LLM reranking. To further validate our findings, we conduct large-scale training with SFT and present new state-of-the-art rerankers on the MRB benchmark. We also provide ablations on SFT settings and expect our findings to benefit future research and applications in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20921v2">LLM-guided Chemical Process Optimization with a Multi-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 16 pages (main manuscript without references), 4 figures
    </div>
    <details class="paper-abstract">
      Chemical process optimization maximizes production efficiency and economic performance, but optimization algorithms, including gradient-based solvers, numerical methods, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable. We present a multi-agent LLM framework that autonomously infers operating constraints from minimal process descriptions, then collaboratively guides optimization. Our AutoGen-based framework employs OpenAI's o3 model with specialized agents for constraint generation, parameter validation, simulation, and optimization guidance. Through autonomous constraint generation and iterative multi-agent optimization, the framework eliminates the need for predefined operational bounds. Validated on hydrodealkylation across cost, yield, and yield-to-cost ratio metrics, the framework achieved competitive performance with conventional methods while reducing wall-time 31-fold relative to grid search, converging in under 20 minutes. The reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs and applying domain-informed heuristics. Unlike conventional methods requiring predefined constraints, our approach uniquely combines autonomous constraint generation with interpretable parameter exploration. Model comparison reveals reasoning-capable architectures (o3, o1) are essential for successful optimization, while standard models fail to converge. This approach is particularly valuable for emerging processes and retrofit applications where operational constraints are poorly characterized or unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20934v2">Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at the International Conference on Software Maintenance and Evolution (ICSME'25)
    </div>
    <details class="paper-abstract">
      MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07887v2">Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14756v1">Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate hardware design tasks, including the generation of Verilog code. While early benchmarks focus primarily on functional correctness, efficient hardware design demands additional optimization for synthesis metrics such as area, delay, and power. Existing benchmarks fall short in evaluating these aspects comprehensively: they often lack optimized baselines or testbenches for verification. To address these gaps, we present Pluto, a benchmark and evaluation framework designed to assess the efficiency of LLM-generated Verilog designs. Pluto presents a comprehensive evaluation set of 114 problems with self-checking testbenches and multiple Pareto-optimal reference implementations. Experimental results show that state-of-the-art LLMs can achieve high functional correctness, reaching 78.3\% at pass@1, but their synthesis efficiency still lags behind expert-crafted implementations, with area efficiency of 63.8\%, delay efficiency of 65.9\%, and power efficiency of 64.0\% at eff@1. This highlights the need for efficiency-aware evaluation frameworks such as Pluto to drive progress in hardware-focused LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14751v1">Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Next-token prediction (NTP) has driven the success of large language models (LLMs), but it struggles with long-horizon reasoning, planning, and creative writing, with these limitations largely attributed to teacher-forced training. Multi-token prediction (MTP) partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement. We propose future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations. We explore two variants of FSP: handcrafted summaries, for example, a bag of words summary of the future of the sequence, and learned summaries, which use embeddings produced by a reverse language model trained from right to left. Large-scale pretraining experiments (3B and 8B-parameter models) demonstrate that FSP provides improvements over both NTP and MTP across math, reasoning, and coding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v3">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16191v2">The Last Dependency Crusade: Solving Python Dependency Conflicts with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Pre-print - Accepted at the first annual workshop on Agentic Software Engineering (AgenticSE) co-located with ASE'25
    </div>
    <details class="paper-abstract">
      Resolving Python dependency issues remains a tedious and error-prone process, forcing developers to manually trial compatible module versions and interpreter configurations. Existing automated solutions, such as knowledge-graph-based and database-driven methods, face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This paper investigates the use of Large Language Models (LLMs) to automatically repair dependency issues in Python programs. We propose PLLM (pronounced "plum"), a novel retrieval-augmented generation (RAG) approach that iteratively infers missing or incorrect dependencies. PLLM builds a test environment where the LLM proposes module combinations, observes execution feedback, and refines its predictions using natural language processing (NLP) to parse error messages. We evaluate PLLM on the Gistable HG2.9K dataset, a curated collection of real-world Python programs. Using this benchmark, we explore multiple PLLM configurations, including six open-source LLMs evaluated both with and without RAG. Our findings show that RAG consistently improves fix rates, with the best performance achieved by Gemma-2 9B when combined with RAG. Compared to two state-of-the-art baselines, PyEGo and ReadPyE, PLLM achieves significantly higher fix rates; +15.97\% more than ReadPyE and +21.58\% more than PyEGo. Further analysis shows that PLLM is especially effective for projects with numerous dependencies and those using specialized numerical or machine-learning libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14700v1">LLM Agents for Automated Web Vulnerability Reproduction: Are We There Yet?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in software engineering and cybersecurity tasks, including code generation, vulnerability discovery, and automated testing. One critical but underexplored application is automated web vulnerability reproduction, which transforms vulnerability reports into working exploits. Although recent advances suggest promising potential, challenges remain in applying LLM agents to real-world web vulnerability reproduction scenarios. In this paper, we present the first comprehensive evaluation of state-of-the-art LLM agents for automated web vulnerability reproduction. We systematically assess 20 agents from software engineering, cybersecurity, and general domains across 16 dimensions, including technical capabilities, environment adaptability, and user experience factors, on 3 representative web vulnerabilities. Based on the results, we select three top-performing agents (OpenHands, SWE-agent, and CAI) for in-depth evaluation on our benchmark dataset of 80 real-world CVEs spanning 7 vulnerability types and 6 web technologies. Our results reveal that while LLM agents achieve reasonable success on simple library-based vulnerabilities, they consistently fail on complex service-based vulnerabilities requiring multi-component environments. Complex environment configurations and authentication barriers create a gap where agents can execute exploit code but fail to trigger actual vulnerabilities. We observe high sensitivity to input guidance, with performance degrading by over 33% under incomplete authentication information. Our findings highlight the significant gap between current LLM agent capabilities and the demands of reliable automated vulnerability reproduction, emphasizing the need for advances in environmental adaptation and autonomous problem-solving capabilities.
    </details>
</div>
