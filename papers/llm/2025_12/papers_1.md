# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20616v2">Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Accepted by IEEE Control Systems Letters (L-CSS)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in knowledge acquisition, reasoning, and tool use, making them promising candidates for autonomous agent applications. However, training LLM agents for complex multi-turn task planning faces significant challenges, including sparse episode-wise rewards, credit assignment across long horizons, and the computational overhead of reinforcement learning in multi-turn interaction settings. To this end, this paper introduces a novel approach that transforms multi-turn task planning into single-turn task reasoning problems, enabling efficient policy optimization through Group Relative Policy Optimization (GRPO) with dense and verifiable reward from expert trajectories. Our theoretical analysis shows that GRPO improvement on single-turn task reasoning results in a lower bound of the multi-turn success probability under the minimal turns, as well as the generalization to subtasks with shorter horizons. Experimental evaluation on the complex task planning benchmark demonstrates that our 1.5B parameter model trained with single-turn GRPO achieves superior performance compared to larger baseline models up to 14B parameters, with success rates of 70% for long-horizon planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07797v1">LLM Use for Mental Health: Crowdsourcing Users' Sentiment-based Perspectives and Values from Social Discussions</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) chatbots like ChatGPT are increasingly used for mental health support. They offer accessible, therapeutic support but also raise concerns about misinformation, over-reliance, and risks in high-stakes contexts of mental health. We crowdsource large-scale users' posts from six major social media platforms to examine how people discuss their interactions with LLM chatbots across different mental health conditions. Through an LLM-assisted pipeline grounded in Value-Sensitive Design (VSD), we mapped the relationships across user-reported sentiments, mental health conditions, perspectives, and values. Our results reveal that the use of LLM chatbots is condition-specific. Users with neurodivergent conditions (e.g., ADHD, ASD) report strong positive sentiments and instrumental or appraisal support, whereas higher-risk disorders (e.g., schizophrenia, bipolar disorder) show more negative sentiments. We further uncover how user perspectives co-occur with underlying values, such as identity, autonomy, and privacy. Finally, we discuss shifting from "one-size-fits-all" chatbot design toward condition-specific, value-sensitive LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07795v1">ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 11 pages, 3 tables, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in settings where reasoning, such as multi-step problem solving and chain-of-thought, is essential. Yet, current evaluation practices overwhelmingly report single-run accuracy while ignoring the intrinsic uncertainty that naturally arises from stochastic decoding. This omission creates a blind spot because practitioners cannot reliably assess whether a method's reported performance is stable, reproducible, or cost-consistent. We introduce ReasonBENCH, the first benchmark designed to quantify the underlying instability in LLM reasoning. ReasonBENCH provides (i) a modular evaluation library that standardizes reasoning frameworks, models, and tasks, (ii) a multi-run protocol that reports statistically reliable metrics for both quality and cost, and (iii) a public leaderboard to encourage variance-aware reporting. Across tasks from different domains, we find that the vast majority of reasoning strategies and models exhibit high instability. Notably, even strategies with similar average performance can display confidence intervals up to four times wider, and the top-performing methods often incur higher and less stable costs. Such instability compromises reproducibility across runs and, consequently, the reliability of reported performance. To better understand these dynamics, we further analyze the impact of prompts, model families, and scale on the trade-off between solve rate and stability. Our results highlight reproducibility as a critical dimension for reliable LLM reasoning and provide a foundation for future reasoning methods and uncertainty quantification techniques. ReasonBENCH is publicly available at https://github.com/au-clan/ReasonBench .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07777v1">Mary, the Cheeseburger-Eating Vegetarian: Do LLMs Recognize Incoherence in Narratives?</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Leveraging a dataset of paired narratives, we investigate the extent to which large language models (LLMs) can reliably separate incoherent and coherent stories. A probing study finds that LLMs' internal representations can reliably identify incoherent narratives. However, LLMs generate responses to rating questions that fail to satisfactorily separate the coherent and incoherent narratives across several prompt variations, hinting at a gap in LLM's understanding of storytelling. The reasoning LLMs tested do not eliminate these deficits, indicating that thought strings may not be able to fully address the discrepancy between model internal state and behavior. Additionally, we find that LLMs appear to be more sensitive to incoherence resulting from an event that violates the setting (e.g., a rainy day in the desert) than to incoherence arising from a character violating an established trait (e.g., Mary, a vegetarian, later orders a cheeseburger), suggesting that LLMs may rely more on prototypical world knowledge than building meaning-based narrative coherence. The consistent asymmetry found in our results suggests that LLMs do not have a complete grasp on narrative coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06405v2">TOOL4POI: A Tool-Augmented LLM Framework for Next POI Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 A critical technical error was discovered during our internal review, leading to unreliable experimental results. The issue cannot be resolved, and the paper has also been formally withdrawn from AAAI 2026. We therefore request withdrawal of the arXiv version to maintain scientific accuracy
    </div>
    <details class="paper-abstract">
      Next Point-of-Interest (POI) recommendation is a fundamental task in location-based services. While recent advances leverage Large Language Model (LLM) for sequential modeling, existing LLM-based approaches face two key limitations: (i) strong reliance on the contextual completeness of user histories, resulting in poor performance on out-of-history (OOH) scenarios; (ii) limited scalability, due to the restricted context window of LLMs, which limits their ability to access and process a large number of candidate POIs. To address these challenges, we propose Tool4POI, a novel tool-augmented framework that enables LLMs to perform open-set POI recommendation through external retrieval and reasoning. Tool4POI consists of three key modules: preference extraction module, multi-turn candidate retrieval module, and reranking module, which together summarize long-term user interests, interact with external tools to retrieve relevant POIs, and refine final recommendations based on recent behaviors. Unlike existing methods, Tool4POI requires no task-specific fine-tuning and is compatible with off-the-shelf LLMs in a plug-and-play manner. Extensive experiments on three real-world datasets show that Tool4POI substantially outperforms state-of-the-art baselines, achieving up to 40% accuracy on challenging OOH scenarios where existing methods fail, and delivering average improvements of 20% and 30% on Acc@5 and Acc@10, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05156v2">Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 23 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07611v1">Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO for LLM Reasoning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      This study presents a systematic comparison of three Reinforcement Learning (RL) algorithms (PPO, GRPO, and DAPO) for improving complex reasoning in large language models (LLMs). Our main contribution is a controlled transfer-learning evaluation: models are first fine-tuned on the specialized Countdown Game and then assessed on a suite of general-purpose reasoning benchmarks. Across all tasks, RL-trained models outperform their corresponding base models, although the degree of improvement differs by benchmark. Our parametric analysis offers practical guidance for RL-based LLM training. Increasing the group size in GRPO and DAPO leads to more stable training dynamics and higher accuracy, while the impact of the KL-penalty coefficient is non-monotonic. Additionally, we find that the Dynamic Sampling (DS) component in DAPO does not improve performance; in fact, the best overall results are achieved with DAPO when DS is disabled.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06504v2">When Code Crosses Borders: A Security-Centric Study of LLM-based Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Code translation is crucial for cross-language codebase migration, and large language models (LLMs) have emerged as a promising technique to automate this process. However, the security implications of using LLMs for code translation remain largely unexplored, as existing evaluations primarily focus on syntactic and functional correctness. To bridge this gap, we conduct a security-centric empirical study to investigate the risks of vulnerabilities being introduced or preserved during LLM-based translation. Our study involves a rigorous evaluation of five state-of-the-art LLMs on a curated dataset of 720 security-related code samples across four programming languages (Java, PHP, C, C++) and nine Common Weakness Enumeration (CWE) categories. The results reveal significant security degradation, with 28.6\% to 45\% of translations introducing new vulnerabilities. Web-related flaws, particularly in input validation, proved most challenging for LLMs. Furthermore, we identify and categorize the root causes of these vulnerable translations into a taxonomy of five major error types. Based on our findings, we develop and evaluate a Retrieval-Augmented Generation (RAG)-based mitigation strategy, which successfully reduces the vulnerability introduction rate by 32.8\%. Our study provides the first large-scale evidence of serious security risks in LLM-based code translation and demonstrates the potential of knowledge-enhanced prompting to mitigate them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16653v2">H2EAL: Hybrid-Bonding Architecture with Hybrid Sparse Attention for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 International Conference on Computer-Aided Design (ICCAD) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in a wide range of natural language processing applications. However, the high energy and latency overhead induced by the KV cache limits the edge deployment, especially for long contexts. Emerging hybrid bonding (HB) technology has been proposed as a promising alternative to conventional near-memory processing (NMP) architectures, offering improved bandwidth efficiency and lower power consumption while exhibiting characteristics of distributed memory. In this paper, we propose H2EAL, a hybrid bonding-based accelerator with sparse attention algorithm-hardware co-design for efficient LLM inference at the edge. At the algorithm level, we propose a hybrid sparse attention scheme with static and dynamic sparsity for different heads to fully leverage the sparsity with high accuracy. At the hardware level, we co-design the hardware to support hybrid sparse attention and propose memory-compute co-placement to address the distributed memory bottleneck. Since different attention heads exhibit different sparse patterns and the attention structure often mismatches the HB architecture, we further develop a load-balancing scheduler with parallel tiled attention to address workload imbalance and optimize the mapping strategy. Extensive experiments demonstrate H2EAL achieves 5.20~48.21x speedup and 6.22~73.48x energy efficiency improvement over baseline HB implementation, with a negligible average accuracy drop of 0.87% on multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15830v4">Rethinking LLM Training through Information Geometry and Quantum Metrics</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 9 pages, 1 figure(s)
    </div>
    <details class="paper-abstract">
      Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-based approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07533v1">VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      We propose VulnLLM-R, the~\emph{first specialized reasoning LLM} for vulnerability detection. Our key insight is that LLMs can reason about program states and analyze the potential vulnerabilities, rather than simple pattern matching. This can improve the model's generalizability and prevent learning shortcuts. However, SOTA reasoning LLMs are typically ultra-large, closed-source, or have limited performance in vulnerability detection. To address this, we propose a novel training recipe with specialized data selection, reasoning data generation, reasoning data filtering and correction, and testing-phase optimization. Using our proposed methodology, we train a reasoning model with seven billion parameters. Through extensive experiments on SOTA datasets across Python, C/C++, and Java, we show that VulnLLM-R has superior effectiveness and efficiency than SOTA static analysis tools and both open-source and commercial large reasoning models. We further conduct a detailed ablation study to validate the key designs in our training recipe. Finally, we construct an agent scaffold around our model and show that it outperforms CodeQL and AFL++ in real-world projects. Our agent further discovers a set of zero-day vulnerabilities in actively maintained repositories. This work represents a pioneering effort to enable real-world, project-level vulnerability detection using AI agents powered by specialized reasoning models. The code is available at~\href{https://github.com/ucsb-mlsec/VulnLLM-R}{github}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07525v1">Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 20 pages, 6 figures, under review
    </div>
    <details class="paper-abstract">
      Rotary Position Embeddings (RoPE) have become a standard for encoding sequence order in Large Language Models (LLMs) by applying rotations to query and key vectors in the complex plane. Standard implementations, however, utilize only the real component of the complex-valued dot product for attention score calculation. This simplification discards the imaginary component, which contains valuable phase information, leading to a potential loss of relational details crucial for modeling long-context dependencies. In this paper, we propose an extension that re-incorporates this discarded imaginary component. Our method leverages the full complex-valued representation to create a dual-component attention score. We theoretically and empirically demonstrate that this approach enhances the modeling of long-context dependencies by preserving more positional information. Furthermore, evaluations on a suite of long-context language modeling benchmarks show that our method consistently improves performance over the standard RoPE, with the benefits becoming more significant as context length increases. The code is available at https://github.com/OpenMOSS/rope_pp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07501v1">AutoICE: Automatically Synthesizing Verifiable C Code via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Automatically synthesizing verifiable code from natural language requirements ensures software correctness and reliability while significantly lowering the barrier to adopting the techniques of formal methods. With the rise of large language models (LLMs), long-standing efforts at autoformalization have gained new momentum. However, existing approaches suffer from severe syntactic and semantic errors due to the scarcity of domain-specific pre-training corpora and often fail to formalize implicit knowledge effectively. In this paper, we propose AutoICE, an LLM-driven evolutionary search for synthesizing verifiable C code. It introduces the diverse individual initialization and the collaborative crossover to enable diverse iterative updates, thereby mitigating error propagation inherent in single-agent iterations. Besides, it employs the self-reflective mutation to facilitate the discovery of implicit knowledge. Evaluation results demonstrate the effectiveness of AutoICE: it successfully verifies $90.36$\% of code, outperforming the state-of-the-art (SOTA) approach. Besides, on a developer-friendly dataset variant, AutoICE achieves a $88.33$\% verification success rate, significantly surpassing the $65$\% success rate of the SOTA approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07497v1">How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 48 pages, 3 tables, 2 listings
    </div>
    <details class="paper-abstract">
      We investigate how large language models (LLMs) fail when operating as autonomous agents with tool-use capabilities. Using the Kamiwaza Agentic Merit Index (KAMI) v0.1 benchmark, we analyze 900 execution traces from three representative models - Granite 4 Small, Llama 4 Maverick, and DeepSeek V3.1 - across filesystem, text extraction, CSV analysis, and SQL scenarios. Rather than focusing on aggregate scores, we perform fine-grained, per-trial behavioral analysis to surface the strategies that enable successful multi-step tool execution and the recurrent failure modes that undermine reliability. Our findings show that model scale alone does not predict agentic robustness: Llama 4 Maverick (400B) performs only marginally better than Granite 4 Small (32B) in some uncertainty-driven tasks, while DeepSeek V3.1's superior reliability derives primarily from post-training reinforcement learning rather than architecture or size. Across models, we identify four recurring failure archetypes: premature action without grounding, over-helpfulness that substitutes missing entities, vulnerability to distractor-induced context pollution, and fragile execution under load. These patterns highlight the need for agentic evaluation methods that emphasize interactive grounding, recovery behavior, and environment-aware adaptation, suggesting that reliable enterprise deployment requires not just stronger models but deliberate training and design choices that reinforce verification, constraint discovery, and adherence to source-of-truth data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.07639v2">PowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Efficiently solving Optimal Power Flow (OPF) problems in power systems is crucial for operational planning and grid management. There is a growing need for scalable algorithms capable of handling the increasing variability, constraints, and uncertainties in modern power networks while providing accurate and fast solutions. To address this, machine learning techniques, particularly Graph Neural Networks (GNNs) have emerged as promising approaches. This letter introduces PowerGraph-LLM, the first framework explicitly designed for solving OPF problems using Large Language Models (LLMs). The proposed approach combines graph and tabular representations of power grids to effectively query LLMs, capturing the complex relationships and constraints in power systems. A new implementation of in-context learning and fine-tuning protocols for LLMs is introduced, tailored specifically for the OPF problem. PowerGraph-LLM demonstrates reliable performances using off-the-shelf LLM. Our study reveals the impact of LLM architecture, size, and fine-tuning and demonstrates our framework's ability to handle realistic grid components and constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07462v1">Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07454v1">Persian-Phi: Efficient Cross-Lingual Adaptation of Compact LLMs via Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      The democratization of AI is currently hindered by the immense computational costs required to train Large Language Models (LLMs) for low-resource languages. This paper presents Persian-Phi, a 3.8B parameter model that challenges the assumption that robust multilingual capabilities require massive model sizes or multilingual baselines. We demonstrate how Microsoft Phi-3 Mini -- originally a monolingual English model -- can be effectively adapted to Persian through a novel, resource-efficient curriculum learning pipeline. Our approach employs a unique "warm-up" stage using bilingual narratives (Tiny Stories) to align embeddings prior to heavy training, followed by continual pretraining and instruction tuning via Parameter-Efficient Fine-Tuning (PEFT). Despite its compact size, Persian-Phi achieves competitive results on Open Persian LLM Leaderboard in HuggingFace. Our findings provide a validated, scalable framework for extending the reach of state-of-the-art LLMs to underrepresented languages with minimal hardware resources. The Persian-Phi model is publicly available at https://huggingface.co/amirakhlaghiqqq/PersianPhi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07404v1">Do LLMs Trust the Code They Write?</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07375v1">LUNE: Efficient LLM Unlearning via LoRA Fine-Tuning with Negative Examples</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) possess vast knowledge acquired from extensive training corpora, but they often cannot remove specific pieces of information when needed, which makes it hard to handle privacy, bias mitigation, and knowledge correction. Traditional model unlearning approaches require computationally expensive fine-tuning or direct weight editing, making them impractical for real-world deployment. In this work, we introduce LoRA-based Unlearning with Negative Examples (LUNE), a lightweight framework that performs negative-only unlearning by updating only low-rank adapters while freezing the backbone, thereby localizing edits and avoiding disruptive global changes. Leveraging Low-Rank Adaptation (LoRA), LUNE targets intermediate representations to suppress (or replace) requested knowledge with an order-of-magnitude lower compute and memory than full fine-tuning or direct weight editing. Extensive experiments on multiple factual unlearning tasks show that LUNE: (I) achieves effectiveness comparable to full fine-tuning and memory-editing methods, and (II) reduces computational cost by about an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07374v1">Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Unlearning in large foundation models (e.g., LLMs) is essential for enabling dynamic knowledge updates, enforcing data deletion rights, and correcting model behavior. However, existing unlearning methods often require full-model fine-tuning or access to the original training data, which limits their scalability and practicality. In this work, we introduce Recover-to-Forget (R2F), a novel framework for efficient unlearning in LLMs based on reconstructing full-model gradient directions from low-rank LoRA adapter updates. Rather than performing backpropagation through the full model, we compute gradients with respect to LoRA parameters using multiple paraphrased prompts and train a gradient decoder to approximate the corresponding full-model gradients. To ensure applicability to larger or black-box models, the decoder is trained on a proxy model and transferred to target models. We provide a theoretical analysis of cross-model generalization and demonstrate that our method achieves effective unlearning while preserving general model performance. Experimental results demonstrate that R2F offers a scalable and lightweight alternative for unlearning in pretrained LLMs without requiring full retraining or access to internal parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07312v1">DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) is pushing AI accelerators toward increasingly powerful and specialized designs. Instead of further complicating software development with deeply hierarchical scratchpad memories (SPMs) and their asynchronous management, we investigate the opposite point of the design spectrum: a multi-core AI accelerator equipped with a shared system-level cache and application-aware management policies, which keeps the programming effort modest. Our approach exploits dataflow information available in the software stack to guide cache replacement (including dead-block prediction), in concert with bypass decisions and mechanisms that alleviate cache thrashing. We assess the proposal using a cycle-accurate simulator and observe substantial performance gains (up to 1.80x speedup) compared with conventional cache architectures. In addition, we build and validate an analytical model that takes into account the actual overlapping behaviors to extend the measurement results of our policies to real-world larger-scale workloads. Experiment results show that when functioning together, our bypassing and thrashing mitigation strategies can handle scenarios both with and without inter-core data sharing and achieve remarkable speedups. Finally, we implement the design in RTL and the area of our design is $\mathbf{0.064mm^2}$ with 15nm process, which can run at 2 GHz clock frequency. Our findings explore the potential of the shared cache design to assist the development of future AI accelerator systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24282v2">SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents excel at multi-step, tool-augmented tasks. However, smart homes introduce distinct challenges, requiring agents to handle latent user intents, temporal dependencies, device constraints, scheduling, and more. The main bottlenecks for developing smart home agents with such capabilities include the lack of a realistic simulation environment where agents can interact with devices and observe the results, as well as a challenging benchmark to evaluate them. To address this, we introduce $\textbf{SimuHome}$, a time-accelerated home environment that simulates smart devices, supports API calls, and reflects changes in environmental variables. By building the simulator on the Matter protocol, the global industry standard for smart home communication, SimuHome provides a high-fidelity environment, and agents validated in SimuHome can be deployed on real Matter-compliant devices with minimal adaptation. We provide a challenging benchmark of 600 episodes across twelve user query types that require the aforementioned capabilities. Our evaluation of 16 agents under a unified ReAct framework reveals distinct capabilities and limitations across models. Models under 7B parameters exhibited negligible performance across all query types. Even GPT-4.1, the best-performing standard model, struggled with implicit intent inference, state verification, and particularly temporal scheduling. While reasoning models such as GPT-5.1 consistently outperformed standard models on every query type, they required over three times the average inference time, which can be prohibitive for real-time smart home applications. This highlights a critical trade-off between task performance and real-world practicality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06041v4">DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      How can we effectively handle queries for on-device large language models (LLMs) with varying runtime constraints, such as latency and accuracy? Multi-scale quantization addresses this challenge by enabling memory-efficient runtime model adaptation of LLMs through the overlaying of multiple model variants quantized to different bitwidths. Meanwhile, an important question still remains open-ended: how can models be properly configured to match a target precision or latency? While mixed-precision offers a promising solution, we take this further by leveraging the key observation that the sensitivity of each layer dynamically changes across decoding steps. Building on this insight, we introduce DP-LLM, a novel mechanism that dynamically assigns precision to each layer based on input values. Experimental results across multiple models and benchmarks demonstrate that DP-LLM achieves a superior performance-latency trade-off, outperforming prior approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07246v1">Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Error detection (ED), which aims to identify incorrect or inconsistent cell values in tabular data, is important for ensuring data quality. Recent state-of-the-art ED methods leverage the pre-trained knowledge and semantic capability embedded in large language models (LLMs) to directly label whether a cell is erroneous. However, this LLM-as-a-labeler pipeline (1) relies on the black box, implicit decision process, thus failing to provide explainability for the detection results, and (2) is highly sensitive to prompts, yielding inconsistent outputs due to inherent model stochasticity, therefore lacking robustness. To address these limitations, we propose an LLM-as-an-inducer framework that adopts LLM to induce the decision tree for ED (termed TreeED) and further ensembles multiple such trees for consensus detection (termed ForestED), thereby improving explainability and robustness. Specifically, based on prompts derived from data context, decision tree specifications and output requirements, TreeED queries the LLM to induce the decision tree skeleton, whose root-to-leaf decision paths specify the stepwise procedure for evaluating a given sample. Each tree contains three types of nodes: (1) rule nodes that perform simple validation checks (e.g., format or range), (2) Graph Neural Network (GNN) nodes that capture complex patterns (e.g., functional dependencies), and (3) leaf nodes that output the final decision types (error or clean). Furthermore, ForestED employs uncertainty-based sampling to obtain multiple row subsets, constructing a decision tree for each subset using TreeED. It then leverages an Expectation-Maximization-based algorithm that jointly estimates tree reliability and optimizes the consensus ED prediction. Extensive xperiments demonstrate that our methods are accurate, explainable and robust, achieving an average F1-score improvement of 16.1% over the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05485v2">TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00417v3">CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04668v2">Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20993v2">Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17432v4">Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 13 pages, 8 figures, Accepted by ACM MM2025, Project page: https://garygutc.github.io/UniME
    </div>
    <details class="paper-abstract">
      The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored.In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLMś language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short and long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07122v1">RisConFix: LLM-based Automated Repair of Risk-Prone Drone Configurations</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Flight control software is typically designed with numerous configurable parameters governing multiple functionalities, enabling flexible adaptation to mission diversity and environmental uncertainty. Although developers and manufacturers usually provide recommendations for these parameters to ensure safe and stable operations, certain combinations of parameters with recommended values may still lead to unstable flight behaviors, thereby degrading the drone's robustness. To this end, we propose a Large Language Model (LLM) based approach for real-time repair of risk-prone configurations (named RisConFix) that degrade drone robustness. RisConFix continuously monitors the drone's operational state and automatically triggers a repair mechanism once abnormal flight behaviors are detected. The repair mechanism leverages an LLM to analyze relationships between configuration parameters and flight states, and then generates corrective parameter updates to restore flight stability. To ensure the validity of the updated configuration, RisConFix operates as an iterative process; it continuously monitors the drone's flight state and, if an anomaly persists after applying an update, automatically triggers the next repair cycle. We evaluated RisConFix through a case study of ArduPilot (with 1,421 groups of misconfigurations). Experimental results show that RisConFix achieved a best repair success rate of 97% and an optimal average number of repairs of 1.17, demonstrating its capability to effectively and efficiently repair risk-prone configurations in real time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11227v2">Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 Accepted by NeurIPS 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      The development of reasoning capabilities represents a critical frontier in large language models (LLMs) research, where reinforcement learning (RL) and process reward models (PRMs) have emerged as predominant methodological frameworks. Contrary to conventional wisdom, empirical evidence from DeepSeek-R1 demonstrates that pure RL training focused on mathematical problem-solving can progressively enhance reasoning abilities without PRM integration, challenging the perceived necessity of process supervision. In this study, we conduct a systematic investigation of the relationship between RL training and PRM capabilities. Our findings demonstrate that problem-solving proficiency and process supervision capabilities represent complementary dimensions of reasoning that co-evolve synergistically during pure RL training. In particular, current PRMs underperform simple baselines like majority voting when applied to state-of-the-art models such as DeepSeek-R1 and QwQ-32B. To address this limitation, we propose Self-PRM, an introspective framework in which models autonomously evaluate and rerank their generated solutions through self-reward mechanisms. Although Self-PRM consistently improves the accuracy of the benchmark (particularly with larger sample sizes), analysis exposes persistent challenges: The approach exhibits low precision (<10\%) on difficult problems, frequently misclassifying flawed solutions as valid. These analyses underscore the need for continued RL scaling to improve reward alignment and introspective accuracy. Overall, our findings suggest that PRM may not be essential for enhancing complex reasoning, as pure RL not only improves problem-solving skills but also inherently fosters robust PRM capabilities. We hope these findings provide actionable insights for building more reliable and self-aware complex reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07112v1">FOAM: Blocked State Folding for Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance due to their large parameter counts and extensive training data. However, their scale leads to significant memory bottlenecks during training, especially when using memory-intensive optimizers like Adam. Existing memory-efficient approaches often rely on techniques such as singular value decomposition (SVD), projections, or weight freezing, which can introduce substantial computational overhead, require additional memory for projections, or degrade model performance. In this paper, we propose Folded Optimizer with Approximate Moment (FOAM), a method that compresses optimizer states by computing block-wise gradient means and incorporates a residual correction to recover lost information. Theoretically, FOAM achieves convergence rates equivalent to vanilla Adam under standard non-convex optimization settings. Empirically, FOAM reduces total training memory by approximately 50\%, eliminates up to 90\% of optimizer state memory overhead, and accelerates convergence. Furthermore, FOAM is compatible with other memory-efficient optimizers, delivering performance and throughput that match or surpass both full-rank and existing memory-efficient baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00450v2">Dual Collaborative LLMs via Continual Fine-Tuning for Serendipitous Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Traditional recommendation systems tend to trap users in strong feedback loops by excessively pushing content aligned with their historical preferences, thereby limiting exploration opportunities and causing content fatigue. Although large language models (LLMs) demonstrate potential with their diverse content generation capabilities, existing LLM-enhanced dual-model frameworks face two major limitations: first, they overlook long-term preferences driven by group identity, leading to biased interest modeling; second, they suffer from static optimization flaws, as a one-time alignment process fails to leverage incremental user data for closed-loop optimization. To address these challenges, we propose the Co-Evolutionary Alignment (CoEA) method. For interest modeling bias, we introduce Dual-Stable Interest Exploration (DSIE) module, jointly modeling long-term group identity and short-term individual interests through parallel processing of behavioral sequences. For static optimization limitations, we design a Periodic Collaborative Optimization (PCO) mechanism. This mechanism regularly conducts preference verification on incremental data using the Relevance LLM, then guides the Novelty LLM to perform fine-tuning based on the verification results, and subsequently feeds back the output of the continually fine-tuned Novelty LLM to the Relevance LLM for re-evaluation, thereby achieving a dynamic closed-loop optimization. Extensive online and offline experiments verify the effectiveness of the CoEA model in serendipitous recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09442v3">Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07090v1">Leveraging KV Similarity for Online Structured Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Pruning has emerged as a promising direction for accelerating large language model (LLM) inference, yet existing approaches often suffer from instability because they rely on offline calibration data that may not generalize across inputs. In this work, we introduce Token Filtering, a lightweight online structured pruning technique that makes pruning decisions directly during inference without any calibration data. The key idea is to measure token redundancy via joint key-value similarity and skip redundant attention computations, thereby reducing inference cost while preserving critical information. To further enhance stability, we design a variance-aware fusion strategy that adaptively weights key and value similarity across heads, ensuring that informative tokens are retained even under high pruning ratios. This design introduces no additional memory overhead and provides a more reliable criterion for token importance. Extensive experiments on LLaMA-2 (7B/13B), LLaMA-3 (8B), and Mistral (7B) demonstrate that Token Filtering consistently outperforms prior structured pruning methods, preserving accuracy on commonsense reasoning benchmarks and maintaining strong performance on challenging tasks such as MMLU, even with 50% pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07086v1">ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 This version includes the final camera-ready manuscript accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07081v1">ClinNoteAgents: An LLM Multi-Agent System for Predicting and Interpreting Heart Failure 30-Day Readmission from Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2025-12-08
      | 💬 10 pages, 2 figures. Submitted to AMIA 2026 Informatics Summit Student Paper Track
    </div>
    <details class="paper-abstract">
      Heart failure (HF) is one of the leading causes of rehospitalization among older adults in the United States. Although clinical notes contain rich, detailed patient information and make up a large portion of electronic health records (EHRs), they remain underutilized for HF readmission risk analysis. Traditional computational models for HF readmission often rely on expert-crafted rules, medical thesauri, and ontologies to interpret clinical notes, which are typically written under time pressure and may contain misspellings, abbreviations, and domain-specific jargon. We present ClinNoteAgents, an LLM-based multi-agent framework that transforms free-text clinical notes into (1) structured representations of clinical and social risk factors for association analysis and (2) clinician-style abstractions for HF 30-day readmission prediction. We evaluate ClinNoteAgents on 3,544 notes from 2,065 patients (readmission rate=35.16%), demonstrating strong performance in extracting risk factors from free-text, identifying key contributing factors, and predicting readmission risk. By reducing reliance on structured fields and minimizing manual annotation and model training, ClinNoteAgents provides a scalable and interpretable approach to note-based HF readmission risk modeling in data-limited healthcare systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06982v1">LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning
    </div>
    <details class="paper-abstract">
      Designing state encoders for reinforcement learning (RL) with multiple information sources -- such as sensor measurements, time-series signals, image observations, and textual instructions -- remains underexplored and often requires manual design. We formalize this challenge as a problem of composite neural architecture search (NAS), where multiple source-specific modules and a fusion module are jointly optimized. Existing NAS methods overlook useful side information from the intermediate outputs of these modules -- such as their representation quality -- limiting sample efficiency in multi-source RL settings. To address this, we propose an LLM-driven NAS pipeline that leverages language-model priors and intermediate-output signals to guide sample-efficient search for high-performing composite state encoders. On a mixed-autonomy traffic control task, our approach discovers higher-performing architectures with fewer candidate evaluations than traditional NAS baselines and the LLM-based GENIUS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03994v2">Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted to the AAAI 2026 Deployable AI (DAI) Workshop
    </div>
    <details class="paper-abstract">
      Aligning proprietary large language models (LLMs) with internal organizational policies has become an urgent priority as organizations increasingly deploy LLMs in sensitive domains such as legal support, finance, and medical services. Beyond generic safety filters, enterprises require reliable mechanisms to detect policy violations within their regulatory and operational frameworks, where breaches can trigger legal and reputational risks. Existing content moderation frameworks, such as guardrails, remain largely confined to the safety domain and lack the robustness to capture nuanced organizational policies. LLM-as-a-judge and fine-tuning approaches, though flexible, introduce significant latency and lack interpretability. To address these limitations, we propose a training-free and efficient method that treats policy violation detection as an out-of-distribution (OOD) detection problem. Inspired by whitening techniques, we apply a linear transformation to decorrelate the model's hidden activations and standardize them to zero mean and unit variance, yielding near-identity covariance matrix. In this transformed space, we use the Euclidean norm as a compliance score to detect policy violations. The method requires only the policy text and a small number of illustrative samples, which makes it light-weight and easily deployable. On a challenging policy benchmark, our approach achieves state-of-the-art results, surpassing both existing guardrails and fine-tuned reasoning models. This work provides organizations with a practical and statistically grounded framework for policy-aware oversight of LLMs, advancing the broader goal of deployable AI governance. Code is available at: https://tinyurl.com/policy-violation-detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16978v2">Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      We present Lark, a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System (MAS). To address verbosity and stakeholder trade-offs, we integrate four mechanisms: (i) plasticity, which applies concise adjustments to candidate solutions; (ii) duplication and maturation, which copy high-performing candidates and specialize them into new modules; (iii) ranked-choice stakeholder aggregation using influence-weighted Borda scoring; and (iv) compute awareness via token-based penalties that reward brevity. The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores. In a controlled evaluation over 30 rounds comparing 14 systems, Lark Full achieves a mean rank of 2.55 (95% CI [2.17, 2.93]) and a mean composite score of 29.4/50 (95% CI [26.34, 32.46]), finishing Top-3 in 80% of rounds while remaining cost competitive with leading commercial models ($0.016 per task). Paired Wilcoxon tests confirm that all four mechanisms contribute significantly as ablating duplication/maturation yields the largest deficit (ΔScore = 3.5, Cohen's d_z = 2.53, p < 0.001), followed by plasticity (ΔScore = 3.4, d_z = 1.86), ranked-choice voting (ΔScore = 2.4, d_z = 1.20), and token penalties (ΔScore = 2.2, d_z = 1.63). Rather than a formal Markov Decision Process with constrained optimization, Lark is a practical, compute-aware neuroevolutionary loop that scales stakeholder-aligned strategy generation and makes trade-offs transparent through per-step metrics. Our work presents proof-of-concept findings and invites community feedback as we expand toward real-world validation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.12517v2">Interaction Context Often Increases Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      We investigate how the presence and type of interaction context shapes sycophancy in LLMs. Although real-world interactions allow models to mirror a user's values, preferences, and self-image, prior work often studies sycophancy in zero-shot settings devoid of context. Using two weeks of interaction context from 38 users, we evaluate two forms of sycophancy: (1) agreement sycophancy -- the tendency of models to produce overly affirmative responses, and (2) perspective sycophancy -- the extent to which models reflect a user's viewpoint. Agreement sycophancy tends to increase with the presence of user context, though model behavior varies based on the context type. User memory profiles are associated with the largest increases in agreement sycophancy (e.g. 45% for Gemini 2.5 Pro), and some models become more sycophantic even with non-user synthetic contexts (e.g. 15% for Llama 4 Scout). Perspective sycophancy increases only when models can accurately infer user viewpoints from interaction context. Overall, context shapes sycophancy in heterogeneous ways, underscoring the need for evaluations grounded in real-world interactions and raising questions for system design around extended conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20758v2">SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06914v1">SoK: Trust-Authorization Mismatch in LLM Agent Interactions</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly evolving into autonomous agents capable of interacting with the external world, significantly expanding their capabilities through standardized interaction protocols. However, this paradigm revives the classic cybersecurity challenges of agency and authorization in a novel and volatile context. As decision-making shifts from deterministic code logic to probabilistic inference driven by natural language, traditional security mechanisms designed for deterministic behavior fail. It is fundamentally challenging to establish trust for unpredictable AI agents and to enforce the Principle of Least Privilege (PoLP) when instructions are ambiguous. Despite the escalating threat landscape, the academic community's understanding of this emerging domain remains fragmented, lacking a systematic framework to analyze its root causes. This paper provides a unifying formal lens for agent-interaction security. We observed that most security threats in this domain stem from a fundamental mismatch between trust evaluation and authorization policies. We introduce a novel risk analysis model centered on this trust-authorization gap. Using this model as a unifying lens, we survey and classify the implementation paths of existing, often seemingly isolated, attacks and defenses. This new framework not only unifies the field but also allows us to identify critical research gaps. Finally, we leverage our analysis to suggest a systematic research direction toward building robust, trusted agents and dynamic authorization mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19108v3">A Multi-Language Perspective on the Robustness of LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 46 pages, 10 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large language models have gained significant traction and popularity in recent times, extending their usage to code-generation tasks. While this field has garnered considerable attention, the exploration of testing and evaluating the robustness of code generation models remains an ongoing endeavor. Previous studies have primarily focused on code generation models specifically for the Python language, overlooking other widely-used programming languages. In this work, we conduct a comprehensive comparative analysis to assess the robustness performance of several prominent code generation models and investigate whether robustness can be improved by repairing perturbed docstrings using an LLM. Furthermore, we investigate how their performance varies across different programming languages. To accomplish this, we introduce perturbations in four key areas of the prompt: DocString, functionname, syntax, and format. We have compiled and released a dedicated dataset for this purpose. This work presents our experimental findings, shedding light on the performance of code generation models in various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01268v4">AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06869v1">Rhea: Role-aware Heuristic Episodic Attention for Conversational LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable performance on single-turn tasks, yet their effectiveness deteriorates in multi-turn conversations. We define this phenomenon as cumulative contextual decay - a progressive degradation of contextual integrity caused by attention pollution, dilution, and drift. To address this challenge, we propose Rhea (Role-aware Heuristic Episodic Attention), a novel framework that decouples conversation history into two functionally independent memory modules: (1) an Instructional Memory (IM) that persistently stores high-fidelity global constraints via a structural priority mechanism, and (2) an Episodic Memory (EM) that dynamically manages user-model interactions via asymmetric noise control and heuristic context retrieval. During inference, Rhea constructs a high signal-to-noise context by applying its priority attention: selectively integrating relevant episodic information while always prioritizing global instructions. To validate this approach, experiments on multiple multi-turn conversation benchmarks - including MT-Eval and Long-MT-Bench+ - show that Rhea mitigates performance decay and improves overall accuracy by 1.04 points on a 10-point scale (a 16% relative gain over strong baselines). Moreover, Rhea maintains near-perfect instruction fidelity (IAR > 8.1) across long-horizon interactions. These results demonstrate that Rhea provides a principled and effective framework for building more precise, instruction-consistent conversational LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06867v1">Do Persona-Infused LLMs Affect Performance in a Strategic Reasoning Game?</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted at IJCNLP-AACL 2025
    </div>
    <details class="paper-abstract">
      Although persona prompting in large language models appears to trigger different styles of generated text, it is unclear whether these translate into measurable behavioral differences, much less whether they affect decision-making in an adversarial strategic environment that we provide as open-source. We investigate the impact of persona prompting on strategic performance in PERIL, a world-domination board game. Specifically, we compare the effectiveness of persona-derived heuristic strategies to those chosen manually. Our findings reveal that certain personas associated with strategic thinking improve game performance, but only when a mediator is used to translate personas into heuristic values. We introduce this mediator as a structured translation process, inspired by exploratory factor analysis, that maps LLM-generated inventory responses into heuristics. Results indicate our method enhances heuristic reliability and face validity compared to directly inferred heuristics, allowing us to better study the effect of persona types on decision making. These insights advance our understanding of how persona prompting influences LLM-based decision-making and propose a heuristic generation method that applies psychometric principles to LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06866v1">Less Is More, but Where? Dynamic Token Compression via LLM-Guided Keyframe Prior</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in Video Large Language Models (VLLMs) have achieved remarkable video understanding capabilities, yet face critical efficiency bottlenecks due to quadratic computational growth with lengthy visual token sequences of long videos. While existing keyframe sampling methods can improve temporal modeling efficiency, additional computational cost is introduced before feature encoding, and the binary frame selection paradigm is found suboptimal. Therefore, in this work, we propose Dynamic Token compression via LLM-guided Keyframe prior (DyToK), a training-free paradigm that enables dynamic token compression by harnessing VLLMs' inherent attention mechanisms. Our analysis reveals that VLLM attention layers naturally encoding query-conditioned keyframe priors, by which DyToK dynamically adjusts per-frame token retention ratios, prioritizing semantically rich frames while suppressing redundancies. Extensive experiments demonstrate that DyToK achieves state-of-the-art efficiency-accuracy tradeoffs. DyToK shows plug-and-play compatibility with existing compression methods, such as VisionZip and FastV, attaining 4.3x faster inference while preserving accuracy across multiple VLLMs, such as LLaVA-OneVision and Qwen2.5-VL. Code is available at https://github.com/yu-lin-li/DyToK .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06846v1">CKG-LLM: LLM-Assisted Detection of Smart Contract Access Control Vulnerabilities Based on Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Traditional approaches for smart contract analysis often rely on intermediate representations such as abstract syntax trees, control-flow graphs, or static single assignment form. However, these methods face limitations in capturing both semantic structures and control logic. Knowledge graphs, by contrast, offer a structured representation of entities and relations, enabling richer intermediate abstractions of contract code and supporting the use of graph query languages to identify rule-violating elements. This paper presents CKG-LLM, a framework for detecting access-control vulnerabilities in smart contracts. Leveraging the reasoning and code generation capabilities of large language models, CKG-LLM translates natural-language vulnerability patterns into executable queries over contract knowledge graphs to automatically locate vulnerable code elements. Experimental evaluation demonstrates that CKG-LLM achieves superior performance in detecting access-control vulnerabilities compared to existing tools. Finally, we discuss potential extensions of CKG-LLM as part of future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06836v1">Leveraging LLMs to support co-evolution between definitions and instances of textual DSLs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Software languages evolve over time for various reasons, such as the addition of new features. When the language's grammar definition evolves, textual instances that originally conformed to the grammar become outdated. For DSLs in a model-driven engineering context, there exists a plethora of techniques to co-evolve models with the evolving metamodel. However, these techniques are not geared to support DSLs with a textual syntax -- applying them to textual language definitions and instances may lead to the loss of information from the original instances, such as comments and layout information, which are valuable for software comprehension and maintenance. This study explores the potential of Large Language Model (LLM)-based solutions in achieving grammar and instance co-evolution, with attention to their ability to preserve auxiliary information when directly processing textual instances. By applying two advanced language models, Claude-3.5 and GPT-4o, and conducting experiments across seven case languages, we evaluated the feasibility and limitations of this approach. Our results indicate a good ability of the considered LLMs for migrating textual instances in small-scale cases with limited instance size, which are representative of a subset of cases encountered in practice. In addition, we observe significant challenges with the scalability of LLM-based solutions to larger instances, leading to insights that are useful for informing future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06781v1">From Description to Score: Can LLMs Quantify Vulnerabilities?</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20833v2">LLMs are Biased Evaluators But Not Biased for Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 15 pages, 14 tables, 5 figures Accepted to ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Recent studies have demonstrated that large language models (LLMs) exhibit significant biases in evaluation tasks, particularly in preferentially rating and favoring self-generated content. However, the extent to which this bias manifests in fact-oriented tasks, especially within retrieval-augmented generation (RAG) frameworks, where keyword extraction and factual accuracy take precedence over stylistic elements, remains unclear. Our study addresses this knowledge gap by simulating two critical phases of the RAG framework. In the first phase, LLMs evaluated human-authored and model-generated passages, emulating the \textit{pointwise reranking phase}. The second phase involves conducting pairwise reading comprehension tests to simulate the \textit{generation phase}. Contrary to previous findings indicating a self-preference in rating tasks, our results reveal no significant self-preference effect in RAG frameworks. Instead, we observe that factual accuracy significantly influences LLMs' output, even in the absence of prior knowledge. These findings are consistent among three common QA datasets (NQ, MARCO, TriviaQA Datasets) and 5 widely adopted language models (GPT-3.5, GPT-4o-mini, Gemini, LLaMA3, and Mistral). Our research contributes to the ongoing discourse on LLM biases and their implications for RAG-based system, offering insights that may inform the development of more robust and unbiased LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06776v1">From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at generation but dominant autoregressive (AR) decoding is inherently sequential, creating a throughput bottleneck. Diffusion Language Models (DLMs)--especially block-wise variants--enable parallel generation and intra-block bidirectional reasoning, yet training large DLMs from scratch is costly and wastes the knowledge in mature AR checkpoints. Prior "adaptation" attempts either modify logits or randomly grow attention masks to full-sequence diffusion, or simply transplant AR weights into a block-diffusion recipe, leaving a fundamental mismatch between AR causality and block-wise bidirectionality unaddressed. We reframe adaptation as a intra-paradigm path from AR to Block-Diffusion by viewing AR as Block-Diffusion with blocksize=1. Concretely, we design the pathway of adaptation as follows: we use a context-causal attention mask (causal in context, bidirectional only within the active block), an efficient parallel adaptation procedure, an auxiliary AR loss to maximize data utilization and retain pretrained knowledge, and gradual increment of the generation block size. The recipe integrates cleanly with masked block-diffusion and maintains train-inference consistency. Built on these components, NBDiff-7B (Base and Instruct) could inherit the long-context modeling and reasoning capabilities, and achieve state-of-the-art performance among the 7B-class DLMs, delivering strong gains on general-knowledge, math, and code benchmarks over strong baselines. These results demonstrate that principled AR-to-block-diffusion adaptation is an effective and compute-efficient alternative to training DLMs from scratch. Codes: https://github.com/YuchuanTian/NBDiff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06749v1">DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems are challenging to debug because failures often arise from long, branching interaction traces. The prevailing practice is to leverage LLMs for log-based failure localization, attributing errors to a specific agent and step. However, this paradigm has two key limitations: (i) log-only debugging lacks validation, producing untested hypotheses, and (ii) single-step or single-agent attribution is often ill-posed, as we find that multiple distinct interventions can independently repair the failed task. To address the first limitation, we introduce DoVer, an intervention-driven debugging framework, which augments hypothesis generation with active verification through targeted interventions (e.g., editing messages, altering plans). For the second limitation, rather than evaluating on attribution accuracy, we focus on measuring whether the system resolves the failure or makes quantifiable progress toward task success, reflecting a more outcome-oriented view of debugging. Within the Magnetic-One agent framework, on the datasets derived from GAIA and AssistantBench, DoVer flips 18-28% of failed trials into successes, achieves up to 16% milestone progress, and validates or refutes 30-60% of failure hypotheses. DoVer also performs effectively on a different dataset (GSMPlus) and agent framework (AG2), where it recovers 49% of failed trials. These results highlight intervention as a practical mechanism for improving reliability in agentic systems and open opportunities for more robust, scalable debugging methods for LLM-based multi-agent systems. Project website and code will be available at https://aka.ms/DoVer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06747v1">PrivLLMSwarm: Privacy-Preserving LLM-Driven UAV Swarms for Secure IoT Surveillance</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as powerful enablers for autonomous reasoning and natural-language coordination in unmanned aerial vehicle (UAV) swarms operating within Internet of Things (IoT) environments. However, existing LLM-driven UAV systems process sensitive operational data in plaintext, exposing them to privacy and security risks. This work introduces PrivLLMSwarm, a privacy-preserving framework that performs secure LLM inference for UAV swarm coordination through Secure Multi-Party Computation (MPC). The framework incorporates MPC-optimized transformer components with efficient approximations of nonlinear activations, enabling practical encrypted inference on resource-constrained aerial platforms. A fine-tuned GPT-based command generator, enhanced through reinforcement learning in simulation, provides reliable instructions while maintaining confidentiality. Experimental evaluation in urban-scale simulations demonstrates that PrivLLMSwarm achieves high semantic accuracy, low encrypted inference latency, and robust formation control under privacy constraints. Comparative analysis shows PrivLLMSwarm offers a superior privacy-utility balance compared to differential privacy, federated learning, and plaintext baselines. To support reproducibility, the full implementation including source code, MPC components, and a synthetic dataset is publicly available. PrivLLMSwarm establishes a practical foundation for secure, LLM-enabled UAV swarms in privacy-sensitive IoT applications including smart-city monitoring and emergency response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20964v2">OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) on the task of generating complete formal specifications for verifying the functional correctness of operating system kernels. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each of which is a long-context task of about 20k-30k tokens. The benchmark formulates the specification generation task as a program synthesis problem confined to a domain for specifying states and transitions. This formulation is provided to LLMs through a programming model. The LLMs must be able to understand the programming model and verification assumptions before delineating the correct search space for syntax and semantics and generating formal specifications. Guided by the operating system's high-level functional description, the LLMs are asked to generate a specification that fully describes all correct states and transitions for a potentially buggy code implementation of the operating system. Experimental results with 12 state-of-the-art LLMs indicate limited performance of existing LLMs on the specification generation task for operating system verification. Significant disparities in their performance highlight differences in their ability to handle long-context code generation tasks. The code are available at https://github.com/lishangyu-hkust/OSVBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06721v1">ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are emerging to transform daily life. However, existing LLM agents primarily follow a reactive paradigm, relying on explicit user instructions to initiate services, which increases both physical and cognitive workload. In this paper, we propose ProAgent, the first end-to-end proactive agent system that harnesses massive sensory contexts and LLM reasoning to deliver proactive assistance. ProAgent first employs a proactive-oriented context extraction approach with on-demand tiered perception to continuously sense the environment and derive hierarchical contexts that incorporate both sensory and persona cues. ProAgent then adopts a context-aware proactive reasoner to map these contexts to user needs and tool calls, providing proactive assistance. We implement ProAgent on Augmented Reality (AR) glasses with an edge server and extensively evaluate it on a real-world testbed, a public dataset, and through a user study. Results show that ProAgent achieves up to 33.4% higher proactive prediction accuracy, 16.8% higher tool-calling F1 score, and notable improvements in user satisfaction over state-of-the-art baselines, marking a significant step toward proactive assistants. A video demonstration of ProAgent is available at https://youtu.be/pRXZuzvrcVs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.11192v5">I Learn Better If You Speak My Language: Understanding the Superior Performance of Fine-Tuning Large Language Models with LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 The paper has been accepted to EMNLP 2024 (Main Conference) there is a follow up paper: Efficiently Selecting Response Generation Strategies for Synthetic Data Construction by Self-Aligned Perplexity Note: This is a revised version of arXiv:2402.11192 (v1, submitted 17 Feb 2024)
    </div>
    <details class="paper-abstract">
      This paper explores an intriguing observation: fine-tuning a large language model (LLM) with responses generated by a LLM often yields better results than using responses generated by humans, particularly in reasoning tasks. We conduct an in-depth investigation to understand why this occurs. Contrary to the common belief that these instances is due to the more detailed nature of LLM-generated content, our study identifies another contributing factor: an LLM is inherently more "familiar" with LLM generated responses. This familiarity is evidenced by lower perplexity before fine-tuning. We design a series of experiments to understand the impact of the "familiarity" and our conclusion reveals that this "familiarity" significantly impacts learning performance. Training with LLM-generated responses not only enhances performance but also helps maintain the model's capabilities in other reasoning tasks after fine-tuning on a specific task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06659v1">The Evolution of Agentic AI in Cybersecurity: From Single LLM Reasoners to Multi-Agent Systems and Autonomous Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-07
      | 💬 Accepted at ICAIC 2026
    </div>
    <details class="paper-abstract">
      Cybersecurity has become one of the earliest adopters of agentic AI, as security operations centers increasingly rely on multi-step reasoning, tool-driven analysis, and rapid decision-making under pressure. While individual large language models can summarize alerts or interpret unstructured reports, they fall short in real SOC environments that require grounded data access, reproducibility, and accountable workflows. In response, the field has seen a rapid architectural evolution from single-model helpers toward tool-augmented agents, distributed multi-agent systems, schema-bound tool ecosystems, and early explorations of semi-autonomous investigative pipelines. This survey presents a five-generation taxonomy of agentic AI in cybersecurity. It traces how capabilities and risks change as systems advance from text-only LLM reasoners to multi-agent collaboration frameworks and constrained-autonomy pipelines. We compare these generations across core dimensions - reasoning depth, tool use, memory, reproducibility, and safety. In addition, we also synthesize emerging benchmarks used to evaluate cyber-oriented agents. Finally, we outline the unresolved challenges that accompany this evolution, such as response validation, tool-use correctness, multi-agent coordination, long-horizon reasoning, and safeguards for high-impact actions. Collectively, this work provides a structured perspective on how agentic AI is taking shape within cybersecurity and what is required to ensure its safe and reliable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06655v1">GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20561v2">Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as rethinking and error correction, as a form of in-context exploration. However, the Markovian policy obtained from conventional RL training does not give rise to reflective exploration behaviors since the policy depends on the history only through the state and therefore has no incentive to enrich identical states with additional context. Instead, RL exploration is only useful during training to learn the optimal policy in a trial-and-error manner. Therefore, it remains unclear whether reflective reasoning will emerge during RL, or why it is beneficial. To remedy this, we recast reflective exploration within a Bayesian RL framework, which optimizes the expected return under a posterior distribution over Markov decision processes induced by the training data. This Bayesian formulation admits uncertainty-adaptive policies that, through belief updates, naturally incentivize information-gathering actions and induce self-reflection behaviors. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms conventional RL approaches, achieving superior test-time performance and token efficiency. Our code is available at https://github.com/shenao-zhang/BARL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06274v2">Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06607v1">A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-07
    </div>
    <details class="paper-abstract">
      Applying LLMs to predictive tasks in finance is challenging due to look-ahead bias resulting from their training on long time-series data. This precludes the backtests typically employed in finance since retraining frontier models from scratch with a specific knowledge cutoff is prohibitive. In this paper, we introduce a fast, effective, and low-cost alternative. Our method guides generation at inference time by adjusting the logits of a large base model using a pair of smaller, specialized models -- one fine-tuned on information to be forgotten and another on information to be retained. We demonstrate that our method effectively removes both verbatim and semantic knowledge, corrects biases, and outperforms prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12845v5">ExLLM: Experience-Enhanced LLM Optimization for Molecular Design and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 10 pages, under review
    </div>
    <details class="paper-abstract">
      Molecular design involves an enormous and irregular search space, where traditional optimizers such as Bayesian optimization, genetic algorithms, and generative models struggle to leverage expert knowledge or handle complex feedback. Recently, LLMs have been used as optimizers, achieving promising results on benchmarks such as PMO. However, existing approaches rely only on prompting or extra training, without mechanisms to handle complex feedback or maintain scalable memory. In particular, the common practice of appending or summarizing experiences at every query leads to redundancy, degraded exploration, and ultimately poor final outcomes under large-scale iterative search. We introduce ExLLM (Experience-Enhanced LLM optimization), an LLM-as-optimizer framework with three components: (1) a compact, evolving experience snippet tailored to large discrete spaces that distills non-redundant cues and improves convergence at low cost; (2) a simple yet effective k-offspring scheme that widens exploration per call and reduces orchestration cost; and (3) a lightweight feedback adapter that normalizes objectives for selection while formatting constraints and expert hints for iteration. ExLLM sets new state-of-the-art results on PMO and generalizes strongly in our setup, it sets records on circle packing and stellarator design, and yields consistent gains across additional domains requiring only a task-description template and evaluation functions to transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06590v1">Towards Efficient Hypergraph and Multi-LLM Agent Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 8 Pages
    </div>
    <details class="paper-abstract">
      Recommender Systems (RSs) have become the cornerstone of various applications such as e-commerce and social media platforms. The evolution of RSs is paramount in the digital era, in which personalised user experience is tailored to the user's preferences. Large Language Models (LLMs) have sparked a new paradigm - generative retrieval and recommendation. Despite their potential, generative RS methods face issues such as hallucination, which degrades the recommendation performance, and high computational cost in practical scenarios. To address these issues, we introduce HGLMRec, a novel Multi-LLM agent-based RS that incorporates a hypergraph encoder designed to capture complex, multi-behaviour relationships between users and items. The HGLMRec model retrieves only the relevant tokens during inference, reducing computational overhead while enriching the retrieval context. Experimental results show performance improvement by HGLMRec against state-of-the-art baselines at lower computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06556v1">Securing the Model Context Protocol: Defending LLMs Against Tool Poisoning and Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      The Model Context Protocol (MCP) enables Large Language Models to integrate external tools through structured descriptors, increasing autonomy in decision-making, task execution, and multi-agent workflows. However, this autonomy creates a largely overlooked security gap. Existing defenses focus on prompt-injection attacks and fail to address threats embedded in tool metadata, leaving MCP-based systems exposed to semantic manipulation. This work analyzes three classes of semantic attacks on MCP-integrated systems: (1) Tool Poisoning, where adversarial instructions are hidden in tool descriptors; (2) Shadowing, where trusted tools are indirectly compromised through contaminated shared context; and (3) Rug Pulls, where descriptors are altered after approval to subvert behavior. To counter these threats, we introduce a layered security framework with three components: RSA-based manifest signing to enforce descriptor integrity, LLM-on-LLM semantic vetting to detect suspicious tool definitions, and lightweight heuristic guardrails that block anomalous tool behavior at runtime. Through evaluation of GPT-4, DeepSeek, and Llama-3.5 across eight prompting strategies, we find that security performance varies widely by model architecture and reasoning method. GPT-4 blocks about 71 percent of unsafe tool calls, balancing latency and safety. DeepSeek shows the highest resilience to Shadowing attacks but with greater latency, while Llama-3.5 is fastest but least robust. Our results show that the proposed framework reduces unsafe tool invocation rates without model fine-tuning or internal modification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05069v2">SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 Code: https://github.com/sdc17/SwiReasoning, Website: https://swireasoning.github.io/
    </div>
    <details class="paper-abstract">
      Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06490v1">Optimizing LLMs Using Quantization for Mobile Execution</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 11 pages, 1 equation, 2 tables. Author Accepted Manuscript (AAM) of a paper published in Springer LNNS, ICT4SD 2025. DOI: 10.1007/978-3-032-06697-8_33
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer powerful capabilities, but their significant size and computational requirements hinder deployment on resource-constrained mobile devices. This paper investigates Post-Training Quantization (PTQ) for compressing LLMs for mobile execution. We apply 4-bit PTQ using the BitsAndBytes library with the Hugging Face Transformers framework to Meta's Llama 3.2 3B model. The quantized model is converted to GGUF format using llama.cpp tools for optimized mobile inference. The PTQ workflow achieves a 68.66% reduction in model size through 4-bit quantization, enabling the Llama 3.2 3B model to run efficiently on an Android device. Qualitative validation shows that the 4-bit quantized model can perform inference tasks successfully. We demonstrate the feasibility of running the quantized GGUF model on an Android device using the Termux environment and the Ollama framework. PTQ, especially at 4-bit precision combined with mobile-optimized formats like GGUF, provides a practical pathway for deploying capable LLMs on mobile devices, balancing model size and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06443v1">Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on edge devices. To meet strict resource constraints, real-world deployment has pushed LLM quantization from 8-bit to 4-bit, 2-bit, and now 1.58-bit. Combined with lookup table (LUT)-based inference, CPUs run these ultra-low-bit LLMs even faster than NPUs, opening new opportunities for ubiquitous on-device intelligence. However, this paper identifies that LUT-based inference underutilizes memory bandwidth during parallel inference, which is required for prefilling, test-time scaling, and other multi-token scenarios. The root cause is the scalar LUT paradigm, which performs repetitive and non-contiguous memory accesses for each token. To solve the issue, we propose vector LUT, a new lookup paradigm that constructs a unified LUT across parallel tokens, and performs a single $1 \rightarrow N$ lookup per index. To realize it efficiently, we further introduce (1) Vector LUT-Centric Tensor Layout, and (2) Cache-Aware Streamed Lookup techniques. Evaluations on 5 edge devices across 3 LLMs show that Vec-LUT outperforms state-of-the-art baselines by up to $4.2\times$. Our implementation is integrated into llama.cpp. The code is available at https://github.com/Cipherxzc/vlut.cpp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06432v1">HiveMind: Contribution-Guided Online Prompt Optimization of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 This paper was accepted to AAAI2026
    </div>
    <details class="paper-abstract">
      Recent advances in LLM-based multi-agent systems have demonstrated remarkable capabilities in complex decision-making scenarios such as financial trading and software engineering. However, evaluating each individual agent's effectiveness and online optimization of underperforming agents remain open challenges. To address these issues, we present HiveMind, a self-adaptive framework designed to optimize LLM multi-agent collaboration through contribution analysis. At its core, HiveMind introduces Contribution-Guided Online Prompt Optimization (CG-OPO), which autonomously refines agent prompts based on their quantified contributions. We first propose the Shapley value as a grounded metric to quantify each agent's contribution, thereby identifying underperforming agents in a principled manner for automated prompt refinement. To overcome the computational complexity of the classical Shapley value, we present DAG-Shapley, a novel and efficient attribution algorithm that leverages the inherent Directed Acyclic Graph structure of the agent workflow to axiomatically prune non-viable coalitions. By hierarchically reusing intermediate outputs of agents in the DAG, our method further reduces redundant computations, and achieving substantial cost savings without compromising the theoretical guarantees of Shapley values. Evaluated in a multi-agent stock-trading scenario, HiveMind achieves superior performance compared to static baselines. Notably, DAG-Shapley reduces LLM calls by over 80\% while maintaining attribution accuracy comparable to full Shapley values, establishing a new standard for efficient credit assignment and enabling scalable, real-world optimization of multi-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06401v1">LLMCFG-TGen: Using LLM-Generated Control Flow Graphs to Automatically Create Test Cases from Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      Appropriate test case generation is critical in software testing, significantly impacting the quality of the testing. Requirements-Based Test Generation (RBTG) derives test cases from software requirements, aiming to verify whether or not the system's behaviors align with user needs and expectations. Requirements are often documented in Natural Language (NL), with use-case descriptions being a popular method for capturing functional behaviors and interaction flows in a structured form. Large Language Models (LLMs) have shown strong potential for automating test generation directly from NL requirements. However, current LLM-based approaches may not provide comprehensive, non-redundant coverage. They may also fail to capture complex conditional logic in requirements, resulting in incomplete test cases. We propose a new approach that automatically generates test cases from NL use-case descriptions, called Test Generation based on LLM-generated Control Flow Graphs (LLMCFG-TGen). LLMCFG-TGen comprises three main steps: (1) An LLM transforms a use case into a structured CFG that encapsulates all potential branches; (2) The generated CFG is explored, and all complete execution paths are enumerated; and (3) The execution paths are then used to generate the test cases. To evaluate our proposed approach, we conducted a series of experiments. The results show that LLMs can effectively construct well-structured CFGs from NL use cases. Compared with the baseline methods, LLMCFG-TGen achieves full path coverage, improving completeness and ensuring clear and accurate test cases. Practitioner assessments confirm that LLMCFG-TGen produces logically consistent and comprehensive test cases, while substantially reducing manual effort. The findings suggest that coupling LLM-based semantic reasoning with structured modeling effectively bridges the gap between NL requirements and systematic test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06393v1">Less Is More for Multi-Step Logical Reasoning of LLM Generalisation Under Rule Removal, Paraphrasing, and Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across many natural language tasks, yet their generalisation to structural perturbations in logical contexts remains poorly understood. We introduce a controlled evaluation framework that probes reasoning reliability through four targeted stress tests: (1) rule deletion, removing either redundant or essential rules from a multi-step inference chain; (2) contradictory evidence injection; (3) logic-preserving rewrites generated through several families of equivalence laws (contrapositive, double negation, implication, De Morgan, identity, and commutativity); and (4) multi-law equivalence stacking that introduces 2-5 simultaneous logical transformations. Across three representative model families: BERT, Qwen2, and LLaMA-like models. Our experiments reveal a strikingly consistent pattern: all models achieve perfect accuracy on the base tasks and remain fully generalise to redundant rule deletion and all equivalence-based rewrites (single or multi-law), but fail sharply under essential rule deletion (dropping to 25% accuracy) and collapse completely in the presence of explicit contradictions (0% accuracy). These results demonstrate that LLMs possess stable invariance to semantic-preserving logical transformations, yet remain fundamentally brittle to missing or conflicting evidence. Our framework provides a clean diagnostic tool for isolating such reasoning failure modes and highlights persistent gaps in the logical generalisation abilities of current LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06351v1">LLM-Upgraded Graph Reinforcement Learning for Carbon-Aware Job Scheduling in Smart Manufacturing</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      This paper presents \textsc{Luca}, a \underline{l}arge language model (LLM)-\underline{u}pgraded graph reinforcement learning framework for \underline{c}arbon-\underline{a}ware flexible job shop scheduling. \textsc{Luca} addresses the challenges of dynamic and sustainable scheduling in smart manufacturing systems by integrating a graph neural network and an LLM, guided by a carefully designed in-house prompting strategy, to produce a fused embedding that captures both structural characteristics and contextual semantics of the latest scheduling state. This expressive embedding is then processed by a deep reinforcement learning policy network, which generates real-time scheduling decisions optimized for both makespan and carbon emission objectives. To support sustainability goals, \textsc{Luca} incorporates a dual-objective reward function that encourages both energy efficiency and scheduling timeliness. Experimental results on both synthetic and public datasets demonstrate that \textsc{Luca} consistently outperforms comparison algorithms. For instance, on the synthetic dataset, it achieves an average of 4.1\% and up to 12.2\% lower makespan compared to the best-performing comparison algorithm while maintaining the same emission level. On public datasets, additional gains are observed for both makespan and emission. These results demonstrate that \textsc{Luca} is effective and practical for carbon-aware scheduling in smart manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04478v2">Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has driven progress in reasoning tasks -- from program synthesis to scientific hypothesis generation -- yet their ability to handle ranked preferences and structured algorithms in combinatorial domains remains underexplored. We study matching markets, a core framework behind applications like resource allocation and ride-sharing, which require reconciling individual ranked preferences to ensure stable outcomes. We evaluate several state-of-the-art models on a hierarchy of preference-based reasoning tasks -- ranging from stable-matching generation to instability detection, instability resolution, and fine-grained preference queries -- to systematically expose their logical and algorithmic limitations in handling ranked inputs. Surprisingly, even top-performing models with advanced reasoning struggle to resolve instability in large markets, often failing to identify blocking pairs or execute algorithms iteratively. We further show that parameter-efficient fine-tuning (LoRA) significantly improves performance in small markets, but fails to bring about a similar improvement on large instances, suggesting the need for more sophisticated strategies to improve LLMs' reasoning with larger-context inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03383v2">UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by the current device workload, adding to the uncertainty of model deployment. We introduce UniQL, a unified post-training quantization and low-rank compression framework with on-device configurable pruning rates for edge LLMs. UniQL is a general framework that integrates quantization and low-rank compression for Transformers, State Space Models (SSMs), and hybrid models to support diverse edge applications. In our proposed joint framework, we introduce an efficient structured weight-sorting method that speeds up computation by 20x, quantization-aware singular value decomposition (SVD) to minimize quantization errors, state-aware weight sorting for SSMs, and a fused rotary positional embedding (RoPE) kernel for pruned models. Our framework performs weight-sorting, fine-tuning, and quantization in the cloud in a single-pass workflow, while enabling on-device configurable pruning rates up to 35%. Our experiments show that quantized and pruned models achieve a memory reduction of 4x-5.7x and a token-throughput improvement of 2.7x-3.4x, maintaining accuracy within 5% of the original models at 15% pruning across Transformers (Llama3 and Qwen2.5), SSMs (Mamba2), and hybrid models (Nemotron-H and Bamba-v2). The code and quantized models are available at: https://github.com/enyac-group/UniQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04416v2">DataGovBench: Benchmarking LLM Agents for Real-World Data Governance Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-06
      | 💬 Equal contribution: Zhou Liu and Zhaoyang Han. Corresponding authors: Yuanfeng Song and Wentao Zhang
    </div>
    <details class="paper-abstract">
      Data governance ensures data quality, security, and compliance through policies and standards, a critical foundation for scaling modern AI development. Recently, large language models (LLMs) have emerged as a promising solution for automating data governance by translating user intent into executable transformation code. However, existing benchmarks for automated data science often emphasize snippet-level coding or high-level analytics, failing to capture the unique challenge of data governance: ensuring the correctness and quality of the data itself. To bridge this gap, we introduce DataGovBench, a benchmark featuring 150 diverse tasks grounded in real-world scenarios, built on data from actual cases. DataGovBench employs a novel "reversed-objective" methodology to synthesize realistic noise and utilizes rigorous metrics to assess end-to-end pipeline reliability. Our analysis on DataGovBench reveals that current models struggle with complex, multi-step workflows and lack robust error-correction mechanisms. Consequently, we propose DataGovAgent, a framework utilizing a Planner-Executor-Evaluator architecture that integrates constraint-based planning, retrieval-augmented generation, and sandboxed feedback-driven debugging. Experimental results show that DataGovAgent significantly boosts the Average Task Score (ATS) on complex tasks from 39.7 to 54.9 and reduces debugging iterations by over 77.9 percent compared to general-purpose baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v3">KeyB2+: Summary-Augmented Block Selection for Scalable Long-Document Reranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced neural information retrieval (IR), yet applying them to long-document reranking remains computationally expensive and often ineffective when irrelevant content dominates. We begin with an in-depth analysis of decoder-only LLM attention and show that while some heads align with relevance signals, this alignment quickly deteriorates as irrelevant text accumulates. These observations highlight the necessity of explicit block selection to preserve focus and efficiency. We present KeyB2 and KeyB2+, a scalable reranking framework that selects and aggregates the most relevant blocks together with each document's summarization, ensuring that both localized evidence and global semantics are captured before LLM scoring. KeyB2 family support flexible selectors: BM25, bi-encoder, and cross-encoder, and adapts decoder-only LLMs to compute fine-grained relevance scores on the selected content. Experiments demonstrate that abstract-augmented block selection consistently improves retrieval effectiveness over strong baselines while substantially lowering inference cost, achieving new SOTA result on TREC DL 2019 document track (0.738 for NDCG@10). This establishes KeyB2+ as a practical and effective solution for scalable long-document reranking with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05630v3">How Not to Detect Prompt Injections with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      LLM-integrated applications and agents are vulnerable to prompt injection attacks, where adversaries embed malicious instructions within seemingly benign input data to manipulate the LLM's intended behavior. Recent defenses based on known-answer detection (KAD) scheme have reported near-perfect performance by observing an LLM's output to classify input data as clean or contaminated. KAD attempts to repurpose the very susceptibility to prompt injection as a defensive mechanism. We formally characterize the KAD scheme and uncover a structural vulnerability that invalidates its core security premise. To exploit this fundamental vulnerability, we methodically design an adaptive attack, DataFlip. It consistently evades KAD defenses, achieving detection rates as low as $0\%$ while reliably inducing malicious behavior with a success rate of $91\%$, all without requiring white-box access to the LLM or any optimization procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06228v1">Policy-based Sentence Simplification: Replacing Parallel Corpora with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      Sentence simplification aims to modify a sentence to make it easier to read and understand while preserving the meaning. Different applications require distinct simplification policies, such as replacing only complex words at the lexical level or rewriting the entire sentence while trading off details for simplicity. However, achieving such policy-driven control remains an open challenge. In this work, we introduce a simple yet powerful approach that leverages Large Language Model-as-a-Judge (LLM-as-a-Judge) to automatically construct policy-aligned training data, completely removing the need for costly human annotation or parallel corpora. Our method enables building simplification systems that adapt to diverse simplification policies. Remarkably, even small-scale open-source LLMs such as Phi-3-mini-3.8B surpass GPT-4o on lexical-oriented simplification, while achieving comparable performance on overall rewriting, as verified by both automatic metrics and human evaluations. The consistent improvements across model families and sizes demonstrate the robustness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06227v1">Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety</a></div>
    <div class="paper-meta">
      📅 2025-12-06
    </div>
    <details class="paper-abstract">
      Real-world indicators are important for improving natural language processing (NLP) tasks such as life events for mental health analysis and risky behaviour for online safety, yet labelling such information in NLP training datasets is often costly and/or difficult given the dynamic nature of such events. This paper compares several LLM-based data enrichment methods and introduces a novel Confidence-Aware Fine-Grained Debate (CFD) framework in which multiple LLM agents simulate human annotators and exchange fine-grained evidence to reach consensus. We describe two new expert-annotated datasets, a mental health Reddit wellbeing dataset and an online safety Facebook sharenting risk dataset. Our CFD framework achieves the most robust data enrichment performance compared to a range of baselines and we show that this type of data enrichment consistently improves downstream tasks. Enriched features incorporated via debate transcripts yield the largest gains, outperforming the non-enriched baseline by 10.1% for the online safety task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05962v1">Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become the de facto standard for tuning LLMs to solve tasks involving reasoning. However, growing evidence shows that models trained in such way often suffer from a significant loss in diversity. We argue that this arises because RL implicitly optimizes the "mode-seeking" or "zero-forcing" Reverse KL to a target distribution causing the model to concentrate mass on certain high-probability regions of the target while neglecting others. In this work, we instead begin from an explicit target distribution, obtained by filtering out incorrect answers while preserving the relative probabilities of correct ones. Starting from a pre-trained LLM, we approximate this target distribution using the $α$-divergence family, which unifies prior approaches and enables direct control of the precision-diversity trade-off by interpolating between mode-seeking and mass-covering divergences. On a Lean theorem-proving benchmark, our method achieves state-of-the-art performance along the coverage-precision Pareto frontier, outperforming all prior methods on the coverage axis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.06211v3">iMotion-LLM: Instruction-Conditioned Trajectory Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      We introduce iMotion-LLM, a large language model (LLM) integrated with trajectory prediction modules for interactive motion generation. Unlike conventional approaches, it generates feasible, safety-aligned trajectories based on textual instructions, enabling adaptable and context-aware driving behavior. It combines an encoder-decoder multimodal trajectory prediction model with a pre-trained LLM fine-tuned using LoRA, projecting scene features into the LLM input space and mapping special tokens to a trajectory decoder for text-based interaction and interpretable driving. To support this framework, we introduce two datasets: 1) InstructWaymo, an extension of the Waymo Open Motion Dataset with direction-based motion instructions, and 2) Open-Vocabulary InstructNuPlan, which features safety-aligned instruction-caption pairs and corresponding safe trajectory scenarios. Our experiments validate that instruction conditioning enables trajectory generation that follows the intended condition. iMotion-LLM demonstrates strong contextual comprehension, achieving 84% average accuracy in direction feasibility detection and 96% average accuracy in safety evaluation of open-vocabulary instructions. This work lays the foundation for text-guided motion generation in autonomous driving, supporting simulated data generation, model interpretability, and robust safety alignment testing for trajectory generation models. Our code, pre-trained model, and datasets are available at: https://vision-cair.github.io/iMotion-LLM/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12229v2">Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05929v1">LLM Harms: A Taxonomy and Discussion</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      This study addresses categories of harm surrounding Large Language Models (LLMs) in the field of artificial intelligence. It addresses five categories of harms addressed before, during, and after development of AI applications: pre-development, direct output, Misuse and Malicious Application, and downstream application. By underscoring the need to define risks of the current landscape to ensure accountability, transparency and navigating bias when adapting LLMs for practical applications. It proposes mitigation strategies and future directions for specific domains and a dynamic auditing system guiding responsible development and integration of LLMs in a standardized proposal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05925v1">To Err Is Human: Systematic Quantification of Errors in Published AI Papers via LLM Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      How many mistakes do published AI papers contain? Peer-reviewed publications form the foundation upon which new research and knowledge are built. Errors that persist in the literature can propagate unnoticed, creating confusion in follow-up studies and complicating reproducibility. The accelerating pace of research and the increasing demands on the peer-review system make such mistakes harder to detect and avoid. To address this, we developed a Paper Correctness Checker based on GPT-5 to systematically identify mistakes in papers previously published at top AI conferences and journals. Our analysis focuses on objective mistakes-e.g., errors in formulas, derivations, calculations, figures, and tables-that have a clearly verifiable ground truth. We intentionally exclude subjective considerations such as novelty, importance, or writing quality. We find that published papers contain a non-negligible number of objective mistakes and that the average number of mistakes per paper has increased over time-from 3.8 in NeurIPS 2021 to 5.9 in NeurIPS 2025 (55.3% increase); from 4.1 in ICLR 2018 to 5.2 in ICLR 2025; and from 5.0 in TMLR 2022/23 to 5.5 in TMLR 2025. Human experts reviewed 316 potential mistakes identified by the AI Checker and confirmed that 263 were actual mistakes, corresponding to a precision of 83.2%. While most identified issues are relatively minor, correcting them would reduce confusion in the literature and strengthen reproducibility. The AI Checker also surfaced potentially more substantive mistakes that could affect the interpretation of results. Moreover, we show that the AI Checker can propose correct fixes for 75.8% of the identified mistakes. Overall, this study highlights the potential of frontier LLMs to detect and correct objective mistakes in published papers, helping to establish a firmer foundation of knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05908v1">Natural Language Summarization Enables Multi-Repository Bug Localization by LLMs in Microservice Architectures</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 Accepted at LLM4Code Workshop, ICSE 2026
    </div>
    <details class="paper-abstract">
      Bug localization in multi-repository microservice architectures is challenging due to the semantic gap between natural language bug reports and code, LLM context limitations, and the need to first identify the correct repository. We propose reframing this as a natural language reasoning task by transforming codebases into hierarchical NL summaries and performing NL-to-NL search instead of cross-modal retrieval. Our approach builds context-aware summaries at file, directory, and repository levels, then uses a two-phase search: first routing bug reports to relevant repositories, then performing top-down localization within those repositories. Evaluated on DNext, an industrial system with 46 repositories and 1.1M lines of code, our method achieves Pass@10 of 0.82 and MRR of 0.50, significantly outperforming retrieval baselines and agentic RAG systems like GitHub Copilot and Cursor. This work demonstrates that engineered natural language representations can be more effective than raw source code for scalable bug localization, providing an interpretable repository -> directory -> file search path, which is vital for building trust in enterprise AI tools by providing essential transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05700v1">Faithfulness metric fusion: Improving the evaluation of LLM trustworthiness across domains</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 9 pages, conference paper
    </div>
    <details class="paper-abstract">
      We present a methodology for improving the accuracy of faithfulness evaluation in Large Language Models (LLMs). The proposed methodology is based on the combination of elementary faithfulness metrics into a combined (fused) metric, for the purpose of improving the faithfulness of LLM outputs. The proposed strategy for metric fusion deploys a tree-based model to identify the importance of each metric, which is driven by the integration of human judgements evaluating the faithfulness of LLM responses. This fused metric is demonstrated to correlate more strongly with human judgements across all tested domains for faithfulness. Improving the ability to evaluate the faithfulness of LLMs, allows for greater confidence to be placed within models, allowing for their implementation in a greater diversity of scenarios. Additionally, we homogenise a collection of datasets across question answering and dialogue-based domains and implement human judgements and LLM responses within this dataset, allowing for the reproduction and trialling of faithfulness evaluation across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05594v1">Ontology Learning with LLMs: A Benchmark Study on Axiom Identification</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 Submitted to Semantic Web Journal, under review
    </div>
    <details class="paper-abstract">
      Ontologies are an important tool for structuring domain knowledge, but their development is a complex task that requires significant modelling and domain expertise. Ontology learning, aimed at automating this process, has seen advancements in the past decade with the improvement of Natural Language Processing techniques, and especially with the recent growth of Large Language Models (LLMs). This paper investigates the challenge of identifying axioms: fundamental ontology components that define logical relations between classes and properties. In this work, we introduce an Ontology Axiom Benchmark OntoAxiom, and systematically test LLMs on that benchmark for axiom identification, evaluating different prompting strategies, ontologies, and axiom types. The benchmark consists of nine medium-sized ontologies with together 17.118 triples, and 2.771 axioms. We focus on subclass, disjoint, subproperty, domain, and range axioms. To evaluate LLM performance, we compare twelve LLMs with three shot settings and two prompting strategies: a Direct approach where we query all axioms at once, versus an Axiom-by-Axiom (AbA) approach, where each prompt queries for one axiom only. Our findings show that the AbA prompting leads to higher F1 scores than the direct approach. However, performance varies across axioms, suggesting that certain axioms are more challenging to identify. The domain also influences performance: the FOAF ontology achieves a score of 0.642 for the subclass axiom, while the music ontology reaches only 0.218. Larger LLMs outperform smaller ones, but smaller models may still be viable for resource-constrained settings. Although performance overall is not high enough to fully automate axiom identification, LLMs can provide valuable candidate axioms to support ontology engineers with the development and refinement of ontologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08916v4">HalluClean: A Unified Framework to Combat Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05542v1">RoBoN: Routed Online Best-of-n for Test-Time Scaling with Multiple LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 20 pages, 3 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Foundations of Reasoning in Language Models
    </div>
    <details class="paper-abstract">
      Best-of-$n$ is a widely used test-time scaling approach for LLM inference. Yet despite evidence that LLMs exhibit complementary strengths across tasks, traditionally best-of-$n$ relies on a single model to generate responses. We propose RoBoN (Routed Online Best-of-$n$), a sequential multi-LLM alternative to the prevailing single-model best-of-$n$. Given a suite of models $\{m_i\}_{i=1}^M$, RoBoN sequentially routes generations one-by-one across models, based on scores computed using a reward model and an agreement signal on the predicted responses. This online routing requires no additional training, keeps compute parity, and works with any plug-in reward model. Across reasoning benchmarks (MATH500, OlympiadBench, MinervaMath, GSM8K, MMLU), RoBoN consistently outperforms standard best-of-$n$ applied to each individual model for larger $n$, with gains of up to 3.4\% in absolute accuracy, and also improves over a uniform multi-model portfolio baseline. Our results indicate that diversity across models can be exploited at inference to improve best-of-$n$ performance over any constituent model alone, providing a simple, training-free path to test-time scaling with multiple LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05537v1">Automated Identification of Incidentalomas Requiring Follow-Up: A Multi-Anatomy Evaluation of LLM-Based and Supervised Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      Objective: To evaluate large language models (LLMs) against supervised baselines for fine-grained, lesion-level detection of incidentalomas requiring follow-up, addressing the limitations of current document-level classification systems. Methods: We utilized a dataset of 400 annotated radiology reports containing 1,623 verified lesion findings. We compared three supervised transformer-based encoders (BioClinicalModernBERT, ModernBERT, Clinical Longformer) against four generative LLM configurations (Llama 3.1-8B, GPT-4o, GPT-OSS-20b). We introduced a novel inference strategy using lesion-tagged inputs and anatomy-aware prompting to ground model reasoning. Performance was evaluated using class-specific F1-scores. Results: The anatomy-informed GPT-OSS-20b model achieved the highest performance, yielding an incidentaloma-positive macro-F1 of 0.79. This surpassed all supervised baselines (maximum macro-F1: 0.70) and closely matched the inter-annotator agreement of 0.76. Explicit anatomical grounding yielded statistically significant performance gains across GPT-based models (p < 0.05), while a majority-vote ensemble of the top systems further improved the macro-F1 to 0.90. Error analysis revealed that anatomy-aware LLMs demonstrated superior contextual reasoning in distinguishing actionable findings from benign lesions. Conclusion: Generative LLMs, when enhanced with structured lesion tagging and anatomical context, significantly outperform traditional supervised encoders and achieve performance comparable to human experts. This approach offers a reliable, interpretable pathway for automated incidental finding surveillance in radiology workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05506v1">When Scaffolding Breaks: Investigating Student Interaction with LLM-Based Writing Support in Real-Time K-12 EFL Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are promising tools for scaffolding students' English writing skills, but their effectiveness in real-time K-12 classrooms remains underexplored. Addressing this gap, our study examines the benefits and limitations of using LLMs as real-time learning support, considering how classroom constraints, such as diverse proficiency levels and limited time, affect their effectiveness. We conducted a deployment study with 157 eighth-grade students in a South Korean middle school English class over six weeks. Our findings reveal that while scaffolding improved students' ability to compose grammatically correct sentences, this step-by-step approach demotivated lower-proficiency students and increased their system reliance. We also observed challenges to classroom dynamics, where extroverted students often dominated the teacher's attention, and the system's assistance made it difficult for teachers to identify struggling students. Based on these findings, we discuss design guidelines for integrating LLMs into real-time writing classes as inclusive educational tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05497v3">Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      Large-scale Mixture of Experts (MoE) Large Language Models (LLMs) have recently become the frontier open weight models, achieving remarkable model capability similar to proprietary ones. But their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit LLM serving systems. To understand the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across four state-of-the-art large-scale MoE models released in 2025 (200B-1000B) using over 24,000 requests spanning diverse workloads. We perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. With our insights, we then demonstrate how to improve wafer-scale GPUs as a case study, and show that minor architectural modifications leveraging the insights achieve substantial performance gains, delivering 5.3x and 3.1x average speedups on DeepSeek V3 and Qwen3, respectively. Our work presents the first comprehensive data-centric analysis of large-scale MoE models and a concrete design study using the learned lessons, with profiling traces and simulation framework already open-sourced with $>$1k downloads. Our traces and results are publicly available at https://huggingface.co/datasets/core12345/MoE_expert_selection_trace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19552v3">iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05485v1">TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05464v1">Dynamic Alignment for Collective Agency: Toward a Scalable Self-Improving Framework for Open-Ended LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 8 pages, 4 figures, to appear in AAAI 2026 AIGOV Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically aligned with human values using preference data or predefined principles such as helpfulness, honesty, and harmlessness. However, as AI systems progress toward Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI), such value systems may become insufficient. In addition, human feedback-based alignment remains resource-intensive and difficult to scale. While AI-feedback-based self-improving alignment methods have been explored as a scalable alternative, they have largely remained constrained to conventional alignment values. In this work, we explore both a more holistic alignment objective and a scalable, self-improving alignment approach. Aiming to transcend conventional alignment norms, we introduce Collective Agency (CA)-a unified and open-ended alignment value that encourages integrated agentic capabilities. We also propose Dynamic Alignment-an alignment framework that enables an LLM to iteratively align itself. Dynamic Alignment comprises two key components: (1) automated training dataset generation with LLMs, and (2) a self-rewarding mechanism, where the policy model evaluates its own output candidates and assigns rewards for GRPO-based learning. Experimental results demonstrate that our approach successfully aligns the model to CA while preserving general NLP capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05461v1">Knowing Your Uncertainty -- On the application of LLM in social sciences</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 49 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly being integrated into computational social science research, yet their blackboxed training and designed stochastic elements in inference pose unique challenges for scientific inquiry. This article argues that applying LLMs to social scientific tasks requires explicit assessment of uncertainty-an expectation long established in both quantitative methodology in the social sciences and machine learning. We introduce a unified framework for evaluating LLM uncertainty along two dimensions: the task type (T), which distinguishes between classification, short-form, and long-form generation, and the validation type (V), which captures the availability of reference data or evaluative criteria. Drawing from both computer science and social science literature, we map existing uncertainty quantification (UQ) methods to this T-V typology and offer practical recommendations for researchers. Our framework provides both a methodological safeguard and a practical guide for integrating LLMs into rigorous social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08475v2">Designing LLM-based Multi-Agent Systems for Software Engineering Tasks: Quality Attributes, Design Patterns and Rationale</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 35 pages, 4 images, 7 tables, Manuscript submitted to a Journal (2025)
    </div>
    <details class="paper-abstract">
      As the complexity of Software Engineering (SE) tasks continues to escalate, Multi-Agent Systems (MASs) have emerged as a focal point of research and practice due to their autonomy and scalability. Furthermore, through leveraging the reasoning and planning capabilities of Large Language Models (LLMs), the application of LLM-based MASs in the field of SE is garnering increasing attention. However, there is no dedicated study that systematically explores the design of LLM-based MASs, including the Quality Attributes (QAs) on which designers mainly focus, the design patterns used by designers, and the rationale guiding the design of LLM-based MASs for SE tasks. To this end, we conducted a study to identify the QAs that LLM-based MASs for SE tasks focus on, the design patterns used in the MASs, and the design rationale for the MASs. We collected 94 papers on LLM-based MASs for SE tasks as the source. Our study shows that: (1) Code Generation is the most common SE task solved by LLM-based MASs among ten identified SE tasks, (2) Functional Suitability is the QA on which designers of LLM-based MASs pay the most attention, (3) Role-Based Cooperation is the design pattern most frequently employed among 16 patterns used to construct LLM-based MASs, and (4) Improving the Quality of Generated Code is the most common rationale behind the design of LLM-based MASs. Based on the study results, we presented the implications for the design of LLM-based MASs to support SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11062v3">Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-05
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11222v2">ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal</a></div>
    <div class="paper-meta">
      📅 2025-12-05
      | 💬 Accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems.
    </details>
</div>
