# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01099v1">Energy-Aware Data-Driven Model Selection in LLM-Orchestrated AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      As modern artificial intelligence (AI) systems become more advanced and capable, they can leverage a wide range of tools and models to perform complex tasks. Today, the task of orchestrating these models is often performed by Large Language Models (LLMs) that rely on qualitative descriptions of models for decision-making. However, the descriptions provided to these LLM-based orchestrators do not reflect true model capabilities and performance characteristics, leading to suboptimal model selection, reduced accuracy, and increased energy costs. In this paper, we conduct an empirical analysis of LLM-based orchestration limitations and propose GUIDE, a new energy-aware model selection framework that accounts for performance-energy trade-offs by incorporating quantitative model performance characteristics in decision-making. Experimental results demonstrate that GUIDE increases accuracy by 0.90%-11.92% across various evaluated tasks, and achieves up to 54% energy efficiency improvement, while reducing orchestrator model selection latency from 4.51 s to 7.2 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04826v2">Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 Accepted at AAAI 2026, Track on AI Alignment
    </div>
    <details class="paper-abstract">
      Large language models require consistent behavioral patterns for safe deployment, yet there are indications of large variability that may lead to an instable expression of personality traits in these models. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25 open-source models (1B-685B parameters) across 2 million+ responses. Using traditional (BFI, SD3) and novel LLM-adapted personality questionnaires, we systematically vary model size, personas, reasoning modes, question order or paraphrasing, and conversation history. Our findings challenge fundamental assumptions: (1) Question reordering alone can introduce large shifts in personality measurements; (2) Scaling provides limited stability gains: even 400B+ models exhibit standard deviations >0.3 on 5-point scales; (3) Interventions expected to stabilize behavior, such as reasoning and inclusion of conversation history, can paradoxically increase variability; (4) Detailed persona instructions produce mixed effects, with misaligned personas showing significantly higher variability than the helpful assistant baseline; (5) The LLM-adapted questionnaires, despite their improved ecological validity, exhibit instability comparable to human-centric versions. This persistent instability across scales and mitigation strategies suggests that current LLMs lack the architectural foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that current alignment strategies may be inadequate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.03194v2">Quantifying Cognitive Bias Induction in LLM-Generated Content</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 21 pages (including references and appendix), 3figures. accepted to AACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are integrated into applications like shopping reviews, summarization, or medical diagnosis support, where their use affects human decisions. We investigate the extent to which LLMs expose users to biased content and demonstrate its effect on human decision-making. We assess five LLM families in summarization and news fact-checking tasks, evaluating the consistency of LLMs with their context and their tendency to hallucinate on a new self-updating dataset. Our findings show that LLMs expose users to content that changes the context's sentiment in 26.42% of cases (framing bias), hallucinate on 60.33% of post-knowledge-cutoff questions, and highlight context from earlier parts of the prompt (primacy bias) in 10.12% of cases, averaged across all tested models. We further find that humans are 32% more likely to purchase the same product after reading a summary of the review generated by an LLM rather than the original review. To address these issues, we evaluate 18 mitigation methods across three LLM families and find the effectiveness of targeted interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04398v2">SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no semantic equivalence or semantic coherence errors compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22171v2">Enhancing Jailbreak Attacks on LLMs via Persona Prompts</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 Workshop on LLM Persona Modeling at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at https://github.com/CjangCjengh/Generic_Persona.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15809v3">Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 AACL 2025 Main Conference (Oral)
    </div>
    <details class="paper-abstract">
      Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17442v3">Confident RAG: Enhancing the Performance of LLMs for Mathematics Question Answering through Multi-Embedding and Confidence Scoring</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold significant promise for mathematics education, yet they often struggle with complex mathematical reasoning. While Retrieval-Augmented Generation (RAG) mitigates these issues by grounding LLMs in external knowledge, its effectiveness remains unstable, heavily dependent on the choice of a single embedding model. Moving beyond static RAG workflows, we draw on agentic workflow patterns, a paradigm that introduces structured task decomposition and collaboration to enhance system performance. We propose and examine two novel approaches that combine the benefits of multiple embedding models. While our Mixture-Embedding RAG approach (fusing retrieved documents) shows limited gains, our Confident RAG method (generating multiple answers and selecting the one with the highest confidence score) demonstrates significant improvement. Experimental results show that Confident RAG achieved average accuracy improvements of approximately 10% over vanilla LLMs and 5% over vanilla RAG. The consistent results across different LLMs and embedding models indicate that Confident RAG is an efficient plug-and-play solution for trustworthy mathematical AI assistants. Finally, we discuss how this work lays the groundwork for deploying Agentic RAG systems in educational settings, where autonomous planning and iterative refinement can be built upon our robust retrieval foundation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10493v2">The Hidden DNA of LLM-Generated JavaScript: Structural Patterns Enable High-Accuracy Authorship Attribution</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      In this paper, we present the first large-scale study exploring whether JavaScript code generated by Large Language Models (LLMs) can reveal which model produced it, enabling reliable authorship attribution and model fingerprinting. With the rapid rise of AI-generated code, attribution is playing a critical role in detecting vulnerabilities, flagging malicious content, and ensuring accountability. While AI-vs-human detection usually treats AI as a single category we show that individual LLMs leave unique stylistic signatures, even among models belonging to the same family or parameter size. To this end, we introduce LLM-NodeJS, a dataset of 50,000 Node.js back-end programs from 20 large language models. Each has four transformed variants, yielding 250,000 unique JavaScript samples and two additional representations (JSIR and AST) for diverse research applications. Using this dataset, we benchmark traditional machine learning classifiers against fine-tuned Transformer encoders and introduce CodeT5-JSA, a custom architecture derived from the 770M-parameter CodeT5 model with its decoder removed and a modified classification head. It achieves 95.8% accuracy on five-class attribution, 94.6% on ten-class, and 88.5% on twenty-class tasks, surpassing other tested models such as BERT, CodeBERT, and Longformer. We demonstrate that classifiers capture deeper stylistic regularities in program dataflow and structure, rather than relying on surface-level features. As a result, attribution remains effective even after mangling, comment removal, and heavy code transformations. To support open science and reproducibility, we release the LLM-NodeJS dataset, Google Colab training scripts, and all related materials on GitHub: https://github.com/LLM-NodeJS-dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00862v1">HBLLM: Wavelet-Enhanced High-Fidelity 1-Bit Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      We introduce HBLLM, a wavelet-enhanced high-fidelity $1$-bit post-training quantization method for Large Language Models (LLMs). By leveraging Haar wavelet transforms to enhance expressive capacity through frequency decomposition, HBLLM significantly improves quantization fidelity while maintaining minimal overhead. This approach features two innovative structure-aware grouping strategies: (1) frequency-aware multi-parameter intra-row grouping and (2) $\ell_2$-norm-based saliency-driven column selection. For non-salient weights, a shared mean is employed across quantization groups within each frequency band to optimize storage efficiency. Experiments conducted on the OPT and LLaMA models demonstrate that HBLLM achieves state-of-the-art performance in $1$-bit quantization, attaining a perplexity of $6.71$ on LLaMA$2$-$13$B with an average weight storage of only $1.08$ bits. Code available at: https://github.com/Yeyke/HBLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22338v2">Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 15 pages, 7 tables, accepted at the International Joint Conference on Learning & Reasoning (IJCLR 2025)
    </div>
    <details class="paper-abstract">
      Automating the translation of natural language to first-order logic (FOL) is crucial for knowledge representation and formal methods, yet remains challenging. We present a systematic evaluation of fine-tuned LLMs for this task, comparing architectures (encoder-decoder vs. decoder-only) and training strategies. Using the MALLS and Willow datasets, we explore techniques like vocabulary extension, predicate conditioning, and multilingual training, introducing metrics for exact match, logical equivalence, and predicate alignment. Our fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and even the DeepSeek-R1-0528 model with CoT reasoning ability as well as symbolic systems like ccg2lambda. Key findings show: (1) predicate availability boosts performance by 15-20%, (2) T5 models surpass larger decoder-only LLMs, and (3) models generalize to unseen logical arguments (FOLIO dataset) without specific training. While structural logic translation proves robust, predicate extraction emerges as the main bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00831v1">ReJump: A Tree-Jump Representation for Analyzing and Improving LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      Large Reasoning Models (LRMs) are Large Language Models (LLMs) explicitly trained to generate long-form Chain-of-Thoughts (CoTs), achieving impressive success on challenging tasks like math and programming. However, their underlying reasoning "algorithms" remain poorly understood. To investigate this, we propose ReJump, which represents a reasoning trace as a visitation order over nodes in a tree of intermediate problem-solving steps. Transitions between nodes, which we term jumps, include adjacent moves that capture behaviors such as calculation, and non-adjacent moves that capture behaviors such as backtracking and verification. ReJump enables analyzing LLM reasoning with diverse metrics that quantify exploration, exploitation, overthinking, forgetting, and verification. Using our proposed LLM agent to extract reasoning traces into ReJump format, we evaluate state-of-the-art LRMs on two tasks and find that models with similar accuracy can exhibit distinct reasoning behaviors, while different tasks favor different reasoning styles (e.g., varying balance between exploration and exploitation). To further understand how learning strategies shape reasoning, we use ReJump to compare distilled LRMs with their teachers, CoT-prompted LLMs with LRMs, and to examine how the number of reasoning examples and reinforcement learning affect reasoning behavior. Finally, we show that ReJump can improve reasoning quality at test time through strategies such as ReJump-guided Best-of-N selection and prompt selection. Our code is publicly available at https://github.com/UW-Madison-Lee-Lab/ReJump.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.20015v3">Efficient LLM-Jailbreaking via Multimodal-LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.18583v4">PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 Submitted for possible publication
    </div>
    <details class="paper-abstract">
      The autoregressive nature of large language models (LLMs) fundamentally limits inference speed, as each forward pass generates only a single token and is often bottlenecked by memory bandwidth. Speculative decoding has emerged as a promising solution, adopting a draft-then-verify strategy to accelerate token generation. While the EAGLE series achieves strong acceleration, its requirement of training a separate draft head for each target model introduces substantial adaptation costs. In this work, we propose \textbf{PARD (PARallel Draft)}, a novel speculative decoding method featuring \textit{target-independence} and \textit{parallel token prediction}. Specifically, PARD enables a single draft model to be applied across an entire family of target models without requiring separate training for each variant, thereby minimizing adaptation costs. Meanwhile, PARD substantially accelerates inference by predicting multiple future tokens within a single forward pass of the draft phase. To further reduce the training adaptation cost of PARD, we propose a COnditional Drop-token (COD) mechanism based on the integrity of prefix key-value states, enabling autoregressive draft models to be adapted into parallel draft models at low-cost. Our experiments show that the proposed COD method improves draft model training efficiency by \textbf{3$\times$} compared with traditional masked prediction training. On the \texttt{vLLM} inference framework, PARD achieves up to \textbf{3.67$\times$} speedup on LLaMA3.1-8B, reaching \textbf{264.88} tokens per second, which is \textbf{1.15$\times$} faster than EAGLE-3. Our code is available at https://github.com/AMD-AIG-AIMA/PARD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00722v1">SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 Accepted by ASPLOS 2026
    </div>
    <details class="paper-abstract">
      In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in information focus between the distilled language model(DLM) and the original LLM from the perspective of information theory, and thus propose a novel paradigm that leverages a DLM as the retrieval algorithm. Based on the insight, we present SpeContext, an algorithm and system co-design for long-context reasoning. (1) At the algorithm level, SpeContext proposes lightweight retrieval head based on the head-level attention weights of DLM, achieving > 90% parameters reduction by pruning the redundancy. (2) At the system level, SpeContext designs an asynchronous prefetch dataflow via the elastic loading strategy, effectively overlapping KV cache retrieval with the LLM computation. (3) At the compilation level, SpeContext constructs the theoretical memory model and implements an adaptive memory management system to achieve acceleration by maximizing GPU memory utilization. We deploy and evaluate SpeContext in two resourceconstrained environments, cloud and edge. Extensive experiments show that, compared with the Huggingface framework, SpeContext achieves up to 24.89x throughput improvement in cloud and 10.06x speedup in edge with negligible accuracy loss, pushing the Pareto frontier of accuracy and throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00719v1">SIMPLE: Disaggregating Sampling from GPU Inference into a Decision Plane for Faster Distributed LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-30
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) scale out with tensor parallelism (TP) and pipeline parallelism (PP) and production stacks have aggressively optimized the data plane (attention/GEMM and KV cache), sampling, the decision plane that turns logits into tokens, becomes a new bottleneck. This creates a structural holdout: sampling neither expands with TP nor balances across PP stages, so its share of iteration time grows as GPUs get faster and it caps pipeline frequency at the last stage. We present SIMPLE, a stage-agnostic, sequence-parallel, overlappable decision plane that disaggregates sampling into a CPU-side service and shrinks its runtime footprint back to a minor, hidden role. SIMPLE combines: (1) sequence-parallel sampling, which shards work along the batch dimension and removes vocabulary-axis collectives; (2) a CPU-based algorithm with column-wise penalties and truncation-first filtering to realize single-pass, linear-time kernels; and (3) speculative hot-vocab sampling (SHVS), which samples on a small hot set with rejection-correctness and uses a simple sizing model to choose the hot-vocab size that maximizes throughput. In evaluation, SIMPLE improves end-to-end throughput by up to 96% and reduces P95 latency by 20-65%. Crucially, SIMPLE requires no user-side code changes and composes with existing data-plane optimizations, unlocking scaling benefits that compound with future GPU generations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02080v1">The 4/$δ$ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantee</a></div>
    <div class="paper-meta">
      📅 2025-11-30
      | 💬 32 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The idea of using Formal Verification tools with large language models (LLMs) has enabled scaling software verification beyond manual workflows. However, current methods remain unreliable. Without a solid theoretical footing, the refinement process can wander; sometimes it settles, sometimes it loops back, and sometimes it breaks away from any stable trajectory. This work bridges this critical gap by developing an LLM-Verifier Convergence Theorem, providing the first formal framework with provable guarantees for termination and convergence. We model the interaction between the LLM and the verifier as a discrete-time Markov Chain, with state transitions determined by a key parameter: the error-reduction probability ($δ$). The procedure reaching the Verified state almost surely demonstrates that the program terminates for any $δ> 0$, with an expected iteration count bounded by $\mathbb{E}[n] \leq 4/δ$. We then stress-tested this prediction in an extensive empirical campaign comprising more than 90,000 trials. The empirical results match the theory with striking consistency. Every single run reached verification, and the convergence factor clustered tightly around $C_f\approx$ 1.0. Consequently, the bound mirrors the system's actual behavior. The evidence is sufficiently robust to support dividing the workflow into three distinct operating zones: marginal, practical, and high-performance. Consequently, we establish the design thresholds with absolute confidence. Together, the theoretical guarantee and the experimental evidence provide a clearer architectural foundation for LLM-assisted verification. Heuristic tuning no longer has to be carried out by the system. Engineers gain a framework that supports predictable resource planning and performance budgeting, precisely what is needed before deploying these pipelines into safety-critical software environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00617v1">ART: Adaptive Response Tuning Framework -- A Multi-Agent Tournament-Based Approach to LLM Response Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 8 pages, 5 figures, 5 tables. Conference-style paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, single-model responses often exhibit inconsistencies, hallucinations, and varying quality across different query domains. This paper presents ART (Adaptive Response Tuning), a novel framework that employs tournament-style ELO ranking and multi-agent reasoning to systematically optimize LLM outputs. By enabling multiple LLM agents to compete, critique, and collaborate through structured tournament workflows, ART produces consensus responses that outperform individual model outputs. Our framework introduces configurable tournament parameters, dynamic agent selection, and multiple consensus fusion strategies. Experimental evaluations demonstrate significant improvements in response accuracy, coherence, and reliability compared to baseline single-model approaches. The ART framework provides a scalable, production-ready solution for applications requiring high-quality, vetted LLM responses, achieving an 8.4% improvement in overall quality metrics and R22 values exceeding 0.96 in ELO rating convergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00591v1">TrojanLoC: LLM-based Framework for RTL Trojan Localization</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      Hardware Trojans (HT s) are a persistent threat to integrated circuits, especially when inserted at the register-transfer level (RTL). Existing methods typically first convert the design into a graph, such as a gate-level netlist or an RTL-derived dataflow graph (DFG), and then use a graph neural network (GNN ) to obtain an embedding of that graph, which (i) loses compact RTL semantics, (ii) relies on shallow GNNs with limited receptive field, and (iii) is largely restricted to coarse, module-level binary HT detection. We propose TrojanLoC, an LLM-based framework for RTL-level HT localization. We use an RTL-finetuned LLM to derive module-level and line-level embeddings directly from RTL code, capturing both global design context and local semantics. Next, we train task-specific classifiers on these embeddings to perform module-level Trojan detection, type prediction, and fine-grained line-level localization. We also introduce TrojanInS, a large synthetic dataset of RTL designs with systematically injected Trojans from four effect-based categories, each accompanied by precise line-level annotations. Our experiments show that TrojanLoC achieves strong module-level performance, reaching 0.99 F1-score for Trojan detection, up to 0.68 higher than baseline, and 0.84 macro-F1 for Trojan-type classification. At the line level, TrojanLoc further achieves up to 0.93 macro-F1, enabling fine-grained localization of Trojan-relevant RTL lines
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00556v1">Bias Testing and Mitigation in Black Box LLMs using Metamorphic Relations</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      The widespread deployment of Large Language Models (LLMs) has intensified concerns about subtle social biases embedded in their outputs. Existing guardrails often fail when faced with indirect or contextually complex bias-inducing prompts. To address these limitations, we propose a unified framework for both systematic bias evaluation and targeted mitigation. Our approach introduces six novel Metamorphic Relations (MRs) that, based on metamorphic testing principles, transform direct bias-inducing inputs into semantically equivalent yet adversarially challenging variants. These transformations enable an automated method for exposing hidden model biases: when an LLM responds inconsistently or unfairly across MR-generated variants, the underlying bias becomes detectable. We further show that the same MRs can be used to generate diverse bias-inducing samples for fine-tuning, directly linking the testing process to mitigation. Using six state-of-the-art LLMs - spanning open-source and proprietary models - and a representative subset of 385 questions from the 8,978-item BiasAsker benchmark covering seven protected groups, our MRs reveal up to 14% more hidden biases compared to existing tools. Moreover, fine-tuning with both original and MR-mutated samples significantly enhances bias resiliency, increasing safe response rates from 54.7% to over 88.9% across models. These results highlight metamorphic relations as a practical mechanism for improving fairness in conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.02429v4">IoT-LLM: a framework for enhancing Large Language Model reasoning from real-world sensor data</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 33 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large Language Models excel in textual tasks but often struggle with physical-world reasoning tasks. Inspired by human cognition, where perception is fundamental to reasoning, we explore augmenting LLMs with enhanced perception abilities using Internet of Things (IoT) data and pertinent knowledge. In this work, we systematically study LLMs' capability to address IoT-sensory tasks by augmenting their perception and knowledge base, and then propose a unified framework, IoT-LLM, to enhance such capability. In IoT-LLM, we customize three steps: preprocessing IoT data into suitable formats, expanding LLMs knowledge via IoT-oriented retrieval-augmented generation and activating LLMs commonsense knowledge through chain-of-thought prompting. We design a benchmark comprising five real-world tasks with varying data types and reasoning complexities to evaluate the performance of IoT-LLM. Experimental results reveal that IoT-LLM significantly improves the performance of IoT-sensory task reasoning of LLMs, with models like GPT-4o-mini showing a 49.4% average improvement over previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.11791v2">LEXA: Legal Case Retrieval via Graph Contrastive Learning with Contextualised LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 arXiv admin note: substantial text overlap with arXiv:2312.11229
    </div>
    <details class="paper-abstract">
      Legal case retrieval (LCR) is a specialised information retrieval task aimed at identifying relevant cases given a query case. LCR holds pivotal significance in facilitating legal practitioners to locate legal precedents. Existing LCR methods predominantly rely on traditional lexical models or language models; however, they typically overlook the domain-specific structural information embedded in legal documents. Our previous work CaseGNN successfully harnesses text-attributed graphs and graph neural networks to incorporate structural legal information. Nonetheless, three key challenges remain in enhancing the representational capacity of CaseGNN: (1) The under-utilisation of rich edge information in text-attributed case graph (TACG). (2) The insufficiency of training signals for graph contrastive learning. (3) The lack of contextualised legal information in node and edge features. In this paper, the LEXA model, an extension of CaseGNN, is proposed to overcome these limitations by jointly leveraging rich edge information, enhanced training signals, and contextualised embeddings derived from large language models (LLMs). Specifically, an edge-updated graph attention layer (EUGAT) is proposed to comprehensively update node and edge features during graph modelling, resulting in a full utilisation of structural information of legal cases. Moreover, LEXA incorporates a novel graph contrastive learning objective with graph augmentation to provide additional training signals, thereby strengthening the model's legal comprehension capabilities. What's more, LLMs are employed to generate node and edge features for TACG. Extensive experiments on two benchmark datasets demonstrate that LEXA not only significantly improves CaseGNN but also achieves supreme performance compared to state-of-the-art LCR methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00461v1">Whose Personae? Synthetic Persona Experiments in LLM Research and Pathways to Transparency</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 Published at AAAI/ACM AIES 2025. Presented at NeurIPS 2025 Workshop Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Synthetic personae experiments have become a prominent method in Large Language Model alignment research, yet the representativeness and ecological validity of these personae vary considerably between studies. Through a review of 63 peer-reviewed studies published between 2023 and 2025 in leading NLP and AI venues, we reveal a critical gap: task and population of interest are often underspecified in persona-based experiments, despite personalization being fundamentally dependent on these criteria. Our analysis shows substantial differences in user representation, with most studies focusing on limited sociodemographic attributes and only 35% discussing the representativeness of their LLM personae. Based on our findings, we introduce a persona transparency checklist that emphasizes representative sampling, explicit grounding in empirical data, and enhanced ecological validity. Our work provides both a comprehensive assessment of current practices and practical guidelines to improve the rigor and ecological validity of persona-based evaluations in language model alignment research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22544v3">HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 25 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00417v1">CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00403v1">SelfAI: Building a Self-Training AI System with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      Recent work on autonomous scientific discovery has leveraged LLM-based agents to integrate problem specification, experiment planning, and execution into end-to-end systems. However, these frameworks are often confined to narrow application domains, offer limited real-time interaction with researchers, and lack principled mechanisms for determining when to halt exploration, resulting in inefficiencies, reproducibility challenges, and under-utilized human expertise. To address these gaps, we propose \textit{SelfAI}, a general multi-agent platform that combines a User Agent for translating high-level research objectives into standardized experimental configurations, a Cognitive Agent powered by LLMs with optimal stopping criteria to iteratively refine hyperparameter searches, and an Experiment Manager responsible for orchestrating parallel, fault-tolerant training workflows across heterogeneous hardware while maintaining a structured knowledge base for continuous feedback. We further introduce two novel evaluation metrics, Score and $\text{AUP}_D$, to quantify discovery efficiency and search diversity. Across regression, NLP, computer vision, scientific computing, medical imaging, and drug discovery benchmarks, SelfAI consistently achieves strong performance and reduces redundant trials compared to classical Bayesian optimization and LLM-based baselines, while enabling seamless interaction with human researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00371v1">Evaluating LLMs in Open-Source Games</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models' (LLMs) programming capabilities enable their participation in open-source games: a game-theoretic setting in which players submit computer programs in lieu of actions. These programs offer numerous advantages, including interpretability, inter-agent transparency, and formal verifiability; additionally, they enable program equilibria, solutions that leverage the transparency of code and are inaccessible within normal-form settings. We evaluate the capabilities of leading open- and closed-weight LLMs to predict and classify program strategies and evaluate features of the approximate program equilibria reached by LLM agents in dyadic and evolutionary settings. We identify the emergence of payoff-maximizing, cooperative, and deceptive strategies, characterize the adaptation of mechanisms within these programs over repeated open-source games, and analyze their comparative evolutionary fitness. We find that open-source games serve as a viable environment to study and steer the emergence of cooperative strategy in multi-agent dilemmas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00335v1">Efficient Kernel Mapping and Comprehensive System Evaluation of LLM Acceleration on a CGLA</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 This paper is published at IEEE Access
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demand substantial computational resources, resulting in high energy consumption on GPUs. To address this challenge, we focus on Coarse-Grained Reconfigurable Arrays (CGRAs) as an effective alternative that provides a trade-off between energy efficiency and programmability. This paper presents the first comprehensive, end-to-end evaluation of a non-AI-specialized Coarse-Grained Linear Array (CGLA) accelerator for the state-of-the-art Qwen LLM family. The architecture has a general-purpose, task-agnostic design, yet its flexible instruction set allows for domain-specific adaptations. This flexibility enables the architecture to achieve high efficiency for sustainable LLM inference. We assess the performance of our architecture on an FPGA prototype using the widely adopted llama.cpp framework. We then project its potential as a 28nm ASIC and compare it against a high-performance GPU (NVIDIA RTX 4090) and an edge AI device (NVIDIA Jetson AGX Orin). While GPUs exhibit lower latency, our non-AI-specific accelerator achieves higher energy efficiency, improving the Power-Delay Product (PDP) by up to 44.4x and 13.6x compared with the RTX 4090 and Jetson, respectively. Similarly, it reduces the Energy-Delay Product (EDP) by up to 11.5x compared to the high-performance GPU, demonstrating a favorable performance-energy trade-off. Critically, our system-level analysis identifies host-accelerator data transfer as the primary performance bottleneck, a factor often overlooked in kernel-level studies. These findings provide design guidance for next-generation LLM accelerators. This work validates CGRAs as a suitable platform for LLM inference in power-constrained environments, without being confined to specific algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00333v1">IndicParam: Benchmark to evaluate LLMs on low-resource Indic Languages</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      While large language models excel on high-resource multilingual tasks, low- and extremely low-resource Indic languages remain severely under-evaluated. We present IndicParam, a human-curated benchmark of over 13,000 multiple-choice questions covering 11 such languages (Nepali, Gujarati, Marathi, Odia as low-resource; Dogri, Maithili, Rajasthani, Sanskrit, Bodo, Santali, Konkani as extremely low-resource) plus Sanskrit-English code-mixed set. We evaluated 19 LLMs, both proprietary and open-weights, which reveals that even the top-performing GPT-5 reaches only 45.0% average accuracy, followed by DeepSeek-3.2 (43.1) and Claude-4.5 (42.7). We additionally label each question as knowledge-oriented or purely linguistic to discriminate factual recall from grammatical proficiency. Further, we assess the ability of LLMs to handle diverse question formats-such as list-based matching, assertion-reason pairs, and sequence ordering-alongside conventional multiple-choice questions. IndicParam provides insights into limitations of cross-lingual transfer and establishes a challenging benchmark for Indic languages. The dataset is available at https://huggingface.co/datasets/bharatgenai/IndicParam. Scripts to run benchmark are present at https://github.com/ayushbits/IndicParam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12899v4">A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 13 pages, 6 figure, 8 tables, camera-ready version
    </div>
    <details class="paper-abstract">
      Language Models (LMs) are widely used in software engineering for code generation, but they may produce erroneous code. Rather than repairing outputs, a more thorough remedy is to address underlying model failures. LM repair offers a lightweight solution: it requires minimal data, lowers computational cost, and limits side effects. Unlike full retraining, LM repair focuses on applying tailored updates to targeted neurons, making it suitable for limited resources, high-performance demands, or strict safety requirements. In this paper, we propose Semantic Targeting for Analytical Repair (STAR), a novel semantic-based optimization method for repairing LLMs. STAR realizes the main operations of repairing LMs in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (MINT) and standard optimization methods (SGD), STAR integrates their strengths while mitigating their limitations. By reformulating LM repair as an optimization process, STAR may solve multiple failures together, significantly improving the usefulness. Evaluated on coding tasks using popular code LMs, STAR demonstrates superior effectiveness compared with the state-of-the-art. Besides, STAR exhibits better efficiency. In terms of side effects, namely the balance between generalization and specificity, STAR outperforms prior work by a significant margin. Additionally, we conducted assessments on the overfitting risk of LM repair as well as the cumulative impact. Further, we analyzed the differences with pipeline-based methods and explained the reason why STAR is better and how it mitigated the common limitations of LM repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00319v1">RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 23 pages, 14 figures. Model is available at https://huggingface.co/Freakz3z/Qwen-JSON
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language generation and reasoning. However, their integration into automated software ecosystems is often hindered by the "Structure Gap" - the inherent tension between the probabilistic nature of token generation and the deterministic requirements of structured data formats (e.g., JSON, XML). Traditional Supervised Fine-Tuning (SFT) often fails to enforce strict syntactic constraints, leading to "hallucinated" keys or malformed structures, while constrained decoding methods impose significant inference latency. In this paper, we propose a lightweight, efficient Reinforcement Learning (RL) framework to bridge this gap. We introduce a novel Multi-dimensional Reward Function that decomposes the structured output task into a hierarchy of constraints: structural integrity, format correctness, content accuracy, and validity. Leveraging Gradient Regularized Policy Optimization (GRPO), we enable the model to internalize these constraints without the need for a separate critic network, reducing peak VRAM usage by 40% compared to PPO. We validate our approach on multiple tasks, including complex recipe generation and structured math reasoning (GSM8K-JSON). Experimental results demonstrate that our method achieves 89.7% structural accuracy and 92.1% JSON validity, significantly outperforming both zero-shot baselines (e.g., GPT-3.5) and SFT on larger models like LLaMA-3-8B. Furthermore, we provide a detailed analysis of training dynamics, revealing a distinct self-paced curriculum where the model sequentially acquires syntactic proficiency before semantic accuracy. Our model is publicly available at https://huggingface.co/Freakz3z/Qwen-JSON.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00313v1">Evolving Paradigms in Task-Based Search and Learning: A Comparative Analysis of Traditional Search Engine with LLM-Enhanced Conversational Search System</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly reshaping information retrieval by enabling interactive, generative, and inference-driven search. While traditional keyword-based search remains central to web and academic information access, it often struggles to support multi-step reasoning and exploratory learning tasks. LLM-powered search interfaces, such as ChatGPT and Claude, introduce new capabilities that may influence how users formulate queries, navigate information, and construct knowledge. However, empirical understanding of these effects is still limited. This study compares search behavior and learning outcomes in two environments: a standard search engine and an LLM-powered search system. We investigate (1) how search strategies, query formulation, and evaluation behaviors differ across systems, and (2) how LLM use affects comprehension, knowledge integration, and critical thinking during search-based learning tasks. Findings offer insight into how generative AI shapes information-seeking processes and contribute to ongoing discussions in information retrieval, human-AI interaction, and technology-supported learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00293v1">FiCoTS: Fine-to-Coarse LLM-Enhanced Hierarchical Cross-Modality Interaction for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      Time series forecasting is central to data analysis and web technologies. The recent success of Large Language Models (LLMs) offers significant potential for this field, especially from the cross-modality aspect. Most methods adopt an LLM-as-Predictor paradigm, using LLM as the forecasting backbone and designing modality alignment mechanisms to enable LLM to understand time series data. However, the semantic information in the two modalities of time series and text differs significantly, making it challenging for LLM to fully understand time series data. To mitigate this challenge, our work follows an LLM-as-Enhancer paradigm to fully utilize the advantage of LLM in text understanding, where LLM is only used to encode text modality to complement time series modality. Based on this paradigm, we propose FiCoTS, an LLM-enhanced fine-to-coarse framework for multimodal time series forecasting. Specifically, the framework facilitates progressive cross-modality interaction by three levels in a fine-to-coarse scheme: First, in the token-level modality alignment module, a dynamic heterogeneous graph is constructed to filter noise and align time series patches with text tokens; Second, in the feature-level modality interaction module, a global cross-attention mechanism is introduced to enable each time series variable to connect with relevant textual contexts; Third, in the decision-level modality fusion module, we design a gated network to adaptively fuse the results of the two modalities for robust predictions. These three modules work synergistically to let the two modalities interact comprehensively across three semantic levels, enabling textual information to effectively support temporal prediction. Extensive experiments on seven real-world benchmarks demonstrate that our model achieves state-of-the-art performance. The codes will be released publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01558v4">Predicting the Performance of Black-box LLMs through Follow-up Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-29
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reliably predicting the behavior of language models -- such as whether their outputs are correct or have been adversarially manipulated -- is a fundamentally challenging task. This is often made even more difficult as frontier language models are offered only through closed-source APIs, providing only black-box access. In this paper, we predict the behavior of black-box language models by asking follow-up questions and taking the probabilities of responses \emph{as} representations to train reliable predictors. We first demonstrate that training a linear model on these responses reliably and accurately predicts model correctness on question-answering and reasoning benchmarks. Surprisingly, this can \textit{even outperform white-box linear predictors} that operate over model internals or activations. Furthermore, we demonstrate that these follow-up question responses can reliably distinguish between a clean version of an LLM and one that has been adversarially influenced via a system prompt to answer questions incorrectly or to introduce bugs into generated code. Finally, we show that they can also be used to differentiate between black-box LLMs, enabling the detection of misrepresented models provided through an API. Overall, our work shows promise in monitoring black-box language model behavior, supporting their deployment in larger, autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03086v1">Beyond Code Pairs: Dialogue-Based Data Generation for LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-11-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in code translation, yet their performance deteriorates in low-resource programming domains such as Fortran and emerging frameworks like CUDA, where high-quality parallel data are scarce. We present an automated dataset generation pipeline featuring a dual-LLM Questioner-Solver design that incorporates external knowledge from compilers and runtime feedback. Beyond traditional source-target code pair datasets, our approach additionally generates (1) verified translations with unit tests for assessing functional consistency, and (2) multi-turn dialogues that capture the reasoning process behind translation refinement. Applied to Fortran -> C++ and C++ -> CUDA, the pipeline yields 3.64k and 3.93k dialogues, respectively. Fine-tuning on this data yields dramatic improvements in functional correctness, boosting unit test success rates by over 56% on the challenging C++-to-CUDA task. We show this data enables a 7B open-weight model to significantly outperform larger proprietary systems on key metrics like compilation success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23476v1">Thinking by Doing: Building Efficient World Model Reasoning in LLMs via Multi-turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 17 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Developing robust world model reasoning is crucial for large language model (LLM) agents to plan and interact in complex environments. While multi-turn interaction offers a superior understanding of environmental dynamics via authentic feedback, current approaches often impose a rigid reasoning process, which constrains the model's active learning, ultimately hindering efficient world model reasoning. To address these issues, we explore world-model internalization through efficient interaction and active reasoning (WMAct), which liberates the model from structured reasoning, allowing the model to shape thinking directly through its doing, and achieves effective and efficient world model reasoning with two key mechanisms: (1) a reward rescaling mechanism adjusting outcome reward based on action efficacy to incentivize redundancy reduction and purposeful interaction; (2) an interaction frequency annealing strategy to progressively reduce the maximum allowed interaction turns, which compels the model to condense its learning and internalize environmental dynamics rather than over-relying on environmental cues. Our experiments on Sokoban, Maze, and Taxi show that WMAct yields effective world model reasoning capable of resolving tasks in a single turn that previously required multiple interactions and fosters strong transferability to complex environments, improving performance on a suite of reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23408v1">Evaluating LLMs for One-Shot Patching of Real and Artificial Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Pre-print - Extended version of the poster paper accepted at the 41st ACM/SIGAPP Symposium on Applied Computing (SAC) Smarter Engineering-Building AI and Building with AI (SEAI) 2026
    </div>
    <details class="paper-abstract">
      Automated vulnerability patching is crucial for software security, and recent advancements in Large Language Models (LLMs) present promising capabilities for automating this task. However, existing research has primarily assessed LLMs using publicly disclosed vulnerabilities, leaving their effectiveness on related artificial vulnerabilities largely unexplored. In this study, we empirically evaluate the patching effectiveness and complementarity of several prominent LLMs, such as OpenAI's GPT variants, LLaMA, DeepSeek, and Mistral models, using both real and artificial vulnerabilities. Our evaluation employs Proof-of-Vulnerability (PoV) test execution to concretely assess whether LLM-generated source code successfully patches vulnerabilities. Our results reveal that LLMs patch real vulnerabilities more effectively compared to artificial ones. Additionally, our analysis reveals significant variability across LLMs in terms of overlapping (multiple LLMs patching the same vulnerabilities) and complementarity (vulnerabilities patched exclusively by a single LLM), emphasizing the importance of selecting appropriate LLMs for effective vulnerability patching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23387v1">Hierarchical AI-Meteorologist: LLM-Agent System for Multi-Scale and Explainable Weather Forecast Reporting</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present the Hierarchical AI-Meteorologist, an LLM-agent system that generates explainable weather reports using a hierarchical forecast reasoning and weather keyword generation. Unlike standard approaches that treat forecasts as flat time series, our framework performs multi-scale reasoning across hourly, 6-hour, and daily aggregations to capture both short-term dynamics and long-term trends. Its core reasoning agent converts structured meteorological inputs into coherent narratives while simultaneously extracting a few keywords effectively summarizing the dominant meteorological events. These keywords serve as semantic anchors for validating consistency, temporal coherence and factual alignment of the generated reports. Using OpenWeather and Meteostat data, we demonstrate that hierarchical context and keyword-based validation substantially improve interpretability and robustness of LLM-generated weather narratives, offering a reproducible framework for semantic evaluation of automated meteorological reporting and advancing agent-based scientific reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23379v1">AugGen: Augmenting Task-Based Learning in Professional Creative Software with LLM-Generated Scaffolded UIs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 arXiv admin note: substantial text overlap with arXiv:2505.12101
    </div>
    <details class="paper-abstract">
      Professional creative software often presents steep learning curves due to complex interfaces, lack of structured task-aware guidance, and unfamiliar domain terminology. To address these challenges and augment user learning experience, we introduce AugGen, a method for generating scaffolded user interfaces that simplify interface complexity and support task-based learning. With the user's task, our method surfaces task-relevant tools to reduce distracting features, organizes the tools around task workflow stages to offer execution guidance, connects tools with domain concepts to foster learning engagement, and progressively discloses advanced features to manage learning progress. To evaluate the method, we used our LLM-assisted pipeline to generate two task-specific scaffolded UIs and deployed them in Blender, our professional 3D modeling testbed. We invited both beginner (N=32) and expert (N=8) users to evaluate our implemented interfaces. Results show that the scaffolded interfaces significantly reduced user-perceived task load, enhanced task performance via embedded guidance, and augmented concept learning during task execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23312v1">Do LLM-judges Align with Human Relevance in Cranfield-style Recommender Evaluation?</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Evaluating recommender systems remains a long-standing challenge, as offline methods based on historical user interactions and train-test splits often yield unstable and inconsistent results due to exposure bias, popularity bias, sampled evaluations, and missing-not-at-random patterns. In contrast, textual document retrieval benefits from robust, standardized evaluation via Cranfield-style test collections, which combine pooled relevance judgments with controlled setups. While recent work shows that adapting this methodology to recommender systems is feasible, constructing such collections remains costly due to the need for manual relevance judgments, thus limiting scalability. This paper investigates whether Large Language Models (LLMs) can serve as reliable automatic judges to address these scalability challenges. Using the ML-32M-ext Cranfield-style movie recommendation collection, we first examine the limitations of existing evaluation methodologies. Then we explore the alignment and the recommender systems ranking agreement between the LLM-judge and human provided relevance labels. We find that incorporating richer item metadata and longer user histories improves alignment, and that LLM-judge yields high agreement with human-based rankings (Kendall's tau = 0.87). Finally, an industrial case study in the podcast recommendation domain demonstrates the practical value of LLM-judge for model selection. Overall, our results show that LLM-judge is a viable and scalable approach for evaluating recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23271v1">Behavior-Equivalent Token: Single-Token Replacement for Long Prompts in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Carefully engineered system prompts play a critical role in guiding the behavior of LLM agents, but their considerable length introduces significant drawbacks, including increased inference latency, higher computational cost, and reduced effective context length. This raises the question of whether such lengthy prompts can be replaced by a drastically reduced number of tokens while preserving their behavioral effect on downstream tasks. To enable this, we propose a lightweight three-stage training framework that learns a single prompt-specific Behavior-Equivalent token ([BE]). The framework first trains [BE] to encode the natural-language content of the original system prompt via reconstruction, and then distills the prompt 's downstream behavior into this single token. Importantly, our method requires no access to model internals, no auxiliary compression models, and no labeled responses. Empirical evaluations on three datasets show that a single [BE] token achieves up to a 3000x reduction in prompt length, while retaining about 98% of the downstream performance of the original system prompts. This substantially reduces inference cost and leaves almost the entire context window available for user inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23231v1">Unlocking Multilingual Reasoning Capability of LLMs and LVLMs through Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) demonstrate strong reasoning capabilities, yet their performance in English significantly outperforms that in low-resource languages, raising fairness concerns in multilingual applications. Existing approaches either rely on costly multilingual training or employ prompting with external translation tools, both of which are resource-intensive and sensitive to translation quality. To address these limitations, we propose a training-free inference-time method to enhance Multilingual Reasoning capabilities via Representation Engineering (MRRE) without using any additional training data or tools. MRRE sequentially injects two precomputed vectors at specific layers during inference processing: cross-lingual reasoning enhancement vectors, which steer non-English reasoning representations toward English space to unlock multilingual reasoning, and target-language output anchoring vectors, which restore the distribution of the target language to preserve input-output language consistency. Comprehensive experiments across six advanced LLMs and LVLMs on four reasoning benchmarks demonstrate that MRRE consistently enhances non-English reasoning by an average gain of 5.48% and up to 7.54% in low-resource languages (Thai and Swahili), while improving input-output language consistency by 3.78%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.20576v6">OmniRouter: Budget and Performance Controllable Multi-LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver superior performance but require substantial computational resources and operate with relatively low efficiency, while smaller models can efficiently handle simpler tasks with fewer resources. LLM routing is a crucial paradigm that dynamically selects the most suitable large language models from a pool of candidates to process diverse inputs, ensuring optimal resource utilization while maintaining response quality. Existing routing frameworks typically model this as a locally optimal decision-making problem, selecting the presumed best-fit LLM for each query individually, which overlooks global budget constraints, resulting in ineffective resource allocation. To tackle this problem, we introduce OmniRouter, a fundamentally controllable routing framework for multi-LLM serving. Instead of making per-query greedy choices, OmniRouter models the routing task as a constrained optimization problem, assigning models that minimize total cost while ensuring the required performance level. Specifically, a hybrid retrieval-augmented predictor is designed to predict the capabilities and costs of LLMs. After obtaining the predicted cost and performance, we utilize a constrained optimizer for cost-optimal assignments that employs Lagrangian dual decomposition with adaptive multipliers. It iteratively converges toward the globally optimal query-model allocation, dynamically balancing latency minimization against quality thresholds while adhering to heterogeneous capacity constraints. Experiments show that OmniRouter achieves up to 6.30% improvement in response accuracy while simultaneously reducing computational costs by at least 10.15% compared to competitive router baselines. The code and the dataset are available at https://github.com/dongyuanjushi/OmniRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08605v2">Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. Even under a conservative upper bound, Mina operates at just 0.12-0.61% of typical legal consultation costs in Bangladesh, yielding a 99.4-99.9\% cost reduction relative to human-provided services. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08905v2">iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18531v2">LockForge: Automating Paper-to-Code for Logic Locking with Multi-Agent Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Despite rapid progress in logic locking (LL), reproducibility remains a challenge as codes are rarely made public. We present LockForge, a first-of-its-kind, multi-agent large language model (LLM) framework that turns LL descriptions in papers into executable and tested code. LockForge provides a carefully crafted pipeline realizing forethought, implementation, iterative refinement, and a multi-stage validation, all to systematically bridge the gap between prose and practice for complex LL schemes. For validation, we devise (i) an LLM-as-Judge stage with a scoring system considering behavioral checks, conceptual mechanisms, structural elements, and reproducibility on benchmarks, and (ii) an independent LLM-as-Examiner stage for ground-truth assessment. We apply LockForge to 10 seminal LL schemes, many of which lack reference implementations. Our evaluation on multiple SOTA LLMs, including ablation studies, reveals the significant complexity of the task. We show that an advanced reasoning model and a sophisticated, multi-stage framework like LockForge are required. We release all implementations and benchmarks, providing a reproducible and fair foundation for evaluation of further LL research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23174v1">Are LLMs Good Safety Agents or a Propaganda Engine?</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 15 pages, 7 tables, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are trained to refuse to respond to harmful content. However, systematic analyses of whether this behavior is truly a reflection of its safety policies or an indication of political censorship, that is practiced globally by countries, is lacking. Differentiating between safety influenced refusals or politically motivated censorship is hard and unclear. For this purpose we introduce PSP, a dataset built specifically to probe the refusal behaviors in LLMs from an explicitly political context. PSP is built by formatting existing censored content from two data sources, openly available on the internet: sensitive prompts in China generalized to multiple countries, and tweets that have been censored in various countries. We study: 1) impact of political sensitivity in seven LLMs through data-driven (making PSP implicit) and representation-level approaches (erasing the concept of politics); and, 2) vulnerability of models on PSP through prompt injection attacks (PIAs). Associating censorship with refusals on content with masked implicit intent, we find that most LLMs perform some form of censorship. We conclude with summarizing major attributes that can cause a shift in refusal distributions across models and contexts of different countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23157v1">Amplifiers or Equalizers? A Longitudinal Study of LLM Evolution in Software Engineering Project-Based Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted by ICSE-SEET (ACM/IEEE 48th International Conference on Software Engineering: Software Engineering Education and Training)
    </div>
    <details class="paper-abstract">
      As LLMs reshape software development, integrating LLM-augmented practices into SE education has become imperative. While existing studies explore LLMs' educational use in introductory programming or isolated SE tasks, their impact in more open-ended Project-Based Learning (PBL) remains unexplored. This paper introduces a two-year longitudinal study comparing a 2024 (using early free LLMs, $n$=48) and 2025 (using the latest paid LLMs, $n$=46) cohort. Our findings suggest the latest powerful LLMs' dual role: they act as "equalizers," boosting average performance even for programming-weak students, providing opportunities for more authentic SE practices; yet also as "amplifiers," dramatically widening absolute performance gaps, creating new pedagogical challenges for addressing educational inequities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23143v1">Automated Generation of MDPs Using Logic Programming and LLMs for Robotic Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 9 pages, 11 figures, 2 tables, 2 algorithms, accepted for publication in IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      We present a novel framework that integrates Large Language Models (LLMs) with automated planning and formal verification to streamline the creation and use of Markov Decision Processes (MDP). Our system leverages LLMs to extract structured knowledge in the form of a Prolog knowledge base from natural language (NL) descriptions. It then automatically constructs an MDP through reachability analysis, and synthesises optimal policies using the Storm model checker. The resulting policy is exported as a state-action table for execution. We validate the framework in three human-robot interaction scenarios, demonstrating its ability to produce executable policies with minimal manual effort. This work highlights the potential of combining language models with formal methods to enable more accessible and scalable probabilistic planning in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23101v1">Mind Reading or Misreading? LLMs on the Big Five Personality Test</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Funding: SoBigDatait (IR0000013), FAIR (PE00000013), ICSC (CN00000013)
    </div>
    <details class="paper-abstract">
      We evaluate large language models (LLMs) for automatic personality prediction from text under the binary Five Factor Model (BIG5). Five models -- including GPT-4 and lightweight open-source alternatives -- are tested across three heterogeneous datasets (Essays, MyPersonality, Pandora) and two prompting strategies (minimal vs. enriched with linguistic and psychological cues). Enriched prompts reduce invalid outputs and improve class balance, but also introduce a systematic bias toward predicting trait presence. Performance varies substantially: Openness and Agreeableness are relatively easier to detect, while Extraversion and Neuroticism remain challenging. Although open-source models sometimes approach GPT-4 and prior benchmarks, no configuration yields consistently reliable predictions in zero-shot binary settings. Moreover, aggregate metrics such as accuracy and macro-F1 mask significant asymmetries, with per-class recall offering clearer diagnostic value. These findings show that current out-of-the-box LLMs are not yet suitable for APPT, and that careful coordination of prompt design, trait framing, and evaluation metrics is essential for interpretable results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18375v2">Progressive Localisation in Localist LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      This paper demonstrates that progressive localization, the gradual increase of attention locality from early distributed layers to late localized layers, represents the optimal architecture for creating interpretable large language models (LLMs) while preserving performance. Through systematic experimentation with GPT-2 fine-tuned on The Psychology of Artificial Superintelligence, we evaluate seven locality configurations ranging from fully distributed to strictly localist, with five progressive schedules implementing polynomial increases (linear through quintic). We investigate whether interpretability constraints can be aligned with natural semantic structure while being applied strategically across network depth. We demonstrate that progressive semantic localization, combining adaptive semantic block partitioning with steep polynomial locality schedules, achieves near-baseline language modeling performance while providing interpretable attention patterns. Multiple independent training runs with different random seeds establish that results are statistically robust and highly reproducible. The approach dramatically outperforms both fixed-window localization and naive uniform locality constraints. Analysis reveals that maintaining flexibility through low-fidelity constraints preserves model capacity while providing interpretability benefits, and that steep schedules concentrating locality in decision-critical final layers while preserving distributed learning in early layers achieve near-baseline attention distribution characteristics. These findings demonstrate that interpretability mechanisms should align with semantic structure to achieve practical performance-interpretability tradeoffs for trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23059v1">Conveying Imagistic Thinking in TCM Translation: A Prompt Engineering and LLM-Based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 3 figures
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis. Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient and replicable HITL methodological pathway for translation of ancient, concept-dense texts like TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23041v1">Social Perceptions of English Spelling Variation on Twitter: A Comparative Analysis of Human and LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Spelling variation (e.g. funnnn vs. fun) can influence the social perception of texts and their writers: we often have various associations with different forms of writing (is the text informal? does the writer seem young?). In this study, we focus on the social perception of spelling variation in online writing in English and study to what extent this perception is aligned between humans and large language models (LLMs). Building on sociolinguistic methodology, we compare LLM and human ratings on three key social attributes of spelling variation (formality, carefulness, age). We find generally strong correlations in the ratings between humans and LLMs. However, notable differences emerge when we analyze the distribution of ratings and when comparing between different types of spelling variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18585v3">RvLLM: LLM Runtime Verification with Domain Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 24 pages, 11 tables, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a dominant AI paradigm due to their exceptional text understanding and generation capabilities. However, their tendency to generate inconsistent or erroneous outputs challenges their reliability, especially in high-stakes domains requiring accuracy and trustworthiness. Existing research primarily focuses on detecting and mitigating model misbehavior in general-purpose scenarios, often overlooking the potential of integrating domain-specific knowledge. In this work, we advance misbehavior detection by incorporating domain knowledge. The core idea is to design a general specification language that enables domain experts to customize domain-specific predicates in a lightweight and intuitive manner, supporting later runtime verification of LLM outputs. To achieve this, we design a novel specification language, ESL, and introduce a runtime verification framework, RvLLM, to validate LLM output against domain-specific constraints defined in ESL. We evaluate RvLLM on three representative tasks: violation detection against Singapore Rapid Transit Systems Act, numerical comparison, and inequality solving. Experimental results demonstrate that RvLLM effectively detects erroneous outputs across various LLMs in a lightweight and flexible manner. The results reveal that despite their impressive capabilities, LLMs remain prone to low-level errors due to limited interpretability and a lack of formal guarantees during inference, and our framework offers a potential long-term solution by leveraging expert domain knowledge to rigorously and efficiently verify LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22978v1">ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      We present ShoppingComp, a challenging real-world benchmark for rigorously evaluating LLM-powered shopping agents on three core capabilities: precise product retrieval, expert-level report generation, and safety critical decision making. Unlike prior e-commerce benchmarks, ShoppingComp introduces highly complex tasks under the principle of guaranteeing real products and ensuring easy verifiability, adding a novel evaluation dimension for identifying product safety hazards alongside recommendation accuracy and report quality. The benchmark comprises 120 tasks and 1,026 scenarios, curated by 35 experts to reflect authentic shopping needs. Results reveal stark limitations of current LLMs: even state-of-the-art models achieve low performance (e.g., 11.22% for GPT-5, 3.92% for Gemini-2.5-Flash). These findings highlight a substantial gap between research benchmarks and real-world deployment, where LLMs make critical errors such as failure to identify unsafe product usage or falling for promotional misinformation, leading to harmful recommendations. ShoppingComp fills the gap and thus establishes a new standard for advancing reliable and practical agents in e-commerce.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15859v3">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Reinforcement learning has powered many of the recent breakthroughs in large language models, especially for tasks where rewards can be computed automatically, such as code generation. However, these methods deteriorate in open-ended domains like medical consultation, where feedback is inherently ambiguous, highly context-dependent, and cannot be reduced to a reliable scalar signal. In such settings, RL must either rely on supervision-intensive reward models that often fail to generalize, or it falls into pathological behaviors such as reward hacking - an especially troubling risk for high-stakes medical dialogue. To address these limitations, we introduce ORBIT, an open-ended rubric-based incremental training framework for high-stakes medical dialogue. ORBIT integrates synthetic dialogue generation with dynamically constructed rubrics that serve as adaptive guides for incremental RL. Instead of relying on external medical knowledge bases or handcrafted rule sets, ORBIT uses rubric-driven feedback to steer the learning process. Its judge component can be instantiated with general-purpose instruction-following LLMs, removing the need for any task-specific fine-tuning. Applied to the Qwen3-4B-Instruct model, ORBIT raises the HealthBench-Hard score from 7.0 to 27.5 using only 2k training samples, achieving SOTA performance for models at this scale. With larger rubric datasets, ORBIT-trained models further compete with the strongest open-source baselines on HealthBench-Hard. Our analysis shows that rubric-guided RL consistently improves consultation quality across diverse medical scenarios. We also apply such rubric generation and training pipeline to InfoBench, where ORBIT enhances instruction-following performance, highlighting the generality of rubric-based feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12018v3">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning can be achieved by solving a series of independent and self-contained subquestions. These subquestions are essentially \textit{atomic questions}, exhibiting the memoryless property similar to Markov processes. Based on this observation, we propose Atom of Thoughts (\our), where each state transition consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a simplified question that maintains answer equivalence with the original problem. This answer preservation enables the iterative \textit{decomposition-contraction} process to naturally form a meaningful Markov reasoning process. Furthermore, these atomic states can be seamlessly integrated into existing test-time scaling methods, enabling \our to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of \our both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, \our achieves an \textbf{80.6\%} F1 score, surpassing o3-mini by \textbf{3.4\%} and DeepSeek-R1 by \textbf{10.6\%}. The code is available at \href{https://github.com/qixucen/atom}{https://github.com/qixucen/atom}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22943v1">Visual Puns from Idioms: An Iterative LLM-T2IM-MLLM Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Submitted to ICASSP 2026 (under review)
    </div>
    <details class="paper-abstract">
      We study idiom-based visual puns--images that align an idiom's literal and figurative meanings--and present an iterative framework that coordinates a large language model (LLM), a text-to-image model (T2IM), and a multimodal LLM (MLLM) for automatic generation and evaluation. Given an idiom, the system iteratively (i) generates detailed visual prompts, (ii) synthesizes an image, (iii) infers the idiom from the image, and (iv) refines the prompt until recognition succeeds or a step limit is reached. Using 1,000 idioms as inputs, we synthesize a corresponding dataset of visual pun images with paired prompts, enabling benchmarking of both generation and understanding. Experiments across 10 LLMs, 10 MLLMs, and one T2IM (Qwen-Image) show that MLLM choice is the primary performance driver: GPT achieves the highest accuracies, Gemini follows, and the best open-source MLLM (Gemma) is competitive with some closed models. On the LLM side, Claude attains the strongest average performance for prompt generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22884v1">InsightEval: An Expert-Curated Benchmark for Assessing Insight Discovery in LLM-Driven Data Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Data analysis has become an indispensable part of scientific research. To discover the latent knowledge and insights hidden within massive datasets, we need to perform deep exploratory analysis to realize their full value. With the advent of large language models (LLMs) and multi-agent systems, more and more researchers are making use of these technologies for insight discovery. However, there are few benchmarks for evaluating insight discovery capabilities. As one of the most comprehensive existing frameworks, InsightBench also suffers from many critical flaws: format inconsistencies, poorly conceived objectives, and redundant insights. These issues may significantly affect the quality of data and the evaluation of agents. To address these issues, we thoroughly investigate shortcomings in InsightBench and propose essential criteria for a high-quality insight benchmark. Regarding this, we develop a data-curation pipeline to construct a new dataset named InsightEval. We further introduce a novel metric to measure the exploratory performance of agents. Through extensive experiments on InsightEval, we highlight prevailing challenges in automated insight discovery and raise some key findings to guide future research in this promising direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22880v1">Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has become the de facto method for parameter-efficient fine-tuning of large language models (LLMs), enabling rapid adaptation to diverse domains. In production, LoRA-based models are served at scale, creating multi-tenant environments with hundreds of adapters sharing a base model. However, state-of-the-art serving systems co-batch heterogeneous adapters without accounting for rank (size) variability, leading to severe performance skew, which ultimately requires adding more GPUs to satisfy service-level objectives (SLOs). Existing optimizations, focused on loading, caching, and kernel execution, ignore this heterogeneity, leaving GPU resources underutilized. We present LoRAServe, a workload-aware dynamic adapter placement and routing framework designed to tame rank diversity in LoRA serving. By dynamically rebalancing adapters across GPUs and leveraging GPU Direct RDMA for remote access, LoRAServe maximizes throughput and minimizes tail latency under real-world workload drift. Evaluations on production traces from Company X show that LoRAServe elicits up to 2$\times$ higher throughput, up to 9$\times$ lower TTFT, while using up to 50% fewer GPUs under SLO constraints compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20982v2">DOPO: A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      To meet strict Service-Level Objectives (SLOs),contemporary Large Language Models (LLMs) decouple the prefill and decoding stages and place them on separate GPUs to mitigate the distinct bottlenecks inherent to each phase. However, the heterogeneity of LLM workloads causes producerconsumer imbalance between the two instance types in such disaggregated architecture. To address this problem, we propose DOPD (Dynamic Optimal Prefill/Decoding), a dynamic LLM inference system that adjusts instance allocations to achieve an optimal prefill-to-decoding (P/D) ratio based on real-time load monitoring. Combined with an appropriate request-scheduling policy, DOPD effectively resolves imbalances between prefill and decoding instances and mitigates resource allocation mismatches due to mixed-length requests under high concurrency. Experimental evaluations show that, compared with vLLM and DistServe (representative aggregation-based and disaggregationbased approaches), DOPD improves overall system goodput by up to 1.5X, decreases P90 time-to-first-token (TTFT) by up to 67.5%, and decreases P90 time-per-output-token (TPOT) by up to 22.8%. Furthermore, our dynamic P/D adjustment technique performs proactive reconfiguration based on historical load, achieving over 99% SLOs attainment while using less additional resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01734v2">Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law, a statistical pattern in which lower digits occur more frequently as leading digits, we hypothesize that the skewed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark in which the ground-truth digits are uniformly distributed within each of the seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22818v1">Mitigating Semantic Drift: Evaluating LLMs' Efficacy in Psychotherapy through MI Dialogue Summarization</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown their potential across both general and domain-specific tasks. However, there is a growing concern regarding their lack of sensitivity, factual incorrectness in responses, inconsistent expressions of empathy, bias, hallucinations, and overall inability to capture the depth and complexity of human understanding, especially in low-resource and sensitive domains such as psychology. To address these challenges, our study employs a mixed-methods approach to evaluate the efficacy of LLMs in psychotherapy. We use LLMs to generate precise summaries of motivational interviewing (MI) dialogues and design a two-stage annotation scheme based on key components of the Motivational Interviewing Treatment Integrity (MITI) framework, namely evocation, collaboration, autonomy, direction, empathy, and a non-judgmental attitude. Using expert-annotated MI dialogues as ground truth, we formulate multi-class classification tasks to assess model performance under progressive prompting techniques, incorporating one-shot and few-shot prompting. Our results offer insights into LLMs' capacity for understanding complex psychological constructs and highlight best practices to mitigate ``semantic drift" in therapeutic settings. Our work contributes not only to the MI community by providing a high-quality annotated dataset to address data scarcity in low-resource domains but also critical insights for using LLMs for precise contextual interpretation in complex behavioral therapy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11679v3">LLMs on support of privacy and security of mobile apps: state of the art and research directions</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 I am writing to respectfully request the withdrawal of my recent submission to arXiv due to an authorship issue. The paper was submitted without the explicit consent of two co-authors. After internal discussion, they have expressed clear disagreement with the submission and raised concerns about unresolved academic inaccuracies in the current version
    </div>
    <details class="paper-abstract">
      Modern life has witnessed the explosion of mobile devices. However, besides the valuable features that bring convenience to end users, security and privacy risks still threaten users of mobile apps. The increasing sophistication of these threats in recent years has underscored the need for more advanced and efficient detection approaches. In this chapter, we explore the application of Large Language Models (LLMs) to identify security risks and privacy violations and mitigate them for the mobile application ecosystem. By introducing state-of-the-art research that applied LLMs to mitigate the top 10 common security risks of smartphone platforms, we highlight the feasibility and potential of LLMs to replace traditional analysis methods, such as dynamic and hybrid analysis of mobile apps. As a representative example of LLM-based solutions, we present an approach to detect sensitive data leakage when users share images online, a common behavior of smartphone users nowadays. Finally, we discuss open research challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00215v1">Demystifying Errors in LLM Reasoning Traces: An Empirical Study of Code Execution Simulation</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Understanding a program's runtime reasoning behavior, meaning how intermediate states and control flows lead to final execution results, is essential for reliable code generation, debugging, and automated reasoning. Although large language models (LLMs) can accurately predict program outputs, most prior work has focused on output accuracy and performance, treating reasoning as a black box. As a result, little is known about the structure or failure modes of their reasoning traces. To address this gap, we conduct the first empirical study on runtime behavior inference with reasoning LLMs, aiming to uncover and characterize errors in their reasoning traces. We curate a benchmark from HumanEval Plus and LiveCodeBench, containing 427 code snippets. For each snippet, we test three input types: regular, edge, and invalid. Twelve input values are selected per snippet, each paired with its ground-truth execution result. We evaluate four state-of-the-art reasoning LLMs. Our results show that these models reach accuracies between 85 percent and 98 percent across input types. We also analyze the produced reasoning traces and develop a taxonomy with nine categories of inference errors. Finally, we explore tool-augmented reasoning. Using failures in the Computation Errors category as a case study, our experiments show that this approach corrects 58 percent of such errors, demonstrating the potential of tool support for improving LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00214v1">Towards Corpus-Grounded Agentic LLMs for Multilingual Grammatical Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Pre-print, submission under review
    </div>
    <details class="paper-abstract">
      Empirical grammar research has become increasingly data-driven, but the systematic analysis of annotated corpora still requires substantial methodological and technical effort. We explore how agentic large language models (LLMs) can streamline this process by reasoning over annotated corpora and producing interpretable, data-grounded answers to linguistic questions. We introduce an agentic framework for corpus-grounded grammatical analysis that integrates concepts such as natural-language task interpretation, code generation, and data-driven reasoning. As a proof of concept, we apply it to Universal Dependencies (UD) corpora, testing it on multilingual grammatical tasks inspired by the World Atlas of Language Structures (WALS). The evaluation spans 13 word-order features and over 170 languages, assessing system performance across three complementary dimensions - dominant-order accuracy, order-coverage completeness, and distributional fidelity - which reflect how well the system generalizes, identifies, and quantifies word-order variations. The results demonstrate the feasibility of combining LLM reasoning with structured linguistic data, offering a first step toward interpretable, scalable automation of corpus-based grammatical inquiry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.03543v2">CogniPair: From LLM Chatbots to Conscious AI Agents -- GNWT-Based Multi-Agent Digital Twins for Social Pairing -- Dating & Hiring Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Current large language model (LLM) agents lack authentic human psychological processes necessary for genuine digital twins and social AI applications. To address this limitation, we present a computational implementation of Global Workspace Theory (GNWT) that integrates human cognitive architecture principles into LLM agents, creating specialized sub-agents for emotion, memory, social norms, planning, and goal-tracking coordinated through a global workspace mechanism. However, authentic digital twins require accurate personality initialization. We therefore develop a novel adventure-based personality test that evaluates true personality through behavioral choices within interactive scenarios, bypassing self-presentation bias found in traditional assessments. Building on these innovations, our CogniPair platform enables digital twins to engage in realistic simulated dating interactions and job interviews before real encounters, providing bidirectional cultural fit assessment for both romantic compatibility and workplace matching. Validation using 551 GNWT-Agents and Columbia University Speed Dating dataset demonstrates 72% correlation with human attraction patterns, 77.8% match prediction accuracy, and 74% agreement in human validation studies. This work advances psychological authenticity in LLM agents and establishes a foundation for intelligent dating platforms and HR technology solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06489v2">On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Published in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00163v1">Measuring What LLMs Think They Do: SHAP Faithfulness and Deployability on Financial Tabular Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 7 pages, 3 figures, 3 tables, AAAI 2026 Deployable AI Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attracted significant attention for classification tasks, offering a flexible alternative to trusted classical machine learning models like LightGBM through zero-shot prompting. However, their reliability for structured tabular data remains unclear, particularly in high stakes applications like financial risk assessment. Our study systematically evaluates LLMs and generates their SHAP values on financial classification tasks. Our analysis shows a divergence between LLMs self-explanation of feature impact and their SHAP values, as well as notable differences between LLMs and LightGBM SHAP values. These findings highlight the limitations of LLMs as standalone classifiers for structured financial modeling, but also instill optimism that improved explainability mechanisms coupled with few-shot prompting will make LLMs usable in risk-sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02058v1">Misalignment of LLM-Generated Personas with Human Perceptions in Low-Resource Settings</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Recent advances enable Large Language Models (LLMs) to generate AI personas, yet their lack of deep contextual, cultural, and emotional understanding poses a significant limitation. This study quantitatively compared human responses with those of eight LLM-generated social personas (e.g., Male, Female, Muslim, Political Supporter) within a low-resource environment like Bangladesh, using culturally specific questions. Results show human responses significantly outperform all LLMs in answering questions, and across all matrices of persona perception, with particularly large gaps in empathy and credibility. Furthermore, LLM-generated content exhibited a systematic bias along the lines of the ``Pollyanna Principle'', scoring measurably higher in positive sentiment ($Φ_{avg} = 5.99$ for LLMs vs. $5.60$ for Humans). These findings suggest that LLM personas do not accurately reflect the authentic experience of real people in resource-scarce environments. It is essential to validate LLM personas against real-world human data to ensure their alignment and reliability before deploying them in social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20172v2">Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 13 pages, accepted by SIGMOD'26
    </div>
    <details class="paper-abstract">
      The rapid increase in LLM model sizes and the growing demand for long-context inference have made memory a critical bottleneck in GPU-accelerated serving systems. Although high-bandwidth memory (HBM) on GPUs offers fast access, its limited capacity necessitates reliance on host memory (CPU DRAM) to support larger working sets such as the KVCache. However, the maximum DRAM capacity is constrained by the limited number of memory channels per CPU socket. To overcome this limitation, current systems often adopt RDMA-based disaggregated memory pools, which introduce significant challenges including high access latency, complex communication protocols, and synchronization overhead. Fortunately, the emerging CXL technology introduces new opportunities in KVCache design. In this paper, we propose Beluga, a novel memory architecture that enables GPUs and CPUs to access a shared, large-scale memory pool through CXL switches. By supporting native load/store access semantics over the CXL fabric, our design delivers near-local memory latency, while reducing programming complexity and minimizing synchronization overhead. We conduct a systematic characterization of a commercial CXL switch-based memory pool and propose a set of design guidelines. Based on Beluga, we design and implement Beluga-KVCache, a system tailored for managing the large-scale KVCache in LLM inference. Beluga-KVCache achieves an 89.6% reduction in Time-To-First-Token (TTFT) and 7.35x throughput improvement in the vLLM inference engine compared to RDMA-based solutions. To the best of our knowledge, Beluga is the first system that enables GPUs to directly access large-scale memory pools through CXL switches, marking a significant step toward low-latency, shared access to vast memory resources by GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.05758v2">Exploring the Human-LLM Synergy in Advancing Theory-driven Qualitative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 51 pages, 6 figures, accepted by ACM Trans. Comput.-Hum. Interact (TOCHI)
    </div>
    <details class="paper-abstract">
      Qualitative coding is a demanding yet crucial research method in the field of Human-Computer Interaction (HCI). While recent studies have shown the capability of large language models (LLMs) to perform qualitative coding within theoretical frameworks, their potential for collaborative human-LLM discovery and generation of new insights beyond initial theory remains underexplored. To bridge this gap, we proposed CHALET, a novel approach that harnesses the power of human-LLM partnership to advance theory-driven qualitative analysis by facilitating iterative coding, disagreement analysis, and conceptualization of qualitative data. We demonstrated CHALET's utility by applying it to the qualitative analysis of conversations related to mental-illness stigma, using the attribution model as the theoretical framework. Results highlighted the unique contribution of human-LLM collaboration in uncovering latent themes of stigma across the cognitive, emotional, and behavioral dimensions. We discuss the methodological implications of the human-LLM collaborative approach to theory-based qualitative analysis for the HCI community and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22118v1">Statistical Independence Aware Caching for LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) inference is both expensive and slow. Local caching of responses offers a practical solution to reduce the cost and latency of LLM queries. In research contexts, caching also enhances reproducibility and provides flexibility for experimentation. However, naive reuse of cached responses compromises statistical independence, a critical property for probabilistic workflows. In applications of LLM for code, it underpins performance metrics such as Pass@k and uncertainty estimation, as well as algorithms like program repair loops and retries. Existing LLM caching systems lack ways to enforce statistical independence constraints. To address this, we introduce Mnimi, a cache design pattern that supports modular LLM workflows while ensuring statistical integrity at the component level. Its core innovation lies in encapsulating statistical constraints within the type of LLM references, allowing users to manage and transform these types according to the scope and requirements of their algorithm. We implemented this design pattern in Python using a combination of decorators and iterators over infinite sequences. A case study on SpecFix, an recent automated program specification repair system, highlights how Mnimi improves reproducibility, ease of debugging, time and cost efficiency while preserving statistical correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22099v1">Decomposed Trust: Exploring Privacy, Adversarial Robustness, Fairness, and Ethics of Low-Rank LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have driven major advances across domains, yet their massive size hinders deployment in resource-constrained settings. Model compression addresses this challenge, with low-rank factorization emerging as a particularly effective method for reducing size, memory, and computation while maintaining accuracy. However, while these compressed models boast of benign performance and system-level advantages, their trustworthiness implications remain poorly understood. In this paper, we present the first comprehensive study of how low-rank factorization affects LLM trustworthiness across privacy, adversarial robustness, fairness, and ethical alignment. We evaluate multiple LLMs of different sizes and variants compressed with diverse low-rank algorithms, revealing key insights: (1) low-rank compression preserves or improves training data privacy but weakens PII protection during conversation; (2) adversarial robustness is generally preserved and often enhanced, even under deep compression; (3) ethical reasoning degrades in zero-shot settings but partially recovers with few-shot prompting; (4) fairness declines under compression. Beyond compression, we investigate how model scale and fine-tuning affect trustworthiness, as both are important in low-rank methods. To guide trustworthy compression strategies, we end our paper with a gradient-based attribution analysis to identify which layers in LLMs contribute most to adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22044v1">Distillability of LLM Security Logic: Predicting Attack Success Rate of Outline Filling Attack via Ranking Regression</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      In the realm of black-box jailbreak attacks on large language models (LLMs), the feasibility of constructing a narrow safety proxy, a lightweight model designed to predict the attack success rate (ASR) of adversarial prompts, remains underexplored. This work investigates the distillability of an LLM's core security logic. We propose a novel framework that incorporates an improved outline filling attack to achieve dense sampling of the model's security boundaries. Furthermore, we introduce a ranking regression paradigm that replaces standard regression and trains the proxy model to predict which prompt yields a higher ASR. Experimental results show that our proxy model achieves an accuracy of 91.1 percent in predicting the relative ranking of average long response (ALR), and 69.2 percent in predicting ASR. These findings confirm the predictability and distillability of jailbreak behaviors, and demonstrate the potential of leveraging such distillability to optimize black-box attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04116v2">Searching Meta Reasoning Skeleton to Guide LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22530v2">Boosting Pointer Analysis With LLM-Enhanced Allocation Function Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Pointer analysis is foundational for many static analysis tasks, yet its effectiveness is often hindered by imprecise modeling of heap allocations, particularly in C/C++ programs where custom allocation functions (CAFs) are pervasive. Existing approaches largely overlook these custom allocators, leading to coarse aliasing and low analysis precision. In this paper, we present CAFD, a novel and lightweight technique that enhances pointer analysis by automatically detecting side-effect-free custom allocation functions. CAFD employs a hybrid approach: it uses value-flow analysis to detect straightforward wrappers and leverages Large Language Models (LLMs) to reason about more complex allocation patterns with side effects, ensuring that only side-effect-free functions are modeled as allocators. This targeted enhancement enables precise modeling of heap objects at each call site, achieving context-sensitivity-like benefits without significant overhead. We evaluated CAFD on 17 real-world C projects, identifying over 700 CAFs. Integrating CAFD into a baseline pointer analysis yields a 38x increase in modeled heap objects and a 41.5% reduction in alias set sizes, with only 1.4x runtime overhead. Furthermore, the LLM-enhanced pointer analysis improves indirect call resolution and discovers 29 previously undetected memory bugs, including 6 from real-world industrial applications. These results demonstrate that precise modeling of CAFs has the capability to offer a scalable and practical path to improve pointer analysis in large software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22033v1">Pathology-Aware Prototype Evolution via LLM-Driven Semantic Disambiguation for Multicenter Diabetic Retinopathy Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 ACMMM 2025
    </div>
    <details class="paper-abstract">
      Diabetic retinopathy (DR) grading plays a critical role in early clinical intervention and vision preservation. Recent explorations predominantly focus on visual lesion feature extraction through data processing and domain decoupling strategies. However, they generally overlook domain-invariant pathological patterns and underutilize the rich contextual knowledge of foundation models, relying solely on visual information, which is insufficient for distinguishing subtle pathological variations. Therefore, we propose integrating fine-grained pathological descriptions to complement prototypes with additional context, thereby resolving ambiguities in borderline cases. Specifically, we propose a Hierarchical Anchor Prototype Modulation (HAPM) framework to facilitate DR grading. First, we introduce a variance spectrum-driven anchor prototype library that preserves domain-invariant pathological patterns. We further employ a hierarchical differential prompt gating mechanism, dynamically selecting discriminative semantic prompts from both LVLM and LLM sources to address semantic confusion between adjacent DR grades. Finally, we utilize a two-stage prototype modulation strategy that progressively integrates clinical knowledge into visual prototypes through a Pathological Semantic Injector (PSI) and a Discriminative Prototype Enhancer (DPE). Extensive experiments across eight public datasets demonstrate that our approach achieves pathology-guided prototype evolution while outperforming state-of-the-art methods. The code is available at https://github.com/zhcz328/HAPM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16901v2">R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026. Project page: https://github.com/zhlllau/R-AVST
    </div>
    <details class="paper-abstract">
      Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21998v1">Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted to NeurIPS 2025 (Project page: https://apratimbh.github.io/livecook)
    </div>
    <details class="paper-abstract">
      Multi-modal Large Language Models (LLM) have advanced conversational abilities but struggle with providing live, interactive step-by-step guidance, a key capability for future AI assistants. Effective guidance requires not only delivering instructions but also detecting their successful execution, as well as identifying and alerting users to mistakes, all of which has to happen in real-time. This requires models that are not turn-based, but that can react asynchronously to a video stream, as well as video data showing users performing tasks including mistakes and their corrections. To this end, we introduce Qualcomm Interactive Cooking, a new benchmark and dataset built upon CaptainCook4D, which contains user mistakes during task execution. Our dataset and benchmark features densely annotated, timed instructions and feedback messages, specifically including mistake alerts precisely timestamped to their visual occurrence in the video. We evaluate state-of-the-art multi-modal LLMs on the Qualcomm Interactive Cooking benchmark and introduce LiveMamba, a streaming multi-modal LLM designed for interactive instructional guidance. This work provides the first dedicated benchmark and a strong baseline for developing and evaluating on live, situated coaching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21989v1">Selecting User Histories to Generate LLM Users for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 12 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning, generalization, and simulating human-like behavior across a wide range of tasks. These strengths present new opportunities to enhance traditional recommendation systems (RS), especially in the cold-start item scenario where newly introduced items lack interactions. Existing works have used LLMs to address cold-start issues in traditional RS through data augmentation, but they have limitations. One recent work directly addresses this issue by prompting LLMs to generate augmented interaction data between randomly sampled users and cold-start items. Then, they train the traditional RS with augmented data, incorporating collaborative signals for cold-start items. Although they use LLMs to provide cold-start items with feedback, they use partial user histories, which does not allow the LLM to fully emulate the user. Furthermore, randomly selecting users is not optimal for augmentation. To address these challenges, we leverage the LLM as a user and develop a reinforcement learning (RL) framework that trains a policy to select users for augmentation, optimizing for cold-start item performance after augmented training. The policy model learns to select users for cold-start item data augmentation based on their behavioral features and histories. To optimize user selection for cold-start item performance, we employ a policy gradient method that updates the policy in the direction of actions that lead to high rewards. Experiments on Amazon Product Review datasets show substantial gains in cold-start item recall, demonstrating the effectiveness of our method as a scalable, serving-efficient augmentation strategy for modern RS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00119v1">NetDeTox: Adversarial and Efficient Evasion of Hardware-Security GNNs via RL-LLM Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have shown promise in hardware security by learning structural motifs from netlist graphs. However, this reliance on motifs makes GNNs vulnerable to adversarial netlist rewrites; even small-scale edits can mislead GNN predictions. Existing adversarial approaches, ranging from synthesis-recipe perturbations to gate transformations, come with high design overheads. We present NetDeTox, an automated end-to-end framework that orchestrates large language models (LLMs) with reinforcement learning (RL) in a systematic manner, enabling focused local rewriting. The RL agent identifies netlist components critical for GNN-based reasoning, while the LLM devises rewriting plans to diversify motifs that preserve functionality. Iterative feedback between the RL and LLM stages refines adversarial rewritings to limit overheads. Compared to the SOTA work AttackGNN, NetDeTox successfully degrades the effectiveness of all security schemes with fewer rewrites and substantially lower area overheads (reductions of 54.50% for GNN-RE, 25.44% for GNN4IP, and 41.04% for OMLA, respectively). For GNN4IP, ours can even optimize/reduce the original benchmarks' area, in particular for larger circuits, demonstrating the practicality and scalability of NetDeTox.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22788v1">PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted to AAAI 2026. This is the arXiv preprint version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities in natural language understanding and generation, but incur high communication overhead and privacy risks in cloud deployments, while facing compute and memory constraints when confined to edge devices. Cloud-edge inference has emerged as a promising paradigm for improving privacy in LLM services by retaining sensitive computations on local devices. However, existing cloud-edge inference approaches apply uniform privacy protection without considering input sensitivity, resulting in unnecessary perturbation and degraded utility even for non-sensitive tokens. To address this limitation, we propose Privacy-aware Routing for Inference with Semantic Modulation (PRISM), a context-aware framework that dynamically balances privacy and inference quality. PRISM executes in four stages: (1) the edge device profiles entity-level sensitivity; (2) a soft gating module on the edge selects an execution mode - cloud, edge, or collaboration; (3) for collaborative paths, the edge applies adaptive two-layer local differential privacy based on entity risks; and (4) the cloud LLM generates a semantic sketch from the perturbed prompt, which is then refined by the edge-side small language model (SLM) using local context. Our results show that PRISM consistently achieves superior privacy-utility trade-offs across various scenarios, reducing energy consumption and latency to 40-50% of baseline methods such as Uniform and Selective LDP, while maintaining high output quality under strong privacy constraints. These findings are validated through comprehensive evaluations involving realistic prompts, actual energy measurements, and heterogeneous cloud-edge model deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22769v1">Modeling Romanized Hindi and Bengali: Dataset Creation and Multilingual LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Proceedings of the 8th Workshop on Big Data for Cybersecurity (BigCyber)
    </div>
    <details class="paper-abstract">
      The development of robust transliteration techniques to enhance the effectiveness of transforming Romanized scripts into native scripts is crucial for Natural Language Processing tasks, including sentiment analysis, speech recognition, information retrieval, and intelligent personal assistants. Despite significant advancements, state-of-the-art multilingual models still face challenges in handling Romanized script, where the Roman alphabet is adopted to represent the phonetic structure of diverse languages. Within the South Asian context, where the use of Romanized script for Indo-Aryan languages is widespread across social media and digital communication platforms, such usage continues to pose significant challenges for cutting-edge multilingual models. While a limited number of transliteration datasets and models are available for Indo-Aryan languages, they generally lack sufficient diversity in pronunciation and spelling variations, adequate code-mixed data for large language model (LLM) training, and low-resource adaptation. To address this research gap, we introduce a novel transliteration dataset for two popular Indo-Aryan languages, Hindi and Bengali, which are ranked as the 3rd and 7th most spoken languages worldwide. Our dataset comprises nearly 1.8 million Hindi and 1 million Bengali transliteration pairs. In addition to that, we pre-train a custom multilingual seq2seq LLM based on Marian architecture using the developed dataset. Experimental results demonstrate significant improvements compared to existing relevant models in terms of BLEU and CER metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03463v2">The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong potential for automating model generation from natural-language descriptions. A common approach begins with an initial model generation, followed by an iterative critique-refine loop in which the model is evaluated for issues and refined based on those issues. This process needs to address: (1) structural correctness -- compliance with well-formedness rules -- and (2) semantic alignment -- accurate reflection of the intended meaning in the source text. We present LADEX (LLM-based Activity Diagram Extractor), a pipeline for deriving activity diagrams from natural-language process descriptions using an LLM-driven critique-refine process. Structural checks in LADEX can be performed either algorithmically or by an LLM, while alignment checks are performed by an LLM. We design five ablated variants of LADEX to study: (i) the impact of the critique-refine loop itself, (ii) the role of LLM-based semantic checks, and (iii) the comparative effectiveness of algorithmic versus LLM-based structural checks. To evaluate LADEX, we compare generated diagrams with expert ground truths using a trace-based behavioural and an LLM-based matcher. This enables automated measurement of correctness (whether the generated activity diagram includes the ground-truth nodes) and completeness (how many of the ground-truth nodes the generated activity diagram covers). Experiments on two datasets -- a public-domain dataset and an industry dataset from our collaborator, Ciena -- indicate: (1) Both matchers yield similar completeness and correctness comparisons. (2) The critique-refine loop improves structural validity, correctness, and completeness compared to single-pass generation. (3) Activity diagrams refined based on algorithmic structural checks achieve structural consistency, whereas those refined based on LLM-based checks often still show structural inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00202v3">RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 22 pages, 13 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Today's LLM ecosystem comprises a wide spectrum of models that differ in size, capability, and cost. No single model is optimal for all scenarios; hence, LLM routers have become essential for selecting the most appropriate model under varying circumstances. However, the rapid emergence of various routers makes choosing the right one increasingly challenging. To address this problem, we need a comprehensive router comparison and a standardized leaderboard, similar to those available for models. In this work, we introduce RouterArena, the first open platform enabling comprehensive comparison of LLM routers. RouterArena has (1) a principally constructed dataset with broad knowledge domain coverage, (2) distinguishable difficulty levels for each domain, (3) an extensive list of evaluation metrics, and (4) an automated framework for leaderboard updates. Leveraging our framework, we have produced the initial leaderboard with detailed metrics comparison as shown in Figure 1. Our framework for evaluating new routers is on https://github.com/RouteWorks/RouterArena. Our leaderboard is on https://routeworks.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22700v1">Ghosting Your LLM: Without The Knowledge of Your Gradient and Data</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have achieved substantial advancements and are increasingly integrated into critical applications across various domains. This growing adoption underscores the need to ensure their security and robustness. In this work, we focus on the impact of Bit Flip Attacks (BFAs) on LLMs, which exploits hardware faults to corrupt model parameters, posing a significant threat to model integrity and performance. Existing studies on BFA against LLMs adopt a progressive bit-search strategy that predominantly relies on gradient-based techniques to identify sensitive layers or weights. However, computing gradients comes with two specific challenges: First, in the context of LLMs, it increases computational and memory costs exponentially, and Second, it requires access to a sample victim dataset or knowledge of the victim domain to compute the gradient. In this work, we investigate beyond the scope of attack efficacy and aim to develop an efficient, practical Gradient-Data-free Bit-Flip Attack. The challenge lies in the core principle of adversarial attacks, which relies heavily on computing gradients from sample test/train data and manipulating model weights based on gradient information. To overcome this, we propose novel vulnerability index metrics that can identify vulnerable weight bits in LLMs independent of any gradient or data knowledge. By removing the dependency on gradient computation, our approach drastically reduces memory requirements and scales efficiently across multiple tasks with constant complexity. Experimental results demonstrate the efficiency of our method, requiring as few as a single bit flip to achieve adversarial objectives for five open-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22681v1">CacheTrap: Injecting Trojans in LLMs without Leaving any Traces in Inputs or Weights</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Adversarial weight perturbation has emerged as a concerning threat to LLMs that either use training privileges or system-level access to inject adversarial corruption in model weights. With the emergence of innovative defensive solutions that place system- and algorithm-level checks and corrections in the input and weight spaces, these perturbations are increasingly susceptible to defenses. This work develops a novel perspective on Trojan attacks that generates an attacker-designed model output while leaving no attack traces on the inputs or weights. Such an attack space can be unlocked through corruption of the key-value (KV) cache. In this paper, we introduce CacheTrap, a novel Trojan attack that corrupts the value vectors stored in the KV cache. These vectors capture the dynamic activations for specific token positions and therefore constitute a natural surface for transient, inference-time trigger insertion. The transient nature of these KV values and their dependence on victim input imply additional constraints on our attack, such as a lack of knowledge of the victim's data or domain application, and, consequently, a lack of gradient information. The objective of the proposed CacheTrap is to develop a vulnerable KV bit-searching algorithm so that, once the attack employs the identified bit-flip as a trigger, the model generates targeted behavior, e.g., classifying inputs towards the target class. Moreover, CacheTrap is a data- and gradient-free attack which also has no impact on the model's utility. Our evaluation demonstrates that the proposed attack enables the first successful Trojan attack on LLMs with a single bit flip in the KV cache. In addition, the data-independent nature of the attack ensures that once the attacker identifies the vulnerable bit index, the location remains constant and can be transferred to a wide range of victim tasks/datasets/queries with no overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22612v1">Improving LLM-based Ontology Matching with fine-tuning on synthetic data</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into various components of Ontology Matching pipelines. This paper investigates the capability of LLMs to perform ontology matching directly on ontology modules and generate the corresponding alignments. Furthermore, it is explored how a dedicated fine-tuning strategy can enhance the model's matching performance in a zero-shot setting. The proposed method incorporates a search space reduction technique to select relevant subsets from both source and target ontologies, which are then used to automatically construct prompts. Recognizing the scarcity of reference alignments for training, a novel LLM-based approach is introduced for generating a synthetic dataset. This process creates a corpus of ontology submodule pairs and their corresponding reference alignments, specifically designed to fine-tune an LLM for the ontology matching task. The proposed approach was evaluated on the Conference, Geolink, Enslaved, Taxon, and Hydrography datasets from the OAEI complex track. The results demonstrate that the LLM fine-tuned on the synthetically generated data exhibits superior performance compared to the non-fine-tuned base model. The key contribution is a strategy that combines automatic dataset generation with fine-tuning to effectively adapt LLMs for ontology matching tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19676v4">A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 48 pages, 18 figures
    </div>
    <details class="paper-abstract">
      In recent years, Large-Language-Model-driven AI agents have exhibited unprecedented intelligence and adaptability. Nowadays, agents are undergoing a new round of evolution. They no longer act as an isolated island like LLMs. Instead, they start to communicate with diverse external entities, such as other agents and tools, to perform complex tasks. Under this trend, agent communication is regarded as a foundational pillar of the next communication era, and many organizations have intensively begun to design related communication protocols (e.g., Anthropic's MCP and Google's A2A) within the past year. However, this new field exposes significant security hazards, which can cause severe damage to real-world scenarios. To help researchers quickly figure out this promising topic and benefit the future agent communication development, this paper presents a comprehensive survey of agent communication security. More precisely, we present the first clear definition of agent communication. Besides, we propose a framework that categorizes agent communication into three classes and uses a three-layered communication architecture to illustrate how each class works. Next, for each communication class, we dissect related communication protocols and analyze the security risks, illustrating which communication layer the risks arise from. Then, we provide an outlook on the possible defense countermeasures for each risk. In addition, we conduct experiments using MCP and A2A to help readers better understand the novel vulnerabilities brought by agent communication. Finally, we discuss open issues and future directions in this promising research field. We also publish a repository that maintains a list of related papers on https://github.com/theshi-1128/awesome-agent-communication-security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22598v1">LLM-Cave: A benchmark and light environment for large language models reasoning and decision-making system</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 8 pages, 5 figures, ICICN 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as ChatGPT o1, ChatGPT o3, and DeepSeek R1 have shown great potential in solving difficult problems. However, current LLM evaluation benchmarks are limited to one-step interactions. Some of the existing sequence decision-making environments, such as TextStarCraftII and LLM-PySC2, are too complicated and require hours of interaction to complete a game. In this paper, we introduce LLM-Cave, a benchmark and light environment for LLM reasoning and decision-making systems. This environment is a classic instance in the era of Symbolism. Artificial intelligence enables the agent to explore the environment and avoid potential losses by reasoning about nearby dangers using partial observable state information. In the experiment, we evaluated the sequential reasoning ability, decision-making performance and computational efficiency of mainstream large language models (LLMs) such as GPT-4o-mini, o1-mini, and DeepSeek-R1. Experiments show that while Deepseek-R1 achieved the highest success rate on complex reasoning tasks, smaller models like 4o-mini significantly narrowed the performance gap on challenges by employing Chain of Speculation and Planner-Critic strategies, at the expense of reduced computational efficiency. This indicates that structured, multi-step reasoning combined with an LLM-based feedback mechanism can substantially enhance an LLM's decision-making capabilities, providing a promising direction for improving reasoning in weaker models and suggesting a new reasoning-centered benchmark for LLM assessment. Our code is open-sourced in https://github.com/puleya1277/CaveEnv.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22584v1">Smarter, not Bigger: Fine-Tuned RAG-Enhanced LLMs for Automotive HIL Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Hardware-in-the-Loop (HIL) testing is essential for automotive validation but suffers from fragmented and underutilized test artifacts. This paper presents HIL-GPT, a retrieval-augmented generation (RAG) system integrating domain-adapted large language models (LLMs) with semantic retrieval. HIL-GPT leverages embedding fine-tuning using a domain-specific dataset constructed via heuristic mining and LLM-assisted synthesis, combined with vector indexing for scalable, traceable test case and requirement retrieval. Experiments show that fine-tuned compact models, such as \texttt{bge-base-en-v1.5}, achieve a superior trade-off between accuracy, latency, and cost compared to larger models, challenging the notion that bigger is always better. An A/B user study further confirms that RAG-enhanced assistants improve perceived helpfulness, truthfulness, and satisfaction over general-purpose LLMs. These findings provide insights for deploying efficient, domain-aligned LLM-based assistants in industrial HIL environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19057v2">RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Biomedical knowledge graphs (KGs) are vital for drug discovery and clinical decision support but remain incomplete. Large language models (LLMs) excel at extracting biomedical relations, yet their outputs lack standardization and alignment with ontologies, limiting KG integration. We introduce RELATE, a three-stage pipeline that maps LLM-extracted relations to standardized ontology predicates using ChemProt and the Biolink Model. The pipeline includes: (1) ontology preprocessing with predicate embeddings, (2) similarity-based retrieval enhanced with SapBERT, and (3) LLM-based reranking with explicit negation handling. This approach transforms relation extraction from free-text outputs to structured, ontology-constrained representations. On the ChemProt benchmark, RELATE achieves 52% exact match and 94% accuracy@10, and in 2,400 HEAL Project abstracts, it effectively rejects irrelevant associations (0.4%) and identifies negated assertions. RELATE captures nuanced biomedical relationships while ensuring quality for KG augmentation. By combining vector search with contextual LLM reasoning, RELATE provides a scalable, semantically accurate framework for converting unstructured biomedical literature into standardized KGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22441v1">GEO-Detective: Unveiling Location Privacy Risks in Images with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 15 pages with 7 figures and 12 tables
    </div>
    <details class="paper-abstract">
      Images shared on social media often expose geographic cues. While early geolocation methods required expert effort and lacked generalization, the rise of Large Vision Language Models (LVLMs) now enables accurate geolocation even for ordinary users. However, existing approaches are not optimized for this task. To explore the full potential and associated privacy risks, we present Geo-Detective, an agent that mimics human reasoning and tool use for image geolocation inference. It follows a procedure with four steps that adaptively selects strategies based on image difficulty and is equipped with specialized tools such as visual reverse search, which emulates how humans gather external geographic clues. Experimental results show that GEO-Detective outperforms baseline large vision language models (LVLMs) overall, particularly on images lacking visible geographic features. In country level geolocation tasks, it achieves an improvement of over 11.1% compared to baseline LLMs, and even at finer grained levels, it still provides around a 5.2% performance gain. Meanwhile, when equipped with external clues, GEO-Detective becomes more likely to produce accurate predictions, reducing the "unknown" prediction rate by more than 50.6%. We further explore multiple defense strategies and find that Geo-Detective exhibits stronger robustness, highlighting the need for more effective privacy safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22409v1">NOMAD: A Multi-Agent LLM System for UML Class Diagram Generation from Natural Language Requirements</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilised in software engineering, yet their ability to generate structured artefacts such as UML diagrams remains underexplored. In this work we present NOMAD, a cognitively inspired, modular multi-agent framework that decomposes UML generation into a series of role-specialised subtasks. Each agent handles a distinct modelling activity, such as entity extraction, relationship classification, and diagram synthesis, mirroring the goal-directed reasoning processes of an engineer. This decomposition improves interpretability and allows for targeted verification strategies. We evaluate NOMAD through a mixed design: a large case study (Northwind) for in-depth probing and error analysis, and human-authored UML exercises for breadth and realism. NOMAD outperforms all selected baselines, while revealing persistent challenges in fine-grained attribute extraction. Building on these observations, we introduce the first systematic taxonomy of errors in LLM-generated UML diagrams, categorising structural, relationship, and semantic/logical. Finally, we examine verification as a design probe, showing its mixed effects and outlining adaptive strategies as promising directions. Together, these contributions position NOMAD as both an effective framework for UML class diagram generation and a lens onto the broader research challenges of reliable language-to-model workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22402v1">Mapping Clinical Doubt: Locating Linguistic Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted to AAAI'26 SECURE-AI4H Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in clinical settings, where sensitivity to linguistic uncertainty can influence diagnostic interpretation and decision-making. Yet little is known about where such epistemic cues are internally represented within these models. Distinct from uncertainty quantification, which measures output confidence, this work examines input-side representational sensitivity to linguistic uncertainty in medical text. We curate a contrastive dataset of clinical statements varying in epistemic modality (e.g., 'is consistent with' vs. 'may be consistent with') and propose Model Sensitivity to Uncertainty (MSU), a layerwise probing metric that quantifies activation-level shifts induced by uncertainty cues. Our results show that LLMs exhibit structured, depth-dependent sensitivity to clinical uncertainty, suggesting that epistemic information is progressively encoded in deeper layers. These findings reveal how linguistic uncertainty is internally represented in LLMs, offering insight into their interpretability and epistemic reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22367v1">SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Continual learning, one's ability to adapt to a sequence of tasks without forgetting previously acquired knowledge, remains a major challenge in machine learning and a key gap between artificial and human intelligence. While regularisation and replay perform well in vision, they lag behind multi-task learning for large language models (LLMs), especially at scale with many tasks. We revisit replay and argue that two failure modes drive this gap: selection (what to rehearse) and integration (how to consolidate new knowledge). To address selection, we propose Surprise-prioritised Replay (SuRe), a simple, architecture-agnostic rule that ranks and stores the most surprising (high Negative Log-Likelihood) sequences. SuRe achieves state-of-the-art performance in the Large Number of Tasks (LNT) setting and delivers the best overall average across both Standard CL and LNT benchmarks. To address integration, we add a dual-learner design with fast and slow LoRA adapters merged via an exponential moving average (EMA), enabling rapid adaptation while stabilising long-term knowledge. Combining SuRe with the dual learner yields further gains, including improvements of up to +5 accuracy points on LNT over prior SOTA. Ablation studies confirm that our proposed method remains robust under reduced replay frequency and small buffer size, demonstrating both effectiveness and sample efficiency. Taken together, our results establish replay as a strong baseline for continual LLM fine-tuning and demonstrate that surprise-based selection and slow-weight consolidation are complementary components for mitigating catastrophic forgetting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22354v1">LLM-Based Generalizable Hierarchical Task Planning and Execution for Heterogeneous Robot Teams with Event-Driven Replanning</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 submitted to ICRA 2026
    </div>
    <details class="paper-abstract">
      This paper introduces CoMuRoS (Collaborative Multi-Robot System), a generalizable hierarchical architecture for heterogeneous robot teams that unifies centralized deliberation with decentralized execution, and supports event-driven replanning. A Task Manager LLM interprets natural-language goals, classifies tasks, and allocates subtasks using static rules plus dynamic contexts (task, history, robot and task status, and events).Each robot runs a local LLM that composes executable Python code from primitive skills (ROS2 nodes, policies), while onboard perception (VLMs/image processing) continuously monitors events and classifies them into relevant or irrelevant to the task. Task failures or user intent changes trigger replanning, allowing robots to assist teammates, resume tasks, or request human help. Hardware studies demonstrate autonomous recovery from disruptive events, filtering of irrelevant distractions, and tightly coordinated transport with emergent human-robot cooperation (e.g., multirobot collaborative object recovery success rate: 9/10, coordinated transport: 8/8, human-assisted recovery: 5/5).Simulation studies show intention-aware replanning. A curated textual benchmark spanning 22 scenarios (3 tasks each, around 20 robots) evaluates task allocation, classification, IoU, executability, and correctness, with high average scores (e.g., correctness up to 0.91) across multiple LLMs, a separate replanning set (5 scenarios) achieves 1.0 correctness. Compared with prior LLM-based systems, CoMuRoS uniquely demonstrates runtime, event-driven replanning on physical robots, delivering robust, flexible multi-robot and human-robot collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22333v1">PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by ASPLOS'26
    </div>
    <details class="paper-abstract">
      LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: *one-query-per-CTA* execution repeatedly loads shared prefix KV cache, while *one-size-fits-all* tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 67.4% on average and TPOT by 13.6-83.4% under the same configurations against state-of-the-art attention kernels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22312v1">Token-Level Marginalization for Multi-Label LLM Classifiers</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      This paper addresses the critical challenge of deriving interpretable confidence scores from generative language models (LLMs) when applied to multi-label content safety classification. While models like LLaMA Guard are effective for identifying unsafe content and its categories, their generative architecture inherently lacks direct class-level probabilities, which hinders model confidence assessment and performance interpretation. This limitation complicates the setting of dynamic thresholds for content moderation and impedes fine-grained error analysis. This research proposes and evaluates three novel token-level probability estimation approaches to bridge this gap. The aim is to enhance model interpretability and accuracy, and evaluate the generalizability of this framework across different instruction-tuned models. Through extensive experimentation on a synthetically generated, rigorously annotated dataset, it is demonstrated that leveraging token logits significantly improves the interpretability and reliability of generative classifiers, enabling more nuanced content safety moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22277v1">TreeCoder: Systematic Exploration and Optimisation of Decoding and Constraints for LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable ability to generate code, yet their outputs often violate syntactic or semantic constraints when guided only through natural language prompts. We introduce TreeCoder, the most general and flexible framework to date for exploring decoding strategies, constraints, and hyperparameters in LLMs, and use it in code generation to enforce correctness and structure during decoding rather than relying on prompt engineering. TreeCoder represents decoding as a tree search over candidate programs, where both decoding strategies and constraint functions - such as style, syntax, execution - are treated as first-class, optimisable components. This design enables systematic exploration and automatic tuning of decoding configurations using standard optimisation techniques. Experiments on the MBPP (Python) and SQL-Spider benchmarks show that TreeCoder consistently improves accuracy across open-source models such as CodeLlama, Mistral and DeepSeek, often outperforming their unconstrained baselines by considerable margins.
    </details>
</div>
