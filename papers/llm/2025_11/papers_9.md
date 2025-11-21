# llm - 2025_11

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
- Part 9

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00592v1">Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Accepted at the 34th International Conference on Parallel Architectures and Compilation Techniques (PACT 2025). 12 pages, plus appendix
    </div>
    <details class="paper-abstract">
      Automatic code optimization remains a difficult challenge, particularly for complex loop nests on modern hardware. This paper investigates a novel approach to code optimization where Large Language Models (LLMs) guide the process through a closed-loop interaction with a compiler. We present ComPilot, an experimental framework that leverages off-the-shelf LLMs, without any task-specific fine-tuning, as interactive optimization agents. ComPilot establishes a feedback loop where an LLM proposes transformations for a given loop nest to a compiler. The compiler attempts the transformations, reporting back legality status and measured speedup or slowdown. The LLM utilizes this concrete feedback to iteratively refine its optimization strategy. Our extensive evaluation across the PolyBench benchmark suite demonstrates the effectiveness of this zero-shot approach. ComPilot achieves geometric mean speedups of 2.66x (single run) and 3.54x (best-of-5 runs) over the original code. Furthermore, ComPilot demonstrates competitive performance against the state-of-the-art Pluto polyhedral optimizer, outperforming it in many cases. This experimental study demonstrates that general-purpose LLMs can effectively guide the code optimization process when grounded by compiler feedback, opening promising research directions for agentic AI in code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00556v1">Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Preprint, 14 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreaking attacks despite their impressive capabilities. Investigating these weaknesses is crucial for robust safety mechanisms. Existing attacks primarily distract LLMs by introducing additional context or adversarial tokens, leaving the core harmful intent unchanged. In this paper, we introduce ISA (Intent Shift Attack), which obfuscates LLMs about the intent of the attacks. More specifically, we establish a taxonomy of intent transformations and leverage them to generate attacks that may be misperceived by LLMs as benign requests for information. Unlike prior methods relying on complex tokens or lengthy context, our approach only needs minimal edits to the original request, and yields natural, human-readable, and seemingly harmless prompts. Extensive experiments on both open-source and commercial LLMs show that ISA achieves over 70% improvement in attack success rate compared to direct harmful prompts. More critically, fine-tuning models on only benign data reformulated with ISA templates elevates success rates to nearly 100%. For defense, we evaluate existing methods and demonstrate their inadequacy against ISA, while exploring both training-free and training-based mitigation strategies. Our findings reveal fundamental challenges in intent inference for LLMs safety and underscore the need for more effective defenses. Our code and datasets are available at https://github.com/NJUNLP/ISA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00554v1">Red-teaming Activation Probes using Prompted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Activation probes are attractive monitors for AI systems due to low cost and latency, but their real-world robustness remains underexplored. We ask: What failure modes arise under realistic, black-box adversarial pressure, and how can we surface them with minimal effort? We present a lightweight black-box red-teaming procedure that wraps an off-the-shelf LLM with iterative feedback and in-context learning (ICL), and requires no fine-tuning, gradients, or architectural access. Running a case study with probes for high-stakes interactions, we show that our approach can help discover valuable insights about a SOTA probe. Our analysis uncovers interpretable brittleness patterns (e.g., legalese-induced FPs; bland procedural tone FNs) and reduced but persistent vulnerabilities under scenario-constraint attacks. These results suggest that simple prompted red-teaming scaffolding can anticipate failure patterns before deployment and might yield promising, actionable insights to harden future probes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00488v1">\texttt{ReMind}: Understanding Deductive Code Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in code-related tasks. Despite their advancement, empirical evidence reveals that they still struggle with \emph{deductive code reasoning}, the ability to reason about the program execution process. While prior studies have recognized this limitation, the underlying causes remain largely underexplored. In this paper, we begin by presenting a comprehensive empirical study that reveals three key challenges undermining deductive code reasoning: (1) an intrinsic gap between generation and reasoning abilities, (2) a consistent bias towards code sources, and (3) weak zero-shot generalization on complex benchmarks. In light of these challenges, we propose \texttt{ReMind}, a multi-agent framework composed of \texttt{Mutator}, \texttt{Executor}, and \texttt{Inspector}. The \texttt{Mutator} generates code variants to mitigate bias towards code sources, the \texttt{Executor} traces variable states step-by-step to expose inconsistency, and the \texttt{Inspector} identifies problematic reasoning steps and provides control-flow refinement to bridge the intrinsic reasoning gap. Through their coordinated collaboration, \texttt{ReMind} systematically identifies and refines reasoning flaws, achieving outstanding performance and enabling robust zero-shot generalization. Extensive experiments on two benchmarks with five LLMs demonstrate the superior advantages of \texttt{ReMind} compared to baseline approaches in deductive code reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00476v1">Remembering Unequally: Global and Disciplinary Bias in LLM-Generated Co-Authorship Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Ongoing breakthroughs in Large Language Models (LLMs) are reshaping search and recommendation platforms at their core. While this shift unlocks powerful new scientometric tools, it also exposes critical fairness and bias issues that could erode the integrity of the information ecosystem. Additionally, as LLMs become more integrated into web-based searches for scholarly tools, their ability to generate summarized research work based on memorized data introduces new dimensions to these challenges. The extent of memorization in LLMs can impact the accuracy and fairness of the co-authorship networks they produce, potentially reflecting and amplifying existing biases within the scientific community and across different regions. This study critically examines the impact of LLM memorization on the co-authorship networks. To this end, we assess memorization effects across three prominent models, DeepSeek R1, Llama 4 Scout, and Mixtral 8x7B, analyzing how memorization-driven outputs vary across academic disciplines and world regions. While our global analysis reveals a consistent bias favoring highly cited researchers, this pattern is not uniformly observed. Certain disciplines, such as Clinical Medicine, and regions, including parts of Africa, show more balanced representation, pointing to areas where LLM training data may reflect greater equity. These findings underscore both the risks and opportunities in deploying LLMs for scholarly discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00432v1">G2: Guided Generation for Enhanced Output Diversity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across diverse natural language processing tasks. However, these models exhibit a critical limitation in output diversity, often generating highly similar content across multiple attempts. This limitation significantly affects tasks requiring diverse outputs, from creative writing to reasoning. Existing solutions, like temperature scaling, enhance diversity by modifying probability distributions but compromise output quality. We propose Guide-to-Generation (G2), a training-free plug-and-play method that enhances output diversity while preserving generation quality. G2 employs a base generator alongside dual Guides, which guide the generation process through decoding-based interventions to encourage more diverse outputs conditioned on the original query. Comprehensive experiments demonstrate that G2 effectively improves output diversity while maintaining an optimal balance between diversity and quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00413v1">Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      In agentic LLM scenarios, an agent's interaction process during a single rollout often exhibits branching behaviors. Due to memory retrieval and concurrent tool executions at certain decision points, the token trajectory of one task evolves into a tree-like structure rather than a linear sequence. However, current training pipelines decompose such tree-structured trajectories into separate linear segments, treating each branch as an independent sequence. As a result, shared prefixes across these branches are repeatedly recomputed during both forward and backward passes. To address this inefficiency, we propose Tree Training, a paradigm that computes each shared prefix only once and reuses its intermediate results across related branches during both forward and backward passes, substantially improving computation efficiency in large-scale agentic training. This is achieved via (i) Tree Packing, which efficiently reuses shared computations across trajectories, and (ii) Gradient Restoration, which ensures correct gradient propagation across reused prefixes. Experiments on multiple open-source models demonstrate up to 3.9x reduction in total training time, enabling more efficient agentic LLM SFT and RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00382v1">Efficiency vs. Alignment: Investigating Safety and Fairness Risks in Parameter-Efficient Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Organizations are increasingly adopting and adapting Large Language Models (LLMs) hosted on public repositories such as HuggingFace. Although these adaptations often improve performance on specialized downstream tasks, recent evidence indicates that they can also degrade a model's safety or fairness. Since different fine-tuning techniques may exert distinct effects on these critical dimensions, this study undertakes a systematic assessment of their trade-offs. Four widely used Parameter-Efficient Fine-Tuning methods, LoRA, IA3, Prompt-Tuning, and P-Tuning, are applied to four instruction-tuned model families (Meta-Llama-3-8B, Qwen2.5-7B, Mistral-7B, and Gemma-7B). In total, 235 fine-tuned variants are evaluated across eleven safety hazard categories and nine demographic fairness dimensions. The results show that adapter-based approaches (LoRA, IA3) tend to improve safety scores and are the least disruptive to fairness, retaining higher accuracy and lower bias scores. In contrast, prompt-based methods (Prompt-Tuning and P-Tuning) generally reduce safety and cause larger fairness regressions, with decreased accuracy and increased bias. Alignment shifts are strongly moderated by base model type: LLaMA remains stable, Qwen records modest gains, Gemma experiences the steepest safety decline, and Mistral, which is released without an internal moderation layer, displays the greatest variance. Improvements in safety do not necessarily translate into improvements in fairness, and no single configuration optimizes all fairness metrics simultaneously, indicating an inherent trade-off between these objectives. These findings suggest a practical guideline for safety-critical deployments: begin with a well-aligned base model, favour adapter-based PEFT, and conduct category-specific audits of both safety and fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00346v1">Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 10 pages, 5 figures, 4 tables, Published at the Brazilian Symposium on Cybersecurity (SBSeg 2025)
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has raised significant concerns about their security against adversarial attacks. In this work, we propose a novel approach to crafting universal jailbreaks and data extraction attacks by exploiting latent space discontinuities, an architectural vulnerability related to the sparsity of training data. Unlike previous methods, our technique generalizes across various models and interfaces, proving highly effective in seven state-of-the-art LLMs and one image generation model. Initial results indicate that when these discontinuities are exploited, they can consistently and profoundly compromise model behavior, even in the presence of layered defenses. The findings suggest that this strategy has substantial potential as a systemic attack vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00343v1">LingGym: How Far Are LLMs from Thinking Like Field Linguists?</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      This paper introduces LingGym, a new benchmark that evaluates LLMs' capacity for meta-linguistic reasoning using Interlinear Glossed Text (IGT) and grammatical descriptions extracted from 18 typologically diverse reference grammars. Unlike previous work that focuses on specific downstream tasks, we assess whether LLMs can generalize linguistic inference across low-resource languages and structures not seen during training. We present a controlled evaluation task: Word-Gloss Inference, in which the model must infer a missing word and gloss from context using varying levels of linguistic information (e.g., glosses, grammatical explanations, translations). Our results show that incorporating structured linguistic cues leads to consistent improvements in reasoning performance across all models. This work highlights both the promise and current limitations of using LLMs for typologically informed linguistic analysis and low-resource language documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00340v1">Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 41 pages, 4 images
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into high-stakes legal work has exposed a critical gap: no benchmark exists to systematically stress-test their reliability against the nuanced, adversarial, and often subtle flaws present in real-world contracts. To address this, we introduce CLAUSE, a first-of-its-kind benchmark designed to evaluate the fragility of an LLM's legal reasoning. We study the capabilities of LLMs to detect and reason about fine-grained discrepancies by producing over 7500 real-world perturbed contracts from foundational datasets like CUAD and ContractNLI. Our novel, persona-driven pipeline generates 10 distinct anomaly categories, which are then validated against official statutes using a Retrieval-Augmented Generation (RAG) system to ensure legal fidelity. We use CLAUSE to evaluate leading LLMs' ability to detect embedded legal flaws and explain their significance. Our analysis shows a key weakness: these models often miss subtle errors and struggle even more to justify them legally. Our work outlines a path to identify and correct such reasoning failures in legal AI.
    </details>
</div>
