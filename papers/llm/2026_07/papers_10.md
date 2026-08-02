# llm - 2026_07

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
- Part 10

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01103v1">Clinician-Level Agreement Without Clinical Caution: LLM Evaluator Limits in Medical AI Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Open-response evaluation provides stronger clinical validity than multiple-choice benchmarks but creates a scoring bottleneck that motivates automated LLM-asa-Judge approaches. Whether such evaluators replicate clinical calibration and caution, however, remains untested. We introduce MedQADE, the first standardised open-response clinical benchmark for German, a major clinical language lacking native evaluation infrastructure, comprising 3,800 items annotated by ten practising physicians and nine Large Language Model (LLM) evaluators. The top-performing evaluator model, Gemini 3 Flash, reached alignment consistent with the physician ceiling (\k{appa} = 0.694 vs. \k{appa} = 0.709), though wide confidence intervals limit interpretation. Despite this statistical alignment, automated evaluators exhibited near-absent clinical metacognition: physicians scaled abstention with item difficulty, while frontier models assigned definitive scores in every case. We additionally quantified systematic lineage-dependent biases, where models preferentially scored architectural siblings, an effect independent of language. These results show that statistical alignment does not ensure clinical caution, and that evaluator independence requires explicit verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14660v2">NeuroFilter: Activation-Based Guardrails for Privacy-Conscious LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Agentic Large Language Models (LLMs) are models able to reason, plan, and execute tools over unstructured data. These abilities are enabling transformative applications in domains spanning from personal assistant, financial, and legal domains. While these systems can substantially improve productivity and service quality, effective agency typically requires access to sensitive personal or organizational information. However, this access introduces critical inference-time privacy risks, specifically regarding contextually appropriate information disclosure. While recent studies highlight the inability of agentic LLMs to consistently adhere to privacy norms, existing defenses often rely on auxiliary LLM-based monitors. However, these defenses are expensive and offer limited protection against attacks that are robust to semantic censorship. To contrast this background, this paper proposes a notion of privacy filters based on activation probing. We show that these filters are both computationally efficient and effective for both single-turn and multi-turn conversational settings. Furthermore, this work provides the first systematic investigation into probing model internals across a conversation trajectory, moving beyond static, single-prompt analysis to capture the evolving state of privacy-sensitive interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01047v1">Conversable Complexity: Agentic LLM Collectives as Interpretable Substrates</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Complexity and interpretability rarely coincide: systems rich enough for complex behaviours to emerge are usually too opaque to question, while transparent ones are too simple for anything complex to emerge. A single large language model (LLM) is a static artefact, hardly exhibiting any of the emergent properties we associate with life. This changes through interaction: populations of LLMs display emergent dynamics absent from isolated models. Furthermore, LLMs can be endowed with persistent memory, tools and shared skills, and the capacity to initiate actions unprompted, i.e., turning LLMs agentic. In this paper, we argue that such collectives of agents can serve as a computational substrate for Artificial Life (ALife) research. Critically, since the agents communicate in natural language, their collective behaviour can be directly interrogated by examining textual traces and asking the agents themselves. We outline the notion of interpretability in language-model research and extend it for collectives of agents. Lastly, we survey recent examples of agentic LLM collectives that already instantiate the idea of agentic substrates, from controlled experiments to deployments in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01040v1">As It Was: Aligning LLM Search Evaluation with Historical User Preferences</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Large-scale search systems evolve faster than human quality assurance can scale, especially for long-tail intents and multilingual queries. LLM-as-a-judge approaches provide a scalable alternative for evaluating the relevance of search engine result pages (SERPs), but judgments based solely on semantic similarity or world knowledge can drift from actual user preferences, particularly for ambiguous queries. We introduce a behavior-grounded LLM judge that augments each SERP item with a lightweight and auditable behavioral prior in the form of a Query-Relevance-Impressions (QRI) card. Each card summarizes how users have historically interacted with similar queries and results, providing compact empirical evidence that the judge can cite to resolve ambiguity and make more consistent relevance judgments while still relying on semantic reasoning. In a large-scale music search evaluation at Spotify, using relevance estimates derived from historical user interactions across 6,000 recomposed SERPs, the behavior-grounded judge achieves stronger alignment with user preferences, improving Spearman rank correlation by approximately 5% overall and yielding a 91% relative improvement on disagreement cases. On a multilingual human-judged dataset spanning five languages, grounding further increases correlation with human relevance judgments by 15%. Importantly, when evaluated against outcomes from a live A/B test, the grounded judge shows consistently higher alignment with the observed winning model. While absolute alignment remains moderate, these findings demonstrate that lightweight behavioral grounding can improve the reliability and practical usefulness of LLM-based evaluation in real-world search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08625v2">From Holistic Evaluation to Structured Criteria: Rubrics Across the Evolving LLM Landscape</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) advance toward open-ended autonomous agents, the mechanisms used to evaluate and guide their behavior must evolve accordingly. This work introduces the rubric as a unifying framework capturing this evolution, characterizing rubrics as a dynamic response to successive LLM paradigm shifts that recurs across otherwise independent efforts in evaluation, reinforcement learning, and safety alignment. We define rubrics as explicit criteria sets that transform complex quality judgments into structured and actionable standards, and demonstrate that their recurrence across these research threads is not coincidental. We systematically organize existing rubric designs, examine their construction and optimization, and analyze their role across evaluation and training. Rubrics manifest at three progressively deeper levels: at the evaluative level, they decompose holistic judgments into verifiable dimensions; at the training level, they serve as dense feedback signals providing process-level guidance where scalar rewards fall short; at the intrinsic level, they emerge dynamically from model behaviors, driving self-improvement. We further assess rubric reliability across generation quality, execution fidelity, theoretical constraints, and security threats, before surveying rubric-based benchmarks across diverse domains. By rendering assessment transparent and decomposable, rubrics translate human value expectations into machine-learnable signals, serving as the enduring bridge between human intentions and machine behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01299v1">HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      In retrieval augmented generation (RAG) and agentic LLM serving, prompts are assembled from independent segments into long contexts, making the prefill stage dominate the per-request computation cost. To this cost, two directions have emerged in parallel: position-independent caching (PIC) admits KV reuse for non-contiguous segments shared across different requests, while hybrid-attention models reduce computation complexity by replacing most full-attention layers with linear attention. However, they cannot coexist: applying PIC to hybrid-attention models breaks down because per-token KV-cache reuse primitives do not transfer to the per-request recurrent state. In this work, we present Hypic, the first serving system for hybrid-attention LLMs with position-independent caching. For linear-attention layers, we identify the segment-cumulative transition operator as the missing algebraic primitive, and cache it alongside each segment's zero-start end-state, enabling near-exact and constant-time state composition of independently cached segments. For the remaining full-attention layers, existing PIC methods also fail as linear layers do not expose the per-token hidden states for selective recomputation. We show that the most significant attention deviation concentrates at segment boundaries, so recomputing only a small seam window at each boundary suffices to restore cross-segment lookback. Finally, Hypic exploits segment-level self-containment to parallelize cache-miss prefill across instances, turning long cold requests -- a major tail-latency contributor under both prefix caching and prior PIC -- into an accelerable workload. Evaluated across four hybrid-attention models and five workloads, Hypic reduces time-to-first-token (TTFT) by 2.45x on average and improves peak throughput by up to 2.0x over existing systems, while staying within 3.3 points of full-recompute accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00939v1">Leveraging LLM-Based Agentic Systems to Generate Quantum Applications for Test Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Quantum computing is increasingly explored for software engineering (SE) optimization, but translating natural-language (NL) task-level requirements into executable quantum applications still demands substantial quantum and programming expertise. We present QPipe, a large language model (LLM)-based multi-agent architecture that autonomously turns NL requirements into traceable quantum-application workflows through specialized agents for requirement parsing, formulation, code generation, review, execution, and verification. We evaluate QPipe on 20 NL requirements, each associated with a real-world benchmark and a test-optimization problem. QPipe successfully completes the key stages of quantum-application generation across requirements, achieving average rates of 100% for code compilation and 96.7% for application execution and final-result combination, with average generation costs of 260.1 seconds and 1.89M tokens per requirement. Among the generated quantum applications that execute successfully, the returned solutions outperform the offline genetic algorithm baseline in most cases. Ablation results further show that QPipe's advantage depends on retaining code-generation skills, task knowledge, review feedback, and multi-agent decomposition. These results indicate that agentic coordination can support generation of executable quantum applications for tackling test optimization problems from real-world benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00937v1">Persona Non Grata: LLM Persona-Driven Generations in MCQA are Unstable in Distinct Dimensions</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 23 pages, 12 figures. Under review at ARR
    </div>
    <details class="paper-abstract">
      Persona-driven generations (PDGs) have seen prolific use in research and industry applications, where a large language model (LLM) takes on a 'persona' while completing some task. While persona expressed through free-form text (like dialogue) has substantial work investigating stability or consistency, relatively, persona expressed in non-text-heavy outputs (like in multiple-choice question answering, or MCQA) is often overlooked. We work to address this gap, seeking to understand the instability of LLM PDGs in MCQA tasks. We develop three metrics investigating the performance, outcome, and question correctness stability, evaluating three distinct dimensions. Using these metrics, we find that instability varies consistently between model families and model size, and across question domains, with math/commonsense questions leading to greater instability. We also find task prompt format introduces more prediction instability than other hyperparameters, like temperature. Finally, we find that instability is related to task accuracy, and using our instability metrics, find different experimental settings that result in different best and worst personas for tasks, despite their similarity. This reveals the importance of checking hyperparameter instability in PDGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13445v3">Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 ICLR 2026 Workshop on Principled Design for Trustworthy AI - Interpretability, Robustness, and Safety across Modalities 67 pages, 13 figures
    </div>
    <details class="paper-abstract">
      When asked to explain their decisions, LLMs can often give explanations which sound plausible to humans. But are these explanations faithful, i.e. do they convey the factors actually responsible for the decision? In this work, we analyse counterfactual faithfulness across 75 models from 13 families. We analyze the tradeoff between conciseness and comprehensiveness, how correlational faithfulness metrics assess this tradeoff, and the extent to which metrics can be gamed. This analysis motivates two new metrics: the phi-CCT, a simplified variant of the Correlational Counterfactual Test (CCT) which avoids the need for token probabilities while explaining most of the variance of the original test; and F-AUROC, which eliminates sensitivity to imbalanced intervention distributions and captures a model's ability to produce explanations with different levels of detail. Our findings reveal a clear scaling trend: larger and more capable models are consistently more faithful on all metrics we consider. Our code is available at https://github.com/google-deepmind/corr_faith.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28229v2">Humanizing Automatically Generated Unit Test Suites with LLM-Based Refactoring</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Search-based test generation tools such as EvoSuite produce compilable and high-coverage unit tests at scale, but their suites are often hard to read and maintain. LLMs can generate more natural tests, yet direct generation remains brittle, with compilation rates of only 51-78% in our study. We introduce TestHumanizer, a hybrid SBST+LLM approach that uses LLMs as controlled refactoring layers over compilable SBST suites to improve naming, structure, and developer-oriented clarity while preserving behavior and compilation validity. We evaluate TestHumanizer on 350 classes from Defects4J and SF110. EvoSuite generates 15 suites per class, and each suite is refactored under three context configurations using gpt-4o and mistral-large-2407, yielding 31,500 refactorings. TestHumanizer reaches 88-98% compilation rates, close to EvoSuite's 100% baseline and clearly above direct LLM generation. Structural coverage is largely preserved, typically within 1-2 percentage points, and 86-95% of refactorings satisfy a composite faithful-refactoring threshold. Refactored suites also improve predicted readability, reduce control-flow and cognitive complexity, and mitigate structural smells. The summary-based setting offers the most robust trade-off, while long code-centric prompts are more prone to hallucination-induced failures. A developer study on 30 classes and 444 test methods confirms significant gains in perceived readability and willingness to adopt, with Wilcoxon p less than 0.01 and substantial inter-rater agreement. Overall, LLMs are most effective not as standalone generators but as validation-gated refinement layers over robust SBST outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00910v1">Calibrating the Instrument: Controllability of an LLM-Driven Synthetic Population</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Generative Synthetic Populations (GSP) -- the convergence of population synthesis, agent-based modelling, and LLM agents -- are attracting growing interest for urban simulation and institutional communication research. Before any GSP instrument is used on a real population, a more basic question must be answered: does it respond to stimuli of known valence in an ordered, replicable, group-structured way? We call this controllability. We ask not whether a synthetic population tracks humans, but whether it tracks itself: whether the latent structure we impose on it is recovered in its own responses. This internal-validity question is logically prior to any claim about external validity, just as characterising an instrument's response function must precede using it to test a theory. We report SIVE (Synthetic Instrument Validation Experiment): a fictional municipality (Montelago) with 120 synthetic personas of known latent structure, exposed to seven conditions spanning strongly positive to strongly negative institutional communications about a water network. Seven pre-registered criteria, evaluated across a temperature sweep, jointly assess fidelity, stability, noise floor, specificity, sensitivity, and ordering. All seven pass at every temperature. A central finding turns a calibration failure into a diagnostic success: a message designed as "weakly positive" was identified by the instrument as functionally negative, traced to unresolved problems, uncertainty, and institutional passivity in its text; a redesigned version restored the expected ordering and interacts with agents' latent trust in unanticipated ways. A noise sub-experiment shows the instrument's intrinsic noise is roughly half the cross-agent estimate and stable across temperatures. Individual trajectories reveal coherent micro-dynamics that summary statistics obscure. Full data are available via an interactive explorer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01293v1">RuleChef: Grounding LLM Task Knowledge in Human-Editable Rules</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      We present RuleChef, a framework that uses large language models (LLMs) to generate executable rules for NLP tasks such as text classification, Named Entity Recognition (NER), or relation extraction. Rules are generated based on a task description and a set of labeled examples, then they are iteratively improved based both on additional examples and on human feedback overexisting rules. RuleChef can also be used to bootstrap rules using the observed input-output pairs from any existing model for a given task. LLMs are used only at learning time, synthesizing rules and iteratively patching them based on failures measured on a held-out split. The result of this process is a fast, deterministic, and inspectable rule system. Preliminary evaluation is performed on both classification and NER tasks. We release RuleChef as open-source software under an Apache 2.0
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16511v2">Tail-Shape Estimation in LLM Evaluation Is Fragile: A Protocol for Diagnosing False Positives</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 9 pages of main paper, 4 figures and 4 tables in the main paper, more in the appendix
    </div>
    <details class="paper-abstract">
      Recent work motivates moving large language model (LLM) evaluation from mean-based to tail-aware metrics, including conditional value-at-risk and tail-index estimates of reward-model error. We ask whether the canonical extreme-value-theory tail-index parameter, which isolates how heavy a tail is from how large the tail mass is, adds discriminative information beyond the mean and a standard tail-magnitude statistic in LLM evaluation. We pre-register a protocol covering admissibility, goodness-of-fit, threshold-stability, and effect-size requirements for any positive tail-shape claim. The protocol is the contribution of this paper; the empirical study below is a demonstration of what its gates catch. Applied to a standard LLM toxicity-evaluation setup under two structurally different scorer families, the protocol catches three distinct modes of false positives that a naive analysis would have published, and rejects the headline tail-shape claim on both scorers. We conclude that tail-shape estimation in the LLM toxicity-evaluation setups we examined is more fragile than the recent literature suggests, and recommend the protocol as a starting point for tail-index claims in similar setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00848v1">MetaHOPE: A Metaphor-Oriented Evaluation Framework for Analysing MT and LLM Translation Errors</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      In this opinion paper, we propose MetaHOPE, an error severity-aware annotation framework for evaluating metaphor translations. Metaphors present challenges for machine translation (MT) and natural language understanding and processing (NLU, NLP), because it presents the features of semantic complexity, contextual dependency, and cultural embeddings that can lead to ambiguity issues for NLP models. To investigate how state-of-the-art NLP models perform on translating metaphors, we select three representative systems, i.e., GoogleMT, GPT5.4, and Hunyuan-7b as Neural MT (NMT) models and LLMs. We used two human-annotated metaphor corpora, including VUAMC and PSUCMC for English-to-Chinese and Chinese-to-English translation purposes. The original corpora we used are monolingual, where we carried out error annotation using the MetaHOPE framework, and also produced the human post-edited gold reference for bilingual use as a new resource. We believe the MetaHOPE evaluation framework for metaphor translation annotation, the parallel corpora resources, and the error analysis on SOTA automatic translation models can be useful and shed some light for the field of metaphor translation study. We share our resources publicly upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.10887v4">Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Published at ICLR 2026
    </div>
    <details class="paper-abstract">
      Growing concerns over the theft and misuse of Large Language Models (LLMs) underscore the need for effective fingerprinting to link a model to its original version and detect misuse. We define five essential properties for a successful fingerprint: Transparency, Efficiency, Persistence, Robustness, and Unforgeability. We present a novel fingerprinting framework that provides verifiable proof of ownership while preserving fingerprint integrity. Our approach makes two main contributions. First, a chain and hash technique that cryptographically binds fingerprint prompts to their responses, preventing collisions and enabling irrefutable ownership claims. Second, we address a realistic threat model in which instruction-tuned models' output distribution can be significantly altered through meta-prompts. By incorporating random padding and varied meta-prompt configurations during training, our method maintains robustness even under significant output style changes. Experiments show that our framework securely proves ownership, resists both benign transformations (e.g., fine-tuning) and adversarial fingerprint removal, and extends to fingerprinting LoRA adapters\footnote{We release our code at: https://github.com/microsoft/Chain-Hash.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00760v1">MosaicKV: Serving Long-Context LLM with Dynamic Two-D KV Cache Compression</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Long-context LLM services now sustain prompts with hundreds of thousands to millions of tokens, making the key-value (KV) cache a first-order serving cost. Because the cache grows linearly with context length, it can exhaust GPU memory, force smaller batches, and reduce serving throughput. Prior KV cache compression techniques typically target only the sequence dimension or only the channel dimension, which leaves limited headroom as context windows scale. Compressing both dimensions promises higher memory reduction, but applying the two forms of compression directly leads to significant accuracy loss. This paper introduces MosaicKV, a dynamic two-D (dimensional) KV cache compression system for extremely long-context serving. MosaicKV uses dynamic two-D compression to address the accuracy challenge, exploiting the non-uniform importance distribution of elements within the KV cache. Instead of applying one compression pattern globally, MosaicKV identifies important elements for each KV vector and selects compression strategies at the granularity of KV cache segments. To address the performance challenge, where fine-grained sparsity and compression management overhead can offset the gains from compression, MosaicKV introduces compressed KV cache management. This mechanism uses underutilized GPU and CPU resources to maintain compressed KV caches and accelerate attention computation. Evaluation on an H800 GPU with multiple LLMs shows that MosaicKV delivers up to 16x attention speedup, 4.8x lower decode latency, and 7.3x higher throughput than the uncompressed baseline. At the same time, it reduces memory usage by 3x and incurs only 1.76% average accuracy loss on LongBench and RULER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00733v1">LLM-Guided ODE Discovery and Parameter Inference from Small-Cohort Aggregate Data</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Mechanistic modeling via ordinary differential equations (ODEs) provides interpretable descriptions of complex dynamics and enables inference of underlying mechanisms, which is particularly valuable in clinical settings. However, in rare diseases, both the structure and parameters of the model are typically unknown, while individual-level data is scarce, noisy, heterogeneous, and subject to privacy constraints. In such settings, population-level summary statistics provide a practical privacy-preserving data representation, while capturing heterogeneity further requires modeling parameters as distributions rather than fixed values. Yet no existing method jointly discovers ODE structure and refines parameter distributions solely from summary statistics. We present AgentODE, an end-to-end framework that addresses this gap. An LLM proposes candidate ODE structures, while a tool-augmented inference agent iteratively refines parameter distributions through a diagnosis--update loop, operating on population-level summary statistics alone. We evaluate AgentODE on three benchmark problems across different fields and two clinical datasets, including the rare disease recessive dystrophic epidermolysis bullosa (RDEB), with only 231 observations across 46 patients. AgentODE recovers functionally consistent ODE structures across all settings, and experiments on RDEB demonstrates that in sparse and noisy data settings reasoning from summary statistics promotes mechanistically principled structure discovery, whereas baselines with individual-level data access recover implausible structures despite better predictive performance. AgentODE opens new possibilities for mechanistic modeling of rare diseases directly from population-level summary statistics, where data scarcity and privacy constraints have traditionally limited such analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00711v1">ClarifyCodeBench: Evaluating LLMs on Clarifying Ambiguous Requirements for Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Large Language Models have emerged as programming assistants. However, the efficacy of code generation is constrained by the quality of input requirements, which are frequently ambiguous, incomplete, or underspecified. While LLMs excel at one-shot code synthesis, their ability to proactively clarify intent remains underexplored, as a critical trait for robust software engineering. Existing benchmarks largely overlook this interactive bottleneck, assuming perfectly specified prompts that do not reflect the iterative nature of requirement elicitation. To bridge this gap, we introduce ClarifyCodeBench, a novel interactive benchmark for evaluating LLMs' capability in resolving requirement ambiguity. Constructed from real-world programming tasks, ClarifyCodeBench features high-quality manual annotations, including N unique ambiguity types, associated clarification questions, and corresponding ground-truth answers. Furthermore, we formalize two rigorous metrics to assess the interaction quality: Turn-discounted Key Question Rate, which penalizes inefficient questioning, and Optimal Round Adherence, which measures the precision of the elicitation process. We conduct a systematic evaluation of six state-of-the-art LLMs using ClarifyCodeBench. Our empirical results yield three critical insights: 1) Capability Decoupling: Strong code generation performance does not inherently translate to effective requirement clarification; 2) The Reasoning Paradox: While increased computational thinking enhances code correctness, it yields marginal gains in identifying ambiguities; 3) The Multi-ambiguity Ceiling: LLMs' clarification performance degrades sharply as the density of ambiguities increases, revealing a significant bottleneck in handling complex, real-world specifications. Our work underscores the necessity for future AI4SE research to transition from static synthesis to interactive elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29088v3">Diff-Based Code Corruption using LLMs for Large-Scale Bugfix Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      There are various benchmarks to evaluate bugfixing capabilities of Large Language Models. However, most widespread benchmarks do not fully reflect real-world bugfixing practices. They are small, weakening statistical reliability, and the buggy programs are often similar to one another, potentially distorting evaluation results. The range of bug types can also be narrow, failing to capture a representative range of bugs. To address these issues, we introduce MegaBugFix, a large-scale bugfixing benchmark containing 12,629 buggy Python programs synthesized from correct ones by a Large Language Model. Bug injections were generated as diffs representing code changes. Through this approach, we were able to avoid common pitfalls of LLM-based mutation techniques like injecting overly simplistic bugs or failing to modify the input program. We evaluated 13 open-weight models on MegaBugFix and baseline benchmarks, finding consistently lower performance on MegaBugFix. This reveals that our benchmark presents more challenging bugs and exposes model failures that may remain hidden when evaluating on existing benchmarks. The benchmark and fine-tuned model used for bug injection are available at hf.co/collections/szalontaib/megabugfix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05740v2">GPTKB v1.5: A Massive Knowledge Base for Exploring Factual LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 3 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      Language models are powerful artifacts, yet their factual knowledge is still poorly understood, and inaccessible to ad-hoc browsing and scalable statistical analysis. This demonstration introduces GPTKB v1.5, a densely interlinked 100-million-triple knowledge base (KB) built for $14,000 from GPT-4.1, using the GPTKB methodology for massive-recursive LLM knowledge materialization. This demo focuses on three use cases: (1) link-traversal-based LLM knowledge exploration, (2) SPARQL-based structured LLM knowledge querying, (3) comparative exploration of the strengths and weaknesses of LLM knowledge. Massive-recursive LLM knowledge materialization is a groundbreaking opportunity both for the systematic analysis of LLM knowledge, as well as for automated KB construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00692v1">Self-GC: Self-Governing Context for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents accumulate tool results, files, plans, and user constraints that are too structured to be treated as a disposable text suffix. Current systems mostly rely on in-run heuristics such as chronological pruning and tool-output masking, or on final self-summary near a context limit. Heuristics are cheap but blind to future dependencies; summaries preserve narrative state but often hide exact evidence, locators, and editable artifacts. We present Self-GC, where GC denotes self-governing context while deliberately echoing garbage collection: the system does not merely reclaim unused tokens, but governs the lifecycle of agent context objects. Self-GC turns user turns, tool spans, and skill state into indexed objects; asks a side-channel planner to propose fold, mask, and prune actions; and lets the harness enforce recoverable sidecars, safe commit boundaries, and cache-aware commit. On a 33-session Hard Set, Self-GC prunes 43.95% of prefix tokens while leaving 84.85% of future continuations unaffected, compared with no-impact rates of 54.55% to 69.70% for heuristic baselines. On a 332-session production-derived suite, three planner backbones reach no-impact rates of 91.27% to 94.58%, while baselines remain at 77.71% to 87.46%. In production, an online account-level split reduces daytime average input tokens by 10% to 15%, with peak reductions near 20%. These results point to context management as runtime lifecycle control over indexed, recoverable objects rather than post hoc text cleanup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12535v3">Ghost in the Context: Policy-Carriage Integrity in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      LLM agents choose actions from bounded decision states assembled from system policy, runtime state, tools, workload content, and the final request. We study policy-carriage integrity: applicable trusted policies must remain present, sound, and correctly bound in the decision state immediately before action. Under controlled pressure replay over AutoGen/tau3 and OpenHands/SWE-bench traces, the tested protected-placement configurations preserve policy across the pressure sweep, while task-local placement exhibits eviction, weakening, or over-budget continuation depending on the context manager. We keep this result state-level: a fixed-assembler behavioral calibration produced 0/90 unsafe-action proposals and 0/90 unguarded policy violations, so policy absence alone did not establish unsafe model behavior. The resulting design guidance is systems-level: assign typed provenance to policy state, isolate control budget, check before assembly that the complete active policy set fits, fail closed on overload, and enforce structured policies at the action boundary. We present ControlCapsule as a reference design pattern for these requirements; exact active-policy replay + preflight remains the key baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00664v1">YOMI-Bench: A Benchmark for Evaluating Kanji Reading and Phonological Understanding of LLMs for Japanese</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      We propose YOMI-Bench, a benchmark for evaluating kanji reading and phonological understanding of large language models (LLMs) for Japanese. In Japanese, a single kanji character often has multiple possible readings, making it difficult to infer the correct reading from surface-level text alone. Due to these linguistic characteristics, it is empirically known that LLMs exhibit low performance in kanji reading for Japanese. The proposed YOMI-Bench consists of four tasks specifically designed to evaluate kanji reading performance in Japanese. In our evaluation using YOMI-Bench, we assessed one multilingual open LLM, four Japanese-specific open LLMs, and five commercial LLMs. As a result, we found that even Japanese-specific models show low performance, and that commercial models also perform poorly on generation tasks that require consideration of kanji readings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26666v2">PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 7 pages, 3 tables; workshop paper
    </div>
    <details class="paper-abstract">
      Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce fragmentation, and mature kernels like FlashInfer provide highly optimized decode attention. However, the best single-kernel implementation is not always the best serving schedule: low-active long-context decode can under-utilize GPUs, while mixed sequence lengths introduce tension between many exact-length launches and coarse padded batches. We present PersistentKV, a native block-table decode attention engine and page-aware scheduling study for grouped-query attention (GQA). PersistentKV maps work by KV-head group, executes directly over native page tables, and adds a compact workqueue schedule executing only non-empty row-KV-head-sequence-split tasks. On an RTX 3060 (FP16, page size 16, Hq=32, Hkv=8, d=128), a calibrated roofline-style policy selects FlashInfer for small active batches, PersistentKV sequence splitting for batch size 1 (B1) long-context steps, and PersistentKV workqueue scheduling for supported B8 long-context GQA steps. With cost-model constants fixed on calibration traces, five held-out seeds improve mean wall decode-token throughput by 1.04x to 1.08x on B8 bimodal, uniform, and Zipf-like workloads, and by 1.40x on a B1 bucketed trace. For the B4 boundary case and uncalibrated GQA ratios, the policy avoids regressions by routing to FlashInfer. We also report an attention-plus-MLP timing proxy and workload counters showing workqueue scheduling reduces launch fan-out from 16.00 to 2.00 launches per step on held-out bimodal B8. These results show that work assignment is a decisive serving-system variable.
    </details>
</div>
