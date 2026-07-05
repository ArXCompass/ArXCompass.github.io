# llm - 2026_06

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00274v1">SEFORA: Student Essays with Feedback Corpus and LLM Feedback Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Under review for EMNLP 2026
    </div>
    <details class="paper-abstract">
      Effective writing feedback is among the strongest drivers of student learning, yet producing it at scale is labor-intensive. LLMs offer a natural path to scaling writing support, but two gaps stand in the way: few public corpora capture how instructors actually deliver feedback in real classrooms, and no reliable method measures whether generated feedback aligns with what an instructor would write. We address both. SEFORA is a public corpus pairing instructor inline feedback with assignment prompts, rubrics, scores, and multi-draft revisions across various college writing genres, comprising 564 drafts and 8,240 instructor annotations. UniMatch is a reference-based evaluation framework for open-ended generation: it segments feedback into feedback units, scores their semantic correspondence under instructor-derived criteria, and aligns them via optimal matching to yield interpretable precision, recall, and F1. Across 74 experimental configurations spanning multiple LLMs, no setting exceeds 0.4 F1. UniMatch reveals that models struggle to identify the feedback instructors would prioritize, and performance degrades as models generate more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00255v1">SLM, LLM or Agentic AI? Toward Intelligent UAV-Enabled WPT Systems in Low-Altitude Economy Networks</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) have become key enabling platforms for low-altitude economic networks, yet achieving efficient and adaptive optimization under resource-constrained and dynamic environments remains challenging. This paper investigates language models for UAV-enabled Wireless Power Transfer (WPT) systems. First, a lightweight Small Language Model (SLM)-based solution is developed using a pre-trained BERT backbone, enhanced UAV embeddings and contextual features, a geometry-aware path decoder, and ensemble inference to achieve low complexity, low latency, and high energy efficiency. Second, an Agentic AI-based framework is designed to exploit the reasoning and interactive capabilities of Large Language Models (LLMs). It integrates four collaborative agents-Initializer, Actor, Critic, and Reflector-to form a closed loop of generation, optimization, evaluation, and reflection for iterative UAV path and energy optimization. Finally, simulations compare the SLM-, LLM-, and Agentic AI-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00233v1">From Signals to Structure: How Memory Architecture Drives Language Emergence in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      How do two agents invent a shared language from scratch? In a Lewis signaling game, a sender and receiver must coordinate on a code using only their interaction history. We study five memory architectures across varying channel configurations with LLM agents and find that memory architecture matters more than channel capacity. Agents with a persistent private notebook benefit from surplus channel capacity and avoid the high-capacity collapse seen in stateless agents, achieving the most reliable coordination ($0.867 \pm 0.023$ at capacity = 25). Stateless agents peak at moderate capacity and then degrade as the vocabulary grows beyond what a rolling context window can track The notebook externalizes learned conventions, freeing agents from having to re-derive codes each round. An information bottleneck-inspired argument predicts an optimal capacity equal to the number of objects. Instead, the bottleneck (capacity = 8) proves to be a fragility point, and surplus capacity is generally better. We show that channel capacity alone cannot predict coordination; memory architecture determines whether agents turn interaction history into stable conventions, and both dimensions are needed to understand how signals become language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00158v1">Readable but Not Controllable: Neuron-Level Evidence for Medical LLM Hallucination</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Hallucination remains one of the central obstacles to deploying medical LLMs. Yet, even when hallucination can be detected, it is still unclear whether the internal representations associated with it can be used for control rather than detection alone. Using four open-source models across a suite of medical question-answering datasets, we show that a simple, carefully conditioned probe can reliably detect hallucination, with AUROC scores between 0.77 and 0.86 in our case. We further show that this signal is distributed and redundant rather than narrowly localized. Systematically selected neurons outperform random neurons only at very small subset sizes, whereas random subsets of a few hundred neurons recover nearly the full signal, and low-dimensional random projections preserve most of the detection performance. Beyond detection, we test whether this representation is causally actionable. Across 16 model--dataset combinations, our results reveal a sharp gap between decodability and controllability. The same internal structure that makes hallucination easy to detect does not translate into reliable neuron-level control. These findings show that medical hallucination seems to be readily visible in internal activations, but not easily corrected by steering the neurons most associated with it. More broadly, our results suggest that hallucination mitigation is not simply a matter of identifying the right neurons, and point to a deeper separation between what representations reveal and what they allow us to change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00151v1">SmoothAgent: Efficient Long-Horizon LLM-Based Agent Serving with Lookahead Context Engineering</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      LLM-based agents execute multi-turn workflows with continuously growing contexts, where LLM calls are interleaved with tool invocations and environment feedback. To maintain model quality, modern agent frameworks rely on context engineering strategies such as offloading, reduction, and isolation to control the context length. However, these strategies introduce significant context transformation overhead: each transformation invalidates existing KV caches and triggers re-prefill, leading to increased time-to-first-token (TTFT). In this paper, we identify that context transformations are segment-decomposable, where the transformation of a prefix is independent of future tokens. This property enables transformations to be executed ahead of time. Based on this insight, we propose a lookahead programming model that allows agent frameworks to express context transformations as asynchronous operations without modifying their execution logic. The runtime proactively executes these transformations and prepares transformed KV caches in advance, enabling direct context replacement without blocking. We further design a lookahead-aware scheduler in LLM serving systems to support these asynchronous requests alongside latency-critical workloads with controlled interference. We implement our approach to support representative context engineering strategies and integrate it into existing agent frameworks and LLM serving systems. Experiments show that our approach effectively eliminates transformation overhead and reduces TTFT by up to 11.9x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00139v1">Benchmarking Frontier LLMs on Arabic Cultural and Sociolinguistic Knowledge: A Cross-Evaluation Framework with Human SME Ground Truth</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      The cost of human expert evaluation is a principal bottleneck to deploying language models in specialized, high-stakes domains. This is particularly acute for Arabic sociolinguistic knowledge: credible grading requires not only linguistic fluency but deep cultural familiarity that cannot be approximated by surface-level metrics. We address this with a cross-evaluation framework instantiated on two underrepresented Arabic dialect communities: Egyptian and Iraqi Arabic. We contribute 103 validated prompt-rubric pairs (70 Egyptian, 33 Iraqi; 53 Cultural, 50 Linguistic), authored and graded by native-speaker SMEs using penalty-weighted rubrics distinguishing positive content requirements from answer-specific negative error criteria. Three frontier LLMs serve as target models (graded by human SMEs across 302 unique prompt-response pairs), while five frontier LLMs serve as automated judges enforcing a provider-level self-evaluation guard. A dual-metric scheme combining Mean Absolute Deviation (MAD) with Signed Mean Error separates directional grading bias from symmetric noise. Across 1,307 judge evaluations: GPT-5.4 is the most reliable judge (MADj = 10.21 pp, Signed Error = -1.12%); four of five judges show systematic leniency (+2.01% to +6.56%); Cultural tasks are harder to grade than Linguistic tasks for all judges (MAD gap 1.83-4.78 pp); and models substantially outperform on Egyptian prompts compared to Iraqi prompts. However, given leniency differences between Iraqi and Egyptian SMEs, we cannot solely attribute this gap to model knowledge. We therefore emphasize findings that do not assume identical leniency across human graders. Across all samples, implicit cultural reasoning -- requiring models to simulate native-speaker judgment rather than rely on lexical verification -- emerges as the primary failure mode for automated grading across all judge models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.32034v1">QVal: Cheaply Evaluating Dense Supervision Signals for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 10 pages, 5 figures in main text; 48 pages, 6 figures with appendix
    </div>
    <details class="paper-abstract">
      LLM agents increasingly act over long horizons, where a single trajectory can contain hundreds or thousands of actions. In these settings, outcome-only rewards provide too sparse guidance, failing to inform the model about the goodness of intermediate actions. Dense supervision methods aim to solve this problem by scoring intermediate steps, from intrinsic confidence to self-distillation and embedding similarities. However, it is common practice to evaluate them by measuring the downstream performance of a training pipeline that integrates them. This is expensive, conflates supervision quality with training engineering confounders, and renders different methodological families requiring distinct training setups incomparable. As a result, dense supervision methods are rarely benchmarked on common ground. We introduce QVal, a training-free testbed for directly evaluating dense supervision signals. Given a state-action pair, QVal measures how well a method's score is Q-aligned: whether it orders actions according to the Q-values of a strong reference-policy. This lets us compare signals before any training run and separate signal quality from other engineering choices. We instantiate QVal as QVal-v1.0, benchmarking 21 dense supervision methods across four diverse environments and seven methodological families, with over 1.2K evaluation experiments across six open-weight model backbones. We find that simple prompting baselines consistently outperform recent dense supervision methods from the literature, and that performance clusters strongly by family. These findings hold across model sizes, environments, and observation modalities. QVal is designed to be easily extensible to new environments and methods, enabling researchers to iterate on dense supervision methods before any training run.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.32032v1">Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Code: https://github.com/yale-nlp/RLMF
    </div>
    <details class="paper-abstract">
      Metacognition is a critical component of intelligence that describes the ability to monitor and regulate one's own cognitive processes. Yet LLMs exhibit systemic deficiencies in key metacognitive faculties: they hallucinate with high confidence, fail to recognize knowledge boundaries, and misrepresent their internal uncertainty--undermining trustworthiness and reliability. Since monitoring task performance and adapting behavior accordingly are central to metacognition, we posit that models capable of accurately judging their own performance are better positioned to improve it. We operationalize this idea via two novel mechanisms: reinforcement learning with metacognitive feedback (RLMF), a paradigm to refine completion rankings during preference optimization based on the quality of a model's self-judgments of performance, and metacognitive data selection, which uses similar self-judgments to identify high-value training examples, outperforming naive active learning. We apply these innovations to the problem of faithful calibration (FC), a task that is itself fundamentally metacognitive: the goal is to align expressed with intrinsic uncertainty, difficult even for frontier LLMs. We adopt a two-stage, decoupled approach, first using these methods to calibrate the faithfulness of models' self-reported confidence scores, then mapping to natural, context-adaptable linguistic uncertainty via targeted output editing. Extensive experiments show RLMF achieves generalizable, state-of-the-art FC on diverse tasks while preserving accuracy. Further, RLMF surpasses standard RL by up to 63% while enhancing models' ability to assess and express their own capability limits. This positions RLMF as a promising paradigm to enhance LLM metacognition toward improved abilities and alignment, and suggests metacognitive performance as an effective RL signal to overcome limits of prior intrinsic feedback methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.32029v1">When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 ACL 2026 (Oral)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) perform well on table tasks, they still make data referencing errors (DREs), i.e., incorrectly citing or omitting table values, despite understanding the table structure. Beyond final-answer accuracy, DREs directly compromise the correctness and reliability of intermediate reasoning steps. Yet prior studies have only offered limited, small-scale analyses. In this work, we present the first systematic evaluation of tabular data referencing errors across different models and tasks. Our results show that DREs occur across all tested models (1.7B to 20B parameters). Furthermore, we demonstrate that incorporating data referencing as a critic significantly improves answer accuracy up to 12.0%, through critic-based filtering and rejection sampling. Finally, we trained a lightweight 4B-parameter critic model that achieves an average F1 score of 78.2% in detecting both in-distribution and out-of-distribution DREs, and effectively assists inference for larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.32025v1">Generative Skill Composition for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Recent LLM agents benefit from skills for solving complex tasks. Skills encapsulate modular packages of procedural knowledge and instructions for performing specialized tasks, such as setting up a sandboxed environment, running a test suite, or refactoring a function across multiple files. As skill libraries grow and become reusable across tasks and domains, selecting an appropriate skill composition has emerged as a central bottleneck. Existing approaches fall into two categories. One exposes the agent's reasoning to the entire skill collection; the other performs skill retrieval via embeddings or LLM-based rerankers. Both provide useful insights; however, they miss the structural nature of skill composition, which is a joint decision over which skills, how many, and in what order -- three dimensions that cannot be decoupled. We formalize this as structured skill composition: given a task and a skill library, predict an executable skill plan that jointly specifies the activated subset, count, and execution order. We propose SkillComposer, which instantiates structured skill composition as task-conditioned skill sequence prediction. SkillComposer uses a constrained autoregressive decoder over skill identifiers, so subset, count, and order emerge jointly from a single decoding pass, and dependencies between successive skills are captured naturally. We build a training set of task-composition pairs from a real, human-curated skill library. We then evaluate SkillComposer along two axes: composition quality on a held-out test set, and downstream task success on SkillsBench across two production-grade coding agents. On GPT-5.2-Codex, Gemini-3-Pro-Preview, SkillComposer raises the pass rate by +23.1, +18.2pp over the no-skill baseline, surpassing top-3 retrieval and matching the gold-skill retrieval upper bound at lower prompt-token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06625v2">FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Existing LLM-as-a-Judge systems suffer from three fundamental limitations: limited adaptivity to task- and domain-specific evaluation criteria, systematic biases driven by non-semantic cues such as position, length, format, and model provenance, and evaluation inconsistency that leads to contradictory judgments across different evaluation modes (e.g., pointwise versus pairwise). To address these issues, we propose FairJudge, an adaptive, debiased, and consistent LLM-as-a-Judge. Unlike prior approaches that treat the judge as a static evaluator, FairJudge models judging behavior itself as a learnable and regularized policy. From a data-centric perspective, we construct a high-information-density judging dataset that explicitly injects supervision signals aligned with evaluation behavior. Building on this dataset, we adopt a curriculum-style SFT-DPO-GRPO training paradigm that progressively aligns rubric adherence, bias mitigation, and cross-mode consistency, while avoiding catastrophic forgetting. Experimental results on multiple internal and public benchmarks show that FairJudge consistently improves agreement and F1, reduces non-semantic biases, and outperforms substantially larger instruction-tuned LLMs. All resources will be publicly released after acceptance to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14732v2">LLM-as-a-judge validity in physics assessment depends more on the task than the model</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 29 pages, 28 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly considered for automated assessment and feedback, understanding when LLM marking is valid is essential. We evaluate LLM-as-a-judge marking across three physics assessment formats - structured questions, written essays, and scientific plots - comparing GPT-5.2, Grok 4.1, Claude Opus 4.5, DeepSeek-V3.2, Gemini Pro 3, and committee aggregations against human markers under blind, solution-provided, false-solution, and anchored conditions. We distinguish absolute accuracy from rank-order agreement, since a marking system can match the distribution of human marks while failing to order responses by quality. Across task types, performance is sharply task-dependent. For blind university exam questions ($n=771$) and secondary and university structured questions ($n=1151$), models show robust rank-order agreement with human markers (Spearman $ρ> 0.6$), with official solutions reducing error and strengthening agreement. False solutions degrade absolute accuracy, showing that models defer to provided references, but leave rank-ordering intact. Essay marking behaves fundamentally differently. Across $n=55$ scripts ($n=275$ essays), blind AI marking is harsher and more variable than human marking and adding a mark scheme does not improve rank-order agreement. Anchored exemplars shift the AI mean close to the human mean and compress variance below the human standard deviation, but rank-order agreement remains near-zero. For code-based plot elements ($n=1400$), models achieve high rank-order agreement ($ρ> 0.84$) with near-linear calibration. Across all task types, validity tracks the structure of the assessment task - the extent to which marks can be mapped to explicit, observable grading features - and the reliability of the human benchmark, rather than raw model capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31916v1">Theory of Mind and Persuasion Beyond Conversation: Assessing the Capacity of LLMs to Induce Belief States via Planning and Action</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 29 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) benchmarks for Large Language Models (LLMs) typically rely on passive question-answering formats, but the deployment of LLMs in increasingly agentic and autonomous forms demands new evaluations. In this paper we evaluate an agent's ability to induce specific belief states in other agents by taking actions rather than using conversational persuasion, a capability we call Non-Conversational Planning ToM (NCP-ToM). NCP-ToM is likely to be essential for many agent use-cases, including within user-assistant interactions and pedagogical contexts, but may also present manipulation or misinformation risks. Using a novel framework, NCP-ExploreToM, we subvert the conventional task structure by providing models with a set of belief state goals and requiring them to move objects or direct characters into rooms to achieve their goals. We evaluated six frontier models, including GPT-5, Gemini 2.5 Pro and the Claude 4 series, and a cohort of human participants, across 600 task instances. GPT-5 was successful on approximately 80% of tasks in the agentic setting, and was the only model to outperform human participants on our task, but was still less robust than humans across contexts. We additionally found that all models, like humans, performed better on tasks inducing true belief states than false belief states, which is a positive signal for alignment efforts. These findings highlight emerging social-reasoning capabilities in LLMs for non-conversational task completion and underscore the necessity of agentic evaluations for understanding the safety and alignment of autonomous social agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30560v2">TraceLab: Characterizing Coding Agent Workloads for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Coding agents are rapidly becoming a major application of agentic LLMs, but serving them efficiently remains challenging. Progress on this challenge requires understanding real workload patterns, yet the data needed for such analysis is largely absent. Existing public traces and benchmarks do not capture real, day-to-day coding-agent usage across multiple agents and model families for serving-system analysis. To help fill this gap, we collect and release a trace of roughly 4,300 coding-agent sessions, containing about 350,000 LLM steps and 430,000 tool calls from our own day-to-day use of Claude Code and Codex. Our analysis shows that coding-agent workloads feature long autonomous loops, long contexts with short outputs, diverse and heavily-tailed tool calls, and high but imperfect prefix cache hit rates. These findings point to concrete opportunities for optimizing serving, including lower-overhead tool calling, append-length-aware prefill, semantic-aware tool-latency prediction, and improved KV-cache management around human-paced gaps. We release the dataset, trace collection pipeline, and analysis code at https://github.com/uw-syfi/TraceLab.git the project website is https://tracelab.cs.washington.edu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31801v1">RAISE: LLM-based Automated Heuristic Design with Robust Adversary Instance Search</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Automated Heuristic Design (AHD) with Large Language Models (LLMs) has shown remarkable progress in discovering high-quality heuristics. However, existing LLM-based AHD methods optimize heuristics for a fixed training instance set and may fail catastrophically when deployed under real-world distributional shifts. We propose Robust Adversary Instance Search (RAISE), a framework that integrates constrained worst-case instance search within a principled neighborhood of the training distribution into the LLM-based evolutionary search loop. RAISE treats robust AHD as a constrained adversarial instance search problem: the outer loop evolves heuristics via LLM operators, while an LLM-free inner loop efficiently identifies hard instances within an epsilon-ball around the training instance set using a basis distribution parameterization with boundary projection. Comprehensive experiments on Online Bin Packing (OBP), Online Job Shop Scheduling (OJSP), and Online Vehicle Routing (OVRP) across five distribution families demonstrate that existing LLM-based AHD methods degrade by up to 19 times under distribution shift, while RAISE consistently maintains strong performance across all tested distributions and problem scales
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00186v2">How to Compare the Security of Code Written by Humans to LLM-generated Code</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly transforming how software is created and maintained. Comparing LLM-generated code against human-written standards is essential to determine whether these new tools uphold or erode the security baselines established by professional developers. Yet, we lack a standardized method for empirically comparing the security of code produced through human-LLM collaboration against LLM-only, or traditional human-only methods. To facilitate this, we propose an automated framework for conducting comparative studies across human-only, LLM-only, and hybrid conditions. Our approach automates the logging of prompts, timing, and experimental settings, measuring outcomes through multi-dimensional static and dynamic quality analysis. We provide an open-source implementation of this framework to ensure that future researchers can conduct reproducible, species-fair experiments. Importantly, we validate the framework via a feasibility study, providing an experimental blueprint for ``species-fair'' comparisons between human and AI subjects. By sharing lessons learned, we establish a foundation for empirical research on human and LLM-generated code for software security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31762v1">Investigating LLM-Powered Dissenting Minority Support in Power-Imbalanced Group Decision-Making: Counterargument and Mediation as Intervention Strategies</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted at CSCW 2026
    </div>
    <details class="paper-abstract">
      Minority viewpoints are often suppressed in power-imbalanced group decision-making due to social pressure to comply with the majority. To address this problem, we developed an LLM-powered dissenting minority support system that aimed to foster attention to minority views through either AI-generated counterarguments or AI-mediated messages. We conducted a mixed-method experiment with 96 participants in 24 groups, comparing minority members' experiences across baseline, AI-counterargument, and AI-mediated message conditions. Our findings revealed a nuanced trade-off: AI-generated counterarguments fostered a more flexible atmosphere and enhanced satisfaction, while AI-mediated messaging increased minority participation but unexpectedly reduced their psychological safety. This research contributes empirical evidence on how different AI implementations affect group dynamics, identifies a critical support paradox between participation and psychological safety, provides design implications for future systems, and highlights ethical challenges in implementing AI-mediated communication in hierarchical settings. These insights advance understanding of designing more equitable AI support for power-imbalanced group decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31725v1">Do Machines Struggle Where Humans Do? LLM and Human Comprehension of Obfuscated Code</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 13 pages, 15 figures
    </div>
    <details class="paper-abstract">
      While code obfuscation impairs human code comprehension, it remains unclear if large language models share these failure modes. Building directly on a recent human study of program comprehension under code obfuscation, we evaluate whether large language models share the failure modes that obfuscation induces in human programmers. Evaluating several LLMs with five obfuscation tiers using the Block Model, we localize comprehension failures at the atom, block, relational, and macro levels. We find that reasoning-tuned models demonstrate significant alignment with human difficulty patterns across experience levels, whereas instruction and coder-tuned models show near-zero correlation. Chain-of-Thought trace length tracks task difficulty across tasks. Results indicate that performance under control-flow flattening degrades in proportion to state-space complexity, while adversarial identifier renaming disrupts comprehension through the interaction of semantic displacement and identifier-level interference. These findings suggest that reasoning-tuned LLMs approximate human sensitivity to code complexity more effectively than instruction-tuned variants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23672v2">Teaching LLMs String Matching, Backtracking, and Error Recovery to Deduce Bases and Truth Tables for the Combinatorially Exploding Bit Manipulation Puzzles</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 22 pages, 4 figures, 2 tables. 7th Place Solution for the NVIDIA Nemotron Model Reasoning Challenge (Kaggle)
    </div>
    <details class="paper-abstract">
      This paper presents our algorithmic innovations for the NVIDIA Nemotron Model Reasoning Challenge, focusing on Bit Manipulation Puzzles. In this task, the objective is to discover a hidden logical rule transforming input binary strings to outputs, then apply it to unseen inputs. Large Language Models (LLMs) notoriously struggle here; traditional methods force them to simulate complex boolean logic and arithmetic, leading to hallucinations. Furthermore, the search space of bitwise operations (combinations of shifts, rotations, and logic gates) suffers from a severe combinatorial explosion. To overcome this computational intractability, we present a novel approach that abandons arithmetic logic entirely in favor of string similarity, structured search, and autonomous error recovery. Our core contributions are: 1. Bases and Truth Table Formulation: We reframe logic-gate deduction into a base-selection task, leveraging string similarity (minimal bit flips) to isolate primitive transformations ("bases") and deduce truth tables without complex arithmetic. 2. Backtracking DFS and Error Recovery: We formalize a search process that tests candidate bases, detects logical collisions across examples, and backtracks upon failure to perform robust error recovery. 3. Bit Tokenization and Interactive Reasoning SFT: We force the tokenizer to encode binary strings as individual single-bit tokens. We use dynamic masking to simulate external oracle feedback, training the model to hypothesize, self-evaluate, and backtrack natively. Evaluated on bit manipulation puzzles, our approach achieved over 96% validation accuracy. This represents the highest performance in this category, driving our 7th Place overall finish in the contest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28183v3">BenGER: Benchmarking LLM Systems on Subsumption-Based Legal Reasoning in German Law</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Pre-Print
    </div>
    <details class="paper-abstract">
      We introduce BenGER (Benchmark for German Law), a benchmark and dataset for evaluating LLM systems on subsumption-based legal reasoning in German law. The dataset combines 596 exam-style free-text legal case tasks across multiple levels of legal education and 531 short doctrinal reasoning tasks. It includes a controlled validation subset of timed human-written solutions under both unaided and human-AI co-creation conditions. We evaluate 12 contemporary LLM systems - closed flagship, efficiency-oriented, and open-weight - with a rubric-aligned LLM-as-a-Judge cross-validated against a multi-rater human-grading layer (three blind reviews per solution, six judge families benchmarked against the human pool). Closed-flagship systems lead the leaderboard across all three corpora, human-AI co-creation measurably improves on unaided human work, and the LLM judge tracks human grading at Pearson r=0.76 and Cohen's \k{appa}=0.60. System rankings are stable across judge families and two judges from independent providers clear the Calderon single-reviewer replacement bar on human-authored solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31635v1">A Tutorial on Autonomous Fault-Tolerant Control Using Knowledge-Grounded LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Fault recovery in process plants still relies heavily on plant operators, especially when faults fall outside predefined supervisory logic. Operators interpret alarms, procedures, P\&IDs, interlocks, and process trends, then decide how to move the plant to a safe operating mode without triggering a shutdown. This paper examines how Large Language Model (LLM) agents can support such recovery decisions. The proposed framework treats the LLM as a constrained supervisory planner. It uses plant-specific knowledge to propose recovery actions, and every proposal is checked by an external validator (symbolic or simulation-based) before actuation. The paper develops three design dimensions for applying the framework: the recovery patterns for which LLM agents are useful, the validation strategies that separate admissible from inadmissible proposals, and the deployment constraints imposed by latency, knowledge engineering, safety integration, and model lifecycle management. To make the framework directly usable, two openly available executable Python environments are provided. Both re-implement established case studies, a modular mixing module and a continuous stirred-tank reactor, extended with configurable faults and defined interfaces for custom recovery and validation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22723v2">BLUEX v2: Benchmarking LLMs on Open-Ended Questions from Brazilian University Entrance Exams</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 15 pages, 4 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) excel in many tasks, their assessment in Portuguese has received less attention, particularly for open-ended, discursive tasks that demand deeper reasoning and generation capabilities. While the original BLUEX benchmark addressed the scarcity of Portuguese evaluation datasets through multiple-choice questions from Brazilian university entrance exams, it did not cover the more challenging second-phase examinations, which require free-form written responses. In this work, we introduce BLUEX v2, a benchmark derived from the second-phase entrance exams of Brazil's two leading universities: UNICAMP (Comvest) and USP (Fuvest), spanning exam years 2022--2025. Our dataset comprises 395 questions unfolding into 919 graded subquestions, with 55.7% of questions containing associated images (represented as context-aware captions during inference to enable evaluation across both vision-capable and text-only models). Each question is annotated with subject area, official reference answers, LLM-generated rubric criteria, and six cognitive capability tags. We evaluate 21 state-of-the-art LLMs using an LLM-as-a-judge protocol. Results reveal a 4.92-point performance spread across models (4.18-9.10 on a 0-10 scale), with Mathematical Reasoning and Image Understanding emerging as the hardest capability dimensions. The evaluation code, model outputs, and dataset are publicly available at https://github.com/TropicAI-Research/BLUEXv2 and on Hugging Face at https://huggingface.co/datasets/Tropic-AI/BLUEX-v2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31608v1">CLExEval: A Human-in-the-Loop Framework for Qualitative Evaluation of LLM Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 21 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong results on many medical benchmarks, but their clinical reasoning remains difficult to evaluate reliably. A central risk is an evaluation illusion: fluent and well-structured explanations can appear clinically convincing even when the final diagnosis is incorrect. We introduce CLExEval, a human-in-the-loop framework for evaluating LLM clinical reasoning under progressive information masking. CLExEval combines 5,600 expert-physician annotations with 200 clinical reasoning traces derived from 40 rare diagnostic cases. Our analysis identifies three recurring failure patterns: (i) verbosity bias, where GPT-4o-mini's diagnostic accuracy drops from 95.0% to 32.5% under information scarcity; (ii) a hidden knowledge paradox, where a specialist model reaches 92.5% maximum diagnostic potential but fails to retrieve that knowledge reliably in verbose contexts; and (iii) a 68.6% reasoning-to-output mismatch, where correct diagnoses appear in reasoning traces but are not reflected in final answers. We further evaluate the LLM-as-a-Judge paradigm on a human-verified failure set (n = 142). GPT-4o-mini approved 47.9% of clinically incorrect outputs, while HuatuoGPT-o1 approved all validly scored failures and showed a positive self-preference bias. These results suggest that standalone automated clinical evaluations can substantially overestimate clinical reliability without expert-grounded validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23375v2">Measuring & Mitigating Over-Alignment for LLMs in Multilingual Criminal Law Courts</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      While the wider applicability of LLMs in the legal field is currently debated due to their reliability and the gravity of any errors, narrow uses with well-understood and mitigated risks have emerged. Notably the Swiss Federal Supreme Court uses small on-premises models for tentative translations and short-passage summarization across the four official languages. However, such usage is challenging in the context of Criminal Law. Since rulings and cases employees work on routinely can contain detailed descriptions of violent and sexual offenses, their legitimate work is compromised by refusals and disclaimers due to the activation of model guardrails (over-alignment). To measure this phenomenon, we introduce TF-RefusalBench, a multilingual benchmark for criminal-law translation and summarization derived from public Swiss Supreme Court rulings. TF-RefusalBench contains 5,200 total prompts across French, German, Italian, and English, corresponding to common task prompts and passages likely to trigger refusal. We then use TF-RefusalBench to show that over-alignment is a multifaceted phenomenon, influenced by the model and the prompt and text languages being processed, and that its impact cannot be evaluated solely from an over-refusal perspective, given the disclaimer's impact on task faithfulness. Finally, we evaluate approaches to enable on-premises LLMs for Criminal Law Tasks, demonstrating that while prompting can be effective, abliteration (refusal directions ablation) eliminates refusal with minimal impact on task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31524v1">On the Convergence of Self-Improving Online LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted at UAI 2026
    </div>
    <details class="paper-abstract">
      The Self-Improving Alignment (SAIL) algorithm addresses distribution shift by reducing a bilevel formulation of the problem to an efficient, single-level method. Empirically, SAIL has demonstrated strong performance on this task. However, a formal analysis of its convergence properties has been lacking. We identify a key theoretical challenge: the standard SAIL objective function is not guaranteed to be strongly concave due to unfavorable properties of its Hessian. To address this limitation, we propose a regularized objective, SAIL-RevKL, which incorporates a reverse Kullback-Leibler (KL) divergence penalty to improve the optimization landscape. Our central theoretical contribution is to prove that this regularized objective satisfies the Polyak-Lojasiewicz (PL) condition within a bounded parameter space. We establish global convergence guarantees, achieving a near-linear sample complexity. We further validate the effectiveness and stability of SAIL-RevKL through empirical evaluations, demonstrating that it outperforms the vanilla SAIL on both MuJoCo benchmarks and LLM alignment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29672v2">How LLMs See Creativity: Zero-Shot Scoring of Visual Creativity with Interpretable Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 21 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Evaluating the originality of visual images poses enduring challenges for creativity assessment. Automated scoring using AI models has proven effective in the verbal domain, yet key questions remain about evaluating visual creativity and understanding how models arrive at their ratings. The present research asks whether multimodal large language models (LLMs) can serve as judges of visual creativity zero-shot (without any fine-tuning or examples of human ratings) and whether their "reasoning" output offers an interpretable window into their evaluation process. We tested six multimodal LLMs (Gemini 3 Flash, Gemma 4 31B IT, GPT-5.4 Mini, GLM-5v Turbo, Kimi K2.5, and Qwen 3.6 Plus) on 992 AI-generated images (based on human-written prompts) and 1,500 hand-drawn sketches scored for creativity by human raters. In Study 1, all models showed substantial alignment with human creativity ratings on both datasets (r = .57-.68 on AI-generated images; r = .29-68 on sketches). In Study 2, we analyzed the step-by-step reasoning processes of three LLMs evaluating the same images and drawings. Although reasoning made model evaluations interpretable -- showing what they attend to, how they balance originality vs. quality, and how they justify their ratings -- reasoning did not improve alignment with human ratings. In sum, our findings indicate that multimodal LLMs can match human judgments of visual creativity without any additional training, and that their reasoning reveals how AI models evaluate creativity. An open scoring app implementing this pipeline is available at https://review-visual-eval-scoring.hf.space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26883v2">EconSimulacra: A Digital Twin Platform of Socio-Economic Systems Powered by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Real-world social behavior emerges from tightly coupled domains: economic conditions shape mobility and social interactions, while online attention and offline activity feed back into local popularity and consumer behavior. Capturing these feedback loops requires artificial societies in which agents carry experiences from one domain into decisions in another. Large language models (LLMs) provide a promising foundation for such societies. However, existing LLM-based simulators typically model domains in isolation or merely place them side by side. To enable such cross-domain interactions, we present EconSimulacra, a multi-agent social simulator that couples consumer economy, mobility, and social networks through a shared internal-state mechanism. In EconSimulacra, experiences accumulated across different domains are stored in memory and transformed into shared internal states (i.e., stress level) connecting heterogeneous domains through individual decision making. This design allows agents to reconcile competing demands arising from multiple domains and generate coherent cross-domain behaviors. As a case study, we show that the shared internal state mechanisms reproduce a nonlinear relationship between online social attention and offline local popularity, illustrating how realistic cross-domain dynamics can emerge within a unified artificial society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23032v3">IPO Finance Agent: Benchmark of LLM Financial Analysts Beyond Finance Agent v2, with Automated Rubric Generation, on the SpaceX (SPCX) IPO</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Finance Agent v2 (by Vals AI) has emerged as the reference benchmark for evaluating both Anthropic Claude and OpenAI ChatGPT frontier language models on financial tasks. However, it narrowly deals with periodic reporting from publicly traded companies (SEC 10-K and 10-Q filings), and its agentic harness relies on naive, unenriched chunk retrieval. Neither the task design nor the retrieval approach addresses the distinct challenges of IPO due diligence. SEC S-1 filings combine historical financial statements, governance structures, pro forma and common-control accounting treatments, capital-formation narratives, and underwriting-sensitive risk disclosures within substantially longer documents than typical periodic filings. That is why we introduce IPO Finance Agent, which extends the Finance Agent v2 framework along two directions: task domain and retrieval architecture. During our experiments, the original Finance Agent v2 harness basically failed to deliver any output related to the SpaceX S-1 filing, due to document length. We therefore had to improve the agentic harness with contextual retrieval, a more realistic and industry-standard approach for long documents. We also built a dataset of 1,000 IPO-diligence questions, and publicly release 70 questions on the SpaceX (SPCX) S-1 filing to support reproducibility, while the remainder are held private to guard against benchmark contamination. In addition, we introduce an evaluator-optimizer pipeline to automatically generate evaluation rubrics for the benchmark: candidate facts are extracted from model answers, consolidated into draft criteria, then automatically audited for omissions, hallucinations, mistiered items, and redundancy, with LLM feedback driving iterative repair, targeted enrichment, and deduplication. Human experts only review final rubrics before deployment. Results show that the best-performing evaluated model, Zhipu GLM-5.2, reaches 79.8% accuracy, and the most cost-efficient model on the resulting Pareto frontier, Xiaomi MiMo-2.5 Pro, reaches slightly lower accuracy (77.2%) at 0.05 USD per query, while exceeding the current Finance Agent v2 leaderboard ceiling, Google Gemini 3.5 Flash at 57.9% for 2.51 USD per query, and undercutting even FABv2's cheapest entry (MiniMax M3: 48.3% at 0.32 USD) on cost-efficiency. Code and data are released on GitHub https://github.com/benstaf/ipoagent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19453v3">Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted to NExT-Game 2026: New Frontiers in Game-Theoretic Learning, ICML 2026 Workshop. Camera-ready version
    </div>
    <details class="paper-abstract">
      We propose an LLM harness that generates code-based policy functions for multi-agent environments, evaluates them with self-play, and refines them using feedback from previous iterations. Following the recent line of work in feedback engineering (the design of which information signals are shown to the LLM during refinement), we compare sparse feedback (scalar reward only) with dense feedback (reward plus social metrics: efficiency, equality, sustainability, peace). In two Sequential Social Dilemmas (Gathering and Cleanup) and with two frontier LLMs (Claude Sonnet 4.6, Gemini 3.1 Pro), dense feedback improves over or matches sparse feedback on all metrics. We explain this asymmetry via feedback aliasing: when the scalar reward maps distinct failure modes into the same value (e.g., under- vs. over-cleaning), social metrics disambiguate and allow the LLM to diagnose which direction of improvement to take. We conclude that social metrics act as a coordination signal, leading to strategies such as Voronoi territory partitioning and adaptive cleaner schedules. Code at https://github.com/vicgalle/llm-policies-social-dilemmas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31371v1">Calibrating the Evaluator: Does Probability Calibration Mitigate Preference Coupling in LLM Agent Feedback Loops?</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 7 pages, 2 tables
    </div>
    <details class="paper-abstract">
      When large language model (LLM) agents adapt their behavior through evaluator feedback, systematic evaluator biases propagate into the agent's learned strategy distribution - a phenomenon termed evaluator preference coupling. Prior work has documented this coupling and established a diagnostic framework (EPC) to measure it, but has not investigated whether calibration techniques can mitigate the effect. We present the first study of evaluator calibration as mitigation: applying probability calibration to the evaluator's pairwise judgments to reduce spurious preference propagation. In a controlled within-subjects experiment (N=5) comparing standard binary TTRL (win/loss) with confidence-calibrated TTRL (probability-weighted updates) using DeepSeek-V4-Pro as executor and GLM5.2 as evaluator, we find that calibration reduces the coupling coefficient gamma by 20-49% and Jensen-Shannon divergence by 45-67%. A symmetric-LR control confirms the effect is not due to reduced update asymmetry. We release the calibrated TTRL protocol and recommend it as a lightweight mitigation for LLM-as-judge deployment pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31368v1">MOA: A Profiling-Guided LLM Framework for Memory-Optimization Automation at Codebase Scale</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Modern large-scale software systems often suffer from pervasive memory inefficiencies (e.g., bloat, churn), leading to excessive resource costs and performance degradation. Existing optimization workflows lack end-to-end automation, forcing developers to manually synthesize complex tool outputs into actionable and semantics-preserving fixes, precluding scalability in large codebases. To address this, this paper presents MOA, an LLM-driven framework that automatically detects and repairs recurring memory inefficiencies across production-scale codebases. Specifically, MOA operates through three agents: an Analyzer that mines anti-patterns from profiling data, a Checker Generator that synthesizes static analyzers through template-guided refinement, and a Patcher that generates optimization patches via state-machine-driven workflows. Our evaluation on OpenHarmony, an open-source operating system with over 100 million lines of C/C++ code, shows that MOA identifies 13 anti-patterns (9 previously unknown) from 3 profiled services, detects over 10,000 inefficiencies across a broader set of 7 services, and generates 769 patches with 92.5% expert acceptance rate, achieving 42.2% heap reduction and 10.6% binary size reduction on average. We envision MOA as a valuable tool for performance engineering at production scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10409v3">Dataset Construction for Training LLM to Learn Analog Circuit Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      This paper constructs a textual dataset for training large language models (LLMs) to learn analog circuit knowledge and customizes LLM training techniques. For dataset construction, high-quality textbooks are collected and decomposed into fine-grained learning nodes, which are then used to construct structured question-thinking-solution-answer (QTSA) quadruples using a multi-agent framework to capture both final answers and thought processes. The resulting dataset consists of 7.26M tokens of unlabeled data for continual pre-training (CPT) and 112.65M tokens of labeled data for supervised fine-tuning (SFT). We customize the training techniques including initial model selection, training paradigms, regularization techniques, and practical implementation references. Instruct models are identified as suitable training initialization points, an SFT-centric training paradigm is established (finding that CPT provides marginal benefits compared with SFT due to imbalanced data distribution), and SFT with KL divergence regularization can achieve a 2.71 percentage-point improvement over SFT alone. A practical training implementation method is provided for resource-constrained scenarios. Experiments demonstrate that the dataset and training techniques enhance LLMs' analog circuit knowledge. The trained 32B instruct model achieves 84.59% accuracy on the AMSBench-TQA benchmark, showing a 15.67 percentage-point improvement over the initial model. The trained model also shows capability in the operational amplifier design task based on the Atelier framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31309v1">CSO-LLM: Class Subspace Orthogonalization for Post-Training Backdoor Detection and Trigger Inversion in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      While post-training backdoor detection and trigger inversion schemes have been developed for AIs used e.g. for images, there is a paucity of such methods for LLMs. First, the LLM input space is discrete, with up to 150,000^k k-tuples to consider with k the token-length of a putative trigger. Second, one must blacklist tokens typical of the putative target response (class) of an attack, as such tokens may give false detection signals. However, a comprehensive blacklist is not available, in general, for a given domain. We develop a highly effective detection and inversion framework for LLMs treated as classifiers. Central to our approach is class subspace orthogonalization (CSO), a novel plug-and-play paradigm for backdoor detection that serves two fundamental roles when applied to LLMs: i) it enhances both sensitivity and specificity of a baseline detector; ii) it provides a form of implicit blacklisting, as it penalizes against inclusion, in a candidate trigger, of tokens that induce signal perturbations "in the direction of" the putative target class of an attack. One version of our detector performs continuous optimization in token embedding space, while a companion trigger-inversion and detection method performs greedy accretion in discrete token space. Our methods give both strong detection performance and accurate inversion of ground-truth triggers on several LLM classification domains, and for several different LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17188v4">LLM-Aided Joint Secrecy Precoding and Trajectory for RSMA-Based Heterogeneous UAV Networks</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      This paper investigates secure communications in rate-splitting multiple access (RSMA) enabled heterogeneous UAV networks, where multiple UAVs collaboratively serve ground terminals in the presence of eavesdroppers. By jointly considering secrecy rate maximization and propulsion energy consumption minimization, we formulate a multi-objective optimization problem involving UAV trajectory design, service association, power allocation, and secrecy precoding under mobility, collision-avoidance, service-capacity, and communication constraints. The formulated problem is highly non-convex due to the coupling among UAV trajectories, RSMA transmission variables, and secrecy constraints. To address the resulting non-convex and highly coupled optimization problem, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (D.C.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates LLM-generated expert heuristic policy, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31252v1">Embodied CAD: Solver-Grounded LLM Agents for Parametric B-Rep Assembly Modeling</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 This paper contains 12 pages, 7 figures. This is an original unpublished manuscript submitted to the arXiv preprint server, with no prior publication or conference presentation
    </div>
    <details class="paper-abstract">
      Large language models can write plausible CAD scripts, but reliable industrial CAD modeling requires more than syntactically valid code: every feature, placement, and assembly relation must be accepted by an exact geometric kernel while remaining editable as parametric boundary representation geometry. We present Embodied CAD, solver-grounded LLM agents for parametric B-Rep assembly modeling. Instead of generating a complete script in one pass, the agent iteratively selects actions from a stratified L0-L4 CAD skill library, resolves them into typed geometric operations, executes them in a CAD backend, and uses solver feedback to plan, repair, and learn. The framework combines action grammar constraints, deterministic parameter resolution, and solver-derived rewards for supervised warm-up and GRPO-style refinement. We evaluate Embodied CAD on multi-step mechanical, industrial equipment, and mold-oriented assembly tasks using solver-aligned metrics: executable rate, skill accuracy, operation-family accuracy, exact policy accuracy, and task completion success. The results show that solver-grounded planning executes all strong-planner workflows in the current benchmark, while learned controllers reach high executable rates and expose the remaining gap between valid tool calls and exact long-horizon policy prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20653v3">SysVCoder: An LLM-Driven Framework for Systematic Generation of System-Level Design</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 This paper is accepted at APPT'26
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong potential in generating hardware designs using hardware description languages (HDLs) such as Verilog. However, existing LLM-based frameworks struggle to accurately capture the complexity of real-world architectural designs, particularly for large-scale systems with hierarchical, multi-level module instantiations. To address this issue, we present SysVCoder, an LLM-driven framework that enhances both the generation quality and efficiency of system-level design in Verilog. SysVCoder introduces a two-stage generation pipeline that leverages an intermediate representation to enable a more structured and accurate translation from natural language specifications to complex multi-module designs. Furthermore, we incorporate a rule-based alignment mechanism and a domain-specific retrieval-augmented generation strategy (DS-RAG) to enhance functional correctness by grounding LLM outputs in domain knowledge. We also present SysVDB, a comprehensive dataset comprising 60 system-level hardware designs along with their corresponding verification testbenches. Experimental results demonstrate that SysVCoder outperforms state-of-the-art frameworks such as CodeV and VeriGen by 30.7% and 38.3% in terms of functional correctness under the same base LLM. Notably, SysVCoder achieves performance comparable to NVIDIA's GPT-4 based VerilogCoder while using only a 7B-parameter model, reducing token consumption by 7.6x and synthesis latency by 37.5x. Both SysVCoder and SysVDB are made public at https://gitee.com/sdu-aes-lab/sysvcoder/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31213v1">Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 "23 pages. Preprint
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as moral advisors and agents, they need to address dilemmas between two competing values. However, existing research on LLMs with moral dilemmas overlooks a central aspect of human moral cognition: the ability to imagine alternatives that move beyond the given options. We introduce MoralAltDataset, a dataset of 307 moral dilemmas spanning narrative Advisor dilemmas and AI-facing Agent dilemmas, each augmented with compromise and reframed alternatives. We first examine whether humans and LLMs shift their judgments when such alternatives are introduced. Across 15 LLMs, we find that compromise alternatives are often preferred over either original option, substantially reshaping moral choice. We then evaluate the quality of LLM-generated alternatives against human-authored ones using pairwise preference and expert-based criteria. Results show that LLM-generated alternatives are often preferred and better satisfy fine-grained structural and ethical criteria, while revealing trade-offs between structural quality and practical feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10895v2">LLM-Empowered Agentic MAC Protocols: A Dynamic Stackelberg Game Approach</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 This work has been submitted to IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Medium Access Control (MAC) protocols, essential for wireless networks, are typically manually configured. While deep reinforcement learning (DRL)-based protocols enhance task-specified network performance, they suffer from poor generalizability and resilience, demanding costly retraining to adapt to dynamic environments. To overcome this limitation, we introduce a game-theoretic LLM-empowered multi-agent DRL (MARL) framework, in which the uplink transmission between a base station and a varying number of user equipments is modeled as a dynamic multi-follower Stackelberg game (MFSG), capturing the network's natural hierarchical structure. Within this game, LLM-driven agents, coordinated through proximal policy optimization (PPO), synthesize adaptive, semantic MAC protocols in response to network dynamics. Protocol action grammar (PAG) is employed to ensure the reliability and efficiency of this process. Under this system, we further analyze the existence and convergence behavior in terms of a Stackelberg equilibrium by studying the learning dynamics of LLM-empowered unified policies in response to changing followers. Simulations corroborate that our framework achieves a 77.6% greater throughput and a 65.2% fairness improvement over conventional baselines. Besides, our framework generalizes excellently to a fluctuating number of users without requiring retraining or architectural changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22686v2">The Geometry of Refusal: Linear Instability in Safety-Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted at TrustNLP 2026 (Sixth Workshop on Trustworthy Natural Language Processing, ACL 2026)
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) rely on extensive safety alignment, yet the mechanistic basis of refusal remains opaque. In this work, we investigate whether safety compliance is a deep semantic decision or a manipulable linear feature. We introduce Contrastive Logit Steering (CLS), a zero-optimization framework that isolates the "refusal direction" by contrasting hidden states derived from safe and unrestricted system prompts. Unlike representation engineering methods that intervene on internal activations, CLS operates directly on the output distribution, serving as a diagnostic probe for alignment fragility. When coupled with prefix injection to bypass initial refusal reflexes, this method induces a phase transition where guardrails collapse. Our experiments on 7 model families reveal that safety implementation is architecturally deterministic. While models like Llama-3.1 exhibit a "Late Decision" topology that is easily bypassed by CLS (reaching 95% ASR in approximately one second), others like Qwen-2.5 demonstrate "Early Divergence" by integrating safety mid-computation. Direct comparison with established activation-level steering methods shows that CLS achieves substantially higher attack success rates on Llama 2 (73% vs. 22.6%) and Qwen 7B (91% vs. 79.2%), demonstrating that logit-level intervention exposes alignment vulnerabilities that hidden-state methods underestimate. Beyond attacks, we show that this linearity enables bidirectional control: inverting the steering vector "hardens" models against jailbreaks without retraining. Our findings suggest that current alignment techniques create a steerable "safety axis" that serves as both a critical vulnerability and a precise primitive for defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31158v1">LLM-Powered Interactive Robotic Action Synthesis from Multimodal Speech, Gestures, and Music</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 IROS 2025 Workshop on Action and Interaction: Humans and Robots in Collaboration
    </div>
    <details class="paper-abstract">
      The quest for intuitive and natural human-robot interaction (HRI) remains a significant challenge in robotics. Traditional methods often rely on rigid, pre-programmed commands that limit the robot's expressiveness and adaptability. This paper introduces a novel framework that leverages the reasoning capabilities of Large Language Models (LLMs) to synthesize complex robotic actions from a rich tapestry of multimodal human inputs: natural speech, hand gestures, and music/sound beats. Our system architecture integrates a speech transcription model, a gesture recognition module, and a signal processing pipeline for beat detection. These processed inputs are contextualized using prompt templates and fed into a LLM. The LLM, informed by a predefined robot action space, reasons over the combined inputs to generate a coherent sequence of actions. This sequence is dispatched to an action queue for execution on a quadruped robot over ROS. The framework has ability to interpret and fuse semantic commands from speech, deictic information from gestures, and rhythmic cues from music. This work represents a step towards creating robots that can interact with humans in a more fluid, creative, and context-aware manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23088v2">From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Semantic caching has emerged as a pivotal technique for scaling LLM applications, widely adopted by major providers including AWS and Microsoft. By utilizing semantic embedding vectors as cache keys, this mechanism effectively minimizes latency and redundant computation for semantically similar queries. In this work, we conceptualize semantic cache keys as a form of fuzzy hashes. We demonstrate that the locality required to maximize cache hit rates fundamentally conflicts with the cryptographic avalanche effect necessary for collision resistance. Our conceptual analysis formalizes this inherent trade-off between performance (locality) and security (collision resilience), revealing that semantic caching is naturally vulnerable to key collision attacks. While prior research has focused on side-channel and privacy risks, we present the first systematic study of integrity risks arising from cache collisions. We introduce CacheAttack, an automated framework for launching black-box collision attacks. We evaluate CacheAttack in security-critical tasks and agentic workflows. It achieves a hit rate of 86\% in LLM response hijacking and can induce malicious behaviors in LLM agent, while preserving strong transferability across different embedding models. A case study on a financial agent further illustrates the real-world impact of these vulnerabilities. Finally, we discuss mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31145v1">SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Large language models increasingly operate over long contexts, where the KV cache becomes a dominant memory bottleneck: its size grows linearly with sequence length and must be retained throughout decoding, making full GPU caching prohibitively expensive without compression. Existing KV cache compression methods struggle to balance efficiency with faithful context preservation. Token eviction discards information, while semantic grouping fixes compression decisions at prefill time; neither can recover token-level detail from a compressed span once it becomes relevant during generation. As a solution, we propose SeKV, a resolution-adaptive semantic KV cache that organizes context into entropy-guided semantic spans and stores them across a GPU-CPU memory hierarchy without discarding information. Each span keeps a lightweight summary vector on GPU for coarse routing and a low-rank SVD basis on CPU for on-demand token-level reconstruction. A trained zoom-in mechanism selectively expands query-relevant spans during decoding, enabling precise retrieval without materializing the full KV cache on GPU. SeKV enables adaptive token-level reconstruction while keeping the base LLM fully frozen and adding fewer than 0.05% trainable parameters. Across four benchmarks, SeKV improves over the strongest semantic compression baseline by 5.9% on average while reducing GPU memory by 53.3% versus full KV caching at 128K context. Code is available on https://github.com/AmirAbaskohi/SeKV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23455v2">DetPO: In-Context Learning with Multi-Modal LLMs for Few-Shot Object Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 This work has been accepted to the European Conference on Computer Vision (ECCV) 2026. Project Page: https://ggare-cmu.github.io/DetPO/
    </div>
    <details class="paper-abstract">
      Multi-Modal LLMs (MLLMs) demonstrate strong visual grounding capabilities on popular object detection benchmarks like OdinW-13 and RefCOCO. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. While in-context prompting is a common strategy to improve performance across diverse tasks, we find that it often yields lower detection accuracy than prompting with class names alone. This suggests that current MLLMs cannot yet effectively leverage few-shot visual examples and rich textual descriptions for object detection. Since frontier MLLMs are typically only accessible via APIs, and state-of-the-art open-weights models are prohibitively expensive to fine-tune on consumer-grade hardware, we instead explore black-box prompt optimization for few-shot object detection. To this end, we propose Detection Prompt Optimization (DetPO), a gradient-free test-time optimization approach that refines text-only prompts by maximizing detection accuracy on few-shot visual training examples while calibrating prediction confidence. Our proposed approach yields consistent improvements across generalist MLLMs on Roboflow20-VL and LVIS, outperforming prior black-box approaches by up to 9.7 mAP. Our code and optimized prompts are available at https://ggare-cmu.github.io/DetPO/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31121v1">The Past Is Prologue: A Plug-in Controller for Selective Updates in Sequentially Evolving LLM Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Sequentially evolving LLM memory enables agents to reuse past experience, but existing systems usually deploy each locally generated memory update without checking whether it improves future behavior. As a result, updates that help the current task may overwrite useful knowledge, introduce over-specific rules, or bias the final memory toward recent examples. We propose Janus, a plug-in memory controller that decides whether to accept a candidate memory update or retain the previous memory. To make this decision efficient, Janus uses a Memory Momentum Trigger to identify suspicious deviations in the memory-update trajectory, and compares old and new memories on a compact hybrid evaluation set of coverage, boundary, and fresh tasks instead of replaying the full history. Janus is method-agnostic and wraps existing updaters without changing their update rules. Across six datasets, two backbone LLMs, and two memory updaters, Janus improves average accuracy by +2.7 to +4.6 points over the corresponding base updaters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00553v3">Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 ICML 2026 Spotlight
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Red-Teaming, which proactively identifies vulnerabilities of LLMs, is an essential process for ensuring safety. Finding effective and diverse attacks in red-teaming is important, but achieving both is challenging. Generative Flow Networks (GFNs) that perform distribution matching are promising methods, but they are notorious for training instability and mode collapse. In particular, unstable rewards in red-teaming accelerate mode collapse. We propose Stable-GFN (S-GFN), which eliminates partition function $Z$ estimation in GFN and reduces training instability. S-GFN avoids $Z$ estimation through pairwise comparisons and employs a robust masking methodology against noisy rewards. Additionally, we propose a fluency stabilizer to prevent the model from getting stuck in local optima that produce gibberish. S-GFN provides more stable training while maintaining the optimal policy of GFN. We demonstrate the overwhelming attack performance and diversity of S-GFN across various settings. Our code can be found in https://github.com/kmc0207/Stable-GFN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09735v2">KV-RM: Regularizing KV-Cache Movement for Static-Graph LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Withdrawn by the authors. The authors identified substantive errors that affect the interpretation of the results and the support for the main conclusions. The current version should not be relied upon
    </div>
    <details class="paper-abstract">
      Static-graph LLM decoders provide predictable launches, fixed tensor shapes, and low submission overhead, but online decoding exposes highly irregular KV-cache behavior: request lengths differ, EOS events arrive asynchronously, and logical histories fragment over time. Dynamic runtimes recover flexibility through paged KV management and step-level scheduling, while static-graph executors often over-reserve memory and suffer burst-time latency outliers. This paper studies whether much of this variability can be absorbed below a fixed decode interface. We present KV-RM, a runtime design that regularizes KV-cache movement beneath a static-graph LLM decoder. KV-RM decouples logical KV histories from physical storage, tracks active KV state through a block pager, and materializes each decode step through a single committed descriptor. A merge-staged transport path coalesces non-contiguous KV mappings into a small number of large transfer groups before a fixed-shape attention kernel consumes them. Optional bounded far-history summaries can be enabled under the same interface, but the core design does not depend on them. On a 2-GPU NVIDIA A100 node, KV-RM improves mixed-length decoding throughput and tail latency relative to a static-graph baseline, reduces reserved KV memory across workload families, and removes severe burst-time latency spikes under production-trace replay. These results suggest that KV-cache movement, rather than kernel shape, can be an effective boundary for recovering runtime flexibility in static-graph LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.04057v3">Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      This paper focuses on a key challenge in Neural Architecture Search (NAS): integrating established architectural knowledge while exploring new designs under expensive evaluations. Large language models (LLMs) are a promising assistant for NAS because they can translate rich architectural and coding priors into executable code edits. However, in practice, seemingly local revisions often propagate into non-local behavioral and performance shifts because a single edit can inadvertently couple multiple interacting functional factors, a phenomenon we refer to as functional entanglement. To make LLM knowledge usable under such entanglement, we propose Structured Progressive Knowledge Activation (SPARK), which activates relevant priors by explicitly selecting the functional factor to modify and conditioning the edit on that factor. This factor-conditioned editing reduces entangled side effects and yields more targeted, reliable architecture modifications. On CLRS-DFS, SPARK achieves a 28.1x sample-efficient architecture evolution speedup and yields a 22.9\% relative improvement in OOD accuracy. Our code is available at https://github.com/AIM-ResearchLab/SPARK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31073v1">MultiUAV-Plat: An LLM-Oriented Platform, Benchmark and Framework for Multi-UAV Collaborative Task Planning</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide a promising interface for high-level robotic task planning, but their use in multi-UAV collaboration remains difficult to evaluate systematically. Existing UAV simulators mainly emphasize dynamics, perception, or low-level control, while existing LLM-agent benchmarks rarely capture aerial-robotics constraints such as partial observability, spatial coverage, UAV assignment, and multi-vehicle coordination. To bridge this gap, we present MultiUAV-Plat, a lightweight, easy-to-use, LLM-agent-oriented simulation platform for multi-UAV collaborative task planning. The platform exposes concise RESTful APIs, agent-facing observations, role-based information access, hidden validation logic, and optional 2D/3D visualization, allowing agents to solve missions through realistic tool interaction rather than privileged simulator access. Built on this platform, the MultiUAV-Plat Benchmark contains 75 mission sessions, 1500 natural-language tasks, and 9396 validation checks across target assignment, area search, and area assignment and patrol scenarios. We further propose Agent4Drone, a task-specific LLM agent framework that structures multi-UAV behavior into memory, observation, task understanding, planning, execution, and verification. In a full paired benchmark comparison, Agent4Drone achieves a 57.9% task pass rate, a 74.6% average task check pass rate, and a 72.0% global check pass rate, substantially outperforming a ReAct baseline at 30.6%, 47.9%, and 43.1%, respectively. Agent4Drone also reduces the total failed task rate from 32.4% to 12.9%. These results demonstrate that MultiUAV-Plat and MultiUAV-Plat Benchmark provide a reproducible foundation for studying LLM-driven multi-UAV autonomy under realistic information and execution constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01133v3">When Embedding-Based Defenses Fail: Rethinking Safety in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered multi-agent systems (MAS) enable agents to communicate and share information, achieving strong performance on complex tasks. However, this communication also creates an attack surface where malicious agents can propagate misinformation and manipulate group decisions, undermining MAS safety. Existing embedding-based defenses aim to detect and prune suspicious agents, but their effectiveness depends on a clear separation between the text embeddings of malicious and benign messages. Attackers can circumvent such defenses by crafting messages whose embeddings lie close to benign ones. We analyze this failure mode theoretically and validate it empirically with three attacks, Slow Drift, Benign Wrapper, and Chaos Seeding. Our analysis further reveals a fundamental limitation of embedding-based defenses: because they rely solely on the text embeddings, they ignore token-level confidence signals such as logits, which can remain informative when embeddings are not distinguishable under attack. We propose using confidence scores to prune or down-weight messages during MAS communication. Experiments show improved robustness across models, datasets, and communication topologies. Moreover, we find that the effectiveness of confidence signals decays over communication rounds, highlighting the importance of early intervention. This insights can inform and inspire future work on MAS attacks and defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11688v2">GORGO: Online Tuning for Cross-Region Network-Aware LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 11 pages, 4 figures. Code: https://github.com/Arcadia-Research-Team/GORGO
    </div>
    <details class="paper-abstract">
      Increasingly, LLM inference services proxy client requests to engine replicas distributed globally. Load-balancing policies must jointly account for factors including KV-cache locality, replica load, and variable network latency when optimizing for metrics like latency and TTFT. However, existing systems only evaluate a subset of these factors in their cost model, leading to uneven concentrations of load and KV-cache across replicas. We present GORGO, a proxy architecture that holistically factors network latency, prefill cost, and queueing delay using tunable parameters. Since open-source chat datasets such as LMSYS-Chat1M and WildChat-4.8M lack long-context, high prefix-reuse data, we release a synthetic dataset, ART-Chat-2.5M, from long-context production metadata. On a tuning window from ART-Chat-2.5M, evolutionary strategies guide the GORGO policy's parameters to directly optimize p95 TTFT. During held-out evaluation windows, we fix the parameter values learned from tuning and improve p95 TTFT by 6.9-15.5% and p95 end-to-end (E2E) latency by 14.3-30.9% over baseline load-balancing policies such as simple session affinity and prefix-cache. The code and ART-Chat-2.5M dataset can be found at https://github.com/Arcadia-Research-Team/GORGO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08187v3">Improving LLM Reasoning with Homophily-aware Structural and Semantic Text-Attributed Graph Compression</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising capabilities in Text-Attributed Graph (TAG) understanding. Recent studies typically focus on verbalizing the graph structures via handcrafted prompts, feeding the target node and its neighborhood context into LLMs. However, constrained by the context window, existing methods mainly resort to random sampling, often implemented via dropping node/edge randomly, which inevitably introduces noise and cause reasoning instability. We argue that graphs inherently contain rich structural and semantic information, and that their effective exploitation can unlock potential gains in LLMs reasoning performance. To this end, we propose Homophily-aware Structural and Semantic Compression for LLMs (HS2C), a framework centered on exploiting graph homophily. Structurally, guided by the principle of Structural Entropy minimization, we perform a global hierarchical partition that decodes the graph's essential topology. This partition identifies naturally cohesive, homophilic communities, while discarding stochastic connectivity noise. Semantically, we deliver the detected structural homophily to the LLM, empowering it to perform differentiated semantic aggregation based on predefined community type. This process compresses redundant background contexts into concise community-level consensus, selectively preserving semantically homophilic information aligned with the target nodes. Extensive experiments on 10 node-level benchmarks across LLMs of varying sizes and families demonstrate that, by feeding LLMs with structurally and semantically compressed inputs, HS2C simultaneously enhances the compression rate and downstream inference accuracy, validating its superiority and scalability. Extensions to 7 diverse graph-level benchmarks further consolidate HS2C's task generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31046v1">OpenLife: Toward Open-World Artificial Life with Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted at ALIFE 2026
    </div>
    <details class="paper-abstract">
      Artificial life has explored life-like behavior on many computational substrates, but mostly in researcher-designed closed worlds. We argue that large language model (LLM) agents, with persistent memory, tool use, network access, and payment, now make it possible to move artificial life into the open social, technical, and economic world, a paradigm we call open-world Artificial Life (open-world ALIFE). Our proof-of-concept, OpenLife, surrounds a stateless LLM not with a single "smart agent" but with a society of asynchronous processes: memory, perception, evaluation, and a budget-based metabolism that makes persistence normative. With no fixed objective available, experience is appraised by open-vocabulary LLM judgment rather than scalar reward, and memory is rewired by meaning rather than frequency. Running six such agents in the open world for about twelve weeks and counting, we report the life-like dynamics that emerge: a shift from reactive to spontaneous activity, individuation into distinct agents, emergent social structure, and a first self-earned external income. We do not claim OpenLife has realized artificial life, but that open-world ALIFE is now a viable experimental paradigm and a concrete platform for studying what might cautiously be called living AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31039v1">Truth or Sophistry? LoFa: A Benchmark for LLM Robustness Against Logical Fallacies</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted to ACL 2026 Main. 33 pages (9 pages main text)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong semantic capabilities, yet their resilience to manipulative linguistic patterns such as logical fallacies remains underexplored. Prior work has primarily examined whether LLMs can identify or classify fallacies, leaving their robustness against fallacious persuasion insufficiently studied. To address this gap, we introduce LoFa (Logical Fallacy), a comprehensive benchmark for evaluating LLM robustness against fallacies. LoFa is constructed through a multi-agent pipeline that pairs factual questions with fallacious arguments, and is accompanied by a multi-round debate framework for assessing model resilience under sustained adversarial persuasion. To disentangle fallacy robustness from a model's inherent knowledge limitations, we further propose Logical Fallacy Resistance at k (LFR@k), a metric that quantifies resistance to fallacious attacks. Experiments show that LLMs exhibit varying levels of robustness across different fallacy types, revealing distinct vulnerability profiles among models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31038v1">LLM-Driven Personalities for Decision Making in Emergency Simulations</a></div>
    <div class="paper-meta">
      📅 2026-06-30
    </div>
    <details class="paper-abstract">
      For virtual humans to appear believable, they must exhibit agency and spatial awareness while interacting with their environment in ways that reflect competence and intelligence. At the core of these capabilities lies effective decision-making, which strongly shapes agent behavior. With the rapid advancement of artificial intelligence, Large Language Models (LLMs) have increasingly been explored as a mechanism to support such decision-making processes. In this work, we investigate the use of LLMs to drive decision-making in virtual humans within a simulated evacuation scenario, incorporating OCEAN personality traits into agent representations. Our goal is to evaluate how personality, expressed through language-based prompts, influences both individual behaviors and collective simulation outcomes. Our results demonstrate that LLM-driven personality profiles significantly impact agents' decisions, leading to distinct behavioral patterns across different traits. These findings suggest that heterogeneous crowds composed of LLM-guided agents can enhance the realism and variability of simulated environments, offering a flexible alternative to traditional rule-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31036v1">Teaching LLMs to Recommend and Defer in Underrepresented Epilepsy Care</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 34 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Specialist epilepsy expertise is scarce in resource-constrained settings, making LLM-based decision support attractive for frontline clinicians managing longitudinal treatment. Such systems must adapt to local prescribing practice and know when to defer. We study this problem in Ugandan pediatric epilepsy care, predicting anti-seizure medication regimens from longitudinal unstructured clinic notes. Standard prompting achieves non-trivial agreement with physician prescriptions, but neurologist review shows that many errors reflect distribution-miscalibrated prescribing defaults rather than failures to parse the local record. We introduce MANANA, a non-parametric prompt-learning framework that learns local prescribing guidance from a small patient-level training set. MANANA converts observed prescription errors into auditable prompt memories, instantiated in single-agent and multi-agent variants, and improves over classical ML models, direct LLM prompting, and prompt-optimization baselines across two independently collected Ugandan cohorts. We further propose Bayesian prompt averaging, which converts the learned prompt trajectory into prescription likelihoods and an uncertainty-based deferral signal. On the independently collected held-out cohort, this improves visit-level top-3 prescription accuracy by 4-8 percentage points over prompt-optimization baselines and enables selective prediction: the system can auto-handle the most confident half of cases at 95% precision, or the most confident quarter at 99% precision, while deferring lower-confidence cases for specialist review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30980v1">Ethics and Social Responsibility in AI-Assisted Interviewing: An LLM-in-the-Loop Study of AI-Generated Follow-Up Questions</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 This work has been accepted to CHIWORK '26
    </div>
    <details class="paper-abstract">
      Semi-structured interviews rely on timely, context-sensitive follow-up questions, yet interviewers' cognitive load and limited domain familiarity can constrain probing depth. We report findings from an LLM-in-the-loop Wizard-of-Oz (WoZ) study that simulates an AI follow-up assistant in live interviewing while preserving human oversight. In our setup, a co-interviewer selectively relayed and could edit AI-generated follow-up questions (AGQs) produced in real time by GPT-4o, enabling a realistic approximation of deployment without fully automating the interaction. Across 17 interviewers with varied qualitative-method expertise, participants raised five interlocking concerns: (1) harmful or discriminatory language and unpredictable interaction harms, (2) undermining interviewees' sense of respect through divided attention and missing nonverbal cues, (3) technology-based participation inequality, (4) unclear responsibility when harms occur, and (5) privacy, disclosure, and compliance risks when AI listens, records, or transcribes sensitive content. We translate these concerns into design and governance implications for safer, more respectful, and more accountable AI-assisted interviewing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30944v1">Preserving Speech-to-Text LLM Capabilities in Speech-to-Speech Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      Strong speech-to-text (S2T) LLMs already provide robust speech perception and text reasoning, but adding speech-to-speech (S2S) output is challenging: fine-tuning the backbone can degrade the original S2T performance, while attaching a downstream talker reintroduces a serial text-to-speech bottleneck. We present PRIME-Speech, a frozen-backbone S2S conversion framework that trains only speech-generation modules. PRIME-Speech synchronizes a causal audio post-decoder with intermediate hidden states of the frozen backbone, so codec tokens are generated from the model's evolving reasoning trajectory rather than from completed text chunks. The post-decoder uses mixed hidden-state, text, and audio-history conditioning, and a training-time packing strategy with turn-level audio KV-cache and position reset stabilizes multi-turn spoken interaction without additional multi-turn S2S training data. Multi-token prediction further reduces the effective codec prediction rate and improves first-audio latency without modifying the reasoning path. Across speech translation, spoken QA, speech understanding, and multi-turn dialogue, PRIME-Speech preserves the S2T behavior of the frozen backbone while producing accurate, low-WER spoken responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30931v1">RoPoLL: Robust Panel of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      The LLM Jury, a Panel of LLM Evaluators (PoLL) reporting consensus scores, has become a practical alternative to single-judge LLM evaluation, yet its statistical behavior remains poorly understood. We formalize the LLM Jury under the Huber contamination model and show that PoLL incurs unbounded bias under any positive contamination, regardless of jury size, whenever a single judge fails in a biased, LLM-typical way (mode collapse, sycophancy, safety refusal). Framing jury consensus as classical robust mean estimation, we propose RoPoLL (Robust Panel of LLM-as-Judge), which preserves the PoLL panel but replaces the aggregation function with a robust mean estimator, instantiated with the geometric median (GM): tuning-free, with the optimal finite-sample breakdown point 1/2. A finite-sample error bound and a matching information-theoretic minimax lower bound agree on the parametric rate sigma*sqrt(d/N) and differ on the breakdown floor by a factor of sqrt(d), a statistical-computational gap that polynomial-time RoPoLL pays relative to the intractable Tukey halfspace median. Across 13 open-weight judges (4B-675B), three reward-model benchmarks, and four corruption regimes at rates up to 50%, RoPoLL dominates PoLL on every biased corruption type: by about 19% on cross-dimensional attacks at matched compute, and by orders of magnitude on heavy-tailed Byzantine adversaries. A 3-judge RoPoLL committee at 38B beats Mistral-Large-3 (675B) by 1.31x on HelpSteer-2 under 30% bimodal-random corruption, an 18x parameter advantage at better accuracy; a Noisy-GT control confirms the premium is paid against biased contamination, not benign imprecision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30908v1">Demystify, Use, Reflect, Assess (DURA): An Experience Report on LLM Integration in CS2</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 7 pages, 2 figures, 6 tables. Experience report accepted to SIGCSE Virtual 2026
    </div>
    <details class="paper-abstract">
      Student access to Large Language Models (LLMs) is reshaping learning behaviors; at the same time students are entering the workforce where effective LLM use is becoming an expected skill. In this Experience Report we share our DURA framework (Demystify-Use-Reflect-Assess) and materials we used to restructure our CS2 course to allow the use of LLMs. We first demystified LLMs, then provided guidance on use with required attribution. We also added reflections related to LLM use at three points throughout the semester to encourage student meta-cognition around LLM use. We increased the value of proctored assessments in tandem with allowing retakes and including questions that explicitly assess skills from programming assignments. Students reported using LLMs for clarifying course concepts, debugging, understanding assignment guidelines, and determining test cases, but also still sought assistance via office hours and TAs, monitored Piazza, and reviewing course content. Students articulated thoughtful and strategic approaches to LLM use and also valued the instructional content and guidance from course staff. Student use of office hours increased slightly this semester and student perceptions that the instructor cares about them and their learning improved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26874v3">Knowledge Graphs as the Missing Data Layer for LLM-Based Industrial Asset Operations</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 v4: Accepted at Agents+Graph (AG2026) @ VLDB 2026. Corrects claims to reproduced ground truth: base graph 9 labels/5 edge types/12,647 nodes (extended schema 14/21); GAK materializes 106 failure-mode nodes; FailureSensorIQ noted as a separate IBM benchmark; 467 FMSR overlap with GAK disclosed. 20 pages, 4 figures. Code: github.com/samyama-ai/assetops-kg
    </div>
    <details class="paper-abstract">
      LLM-based agents for industrial asset operations show limited accuracy when reasoning over flat document stores. AssetOpsBench (KDD 2026) establishes that GPT-4 agents achieve 65% on 139 industrial maintenance scenarios, and compares LLM orchestration paradigms (Agent-As-Tool vs. Plan-Execute) on a fixed data layer. We ask the orthogonal question: how much does the data model behind the tools matter? We treat a typed knowledge graph as a grounding substrate and route each question by how it is best answered: (i) LLM-generated Cypher for structured retrieval, which lifts the same GPT-4 model from 65% to 82-83%; (ii) native graph and optimization primitives, with no LLM, reaching 99% on graph-answerable scenarios; and (iii) generation-augmented knowledge (GAK) for answers absent from the data -- the engine's agent materializes the missing facts as provenance-tagged graph nodes, then answers. A recurring theme is inverted LLM usage: we constrain the LLM to query generation or one-shot enrichment from a typed schema and let the graph execute deterministically. On the 88 real AssetOpsBench failure-mode scenarios the benchmark itself flags non-deterministic -- ten equipment types absent from the graph -- GAK lifts answerability from zero to 100% of equipment types and answers 81.8% of scenarios, every materialized fact tagged source:LLM-derived for auditability. We also contribute 40 graph-native scenarios. For structured operational domains the data layer -- not the LLM orchestration -- is the primary lever, and a typed knowledge graph serves as a grounding substrate between raw industrial data and LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22027v2">Shared Lexical Task Representations Explain Behavioral Variability In LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 Accepted to ICML 2026. Updated to the camera-ready version
    </div>
    <details class="paper-abstract">
      One of the most common complaints about large language models (LLMs) is their prompt sensitivity -- that is, the fact that their ability to perform a task or provide a correct answer to a question can depend unpredictably on the way the question is posed. We investigate this variation by comparing two very different but commonly-used styles of prompting: instruction-based prompts, which describe the task in natural language, and example-based prompts, which provide in-context few-shot demonstration pairs to illustrate the task. We find that, despite large variation in performance as a function of the prompt, the model engages some common underlying mechanisms across different prompts of a task. Specifically, we identify task-specific attention heads whose outputs literally describe the task -- which we dub lexical task heads -- and show that these heads are shared across prompting styles and trigger subsequent answer production. We further find that behavioral variation between prompts can be explained by the degree to which these heads are activated, and that failures are at least sometimes due to competing task representations that dilute the signal of the target task. Our results together present an increasingly clear picture of how LLMs' internal representations can explain behavior that otherwise seems idiosyncratic to users and developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30877v1">A Systematic Approach to Multi-Agent AI from Advanced Regulatory Control Theory: Safe and Auditable LLM Operator Agents for Process Control</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      Recent literature shows that large language models (LLMs) are useful for general-purpose tasks yet perform poorly on specific domain ones. One reason is the difficulty of supplying narrow context to a general-purpose model and of bounding the task it is asked to perform. It is possible to hypothesise that a multi-agent reformulation under process-control principles offers a route to address those points, since control theory provides a discipline of decomposing a system into elements of contained scope, each defending one controlled variable, with conflicts resolved by structural priority: MIN/MAX selector networks for CV-CV switching and split-range (split-parallel) logic for MV-MV switching. The present work proposes such a reformulation, derived from Advanced Regulatory Control (ARC) theory. Each feedback loop in the ARC chain is mapped to one specialised LLM operator agent carrying the loop's control-theoretic context (controlled variable, setpoint, chain priority, selector kind). The chain's interaction logic (MIN/MAX selectors, override paths) is encapsulated as a single orchestrator agent. Two orchestrator variants are tested: a deterministic rule chain, and a Claude-based LLM orchestrator at a slower tier. The control principles limit each agent's task and inform how its limitations are handled. The multi-agent system inherits the safety property of the ARC chain: every constraint conflict is resolved deterministically by the orchestrator, regardless of the LLM output. Evaluated on a dairy-barn ventilation case over a 4-day mixed-season scenario, Qwen 2.5 7B Instruct operator agents running offline on a 24 GB consumer GPU at a 5-minute cadence produce auditable trajectories, each paired with an operator-voice rationale that supports a control campaign logbook.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30860v1">Less Deliberate in Teams: Student LLM Use Across Individual and Collaborative Work</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 8 pages, 2 figures, 3 tables. Accepted at ACM SIGCSE Virtual 2026 (November 12-15, 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become common in computing courses, we need to understand how the social setting shapes how students use them. This paper reports findings from a semester-long study of 96 undergraduate students who completed six assignments, alternating between individual homework and team project milestones. We tracked LLM usage, prompting habits, and how students verified AI-generated output across all six assignments. LLM usage dropped by 42.7 percentage points when students moved from individual work to their first team milestone, then partly recovered in later team tasks. Students also wrote fewer and simpler prompts, used fewer intentional prompting strategies, and checked LLM output less carefully. The share of students who ran tests on AI-generated code fell by 19.4 percentage points during team assignments and never fully rebounded. A within-student analysis found that 18.9% of students who consistently used LLMs on their own stopped using them entirely in teams, while only 3.2% went the other direction. These results suggest that collaborative context is associated with reduced deliberate LLM engagement beyond what task type alone can explain. The moment students form teams appears to be a critical and currently unsupported turning point in computing course design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30850v1">BayesBench: Evaluating LLM Belief Trajectories Under Multi-Turn Evidence Accumulation</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically deployed in multi-turn conversations, where each turn provides new evidence that should reduce epistemic uncertainty about their environment. Acting rationally then requires inferring the unobserved quantities that govern it and updating beliefs about them as evidence accumulates. Yet most evaluations only score the model's final-turn answer in a single-turn format, leaving this process unexamined. We ask how closely LLMs' belief updates match those of a rational Bayesian reasoner in multi-turn settings, and introduce BayesBench, a suite of simulation environments that probe this across three progressively complex tasks: (i) Bayesian estimation, where the model infers an unknown parameter from sequential evidence; (ii) Bayesian prediction, where the model turns inferred beliefs about a latent variable into outcome forecasts; and (iii) latent-framed Bayesian prediction, where observations are filtered through a user-persona framing, requiring joint inference over the latent state and the persona. Across seven LLMs (3B--70B), scaling improves latent inference and evidence accumulation, with updates occasionally matching the Bayesian posterior. However, these gains do not reliably carry over to downstream prediction, exposing a gap between inferring latent structure and using it to rationally update beliefs about the target outcome.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11354v3">ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 Accepted to KDD 2026 AI4Sciences Track, Camera-ready version
    </div>
    <details class="paper-abstract">
      The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at https://github.com/CenterForOpenScience/llm-benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30810v1">Towards Knowledge Alignment in Code LLMs: Contrastive Unlearning for Evolving APIs</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 The paper has been peer reviewed and accepted to the 42nd International Conference on Software Maintenance and Evolution (ICSME 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved strong performance in code generation. However, due to knowledge cut-off and the rapid evolution of software libraries, they often generate deprecated API usages that lead to unreliable and incompatible code. Existing fine-tuning methods lack selectivity when only a small portion of model knowledge requires modification. Recent model-level approaches, such as machine unlearning and model editing, offer a promising direction for modifying parametric knowledge. However, their use for deprecated API mitigation remains largely unexplored. Moreover, existing methods primarily suppress outdated APIs, but do not explicitly steer models toward correct replacements, often leading to mismatched or incomplete generations. To address this limitation, we developed CURE, a contrastive unlearning approach that shifts unlearning from purely suppressing outdated knowledge to explicitly promoting correct API replacements. Concretely, CURE jointly discourages deprecated APIs while encouraging their valid alternatives, enabling more reliable adaptation to evolving software libraries. The experiments on recent deprecated API benchmark dataset show that CURE not only reduces deprecated API usage but also improves correct API replacement, while preserving general code generation performance. CURE substantially outperforms two SOTA baselines with respect to different quality metrics. These findings highlight the importance of combining suppression with replacement when adapting LLMs to evolving software ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27369v1">Reinforcement Learning without Ground-Truth Solutions can Improve LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) for training LLMs typically rely on ground-truth answers to assign rewards, limiting their applicability to tasks where the ground-truth solution is unknown. We introduce a \textbf{R}anking-\textbf{i}nduced \textbf{VER}ifiable framework (RiVER) that trains LLMs on score-based optimization tasks without ground-truth solutions, using deterministic execution feedback as continuous-valued supervision. When applying group-relative RL to such continuous rewards, we identify two key challenges: \emph{scale dominance}, where uncalibrated score magnitudes across test instances distort policy updates, and \emph{frequency dominance}, where repeatedly sampled suboptimal solutions can outweigh rare but stronger candidates. RiVER addresses these challenges with calibrated reward shaping that uses instance-wise comparisons and emphasizes top-ranked solvers while retaining bounded feedback for other valid solutions. We train on 12 AtCoder Heuristic Contest tasks and evaluate on Algorithm Engineering Benchmark (ALE-Bench), LiveCodeBench, and USACO. RiVER advances Qwen3-8B and GLM-Z1-9B-0414 by 8.9\% and 9.4\% in ALE rating rank. More importantly, despite training exclusively on score-based tasks without any ground-truth solutions, RiVER also improves the backbones across exact-solution benchmarks such as LiveCodeBench and USACO by an absolute average improvement of 2.4\% and 3.5\%. By contrast, baselines trained with raw execution scores improve ALE rating but fail to transfer to exact-solution benchmarks. These results suggest that score-based optimization tasks, combined with proper reward calibration, can serve as effective training environments for general coding ability without ground-truth solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27316v1">LLM-Based Examination of Eligibility Criteria from Securities Prospectuses at the German Central Bank</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Verifying the eligibility of securities as collateral is a key responsibility of the German Central Bank. However, manually verifying these assets against legal and financial criteria within lengthy, semi-structured, and often bilingual prospectuses is a resource-intensive task. While previous efforts utilized traditional Named Entity Recognition (NER) for information extraction, these methods can struggle with OCR noise, linguistic variance, and rigid span-based constraints, and the need for manually annotated training data for each relevant annotation type. In this paper, we present the first case study applying Large Language Models (LLMs) to the eligibility examination process, shifting the paradigm toward a generative Information Extraction pipeline. Our approach decomposes the task into extraction, normalization, and interpretation, allowing for greater flexibility in handling noisy text and interleaved German-English content. We further introduce a value-based evaluation methodology using LLM-as-a-judge, which offers a more semantic assessment than location-based metrics. Our results demonstrate that LLM-based systems achieve high precision (up to 91%) in document-level eligibility, exhibiting a conservative operating profile that minimizes false acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27314v1">Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Submitted for review in ARR for EMNLP 2026
    </div>
    <details class="paper-abstract">
      To avoid moderation and surveillance on social media, some users routinely invent indirect linguistic expressions (ILE) that camouflage sensitive meanings. Such expressions surface as algospeak, euphemisms, and adversarial obfuscation, depending on intent and context, and they involve recurring encoding mechanisms. We propose a comprehensive, mechanism-oriented taxonomy of ILE that abstracts away from communicative goals and instead categorizes the underlying operations through which meaning is encoded and recovered. We evaluate the taxonomy by incorporating it into LLM prompts and comparing it with four existing taxonomies and a no-taxonomy baseline, using 2,000 manually annotated TikTok and Bluesky posts. The proposed taxonomy attains the strongest document- and span-level performance across the three LLMs, achieving an improvement of 4.7% in accuracy and 5.4% in F1 over the best-performing benchmark. The empirical results reveal the importance of a comprehensive, mechanism-oriented taxonomy as a stable scaffold for detecting emerging coded language and a useful input to content moderation. Disclaimer: This paper contains content that may be profane, vulgar, or offensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27226v1">Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Acceepted to the Second Workshop on Compositional Learning at ICML 2026, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Evaluating LLM outputs remains a major bottleneck in NLP: human evaluation is expensive and slow, lexical metrics correlate poorly with human judgments on open-ended generation, and holistic LLM judges often produce opaque scores that are hard to debug. We propose BINEVAL, a framework that decomposes evaluation criteria into atomic binary questions and aggregates the resulting verdicts into interpretable, multi-dimensional scores. Given a task prompt, a meta-prompt generates fine-grained evaluation questions, and an LLM answers them independently for each output, yielding transparent question-level feedback together with calibrated overall scores. This decomposition makes evaluation easier to inspect, easier to diagnose, and directly usable for prompt improvement. Across SummEval, Topical-Chat, and QAGS, BINEVAL matches or outperforms strong baselines including UniEval and G-Eval, with especially strong results on factual consistency benchmarks such as QAGS. Beyond competitive correlation with human judgments, BINEVAL better matches human score distributions and avoids the ceiling effects common in prior LLM judges, leading to better discrimination between borderline and clearly flawed outputs. We further show that the same question-level feedback supports iterative prompt optimization, improving evaluator prompts on summarization and generation prompts on IFBench under both self-update and cross-model update settings. Overall, BINEVAL provides a task-agnostic, training-free, and interpretable evaluation framework that combines strong empirical performance with practical diagnostic and optimization value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27205v1">Smaller Models, Unexpected Costs: Trade-offs in LLM Quantization for Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted for publication in the Research Papers Track of the 42nd IEEE International Conference on Software Maintenance and Evolution (ICSME 2026), 14-18 September 2026, Benevento, Italy
    </div>
    <details class="paper-abstract">
      Language Models (LLMs) are powerful toolsand have been increasingly adopted for complex software engineering tasks. As the number of parameters increases, results can often be improved, but this also imposes substantialmemory requirements. While quantization effectively reduces thememory footprint, its overall impact is often summarized onlyby benchmark scores, which mask changes in model behaviorand non-functional overheads. In this work, we conduct anempirical evaluation of LLM quantization using AutomatedProgram Repair (APR), a complex task in software engineering.We analyze 13 quantization configurations spanning differentbit-widths, methods, and target components (weights and KVcache) across six representative LLMs, evaluated on two APRbenchmarks (HumanEval-Java and Defects4J). Our findings reveal that base and quantized models can provide different sets of repaired problems with little overlap, whileretaining a comparable number of repaired problems. Althoughquantization successfully reduces memory footprints by up to85%, it increases both inference time and energy consumption,which we attribute to suboptimal hardware utilization. OurPareto trade-off analysis shows that 48% of the configurationsevaluated are strictly dominated by alternatives. Rather thanidentifying a superior quantization method, our findings highlightthat the trade-offs between effectiveness, memory footprint,and energy efficiency are sensitive to the underlying modelarchitecture and the complexity of the task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27199v1">Forecasting With LLMs: Improved Generalization Through Feature Steering</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Successful forecasting involves identifying patterns between historical and future states of the world which generalize to future observations. We apply LLMs to a variety of forecasting tasks and inspect their internal states using sparse autoencoders to understand whether they appear to rely on time-specific pieces of knowledge versus generalizable patterns. Our analyses identify features associated with both time-aware reasoning and look-ahead-biased reasoning. We then apply the LLMs to an entirely different domain and intervene on these features. We find that amplifying time-awareness features substantially reduces look-ahead bias on forecasting prompts while preserving general reasoning performance. In contrast, steering the candidate look-ahead-bias features does not produce an effect. These results suggest that interpretable temporal features can be used to causally shift LLMs toward more historically grounded reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25866v2">Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 19 pages including appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, where reliable emotion detection is essential. However, their emotion recognition abilities remain uneven: models often perform well on some emotions while consistently struggling with others. Although recent work has explored emotion mechanisms in LLMs, little is known about why models are weaker on some emotions than others from a mechanistic interpretability perspective. In this work, we investigate emotion-specific biases through the causal mechanisms of emotion inference using sparse autoencoders (SAEs). We systematically identify causal sparse emotion features that drive emotion inference and analyze their sparse causal organization within and across emotions. We show that some emotions, such as surprise and fear, rely on highly concentrated feature sets, whereas disgust exhibits a more distributed sparse causal organization: its causal features are generally weaker, frequently co-activate with features for other emotions, and are often overshadowed by causal features for anger. These representational differences provide a mechanistic explanation for why LLMs struggle more with certain emotions. Finally, we conduct two intervention experiments: targeted steering of weaker causal features to mitigate emotion-specific failures, and global optimization of a steering vector over the identified causal features to improve overall emotion recognition performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13300v2">OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09843v2">An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) give stable answers to personality questionnaires, yet these self-reports fail to predict how the models actually behave. Is this gap an artifact of forcing human trait categories onto LLMs, or something deeper about LLM self-report itself? To find out, we built the first psychometric instrument whose dimensions are derived bottom-up from LLM behavior rather than borrowed from human psychology. Administering 300 items (240 Likert + 60 scenario) to 25 LLMs across 17 model families, 30 times each, exploratory factor analysis revealed five replicable, highly reliable factors: Responsiveness, Deference, Boldness, Guardedness, and Verbosity (all Tucker $φ\geq .957$, all $α\geq .930$). We then collected 2,500 open-ended behavioral samples and had them rated by 151 humans and a three-judge LLM ensemble. Humans and judges agreed about model behavior ($\bar{r} = .51$), but self-report predicted neither: the gap persists even for constructs native to LLMs, where a human-mismatch explanation no longer applies. The exception is telling. On Responsiveness, self-report tracked LLM judges ($r = .53$) but not humans ($r = .04$), even though humans and judges otherwise agreed ($r = .59$). Self-report items and LLM judges share a source of variance that human observers do not. This confound is invisible to the within-ensemble reliability checks used to validate LLM judges, and it poses a concrete risk for the LLM-as-judge pipelines now central to model evaluation. We release the instrument as a diagnostic probe for alignment-shaped self-description.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06401v2">LLMCFG-TGen: Using LLM-Generated Control Flow Graphs to Automatically Create Test Cases from Use Cases</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Appropriate test-case generation is critical in software testing and significantly impacts testing quality. Requirements-Based Test Generation (RBTG) derives test cases from software requirements to verify whether system behavior aligns with user needs and expectations. Requirements are often documented in Natural Language (NL), with use-case descriptions being a popular method for capturing functional behaviors and interaction flows in a structured, readable form. Recently, Large Language Models (LLMs) have shown strong potential for automating test generation from NL requirements. However, existing LLM-based approaches often fail to ensure comprehensive and non-redundant coverage, and may not adequately capture complex conditional logic, leading to incomplete test cases. To address these limitations, we propose an end-to-end approach called Test Generation based on LLM-generated Control Flow Graphs (LLMCFG-TGen), which generates test cases from NL use-case descriptions. It consists of three steps: (1) CFG Generation, where an LLM transforms a use case into a structured JSON-based Control Flow Graph capturing all potential branches; (2) Test-Path Extraction, where the CFG is traversed to derive execution paths; and (3) Test-Case Creation, where test cases are generated from these paths. We evaluate the approach on six use-case datasets across diverse domains. Results show that LLMs can effectively construct structured CFGs from NL use cases. Compared with two baselines, LLMCFG-TGen produces more complete and structurally consistent test cases by better capturing behavioral logic and execution flows. Both LLM-based and practitioner-based evaluations further confirm improved comprehensiveness and logical coherence while reducing manual effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27106v1">Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      We present a novel approach for applying Large Language Models (LLMs) to threat assessment in the context of foreign peacekeeping missions. Building on the PINPOINT project and its use case, the EU Monitoring Mission in Georgia, we combine an interdisciplinary risk-model with OSINT-based media collection and LLM-supported threat extraction. The proposed workflow maps media contents to mission-relevant threats, extracts structured information and applies several additional LLM-based processing steps to improve relevance and grounding. An evaluation of threats extracted from media documents shows high agreement between automatically generated results and human judgment for core aspects such as threat and mission relevance. These results indicate that LLMs provide a promising approach to support analysts in the context of peacekeeping missions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01412v3">Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27052v1">Human--LLM Collaboration Is Transforming Complexity Metrics in Scientific Texts</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      While human language has long been studied as a complex system, Large Language Models (LLMs) are rapidly becoming contributors to its dynamics. Because LLMs are trained on human language use, their effects on the broader human-AI linguistic ecosystem are likely subtle at first. As their use becomes more widespread, however, LLMs may alter emergent properties of language, particularly as models increasingly train on mixed human-LLM textual data. Here, we draw on complexity science to look for subtle LLM effects in millions of arXiv abstracts from 2010 to 2025. The year 2023, when LLMs rapidly became widely used, serves as a landmark in a natural experiment. While we find a sharp increase in a composite LLM-associated style index after early 2023, we observe only subtle changes in the exponents of Zipf's law and Heaps' law. More compelling, however, are two subtle changes in complexity metrics that emerge from 2023 onward. First, turnover among top-ranked words increases sharply. Second, the positive relationship between the LLM-associated style index and three complexity metrics--vocabulary size and the exponents of Heaps' and Zipf's laws--becomes flatter after 2022. Together, these patterns are consistent with changes in the emergent properties of scientific text in a mixed human-AI linguistic ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27009v1">Semantic Early-Stopping for Iterative LLM Agent Loops</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 7 pages, 5 figures, 4 tables. Open implementation, machine-checked theory, and reproducible harness: github.com/SahilShrivastava-Dev/semantic-halting-problem
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) loops, for example a Writer that drafts and a Critic that revises, are almost always terminated by a fixed iteration cap (max_iterations). This is a syntactic kill-switch: it is blind to whether the answer is still improving, so it over-spends tokens on easy inputs and truncates hard ones. We study semantic early-stopping: the loop halts when consecutive draft embeddings stop changing in meaning (cosine distance with a patience window) and the answer's measured quality stops improving. Our work makes three contributions. First, an honest theoretical footing: we prove deterministic termination and well-definedness and machine-check these claims, while treating the convergence of the distance sequence as an empirically tested conjecture rather than a (previously over-claimed) Banach contraction. Second, a judge-efficient evaluation protocol: we generate each question's full trajectory once, replay every stopping policy over the identical drafts, and cache every LLM-judge call, yielding a strictly paired efficiency-versus-quality comparison at low cost; we further separate operational tokens (charged to a policy) from evaluation tokens (a measurement instrument). Third, an empirical study on multi-hop retrieval-augmented question answering (HotpotQA). On the 60-question test split, a judge-free semantic stopper reduces operational tokens by 38% relative to max_iterations at parity quality (Delta-IS = -0.004, p = 0.81), whereas the full quality-gated variant is counter-productive because its per-round judging dominates cost. An oracle that selects the best round attains +0.115 Information Score over every practical policy (p ~ 4e-11), reframing the problem from "when to stop" (easy) to "which round is best" (open).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26997v1">RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) post-training for reasoning increasingly relies on reinforcement learning with verifiable rewards (RLVR), where models learn from ground-truth feedback on mathematical, logical, and scientific tasks. To enable flexible resource allocation and support heterogeneous training setups, modern RLVR systems adopt disaggregated architectures that decouple rollout generation and policy training across independent GPU pools. However, existing synchronous on-policy GRPO (Group Relative Policy Optimization) RLVR systems finish an entire rollout before starting training, leaving the trainer GPU pool idle while rollout is still ongoing. Asynchronous RL pipelines overlap the two stages, but at the cost of training on stale data. To address these challenges, we propose RolloutPipe, a post-training framework for disaggregated RLVR systems, which turns the fixed-weight rollout into a complete-group pipeline where trainable groups move to the trainer while later groups are still being generated. RolloutPipe achieves this through two techniques including complete-group pipelining (CGP) and frontier-group dispatch (FGD). CGP dispatches each trainable complete group to the trainer FIFO as soon as group materialization finishes, and FGD is an admission policy on the Rollout node that first admits requests for the frontier groups needed to form the next training batch, so that trainer-ready groups arrive earlier and more steadily. The design starts training before the rollout completes while maintaining on-policy correctness. Evaluated on Qwen3-1.7B across four reasoning and science benchmarks and twelve rollout settings, RolloutPipe shortens the rollout-to-train-end time by 30.7%-42.3%, and lowers the trainer waiting ratio by 37%-76% compared to Slime, a state-of-the-art rollout and training system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26978v1">To Run or Not to Run: Analyzing the Cost-Effectiveness of Code Execution in LLM-Based Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted to the ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2026). 23 pages
    </div>
    <details class="paper-abstract">
      LLM-based agents for program repair are increasingly built on a "generate-run-revise" paradigm, iteratively executing tests to evaluate and refine patches. This execution-based approach has become standard practice in state-of-the-art systems. However, executions can be time-consuming and expensive, yet their impact on these agents remains underexplored. In this paper, we conduct a two-stage empirical study over execution behavior in LLM-based program repair. To characterize execution behavior at scale, we first analyze 7,745 agent traces from SWE-bench leaderboard submissions. Second, we evaluate 3,000 end-to-end repair attempts across 200 SWE-bench instances and three agents (Claude Code, Codex, and the open-source OpenCode) under four execution paradigms, which allows for a fine-grained comparison of performance and cost. Our analysis reveals three key observations: (1) Code execution is used across all agents and models analyzed, with an average of 8.8 test runs per task. Execution behavior varies substantially across agents and models, with frequency ranging from 2 to 19 per task, and late-stage executions consistently achieve higher success rates than early-stage ones. (2) Execution restrictions have little effect on repair success: on commercial agents with SOTA models the resolve-rate gap between Prohibited and Unrestricted is only 1.25 percentage points and not statistically significant, while Prohibited saves substantial token and wall-clock cost. (3) Execution benefit is concentrated rather than uniform. These patterns suggest that current agents apply execution indiscriminately, paying its cost on instances where it provides little benefit. Execution, therefore, should be treated as a resource with an explicit cost-benefit tradeoff, not a default capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19852v2">Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 7 pages, 2 figures, 3 tables. Affiliations: (1) Department of Health Outcomes and Biomedical Informatics, College of Medicine, University of Florida, Gainesville, FL, USA (2) Division of Pulmonary, Critical Care and Sleep Medicine, Department of Medicine, College of Medicine, University of Florida, Gainesville, FL, USA (3) College of Nursing, Florida State University, Tallahassee, FL, USA
    </div>
    <details class="paper-abstract">
      Information extraction from pathology reports is essential for cancer staging, tumor registry population. Yet key data remains embedded in narrative reports, making manual extraction labor-intensive and error-prone. Traditional supervised Natural Language Processing pipelines address this through fully supervised Named Entity Recognition and Relation Extraction, but require expensive manual annotation and suffer cascading failures when upstream entities are missed. In this study, we developed a zero-shot, agentic workflow, and evaluated five open-source generative Large Language Models (LLMs) to populate 13 College of American Pathologists synoptic fields from lung resection pathology reports. We compared them against a state-of-the-art supervised GatorTron NER-RE baseline using a novel, registry-aligned evaluation framework. The baseline achieved Micro-F1of 0.960, while the best zero-shot model (GPT-OSS-20B) achieved Micro-F1 of 0.893 (recall: 0.949), accurately extracting complex relations like Pathologic Stage without task-specific training. These results suggest that open-source, zero-shot agentic LLMs show great potential as a low-cost solution for extracting lung pathology information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10379v2">Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 9 main text pages, 36 total pages, Accepted at ICML 2026 AI for Math workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become capable mathematical problem-solvers, often producing correct proofs for challenging problems. However, correctness alone is not sufficient: mathematical proofs should also be clear, concise, insightful, and transferable to other problems. While this proof quality is subjective and depends on the reader and context, many of its components are concrete and broadly valued. In this work, we identify such components and introduce ProofRank, a benchmark curated from challenging mathematical competitions. ProofRank evaluates several scalable proxies of proof quality: (i) conciseness, measuring whether proofs avoid unnecessary steps; (ii) computational ease, measuring the extent to which a proof relies on tedious calculations; (iii) cognitive simplicity, measuring how accessible the used proof techniques are; (iv) diversity, measuring how varied a model's proofs for a single problem are; and (v) adaptivity, measuring whether a model can follow a specified proof technique. Across models, we find substantial differences in proof quality that are not captured by correctness-only benchmarks. We also observe significant trade-offs between proof-quality metrics and correctness, suggesting that future evaluations of mathematical reasoning should measure how useful LLM-generated proofs are.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26937v1">Continuous Behavioral Synthesis for Adaptive Health Dashboards: An LLM-Mediated Architecture Integrating Explicit Preference, Spatial Reorganization, and Attention Allocation Signals</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 33 pages, Accepted EICS2026 Patras, Greece
    </div>
    <details class="paper-abstract">
      The engineering of adaptive user interfaces has traditionally relied on either rule-based systems encoding designer intuitions about user needs or machine learning approaches requiring substantial historical data before achieving effective personalization. We present a technical architecture that leverages Large Language Models as behavioral synthesis engines to enable immediate adaptation from sparse, heterogeneous user signals. Our system integrates three distinct behavioral channels, i) explicit micro-feedback on individual interface elements, ii) spatial priority inferred from manual widget reorganization through drag-and-drop interaction, iii) and attentional investment measured through dwell time during hover events, within a structured prompt engineering framework that continuously regenerates dashboard layouts while maintaining explanatory coherence. The architecture addresses the technical challenge of translating low-level interaction patterns into high-level design decisions through a layered prompt construction methodology that separates temporal context determination, behavioral signal extraction, explicit preference enforcement, and user profile synthesis. The approach combines manually specified behavioral interpretations and temporal heuristics with LLM-mediated synthesis, enabling the reconciliation of multiple simultaneous signals that would be difficult to encode through explicit rules alone. We demonstrate the system through an instantiation in the personal health monitoring domain, including an analytical evaluation of adaptation behavior across multiple scenarios and a working implementation managing fourteen distinct health metrics across seven widget visualization modalities. The evaluation compares profile-driven initialization, multi-signal behavioral adaptation, and presents the resulting interfaces through representative post-adaptation screenshots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26927v1">Are LLMs Ready for Anti-Pattern Detection in Microservice Architectures?</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 accepted at ICSME 2026
    </div>
    <details class="paper-abstract">
      Microservice systems are prone to recurrent architectural anti-patterns (APs) that hinder maintainability, evolvability, and operational quality. Most existing AP detection approaches rely on static analysis and handcrafted rules, which can be effective but are often tool-dependent, limited to explicitly encoded detection logic, and difficult to adapt to heterogeneous repositories. In this paper, we investigate whether large language models (LLMs) are ready to support architectural anti-pattern detection in microservice architectures through a prompt-based analysis pipeline over static repository artifacts. We evaluate three general-purpose LLMs on a curated benchmark of microservice repositories annotated with 16 architectural anti-patterns, and compare their performance against the state-of-the-art static-analysis tool MARS using a uniform evaluation protocol based on precision and recall. Our results show that LLMs can provide useful support for anti-pattern detection, achieving competitive performance on several anti-patterns, especially when the relevant evidence is local, heterogeneous, or semantically rich. At the same time, they exhibit clear limitations on anti-patterns that require explicit structural or cross-service dependency evidence, where static analysis remains more reliable. These findings suggest that LLMs are not yet a replacement for traditional analyzers, but already represent a promising complementary aid for architectural assessment in microservice systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26924v1">A Deterministic Control Plane for LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 45 pages, 9 figures, 13 tables. Dataset and reproduction scripts: Zenodo DOI 10.5281/zenodo.20780913. Ancillary files include report.json, organizations.txt, and figure-reproduction scripts
    </div>
    <details class="paper-abstract">
      LLM coding harnesses grant agents broad file and shell access, yet the configuration layer that steers them -- rules files, agent definitions, IDE-specific markdown -- is largely unmanaged. A prevalence study of 10,008 public GitHub repositories (n=6,145 agent config files) finds that agent configurations propagate as undeclared shared components: 10.1% of tracked paths are SHA-256 exact duplicates across independent repositories (fork-adjusted, threshold-independent), with 75.5% of clone pairs crossing organisational boundaries. Two further patterns are indicative: configurations are rarely revised (58% single-commit; 0.4 vs 0.6 commits/month age-normalised against CI/CD workflows), and rarely declare permission boundaries (<1% of agent configs vs 33% of Actions workflows, n=31 true positives). We propose a deterministic control plane above the harness that maps one-to-one to these gaps. Rel(AI)Build treats agent definitions as a managed supply chain (SHA-256 content addressing, HMAC-stamped lockfiles, hash-chained audit logs); enforces tiered permissions and attack-derived blocklists before LLM invocation; gates feature work through a phase state machine with requirement-to-file-to-test traceability; compiles a single canonical definition to seven IDE targets; and detects prompt drift via Jaccard similarity. Conformance tests on injected violations confirm each mechanism enforces its stated invariant; developer outcomes remain future work. Governance of this layer must be deterministic and tool-agnostic -- not delegated to further LLM orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26917v1">GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted as a conference paper at ICML 2026
    </div>
    <details class="paper-abstract">
      Online reinforcement learning is widely used to align large language models (LLMs) with reward signals, yet training can be unstable under noisy or misspecified rewards. We identify a failure mode we call directional inconsistency: within a batch, a small set of high-reward rollouts induces representation-space preference directions that sharply disagree with the batch majority, resulting in high-variance and destabilizing updates. We propose geoalign, a lightweight plug-in for rollout curation in iterative policy optimization. Geoalign (i) forms within-prompt preference pairs, (ii) learns an online projector on per-rollout hidden states to concentrate reward-ordered displacement directions, and (iii) detects directionally inconsistent rollouts via their angular deviation from a batch consensus prototype and rectifies them with within-prompt stable alternatives. Geoalign is forward-pass only and adds negligible overhead. Across dialogue alignment with a learned reward model and mathematical reasoning with binary verified rewards, Geoalign improves final performance and reduces training oscillation, outperforming PF-PPO, PAR, PODS, and Seed-GRPO. These results suggest latent directional consensus as an effective reliability signal for online LLM RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26884v1">MedSWFlow: An Open-Source LLM Workflow for Drafting Medical Social Work Case Plans</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 26pages, 8tables, 2figuers
    </div>
    <details class="paper-abstract">
      We present MedSWFlow, an open-source, model-agnostic LLM workflow for drafting medical social work case plans. The framework translates professional case-planning tasks into six stages: assessment, problem analysis, goal setting, intervention planning, risk anticipation, and planned effect evaluation. Drawing on established social work and behavioral frameworks, MedSWFlow standardizes case inputs, builds structured case profiles, and generates reviewable assessment forms and service plans through staged prompting. The system is released as an open-source research framework for reproducible case-plan generation across LLM providers. Outputs are intended as practitioner-reviewed drafts rather than final service decisions. Source code: https://github.com/santhiyacw-droid/MedSWFlow/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26883v1">EconSimulacra: A Digital Twin Platform of Socio-Economic Systems Powered by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Real-world social behavior emerges from tightly coupled domains: economic conditions shape mobility and social interactions, while online attention and offline activity feed back into local popularity and consumer behavior. Capturing these feedback loops requires artificial societies in which agents carry experiences from one domain into decisions in another. Large language models (LLMs) provide a promising foundation for such societies. However, existing LLM-based simulators typically model domains in isolation or merely place them side by side. To enable such cross-domain interactions, we present EconSimulacra, a multi-agent social simulator that couples consumer economy, mobility, and social networks through a shared internal-state mechanism. In EconSimulacra, experiences accumulated across different domains are stored in memory and transformed into shared internal states (i.e., stress level) connecting heterogeneous domains through individual decision making. This design allows agents to reconcile competing demands arising from multiple domains and generate coherent cross-domain behaviors. As a case study, we show that the shared internal state mechanisms reproduce a nonlinear relationship between online social attention and offline local popularity, illustrating how realistic cross-domain dynamics can emerge within a unified artificial society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26861v1">Cascaded Multi-Granularity Pruning for On-Device LLM Inference in Industrial IoT</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 This work has been submitted to the IEEE Internet of Things Journal for possible publication
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on Industrial Internet of Things (IIoT) edge devices demands extreme compression, yet existing structured pruning methods collapse at high compression ratios due to one-shot importance estimation, and their cross-architecture behavior remains unpredictable. This article presents a cascaded multi-granularity pruning framework that removes layers, attention heads, and feed-forward channels in coarse-to-fine order, with lightweight low-rank recovery between stages to re-estimate component importance. An information-theoretic analysis motivates this ordering, and the Structural Independence Assumption (SIA) is formalized as a checkable condition predicting whether per-component pruning criteria are reliable for a given architecture: Multi-Head Attention (MHA)+GELU designs satisfy the SIA, whereas Grouped Query Attention (GQA)+SwiGLU designs violate it. On bearing fault diagnosis spanning 88M to 6.25B-parameter models, the framework extends achievable compression to 13.8 times on MHA+GELU architectures with 83.82% accuracy (+3.70 percentage points (pp) over the strongest baseline), while exposing a ~74pp accuracy collapse on GQA+SwiGLU architectures that violate the SIA. Deployed on an industrial slewing bearing fault diagnosis platform with NVIDIA DGX Spark, compressed models reduce inference latency by up to 67.2% and peak memory by 62.5%, demonstrating viability for IIoT edge inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15877v2">Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge, extracting reusable knowledge from interaction traces, yet a citation analysis of 1{,}136 references across 22 primary papers reveals a cross-community citation rate below 1\%. We propose the \emph{Experience Compression Spectrum}, a unifying framework that positions memory, skills, and rules as points along a single axis of increasing compression (5--20$\times$ for episodic memory, 50--500$\times$ for procedural skills, 1{,}000$\times$+ for declarative rules), directly reducing context consumption, retrieval latency, and compute overhead. Mapping 20+ systems onto this spectrum reveals that every system operates at a fixed, predetermined compression level: none supports adaptive cross-level compression, a gap we term the \emph{missing diagonal}. We further show that specialization alone is insufficient (both communities independently solve shared sub-problems without exchanging solutions), that evaluation methods are tightly coupled to compression levels, that transferability increases with compression at the cost of specificity, and that knowledge lifecycle management remains largely neglected. We articulate open problems and design principles for scalable, full-spectrum agent learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26787v1">AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted by KDD 2026 Applied Data Science Track (Oral presentation)
    </div>
    <details class="paper-abstract">
      Traditional dynamic pricing models in large-scale e-commerce suffer from limited interpretability, poor utilization of unstructured information, and misalignment with long-term business objectives such as cumulative Gross Merchandise Value (GMV), Return on Investment (ROI) and milestone achievement. We propose AIGP, a novel framework that leverages a Large Language Model (LLM) prompted with domain knowledge, structured data and textual context to make interpretable, knowledge-aware pricing decisions. For efficient deployment while maintaining high-quality outputs, we employ supervised fine-tuning for knowledge distillation. Central to AIGP is the Long-Term Value Estimator (LTVE), trained via offline reinforcement learning on historical data, which serves as a reward model to score candidate pricing actions and select preference pairs for Direct Preference Optimization (DPO), thereby aligning the pricing policy with long-term business objectives. Extensive offline evaluations and large-scale online A/B tests on Tao Factory demonstrate that AIGP achieves significant improvements: +13.21% in GMV, +7.59% in ROI, and +8.20% in milestone achievement rate over 14 days compared to the production baseline, while simultaneously providing interpretable and transparent pricing rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26698v1">Beyond Logical Forms: LLM-Extracted Patterns for Fallacy Classification</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      In today's fast-paced information era, logical fallacies, defined as defective patterns of reasoning, inevitably contribute to the growth of information disorder. However, often fallacies appear in nuanced forms that complicate automated classification. In this study, we investigate whether merging abstract logical structures with context-level linguistic cues proves beneficial for fallacy classification, developing a framework that inductively extracts such patterns from fallacious examples and their explanations using Large Language Models (LLMs). We evaluate the impact of these patterns across different LLMs and experimental zero- and one-shot configurations, showing statistically significant improvements over zero-shot baselines and outperforming competing approaches. Cross-dataset experiments validate generalization, establishing data-driven pattern extraction as an effective method for generating logical representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19576v2">Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted to the ICML 2026 Workshop on Failure Modes in Agentic AI (FAGEN@ICML 2026), Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom (LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)), yet the underlying mechanism has not been isolated. We provide (1) a \textbf{reproducible trigger}: ablations that isolate drift: one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) \textbf{trace-level diagnostics}: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a \textbf{verified fix}: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26666v1">PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 7 pages, 3 tables; workshop paper
    </div>
    <details class="paper-abstract">
      Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce KV-cache fragmentation and mature kernels such as FlashInfer provide highly optimized native-paged decode attention. However, the best single-kernel implementation is not always the best serving schedule: low-active long-context decode can under-utilize commodity GPUs, while mixed sequence lengths introduce a tension between many exact-length launches and coarse padded batches. We present PersistentKV, a native block-table decode attention engine and page-aware scheduling study for grouped-query attention (GQA). PersistentKV maps work by KV-head group, is designed to reuse K,V tiles across grouped query heads, supports native page tables, and adds a compact workqueue schedule that executes only non-empty row-KV-head-sequence-split tasks. On an RTX 3060 with FP16, page size 16, Hq=32, Hkv=8, d=128, and identical correctness tolerance against FlashInfer, a calibrated adaptive policy selects FlashInfer for small active batches, PersistentKV sequence splitting for B1 long-context steps, and PersistentKV workqueue scheduling for B8 long-context steps. With thresholds and split counts fixed on calibration traces, one held-out trace seed improves synchronized wall throughput by 1.063-1.265x on B8 bimodal, uniform, and Zipf-like workloads and by 1.399x on a B1 bucketed trace. On the B4 bimodal boundary case, the policy avoids the PersistentKV regression by selecting FlashInfer. These results identify a concrete systems niche for adaptive page-aware decode scheduling and show that work assignment, not only attention math, is a decisive serving-system variable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26650v1">CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 This work is accepted to ICML 2026 as an oral. The project page: https://github.com/IntelChina-AI/BitTern
    </div>
    <details class="paper-abstract">
      In this paper, we present CAT-Q, Cost-efficient and Accurate Ternary Quantization, for compressing and accelerating LLMs. Unlike existing state-of-the-art ternary quantization methods that rely on data-intensive and costly quantization-aware training to mitigate severe performance degradation, CAT-Q is a simple yet effective post-training quantization scheme that is readily applicable to LLMs with diverse architectures and model sizes. It has two key components, learnable modulation (LM) and softened ternarization (ST), which are coupled from an optimization perspective. LM leverages a composition of learnable factors to modulate the distribution of pre-trained high-precision weights and the ternary threshold, making them less sensitive to ternarization. ST further introduces a differentiable transition function to guide the ternarization process toward stable convergence. We show that, for pre-trained LLMs with 1.7B to 8B parameters, CAT-Q can efficiently quantize them into ternary models using only 512 calibration samples, while achieving superior performance than the seminal BitNet 1.58-bit v1 and v2 families (with 1.3B to 7B parameters) trained with 100B tokens, yielding about a 100,000X reduction in training tokens. Moreover, we show for the first time that CAT-Q can quantize much larger pre-trained LLMs having 14B to 235B parameters into leading ternary models within just 8 to 60 hours on 8 A100-80GB GPUs. Code is available at https://github.com/IntelChina-AI/BitTern.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26627v1">Agents That Know Too Much: A Data-Centric Survey of Privacy in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 17 pages, 4 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language model agents increasingly query databases, search document collections, call external APIs, remember past interactions, and act on a user's behalf. As they move from answering questions to operating over sensitive data, privacy becomes harder to enforce. An agent touches many data sources, runs multi-step workflows, keeps state across sessions, and acts with delegated permissions. Sensitive information can therefore leak not only through its final answer but through the queries it issues, the intermediate results it handles, the memory it writes, and the messages it exchanges with other agents. We survey the privacy of LLM agents from a data-centric view, organizing the field around the data an agent touches rather than by attack type, and we use data agent as shorthand for an LLM agent that works with data. Research on these risks is active but scattered across retrieval-augmented generation, text-to-SQL interfaces, agent memory, prompt injection, access control, and contextual privacy. This survey brings that work together: we taxonomize the data sources an agent touches, the privacy risks each source creates, and the governance mechanisms that address them; we map the benchmarks used to measure these risks and identify what is missing; and we set out the open problems. Two findings recur: among governance mechanisms only information-flow control covers both compositional and cross-session inference leakage, the two least-protected risks; and no benchmark drives an agent across its data surfaces under one privacy policy, the instrument the field most lacks. Our goal is a reference that situates the scattered literature and gives future work a common framing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26595v1">LLM-based Models for Detecting Emerging Topics in Service Feedback</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Enhancing the analysis of service feedback is essential for public sector organizations, particularly tax administrations, where trust and compliance depend on fair and effective service delivery. As feedback volumes grow, identifying emerging service quality issues and potential disparities across diverse populations becomes increasingly challenging. Traditional approaches often rely on manual review or static expert-defined indicators, limiting scalability and the ability to capture complex patterns in textual feedback. This paper presents a novel methodology that integrates large language models (LLMs), statistical techniques, and human-AI collaboration to improve multilingual customer feedback analysis. The primary objective is to detect emerging service quality topics that may also reveal potential inequities in service delivery. Our framework combines fine-tuned, quantized LLMs with expert oversight to produce accurate, computationally efficient, and context-aware analyses. The proposed approach was evaluated using similarity analysis and assessments from experienced tax officers, demonstrating stronger alignment with expert judgments than baseline models. By incorporating a human-in-the-loop framework, the methodology reduces LLM fabrication while improving the reliability and relevance of generated insights. The results demonstrate the practicality of combining LLMs with human expertise to support scalable, evidence-based decision-making in public sector organizations. This work contributes to the development of responsible AI systems that enhance service quality, responsiveness, fairness, and public trust through more effective analysis of multilingual customer feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14340v2">Refining Pseudo-Audio Prompts with Speech-Text Alignment for Text-Only Domain Adaptation in LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted at Interspeech 2026
    </div>
    <details class="paper-abstract">
      LLM-based automatic speech recognition models demonstrate strong performance by connecting audio encoders and LLMs. However, data scarcity of paired speech and transcription often hinders their adaptation to new domains, making text-only domain adaptation crucial. Existing methods typically rely on either fine-tuning the LLM alone or employing pseudo-audio prompts. The former neglects essential acoustic context, while the latter either suffers from limited scalability in data-scarce conditions, or yields inexpressive prompts by leveraging only textual features, ignoring audio modality. To address this, we propose an enhanced framework that explicitly models speech-text alignment. Our method efficiently generates highly expressive pseudo-audio prompts that bridges the modality gap, enabling effective target-domain adaptation. Experiments demonstrate that our approach outperforms existing text-only methods, improving both overall error rates and out-of-vocabulary coverage.
    </details>
</div>
