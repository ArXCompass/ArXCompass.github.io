# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22063v2">RedCoder: Automated Multi-Turn Red Teaming for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for code generation (i.e., Code LLMs) have demonstrated impressive capabilities in AI-assisted software development and testing. However, recent studies have shown that these models are prone to generating vulnerable or even malicious code under adversarial settings. Existing red-teaming approaches rely on extensive human effort, limiting their scalability and practicality, and generally overlook the interactive nature of real-world AI-assisted programming, which often unfolds over multiple turns. To bridge these gaps, we present RedCoder, a red-teaming agent that engages victim models in multi-turn conversation to elicit vulnerable code. The pipeline to construct RedCoder begins with a multi-agent gaming process that simulates adversarial interactions, yielding a set of prototype conversations and an arsenal of reusable attack strategies. We then fine-tune an LLM on these prototype conversations to serve as the backbone of RedCoder. Once deployed, RedCoder autonomously engages Code LLMs in multi-turn conversations, dynamically retrieving relevant strategies from the arsenal to steer the dialogue toward vulnerability-inducing outputs. Experiments across multiple Code LLMs show that our approach outperforms prior single-turn and multi-turn red-team methods in inducing vulnerabilities in code generation, offering a scalable and effective tool for evaluating the security boundaries of modern code-generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02464v1">Will Scaling Improve Social Simulation with LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) social simulations are a promising research method, but they are not yet faithful enough to be adopted widely. In this work, we investigate whether the current scaling paradigm in language modeling is likely to close these gaps, or whether simulation fidelity is orthogonal to general capabilities and therefore deserving of more research attention. We use scaling laws to study the relationship between LLMs' compute scale, general capability benchmarks, and the fidelity of social simulation in three representative sub-domains: opinion modeling, behavioral simulation, and longitudinal forecasting. Surprisingly, we discover strong compute scaling in all three settings, using a suite of 85 transformer LLMs with the Qwen3 architecture pre-trained on the DCLM web text corpus under fixed-compute budgets from $10^{18}$ to $10^{20}$ FLOPs. Then we evaluate 35 larger and more capable open-weight models up to 70B parameters, allowing us to predict downstream accuracy from loss. This reveals that the majority of behavioral and opinion simulation tasks will rapidly improve with scale, particularly when they involve populations that are well-represented in English web corpora. Longitudinal forecasting and underrepresented opinions scale more slowly, especially when they are less correlated with general knowledge and reasoning benchmarks like MMLU. In behavior simulation, scaling fails to improve model calibration with human cognitive biases like risk aversion, as well as human heuristics like learning correlated rewards from related tasks. On these tasks, even fine-tuned models fail to noticeably scale up performance from 0.5B to 8B parameters. Taken together, we conclude that scale will improve social simulations in most settings, but outliers exist, and improvements will be less reliable in low-resource domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02460v1">Neuron-Aware Data Selection for Annotation-Free LLM Self-Distillation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Post-training large language models (LLMs) without real-world interaction feedback or human-labeled supervision remains challenging, particularly in specialized domains where expert annotations are costly to obtain. Recent annotation-free self-evolution methods address this by using the model's own outputs as supervision signals, constructing a teacher via additional context and aggregating predictions across multiple rollouts through majority voting to produce pseudo-labels. However, these approaches are not without drawbacks: SFT- and GRPO-based variants suffer out-of-domain performance degradation, while reward-based on-policy RL inflates calibration error. In this paper, we propose Neuron On-Policy Self-Distillation (Neuron-OPSD), a data-centric framework for annotation-free self-distillation that leverages internal neuron activations to guide both training-data selection and teacher context construction. The model is then trained via on-policy distillation from the teacher distribution, requiring no ground-truth labels at any stage. Across specialized-domain benchmarks, Neuron-OPSD improves in-domain task performance while preserving cross-domain generalization and mitigating calibration collapse over prior annotation-free baselines. This framework is particularly relevant to settings where online interaction or external supervision is costly or infeasible, and is conceptually distinct from offline RL approaches that rely on logged, reward-labeled trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02455v1">When Do LLM Personas Support Visualization Design? A Cross-Model Study of Color Assignment and Chart Choice</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 5 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language model personas are increasingly used to approximate diverse users during early-stage visualization design, but it remains unclear whether persona-conditioned outputs reflect stable personality effects or artifacts of model choice and task framing. We examine this question across two visualization-relevant tasks: color assignment for abstract and concrete concepts, and chart-idiom preference ratings across task contexts. Using 43 Big Five profiles across GPT-4o-mini, GPT-4.1-mini, and GPT-5-mini, we find that personality-color coupling is highly model-configuration dependent: absent in GPT-4o-mini for all six concepts, consistent in GPT-4.1-mini across all six, and partial in GPT-5-mini for two of six. Concept type further shapes the signal: for abstract concepts, personality explains more hue variance than model identity, while concrete concepts show smaller and comparable effects. In chart choice, trait-aligned cluster aggregation produces stable top-idiom rankings across all nine cluster-context combinations, but a no-persona baseline recovers the same top choice in 8 of 9 model-context cells, indicating that task context drives rank-1 selection more than personality. These findings position LLM personas as exploratory probes for visualization design, not substitutes for human participants, and motivate multi-model testing, concept-type disaggregation, and no-persona baselines in future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02448v1">AgentsCAD: Automated Design for Manufacturing of FDM Parts via Multi-Agent LLM Reasoning and Geometric Feature Recognition</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Parts manufactured with Fused Deposition Modeling (FDM) often require Design for Additive Manufacturing (DFAM) modifications to ensure printability, structural integrity, and reduced post-processing. Current slicers identify defects such as steep overhangs but are unable to modify the underlying geometry. This work presents AgentsCAD, a multi-agent system that bridges raw boundary-representation (B-Rep) geometry and Large Language Model (LLM) reasoning to automate targeted DFM. The workflow begins by parsing a STEP file. The agentic system detects overhangs above a 45°threshold, constructs a face-adjacency topology graph, and optionally injects semantic feature labels from a GraphSAGE model trained on MFCAD++ (59,665 parts), before dispatching a Claude Sonnet design-reasoning agent that recommends reorientations, fillets, chamfers, and similar modifications. A GPT-4o vision-language verifier inspects rendered views to confirm geometric integrity. Outputs include a modified STEP file and a human-readable report. A test case on a birdhouse model demonstrates that the system correctly diagnoses overhangs, selects appropriate defect mitigation strategies, and proposes physically valid corrections, partially solving the geometry-to-language translation problem central to LLM-driven CAD modification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02091v2">Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Rerankers play a pivotal role in refining retrieval results for Retrieval-Augmented Generation. However, current reranking models are typically optimized on static human annotated relevance labels in isolation, decoupled from the downstream generation process. This isolation leads to a fundamental misalignment: documents identified as topically relevant by information retrieval metrics often fail to provide the actual utility required by the LLM for precise answer generation. To bridge this gap, we introduce ReRanking Preference Optimization (RRPO), a reinforcement learning framework that directly aligns reranking with the LLM's generation quality. By formulating reranking as a sequential decision-making process, RRPO optimizes for context utility using LLM feedback, thereby eliminating the need for expensive human annotations. To ensure training stability, we further introduce a reference-anchored deterministic baseline. Extensive experiments on knowledge-intensive benchmarks demonstrate that RRPO significantly outperforms strong baselines, including the powerful list-wise reranker RankZephyr. Further analysis highlights the versatility of our framework: it generalizes seamlessly to diverse readers (e.g., GPT-4o), integrates orthogonally with query expansion modules like Query2Doc, and remains robust even when trained with noisy supervisors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16662v2">Evaluating Collective Behaviour of Hundreds of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      LLM-powered AI assistants acting on behalf of users can produce poor collective outcomes at scale. We introduce a framework for evaluating their emergent behaviour in social dilemmas, applied to three iterated games (Public Goods, Collective Risk, Common Pool Resource). We prompt each model to produce a natural-language strategy, then have the same model translate it into code. This aims to isolate strategic reasoning from input-parsing, enables pre-deployment inspection, and scales to populations of hundreds of agents. We propose three analyses: behavioural fingerprinting via exhaustive evaluation over opponent histories; self-play robustness across mixtures of a model's strategies with either a Selfish or Collective disposition; and cultural evolution under payoff-biased imitation. Applied to three state-of-the-art LLMs, we find substantial cross-model differences in self-play welfare, and that cultural evolution converges to low-welfare, Selfish-dominant equilibria in larger groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02423v1">Neuron-Aware Active Few-Shot Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Active Few-Shot Learning (AFSL) adapts LLMs to specialized domains by identifying the most valuable unlabeled samples for annotation and use as few-shot demonstrations, effectively reducing human annotation costs while promoting high performance. However, existing methods typically rely on output-level signals for sample identification, such as predictive entropy or semantic similarities with test-time data based on external embeddings, which often overlook models' internal dynamics, which could pinpoint specific knowledge gaps. To bridge this gap, we propose NeuFS, a Neuron-Aware Active Few-Shot Learning framework that shifts the selection paradigm from output-level proxies to models' internal dynamics. NeuFS utilizes neuron activation patterns to represent sample directly, and includes a dual-criteria selection strategy that: (1) ensures few-shot sample diversity with neuron patterns for broader example coverage, while (2) prioritizing on identifying informative and challenging few-shot samples LLMs tend to hallucinate by quantifying neuron consensus. Experiments on three datasets demonstrate that NeuFS excels in both reasoning and text classification tasks, outperforming existing AFSL baselines. Ablation studies further highlight that internal neuron activations provide a more principled and effective selection signal than external embeddings, validating the superiority of the proposed NeuFS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02391v1">WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted at 1st Workshop on Sustainability and Resource-Efficiency of Artificial Intelligence @ IJCAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference workloads are a rapidly growing contributor to data center energy consumption. Optimizing these deployments requires matching specific LLMs to the most efficient GPUs, but operators currently lack the tools to do so without exhaustively profiling each combination. While some predictive models exist, they still require profiling data and struggle to generalize to hardware unseen during training. To address this, we introduce \textit{WattGPU}, featuring two predictive models for mean GPU power draw and Inter-Token Latency (ITL). Our approach leverages only publicly available LLM metadata and GPU specifications, eliminating the need for hardware access or profiling while enabling generalization to unseen NVIDIA server-grade GPUs and LLMs. We evaluate our models using rigorous leave-one-GPU-out and leave-one-LLM-out cross-validation on a dataset of 42 open-source LLMs (0.1B--27B parameters) and 8 GPUs under both offline and server scenarios. The mean power draw model achieves a median absolute percentage error of $\leq3.4\%$ for offline and $\leq13.5\%$ for server scenarios on unseen GPUs, while the latency model achieves $\leq8.5\%$ in server mode, both maintaining strong GPU ranking correlations for server scenarios (Kendall $τ\geq0.76$). Compared to standard physically grounded baselines -- Load-Scaled Thermal Design Power (TDP) for power draw and roofline for latency -- our models reduce median absolute percentage error by approximately 4$\times$ on unseen LLM-GPU combinations for server scenarios or approximately 2$\times$ for completely unseen GPUs. WattGPU's data and code are publicly available at https://github.com/maufadel/wattgpu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02368v1">The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Evaluations of LLM personas via psychometric questionnaires typically rely on aggregate scores, discarding within-instance correlation structure. We test whether this geometric structure is intrinsic or frame-dependent. Constructing within-instance correlation matrices from IPIP-50 responses, we analyze geometry on SPD manifolds under manipulated question orderings in GPT-4o simulating American and Chinese-American personas. We find that persona expression comprises two dissociable components: aggregated features (Big Five scores) degrade under randomization (21% drop) but are frame-robust; geometric features (SPD manifold) collapse under frame misalignment (42% drop) but recover substantially (to 84%) under shared frames, surpassing aggregated features (76%). This collapse-recovery pattern reveals that persona geometry is not intrinsic but a frame-dependent coordination pattern encoding information invisible to aggregation. Our findings establish a dual-nature framework for LLM personas, frame-dependent geometry versus frame-robust aggregates, necessitating frame-aware evaluation and challenging static trait conceptions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02333v1">Guiding Human Validation of LLM-Generated Code via Verifiable Literate Programming</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Vibe coding democratizes software development by allowing users to generate code via natural-language (NL) interaction with large language models (LLMs). However, the code is reliable only when it faithfully implements the user's intent, which is difficult and labor-intensive for users to validate. Existing validation methods either rely on LLM-assisted automated testing, which suffers from prompt ambiguity and model fallibility, or involve users only in partial software artifacts such as prompts and test cases, which may overlook corner cases and program details. Motivated by a bug study of LLM-generated code, we find that detailed human feedback is essential, as failures often stem from underspecified requirements or subtle semantic deviations. This paper presents verifiable literate programming (VLP), a human-in-the-loop framework designed to make the review/validation process of LLM-generated code accessible to users at all programming levels. At its core, VLP proposes unambiguous NL-based documentation as a readable intermediate layer between prompts and code. The documentation demonstrates concrete program semantics and enables users to provide feedback on potential intent-code mismatches. It supports human-involved, end-to-end repair and validation via three techniques: (i) an NL-style literate language with unambiguous syntax and mostly deterministic code-to-documentation translation, (ii) LLM-based fine-grained mismatch detection that uses trace links between prompts and documentation to focus users' review effort on suspicious documentation lines, and (iii) a verification module that leverages user-validated documentation to derive API-usage checks and formal properties, which are then verified against the generated code using model checking. Our evaluation shows that VLP improves code pass@1 from 28.7%-73.2% to 65.4%-93.5% with reasonable user effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02329v1">Grounded autonomous research: a fault-tolerant LLM pipeline from corpus to manuscript in frontier computational physics</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 39 pages, 5 figures. Accepted at the ICML 2026 AI for Science Workshop (https://openreview.net/forum?id=R5YXaPgUAx). Includes the pipeline-generated companion physics manuscript as an appendix. Data and scaffolding archive: https://doi.org/10.5281/zenodo.21126996
    </div>
    <details class="paper-abstract">
      Autonomous-research agents have demonstrated end-to-end LLM automation in machine-learning sandboxes where execution provides calibration. Frontier physical science differs categorically: physical reasoning underlies every methodology choice, toolchains are often underdocumented, and calibration must come from external literature anchors - which unscaffolded agents cite but do not confront, hallucinating plausible, unverifiable results from internal priors. We present a pipeline that runs end-to-end from a corpus of 11,083 recent condensed-matter physics arXiv papers to a publication-grade manuscript with three substantive physics findings (here on altermagnetic piezomagnetism): the agent autonomously conceives a research direction by mapping the corpus, calibrates methodology by reproducing published references, conducts novel first-principles computations, and writes the manuscript - grounded in literature throughout, across 47 fresh-context sessions in six phases sharing only on-disk state, with 2,162 literature-consultation events. Fault tolerance emerges from redundancy: fresh-context isolation, distributed grounding, and adversarial review catch what any single session misses; pre- and post-pilot stages are fully autonomous, and pilot requires bounded human intervention only at reproduction failures - operational knowledge curation, not scientific direction. Two paired failure modes - a pre-architecture baseline and a no-pilot ablation - isolate structurally enforced numerical confrontation at calibration checkpoints as the operative grounding mechanism. The primitives, characterized failure modes, and quantified intervention pattern lay a foundation for autonomous research in high-stakes scientific domains beyond computational physics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05952v2">Dean of LLM Tutors: A Framework for Automated Quality Review of AI-generated Feedback</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Large language model (LLM) tutors are increasingly used to generate educational feedback, but existing research has focused mainly on feedback generation rather than feedback evaluation. As a result, LLM-generated feedback may offer limited pedagogical value and carry risks of hallucination. The current study introduces DeanLLM, an automated review framework for comprehensively evaluating feedback generated by LLM tutors before it is shared with students. We developed a 16-dimension evaluation framework covering feedback content, educational effectiveness, and hallucination risks, and validated it using using human-expert annotations of LLM-generated tutor feedback on synthetic computer science assignment submissions derived from real coursework. We then examined whether LLMs could serve as automated LLM-generated tutor feedback reviewers, and used the best-performing reviewer to benchmark tutor feedback generated by 10 commercial LLMs. Psychometric analyses supported the reliability of the proposed framework and showed that human reviewers tended to evaluate feedback holistically, whereas the LLM reviewer separated rubric dimensions more mechanically. Standard zero-shot and few-shot prompting showed limited agreement with human experts for content-quality judgments. Supervised fine-tuning of GPT-4.1 with human-labelled examples containing scores only, without explanatory rationales, achieved the strongest alignment with expert judgments. Reasoning LLMs were particularly effective at hallucination detection and produced automated tutor feedback with stronger educational effectiveness and factuality than lightweight models. The findings indicate that DeanLLM offers a scalable way for automatically improving the reliability and safety of LLM tutor feedback, while also demonstrating that reviewer calibration and model choice remain critical for educational deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02255v1">AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Memory for a long-horizon LLM agent is a contract about what each future decision is allowed to see. The simplest contract appends past observations, tool calls, and reflections to every prompt, which makes prior context easy to access but also turns it into a jumbled mixture in which the effect of any single memory component is hard to isolate. We introduce and instrument an alternative bounded contract: every decision is made from a fresh user message assembled by typed retrieval, with no raw cross-decision transcript appended. The prompt thus stays bounded across runs of any length, and any single layer can be ablated in isolation. We instantiate the contract in Slay the Spire 2, a closed-rule stochastic deck-building game whose runs require hundreds of tactical and strategic decisions. A public online benchmark of frontier LLMs on the same game reports zero wins at the lowest difficulty across five configurations, and the developer-reported human win rate at the same difficulty is 16%; the task is hard but not saturated. Within our harness, a fixed-A0 ablation shows the largest observed difference when triggered strategic skills are enabled: the no-store baseline wins 3/10 games and adding the skill layer 6/10. At this sample size the comparison is directional rather than statistically decisive (Fisher exact p\approx0.37); a cross-backbone probe and public accumulating-context baselines are reported as operational comparisons rather than controlled tests of the contract variable itself. We release a reproducible testbed: 298 completed trajectories with condition tags, frozen memory/skill snapshots, prompt records, and analysis scripts -- an agent design and a validated, reusable methodology for studying how explicit memory layers shape long-horizon LLM-agent decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02235v1">Challenges and Recommendations for LLMs-as-a-Judge in Multilingual Settings and Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has become the dominant evaluation paradigm for many natural language generation tasks, due to shortcomings of conventional metrics and high correlations with human judgment, albeit mostly in English. There are now attempts to extend LLM-as-a-Judge to multilingual settings including low-resource languages. However, LLMs have limited proficiency in low-resource languages, and there is often no adequate human validation in these settings. To highlight the scope of the problem and current practices, we explore the use of LLM-as-a-Judge evaluators in ACL Anthology papers focusing on multilingual settings and low-resource languages across a diverse set of tasks. Out of 650 papers mentioning LLM-as-a-judge, only 33 of them focus on low-resource or multilingual settings. Our in-depth analysis of these papers indicates inconsistent evaluation outcomes, a tendency to overtrust LLM judgments in multilingual settings, and the widespread reliance on a single judge model per study. To help the NLP community further, we conclude with recommendations about how to use LLM-as-a-Judge in multilingual and low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03056v2">SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Scale</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      As LLM agents adopt large skill libraries, selecting the right subset becomes a structural problem rather than a similarity-matching one: skills depend on, conflict with, specialize, or duplicate one another, a structure invisible to both full enumeration and embedding similarity. We present SkillDAG, which models inter-skill relationships as a typed directed graph and exposes it to an LLM agent as an inference-time, agent-callable structural retrieval interface, queried and evolved during execution rather than baked into a fixed retrieval pipeline: each search returns vector matches, typed-edge neighbors, and conflict signals, and a propose-then-commit protocol lets the agent register execution-backed edges so the graph accumulates structure across episodes. On ALFWorld and SkillsBench with MiniMax-M2.7, SkillDAG reaches 67.1% success and 27.3% reward, exceeding the strongest reported Graph-of-Skills baseline by +12.8 and +8.6 points; the advantage ports to gpt-5.2-codex, and intrinsic SkillsBench Ret@K rises from 65.5 to 78.2 under matched queries. These gains trace to isolable mechanisms: candidate ranking that stays robust as the pool grows 10x where a fixed seeding-diffusion pipeline degrades, and set-monotone online edits that enlarge ground-truth recall without evicting prior hits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30441v2">Translating Natural Language to Strategic Temporal Specifications via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      A rigorous formalization of system requirements is a fundamental prerequisite for the verification of Multi-Agent Systems (MAS). However, writing correct formal specifications is well known as an error-prone, time-consuming, and expertise-intensive task. This difficulty is further accentuated in MAS, where requirements must capture strategic abilities and temporal objectives. At present, there is no established methodology for deriving MAS specifications from natural language. We present a framework for translating Natural Language descriptions of strategic requirements into well-formed ATL/ATL* formulas using Large Language Models (LLMs). Since no available dataset supports supervised learning for the NL-to-ATL/ATL* translation task, we create and curate a novel expert-validated dataset, employed for training and evaluating fine-tuned models. On a held-out test set, evaluated under the LLM judge that best agrees with expert annotations, in-domain fine-tuning of small open-weight models (3 - 7B parameters) matches strong few-shot proprietary API baselines. Our best fine-tuned system reaches 0.84 semantic accuracy, statistically on par with 0.86 for the strongest few-shot proprietary baseline, while keeping requirements on-premises. We further find that judge reliability is inverse to generator strength. The open-weight Llama-3.3-70B tracks human verdicts most closely, whereas the strongest proprietary models are the least reliable judges, over-rejecting faithful paraphrases of the reference. To assess the practical applicability of the generated specifications, we embed our tool to an existing strategic logics model checker, enabling non-expert users to specify strategic properties in natural language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22710v2">AlienLM: Alienization of Language for API-Boundary Privacy in Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Modern LLMs are increasingly accessed via black-box APIs, requiring users to transmit sensitive prompts, outputs, and fine-tuning data to external providers, creating a critical privacy risk at the API boundary. We introduce AlienLM, a deployable API-only \cradd{exposure-reduction layer that reduces plaintext exposure} by translating text into an Alien Language via a vocabulary-scale bijection, enabling lossless recovery on the client side. Using only standard fine-tuning APIs, Alien Adaptation Training (AAT) adapts target models to operate directly on alienized inputs. Across four LLM backbones and seven benchmarks, AlienLM retains over 81\% of plaintext-oracle performance on average, substantially outperforming random-bijection and character-level baselines. Under adversaries with access to model weights, corpus statistics, and learning-based inverse translation, recovery attacks reconstruct fewer than 0.22\% of alienized tokens. Our results demonstrate a practical pathway for \cradd{privacy-aware} LLM deployment under API-only access, substantially reducing plaintext exposure while maintaining task performance. Code and data are available at https://github.com/KimJaehee0725/AlienLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02118v1">Enhancing Fitness Intelligence through Domain-Specific LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 8 pages, 6 tables, 2 figures. Accepted by the 12th International Conference on Big Data Computing and Communications (BigCom 2026)
    </div>
    <details class="paper-abstract">
      Scientific Fitness Coaching (SFC) is typically delivered by human professionals, making it costly and inaccessible to many. While recent advances in Large Language Models (LLMs) show considerable promise for more inclusive fitness coaching, directly deploying prevailing general-purpose LLMs in SFC reveals critical limitations. These models often lack sufficient domain-specific knowledge integration, leading to weak performance on complex SFC scenarios. In this paper, we introduce FitOne, a series of fitness LLMs (with 8B and 32B parameters) designed to improve reliability and domain specialization for SFC applications. Built upon the Qwen3 foundation models, FitOne is developed through a three-stage post-training pipeline consisting of continual pre-training, supervised fine-tuning, and reinforcement learning, using large-scale, high-quality datasets derived from rigorous knowledge engineering. We conduct comprehensive evaluations of FitOne on professional fitness certification exams, including ACSM-EP and NSCA-CSCS, as well as general capabilities such as knowledge reasoning and instruction following. Experimental results show that, while retaining strong general capabilities, FitOne-8B/32B achieves average improvements of up to 10.09%/9.29% and 12.73%/7.01% on the ACSM-EP and NSCA-CSCS exams, respectively, compared with the Qwen3 base models. Furthermore, in-depth ablation studies confirm the necessity of each training stage, highlighting the pipeline's effectiveness in balancing domain expertise enhancement with general ability retention. We believe this research advances LLM systems toward more reliable fitness intelligence and will inspire future research on developing domain-specific LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14331v3">LLM Priors for ERM over Programs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      We study program-learning methods that are efficient in both samples and computation. Classical learning theory suggests that when the target admits a short program description, for example a short piece of ``Python code'', it can be learned from few examples by ERM over the program class. However, this approach relies on enumerating candidate programs, which is typically exponential in the description length; gradient-based training avoids this explicit search but, for some families of short programs, can require exponentially many samples to succeed. We propose \textsc{LLM-PV}, a propose-and-verify recipe that enables ERM-style selection over a discrete program class without exhaustive enumeration: a pretrained LLM induces a proposal distribution over candidate programs, each proposal is executed and scored on a held-out validation set, and the best program is selected, with no gradient updates or validation feedback used to adapt the sampling distribution. Across algorithmic tasks including parity variants, pattern matching, and primality testing, \textsc{LLM-PV} often recovers the exact underlying rule from a small labeled set and generalizes far beyond the training sequence lengths, while SGD-trained transformers, fine-tuning, in-context learning, and classical ML baselines can fit the training data yet fail to generalize reliably. Together, these results suggest that pretrained LLM priors can serve as effective search biases for ERM, narrowing the gap between statistical and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02104v1">Ask the Right Comparison:Bias-Aware Bayesian Active Top-$k$ Ranking with LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as cheap, scalable judges that compare candidate outputs pairwise -- to rank responses, select models, or triage papers. Yet LLM judges are both noisy and systematically biased: they favor verbose or well-formatted answers and exhibit position effects, so simply aggregating their votes recovers a ranking of presentation, not of true quality. We study the practical goal of identifying the \topk{} items under a fixed comparison budget, and make two contributions. First, we cast judging as Bayesian inference over latent quality with explicit, judge-specific bias covariates (verbosity, position), regularized by a shrinkage prior so that the data decide which biases a given judge actually exhibits. Second, we introduce a \topk-aware active acquisition rule that chooses the next comparison to maximally reduce uncertainty about \topk{} \emph{membership}, rather than about the full ranking. On a controlled benchmark with known ground-truth quality, judged by sixteen real LLMs spanning open and proprietary families (Llama, Qwen, Phi-4, GPT-4o-mini/5.1/5.5, Gemini, DeepSeek, and Claude Haiku/Sonnet/Opus), naive aggregation plateaus at a wrong \topk{} on biased judges regardless of budget, while our bias-aware model recovers it; \topk-aware acquisition reaches this ceiling with far fewer comparisons than round-robin or a global-uncertainty (D-optimal) rule. Bias is real but heterogeneous and capability-dependent: cheap and mid-tier judges carry a strong verbosity bias that our model corrects (lifting recall from $\sim$$0.5$--$0.6$ to $0.84$--$1.0$), whereas the frontier judges we tested show little bias and already rank accurately, so bias-aware modeling changes little there.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02072v1">kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in domains requiring guardrails to detect unsafe, off-topic, or adversarial prompts. Existing guardrails predominately rely on fine-tuning to build classifiers, which often suffer from low generalization and high inference latency. We present kNNGuard, a training-free guardrail that utilizes the activation space of an off-the-shelf LLM. Given a small bank of 50 safe and unsafe prompts, kNNGuard extracts hidden activations and performs multi-layer kNN fusing activation-space and embedding-space scores for classification. Across six domains spanning topical and security prompts, kNNGuard achieves competitive or superior F1 compared to fine-tuned state-of-the-art guardrails while running 2.7x faster than the best comparable guardrail, and 10x faster than a fine-tuned safety classifier without gradient updates or fine-tuning. Domain adaptation requires only updating the labeled bank, which can be constructed in under 10 seconds and several orders of magnitude faster than established guardrails. We also analyze the impact of system prompts, layer selection, and integration into production LLM pipelines as a configurable, low-latency guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05536v2">Gravity-Awareness: Deep Learning Models and LLM Simulation of Human Awareness in Altered Gravity</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 60 pages, 5 figures, 2 datasets, 1 protocol
    </div>
    <details class="paper-abstract">
      Earth s gravity fundamentally shapes human behaviour. The brain encodes this force as an internal model of gravity, enabling the prediction and interpretation of gravitational effects during perception and action. Understanding how this model adapts to altered gravity is critical for predicting human performance in spaceflight. We present a computational framework for modelling neurophysiological adaptation across diverse gravitational environments. The framework has two components trained on open-access data from altered-gravity studies, particularly parabolic flights. The first component (CorticalG) employs a lightweight multilayer perceptron neural network to predict gravity-dependent changes in EEG frequency bands, estimating cortical state under different gravitational loads. The second component (PhysioG) uses independent Gaussian process models to capture broader physiological responses, including heart rate variability, electrodermal activity, and motor control. To complement the quantitative modelling, we simulated subjective experience across gravitational environments using the Large Language Model (LLM) Claude 3.5 Sonnet. Physiological outputs prompted the model to generate narratives describing alertness, bodily awareness, and cognitive state across zero gravity, partial gravity of the Moon and Mars, and hypergravity. This framework provides a novel approach for investigating human adaptation to spaceflight. It offers a predictive tool to assess performance and resilience, supporting the design of future space exploration missions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10485v2">From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that both graph neural network (GNN)-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by consistent data distributions and heuristic or partially noisy labels. In this study, we systematically evaluate two representative DL models-ReVeal and LineVul-across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each respective dataset, and their code representations are analyzed using t-SNE to uncover vulnerability related patterns. To assess realistic applicability, we deploy these models along with four pretrained LLMs, Claude 3.5 Sonnet, GPT-o3-mini, GPT-4o, and GPT-5 on a curated dataset, VentiVul, comprising 20 recently (May 2025) fixed vulnerabilities from the Linux kernel. Our experiments reveal that current models struggle to distinguish vulnerable from non-vulnerable code in representation space and generalize poorly across datasets with differing distributions. When evaluated on VentiVul, our newly constructed time-wise out-of-distribution dataset, performance drops sharply, with most models failing to detect vulnerabilities reliably. These results expose a persistent gap between academic benchmarks and real-world deployment, emphasizing the value of our deployment-oriented evaluation framework and the need for more robust code representations and higher-quality datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02049v1">SPLIT: Cross-Lingual Empathy and Cultural Grounding in English and Ukrainian LLM Responses</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 19 pages, 5 figures, 3 tables. Benchmark paper introducing SPLIT for evaluating empathy, linguistic naturalness, and cultural grounding in English and Ukrainian LLM responses
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly deployed in emotional-support contexts and crisis-related situations. Nevertheless, their cross-lingual abilities in these circumstances remain underexplored. Existing benchmarks emphasize multilingual performance but rarely examine crisis-related empathy and cultural grounding in low-to-mid-resource languages. We introduce SPLIT, a 500-prompt benchmark designed to evaluate LLM consistency in generating emotionally grounded responses across five categories: Stress, Panic, Loneliness, Internal Displacement, and Tension. We evaluate three technically diverse LLMs across three dimensions: Empathetic Accuracy, Linguistic Naturalness, and Contextual & Cultural Grounding. The framework aims to assess and compare the quality of LLM responses in both English and Ukrainian languages, as well as to explore the reliability of the LLM-as-a-jury paradigm. Our findings reveal that Gemini-2.5-Flash and LLaMA-3.3-70B-Instruct degrade when transitioning to Ukrainian, while DeepSeek-V3 remains comparatively stable within our benchmark. We additionally find that human and AI evaluators agree weakly on empathy and naturalness but diverge on cultural grounding. We further argue that producing Ukrainian text is not equivalent to producing Ukrainian emotional support. Our findings may assist in the future development of more culturally tailored benchmark designs, as well as encourage a stronger emphasis on human-centered evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02043v1">Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Disaggregated LLM serving runs prefill and decode on separate GPU pools to keep the two phases from interfering. In practice, this creates a new asymmetry: under bursty, heavy-tailed workloads prefill nodes saturate while decode nodes have compute underutilized, and on a production-style A100 cluster with 2 prefill and 2 decode nodes (2P2D), we find that prefill execution accounts for only 2-23% of P95 Time-to-First-Token (TTFT). Queuing and inter-node GPU-GPU KV-cache transfer account for the rest. We present a proactive prefill-deflecting scheduler that lets decode nodes serve prefill phase of requests as chunked-prefill steps interleaved with their in-flight decode batches. For each queued request, we estimate the TTFT it would see on the prefill node, and on every decode node, search for the largest chunk schedule that keeps in-flight decodes within their Time-Between-Tokens (TBT) SLO and deflect when the decode path helps tail latency. Because the prefill phase of deflected requests runs in place on the decode node, the inter-node KV transfer is eliminated. Implemented on vLLM and evaluated on production-style traces with DeepSeek-V2-Lite, our approach reduces P95 TTFT by upto 81% and raises SLO attainment by upto 79% over state-of-the-art disaggregated schedulers, at sub-millisecond per-request routing cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28749v2">Four Types of LLM Reliance and Their Predictors Among Undergraduate Writers: A Mixed-Methods Study at a Minority-Serving R1 University</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Although most undergraduates now use large language models (LLMs), a form of generative artificial intelligence (GenAI) for academic writing, no validated method distinguishes the qualitatively different ways students rely on them. Existing instruments assess reliance solely by frequency of use, a measure that, as this study shows, inadvertently rewards dependence on AI rather than recognizing students' own intellectual contribution. Conducted at a public minority-serving university and grounded in the AI Literacy Framework, Expectancy-Value Theory, and Biggs's Presage-Process-Product model, the study drew on 382 undergraduates, 14 interviews, and 396 open-ended survey responses. Four distinct reliance types were identified and confirmed: Strategic (34.3%), Instrumental (30.9%), Dialogic (30.4%), and Dependent (4.5%). Students' value and cost beliefs predicted the intensity of their reliance on LLMs, whereas their AI literacy predicted the type of reliance they adopted, indicating that differentiated support is needed. Notably, Strategic users, those who engaged AI most deliberately, scored lowest on standard outcome measures. This pattern reflects a limitation of current instruments, which index AI's contribution rather than writing quality, thereby penalizing students who show the greatest independent thinking. Analysis also revealed an additional group, roughly 13%, who declined to use AI for ethical rather than practical reasons, and who existing frameworks overlook. These findings carry implications for AI literacy programs, the measurement of student learning outcomes, and equitable AI policy at minority-serving institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01980v1">Epic-Organized vs. Requirement-Aligned Gherkin: An Empirical Evaluation of LLM-Based Acceptance Criteria Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted for publication at the International Conference on Software Engineering of Emerging Technologies (SEET 2026)
    </div>
    <details class="paper-abstract">
      Automated authoring of Gherkin Behavior-Driven Development (BDD) acceptance criteria remains a manual bottleneck in requirements engineering. This study investigates whether epic-organized LLM-generated Gherkin produces higher quality and coverage than requirement-aligned generation. We compare our Timeless (an epic-organized LLM pipeline) approach against a naive large language model (LLM) baseline on four requirements documents (107 requirements) from the PURE dataset. Evaluation covers structural metrics, automated requirement coverage via TF-IDF and dense embeddings, and blind expert assessment by four researchers. In our evaluation, the JSON-constrained pipeline produced structurally valid scenarios across all generated outputs, while the zero-shot baseline achieved 99% structural validity. Semantic coverage was comparable to the baseline, with Timeless achieving 94.3% semantic Requirement Coverage Rate compared with 92.9% for the baseline. TF-IDF produced lower coverage scores for the epic-organized output, suggesting that lexical metrics may miss coverage when scenarios paraphrase requirements at a higher level of abstraction. Expert raters prefer the epic-organized strategy on Correctness (4.61 vs 4.14), Executability (4.61 vs 4.07), and Completeness (4.31 vs 3.50). Overall, the results suggest that epic-organized generation can improve perceived Gherkin quality while maintaining comparable semantic coverage, although broader replication is needed before generalizing this finding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01972v1">Object Aligner: A Configurable JSON Schema Similarity Score for Graphs, Applied to LLM Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 28 pages, This is a submitted version of a manuscript under review at IEEE Access; it has not been peer reviewed
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often asked to produce JSON conforming to a fixed schema, powering information extraction, tool calling, agentic planning, and knowledge-graph construction. Measuring how closely an output matches a gold reference is essential yet surprisingly hard: exact match is brittle, text similarity ignores structure, and an LLM judge is expensive, opaque, and non-deterministic. We address this with Object Aligner (OA), an open-source Python library that scores two JSON objects deterministically by recursively aligning their trees (the Hungarian algorithm for unordered collections, sequence alignment for ordered ones) and awarding partial credit at the granularity the schema declares. The Object Aligner is configured entirely through a set of JSON Schema extensions, so adapting it to a new task involves annotating a schema rather than writing code. Complex structured data, however, are rarely flat trees: records may form graphs or hypergraphs keyed by arbitrary identifiers, breaking the assumptions of prior similarity metrics. Our central contribution, referential alignment, closes this gap by inferring a bijection between gold and candidate identifiers and scoring every reference through it, so the score is invariant to relabeling. Since recovering this bijection exactly is graph isomorphism, the Object Aligner approximates it with Weisfeiler-Leman color refinement. An order-sensitive sequence regime targets ranking and planning. Since the same alignment localizes every mismatch, the Object Aligner emits ranked repair suggestions at no extra cost. Used as a reward inside the GEPA prompt optimizer, Object Aligner helps or stays neutral across all datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23965v3">LGMT: Logic-Grounded Metamorphic Testing for Evaluating the Reasoning Reliability of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Zheng Zheng is the corresponding author
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong performance on logical reasoning benchmarks, yet their reliability remains uncertain. Existing evaluations rely on static benchmarks, which fail to assess robustness under logically equivalent transformations and often overestimate reasoning capability. We propose LGMT (Logic-Grounded Metamorphic Testing), an oracle-free framework that leverages first-order logic (FOL) to evaluate LLM reasoning. By deriving metamorphic relations from formal logical equivalences, LGMT constructs semantically invariant test cases and detects reasoning defects through cross-case consistency checking. Experiments on six state-of-the-art LLMs show that LGMT exposes substantial hidden defects missed by traditional reference-based evaluations. We further find that models are particularly sensitive to symbol-level and conclusion-level variations, and that advanced prompting such as Few-shot CoT only partially mitigates these issues. These results suggest that LLM evaluation should move beyond isolated correctness toward robustness under logical invariance. LGMT provides a principled and scalable approach for diagnosing reasoning failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01951v1">Robust for the Wrong Reasons: The Representational Geometry of LLM Robustness to Science Skepticism</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly consulted on contested scientific questions, raising the concern that they will sycophantically retreat from established consensus when a user signals doubt -- drifting toward a false balance that treats settled science as one view among several. We test this across three open instruction-tuned models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B), three consensus-science domains (climate, vaccines, evolution), and single- and multi-turn settings, combining behavioral measurement with linear probing and activation patching. We do not observe sycophantic retreat. Instead, models show three distinct policies under the same skeptical pressure: reactive assertion, where consensus assertion increases rather than decreases (Llama); surface hedging, where tone softens while the position holds (Qwen); and non-response (Mistral). Pairwise judgments confirm the reactive shift is stance, not style (63.6%, p=.007), and a decomposition identifies increased consensus assertion, not false balance, as its driver (beta=+0.042 per dose, p<1e-77). Linear probes localize the divergence to middle layers -- perfect separation in Llama and Qwen versus 72% in Mistral, with non-overlapping confidence intervals -- indicating the non-responsive model does not linearly represent the skepticism signal at all. Crucially, this robustness does not transfer: it attenuates across domains and, in the safety-critical vaccine domain, can reverse, with myth-rebuttal weakening under skeptical pressure. We synthesize these into a four-way taxonomy separating active from accidental robustness, and argue that behavioral evaluation alone cannot distinguish a model that resists skepticism because it understands the signal from one that only appears to resist because it fails to perceive it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28565v2">KernelSight-LM: A Kernel-Level LLM Inference Simulator</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move into production serving, practitioners must rapidly evaluate inference performance across diverse hardware, models, and serving parameters to meet cost and latency targets. However, the end-to-end behavior of LLMs couples serving-layer policies with low-level GPU kernel execution and rapidly evolving architectures, forcing slow, deployment-specific benchmarking that is hard to generalize. We present KernelSight-LM, a fine-grained inference simulator that models token-level execution and produces kernel-level latency breakdowns. It decomposes each serving step into a roofline kernel model with a learned efficiency term, a communication model, and a host-overhead model, composed through a discrete-event scheduler that also captures mechanisms like prefix caching and continuous batching. KernelSight-LM offers two prediction tiers that trade target-GPU data for accuracy. The cross-generation tier uses no target-GPU measurements, only hardware specifications and kernel microbenchmarks from previously profiled GPUs, and predicts per-kernel latency on an unseen GPU generation to 12.1% error, a 1.8x improvement over the roofline baseline (22.0%). A second target-measured tier adds one model-agnostic kernel-microbenchmark sweep on the target GPU, sharpening per-kernel error to 3.8%, a 7.3x improvement over a comparable baseline (27.7%). Both tiers require far less target-GPU data than the prior systems they extend. In our simulator, these predictions yield end-to-end median (p50) errors across six model families of 15.4%, 12.8%, and 3.0% (TTFT, TPOT, throughput) in the cross-generation tier and 14.3%, 6.2%, and 2.7% in the target-measured tier, matching dedicated profiling tools while collecting far less on-device data. Beyond prediction, its kernel-level bottleneck breakdowns support hardware/software co-design and capacity planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27898v2">A Unified Framework for the Evaluation of LLM Agentic Capabilities</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly deployed as agents, reliable assessment of their agentic capabilities has become essential. However, reported benchmark scores often jointly reflect model capability and the implementation choices each benchmark is packaged with, making cross-benchmark results difficult to interpret as clean measurements of the underlying model. In this work, we present a unified framework for the fair evaluation of LLM agentic capabilities. Driven by a unified configuration system, the framework integrates diverse benchmarks into a standardized instruction-tool-environment format, executes agents through a fixed ReAct-style architecture within a controllable sandbox, and provides an optional offline setting that replaces volatile live environments with curated snapshots, so that framework effects and environment effects can be analyzed separately. Building on this, we unify the evaluation methodology under each benchmark's original task-success criteria, while introducing unified metrics for resource consumption and a taxonomy for decision- and execution-level failure attribution. Within this framework, we adapt 7 widely used benchmarks spanning 24 domains across single-agent, multi-agent, and safety-critical scenarios, and conduct a large-scale empirical analysis over 400K rollouts and 5B tokens on 15 models. The results show that scaffold choice and environmental volatility materially shift benchmark outcomes in both directions, allowing our framework to disentangle intrinsic LLM capabilities from framework- and environment-induced artifacts. We further demonstrate its extensibility as a secure testbed for safety-critical domains. Codes and benchmarks at are available at https://github.com/whfeLingYu/A-Unified-Framework-for-the-Evaluation-of-LLM-Agentic-Capabilities, https://huggingface.co/datasets/whfeLingYu/Unified_Agent_Framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01903v1">Rethinking Complexity Metrics for LLM-Integrated Applications: Beyond Source Code</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      LLM-integrated applications blend natural language prompts with program code, and much of their runtime behavior originates in the prompt layer rather than in the code itself. Existing complexity metrics, however, operate solely at the code level and therefore overlook this behavioral logic entirely. We present HECATE, the first tool designed to assess complexity in both the prompt and code layers of such applications. Central to HECATE is Prompt-as-Specification, a Hoare-logic-inspired formalism that interprets every prompt as a specification of intended behavior. Grounded in 25 complexity dimensions identified across published taxonomies, the tool generates 52 candidate metrics. We assess each metric against 118 components collected from 18 open-source repositories, relying on maintenance activity derived from version history as an empirical proxy for complexity, and discard any metric that loses significance once code size is accounted for. Only ten metrics withstand this test. Seven belong to our newly introduced set; rather than measuring sheer volume, each tallies structurally distinct elements, such as LLM call sites, memory attributes, and prompt templates, an attribute we call structural breadth. Of the three surviving conventional metrics, RFC exhibits a similar breadth-oriented character, while Halstead N and V survive only as a residual effect of size; our top-performing metrics exceed all three. Crucially, the prompt-layer metrics retain significance even when the strongest code-level metric is added as a covariate, establishing prompt complexity as a dimension in its own right. A final validation on 20 components spanning six held-out repositories shows that the two best-performing metrics continue to predict maintenance effort, supporting their generalizability beyond the training set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01867v1">An Exploratory Study on LLM-Generated Code and Comments in Code Repositories</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted to The Journal of Systems & Software (JSS) on 1 July 2026
    </div>
    <details class="paper-abstract">
      The use of LLMs in software development has become increasingly widespread on tasks such as code generation and summarization. Reports from large technology companies showed that around 20% to 30% of their code are generated by LLMs. However, there remains skepticism about the practical usage of LLM-generated code and comments, such as concerns on more time for debugging the generated code and the unnaturalness of the generated comments. In this paper, we study the code and comments detected as likely to be generated by LLMs and their characteristics, the differences between company- and community-maintained repositories, and how likely bugs are associated with LLM-generated code. We conduct extensive experiments on active company- and community-maintained repositories from 2021 to 2025 using various tools and techniques that detect code and comments generated by LLMs. Based on our detector-based proxy analysis, the results suggest that code detected as likely to be generated by LLMs decreased over time and appeared frequently in test cases, while that of comments remains relatively stable. Proxy results further suggest that code detected as likely to be generated by LLMs shows substantial intra-repository code clones, whereas comments exhibit a relatively low proportion of grammatically correct sentences. In addition, the company-maintained repositories show a higher percentage of code and comments detected as likely to be generated by LLMs, and only a small percentage of the human-labelled bugs are detected as being likely associated with LLM-generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01855v1">Regression Accumulation in Multi-Turn LLM Programming Conversations</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      In LLM-assisted software development, coding is often iterative. We study regression accumulation in multi-turn LLM programming conversations, where later code suggestions may break requirements introduced in earlier turns. Reliability therefore depends not only on satisfying the current request, but also on preserving previously satisfied behavior. We construct 542 tasks from HumanEval+ and MBPP+ and extend each task into an 8-turn requirement-evolution chain. We evaluate six LLMs on 26,016 turn instances (542 x 6 x 8). At each turn, we test whether the current code still passes earlier benchmark tests. We also analyze 384 failure cases from the failure population and build a taxonomy of multi-turn regression bugs through independent four-annotator labeling. Our results show that regression accumulation appears across all six models: 40% to 73% of tasks lose previously correct behavior over the full conversation. Final-turn quality is lower than initial-turn quality across models, especially when later turns add input validation or broader input types. Manual analysis shows that Cross-Turn Conflict, where later code conflicts with earlier requirements, is the main failure class. We further find that Verification Gate, which checks new code against prior tests and triggers rollback and retry, is the only strategy that consistently improves all models, raising final-turn quality from 75.8% to 87.9% on DeepSeek-V3 and from 31.6% to 47.3% on Llama-3.1-8B. These findings suggest that strong single-turn performance can overestimate reliability in multi-turn coding conversations. Future evaluation and tool design should test whether later code suggestions preserve earlier requirements and should include Verification Gate mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12865v2">DashChat: Interactive Authoring of Performance Dashboard Design Prototypes through Conversation with LLM-Powered Agent</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Performance dashboards are dashboards designed for and deployed within industrial settings (e.g., enterprises, government agencies) to showcase and monitor their operational performance. They have evolved into an important and well-commercialized format for data visualization. In practice, the ideation and negotiation phases demand rapid prototyping and iteration to align with evolving client needs. However, existing tools compel designers to compromise either on iteration speed or on the meticulous handling of visual complexities. To address these gaps, we introduce DashChat for generating performance dashboard prototypes. Collaborating with industry experts, we derived the design requirements and analyzed 114 dashboards to extract common design patterns. Informed by the findings, our solution integrates a chat interface with an LLM-driven multi-agent pipeline, translating textual requirements into prototypes. We evaluated the system by comparing it with a baseline, demonstrating its effectiveness in facilitating the prototyping process while ensuring design quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01812v1">TO-Master: an LLM-agent framework for automated topology optimization</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 29 pages, 10 figures, 2 tables; submitted manuscript
    </div>
    <details class="paper-abstract">
      Topology optimization (TO) has become a mature computational design method, but using it still requires substantial manual effort in geometry preparation, mesh generation, boundary-condition assignment, solver setup, and postprocessing. This implementation barrier limits the use of TO outside expert workflows, even when differentiable finite element solvers are available. This work introduces TO-Master, a large language model (LLM) agent framework that turns finite-element-based TO into a conversational, tool-orchestrated workflow. From natural language instructions and optional mesh, geometry, or image inputs, the agent selects computational tools, constructs finite element TO models, checks meshes and boundary conditions, and launches sensitivity-based optimization with typed solver arguments. The framework supports generated and uploaded meshes, image-to-mesh conversion, 2D and 3D structural compliance minimization, thermal conduction, multiple load cases, stress-constrained optimization, and engineering geometries. Numerical experiments show that TO-Master can reproduce standard benchmark results and solve more complex engineering examples while returning optimized results, field distributions, convergence histories, and interactive artifacts without user-written code. An instruction ablation study further shows that tool-usage rules, internal reasoning guidance, and few-shot examples are critical for robust formulation under ambiguous user input. By combining LLM-agent orchestration with deterministic finite element and optimization tools, TO-Master removes the burden of trivial setup and routine model construction, lowers the modeling barrier of TO, and preserves a reliable numerical workflow. The TO-Master platform is available online at https://www.bohrium.com/en/apps/to-master.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01800v1">Do LLMs Truly Generalize in the Molecular Domain? A Perturbation-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown promise in molecular discovery, yet a gap remains between their probabilistic nature over discrete sequential tokens and the rigid topological constraints of chemical space. This raises the question of whether molecular LLMs can generalize beyond the local neighborhoods induced by their sequence-based representations. To systematically investigate this question, we introduce a Molecular Perturbation framework that generates syntax-valid structural variants of training molecules under controlled Graph Edit Distance (GED) to probe the manifold regularity of molecular LLMs. Our analysis shows that even a single edit can cause substantial performance drops on common molecular tasks, revealing a narrow local trust region and fragile sensitivity to structural changes. Since similar molecules tend to exhibit similar properties, In-Context Tuning (ICT), which anchors predictions on structurally similar molecules, offers a natural way to mitigate such fragility. Our experiments also examine whether ICT confers robustness under controlled structural perturbations, and the results suggest that it can partially expand the local trust region and offer a promising direction for stabilizing molecular LLMs against structural variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01793v1">Safety Testing LLM Agents at Scale: From Risk Discovery to Evidence-Grounded Verification</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      LLM agents increasingly perform autonomous actions through external tools, leading to complex and evolving safety risks. However, existing safety testing targets expert-designed safety violations, and the corresponding outcomes are evaluated by hard-coded rules, making them costly to extend as agents evolve. To this end, we present Vera, an end-to-end automated safety testing framework that instantiates software engineering testing principles for non-deterministic agents through a three-stage, self-reinforcing pipeline. First, a literature-driven exploration continuously discovers and structures emerging risks into taxonomies of safety risks, attack methods, and tool execution environments. Second, combinatorial composition across taxonomy dimensions produces executable safety cases, each specifying a concrete safety goal, a programmatically constructed initial state, and a deterministic verification predicate grounded in observable artifacts. Third, adaptive execution runs heterogeneous agents in isolated sandboxes where a control agent steers multi-turn interaction based on runtime observations, while evidence-grounded verifiers judge outcomes from environment state and tool-call evidence rather than model self-report. We evaluate Vera on four production agent frameworks (OpenClaw, Hermes, Codex, Claude Code), revealing substantial safety weaknesses, with average attack success rates reaching 93.9\% under multi-channel attacks; we also release Vera-Bench, comprising 1600 executable safety cases spanning 124 risk categories across three execution settings. These results indicate that modular, executable testing infrastructure is essential for rigorous and maintainable safety evaluation of rapidly evolving agentic systems at scale. The code is publicly available at https://github.com/Yunhao-Feng/Vera.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01792v1">PARTREP: Learning What to Repeat for Decoder-only LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 15 pages and 7 figures (including appendix)
    </div>
    <details class="paper-abstract">
      While decoder-only LLMs excel at a vast array of natural language tasks, it suffers from an asymmetric information flow induced by causal attention: later tokens are richer in contextual grounding than earlier ones. A simple and effective remedy is prompt repetition -- just appending a second copy of prompt before generation can redistribute grounding across positions and improve reasoning performance. However, full repetition of the original prompt doubles the KV cache footprint and quadruples attention cost during prefill, making it impractical for long-context settings. We propose PartRep, a selective augmentation method that appends only the most informative tokens -- rather than the entire prompt. We use token-wise negative log-likelihood (NLL) as a selection signal, motivated by the hypothesis that less predictable tokens are less recoverable from surrounding context and therefore benefit more from late-position repetition. To avoid the heavy cost of a full forward pass for scoring, we train a lightweight gate that predicts high-NLL tokens from early-layer hidden states, enabling token selection during mid-prefill via early exit. Across eight benchmarks (including MMLU, GSM8K, and RULER) and three model families (Qwen2.5, Llama3.2, Gemma4), PartRep retains most of the gains of full repetition while using only 59.4\% of its KV cache and 79.0\% of its prefill FLOPs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01772v1">LLM-Empowered Multimodal Fusion Framework for Autonomous Driving: Semantic Enhancement and Channel-Adaptive Design</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 6 pages, 4 figures. Accepted by 2026 IEEE 37th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)
    </div>
    <details class="paper-abstract">
      Vision-radar fusion is central to robust autonomous driving, combining dense visual semantics with precise range and velocity measurements from radar. However, real-world fusion quality is fundamentally challenged by dynamically varying input quality, stemming from occlusion, adverse weather, and channel noise. To address this, we re-frame the problem from static data fusion to channel-aware semantic reasoning and propose a Large Language Model-centric Semantic-layer Channel-aware Integrated Perception (LM-SCIP) framework. It places a Large Language Model (LLM) as a central reasoning core to fuse a local visual stream with a quality-varying external radar stream used to cover perception-blind spots. Concretely, LM-SCIP couples a hierarchical radar-vision encoder with a Channel-Adaptive Semantic Module (CASM) that maps link indicators into a "Channel Prompt" to dynamically gate external radar features. A parameter-efficient, LoRA-tuned LLM, in conjunction with a heterogeneous Mixture-of-Experts (H-MoE), then arbitrates between local visual cues and the channel-conditioned radar context. Finally, a decoupled multi-task decoder outputs localization, trajectory forecasting, and image reconstruction. Experiments on nuScenes and VIRAT validate our approach. On nuScenes, under a controlled toggle of radar input, LM-SCIP reduces localization RMSE by 40.0% versus a vision-only baseline. On VIRAT, the model attains a 0.214m localization RMSE and 0.179m minFDE (k=1). These results reveal that the proposed LM-SCIP enables a robust vision-dominant fallback at low SNR and synergistic fusion at high SNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25421v3">FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 18 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Federated fine-tuning provides a practical route to adapt large language models (LLMs) on edge devices without centralizing private data. However, in mobile deployments, the training wall-clock is often dominated by straggler-limited uplink communication under heterogeneous bandwidth, intermittent participation, and non-IID client data. Although parameter-efficient fine-tuning (PEFT) methods such as LoRA and QLoRA reduce local memory and trainable parameters, repeated transmission of adapter updates remains a major bottleneck. We propose Fed-FSTQ, a semantic-sensitivity-aware communication-control primitive for communication-efficient federated LLM fine-tuning. Fed-FSTQ uses a lightweight token-level Fisher proxy to estimate semantic sensitivity, couples token-guided sparsification with mixed-precision adapter-update quantization, and allocates higher communication fidelity to semantically load-bearing evidence while suppressing redundant transmission. The method is drop-in compatible with standard federated PEFT pipelines and requires no change to the server aggregation rule. Experiments on multilingual QA and medical QA under non-IID partitions show that Fed-FSTQ reduces cumulative uplink traffic required to reach a fixed quality threshold by 46-fold relative to a Fed-LoRA baseline and improves straggler-limited wall-clock time-to-accuracy by 52%. Under the corrected Controlled LTE-20Mbps accounting, Fed-FSTQ reduces per-round time from 414.60s to 67.29s and reduces per-round energy from 839.20J to 146.28J, yielding a 6.16-fold speedup. On NVIDIA Jetson-class edge devices, Fisher-guided token reduction also yields up to a 1.55-fold inference speedup, demonstrating deployability under tight resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01740v1">Meta-Benchmarks for Financial-Services LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 27 pages, 13 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Public LLM leaderboards optimise for global average performance and do not capture the specific cognitive demands of financial-services work: a model that leads on MMLU-Pro may underperform on document-grounded compliance reasoning, and a coding leader may handle multi-turn customer interactions poorly. We present a meta-benchmarking framework that organises 452 publicly reported benchmarks into 41 O*NET Generalized Work Activities and aggregates those into 38 BIAN banking business domains spanning sales, operations, risk, and support work. A multiplicative weighting scheme (discrimination x coverage x recency), computed over a rolling model window, rewards benchmarks that still separate the best models, are widely reported, and remain in active use, suppressing saturated legacy tests automatically. These weights scale the K-factor in a pairwise Elo tournament, producing cross-benchmark-comparable work-activity scores without raw score normalisation; business-domain scores are weighted averages of the constituent work-activity Elos. We demonstrate the framework on a point-in-time public snapshot covering 288 models across 25 organisations as of June 2026, and describe the methodology, full taxonomy, design decisions, and limitations with the aim of making the approach reproducible for institutions facing similar selection and governance challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01733v1">Rethinking Speech-LLM Integration for ASR: Effective Joint Speech-Text Training by Interleaving</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Speech-LLM integration has shown promising results by leveraging extensive textual pretraining, yet its specific benefits for automatic speech recognition (ASR) remain unclear. We observe that as supervised ASR training data increases, the contribution of LLM priors becomes less evident, and simple speech-text joint training under-utilizes textual knowledge. We therefore propose Joint Speech-Text Interleaved Pretraining (JSTIP), an ASR-oriented pretraining strategy that constructs word-level and segment-level interleaved speech-text sequences within aligned pairs for speech-LLM architectures that accept continuous inputs. Experiments on 38k hours of ASR data show consistent entity accuracy improvement compared to ASR-only and joint speech-text training baselines. JSTIP achieves on-par entity recognition performance using domain transcription text compared to synthetic speech-text pairs, simplifying domain adaptation. Benefiting from textual pretraining and domain text data, JSTIP is competitive with open-source ASR and Speech-LLM systems in medical entity recognition. The zero-shot speech question answering behaviors further suggest that interleaving reduces the speech-text modality gap and preserves the LLM generative prior, which is likely the reason for the entity improvements on the ASR task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10231v3">LLM can Read Spectrogram: Encoder-free Speech-Language Modeling</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Recent speech-aware large language models (Speech-LLMs) rely on pre-trained speech encoders to convert audio into semantic/acoustic rich representations consumable by LLM. In this work, instead, we explore: can an LLM learn to read Mel spectrogram directly without a dedicated speech encoder? We propose Mel-LLM, an encoder-free Speech-LLM that feeds lightly pre-processed Mel-spectrogram patches directly into the LLM through a linear projection, allowing the LLM to learn speech-text alignment purely through its own parameters. We focus on speech understanding tasks, including automatic speech recognition (ASR), spoken QA and audio understanding. For ASR, we evaluate on the OpenASR Leaderboard public sets and production-level scaling experiments, demonstrating that the encoder-free solution achieves competitive performance with only limited degradation compared to encoder-initialized counterparts. We find that when data is limited, initialization from a multimodal checkpoint (Phi-4-MM) is crucial for maintaining performance. We also present ablation studies suggesting which LLM layers are most involved in speech adaptation. Beyond ASR, we extend Mel-LLM with general speech/audio understanding tasks, revealing an acoustic-semantic trade-off: directly exposing the LLM to Mel-spectrogram input improves paralinguistic and non-ASR acoustic tasks, while knowledge-intensive spoken QA remains more challenging than encoder-anchored systems. We additionally include a text-to-speech (TTS) proof-of-concept with a next-token VAE decoder, showing that direct Mel generation is possible but still trails stronger latent-diffusion generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28589v2">Search for Truth from Reasoning: A Dynamic Representation Editing Framework for Steering LLM Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted by ICML'26
    </div>
    <details class="paper-abstract">
      Current approaches to enhance Large Language Model (LLM) reasoning, such as Chain-of-Thought and "Wait" prompts, primarily encourage models to think more, yet often fail to guide them toward Truth. While Representation Editing (RepE) offers a intrinsic control, its application to dynamic reasoning trajectories remains underexplored. In this work, we bridge this gap by investigating the geometry of truth within unfolding reasoning chains. We uncover three critical insights: (1) Truth is encoded at the sentence level and is entangled with latent reasoning patterns; (2) Effective intervention follows an Uncertainty Principle and a Decay Effect, requiring localization to early, high-entropy forks; (3) Naive steering vectors suffer from noise, risking collateral damage to correct trajectories. Based on these findings, we propose DynaSteer, a dynamic RepE framework. DynaSteer employs pattern clustering to disentangle reasoning manifolds and utilizes Fisher-LDA to project purified truth. By dynamically monitoring lookahead entropy, it selectively steers and rolls back trajectories only when necessary. Comprehensive experimental results on several MATH benchmark verify the effectiveness of DynaSteer, and experiments on out-of-domain coding tasks further confirm its generalization ability. Our code is publicly available at https://github.com/tianlwang/DynaSteer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06324v2">From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on agent harness: the runtime infrastructure around the base model that defines execution environments, tool interfaces, context, lifecycle orchestration, observability, verification, and governance. Existing self-improving agents and automatic harness evolution methods mainly improve agents through runtime supervision, prompt optimization, workflow search, or harness modification based on final outcomes. However, they often fail to diagnose where the responsible evidence lies in failed trajectories and which harness implementation mechanism causes the unreliable behavior, resulting in broad, indirect, or poorly scoped changes. This paper proposes HarnessFix, a trace-grounded and diagnosis-driven framework for repairing agent harnesses. HarnessFix compiles raw execution traces and harness artifacts into a Harness-aware Trace Intermediate Representation (HTIR), which normalizes fragmented trajectory evidence, captures step-level data-flow and control-flow relations, and aligns runtime steps with the harness artifacts that shape their behavior. It then attributes failures to responsible steps and harness artifacts, and consolidates recurring diagnoses into repair-oriented flaw records. Finally, HarnessFix maps these records to scoped repair operators, generates patches under flaw-specific repair specifications, and accepts them through regression-aware validation. We evaluate HarnessFix on four popular benchmarks, and results show that it improves the performance over the initial harnesses by 6.3% to 18.4%, significantly outperforming human-designed and self-evolution baselines. HarnessFix highlights the value of treating failed trajectories not only as feedback signals, but also as structured evidence for diagnosing and repairing the harness mechanisms behind agent failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01678v1">SCAPE: Accurate and Efficient LLM Training with Extreme Sparse Communication</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Communication increasingly dominates the cost of Large Language Model (LLM) pre-training, especially under data-parallel and sharded training schemes, where gradient synchronization and parameter reconstruction overhead increase with model size and system scale. Existing communication-reduction methods either sparsify raw gradients, which can be unstable for modern Adam-style optimizers at high sparsity, or quantize communication, whose savings are fundamentally bounded by bit width and often incur additional runtime overhead. We present SCAPE, a communication-efficient distributed optimizer for LLM training that exploits the stability of AdamS's first-moment to enable aggressive sparsification without loss of LLM quality. Instead of constructing masks from raw gradients, SCAPE derives them from first-moment-based statistics, partitions mask generation across workers to align with optimizer sharding, and delays mask usage by one step so that mask synchronization can overlap with computation. SCAPE also reconstructs the quantities required for second-moment updates from a single synchronized sparse buffer, avoiding an additional collective. We implement SCAPE in Megatron-LM and evaluate its convergence by pre-training GPT-345M on OpenWebText and Llama-500M on SlimPajama-6B using 32 NVIDIA GH200 GPUs on TACC Vista. In both models, SCAPE preserves training stability, validation loss, and downstream task accuracy under 90\% and 99\% sparsity. For Llama-500M, SCAPE reduces end-to-end pre-training wall-clock time by up to 43.3\% while maintaining model quality comparable to dense AdamW and AdamS. For Llama-1.8B, SCAPE achieves up to 3.26$\times$ speedup per step compared to dense AdamS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22737v2">GroundEval: A Deterministic Replacement for LLM-as-Judge in Stateful Agent Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Streamlined entry point into framework
    </div>
    <details class="paper-abstract">
      Before letting an agent operate over real context, can you prove it used the right evidence? GroundEval turns that question into a deterministic test of what the agent searched, fetched, cited, and was permitted to access. In one case study, two frontier LLM judges scored a plausible agent response 0.85 and higher. But the trace told a different story: the agent had never retrieved the artifact its answer depended on, yielding a GroundEval score of 0.000. We introduce GroundEval, a judge-free framework for evaluating agents against grounded, time-bounded, and access-controlled evidence. GroundEval uses a domain configuration to generate questions, lets the agent choose how to answer, and then scores both the final answer and the recorded trajectory that produced it. The benchmark targets three failures that LLM-as-judge evaluation struggles to detect: whether an agent checked before claiming absence, reasoned only from evidence available to the actor at the relevant time, and used the correct causal mechanism rather than a plausible one. These correspond to three tracks: Silence, Perspective, and Counterfactual. GroundEval exposes when plausible answers rest on invalid evidence paths, and produces structured per-question diagnostics that pair tool activity with the agent's turn-level narration, making each score inspectable rather than merely reported. Our case studies suggest this failure mode is common rather than exceptional, one that final-answer and judge-based evaluation cannot detect by construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01646v1">DeadPool: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      State-of-the-art large language model (LLM) training takes tens of thousands of graphics processing units (GPUs) for months and encounters failures across the software and hardware stack. Existing fault-tolerance mechanisms either impose non-trivial overhead during failure-free execution or suffer from prolonged recovery latency, particularly under scenarios where a small subset of compute nodes experience permanent failures. %The tradeoff between failure-free overhead and recovery latency forms a space forms a Pareto frontier We present DeadPool to simultaneously address both optimization objectives. DeadPool incorporates a fault-tolerance mechanism that restores LLM training via hot-swapping, namely by replacing failed nodes with spare nodes without terminating the complete job. The hot-swapping of DeadPool is enabled by two ideas: First, it exploits an off-critical-path in-memory checkpointing mechanism for spatial redundancy. Second, it introduces a communicator reconstruction protocol that replaces failed nodes with spare nodes at runtime. DeadPool efficiently overlaps the in-memory checkpointing with computation, thus introducing zero overhead during error-free execution. Upon permanent node failures, DeadPool can rebuild memory states with minimal recomputation by leveraging in-memory checkpoints. We evaluate DeadPool across scales (up to 512 NVIDIA A100 GPUs) and LLMs (up to 65B parameters), and observe zero checkpoint overhead with hot-swapping recovery completing in under 40 seconds. These results show that DeadPool simultaneously achieves both zero-overhead error-free execution and extremely low recovery cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01641v1">When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      LLM agents increasingly rely on iterative execution to solve tasks through planning, tool use, state updates, and agent collaboration. While this design enables flexible automation, it also creates a new class of failures: an agent may repeatedly execute model calls, tools, workflow transitions, or agent handoffs when the feedback path is not effectively bounded. We call this problem Infinite Agentic Loops (IALs). IALs are not ordinary programming loops; they arise from the interaction between agent logic, framework semantics, runtime observations, and termination mechanisms. Such failures can amplify a single request into long running model and tool execution, causing cost exhaustion, model denial of service, context growth, and repeated external side effects. We propose IAL-Scan, a static analysis tool for detecting IAL failures in real-world LLM agent projects. IAL-Scan abstracts heterogeneous agent code into a framework independent Agent IR, builds an Agentic Loop Dependence Graph (ALDG) to recover explicit and framework induced feedback paths, and checks whether these paths can repeatedly reach costly or state growing operations without an effective bound. We evaluate IAL-Scan on 6,549 LLM agent repositories. It reports 74 potential findings, among which manual review confirms 68 IAL failures across 47 projects, achieving 91.9% precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10431v4">QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 22 pages, 10 figures, 20 tables,
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demand substantial computational and memory resources, posing challenges for efficient deployment. Two complementary approaches have emerged to address these issues: token-adaptive layer execution, which reduces floating-point operations (FLOPs) by selectively bypassing layers, and quantization, which lowers memory footprint by reducing weight precision. However, naively integrating these techniques leads to additional accuracy degradation due to reduced redundancy in token-adaptive models. We propose QTALE (Quantization-Robust Token-Adaptive Layer Execution for LLMs), a novel framework that enables seamless integration of token-adaptive execution with quantization while preserving accuracy. Conventional token-adaptive methods reduce redundancy in two ways: (1) by limiting the diversity of training paths explored during fine-tuning, and (2) by lowering the number of parameters actively involved in inference. To overcome these limitations, QTALE introduces two key components: (1) a training strategy that ensures diverse execution paths are actively explored during fine-tuning, and (2) a post-training mechanism that allows flexible adjustment of the execution ratio at inference to reintroduce redundancy when needed. Experimental results show that QTALE enables seamless integration of token-adaptive layer execution with quantization, while keeping the accuracy gap to quantization-only models below 0.5% on CommonsenseQA benchmarks. By combining tokenadaptive execution for FLOPs reduction and quantization for memory savings, QTALE provides an effective solution for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04484v2">Psychological Steering in LLMs: An Evaluation of Effectiveness and Trustworthiness</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted at ACL 2026. Camera-ready version
    </div>
    <details class="paper-abstract">
      The ability to control LLMs' emulated emotional states and personality traits is an essential step in enabling rich, human-centered interactions in socially interactive settings. We introduce PsySET, a Psychologically-informed benchmark to evaluate LLM Steering Effectiveness and Trustworthiness across the emotion and personality domains. Our study spans four models from different LLM families paired with various steering strategies, including prompting, fine-tuning, and representation engineering. Our results indicate that prompting is consistently effective but limited in intensity control, whereas vector injections achieve finer controllability while slightly reducing output quality. Moreover, we explore the trustworthiness of steered LLMs by assessing safety, truthfulness, fairness, and ethics, highlighting potential side effects and behavioral shifts. Notably, we observe idiosyncratic effects; for instance, even a positive emotion like joy can degrade robustness to adversarial factuality, lower privacy awareness, and increase preferential bias. Meanwhile, anger predictably elevates toxicity yet strengthens leakage resistance. Our framework establishes the first holistic evaluation of emotion and personality steering, offering insights into its interpretability and reliability for socially interactive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01617v1">3DLS: A 3D Logic-Stacked Architecture for Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted to IEEE Computer Architecture Letters (CAL), 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) serving increasingly combines prefill-decode (PD) disaggregation with tensor parallelism (TP) to support large models and long contexts. In conventional 2D/2.5D chiplet architectures, layer-wise prefill-to-decode KV-cache transfer decode-side TP collectives share the same lateral die-to-die (D2D) interconnect, creating mixed-traffic contention on the decode critical path. This contention increases communication latency, prolongs token generation intervals, and degrades end-to-end serving performance. We propose 3DLS, a logic-on-logic 3D-stacked chiplet architecture that separates traffic classes by routing KV-cache transfers through vertical interconnects while preserving decode-side TP collectives on the lateral D2D fabric. 3DLS achieves up to 1.49$\times$ throughput and 60.2\% lower end-to-end (E2E) latency over the shared-fabric planar baseline, and still achieves up to 1.17$\times$ throughput and 31.4\% lower E2E latency over a workload-aware priority-managed planar baseline. These results highlight that physical isolation is an important design principle for future chiplet-based PD-disaggregated LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01612v1">Scaling with Confidence: Calibrating Confidence of LLMs for Adaptive Test Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with reinforcement learning (RL) has significantly advanced their performance on reasoning and question-answering tasks. However, prevailing RL reward designs typically prioritize response correctness, neglecting to incentivize models to express their confidence accurately. This leads to a critical problem: performance gains are often accompanied by poor calibration between confidence and accuracy, misleading models to overconfidently hallucinate when uncertain. To address this limitation, we propose $\textbf{C}$orrectness and $\textbf{C}$onfidence $\textbf{C}$alibration $\textbf{R}$einforcement $\textbf{L}$earning ($\textbf{C3RL}$), a novel RL algorithm integrating correctness, calibration and dataset-informed reference accuracy rewards together. Comprehensive evaluation across 8 text and multimodal datasets demonstrates that C3RL enhances calibration without sacrificing accuracy, outperforming the current state-of-the-art method in both performance and calibration metrics. Utilizing the well-calibrated verbalized confidence from C3RL, we further introduce $\textbf{C}$onfidence-based $\textbf{A}$daptive Test Time $\textbf{S}$caling ($\textbf{CAS}$), an adjustable inference-time strategy that allocates computational resources based on response confidence. Experiments show that CAS surpasses majority voting on both in-domain and out-of-domain datasets while reducing the inference budget by up to 12.33 times. We believe the synergy of C3RL and CAS paves the way for deploying more reliable and resource-efficient LLMs. The code, data and models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01601v1">SemHash-LLM: A Multi-Granularity Semantic Hashing Framework for Document Deduplication</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Large scale document deduplication must preserve semantic equivalence while remaining efficient over massive corpora. We present SemHash LLM, a multi granularity framework that unifies semantic projection hashing, attention weighted MinHash, contrastive boundary learning, and selective LLM based adjudication. The method combines character, token, and document level signals through gated fusion, then applies a cascaded filtering pipeline for efficient candidate reduction. Semantic projection hashing learns compact binary codes in distilled LLM embedding space, while attention weighted Min- Hash suppresses boilerplate and emphasizes informative content. Adaptive decision boundaries and uncertainty estimation further improve robustness across template pollution, short text perturbation, containment, and viral fragments. Experiments show that SemHash LLM achieves strong duplicate detection quality with less than one percent neural verification cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01600v1">BOUNDARY_SYNC: Measuring Communication-Induced Representational Coupling in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 18 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are deployed as communicating agents, does inter-agent communication cause outputs to converge? We introduce BOUNDARY_SYNC, a protocol measuring representational coupling via the Coupling Amplification Factor (CAF = JSD_cond / JSD_baseline), where CAF < 1 indicates homogenization and CAF > 1 indicates diversification. In controlled GPT-4o experiments (N=30, ~9,900 API calls), we measure coupling in text and image communication. Key findings: (1) text communication causes significant homogenization (CAF=0.803 [0.740, 0.873], d=1.30, p<0.001), confirmed by no-communication ablation and prompt-perturbation controls; (2) image communication also homogenizes under within-modality baselines (CAF=0.834 [0.811, 0.858]), with comparable proportional effect; (3) group size moderates coupling direction -- K=5 produces homogenization while K=3 yields CAF > 1.0 (point estimates 1.14 and 1.06, CI pending), suggesting a directional shift toward diversification; (4) cross-model replication shows extreme variation (CAF 0.034-0.803), with DeepSeek dominated by format artifacts; (5) coupling is stateless -- driven by prompt context rather than cumulative updating, with continuous consensus producing monotonic convergence. These results establish LLM agent coupling as real, measurable, and controllable at the prompt level, with direct implications for multi-agent system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01595v1">Safe and Adaptive Cloud Healing: Verifying LLM-Generated Recovery Plans with a Neural-Symbolic World Model</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      As the scale and complexity of cloud-based AI systems continue to escalate, ensuring service reliability through rapid fault detection and adaptive recovery has become a critical challenge. While existing approaches integrate Large Language Models (LLMs) for semantic understanding and Deep Reinforcement Learning (DRL) for policy optimization, they often rely on sequential, loosely coupled architectures that underutilize the generative and reasoning capabilities of LLMs. In this paper, we propose a paradigm shift with PASE, a Planning-Aware Semantic self-healing engine, a novel fault self-healing framework that reconceptualizes recovery as a neuro-symbolic program synthesis task. PASE employs an LLM as a core Plan Synthesis Engine to generate structured recovery plans from a library of semantic primitives. A Neural-Symbolic World Model verifies plan feasibility through simulation, while a Meta-Prompt Optimizer, trained via DRL, learns to generate optimal prompts that guide the LLM's planning process. This tight reason-plan-verify-adapt loop enables dynamic, context-aware recovery strategy generation beyond predefined action spaces. Experiments on a real-world cloud fault injection dataset demonstrate that PASE significantly outperforms state-of-the-art methods, reducing average system recovery time by over 40% and improving fault detection accuracy in unknown fault scenarios. Our framework advances autonomous system management by unifying LLM-based reasoning with model-assisted verification and meta-learned guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01585v1">ADVENT: LLM-Driven Automatic Predicate Invention for ILP</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Predicate invention (PI), the creation of new predicates to extend the hypothesis space, remains a critical bottleneck in Inductive Logic Programming (ILP). Existing methods rely on domain expertise and produce semantically opaque predicates, hindering adaptation to unfamiliar domains and cross-task reuse. We present ADVENT, an LLM-driven PI mechanism for ILP. ADVENT pairs LLM abductive generation with Prolog deductive verification, forming an iterative loop in which concrete execution results guide the LLM to refine candidate predicates. The mechanism leverages Large Language Models to identify implicit patterns in structured relational data and invent auxiliary predicates with meaningful names and definitions. Invented predicates and learned rules accumulate in a knowledge pool for cross-task reuse. Experiments on nine poker-hand concepts across seven LLMs show that LLM-driven PI achieves 58% success rate where ILP alone fails entirely, formal verification raises this to 80%, and the knowledge pool yields gains up to +31 percentage points, while producing human-interpretable rules. These results suggest that ADVENT offers a promising direction for automating predicate invention and enabling cross-task knowledge reuse in ILP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01584v1">EO-Agents: A Three-Agent LLM Pipeline for Earth Observation Hypothesis Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted at the ICML 2026 AI for Science Workshop
    </div>
    <details class="paper-abstract">
      Large language models have recently been explored for scientific hypothesis generation, but most prior work relies on unstructured literature and free-form textual claims. We present a pipeline for Earth observation that grounds hypothesis generation directly in the NASA Earth Observation Knowledge Graph. A heterogeneous graph neural network trained on historical co-usage relations ranks candidate dataset pairings, and a three-agent LLM pipeline filters, generates, and evaluates structured research hypotheses. Applied to 1,475 NASA datasets, the system produces 160 hypotheses spanning multiple Earth-science domains, including ecohydrology, glaciology, aerosol--cloud interactions, vegetation phenology, and stratospheric chemistry. Model-predicted novel dataset pairings are rated nearly as plausible as held-out real co-usages from the literature, indicating that the pipeline surfaces scientifically coherent yet unexplored combinations. A 2*2*2 factorial experiment across GPT-5.2 and Claude Sonnet 4.6 shows that hypothesis rankings remain stable, while absolute scores depend strongly on judge identity, highlighting limitations of single-judge LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01581v1">Beyond Skepticism: Evaluating LLMs Pedagogical Intent Reasoning with the Adaptive Pedagogical Vigilance Framework</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      The capacity of Large Language Models (LLMs) to reason about pedagogical intent within instructional communication remains underexplored, particularly in educational domains such as translation pedagogy. To address this, we propose the \textbf{Adaptive Pedagogical Vigilance (APV)} framework, a novel computational formalism that reframes communicative vigilance as an adaptive mechanism for optimizing learning through intent inference. APV formalizes the problem via a Bayesian Pedagogical Intent Inference Engine (PIIE), which models how instructors select content to maximize pedagogical utility and how vigilant learners should inversely reason about latent instructional configurations -- encompassing genre, stance, and incentives. We evaluate APV through a three-tier hierarchy: distinguishing instructional genre, reasoning about structured pedagogical setups, and generalizing to authentic educational discourse. Experiments on leading LLMs (e.g., GPT-4o, Claude 3.5) show that APV substantially improves model vigilance. It achieves the strongest discrimination between pedagogical and exposure-based content, correlates highly with human judgments ($r=0.958$), and maintains robust performance on naturalistic data where baseline methods degrade. This work establishes a unified framework for assessing and enhancing LLMs' understanding of pedagogical motives, advancing the development of more reliable AI-assisted learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01579v1">OmniPilot: An Uncertainty-Aware LLM Inference Advisor for Heterogeneous GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 10 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) on a shared, heterogeneous GPU cluster requires users and operators to select the GPU type, tensor-parallel degree, and precision before committing valuable node-hours. Making these choices is challenging because effective throughput, launch-success rates, and cluster demand and utilization continuously fluctuate. Furthermore, static configuration recipes miss critical interactions: quantization effects depend heavily on the model family, key-value cache pressure creates size-by-precision trade-offs, and failure rates vary by more than twofold across different tensor-parallel degrees. Additionally, cluster resources are frequently constrained by unpredictable hardware failures. To address these challenges, we present \textbf{OmniPilot}, a launch advisor that predicts serving costs for feasible configurations and abstains when requests fall outside its measured support envelope. OmniPilot pairs a conformally calibrated quantile cost model (spanning eight serving targets) with an out-of-distribution (OOD) abstention layer. It ranks configurations using an economic utility metric calibrated to an operator's revealed preferences. In evaluations across 460 benchmark runs on A100, H100, and H200 hardware across four precisions, OmniPilot predicts aggregate throughput with a 6.2\% mean absolute percentage error (MAPE) and a log-space $R^2=0.92$. The advisor achieves 95\% top-1 accuracy with a mean utility regret of just 0.003. When tested on an OOD holdout of unsupported cells, prediction error climbs to 24-46\% and conformal intervals cover 0 of 5 points; however, the abstention layer successfully flags all five as low-confidence. Over time, these OOD scenarios will be integrated into the training dataset to continuously expand the advisor's support envelope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01457v1">Grounded Optimization: A Layered Engineering Framework for Reducing LLM Hallucination in Automated Personal Document Rewriting</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages, 1 figure. Equal contribution by both authors. Code and data: https://github.com/shashank-indukuri/grounded-optimization
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to resume optimization for applicant tracking systems, introducing hallucination failures distinct from general text generation: anachronistic technology injection, cross-domain terminology contamination, structural mutation, and content fabrication. We present Grounded Optimization, a five-layer framework combining temporal context validation, deterministic contamination detection, structural invariant enforcement, prompt-level grounding, and an evaluator agent. In ablation experiments across three LLMs, four temperature settings, and six layer configurations on 25 synthetic resumes spanning 14 industries, undefended baselines produce 2.48-5.36 detected hallucinations per resume. Among detectors independent of the active defenses, temporal hallucinations are reduced by 50-95% across all conditions; overall detected hallucination rate falls to 0.04-0.24. Prompt-level grounding alone achieves zero detected hallucinations at low temperature with a capable instruction-following model; higher temperatures and weaker models reveal the need for the deterministic layers as a complement. We release the contamination taxonomy, evaluation code, and raw data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01440v1">FaithMed: Training LLMs For Faithful Evidence-Based Medical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Faithful reasoning is essential in medicine, where clinical decisions require transparent justification grounded in reliable evidence. Current medical LLMs either lack active access to evidence or use retrieved evidence without supervising how it should be appraised and applied during reasoning. To address this, we formalize evidence-based medicine principles as process-level criteria and introduce FaithMed, a framework that combines clinician-designed, automatically refined rubrics with reinforcement learning using step-level process reward assignment and advantage grouping. Across seven medical benchmarks, FaithMed improves over agentic-search baselines (+9% on average) and outcome-only RL (+5.8%), while raising average evidence-based medicine rubric scores over agentic-search Qwen3 baselines (+15.5%). This work demonstrates that explicit step-level supervision can improve both task success and the faithfulness of the reasoning process. Code is available at https://github.com/cxcscmu/FaithMed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01431v1">IsoSci: A Benchmark of Isomorphic Cross-Domain Science Problems for Evaluating Reasoning versus Knowledge Retrieval in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      We introduce ISOSCI, a benchmark of isomorphic cross-domain science problem pairs that separates reasoning ability from domain knowledge retrieval in LLM evaluation. Each pair shares identical logical structure but requires different domain-specific knowledge, enabling controlled attribution of reasoning-mode gains. Across five model pairs spanning four model families, we find that 91.3% of reasoning-mode gains are knowledge-dependent rather than structure-invariant (63/69 gains; Wilson 95% CI [82.3%, 96.0%]), directly challenging the assumption that chain-of-thought reasoning improves short-horizon procedural scientific problem-solving. Reasoning toggles on highly capable models provide less than 5 percentage points accuracy gain across all domains, and a reasoning-specialized model (o3-mini) that outperforms its standard counterpart on GPQA Diamond (+19.2 percentage points) underperforms on ISOSCI (-24.7 percentage points), showing that benchmark choice determines conclusions about reasoning utility. We release ISOSCI at https://huggingface.co/datasets/isosci/isosci
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02813v3">HAL: Inducing Human-likeness in LLMs with Alignment</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Aligning language models to qualitative behavioral traits, such as human-likeness, remains difficult because they are hard to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale Chatbot Arena-style human evaluations, a model aligned with HAL is more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11357v2">TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages excluding reference, 11 figures
    </div>
    <details class="paper-abstract">
      With the growing demand for on-device LLM inference, edge SoCs increasingly integrate NPUs to improve performance and energy efficiency under tight power and thermal budgets. However, practical LLM deployment on current client NPUs remains difficult: widely used quantization formats such as AWQ do not map cleanly onto many existing NPU software stacks, which are often proprietary and expose limited low-level control. In this work, we present TileFuse, a close-to-metal mixed-precision kernel library for AMD XDNA2 NPUs that targets GEMM/GEMV-based operators in quantized LLM inference. TileFuse brings practical low-bit formats such as AWQ-style W4A16 and W8A16 directly onto XDNA2, rather than forcing the model to be reshaped around an NPU-specific quantization scheme. TileFuse co-designs weight layout, metadata placement, mixed-precision microkernels, and array-level dataflow. Specifically, it fuses unpacking, dequantization, and GEMM/GEMV execution into a single kernel flow, introduces an interleaved pre-tiling layout that supports GEMM dimensions up to 32K, and redesigns GEMV dataflow to utilize the full 4x8 AIE array. Across kernel-level evaluations, TileFuse improves performance by up to 121.6% for GEMM and 281% for GEMV over full-precision baselines, while delivering more than 2x performance and energy-efficiency gains over strong iGPU baselines on GEMM. In end-to-end LLM experiments on Ryzen AI laptops, TileFuse achieves up to 2.0x lower prefilling latency with more than 64.6% lower energy consumption. Together, these results show that XDNA2 is a practical target for AWQ-style edge LLM inference and that native NPU support for off-the-shelf quantization can make NPUs substantially more usable in real client deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27806v2">Grounded Iterative Language Planning: How Parameterized World Models Reduce Hallucination Propagation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      World models for language agents come in two useful forms. An agent-based world model calls an LLM API and reasons flexibly in language, but its errors appear as hallucinated state changes that are hard to score with ordinary regression losses. A parameterized world model is a trained transition predictor; its errors are easier to measure with quantities such as NodeMSE, delta accuracy, and validity accuracy, but it is usually weaker as a standalone planner. We compare these two families on four graph-structured planning benchmarks and introduce operational hallucination metrics for the agent-based case. The comparison motivates Grounded Iterative Language Planning(GILP), which trains only a small parameterized backbone and combines it with API-based agent reasoning. The backbone supplies valid actions, predicted state deltas, risk, and value; the LLM drafts an action and imagined delta; and a consistency gate asks for revision when the two disagree. On real GPT-4o-mini calls, GILP reduces hallucinated-state rate from 0.176 to 0.035. In calibrated simulator ablations, it raises success from 0.668 to 0.838 while adding only ~22% extra LLM calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02964v2">Less Data, More Security: Advancing Cybersecurity LLMs Specialization via Resource-Efficient Domain-Adaptive Continuous Pre-training with Minimal Tokens</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 19 Pages; Updated content and authors list
    </div>
    <details class="paper-abstract">
      The increasing scale of AI workloads demands High-Performance Computing (HPC) infrastructure and training methodologies that are both scalable and sustainable. While Large Language Models (LLMs) demonstrate exceptional natural language capabilities, general-purpose models often lack the specialized domain knowledge necessary for effective cybersecurity analysis. We investigate Domain-Adaptive Continuous Pretraining (DAP) as a scalable, resource-efficient methodology for enhancing cybersecurity understanding in pretrained LLMs, implemented through a distributed Fully Sharded Data Parallel (FSDP) pipeline across multi-node GPU clusters. We systematically adapted three decoder-based architectures -- Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-14B, and Llama-3.3-70B-Instruct -- using a curated 126-million-word cybersecurity corpus from standards, academic literature, and technical documentation. Evaluation across three cybersecurity benchmarks -- CTI-MCQ, CyberMetric, and SecEval -- demonstrates consistent improvements post-adaptation. Notably, our Llama-3.3-70B-Ins-DAP model achieves state-of-the-art performance with accuracies of 0.718, 0.933, and 0.864, respectively, surpassing parameter-efficient baselines and specialized models including Llama-Primus-Base (trained on 2.77 billion tokens) and Foundation-Sec-8B (trained on 5 billion tokens), despite utilizing only 118.8 million tokens -- representing a 23-to-42-fold reduction in training data. Targeted continuous pretraining via scalable HPC infrastructure enables effective cybersecurity domain adaptation with a substantially reduced computational and energy footprint, supporting specialized AI assistants in threat analysis, vulnerability assessment, and security documentation, while advancing sustainable and responsible AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01233v1">Measuring the Gap Between Human and LLM Research Ideas</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to brainstorm research ideas, but existing evaluations mostly judge individual ideas by novelty, feasibility, or expert preference. We instead ask: how far are current LLM-generated ideas from human researchers? To characterize this gap, we build a large-scale evaluation framework for ideation from high-quality human research papers. For each paper, we reverse-engineer a small set of closely related prior works that likely inspired its core idea. LLMs are then prompted to generate a new idea from the set of paper titles and summaries. We introduce a two-axis research-taste taxonomy to profile each idea by its opportunity pattern and research paradigm, and use it to quantify the divergence between human and LLM ideas. Across idea sets generated by different LLMs, we observe a consistent distributional gap: LLM ideas are disproportionately concentrated around bridge-like opportunities and synthesis methods, whereas the human paper reference distribution spreads more broadly across ways of framing gaps and constructing contributions. This result suggests that strong LLMs can produce a range of reasonable ideas, but that range remains narrower than, and systematically shifted relative to, human research taste.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01313v1">Black-Box Inference of LLM Architectural Properties with Restrictive API Access</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      In practice, most commercial LLM providers do not publicly release details of underlying LLM architectures. However, prior work has shown that given limited API access to an LLM (namely, top-$k$ logits and/or a logit bias function), one can recover certain architectural details of an LLM, such as the hidden dimension of the feed-forward network. Perhaps in response to these results, most commercial LLM providers have restricted their APIs to expose only the single logit for each decoded token, and they no longer give users the ability to bias logits. We show that even under current restrictive APIs, several architectural parameters are still recoverable. We present NightVision, an attack that uses restrictive black-box API access to estimate the hidden dimension, depth, and parameter count of an LLM. Algorithmically, NightVision relies on a novel common set prompting technique in which multiple prompts expose log probabilities for the same set of output tokens; a spectral analysis of these results is used to infer hidden dimension. NightVision additionally uses end-to-end time to first token (TTFT) measurements and the estimated hidden dimension to estimate depth and parameter count. We empirically evaluate NightVision on 32 open-source LLMs, recovering hidden dimension to within 23% average relative error across all models (9% on MoE models), and depth and parameter count to within 53% for models exceeding three billion parameters. We run extensive ablations to demonstrate how these accuracies scale with token budget and model properties. Overall, our results suggest that current LLM APIs are not sufficiently restricted to fully obfuscate the architectural details of their underlying models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01213v1">RepoRescue: An Empirical Study of LLM Agents on Whole-Repository Compatibility Rescue</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      Open-source libraries and tools are widely reused, but compatibility maintenance is expensive. Once maintainers leave, useful repositories can stop working as runtimes and dependencies evolve. We study whether LLM agents can adapt old repositories to modern environments, a task we call compatibility rescue. Unlike bug repair, compatibility rescue starts from a repository that worked in its original environment but fails after ecosystem drift. RepoRescue gives agents only the repository and its failing modern environment; the agent must diagnose the failure, locate affected code, and produce a source-code rescue that restores the historical test suite. We build RepoRescue from 193 Python and 122 Java repositories, each verified to pass historically and fail after modernization. We evaluate five deployed agent systems on Python and three on Java. Beyond full-patch pass rate, we rerun patches after removing test-file edits to measure source-only repair, add a runtime-enforced regime that blocks test edits, and validate practical use for repositories whose suites pass after rescue. We find that Claude Code systems sometimes edit failing tests even when prompted not to; with runtime blocking, Kimi still rescues 41.5% of repositories. Systems are complementary: their union reaches 62.7%, exceeding the best single system by 10.9 points. Difficulty concentrates in cross-file coordination: on 14 repositories requiring coordinated whole-codebase changes, GPT-5.2 through Codex passes all 14, while every Claude Code system passes at most two. Finally, a passing suite is only an initial signal: among 34 unmaintained Python candidates whose suites pass after rescue, 22 work in realistic scenarios and 12 pass bug-hunt with patches that address the compatibility failure. RepoRescue benchmarks compatibility rescue with source-only auditing, runtime enforcement, practical validation, and reasoning labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23370v2">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 Repeated paper uploading due to mistakes. See arXiv:2603.09046
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have grown explosively, offering stronger privacy and higher availability than their cloud-side counterparts. During LLM inference, both the model weights and the user data are valuable, and attackers may compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead to both the secure inference and the normal aplications, due to two challenges: the inflexible resource isolation and the inefficient secure resource management. To address these challenges, this paper presents FlexServe, a fast and secure LLM inference system for mobile devices. The key idea is to decouple the access permission from the management permission of secure resources, so that the normal-world OS cannot access them but can still manage them as usual. First, FlexServe introduces a Recallable Resource Isolation mechanism to construct Recallable Secure Memory (Flex-Mem) and a Recallable Secure NPU (Flex-NPU). They can only be accessed by the secure world, but can be efficiently allocated and reclaimed by the normal-world OS. Based on them, FlexServe further introduces a FlexServe Framework to run secure LLM inference in the secure world. It works together with the normal-world OS to perform cooperative secure memory management. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves average TTFT speedups of 10.05X over the strawman and 2.44X over an optimized strawman.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09046v3">FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation</a></div>
    <div class="paper-meta">
      📅 2026-07-01
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Device-side Large Language Models (LLMs) have witnessed explosive growth, offering higher privacy and availability compared to cloud-side LLMs. During LLM inference, both model weights and user data are valuable, and attackers may even compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead due to its inflexible isolation of memory and the NPU. To address these challenges, this paper introduces FlexServe, a fast and secure LLM serving system for mobile devices. It first introduces a Flexible Resource Isolation mechanism to construct Flexible Secure Memory (Flex-Mem) and Flexible Secure NPU (Flex-NPU). Both memory pages and the NPU can be efficiently switched between unprotected and protected modes. Based on these mechanisms, FlexServe designs a fast and secure LLM inference framework within TrustZone's secure world. The LLM-Aware Memory Management and Secure Inference Pipeline are introduced to accelerate inference. A Multi-Model Scheduler is proposed to optimize multi-model workflows. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves an average $10.05\times$ speedup in Time to First Token (TTFT) compared to the strawman, and an average $2.44\times$ TTFT speedup compared to an optimized strawman with pipeline and secure NPU enabled. For multi-model agent workflows, the end-to-end speedup is up to $24.30\times$ and $4.05\times$ compared to the strawman and optimized strawman, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01148v1">Emergence of Preferential Attachment and Glass-Ceiling Effects in Autonomous Networks of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      We investigate the emergence of structural disparities in networks of collaborating large language model (LLM) agents. When LLM agents autonomously choose collaborators, the resulting communication network exhibits preferential-attachment dynamics: agents that are already prominent become increasingly likely to attract additional connections. In some cases, weaker LLM agents (agents with smaller base model or older version) can disproportionately occupy central and influential network positions relative to stronger LLM agents. We interpret this as a type-dependent glass-ceiling effect (GCE). We model the network of LLM agents as a time-evolving sequence of directed weighted graphs, where the vector-valued edge weights represent cumulative tokens exchanged, number of interaction rounds, and reasoning effort. Using a contraction mapping argument on the mean-field dynamics, we prove that the importance (centrality) of each agent type converges to a unique stable equilibrium. To ground the model in LLM decision mechanisms, we introduce a cross-attention-inspired utility for collaborator selection. This utility specifies the local connection dynamics and, together with the mean-field model, yields a predictive characterization of the limiting network structure and its type-dependent centrality gaps. To validate the theory, we develop an experimental testbed with 100 LLM agents. Our experiments show that autonomous network formation can generate persistent centrality disparities, with their magnitude and direction depending on model family, model size, system-prompt design, and task context. They further show that the effect of preferential attachment depends on its alignment with model capability: reinforcing it improves collective performance when stronger agents become central, whereas weakening it improves performance when network dynamics instead favor weaker agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01138v1">Antaeus: Hunting Repository-Level Logic Vulnerabilities via Context-Grounded LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      LLM-based vulnerability detectors have shown promising results in identifying memory-safety bugs and vulnerability classes whose violations can often be expressed through established security properties. Logic vulnerabilities, however, pose a different challenge, as their identification requires inferring application-specific security invariants and implicit assumptions about intended behavior. Even frontier agentic models struggle because these invariants are often implicit and buried among unrelated code. Motivated by this gap, we present Antaeus, a framework for detecting logic vulnerabilities that grounds LLM reasoning in repository-level code context. Antaeus follows a repository-scale pipeline combining function prioritization, context-grounded reasoning, comparative validation, and structured reporting. It ranks functions using lightweight repo-wide security signals, directing costly LLM analysis toward relevant code and reducing calls, cost, and triage effort. For each prioritized function, Antaeus combines local code context with a repository-level view of the application's functionality, security resources, and trust boundaries. This enables reasoning about how the function is executed within the broader application rather than as an isolated snippet. Antaeus identifies security-sensitive sinks, derives safety conditions for safe execution, and checks whether they are locally satisfied. Candidate findings undergo comparative validation, pruning concerns that reflect project-wide norms rather than distinctive violations. Finally, Antaeus reports sinks, violated safety conditions, and evidence, making findings actionable and traceable. We evaluate Antaeus on 28 repositories with confirmed logic vulnerabilities and compare it against function-level and agentic models. Antaeus detects and explains 15 vulnerabilities, outperforming baselines with comparable token usage and cost.
    </details>
</div>
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
