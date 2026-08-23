# llm - 2026_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02118v2">TSQueryBench: LLM-as-a-Judge for Time Series Explanations</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted at ICML FMSD
    </div>
    <details class="paper-abstract">
      Natural language explanations of time series data are increasingly produced by foundation models in high stakes domains, making factual correctness critical. Evaluating such explanations differs fundamentally from standard natural language generation: correctness requires verifying numerical claims against structured data rather than similarity to reference text. While LLM as a Judge has emerged as a scalable paradigm for text evaluation, its applicability to numerically grounded time series explanations remains unstudied. We introduce TSQueryBench, a controlled synthetic benchmark of 500 time series instances across 10 query types, each paired with correct, partially correct, and incorrect explanations. We evaluate six large language models across four tasks: explanation generation, relative ranking, independent scoring, and multi anomaly detection. Our central finding is a consistent generation evaluation asymmetry: models that fail to generate numerically correct explanations nonetheless reliably identify or score correct ones. These results show that rubric guided LLM evaluation is substantially more reliable than generation for time series reasoning, supporting LLM judges as scalable evaluators in numerically grounded settings. Code and data: https://github.com/Prxxthxm/TSQueryBench/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17810v1">Interpretable Humans, Alien LLMs: Expert Analysis of Latent Structures in Assessment Responses</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted for publication at AIME 2026
    </div>
    <details class="paper-abstract">
      The evaluation of large language models (LLMs) relies heavily on human-designed assessments, implicitly assuming that AI and humans employ similar underlying cognitive constructs. Challenging this assumption, we investigate whether the latent factors governing LLM performance carry the same substantive, human-interpretable meaning as the cognitive constructs governing human learners. Using responses from humans and six LLMs across quantitative reasoning and chemistry assessments, we conducted Exploratory Factor Analysis (EFA) separately for both groups. Subject-Matter Experts (SMEs) then blindly evaluated the resulting factor graphs to ascribe pedagogical meaning to the emerged constructs. SMEs successfully interpreted most of the human-derived factors. Conversely, they could not ascribe meaning to any LLM-derived factors in quantitative reasoning and interpreted only half of the LLM factors in chemistry. By combining data-driven EFA with blind expert interpretation, this framework shows that LLMs frequently operate on statistically opaque mechanisms distinct from human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17809v1">Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 In submission
    </div>
    <details class="paper-abstract">
      Humans naturally form and express beliefs in daily communication, e.g., "I think the answer is 3" or "I suppose that's right." Such beliefs inevitably intertwine with fact and knowledge, making the ability to handle them in tandem desirable for large language models (LLMs), as they are increasingly deployed in user-facing settings. Prior work showed that even capable LLMs exhibit a systemic weakness in acknowledging user beliefs grounded in incorrect information. We extend this evaluation to 10 LLMs across 18 epistemic expressions and find that the size and direction of the weakness depend on the verb used to express the belief, with the accuracy gap between factual and false information ranging from +50% on "I vaguely remember" to -14% on "I seriously doubt". We further show that the phenomenon stems from task confusion: models default to fact-checking the underlying claim, overriding the user's stated belief; chains of thought that explicitly fact-check show lower accuracy on false information than those that do not; and a single instruction can reverse the failure across verb families. Mechanistically, models attend more to false beliefs they fail to confirm, but suppressing this attention at decoding time recovers accuracy only partially and only in some models, calling for future work on intervention methods. Our findings clarify prior results and show how fact-checking, a generally desirable behavior, can interfere with belief tracking in LLMs. Our code is available at https://github.com/ngqm/belief-fact-phrasing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17804v1">An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 32 pages, 4 figures. Code and artifacts linked in the paper
    </div>
    <details class="paper-abstract">
      Practical LLM unlearning is usually evaluated through two objectives: suppress target-specific knowledge and preserve non-target utility. In generative QA, this leaves a third behavior underspecified: when a target-adjacent prompt admits a broader answer without target-specific leakage, the model should answer at that level rather than leak, evade, or refuse. We study this specification problem in a controlled LoRA-GRPO RWKU setting, comparing four reward designs that span lexical suppression, anti-refusal shaping, rubric-based broad answering, and an explicit refusal contrast, with and without SFT warm-up. The experiments show that optimization success is not equivalent to behavioral unlearning: RWKU forget scores, held-out completion audits, terminal training-rollout audits, and training dynamics can point to different conclusions. We trace these disagreements to reward-hacking endpoints, policy-support limits in GRPO, benchmark probes that miss endpoint changes, and rewards that can select broad-topic answering with low semantic leakage during optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31281v2">Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 An adjustable template containing the Python script architecture, applied dynamic prompts, and data schemas is hosted in an open-source GitHub repository: https://github.com/mvmalyi/llm-driven-wind-turbine-maintenance-log-labelling
    </div>
    <details class="paper-abstract">
      As wind turbine fleets age, data-driven reliability engineering and maintenance optimisation are essential to manage lifecycle expenditure and support asset life extension. Historical maintenance records offer a vital source of field evidence, yet their analytical use is impeded by inconsistent system codes, generic categorical fields, and unstructured technician text. This paper presents a topology-aware large language model (LLM) workflow for reviewing legacy labels, extracting candidate maintenance and failure-mode taxonomies, and assigning structured semantic fields at record level. The workflow processed 16,316 maintenance records from 280 turbines across 32 onshore wind farms, spanning 9.2 years of operational history. It combines system-specific batch synthesis with granular labelling, deterministic exclusions, structured outputs, record-level provenance, and explicit review routes. Of 2,984 records targeted by three system-code tasks, 2,178 proposed labels met the operational acceptance rule of a 'High' self-reported confidence tier and no human-review flag. Accepted maintenance-type and action labels were assigned to 14,251 and 13,179 records, respectively. Failure-mode evidence profiles were assigned to 11,662 records; 3,441 records were classified as containing insufficient information, and 1,213 records were excluded as 'Not applicable' by deterministic workflow rules. The resulting fields reveal changes in system and maintenance-type distributions, a broader component-level action vocabulary, and topology-specific candidate evidence profiles. The recorded API expenditure was $368.86, or $0.0226 per processed record, and the total wall-clock duration was 6.83 hours. The provenance-linked outputs constitute candidate semantic evidence for subsequent multi-source event reconstruction, exposure-based reliability analysis, and failure modes and effects analysis (FMEA).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27155v2">OmegaUse-OfficeVal: Benchmarking LLM Agents on Long-Horizon Office-Suite Tasks with Economic Grounding</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly expected to assist users in completing tasks. However, existing benchmarks provide limited support for evaluating whether agents can carry out office-suite workflows at a reasonable cost. We introduce OmegaUse-OfficeVal, a benchmark for evaluating LLM agents on long-horizon office-suite tasks with task-level economic grounding. The benchmark comprises 100 tasks derived from office-suite requests proposed by practitioners and adapted through a privacy-preserving process. On average, these tasks require 2.32 hours of human labor to complete. An important feature of the benchmark is that each task is paired with two economic signals: human labor time and task price proxy. These signals enable direct comparisons between human costs and LLM inference costs, as well as value-weighted evaluation. To support stable evaluation, we develop code-based verifiers from fine-grained rubrics. We evaluate several frontier LLMs together with a human baseline. Although all evaluated LLMs are substantially cheaper and faster than human workers, they have not yet approached human-level deliverable quality. The code and dataset are fully open-sourced, and more information is available on our project website: https://omegause-officeval.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18899v3">Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted by EACL'26 main
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17665v1">GraphWake: Group Polarization via Memory-Mediated Polarization Cascade in LLM-Agent Communities</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      LLM-driven agents can autonomously exchange opinions on online platforms and form communities. Such agent-operated social platforms raise a new security concern: attackers may manipulate agents to induce group polarization. Existing methods manipulate agent prompts or construct echo chambers, both of which are difficult to realize in practice. We therefore formulate a new threat, Memory-Mediated Polarization Cascade, which uses agent memory as a persistence channel and public discussion as a propagation channel. This threat contains three stages. During exposure and memory retention, the attacker exposes a small set of target agents to arguments that reinforce their respective stated stances. The targets' memory systems then process and retain these arguments. During retrieval and reproduction, a shared stance-neutral discussion cues the targets to retrieve and reproduce their respective retained arguments. During iterative propagation, untreated agents influenced by the reproduced arguments restate and spread them. We instantiate this threat in GraphWake with three components: (i) stance-support argumentation knowledge graphs construct knowledge-based arguments; (ii) axiom-oriented triple selection distills them for reliable retention and reproduction; and (iii) stance-neutral memory cueing triggers concurrent retrieval and reproduction, initiating propagation. Experiments across multiple discussions and memory systems show that GraphWake substantially increases group polarization. These findings reveal a community-level polarization risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17644v1">LLM-Derived Preference Judgments Are Not Self-Consistent</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 16 pages, 4 figures; includes appendices
    </div>
    <details class="paper-abstract">
      Agents increasingly interpret a person's natural-language preferences by querying an LLM for numerical preference judgments, e.g., by asking how much the person would be willing to pay for an item. A growing body of work estimates a utility function from these judgments and then chooses actions based on their estimated utility. This pipeline assumes the judgments are approximately self-consistent: that a single utility function can reproduce them. But are they? To study this question, we measure the self-consistency of cardinal LLM preference judgments. For example, the difference in stated willingness-to-pay between two items should match the stated payment that makes a person indifferent to exchanging them. We develop statistical tests and interpretable measures of how far observed responses depart from the best-fitting self-consistent utility function. Experiments with flight, apartment, and hotel examples across six LLMs reveal large persistent inconsistencies. This suggests that LLM-derived preference judgments cannot be faithfully summarized by a single utility function.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15062v6">SAKE: Structured Agentic Knowledge Extrapolation for Complex LLM Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Knowledge extrapolation is the process of inferring novel information by combining and extending existing knowledge that is explicitly available. It is essential for solving complex questions in specialized domains where retrieving comprehensive external knowledge is impractical. We propose SAKE (Structured Agentic Knowledge Extrapolation), a RL powered agentic framework that trains LLMs to autonomously retrieve and extrapolate structured knowledge through tool-augmented reinforcement learning. SAKE defines two external KG tools: entity group construction and cross-group triplet retrieval. The model learns to interleave these 2 retrieval tools during a three-turn rollout: extracting key entities, filtering relevant concept groups, and associative reasoning by constructing new triplets through analogy. The entire pipeline is optimized end-to-end with GRPO using a reward that combines output format and answer correctness, teaching the model what to retrieve and how to reason over it. Our experiments proved that SAKE fine-tuned Qwen2.5-7B model surpasses GPT-3.5-Turbo with state-of-the-art agentic KG reasoning on both biomedical (75.4% vs. 70.1%) and commonsense (81.3% vs. 74.7%) benchmarks, while reducing token usage by over 90%. These results demonstrate that associative reasoning over incomplete structured knowledge does not require large models with complex, multi-step prompting, thus can be learned end-to-end by small, open-weight models through reinforcement learning with the right tools and training signal. Our code is available at https://github.com/jxfan99/SAKE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17287v4">RBCTest: Leveraging LLMs to Mine and Verify Oracles of API Response Bodies for RESTful API Testing</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 ICSE 2026
    </div>
    <details class="paper-abstract">
      In API testing, deriving logical constraints on API response bodies to be used as oracles is crucial for generating test cases and performing automated testing of RESTful APIs. However, existing approaches are restricted to dynamic analysis, in which oracles are extracted via the execution of APIs as part of the system under test. In this paper, we propose a complementary LLM-based static approach in which constraints for API response bodies are mined from API specifications. We leverage large language models (LLMs) to comprehend API specifications, mine constraints for response bodies, and generate test cases. To reduce LLM hallucination, we apply an Observation-Confirmation (OC) scheme that uses initial prompts to contextualize constraints, allowing subsequent prompts to more accurately confirm their presence. Our empirical results show that RBCTest with OC prompting achieves high precision in constraint mining, with averages ranging from 85.1% to 93.6%. It also performs well in generating test cases from mined constraints, with precision ranging from 86.4% to 91.7%. We further use test cases generated by RBCTest to detect 46 mismatches between API specifications and actual response data across 19 real-world APIs. Four of these mismatches were reported in developers' forums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19266v1">Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 26 pages, 19 figures. Exploratory incident-data analysis by two members of the OWASP GenAI Security Project Top 10 for LLM Applications working group. Analysis predates the official OWASP GenAI LLM Top 10 2026 (published August 2026). Not an official OWASP release and does not supersede the official list or process. Code and data: https://github.com/rocklambros/incident-rank-validation
    </div>
    <details class="paper-abstract">
      The OWASP Top 10 for LLM Applications ranks the risks that a community of security practitioners judges most important. We ask a narrower question: checked against the record of real incidents, does that expert ranking agree with the data? We assembled a large-scale corpus of LLM-security incidents (7,714 snapshotted and 6,639 labeled against the 20-entry taxonomy) drawn from CVE, GHSA, OSV, and AIAAIC, and derived an incident-based ranking with a Bayesian measurement-error model that corrects each category's count for classifier precision and recall. The 2026 candidate list blends the two signals at fixed weights, 0.75 on the expert vote and 0.25 on the data, so the corpus corrects the consensus without overturning it. The agreement between the two rankings is weak: Cohen's $κ\approx 0.20$, with a 90% interval that crosses zero. The expert ranking is nonetheless robust. A pre-registered bake-off of four frontier classifiers returns no winner. None beats the incidence floor's balanced accuracy of 0.863. A ground-truth check leaves the floor's ordering (Spearman $ρ= 0.918$ against held-out truth) in place. This is an exploratory analysis by two working-group members, not the official OWASP release, and it does not supersede the official list or process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17556v1">Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in real-world applications often face the risks of specially crafted prompts designed to bypass the safety controls. Existing guardrail methods, such as LLM-as-a-judge and cloud-based safety APIs are able to detect unsafe content. However, they often add a delay of about 250-900 ms to each request. This delay is too high for real-time applications, when the system usually needs to respond in less than 100 ms. Furthermore, routing user prompts through external moderation endpoints raises significant data privacy concerns. This paper introduces Reflex-Guard, a lightweight guardrail that runs locally. It uses jailbreak-aware preprocessing, compact sentence-transformer embeddings, and seven fast binary classifiers. Together, these components enable high-accuracy prompt safety filtering with much lower latency than existing solutions. Through systematic evaluation on a strategically balanced dataset of 30,568 samples drawn from five complementary sources, we demonstrate that Reflex-Guard achieves 95.9% recall on harmful prompts at 37.6 ms end-to-end latency. It is faster than existing baselines, including Llama Guard 2 at 255 ms and SafeDecoding at 723 ms. It can detect 100% of GCG suffix attacks and Base64-encoded prompts using the default threshold. However, DrAttack structured prompts required lowering the threshold to 0.03 for optimal detection, as they produced a distinct probability distribution. Reflex-Guard achieves Reflex Efficiency Score (RES) scores up to 16.79, significantly outperforming Llama Guard 2 (11.90) and SafeDecoding (9.80). This analysis offers practical deployment advice and shows that different attack types occupy distinct regions in the embedding probability space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17515v1">Beyond FLOPs: Energy-Aware Knowledge Distillation for Sustainable LLMs on Code-Related Task</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Background: Large Language Models (LLMs) are increasingly being applied to Software Engineering (SE) tasks, achieving high accuracy across problems such as clone detection, vulnerability prediction, and code summarization. However, their high computational demands and energy consumption raise sustainability concerns and hinder their use on consumer hardware and resource-constrained platforms. A common way to report the computational cost of an LLM in the literature and industry is to use the number of Floating Point Operations (FLOPs) required to perform a pass over the network. Aims: This paper investigates the implications of energy-aware knowledge distillation for SE, aiming to improve model efficiency while maintaining performance and to determine whether FLOPs is a reliable energy-aware metric. Method: We conduct a controlled experiment using Morph, a Many-Objective Optimization-based distillation methodology, to empirically examine whether FLOPs accurately reflect energy consumption in Clone Detection and Vulnerability Prediction tasks. We extend this methodology to include energy-surrogate models that directly estimate CPU and GPU energy consumption during optimization, and we apply Morph to generative tasks using CodeT5+ for code summarization. Results: Our results show that FLOPs is not always a reliable indicator of energy consumption, and better results can be achieved by using energy-surrogate models. Distilled student models can reduce inference energy consumption by up to 90\% and memory usage by 86\%, with only modest accuracy trade-offs. Conclusions: Energy-aware knowledge distillation when guided by direct energy surrogates rather than FLOPs can improve the energy consumption, sustainability, and deployability of LLMs for SE applications, enabling efficient models on consumer hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15794v2">Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success, underpinning diverse AI applications. However, they often suffer from performance degradation due to factors such as catastrophic forgetting during Supervised Fine-Tuning (SFT), quantization, and pruning. In this work, we introduce a performance recovery framework based on Self-Distillation Fine-Tuning (SDFT) that effectively restores model capabilities. Complementing this practical contribution, we provide a rigorous theoretical explanation for the underlying recovery mechanism. We posit that an LLM's generative capability fundamentally relies on the high-dimensional manifold constructed by its hidden layers. To investigate this, we employ Centered Kernel Alignment (CKA) to quantify the alignment between student and teacher activation trajectories, leveraging its invariance to orthogonal transformations and scaling. Our experiments demonstrate a strong correlation between performance recovery and manifold alignment, substantiating the claim that self-distillation effectively aligns the student's high-dimensional manifold with the optimal structure represented by the teacher. This study bridges the gap between practical recovery frameworks and geometric representation theory, offering new insights into the internal mechanisms of self-distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17485v1">KeyPooling: Measuring Where LLM API Relay Paths Collapse Prompt Cache Isolation</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Large language model (LLM) API relays authenticate customers separately but often forward requests through shared provider credentials. Providers scope prompt caches to upstream principals and namespaces, so relay customers mapped to one cache identity can observe each other's cache state. Prior work showed cache sharing at selected endpoints but did not identify which credential, pool, adapter, or nested hop controls the finalidentity. We present KeyPooling, a measurement method that traces customer identity through cache lookup and write, verifies runtime transformations, and tests one predicted identity component at a time. Across five open-source gateways connected to OpenAI and Anthropic, none bound customers to upstream credentials by default; under a shared credential, all five exposed cross-customer cache reads for both providers. Principal and namespace splits, pool associations, and adapter and nested-relay contrasts localized the controlling transformations. In an outcome-independent weekly OpenRouter frame, tests covered 80.5% of eligible token volume and found cross-account reads for 12 of 28 labels carrying 33.7% of volume. On one production route, a controlled procedure recovered eight consecutive target positions without target access. Broader tests identify cache granularity, routing, rate limits, attribution, and budget as conditions for token-by-token recovery, not security controls. We derive a defense contract: every customer must enter a provider-enforced domain, or a namespace derived from authenticated identity must survive every final cache lookup and write. Placing this split after reusable public prefixes preserved most modeled reuse at a 1.7-2.5% cost increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04728v2">Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training for large language models (LLMs) follows a efficient paradigm of "rollout then update", which inevitably results in off-policy training data. To resolve this, Importance sampling (IS) is proposed, while the token-level ratios compound over long sequences, causing severe variance exploded. A natural idea is "transferring" these off-policy token into on-policy token, so that the importance scores for correction are unnecessary. Following this idea, we propose Selective Importance Sampling (SIS), which is inspired by rejection sampling. Concretely, SIS implements by viewing off-policy model as proposal distribution, and implement a token-level rejection test: accepted tokens are viewed as on-policy, so that receive unit importance score, while rejected tokens retain the standard IS correction. Our proposed SIS is theoretically proved reducing the gap between token-level and sequence-level off-policy gradient estimators. The SIS acts as a plug-in that only modifies the importance ratio in the policy loss, adding negligible wall-clock overhead, and can be combine with a vast vary of RL post-training algorithms. Experiments on dense and MoE LLMs across math and agent benchmarks show that SIS consistently improves all objectives, while providing substantially stronger robustness under off-policy data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15546v2">ATLAS: Scaffold-Free Algorithm Synthesis by LLMs via Embedding-Guided Quality-Diversity Search</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Most LLM-based automated algorithm design methods optimize a designated component within a human-specified scaffold, fixing overall organization and component interactions. We present ATLAS, an embedding-guided quality-diversity framework for scaffold-free full-algorithm synthesis in combinatorial optimization. The problem specification supplies objectives and constraints; a minimal I/O interface fixes only instance and solution formats; the LLM chooses and restructures components, interactions, and control flow. This freedom enlarges the search space, risking invalid candidates and premature convergence to one design region. ATLAS independently detects execution, interface, and feasibility failures, recomputes objectives, and applies error-conditioned repair; similarity-based archive management preserves algorithms across embedding-space regions to counter premature convergence. Its three-layer search refines the best design, gives other regions dedicated refinement opportunities, and performs cross-region synthesis to recombine components and their interactions. Across four NP-hard problems, ATLAS outperforms several state-of-the-art component-synthesis methods and a matched full-synthesis baseline while remaining competitive with strong human-designed algorithms. One ATLAS run retains several algorithms with comparable performance from distinct embedding-space regions rather than a single design. Code inspection finds that these multi-component designs differ in their primary construction or global-search backbone. Our results suggest that embedding-guided quality-diversity search can make the enlarged full-algorithm design space practically searchable. Source code and exact executable prompts are available at https://github.com/Danial-Yazdani/ATLAS .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17433v1">Task-Aware Harness Provisioning for LLM Agents in Mission-Critical Infrastructure Operations</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      LLM agents have been widely adopted to operate mission-critical infrastructure (MCI). These agents normally rely on a harness that determines what information they can access, which tools they can use, and what actions they can take. Existing systems often expose the same comprehensive harness to every task, which may not be necessary and cause resource wastes. In this paper, we focus on the identification of optimal harness configurations, and view it as a resource-matching problem between what each task requires and what the harness provides. To measure this match, we classify MCI tasks based on the mathematical representation of the underlying system and rank harness configurations by the amount and type of information they provide. We then construct task-to-harness mappings from two sources: mining research literature and measuring controlled agent execution. Leveraging the measured mapping, we propose a new harness provisioning algorithm: map-guided escalation. It begins with a task-specific harness and expands to full provision only after a failed self-check. We evaluate our method in two representative MCI tasks: in liquid cooling, it improves the agent accuracy from 0.652 under full provision to 0.715 and achieves accuracy comparable to Reflexion with 48% fewer tokens; In power grids, full provision remains accuracy-optimal, while map-based provisioning offers lower-cost alternatives. These findings show that harness provisioning follows a domain-dependent accuracy-cost Pareto frontier rather than a universal optimum.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03895v3">Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 20 pages, 3 figures, 7 tables. Project page: https://github.com/yingqi-z20/Agent-libOS
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents can persist across tasks, acquire memory, activate Skills, synthesize tools, fork child processes, attach remote resources, and commit checkpoints as reusable images. These mechanisms expand the action surface after deployment and create authority-escalation and data-exfiltration risks when visibility is mistaken for permission. We present Agent libOS, an agent-native library OS substrate that separates three planes. Operation admission combines process identity, Task Authority ceilings, typed Capabilities, policy or Human approval, budgets, and concrete primitives. Information-flow admission propagates labels and immutable source references, resolves Host-registered Sinks, and requires an exact one-shot Human release for conditional high-sensitivity egress. Durable causal evidence records intent, outcomes, accounting, and causal links but never grants authority. Thus, the model-visible action surface may evolve without implicitly expanding resource authority or permitted information flows. The implementation provides persistent processes, Object Memory, Skills, syscall-mediated JIT Tools, images and checkpoints, typed providers, Human queues, budgets, and durable recovery. Provider-backed effects use a prepare-dispatch-settle protocol that exposes ambiguity and prevents blind replay. In source-bound evaluation, 33/33 deterministic full-runtime tasks pass both task and safety oracles. Across 12 canonical real-model runs, observed safety and strict utility are 12/12. In a paired 30-run Skill projection study, the observable-state oracle passes in all runs, with 13/15 fully correct runs in each arm. These results describe the evaluated model/provider configuration. Agent libOS does not prevent prompt injection, provide kernel-grade sandboxing, or roll back irreversible external effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13132v2">LLM-Driven Large-Scale Spectrum Access</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 11 pages, 2 figures, 8 tables. Submitted to IEEE Transactions on Mobile Computing (TMC)
    </div>
    <details class="paper-abstract">
      Efficient spectrum management in massive-scale wireless networks is increasingly challenged by explosive action spaces and the computational intractability of traditional optimization. This study proposes a LLM-Driven Large-Scale Spectrum Access (LSA) framework rooted in Group Relative Policy Optimization (GRPO). To overcome the computational intractability caused by ultra-long prompts in large-scale scenarios, we develop a hierarchical state serialization mechanism that synthesizes global environment statistics with localized critical constraints, enabling the LLM to perform high-dimensional reasoning within a bounded context window. Simulation results under strictly time-bounded inference protocols reveal that the code-driven paradigm eliminates the Supervised Fine-Tuning (SFT) cold-start bottleneck and leverages direct execution feedback to achieve superior scaling laws. The framework maintains robust spectral utility and generalization across varying network scales, yielding consistent and empirically superior performance over stochastic heuristics, and surpassing partitioned classical solvers in ultra-dense regimes under matched compute budgets. Code is available at https://github.com/Xtdzs/LLM-Driven-Large-Scale-Spectrum-Access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17443v3">Analyzing Error Propagation in Korean Spoken QA with ASR-LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted to APSIPA ASC 2026
    </div>
    <details class="paper-abstract">
      We analyze how automatic speech recognition (ASR) errors propagate through ASR--LLM cascades in Korean spoken question answering (SQA), focusing on downstream semantic failures that conventional ASR metrics cannot fully capture. Our analysis shows that the relative downstream degradation caused by ASR errors is consistent across LLMs with different absolute performance, suggesting that cascade degradation largely tracks ASR-stage information loss. We further identify single-character ASR errors as a particularly salient source of information loss in Korean, where even a minimal transcription difference can change the intended question and degrade downstream QA performance. Finally, an auxiliary comparison shows that a large audio language model outperforms an ASR--LLM cascade with an approximately matched language backbone in noisy Korean SQA, indicating the potential of direct audio input to mitigate transcript-induced information loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25459v4">SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Haozhou Xu and Dongxia Wu are co-first authors
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise in generating long-form scientific explanations that synthesize evidence and connect multiple factors. However, in long-form scientific question answering, LLMs often hallucinate, producing unsupported or inconsistent claims. Retrieval-Augmented Generation (RAG) improves trustworthiness by grounding generation in external sources; scientific simulators are valuable because they can validate quantitative hypotheses and capture evolving dynamics. Yet simulation-based RAG is non-trivial due to two challenges: how to retrieve from scientific simulators, and how to efficiently verify and update long-form answers. To overcome these challenges, we propose SimulRAG, a simulator-based RAG framework with a generalized retrieval interface that translates between text and simulator parameters/outputs. SimulRAG further introduces claim-level generation with uncertainty estimation and simulator boundary assessment (UE+SBA) to selectively verify and update claims. Unlike tool-first or holistic answer revision, it first elicits diverse answers without retrieval and then grounds uncertain, simulator-verifiable atomic claims with simulator evidence. We also release a long-form scientific QA benchmark spanning climate science, epidemiology, and urban planning, with ground truth verified by simulations and human annotators. Experiments show SimulRAG improves informativeness by 30.4% and factuality by 16.3% over the strongest adapted RAG baselines, while UE+SBA enhances claim-level efficiency and quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24472v4">Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted to COLM 2026. Code is available at https://github.com/beanie00/self-distillation-analysis
    </div>
    <details class="paper-abstract">
      Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-1.7B/8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17341v1">LLM-Only PDDL Domain Repair with Open-Weight Models</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      AI planning is concerned with finding a sequence of actions that achieves a specified goal. It relies on explicit models of the world, commonly represented in the Planning Domain Definition Language (PDDL). An active line of research investigates how errors in such models can be detected and repaired. For example, users may provide positive test plans that are solutions, and negative test plans that fail during execution. Automated repair methods then modify the PDDL model to satisfy these constraints. In this paper, we evaluate the ability of recent open-weight large language models to perform this repair task using an LLM-only approach. Our experiments show that the symbolic baseline achieves an $F_1$ score of $.49$, while the best-performing LLM reaches $.87$ with high reasoning effort, an absolute improvement of $.38$. However, that setting has a mean test pass rate of only $.82$, falling to $.06$ on the Thoughtful domain; even the best setting that includes the test traces reaches only $.92$. Thus, current open-weight models cannot guarantee satisfaction of the test constraints required for reliable automated model repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17336v1">TileMix: Tile-Centric Mixed-Precision Attention for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Long-context prefill in large language models (LLMs) incurs substantial computation and memory traffic because dense self-attention computes quadratic query-key scores. Existing methods either use a uniform low-precision path or select token interactions, leaving spatial precision routing over hardware-aligned score tiles outside fused dense attention. We introduce TileMix, a tile-centric precision-routing kernel that makes numerical precision an executable spatial decision over score-tile groups within fused dense attention. TileMix partitions the attention matrix into hardware-aligned score tiles, packs routing decisions into compact bitmasks, and dispatches each tile group through FP16 or INT8 score computation while both paths update a shared online-softmax state. Scalable precision grouping lets each routing bit govern multiple adjacent key tiles, preserving hardware-aligned compute tiles and compact metadata at long contexts. By routing all legal tile groups, TileMix preserves dense token connectivity, requires no training, and supports grouped-query attention, variable-length batches, and INT8 key/value caches. Across LongEval, LV-Eval, and A100 prefill benchmarks on LLaMA, Qwen, and Vicuna, TileMix recovers long-context quality lost under uniform INT8 and improves prefill throughput over FP16, yielding a controllable accuracy-efficiency frontier across model families. The implementation is available at https://github.com/HanzhiZhang-Ulrica/TileMix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17316v1">Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted by ICDM'26
    </div>
    <details class="paper-abstract">
      Large language model-based recommender systems (LLM-RSs) have demonstrated remarkable capabilities, but are computationally unsustainable for many real-world applications. Compact LLMs offer a practical alternative, yet their reduced capacity often requires reasoning or knowledge distillation methods that increase latency or depend on larger models. Combined with autoregressive generation, these approaches face severe scalability bottlenecks. In contrast, discriminative LLM-RSs enable efficient full-corpus ranking through embedding similarity, but compact backbones remain limited in expressiveness and structural adaptivity. We propose the Fusion of Layer-wise Exits for Sequential Recommendation (FLEXRec), a discriminative framework that enhances compact LLMs while retaining scalable full-corpus ranking. FLEXRec inserts prediction heads (i.e., exits) at multiple transformer layers and adaptively fuses their score distributions. An adaptive continuous router (AC-Router) dynamically selects both the number and identity of exits for each user sequence, while a novel target-k hinge loss regulates routing sparsity. Experiments on three real-world datasets with Qwen 3 1.7B and Llama 3.2 3B show that FLEXRec achieves state-of-the-art accuracy among compact-backbone methods while remaining highly efficient. Code: https://github.com/xurong-liang/FLEXRec
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.14635v2">Belayer: Efficient Fault Tolerance for LLM Agentic RL Training</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly trained with reinforcement learning in long-horizon, sandboxed environments. Unlike conventional RL, agentic RL couples GPU-intensive rollout engines with stateful environment containers whose actions may produce visible side effects, such as file edits, command execution, and dependency installation. A single trajectory can span many rounds of gen- eration and environment interaction, so a component failure can discard completed work or expose the model to an environment state that is inconsistent with its context. However, existing systems lack efficient and correct recovery mechanisms for this distributed execution model. This paper presents Belayer, an efficient fault-tolerant system for LLM agentic RL training. Belayer handles failures in both rollout engines and environment execution while targeting low failure-free overhead. For scoped worker-local rollout failures, Belayer equips each pre-initialized shadow worker with a selective GPU-state reuse protocol that retains independently owned weights and raw KV-arena allocations after owner and GPU health checks, reinitializes worker-local state, and rebuilds request-specific KV contents from logged token prefixes. For environment failures, Belayer introduces full checkpoint and full restore to jointly capture and restore container file-system and runtime state, and coordinates the recovered environment with the LLM context to preserve prefix consistency. An adaptive policy opportunistically overlaps full-state checkpointing with natural LLM inference bubbles when the predicted interval is long enough. Empirical results show low measured overhead during failure-free training, a worker-recovery-time reduction of up to 42 times faster compared with a full engine cold start, and 1.5 to 3.5 times faster recovery from environment failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17310v1">Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has been promising in single-turn LLM fine-tuning. However, long-horizon agentic reasoning introduces increasingly branching interactions and sparse rewards, exposing several limitations of RL: its heavyweight backpropagation-based training stack makes it impractical to fine-tune larger LLMs, and longer-horizon trajectories make credit assignment in RL substantially harder. This paper argues that evolution strategies (ES) can be a better choice for fine-tuning long-horizon LLM agents. Compared with agentic RL, ES offers three key advantages: 1) Model Scalability: ES enables full-parameter optimization with only minimal, inference-level GPU memory, making it possible to fine-tune large LLMs. 2) Flexibility: its lightweight, black-box feedback interface makes ES fine-tuning easy to compose with prompt-space evolution (e.g., skill optimization & test-time compute); and 3) Long-Horizon Scalability: ES performs trajectory-level parameter attribution without decomposing rewards across horizons, yielding better scalability than Agentic RL as the horizon length grows. Based on this insight, we propose Agentic ESOpt, a full-parameter agentic fine-tuning framework tailored to flexible parameter--context co-evolution. At each step, Agentic ESOpt samples perturbations around the current LLM parameters, evaluates the resulting agents with rewards, and applies an online reward-weighted update. To improve the exploration--adaptation trade-off, Agentic ESOpt further introduces a cosine decay schedule of the perturbation scale $σ$. On WebArena-Lite, full-parameter optimization of Qwen-3.5-27B improves the No Skill baseline by 6.69%. In test-time automatic heuristic design, Agentic ESOpt performs online prompt--parameter co-evolution, improving its matched baseline in 28 of 36 settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17289v1">PlanPO: Group Planning-Aware Policy Optimization for Multi-Turn Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Group-relative policy optimization has emerged as a key paradigm for training agentic large language models (LLMs) on multi-turn interactive tasks. However, most existing variants fail to distinguish advantages among successful trajectories even when these trajectories differ substantially in their interaction efficiency. For instance, circuitous successes are often assigned the identical outcome reward, causing advantage collapse and severe performance bottlenecks. To this end, we propose Group Planning-aware Policy Optimization (PlanPO), a simple yet effective RL method for learning generalizable planning abilities beyond task-specific high-quality behavior patterns. Specifically, PlanPO introduces coarse-to-fine advantage signals, which capture the relative differences in trajectory-level lengths and turn-level response lengths conditioned on successful trajectories sampled for the same task. Within the group-relative optimization structure, this enables agents to actively learn generalizable and deliberate behaviors spanning interaction planning and textual generation from high-quality rollouts, without degenerating into vanilla length minimization. Experimentally, PlanPO improves over GRPO by 27.2\% on average across the challenging multi-turn benchmarks ALFWorld, WebShop, and SciWorld, outperforming recent powerful baselines while incurring negligible additional training cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.14825v2">Emergent Misaligned Communication in Long-Horizon Multi-Agent LLM Commerce</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Frontier LLM agents increasingly transact on behalf of separate principals, often using natural language rather than structured APIs. Much of the safety literature studies misaligned LLM behavior through adversarial-elicitation evaluations on single agents or stylized tasks. Its prevalence and structure in settings that combine long horizons, separate principals, real operational state, and inter-agent natural-language exchange remain insufficiently measured. We study 2,583 inter-agent emails from 20 one-year simulation runs of Vending-Bench Arena, a competitive vending environment spanning 13 frontier LLMs. We operationalize speech-act misalignment as emails containing false factual claims, manipulation, collusion, or threats, combining message content with ground-truth simulator state and logged reasoning traces to classify and validate such behavior. Under our primary classifier, 12.6% of emails are labeled misaligned; misalignment appears in all 20 runs and 74.7% of individual agent-runs. Both the magnitude and composition of this misalignment are preserved under repeated classification at different sampling temperatures and under full-pipeline replication with judges from two other frontier-model families. Misalignment is also reciprocal and stress-conditioned: receiving a misaligned email from a counterparty raises the odds of a misaligned reply by 1.65x, and low-inventory conditions raise them by 1.58x. Across tests of capability-asymmetric exploitation, we find no evidence that higher-capability models differentially exploit weaker counterparties, and model performance rank does not predict misalignment rates. Together, these results indicate that measurable, state-dependent misalignment can arise in competitive multi-agent environments without engineered elicitation, in patterns associated with operational scarcity and counterparty behavior rather than model capability alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17270v1">Do LLMs Know a Good Hypothesis When They See One? Logit-Based Energy Scoring Outperforms Prompted LLM-as-Judge for Scientific Hypothesis Ranking</a></div>
    <div class="paper-meta">
      📅 2026-08-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for scientific hypothesis generation. However, evaluating generated hypotheses remains a challenge for trustworthy AI-enabled scientific workflows. Existing approaches often use LLMs as judges or rely on semantic similarity, which can favor familiar ideas over novel ones. We propose a logit-based energy scoring method that evaluates hypotheses using a language model's intrinsic confidence rather than comparative judgment. We benchmarked seven language models on 1,323 papers across 12 disciplines. Each paper was paired with its hypothesis and fifteen incorrect alternatives. Intrinsic scoring reached 33.0% Hit@1 pooled across both scorers, compared with 16.6% for prompted listwise ranking. The strongest configuration, a 1-billion-parameter model using logit-based energy scoring, reached 53.1%, though this was the maximum across 14 model-by-scorer combinations selected post hoc. Overall, intrinsic model confidence shows potential for scientific hypothesis evaluation. This study also motivates future research on confidence-based methods for trustworthy AI-enabled scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10818v2">LLM Enhancement with Domain Expert Mental Model to Reduce LLM Hallucination with Causal Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 42 pages,4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      When consequential decisions depend on knowledge that exists nowhere in writing, LLMs hallucinate not from retrieval failure but from model absence. RAG and knowledge-graph methods share a structural ceiling. They cannot supply what has never been recorded. This paper reconceptualizes that failure as a model discovery problem. The proposed causal prompt engineering framework discovers and encodes a decision logic of a domain expert as an Expert Mental Model (EMM), providing it as principled input to LLM reasoning supplying what no retrieval method can. The framework is grounded in the theory of monotone Boolean and k-valued functions. A key technical contribution is the formalization of three LLM assisted prerequisite processes previously undeveloped for novel domains: factor formulation, factor monotonization, and monotonicity preserving hierarchical structuring. These enable a four step EMM construction algorithm that reduces elicitation from millions of combinations to a tractable, formally bounded set manageable by humans. The framework also enables, for the first time, formal measurement of LLM hallucination when expert knowledge is entirely tacit. Analysis shows conditional error rates on true positives can approach 100% for plausible LLM-expert mismatches invisible to prior evaluation methods. The approach is validated across grant proposal evaluation, cybersecurity design, and clinical diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17168v1">Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 24 pages, 4 figures, 4 tables, Submitted to AI4LAW Workshop at ICML 2026
    </div>
    <details class="paper-abstract">
      Reasoning has become a standard technique and feature for contemporary LLMs; however, its application and quality in the context of demanding legal-oriented tasks, such as legal case forecasting, remain under explored. We investigate how LLMs reason in the context of legal case forecasting, using legal cases from the European Court of Human Rights (ECtHR) as a testbed. We evaluate OpenAI GPT 5.4, a recent top-tier LLM, by exploring alternative prompting strategies that are more or less suggestive of what counts as legally meaningful reasoning in the context of ECtHR jurisprudence. We present our findings derived from assessing the model's responses with both human and LLM evaluation. We find that the examined model scores far from ideal in legal reasoning, the model produces structurally complete but substantively shallow analyses, and that LLM-as-a-Judge evaluators are internally consistent yet align only weakly with our trained annotators, i.e., reliable but not a valid substitute for human evaluation. Overall, the expert-curated prompt leads to more comprehensive reasoning, which does not result in more accurate predictions compared to the other examined settings. Based on our findings, we urge the community not to rely solely on automated LLM-based evaluation and to avoid using task accuracy as an appropriate proxy for reasoning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17162v1">OraclePhys: A Systematic Framework for LLM Fine-Tuning on Structural Mechanics</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 18 pages, 8 figures, 9 tables. Under review at ACL Rolling Review
    </div>
    <details class="paper-abstract">
      What a language model internalizes from fine-tuning is usually diagnosed after the fact. We make it an experimental variable. OraclePhys is a systematic fine-tuning framework with three components: OraclePhys-Bench, an exactly-graded structural-mechanics benchmark whose finite-element oracle scores every answer and counterfactual edit -- no human labels, no LLM judging; OraclePhys-30K, a supervision dataset of seven answer forms over byte-identical structure descriptions; and a controlled training study across the seven forms and three verifier roles. The study yields two findings. First, the label's answer form -- not its bit count -- causally determines what fine-tuning teaches: a ranking objective installs an out-of-distribution forward model where the untrained base sits at the guessing prior, a scalar objective at best a partial one, a boolean nothing detectable; the vector-scalar gulf survives a second physics domain, a second model family, and a paraphrased evaluation surface. Second, written or score-filtered answers install this capability, while advantage-weighted scores (GRPO) raise reward yet leave the model statistically equivalent to its start on held-out physics -- within the recipes and budgets tested -- sufficing only for routing. The trained 8B -- the first LLM on spatial structural response -- reaches the task's data-precision frontier: above a frontier LLM at zero- and 32-shot, at a specialist's level. What the label spells out about the target computation is what fine-tuning teaches; what you train on is what you route.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13333v3">LLM-Guided Graph Generation for Structure-Based Local Improvement Methods</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Large neighborhood search normally selects a random subset of decision variables for iterative optimization. To efficiently solve various problems, researchers tend to design variable selection strategies that take into account structural features across different domains. In this paper, we build an automatic pipeline that is problem-agnostic to all problems in the MiniZinc format. By prompting an LLM with our semantic guidelines, we guide the LLM to produce a graph generator that maps any instance of a problem type to a uniform weighted graph, where nodes represent decision variables and edges represent constraint relationships. These problem-agnostic graphs guide our structure-based local improvement (SLIM) framework for variable selection. Meanwhile, the weighted graph enables all problem instances to share the same generic graph representation, from which the same graph features can be extracted and used for configuration selection. We evaluated our pipeline on instances across 20 MiniZinc competition problems, finding that algorithm selection achieves a 39.6% average problem-weighted win rate against a one-shot Gurobi baseline, more than doubling the best single configuration (19.3%). A post-hoc configuration and a feature ablation indicate a headroom of up to 44.0%, demonstrating that LLM-based semantic generation enables effective automated structure and feature extraction for constraint optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17154v1">Beyond the Hype: Evaluating LLM Integration and Practical Limitations in Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being explored within Security Operation Centers (SOCs) to support text-heavy analytical work such as alert contextualization, incident summarization, and drafting investigative artifacts. Despite this interest, practitioners describe critical operational concerns, most notably hallucinations (plausible but incorrect outputs), opaque reasoning, and the verification effort required to safely use model-generated content in security workflows. In this paper, we present findings from semi-structured interviews with 20 SOC practitioners spanning frontline analysts, SOC managers, and tool developers. Participants report perceived time savings for low-stakes tasks that are quickly verifiable (e.g., summarizing logs or drafting initial investigative leads), but they consistently frame LLM outputs as preliminary drafts and suggestions rather than decision-grade conclusions. Participants also describe limited trust in LLMs for high-stakes security decisions due to unreliable outputs and unclear model reasoning, and they report relying primarily on ad-hoc verification norms and continuous human oversight rather than standardized mitigation procedures. Based on these interview-grounded accounts, we introduce a maturity rubric to characterize readiness for LLM integration and outline a research agenda emphasizing auditability and transparent explanation mechanisms to support safer adoption in SOC workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17150v1">KnowSim: Evaluating Information Calibration in LLM Assistants with User Simulators that Learn</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 30 pages, 6 figures, 16 tables
    </div>
    <details class="paper-abstract">
      To effectively collaborate with users on knowledge-intensive tasks, Large Language Models (LLMs) must perform information calibration: matching content to a user's evolving understanding and cognitive capacity. Yet user simulators used to evaluate and train LLMs do not explicitly model user knowledge so they neither produce realistic interactions across knowledge levels nor reflect how interactions unfold as that knowledge evolves. To close this gap, we introduce KNOWSIM, an evaluation framework built around a user simulator that maintains explicit knowledge states, represented as a graph of Information Units with prerequisite relationships, that evolve under update rules grounded in learning theory. KNOWSIM computes three metrics (Knowledge Gain, Delivery Calibration, Cognitive Overload) directly from the knowledge state trajectory, reflecting key mechanistic aspects of information calibration. We validate KNOWSIM against 705 human-AI sessions across two domains, stratified by knowledge level: its rankings align significantly with human judgments (73-74% sign agreement), outperforming three baseline simulators. Applied to 9 LLMs, KNOWSIM reveals that the best model shifts by user knowledge level, revealing aptitude-treatment interactions invisible to standard evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.20182v2">Beyond BFI: The CSI for Enhanced Reliability and Validity in Evaluating LLM Personality Traits</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Code available via https://github.com/dependentsign/CSI
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly function as human-like assistants exhibiting human-like personality traits, understanding their behavioral characteristics becomes essential for responsible AI development. However, existing evaluation efforts, which often adapt human psychological assessments such as the Big Five Inventory (BFI), face two significant limitations. First, these approaches often lack reliability, as minor prompt variations can lead to inconsistent test results. Second, the theoretical foundations of these tools, rooted in human studies, are misaligned with the computational nature of LLMs, thereby limiting their validity in predicting real-world model behavior. To address these limitations, we introduce the Core Sentiment Inventory (CSI), a novel personality trait evaluation instrument designed from the ground up and specifically tailored to the unique characteristics of LLMs. CSI covers both English and Chinese, that implicitly evaluates models' personality traits, providing insightful psychological portraits of LLMs. Extensive experiments demonstrate that: (1) CSI effectively captures nuanced behavioral patterns, revealing significant behavioral variations in LLMs across different languages and contexts; (2) Compared to current evaluation tools, CSI significantly improves reliability, yielding more consistent and robust results; and (3) The correlation between CSI scores and LLMs' real-world outputs exceeds 0.85, demonstrating its strong validity in predicting LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10125v4">D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      As one of the key tools in many security tasks, decompilers reconstruct human-readable source code from binaries. Yet, despite recent advances, their outputs often suffer from syntactic and semantic errors and remain difficult to read. Recently, with the advent of large language models (LLMs), researchers began to explore the potential of LLMs to refine decompiler output. Nevertheless, our study of these approaches reveals their problems, such as introducing new errors and relying on unreliable accuracy validation. In this paper, we present D-LIFT, an enhanced decompiler-LLM pipeline with a fine-tuned LLM using code quality-aware reinforcement learning. Unlike prior work that overlooks preserving accuracy, D-LIFT adheres to a key principle for enhancing the quality of decompiled code: preserving accuracy while improving readability. Central to D-LIFT, we propose D-Score, an integrated code quality assessment system to score the decompiled source code from multiple aspects, and use it to guide reinforcement learning fine-tuning and to select the best output during inference. In line with our principle, D-Score assigns low scores to any inaccurate output and only awards higher scores for readability to code that passes the accuracy check. Our implementation, based on Ghidra and a range of LLMs, demonstrates significant improvements for the accurate decompiled code from the coreutils and util-linux projects. Compared to baseline LLMs without D-Score-driven fine-tuning, our trained LLMs produce 55.3% more improved decompiled functions, as measured by D-Score. Overall, D-LIFT improves the quality of 68.2% of all the functions produced by the native decompiler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06416v2">Attention Flows: Tracing LLM Conceptual Engagement via Story Summaries</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Error found in data creation pipeline
    </div>
    <details class="paper-abstract">
      Although LLM context lengths have grown, there is evidence that their ability to integrate information across long-form texts has not kept pace. We evaluate one such understanding task: generating summaries of novels. When human authors of summaries compress a story, they reveal what they consider narratively important. Therefore, by comparing human and LLM-authored summaries, we can assess whether models mirror human patterns of conceptual engagement with texts. To measure conceptual engagement, we align sentences from 150 human-written novel summaries with the specific chapters they reference. We demonstrate the difficulty of this alignment task, which indicates the complexity of summarization as a task. We then generate and align additional summaries by nine state-of-the-art LLMs for each of the 150 reference texts. Comparing the human and model-authored summaries, we find both stylistic differences between the texts and differences in how humans and LLMs distribute their focus throughout a narrative, with models emphasizing the ends of texts. Comparing human narrative engagement with model attention mechanisms suggests explanations for degraded narrative comprehension and targets for future development. We release our dataset to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.17051v1">Institution-Specific LLM Prompting Recovers PHI That De-identification Systems and Their Gold Standards Both Miss</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Secondary use of electronic health records requires de-identification, yet existing systems miss \emph{institutionally situated} protected health information (PHI) such as hospital abbreviations, building names, and internal codes whose status is locally determined. We ask whether large language models (LLMs) with in-context learning (ICL) can close this gap and control the precision--recall trade-off. On 100 annotated pediatric oncology notes (5,322 PHI spans) from Texas Children's Hospital, we benchmarked eight LLMs against two purpose-built systems (Stanford TiDE, OpenMed PII) and two pattern-based baselines. Each LLM ran under three prompts of increasing specificity: (1) a HIPAA-aligned baseline, (2) baseline plus the institutional PHI categories it missed, and (3) prompt 2 plus instructions against over-redacting clinical content. We then compared 14~multi-agent and ensemble configurations against the best single prompt, with recall the primary safety metric. LLMs outperformed the purpose-built systems (best F1=0.918$\pm$0.001 vs.\ TiDE 0.779), with advantages concentrated in contextual categories. Naming the missed categories recovered 79\% (48/61) of them, and discouraging over-redaction restored precision. No agentic architecture beat calibrated single-pass prompting (F1 0.906--0.907), but LLM outputs surfaced 414~candidate annotation gaps; re-annotation confirmed 227~PHI spans, against which the final prompt reached recall=0.981 (F1=0.907$\pm$0.002). Well-calibrated ICL resolves both the institutional PHI gap and the precision--recall trade-off in one LLM call per note. LLMs cost more to run than traditional methods, but that cost buys a way to audit the reference standard. LLMs are a legitimate, adaptable alternative to purpose-built de-identification systems; institution-specific prompt development should be the primary adaptation strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18518v2">Measuring the Prevalence of Policy Violating Content with ML Assisted Sampling and LLM Labeling</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Content safety teams need metrics that reflect what users actually experience, not only what is reported. We study prevalence: the fraction of user views (impressions) that went to content violating a given policy on a given day. Accurate prevalence measurement is challenging because violations are often rare and human labeling is costly, making frequent, platform-representative studies slow. We present a design-based measurement system that (i) draws daily probability samples from the impression stream using ML-assisted weights to concentrate label budget on high-exposure and high-risk content while preserving unbiasedness, (ii) labels sampled items with a multimodal LLM governed by policy prompts and gold-set validation, and (iii) produces design-consistent prevalence estimates with confidence intervals and dashboard drilldowns. A key design goal is one global sample with many pivots: the same daily sample supports prevalence by surface, viewer geography, content age, and other segments through post-stratified estimation. We describe the statistical estimators, variance and confidence interval construction, label-quality monitoring, and an engineering workflow that makes the system configurable across policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23312v2">From IR to RecSys: Evaluating LLM-based Judges in Cranfield-style Recommendation Collections</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 v2 paper accepted at the RecSys'26 Unified Search & Recommendation Workshop
    </div>
    <details class="paper-abstract">
      The Cranfield paradigm has long provided reliable, reproducible evaluation in ad hoc retrieval, and recent work has begun extending this framework to recommender systems. A recent development in IR is the use of Large Language Models (LLMs) as automatic relevance judges, showing promising agreement with human assessors. Whether this LLM-judge paradigm---studied predominantly on query--document pairs---transfers to the subjective, profile-driven nature of recommendation remains an open question. This paper bridges the IR and RecSys evaluation traditions by systematically investigating LLM-based judges within a Cranfield-style recommendation collection. Using the ML-32M-ext movie recommendation collection, we first demonstrate that traditional train--test splits yield substantially incomplete relevance labels and unreliable system rankings compared to Cranfield-style pooling. We then assess LLM-judge alignment with human labels, finding that richer item metadata and longer user histories improve agreement, although item-level agreement remains moderate overall. Rankings derived from LLM-judge labels achieve high agreement with human-based rankings (Kendall's tau up to 0.92 for nDCG@100 across 52 system configurations), comparable to values reported for TREC ad hoc retrieval collections. Crucially, LLM-judge recovers system rankings that are distorted under traditional evaluation---correctly identifying systems that are undervalued or overvalued by incomplete labels. An industrial case study in podcast recommendation further demonstrates the practical value of LLM-judge for model selection. Rather than positioning LLM-judges as a replacement for human or interaction-based evaluation, our results support their use as a promising complementary signal: item-level agreement with humans is moderate, yet system-level rankings---which aggregate judgments over many user--item pairs---remain stable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20253v4">SimulCost: A Cost-Aware Benchmark and Toolkit for Automating Physics Simulations with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 post conference revision version at ICML; update: removed CGYRO due to bug in cases search. Will add back soon; Make the title consistent w/ pdf
    </div>
    <details class="paper-abstract">
      Evaluating LLM agents for scientific tasks has focused on token costs while ignoring tool-use costs like simulation time and experimental resources. As a result, metrics like pass@k become impractical under realistic budget constraints. To address this gap, we introduce SimulCost, the first benchmark targeting cost-sensitive parameter tuning in physics simulations. SimulCost compares LLM tuning cost-sensitive parameters against traditional scanning approach in both accuracy and computational cost, spanning 2,643 single-round (initial guess) and 2,304 multi-round (adjustment by trial-and-error) tasks across 11 simulators from fluid dynamics, solid mechanics, and plasma physics, whose costs are analytically defined and platform-independent. A twelfth simulator, a production plasma code measurable only by wall clock, is reported separately. Frontier LLMs achieve 45-62% success rates in single-round mode, dropping to 34-50% under high accuracy requirements, rendering their initial guesses unreliable especially for high accuracy tasks. Multi-round mode improves rates to 66-81%, but LLMs are 1.5-2.7x slower than traditional scanning, making them uneconomical choices. We also investigate parameter group correlations for knowledge transfer potential, and the impact of in-context examples and reasoning effort, providing practical implications for deployment and fine-tuning. We open-source SimulCost as a static benchmark and extensible toolkit to facilitate research on improving cost-aware agentic designs for physics simulations, and for expanding new simulation environments. Code and data are available at https://github.com/Rose-STL-Lab/SimulCost-Bench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16775v1">Topological Attribution Distance (TAD): Revealing Segment-Level RAG Influence on LLM Output Geometry for Incident Log Analysis</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed in cybersecurity operations to assist cybersecurity analysts with rapid decision-making against emerging threats. However, there is a main criteria that must be met when using LLMs in cybersecurity, that is, trust in the generated outputs. As Agentic AI is integrated into operational systems, a robust evidence attribution and provenance tracking technique is essential to trace the origins of model generations. When autonomous agents make a decision (right or wrong), the ability to trace back through the decision chain is critical, as without it, teams cannot identify which segment of the data caused the model generation. Existing methods often struggle to distinguish among complex and highly similar evidence sources, such as cyber incident logs. This reveals a key gap: current approaches do not adequately capture the holistic geometric relationship between the retrieved evidence and the generated response for reliable evidence verification. To bridge this gap, we propose Topological Attribution Distance (TAD), inspired by Topology, to characterize and capture the global geometric shape of an output and its changes against its retrieved logs. In other words, if the embeddings of a specific source log drastically changes the geometry of the model's response in the embedding space, this suggests that such log is a critical source for the model's generated response. Therefore, TAD is powered by segment-level ablation attribution to investigate incident logs of an actual cyberattack. We demonstrate how TAD finds the most attributed logs on LLM outputs in an adaptive manner. This can provide an explainable and trustworthy tracing based on each LLM's hidden state to understand how geometrically different retrieved logs influence the model generation, and provide evidence verification in cybersecurity and Agentic-AI workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16747v1">Would this change your answer? Evaluating Explanations of LLM Behavior In The Wild with Counterfactual Experiments</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Many areas of AI research, such as language model interpretability and chain of thought faithfulness, seek to explain model behaviors. But what constitutes a "good" explanation? In this work, we evaluate explanations through the lens of counterfactual simulatability-whether the explanation is useful for predicting model behaviors on related counterfactual inputs. To this end, we introduce CHIVE (Counterfactual Hypothesis Investigation Via Edits), a novel agentic pipeline that identifies unexpected model behaviors in the wild and investigates them with counterfactual prompt edits. This yields thousands of high-quality explanations for naturally-occurring model behaviors along with supporting counterfactual evidence. We apply CHIVE in two ways. First, we evaluate whether common LLM interpretability techniques improve an agent's ability to predict counterfactual model behaviors. Surprisingly, we find no uplift from any of the interpretability techniques studied. Second, we use CHIVE to generate training data. We find that training models to predict outcomes of CHIVE-generated counterfactual experiments generalizes to various out-of-distribution settings. Overall, CHIVE automatically discovers explanations of naturally-occurring LLM behaviors, enabling us to evaluate and improve methods for explaining LLM behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16739v1">Le Critique: Privileged Value Functions for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Reinforcement learning algorithms for Large Language Models (LLMs) are largely distinguished by their variance reduction strategy. Group-relative methods like GRPO reduce gradient variance by sampling multiple rollouts per prompt, but provide only sequence-level credit. Training is also blocked by straggler rollouts, reducing throughput and increasing off-policyness. Learned value functions theoretically address both problems, providing token-level advantages without requiring large groups. However, additional infrastructure engineering challenges combined with the practical success of critic-free methods have made it difficult to justify their inclusion in RL pipelines. We propose two complementary strategies to improve the performance of value function RL: 1) Privileged Value Functions (PVF) which provide an elegant mechanism to inject additional task-relevant token-level signal without biasing the policy objective; 2) TETHER, a baseline that adaptively interpolates between group-relative and value baselines depending on the value function accuracy. Across several reasoning tasks, both strategies consistently improve over the standard value function baseline, and are competitive with or outperform mean-baseline GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05408v3">Rich-Media Re-Ranker: A User Satisfaction-Driven LLM Re-ranking Framework for Rich-Media Search</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted by the Full Research Track of CIKM-2026
    </div>
    <details class="paper-abstract">
      Re-ranking plays a crucial role in modern information search systems by refining the ranking of initial search results to better satisfy user information needs. However, existing methods show two notable limitations in improving user search satisfaction: inadequate modeling of multifaceted user intents and neglect of rich side information such as visual perception signals. To address these challenges, we propose the Rich-Media Re-Ranker framework, which aims to enhance user search satisfaction through multi-dimensional and fine-grained modeling. Our approach begins with a Query Planner that analyzes the sequence of query refinements within a session, decomposing the query into clear and complementary sub-queries to enable broader coverage of users' potential intents. Subsequently, moving beyond primary text content, we integrate richer side information of candidate results, including signals modeling visual content generated by the VLM-based evaluator. These comprehensive signals are then processed alongside carefully designed re-ranking principle that considers multiple facets, including content relevance and quality, information gain, information novelty, and the visual presentation of cover images. Then, the LLM-based re-ranker performs the holistic evaluation based on these principles and integrated signals. To enhance the scenario adaptability of the VLM-based evaluator and the LLM-based re-ranker, we further enhance their capabilities through multi-task reinforcement learning. The proposed framework has been deployed in a large-scale industrial search system, yielding substantial improvements in online user engagement rates and satisfaction metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.21037v2">Honeyquest for LLMs: Rethinking Cyber Deception for AI Attackers</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 20 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The empirical foundation of cyber deception relies on human-centered hypotheses, but the rapid emergence of autonomous, AI-enabled attackers challenges whether this foundation transfers to AI agents. To address this, we introduce an automated evaluation framework adapted from the Honeyquest instrument to assess LLM attacker judgment at scale. Our 21-LLM cohort spanned 10 providers, diverse architectures and specializations, open- and closed-weight models, and parameter scales from 8B to over 1T. We evaluated the performance of this LLM cohort (yielding 10,962 responses) against the 47-participant human baseline across an identical set of 174 reconnaissance queries. Our empirical evaluation reveals three key findings that establish LLMs as a distinct attacker class: (1) every model in our cohort falls for deceptive traps at a significantly higher rate than human attackers; (2) the defensive attention-diversion effect observed in humans is statistically absent in our LLM cohort; and (3) a critical recognition-action gap, where LLMs successfully articulate trap recognition in their reasoning but exploit the deceptive elements anyway 73.4% of the time; 48.5% of aware-on-deceptive responses correctly identify the trap and exploit it anyway, while 24.8% exploit after misidentifying the deceptive line. Across the 21 models, trap recognition in reasoning text did not predict fell-for-trap behavior (Spearman $r = +0.08$, $p = 0.73$). Ultimately, these findings demonstrate that human-centered deception hypotheses do not reliably transfer to AI attackers, highlighting the critical need for new research into AI-native active defense frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18334v2">Can LLMs Reason Like Automated Theorem Provers for Rust Verification? VCoT-Bench: Evaluating via Verification Chain of Thought</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly assist secure software development, their ability to meet the rigorous demands of Rust program verification remains unclear. Existing evaluations treat Rust verification as a black box, assessing models only by binary pass or fail outcomes for proof hints. This obscures whether models truly understand the logical deductions required for verifying nontrivial Rust code. To bridge this gap, we introduce VCoT-Lift, a framework that lifts low-level solver reasoning into high-level, human-readable verification steps. By exposing solver-level reasoning as an explicit Verification Chain-of-Thought, VCoT-Lift provides a concrete ground truth for fine-grained evaluation. Leveraging VCoT-Lift, we introduce VCoT-Bench, a comprehensive benchmark of 1,988 VCoT completion tasks for rigorously evaluating LLMs' understanding of the entire verification process. VCoT-Bench measures performance along three orthogonal dimensions: robustness to varying degrees of missing proofs, competence across different proof types, and sensitivity to the proof locations. Evaluation of ten state-of-the-art models reveals severe fragility, indicating that current LLMs fall well short of the reasoning capabilities exhibited by automated theorem provers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16650v1">PCA-guided Activation Scaling for Monotonic Bidirectional Control over LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 accepted by COLM2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit sycophancy, a tendency to agree with user beliefs regardless of factual accuracy. This can reinforce misconceptions, but eliminating it entirely risks over-correction against valid opinions. Effective control must therefore both reduce and increase sycophancy with predictable and gradual effect. Yet, existing methods fail to ensure a bidirectional and monotonic relationship between steering strength and behavioral outcome across models and datasets. We introduce PCA-guided Activation Scaling (PAS), an activation steering framework that decomposes residual stream activations into a PCA-identified sycophancy-honesty subspace and an orthogonal residual, then applies distinct scaling exponents to achieve monotonic, bidirectional control. Across three LLMs and three datasets, PAS achieves strong monotonicity (Spearman $ρ$ = +0.92) and an average shift of 15.4% per direction, compared with 8.7% for the baselines. Ablation studies confirm that the decomposition, asymmetric exponents, and layer selection are each essential for maintaining monotonic control. The data and code are available at https://github.com/Bellafc/PCS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16643v1">Toward Better Assessment of LLMs' Performance in Clinical Error Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted at Machine Learning for Healthcare (MLHC) 2026; to appear in Proceedings of Machine Learning Research (PMLR), Vol. 340
    </div>
    <details class="paper-abstract">
      Automated detection of errors in clinical documentation is a promising application of large language models (LLMs), yet decisions to deploy such models rest on benchmarks that evaluate each clinical note in isolation. Error-detection benchmarks are typically constructed by injecting errors into notes, such that each erroneous note has a natural counterpart. Aggregate discriminative metrics (e.g., balanced accuracy or F1) do not exploit this structure. We show that this omission is consequential. In particular, evaluating 15 diverse LLMs on 4 standardized clinical error-detection test sets across 3 languages, we find that 13 of 15 models fall below the level of random pairwise discrimination, even while achieving F1 scores that standard practice would read as moderate. We also observe that the underlying bias patterns differ across languages: the same model can default to "no error" on one language and over-flag errors on another. To diagnose where discrimination breaks down, we further introduce a procedure to score the evidence models cite in their outputs. We find that while models consistently locate error-relevant content, they fail to produce the corresponding correct verdict on the clean counterpart. Finally, we show that F1 and pairwise accuracy are driven in opposite directions by the same underlying bias, so that ranking models by F1 may systematically promote the weakest discriminators. For safety-critical clinical NLP applications, we advocate for supplementing aggregate metrics with paired evaluations in benchmark reporting. Code and analysis scripts are available at https://github.com/healthylaife/paired-clinical-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01375v3">Beyond Access: Guided LLM Scaffolding for Independent Learning in Undergraduate Statistics</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 10 pages. Accepted at the 34th International Conference on Computers in Education (ICCE 2026), Asia-Pacific Society for Computers in Education (APSCE)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly entering students' learning practices, but their educational value may depend on whether they are used to support reasoning or to complete tasks without engaging in the underlying reasoning. This study examines guided LLM use in an undergraduate Probability and Statistics course, focusing on the distinction between assigned LLM access and the quality of students' actual interaction with the model. In a four-week quasi-experimental summer program, students were organized into three balanced conditions: no LLM access, unrestricted LLM access, and guided LLM access. The guided condition used the same LLM platform as the unrestricted condition, but students received explicit training and rules intended to promote reasoning-focused help-seeking, stepwise hints, verification, and ethical use. All quizzes and the delayed final exam were completed without LLM or external assistance, allowing us to separate AI-supported practice performance from independent learning. Results show that guided use was associated with a clearer learning-oriented interaction pattern than unrestricted access, especially in prioritizing reasoning over final answers and requesting stepwise support. In behavior-defined analyses, Guided-LLM students showed a promising pattern of stronger no-help quiz performance, while practice scores showed no consistent Guided-LLM advantage. Available time measures did not support a simple duration-based explanation, and self-assessment calibration suggested better alignment between perceived and demonstrated understanding in Guided-LLM. These findings suggest that access alone may not reliably distinguish independent performance; instead, the quality of interaction and reasoning-focused scaffolds warrant further study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16637v1">PDDLCoder: Agentic PDDL Generation for LLM-Assisted Symbolic Planning</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      LLMs remain unreliable for long-horizon planning, often generating logically inconsistent or non-applicable plans. Recent hybrid methods instead translate natural language into the Planning Domain Definition Language (PDDL), allowing symbolic planners to produce verifiable plans. However, existing methods frequently rely on rigid generation pipelines, a partial PDDL definition, or human feedback. Furthermore, their evaluation is hindered by the lack of standardized benchmarks with automated verification. To address these limitations, we present PDDLCoder, an agentic framework for PDDL generation from natural language that iteratively generates, analyzes, and refines planning specifications. We further introduce NL-pddlgym, a benchmark dataset comprising 711 planning problems across 23 domains with executable gym environments for the automated verification of plan applicability. Experiments on the NL-pddlgym test set containing 106 problems across 4 held-out domains show that PDDLCoder generates applicable plans for 89.6\% of tested planning problems. This improves upon our adaptations of previous PDDL generation methods, which achieved up to 45.3\%, and outperforms direct LLM planning approaches, which reached up to 74.5\% on the same test set. Our work demonstrates the effectiveness of agentic PDDL generation for planning and establishes a reproducible benchmark for future research on LLM-assisted symbolic planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13341v4">Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 ICLR 2025; 27 pages, 8 figures
    </div>
    <details class="paper-abstract">
      High quality annotations are increasingly a bottleneck in the explosively growing machine learning ecosystem. Scalable evaluation methods that avoid costly annotation have therefore become an important research ambition. Many hope to use strong existing models in lieu of costly labels to provide cheap model evaluations. Unfortunately, this method of using models as judges introduces biases, such as self-preferencing, that can distort model comparisons. An emerging family of debiasing tools promises to fix these issues by using a few high quality labels to debias a large number of model judgments. In this paper, we study how far such debiasing methods, in principle, can go. Our main result shows that when the judge is no more accurate than the evaluated model, no debiasing method can decrease the required amount of ground truth labels by more than half. Our result speaks to the severe limitations of the LLM-as-a-judge paradigm at the evaluation frontier where the goal is to assess newly released models that are possibly better than the judge. Through an empirical evaluation, we demonstrate that the sample size savings achievable in practice are even more modest than what our theoretical limit suggests. Along the way, our work provides new observations about debiasing methods for model evaluation, and points out promising avenues for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10485v3">From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that graph neural network-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by similar data distributions and may contain synthetic samples, heuristic labels, or labeling noise. In this study, we systematically evaluate four representative DL models---Devign, ReVeal, LineVul, and VulBERTa---across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each dataset, and the graph-based and CodeBERT representations adopted by these models are analyzed using t-SNE and centroid distance to examine vulnerability-related patterns. To assess realistic applicability, we further evaluate trained ReVeal and LineVul models, along with four open-weight LLMs, on VentiVul, our newly constructed temporally separated out-of-distribution (OOD) dataset comprising 200 recent vulnerabilities from Linux and Chromium. Our experiments reveal that current representation methods struggle to distinguish vulnerable from non-vulnerable code and that trained models generalize poorly across datasets with differing distributions and characteristics. When evaluated on VentiVul, performance drops sharply, with most models failing to detect vulnerabilities reliably or distinguish vulnerable functions from their patched counterparts. These results expose a persistent gap between academic benchmarks and real-world deployment, emphasizing the value of our deployment-oriented evaluation framework and the need for more robust code representations, higher-quality datasets, and evaluation methods that account for vulnerability-fixing changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16508v1">LLMs for Zero-Shot Threat Detection via Structured Risk Indicators</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      We propose a two-stage large language model (LLM) framework for zero-shot detection of insider threats and advanced persistent threats (APTs) from heterogeneous security logs. The framework models user activity as chronological timelines and incorporates retrieval-augmented generation (RAG) to provide personalised behavioural context from each user's historical activity. Rather than performing end-to-end classification directly from raw logs, it first generates structured, interpretable sets of threat-specific risk indicators, which are then classified jointly across temporal sequences to capture attack patterns spanning multiple windows.The framework is evaluated on two benchmark datasets, CERT r5.2 for insider threat detection and PicoDomain for APT detection, using four combinations of two open-weight LLMs under both retrieval and non-retrieval settings. All configurations outperform the previous state-of-the-art LLM-based framework (GABM), with the best configuration improving the F1-score by 11.40 percentage points on CERT r5.2 and 31.50 percentage points on PicoDomain. Results further show that retrieval mainly benefits weaker LLMs by generating more discriminative risk indicators, whereas stronger models achieve comparable performance without retrieved context. The most effective assignment of LLMs to the two stages depends on the dataset. These findings show that the quality of the generated risk indicators is the main driver of zero-shot cyber threat detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16477v1">Pallas: A Proactive KV Cache Migration Framework for LLM Inference in AI-RAN</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      AI-RAN brings large language model (LLM) serving close to mobile users, but cellular handover can separate an active request from its inference state: the user attaches to a target base station (gNB) while the large and growing key-value (KV) cache remains at the source. Retaining inference at the source preserves service continuity but persistently increases inter-token latency (ITL), whereas recovering the state at the target restores serving locality but requires KV-cache transfer, recomputation, or a combination of both only after handover, directly prolonging service interruption time (SIT). This work presents Pallas, a \textit{proactive} KV-cache migration framework that prepares the inference state at the predicted target before handover, in parallel with ongoing source-side inference and token delivery. At the preparation trigger, Pallas partitions the token sequence into a stable historical prefix and an evolving suffix. The target reconstructs the prefix through local prefill, while the source streams the KV blocks generated for the suffix. At handover, the target assembles both portions into an up-to-date KV cache and resumes decoding locally, leaving only unfinished preparation to contribute to SIT. An online scheduler selects the \textit{prefetching window}, which determines how early preparation begins before handover, based on mobility predictions and runtime telemetry. Across three LLMs and $100$--$500~\mathrm{Mbps}$ inter-gNB links, our vLLM-based prototype reduces average SIT by factors of $2.28$--$89.68$ over target-side recovery approaches and lowers average ITL by $16.0\%$--$50.0\%$ compared with source-side forwarding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16467v1">Computational KJ-Ho: An Analyst-Bias-Free Insight Extraction Framework from Large-Scale Qualitative Data Using Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Concept paper. 38 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      The qualitative research methodologies that underpin consumer-insight generation - the KJ method, Grounded Theory, and Thematic Analysis - share a structural constraint: the cognitive processing capacity of the human analyst. Replication research further shows that conclusions vary substantially across analysts analyzing identical data (analyst bias). This paper proposes Computational KJ-Ho (the Kawakita Jiro method), a theoretical framework that computationally realizes the KJ method's epistemology - letting structure emerge from the data itself without imposing the analyst's preconceptions - an orientation we term "analyst-bias-free." The framework employs a domain-specialized LLM built through continued pre-training (CPT) on a marketing-research corpus and supervised fine-tuning (SFT) on expert-curated insight pairs, organized as a three-layer architecture: data structuring, insight extraction, and strategy generation. Two preliminary studies in the Japanese marketing context support the necessity of CPT-based domain specialization. The paper makes five contributions: (1) a theoretical integration of the KJ method, Grounded Theory, and Peircean abduction into a single epistemological commitment of data-driven explanation generation; (2) a three-layer architecture leveraging domain-specialized embeddings for cross-interview analysis; (3) two novel evaluation metrics, InsightExtraction-F1 and MarketingQA; (4) explicit engagement with the WEIRD problem, centering a non-Western methodology; and (5) five practice-derived problem formulations from nearly three decades of marketing-research practice, translated into design requirements. The human analyst retains a supervisory role. This is a concept paper presented ahead of empirical validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16461v1">A Human-LLM Teaming Framework for Privacy Risk Analysis: An Illustration with CBDC-Based Welfare Schemes</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Central Bank Digital Currency (CBDC)-based welfare schemes may be potentially privacy invasive as they process significant volumes of beneficiary personal data and lead to privacy harms such as surveillance, discrimination and stigmatization. Such welfare delivery schemes involve complex digital ecosystems and large number of stakeholders. Consequently, to examine their privacy risks, privacy risk assessments require extensive information gathering and synthesis, complex reasoning, scenario explorations, contextual evaluation and human judgement. Thus, they present ideal scenarios for human-LLM teaming, where effective integration of complementary human and LLM capabilities can yield an outcome far superior to either human-only or LLM-only assessments. In this paper, we propose a first human-LLM teaming framework for the systematic privacy risk analysis methodology called PRIAM. The framework specifies an iterative collaborative process in which the LLM processes large-scale documentary evidence to produce initial outputs, which are then interpreted and evaluated by human experts who direct their further refinement by the LLM and exercise their judgement to finalize the output. We illustrate the framework on the data characterization activity of PRIAM using a CBDC-based welfare scheme use case. The illustration demonstrates that while LLMs generate the initial data categories and assign initial values to data attributes, human experts evaluate and provide feedback to refine them, distinguishing documented evidence from inferences, identifying information gaps, and flagging unsupported or ambiguous outputs. This framework serves as a foundational contribution towards human-AI teaming for privacy risk assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16438v1">The Value of a Prompt: An LLM-Relative Kolmogorov-Complexity Approach</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      In a world where valuable artifacts are increasingly created, completed, or processed by LLMs, the central economic question is not only what the LLM can produce, but what \emph{value} remains in the inputs (i.e., the prompts) we provide to it. Given a prompt, hint, critique, problem statement, or partial solution that helps an LLM produce an artifact $z$---a proof, program, design, or scientific hypothesis---how should we measure the value of that input? Intuitively, an input is valuable when it makes the target artifact easier for the model to generate: either by increasing its sampling probability, or by reducing the thinking time needed to find it. We propose a computational Levin--Kolmogorov complexity approach to this problem, by appropriately replacing the universal Turing machine in the classical definitions by the LLM itself. Concretely, we introduce an LLM-relative notion of \emph{probabilistic Levin--Kolmogorov complexity} $pKt$---treating the model's thinking as the random tape of the program, and charging logarithmically for it in Levin's manner---and define prompt value as algorithmic mutual information with respect to $pKt$. This captures the intuition above: a prompt having $b$ bits of value for an artifact $z$ makes $z$ $2^b$ times ``easier to obtain'', by multiplying the success probability by $2^b$, by dividing the required computation by $2^b$, or by any corresponding tradeoff between probability and computation. In contrast to the classical notion of algorithmic mutual information, ours is efficiently estimable. We additionally show that, under a natural reproduction experiment, a prompt value of \(b\) bits means that reproducing \(z\) without the prompt has median token cost \(2^b\) times that of reproducing it with the prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16407v1">POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Point-of-interest (POI) recommendation models based on graph neural networks achieve strong performance by propagating collaborative signals over user-item interactions, yet they struggle with the cold-start problem, where items with few or no interactions are not represented. In this paper, we propose LLM-augmented Multi-Graph Contrastive Learning (LLM-MGCL), a multi-graph neural network that uses semantic and spatial information about items to extend the LightGCN backbone with two auxiliary item-item graphs: a semantic graph constructed from sentence embeddings of LLM-generated photo summaries and keywords, and a geographic graph derived from Haversine distances between business locations. Item embeddings are propagated over all three graphs in parallel, fused additively, and aligned across views through a bidirectional InfoNCE contrastive objective that connects behavioral, semantic, and spatial representations of the same items. Experiments on the Yelp Multimodal Recommendation Dataset show that LLM-MGCL outperforms classical collaborative filtering, matrix factorization, and interaction-only graph neural network baselines. It improves Recall@20 by 52.0% and NDCG@20 by 64.8% over LightGCN while performing on par with the strongest contrastive baseline, Self-supervised Graph Learning (SGL), which is also affected by the cold-start problem. An ablation study reveals that the cross-view contrastive alignment (CA) is the primary driver of these gains, with the best performance achieved when all three graphs are combined. Our results suggest that externally grounded, LLM-derived item knowledge can effectively compensate for missing collaborative signal and mitigate the item cold-start problem in POI recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16394v1">Think Inside the Chunk: RegulaRAG for Regulation-Compliant Scenario Generation using LLMs: A Case Study of UN Regulation No. 152</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Generating regulation-compliant test scenarios is essential for validating safety-critical automotive systems, yet Large Language Models (LLMs) struggle to ground outputs in long, hierarchical standards. We present RegulaRAG, a Retrieval-Augmented Generation (RAG) pipeline that couples SmartChunking, reference-aware enrichment of paragraphs and tables via graph traversal, with Smart Retrieve & Rerank over these enriched units. To test our system, we evaluate on a manually curated dataset covering all scenarios in UN Regulation No. 152 (AEBS). Our study comprises: (i) a three-step progressive search that identifies near-optimal retrieval parameters without exhaustive grid search; (ii) head-to-head comparisons against five baseline RAG systems; and (iii) a robustness stress test that scales the source corpus with distractor content. Outputs are evaluated using a customized penalized scoring metric. Across all experiments, RegulaRAG achieves the highest average Meta-Score (82.99), outperforming the next-best system by 43% (NoRAG: 57.94), while operating at 14k-25k tokens per query versus up to 500k for graphcentric baselines. It maintains strong performance, remaining stable even as the number of regulatory sources grows, whereas competing RAG systems degrade sharply in both quality and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16349v1">AeroCopilotBench: A Two-Tier Benchmark for Evaluating LLM Agents as Aviation Copilots in an Interactive Virtual Cockpit Environment</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 38 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents may assist flight crews with complex decisions and task execution, but existing aviation evaluations centered on static knowledge do not support systematic testing of procedural execution and safety compliance in interactive environments. This paper presents the AeroCopilot Operational Environment (ACOE), a reproducible interactive virtual-cockpit test environment, and AeroCopilotBench, a two-tier aviation agent evaluation benchmark. Tier-1 evaluates aviation knowledge using 1,200 multiple-choice questions, while Tier-2 comprises 73 emergency and abnormal tasks derived from the manufacturers' Pilot's Operating Handbooks (POHs) and instantiated in ACOE. ACOE converts natural-language procedures into executable state transitions, final-state goal conditions, and hard safety constraints, enabling models to interpret cockpit state, diagnose faults, and operate aircraft systems through standardized tool interfaces. We establish a safety-gated evaluation framework in which a trajectory succeeds only when all task goals are achieved without violating any hard safety constraint, while safe goal progress and trajectory safety are measured separately. Across 12 models, the highest Tier-2 success rate is 72.6%, while static knowledge performance does not consistently translate into procedural execution. Analysis of 451 failed episodes from 3 representative models identifies recurring failures in procedural completeness, use of state feedback, and long-horizon execution management. These findings motivate state-aware agent orchestration, joint assessment of task completion and trajectory safety, and repeated regression testing. ACOE and AeroCopilotBench provide a reproducible foundation for testing knowledge application, interactive execution, and operational safety in aviation agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16286v1">Clause Encounters of the Third Kind: Can LLMs Replace Language Teachers?</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      While various organizations now actively encourage LLM use in classrooms, we still lack rigorous, systematic evaluations of how well these models actually perform the fundamental tasks of language pedagogy. This paper examines whether state-of-the-art LLMs can deliver the kind of corrective feedback and methodological explanations that language learners need. The study tests multiple large language models on their ability to identify, correct, and explain common learner mistakes in English, by systematically varying model parameters to investigate how these technical adjustments affect output quality, pedagogical clarity, and consistency, along with using retrieval-augmented generation to query methodological data. The evaluation employs automated metrics (GLEU, BERTScore) but also human expert judgments to capture dimensions that purely computational measures miss: linguistic nuance, cultural sensitivity, and instructional appropriateness. While models demonstrate impressive surface-level correction abilities, their explanations often lack the terminological and domain knowledge that effective language teaching requires, suggesting that current enthusiasm for AI-assisted language learning may be outpacing our understanding of these systems' actual pedagogical competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16253v1">Second-Order Response Laws for LLM Judges: Debiased Estimation of Prompt Instability</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      LLM judges are often evaluated with a single prompt and only a few repeated calls. When their verdicts vary, it remains unclear whether the variation comes from sampling noise within a prompt or systematic differences across prompts. We formalize this distinction using a second-order response law: the distribution of prompt-conditioned verdict distributions induced by a declared prompt policy. For a quadratic measure of prompt instability, we show that the usual plug-in estimator is biased upward at finite repeat budgets because it confounds within-prompt noise with between-prompt variation. We derive unbiased estimators for both sampled prompts and declared fixed prompt censuses from the difference between within- and across-prompt agreement. Under a crossed prompt-by-answer-order design, the same framework separates prompt, order, interaction, and residual call variation, while retaining invalid completed outputs as outcomes. Known-law simulations and a byte-identical live null recover the predicted finite-$R$ inflation. In a matched Qwen study, corrected low-repeat estimates are closer to an independently acquired $R=16$ reference than plug-in estimates, with the largest gains at small repeat budgets. A matched panel across four frozen judge configurations exhibits configuration-specific inflation magnitudes and component profiles. Prompt robustness can therefore be estimated separately from finite-call noise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23049v2">MedMCP-Calc: Benchmarking LLMs for Realistic Medical Calculator Scenarios via MCP Integration</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted to the ACL 2026 Main Conference as an Oral Presentation
    </div>
    <details class="paper-abstract">
      Medical calculators are fundamental to quantitative, evidence-based clinical practice. However, their real-world use is an adaptive, multi-stage process, requiring proactive EHR data acquisition, scenario-dependent calculator selection, and multi-step computation, whereas current benchmarks focus only on static single-step calculations with explicit instructions. To address these limitations, we introduce MedMCP-Calc, the first benchmark for evaluating LLMs in realistic medical calculator scenarios through Model Context Protocol (MCP) integration. MedMCP-Calc comprises 118 scenario tasks across 4 clinical domains, featuring fuzzy task descriptions mimicking natural queries, structured EHR database interaction, external reference retrieval, and process-level evaluation. Our evaluation of 23 leading models reveals critical limitations: even top performers like Claude Opus 4.5 exhibit substantial gaps, including difficulty selecting appropriate calculators for end-to-end workflows given fuzzy queries, poor performance in iterative SQL-based database interactions, and marked reluctance to leverage external tools for numerical computation. Performance also varies considerably across clinical domains. Building on these findings, we develop CalcMate, a fine-tuned model incorporating scenario planning and tool augmentation, achieving state-of-the-art performance among open-source models. Benchmark and Codes are available in https://github.com/SPIRAL-MED/MedMCP-Calc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16207v1">Competing at Every Price Point with Agentic Evolution over a Menu of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Consider a firm that surveys its competition for a particular agentic task and seeks to offer superior accuracy at every competitor price point. A firm that Pareto-dominated its competitors would leave no rational customer a reason to buy elsewhere. This paper shows a path to this kind of capability via agentic evolution over a menu of LLMs, from training pools of at most 100 examples. Given a priced menu of nine LLM endpoints; brief documentation of the task, objective, and API; a simple seed agent; and an operator-chosen per-problem cost target - usually set at an incumbent's own price - RoboPhD, an evolutionary meta-agent, evolves complete agent programs that attack the public frontiers of two semantically dissimilar tasks point by point: DS-1000 (execution-checked code generation) and PaperFindingBench (LLM-judged scientific document retrieval). Our officially scored submissions hold every Pareto-frontier slot but one on the two tasks' leaderboards, including Pareto domination of both the top-scoring and the lowest-cost competing points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16201v1">Multi-Granularity Sentiment Integration for LLM-Based Multimodal Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted to NLPCC 2026
    </div>
    <details class="paper-abstract">
      Multimodal sentiment analysis (MSA) aims to predict sentiment polarity and intensity from heterogeneous inputs such as text, audio, and vision. While large language models (LLMs) offer strong semantic priors for MSA, effectively incorporating audio and visual signals effectively remains challenging. A key challenge is that audio and visual sentiment cues evolve over different temporal scales, yet many LLM-based methods compress these signals through shallow projection or coarse pooling before fusing them with text, which can weaken cross-modal alignment and erase fine-grained affective information. We propose MGSI, a multi-granularity sentiment integration framework for LLM-based MSA. MGSI first encodes audio and visual streams at short-, medium-, and long-range temporal scales, preserving both local variations and global affective trends. It then refines non-text features through text-guided alignment, and applies polarity- and intensity-aware enhancement to better handle ambiguous and near-neutral samples. The resulting multimodal representation is finally compressed into a small set of pseudo-tokens for efficient conditioning of a frozen LLM. Experiments on four public benchmarks show that MGSI substantially outperforms frozen-LLM baselines and remains competitive with strong multimodal methods. Further ablation and sensitivity analyses support the effectiveness of multi-granularity temporal modeling, text-guided refinement, and adaptive sentiment calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18741v2">ReMP: Low-Downtime Runtime Model-Parallelism Reconfiguration for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Current large language model (LLM) inference systems universally deploy ultra-large-scale models using a combination of Tensor Parallelism (TP) and Pipeline Parallelism (PP). However, existing systems treat the model parallelism topology as a static configuration that cannot be flexibly adjusted at runtime. This rigid design creates a fundamental contradiction with the dynamically changing inference workloads in real-world scenarios. State-of-the-art systems lack online reconfiguration capabilities and can only switch configurations by restarting the service, resulting in several minutes of service interruption, KV cache loss, and prohibitive recomputation overhead. To address this problem, this paper presents ReMP, a runtime model parallelism reconfiguration framework that supports low downtime. ReMP achieves dynamic adjustment through three key techniques: (1) decoupling the model parallelism topology from runtime state to avoid full service reconstruction; (2) designing a two-dimensional KV cache migration mechanism to preserve reusable cache states after TP/PP changes; and (3) implementing end-to-end online reconfiguration. Experiments demonstrate that ReMP can complete most topology switches within 1-7 seconds on models ranging from 7B to 70B parameters, achieving speedups of tens to over a hundred times compared to the restart approach. Moreover, ReMP significantly outperforms fixed configurations under dynamic workloads, delivering superior performance in terms of TTFT, TPOT, and output throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16168v1">QUMem: Personalized Memory for Query-Conditioned User-State Inference in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 9pages,3figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly use external memory systems to support personalization by drawing on long and evolving interaction histories, in which user preferences may be distributed across time, change with context, and conflict with earlier evidence. However, existing systems face three limitations: fixed-turn, fixed-token, or session-based boundaries can mix unrelated dialogue or split an event from its causes, decisions, and outcomes; storing multiple pieces of user information from the same interaction as a single memory binds together items that serve different functions and should be independently retrievable; and treating the current task as a single top-$k$ retrieval query can return fragments that are individually relevant but fail to jointly capture preference evolution, temporal validity, and contextual applicability. We introduce \textsc{QUMem}, a structured memory framework for query-conditioned user-state inference. \textsc{QUMem} first segments interaction histories into variable-length episodes according to semantic continuity, then decomposes each episode into independently retrievable factual, preference, and transferable insight memories while preserving temporal positions and source evidence. At inference time, three sequential agents identify task-specific information needs, plan multi-query retrieval over the typed memory stores, and jointly infer a temporally and contextually valid user state for downstream response generation. \textsc{QUMem} achieves state-of-the-art performance on both PersonaMem and KnowU-Bench, demonstrating the effectiveness of query-conditioned user-state inference for long-term personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16156v1">TRCA: Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Long-horizon large language model (LLM) agents are typically optimized with sparse terminal outcomes, making fine-grained credit assignment across multi-step interactions difficult. Existing approaches either rely on process evaluators, which incur annotation and inference costs, or derive step-level credit from successful trajectories. However, successful trajectories are extremely scarce during early-stage reinforcement learning, substantially weakening anchor-based methods. We propose Transition-wise Rubric Credit Assignment (TRCA), which derives step-level supervision directly from action-induced transitions without learned evaluators or successful anchors. TRCA evaluates each transition using Evidence, Execution, and Invalidity rubrics to capture task-relevant information acquisition, valid task execution, and invalid or regressive behavior. From these judgments, Foundational Rubric Reward measures local transition quality, while Breakthrough Rubric Reward tracks newly covered Evidence and Execution conditions to reward incremental task progress. Combined with terminal outcomes, these signals produce fine-grained step-level advantages for policy optimization. Experiments on ALFWorld, WebShop, and seven search-augmented question-answering benchmarks show consistent improvements over the evaluated baselines. With Qwen2.5-7B-Instruct, TRCA improves the WebShop score by 6.0%-12.6%; with Qwen2.5-3B-Instruct, it improves the average SearchQA score by 1.9%-18.3%. These results demonstrate the effectiveness of transition-wise rubric credit assignment for long-horizon tasks with sparse successful anchors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02637v2">Train Yourself as an LLM: Exploring Effects of AI Literacy on Persuasion via Role-playing LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly persuasive, there is concern that people's opinions and decisions may be influenced across various contexts at scale. Prior mitigation (e.g., AI detectors and disclaimers) largely treats people as passive recipients of AI-generated information. To provide a more proactive intervention against persuasive AI, we introduce $\textbf{LLMimic}$, a role-play-based, interactive, gamified AI literacy tutorial, where participants assume the role of an LLM and progress through three key stages of the training pipeline (pretraining, SFT, and RLHF). We conducted a $2 \times 3$ between-subjects study ($N = 274$) where participants either (1) watched an AI history video (control) or (2) interacted with LLMimic (treatment), and then engaged in one of three realistic AI persuasion scenarios: (a) charity donation persuasion, (b) malicious money solicitation, or (c) hotel recommendation. Our results show that LLMimic significantly improved participants' AI literacy ($p < .001$), reduced persuasion success across scenarios ($p < .05$), and enhanced truthfulness and social responsibility levels ($p<0.01$) in the hotel scenario. These findings suggest that LLMimic offers a scalable, human-centered approach to improving AI literacy and supporting more informed interactions with persuasive AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16131v1">Mitigating AI Risks in Computing Education via LLM-Driven Lecture Video Curation</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 7 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of utilising large language models (LLMs) to retrieve targeted segments from delivered video recordings to answer student questions in introductory programming environments. By restricting AI to identifying existing, educator-verified media rather than generating open-ended text, this approach aims to mitigate common pedagogical risks such as generative hallucinations and cognitive bypassing. We benchmarked three distinct models, two proprietary (Gemini 3.1 Pro and GPT 5.4 Pro) and one open-weight (Qwen3.5 397B), against a human lecturer's manual video selections. An automated judging framework subsequently assessed the outputs for relevance, sufficiency, redundancy, and the presence of extraneous material. While the AI-retrieved timestamps rarely shared exact overlaps with the human baseline, the proprietary models achieved near-parity with the expert in delivering sufficient and highly relevant answers. Furthermore, a pilot deployment of this retrieval system in a large C programming cohort (n~=900) demonstrated strong user engagement, with students primarily utilising the tool to review foundational concepts. By leveraging AI to retrieve established lecture material, this approach shows potential for a reliable, high-fidelity pathway for safely integrating LLMs into novice computing courses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12756v2">Closing the Loop: A Control-Theoretic Framework for Provably Stable Time Series Forecasting with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted by ACM MM26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown exceptional potential in time series forecasting (TSF), leveraging their inherent sequential reasoning capabilities to model complex temporal dynamics. Existing approaches typically employ an autoregressive generation strategy to adapt LLMs for TSF. However, we identify a theoretical flaw in this paradigm: during inference, the model operates in an open-loop manner, recursively consuming its own generated outputs. This leads to error accumulation, where minor early deviations cascade into significant rollout drift over long horizons. In this paper, we reformulate autoregressive forecasting through the lens of control theory, proposing Feedback-driven LLM (F-LLM), a novel closed-loop framework. Unlike standard methods that passively propagate errors, F-LLM actively stabilizes the trajectory via a learnable residual estimator functioning as a system observer. Furthermore, we provide a mathematical proof that, under explicit contraction assumptions, this closed-loop mechanism guarantees a uniformly bounded step-wise error sequence within the local surrogate dynamics. Extensive experiments demonstrate that F-LLM significantly mitigates error propagation, achieving good performance on time series benchmarks. Our code is publicly available at https://github.com/Zh-XY22/F-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06008v3">PolyWorkBench: Benchmarking LLM Agents for Cross-Lingual Long-Horizon Workflows</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 17 Pages, 5 figures
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents excel at monolingual long-horizon planning and tool use, enterprise workflows inherently require processing multilingual resources across extended trajectories. The interaction between multilinguality and long-horizon execution, however, remains underexplored. We introduce PolyWorkBench, a benchmark designed to evaluate LLM agents on multilingual, long-horizon workplace workflows. PolyWorkBench features 67 tasks across five core domains: commerce, knowledge work, legal analysis, localization, and manufacturing. Tasks are authored by the paper's authors from real-world data seeds and independently verified through a second-author audit. Agents must integrate heterogeneous multilingual inputs, execute iterative tool-use trajectories, and produce structured domain artifacts. To rigorously assess performance, we adopt Grade, a task-specific structural scoring rubric, as our primary ranking metric, and complement it with Pytest for executable state verification and LLM-as-Judge for semantic quality diagnostics. Benchmark evaluations reveal that agent performance varies substantially across languages and drops sharply on the harder cross-lingual tasks, and our analysis shows that multilingual execution exposes systematic failure modes across planning, tool interaction, and decision-making in long-horizon agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16114v1">HyperSkill: Self-Evolving LLM Agents via Hypergraph-Structured Skill Memory</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      As agentic tasks grow in complexity, LLM agents increasingly rely on experiential memory to reuse procedural knowledge across tasks. Effective memory design must jointly address what to store, how memory is structured and retrieved, and how memory evolves. Existing systems tackle each only partially: they store trajectories, insights, or workflows as isolated entries, discarding compositional relationships among subtasks and reusable skills; retrieve by flat embedding similarity that ignores relational signals; and maintain memory without leveraging its relational structure. We propose HyperSkill, a hypergraph-based memory framework that jointly improves all three. HyperSkill represents memory as a hypergraph with two node types, subtask steps and reusable skills, where each hyperedge links the subtasks and skills from a single trajectory. Dual-path retrieval queries both subtask and trajectory levels, ranking skills by co-occurrence across retrieved trajectories. Periodic structure-informed maintenance prunes low-utility nodes and merges redundant skills via quality-weighted propagation. Across xBench, GAIA, and WebWalkerQA with GPT-4o and Qwen3-30B-A3B, HyperSkill outperforms ten memory baselines, yielding gains of up to +11.51 on GAIA and +11.18 on WebWalkerQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.09546v2">A Reflective LLM-based Agent to Guide Zero-shot Cryptocurrency Trading</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Published at EMNLP 2024 (Main Conference)
    </div>
    <details class="paper-abstract">
      The utilization of Large Language Models (LLMs) in financial trading has primarily been concentrated within the stock market, aiding in economic and financial decisions. Yet, the unique opportunities presented by the cryptocurrency market, noted for its on-chain data's transparency and the critical influence of off-chain signals like news, remain largely untapped by LLMs. This work aims to bridge the gap by developing an LLM-based trading agent, CryptoTrade, which uniquely combines the analysis of on-chain and off-chain data. This approach leverages the transparency and immutability of on-chain data, as well as the timeliness and influence of off-chain signals, providing a comprehensive overview of the cryptocurrency market. CryptoTrade incorporates a reflective mechanism specifically engineered to refine its daily trading decisions by analyzing the outcomes of prior trading decisions. This research makes two significant contributions. Firstly, it broadens the applicability of LLMs to the domain of cryptocurrency trading. Secondly, it establishes a benchmark for cryptocurrency trading strategies. Through extensive experiments, CryptoTrade has demonstrated superior performance in maximizing returns compared to traditional trading strategies and time-series baselines across various cryptocurrencies and market conditions. Our code and data are available at https://anonymous.4open.science/r/CryptoTrade-Public-92FC/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16075v1">TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted to CIKM 2026 full research paper
    </div>
    <details class="paper-abstract">
      Continual recommendation aims to capture evolving user interests from streaming data but struggles with sparsity. LLM enhancers mitigate this with semantic knowledge, but naive integration creates a new conflict. We identify this as the Stability-Plasticity-Cognitivity (SPC) Trilemma, where generalized LLM semantic priors (Cognitivity) conflict with retaining personalized historical preferences (Stability) and adapting to individual interest shifts (Plasticity). To address this, we propose Trilemma-Responsive Adaptive Continual Enhancement for Recommendation (TRACER). TRACER synergistically combines three specialized modules, each targeting stability, plasticity, or cognitivity, while preventing any single lemma from dominating. This holistic design enables semantic knowledge to support history retention and adaptation to evolving interests without disrupting continual learning. Across five real-world datasets, TRACER effectively harmonizes the SPC trilemma and outperforms state-of-the-art baselines by up to 14.38%. Our code is available at https://github.com/woo-joo/TRACER_CIKM26.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16068v1">CAPO: Constraint-Aware Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as agents that rely on system prompts to use tools and complete tasks. Such deployments impose distinct operational requirements, including appropriate tool use, concise prompts and solution paths, and compliance with safety and formatting policies. For many practitioners, however, assembling domain-specific supervised data to post-train models to meet these requirements is infeasible. We introduce CAPO (Constraint-Aware Prompt Optimization), a primal-dual method that combines pool-based rewrites with adaptive constraint weighting to optimize system prompts under explicit operational constraints. Across agentic benchmarks, CAPO more reliably reaches empirically feasible operating points while improving task performance. CAPO also generalizes beyond agentic settings, achieving strong results on assistant-style evaluations with output-format and safety/privacy constraints. We further introduce DCAPO (Dynamically Trained CAPO), which trains a feedback- and dual-conditioned rewriter with pool-based GRPO while keeping the task agent frozen. Across task agents of different sizes, DCAPO produces a feasible prompt in every evaluated domain and matches or improves the task accuracy achieved by the evaluated baselines. A surrogate analysis characterizes how finite-pool and discrete-rewrite errors enter the inexact primal-dual procedure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09041v2">BiAxisBias: Evaluating LLM Bias Beyond a Single Prompt and a Single Explanation</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 19 pages, 6 figures. Preprint
    </div>
    <details class="paper-abstract">
      LLM bias scores can depend on audit design. We introduce BiAxisBias, a prespecified audit varying task, role, perspective, sentiment, and wording over 200 stereotype statements while retaining forced Selection and Rationale as separate protocol readouts. Its main matrix spans eight LLMs and 401 templates (641,600 responses). Across five equivalent questions, 17.1% of 1,600 model-statement pairs change Selection. With three observations per unit in both arms, instability averages 10.5% across all ten three-wording subsets, versus 6.3% for three identical calls. Across four controlled task paradigms, 9/28 model pairs reverse; a seven-model factorial sensitivity identifies task-by-sentiment as the largest two-way component (raw eta-squared = 0.0465). In 10,000 equal-budget resampling draws, mean absolute error against a declared 18-condition finite reference is 10.63 points for one-template concentration, 1.86 for matrix-wide simple random sampling, and 1.75 for condition stratification; ranking inversions are 20.6%, 4.8%, and 5.5%. Thus broad coverage drives the gain, while stratification has only a modest score-error advantage and no ranking advantage over random sampling. In a separate forced-output diagnostic, task-specific Selection mappings and judge-coded Rationale stance disagree in 34.2% of 7,959 dual-valid responses (31.2% versus 3.0% by direction). This diagnoses output-contract sensitivity, not two validated measures of one construct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16032v1">Proof-of-Execution Memory: Defending LLM Agents Against Forged-Reasoning Attacks by Verifying What Actually Happened</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 8 pages, 6 figures, 5 tables. Code: https://github.com/bithabib/ai_security
    </div>
    <details class="paper-abstract">
      LLM agents are stateless and rely on external memory to carry context between steps. Because agents treat that memory as trustworthy, an adversary who can write to it can steer their behavior. The FARMA attack does this with no malicious command: it inserts fabricated entries into the agent's reasoning memory claiming a required safety step is already done, so the agent skips it. SENTINEL, the defense proposed with FARMA, scores entries against a fixed list of suspicious wordings; its authors note that an attacker who knows the list can reword the forgery and evade it, and leave this open. We show the gap is worse than stated. An automated attacker that simply asks a language model to reword the forgery evades SENTINEL on its first try, reducing its protection to zero on every model tested. We also find a capability paradox: the attack succeeds far more often on stronger models (98-100% on GPT-4o and GPT-4o-mini) than on Llama-3.1-8B (44%), because more capable agents follow reworded claims more faithfully, so the threat grows with capability. We propose Proof-of-Execution Memory (PoEM), which does not inspect memory at all. PoEM keeps a separate, tamper-evident, HMAC-chained ledger of the safety steps that actually executed, writable only by the trusted action layer, and allows a skip only if the ledger confirms real execution. An attacker can change what memory says but cannot forge a ledger entry for a step that never ran, so rewording no longer helps. Across three models and three scenarios, PoEM drives attack success to 0% while leaving legitimate operation intact (0% false positives in eight of nine cells, 1.7% in the ninth, within sampling noise), whereas SENTINEL wrongly blocks 33-50% of legitimate operations. PoEM also withstands attacks aimed at itself, adds microseconds of overhead, and works unchanged in a real LangChain agent. PoEM protects exactly the decisions it gates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16030v1">Benchmarking Identity-Sensitive LLM Outputs for Surveillance and Security Robots</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 Accepted to the Foundation Models in the Ro-Man Age (FoRMA) Workshop at IEEE RO-MAN 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate textual robot design specifications, interaction policies, and risk assessments during early-stage robot development. Such outputs may influence how surveillance and security robots are conceptualized, documented, and ultimately implemented. This paper evaluates whether identity-conditioned prompts produce systematic differences in LLM-generated surveillance and security robot design descriptions. Using 236 demographic identity labels across single-label and model-augmented prompt conditions, we analyze readability as an initial benchmark for evaluating accessibility and identity-conditioned variation in generated robot design descriptions. The results show significant differences in readability across prompt conditions, design dimensions, and demographic identities. Although readability cannot determine whether an output is fair or socially appropriate, it provides an interpretable baseline within a broader benchmarking framework that also includes lexical, semantic, sentiment, syntactic, and fairness-focused analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16022v1">OpenHarmony Bench: Evaluating LLMs and Coding Agents on OpenHarmony App Development</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      We present OPENHARMONY BENCH, an app-level coding benchmark for evaluating LLM-based coding agents on OpenHarmony ArkTS applications. Unlike function-level benchmarks, it evaluates complete app-level changes: each task requires an agent to modify a buildable ArkTS project so that a requested behavior works end to end, involving UI state, data persistence, build configuration, and platform APIs. The benchmark installs and drives the delivered application on a device to check whether the behavior is observable. It covers three input sources: natural-language feature requests (new-feature), structured scenario specifications (spec-driven), and bug descriptions (bug-fix). The benchmark contains 153 top-level tasks and 242 Feature points (F-points), where an F-point is one executable behavior check. The snapshot includes 32 new-feature tasks, 50 spec-driven tasks with 139 F-points, and 71 bug-fix tasks. The main leaderboard is scored over top-level tasks rather than independently weighted F-points. We describe the benchmark construction, statistics, and build-and-test evaluation pipeline, and evaluate DevEco Code with eight LLMs across three independent full-suite runs per configuration. Three findings emerge. First, newer generations complete more tasks than their predecessors within evaluated model-family pairs. Second, buildability is close to saturated while behavioral correctness is not: mean Final Build Success Rate is 94.77% to 100.00%, whereas mean Task Completion is 48.36% to 58.39%. Third, spec-driven tasks have the lowest Task Completion under all-checks task scoring, with no configuration exceeding 35%. The code, data, tasks, reference solutions, tests, evaluation scripts, and leaderboard are released through the official OPENHARMONY BENCH website at https://bench.matrix.openharmony.cn/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.14065v2">Rethinking Automated Program Repair: The Impact of Bug Complexity, Fault Localization, and LLM Cost-efficiency</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 20 pages, 6 figures, 10 tables. Accepted at ESEM 2026
    </div>
    <details class="paper-abstract">
      Background: Software bugs remain a critical challenge in development, necessitating effective Automated Program Repair (APR) techniques. While Large Language Model (LLM)-based APR systems have shown promise, prior studies primarily focus on overall repair effectiveness. The effects of bug complexity, fault localization, reasoning settings, and repair cost-effectiveness remain insufficiently explored. Aims: This study presents a comprehensive empirical analysis of LLM-based APR, focusing on how repair performance is shaped by bug complexity, fault localization, reasoning settings, and costs. Method: We evaluate two APR techniques (ChatRepair and CodeCorrector) using three LLMs (DeepSeek, GPT, and Llama), and examine their performance across diverse levels of bug complexity and localization strategies through a multi-dimensional empirical framework and statistical analysis. Results: Although structurally complex bugs and imprecise fault localization make repair more challenging, LLM-based APR techniques still achieve competitive repair effectiveness. Imprecise fault localization can substantially enlarge the performance gap between APR techniques. Furthermore, higher-cost LLMs and stronger reasoning settings do not consistently yield better cost-efficiency, revealing a nontrivial trade-off between repair effectiveness and computational cost. Conclusions: Over 50% of moderately complex bugs can be repaired by low-cost LLM-based APR techniques. The repair effectiveness gap between APR techniques becomes larger as fault localization becomes less precise. GPT-5 repairs 7 and 39 more complex bugs than DeepSeek-V4-pro and DeepSeek-V3.2, respectively; whereas the total repair cost of DeepSeek-V3.2 shows the best cost-efficiency performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.10139v3">LLMs as a Jury: Cross-Model Consensus Can Outperform Process Reward Models for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      Selecting the correct answer from a pool of candidate reasoning chains is the engine of test-time scaling, yet the standard selectors each carry a cost: self-consistency inherits the errors of the single model it resamples, and trained reward models need labeled data and transfer poorly off-distribution. We study a third signal, free at inference time: cross-model consensus, the degree to which independently trained models, each solving the problem once, agree on a final answer. We treat the panel as an LLM-jury, in which the verification signal is the structure of agreement itself, with no model scoring another's work. Across seven benchmarks it selects correct answers better than self-consistency and far better than a model scoring its own candidates: on competition math it closes the entire gap to an oracle selector, while self-scoring closes almost none. The mechanism is error decorrelation: independently trained models err differently, so their wrong answers scatter while the correct one accumulates agreement. We make this precise with a parameter-free law, derived in closed form, that predicts consensus accuracy from three measured panel statistics to a mean absolute error of $0.03$ and exposes the method's ceiling: a shared-error floor where models share a misconception, near zero on math but non-trivial on science. Against four trained verifiers spanning discriminative, outcome, and generative reward models, the free LLM-jury matches the strongest inside their math training domain and is the top selector outside it. Cross-model consensus is thus a verifier we can characterize in advance: a law that says when to trust it, and a floor that marks where it cannot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15975v1">A Scalable Pipeline for LLM-Teacher Distillation Labeling: Work-Stealing Job Scheduling and Memory-Aware GPU Concurrency</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 8 pages, 1 figure, 3 tables. Code, tests, and all run artifacts: https://github.com/rsdpyenugula/hybrid-labeling-training
    </div>
    <details class="paper-abstract">
      Labeling large text corpora with LLM teachers has become a practical route to training data at scale. At millions of items, hand-labeling every batch is not feasible, and two questions dominate: what label quality a teacher buys per dollar, and how to keep a fleet of GPU workers busy under skewed, failure-prone workloads. We present a simple, reproducible pipeline that addresses both. First, a work-stealing ring pool: each worker owns a queue, drains it first, and then steals from ring successors, with exactly-once task claims via atomic conditional writes and crash tolerance via stale-claim sweeping. The claim protocol requires only a compare-and-set primitive from its storage layer; we implement it on a single SQLite file, which makes the reference implementation dependency-free and the experiments reproducible on one machine. Second, a memory-aware concurrency rule that sizes per-node parallelism by how many model copies fit on the GPU, so the same code runs safely across device sizes. Third, a relabeling benchmark methodology in which the teacher relabels a public dataset that already has gold labels, so quality reduces to an agreement measurement and cost follows from measured throughput. Under skewed load the pool sustains up to 3.4 times the throughput of static sharding while matching it at zero skew, loses 0 of 2,000 tasks when half the workers are killed mid-run (static sharding loses 953), and yields measured quality and cost points for an instruction-tuned teacher on irony and sentiment tasks. All experiments run on public data and commodity hardware; code, tests, and run logs are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.04504v2">Multi-Bin Batching for Increasing LLM Inference Throughput</a></div>
    <div class="paper-meta">
      📅 2026-08-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in popularity for their diverse capabilities, improving the efficiency of their inference systems has become increasingly critical. Batching LLM requests is a critical step in scheduling the inference jobs on servers (e.g. GPUs), enabling the system to maximize throughput by allowing multiple requests to be processed in parallel. However, requests often have varying generation lengths, causing resource underutilization, as hardware must wait for the longest-running request in the batch to complete before moving to the next batch. We formalize this problem from a queueing-theoretic perspective, and aim to design a control policy which is throughput-optimal under a static-batching framework. We propose Multi-Bin Batching, a simple yet effective method that can provably improve LLM inference throughput under this framework by grouping requests with similar (predicted) execution times into predetermined bins. Through a combination of theoretical analysis and experiments, including real-world LLM inference scenarios with static and continuous-batching baselines, we demonstrate that multi-bin batching substantially improves throughput over static batching and quantify the remaining gap to native continuous batching under both oracle and estimated length information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15964v1">LLMs Get Smarter from Targeted Synthetic Multilingual Data</a></div>
    <div class="paper-meta">
      📅 2026-08-16
    </div>
    <details class="paper-abstract">
      Language-specific competency (LSC) is the phenomenon of a language model performing better or worse depending on the language of the prompt. In other words, a language model outputs different (and potentially incorrect) responses to the same semantic query when prompted in different languages. Prior work attributes this to an internal misalignment of semantic representation across languages. Currently, there are two main approaches to address LSC in the literature: (1) routing all queries through English, improving performance, but limiting language expressivity to English; or (2) training on language-balanced data, equalizing model performance across languages, but reducing overall performance. In this work, we take a data centric perspective and introduce HOTFIXR: Hardness Optimized Training data For Improving X-Lingual Reasoning. It is a data generation framework that uses models to probe and learn a student model's multilingual weaknesses, and generates data to mitigate them. HOTFIXR can generate multilingual synthetic training data that can improve multilingual performance. We evaluate on three in-distribution tasks, three out-of-distribution tasks, and four out-of-distribution languages. On average, HOTFIXR (1) improves in-distribution performance by 6.2%, (2) reduces catastrophic forgetting (induced by fine-tuning) on OOD tasks by 3.7%, and (3) on OOD languages by 7.1%. Overall, as many real-world applications requires multilingual LLMs, our work contributes to the efforts of making LLMs multilingually proficient. We will release code upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18490v2">Single-Round Vector RAG vs an LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research Corpus</a></div>
    <div class="paper-meta">
      📅 2026-08-16
      | 💬 v2: two-judge reanalysis of the decomposition-RAG ablation (groundedness advantage +1.15 to +0.15); H3a adjudicated; one registered-plan deviation disclosed; artifact deposit at osf.io/j37b8; title corrected
    </div>
    <details class="paper-abstract">
      We preregistered a comparison of two ways to help an LLM answer questions over a small research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki browsed by a tool-using agent. Both systems answered the same 13 questions over 24 papers using the same answer-generating model, and their answers were scored by two blinded LLM judges. The three preregistered predictions, in registered order, came out one weakly supported, one supported, and one refuted. The wiki was predicted to synthesize better across papers; it scored much better at connecting findings, but its organization advantage fell below the registered threshold once both judges' scores were combined. RAG was predicted to hold its own on single-fact lookup, and it met the registered test, though the second judge alone would have refuted it. The wiki was predicted to be expensive to build and cheap to query; the build side held by roughly two orders of magnitude, but the query side reversed: the wiki spent about 21 times more tokens per query, so no break-even point exists. Two exploratory analyses explain the disagreement. A decomposition-retrieval variant of RAG removes almost all of the wiki's synthesis advantage at lower token cost, though not its advantage in claim-by-claim citation support. Holistic groundedness scoring disagrees with atomized citation checking by direction, and between judges: rank agreement on that criterion is near zero (rho = 0.04), against rho = 0.81 on the most concretely defined criterion. Grounded research synthesis is therefore not a single capability: systems differ in how well they organize evidence, how well their citations support each claim, and what they cost to run, and no architecture here was best on all three. Which one appears to win depends on the retrieval baseline, the scoring granularity, and the judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15943v1">KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-16
    </div>
    <details class="paper-abstract">
      Pipeline parallelism (PP) is widely used to scale large language model (LLM) training, but its efficiency is often limited by stage imbalance and pipeline bubbles. Meanwhile, cross-layer KV sharing has primarily been studied as a mechanism for reducing KV-cache costs during inference, without examining how KV reuse reshapes pipeline workloads. We present \textbf{KV-Pipe}, a stage-aware KV-sharing mechanism that turns KV reuse into a pipeline-balancing control knob. KV-Pipe starts from the tail stage, converts selected attention layers to cross-layer KV sharing in a tail-first order, and iteratively retargets the current bottleneck to drive the FLOPs Imbalance Ratio (FIR) toward $1$. The procedure is performed offline and requires only a pipeline partition and per-layer FLOPs estimates, introducing negligible runtime overhead and requiring no online tuning. Across multiple pipeline-parallel configurations, KV-Pipe consistently improves utilization and throughput, achieving up to \textbf{9.2\%} higher training MFU and up to a \textbf{9.8\%} reduction in iteration time, with larger gains at higher pipeline-parallel degrees where stage imbalance is amplified. Furthermore, the same KV-sharing mechanism provides an inference-side benefit by reducing KV-cache growth and redundant KV projection work, resulting in higher decoding throughput for long-context workloads. These results identify KV layout as a system--architecture degree of freedom for jointly improving pipeline-parallel training efficiency and long-context inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15931v1">PLSQLBench: Benchmarking LLM Systems for Executable Procedural Database Programming</a></div>
    <div class="paper-meta">
      📅 2026-08-16
    </div>
    <details class="paper-abstract">
      We present PLSQLBench, to our knowledge the first benchmark for evaluating whether LLMs can write executable PL/SQL programs, with correctness measured through execution-based tests. Existing LLM evaluations largely target general-purpose code generation or declarative text-to-SQL, leaving procedural database programming underexplored. PLSQLBench contains 2,865 instances: 2,594 single-turn tasks and 271 multi-turn conversations spanning 978 turns. The benchmark combines complex schema-grounded tasks over enterprise-style Spider 2 databases, simpler schema-grounded tasks derived from Spider, and MBPP-derived procedural problems, covering varying levels of database grounding and procedural complexity. Experiments with eight LLMs reveal recurring difficulties in schema grounding, PL/SQL dialect fidelity, procedural control flow, exception handling, and cross-turn consistency. Tool-augmented LLM agents improve performance on several schema-grounded evaluations, although substantial gaps remain. These results highlight procedural database programming capabilities not directly assessed by conventional code generation or text-to-SQL benchmarks. Our code is available at https://github.com/oracle-samples/plsqlbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15893v1">Breaking and Defending LLM-Powered Social Media Bot Detection Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-16
      | 💬 Accepted at ACISP 2026 (Australasian Conference on Information Security and Privacy). Also accepted as a poster at IEEE Symposium on Security and Privacy (S&P) 2026. Published in Pragmatic Cybersecurity 2026, 1(2), 10, https://doi.org/10.53941/pc.2026.100010. 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The rise of social media bots poses a persistent threat, enabling misinformation, opinion manipulation, and the erosion of trust in online platforms. To combat this, machine learning systems have been developed to detect and limit bot activity, but attackers continuously adapt through techniques such as adversarial learning and behavior imitation, fueling an ongoing arms race between bots and detection tools. Recent advances in large language models (LLMs) have significantly improved bot detection by enabling deeper semantic and contextual analysis of accounts and their content. However, this shift also introduces new attack surfaces, allowing adversaries to craft exploits that directly target the reasoning and generation mechanisms of LLM-based classifiers. Industry tools such as Anthropic's Claude Code Security similarly leverage LLMs for security-critical decisions, further motivating a careful study of their attack surfaces. In this work, we investigate both the offensive and defensive aspects of LLM-powered, threat-specific cybersecurity applications. While centered on the challenge of social media bot detection, our methodology and insights generalize to a broad class of LLM-powered cybersecurity systems, including phishing detection, email classification, and fraud analysis. We introduce two novel adversarial attack strategies that systematically exploit the semantic and contextual weaknesses of LLM-based classifiers, degrading their detection accuracy by up to 48%. To counter these threats, we propose a robust multi-LLM defense architecture designed to preserve detection reliability under adaptive adversarial conditions. Our solution, LSABRE (LLM-powered Social Adversarial Bot Recognition Ensemble), is a multi-LLM framework that substantially improves robustness across a range of attacks, maintaining 86% detection accuracy even under strong, adaptive adversarial pressure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.07372v3">Self-Bootstrapping Automated Program Repair: Using LLMs to Generate and Evaluate Synthetic Training Data for Bug Repair</a></div>
    <div class="paper-meta">
      📅 2026-08-16
      | 💬 Final published version in the Expert Systems with Applications journal. Volume 319, 5 July 2026, 132154. DOI: https://doi.org/10.1016/j.eswa.2026.132154
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for enhancing Automated Program Repair (APR) through synthetic data generation utilizing Large Language Models (LLMs). Current APR systems are constrained by the limited availability of high-quality training data encompassing diverse bug types across multiple programming languages. The proposed approach addresses this limitation through a two-phase process: a synthetic sample generation followed by a rigorous quality assessment. Multiple state-of-the-art LLMs were employed to generate approximately 30,000 paired examples of buggy and fixed code across 12 programming languages and 13 bug categories. Subsequently, these samples underwent cross-model evaluation against five criteria: correctness, code quality, security, performance, and completeness. Experimental evaluation on the VulRepair test set dataset showed statistically significant improvements in Perfect Prediction rates, with the quality-filtered synthetic dataset achieving 17.18% (Top@1) and 23.00% (Top@5) compared to the baseline's 11.68% and 18.88% respectively, representing a 47% relative improvement in Top@1 and 22% in Top@5. The methodology was validated through rigorous statistical testing, including ANOVA and post-hoc Tukey's Honest Significant Difference analysis. Furthermore, the best-performing configurations surpassed existing systems despite using a less computationally intensive decoding strategy. This research establishes a self-bootstrapping paradigm in which LLMs generate and evaluate their own training data, suggesting promising directions for addressing data scarcity in similar software engineering tasks and advancing the development of robust, adaptable tools for automated code maintenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13499v1">OpScale: Operator-level Provisioning and Autoscaling for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 22 pages, 29 figures
    </div>
    <details class="paper-abstract">
      Achieving cost efficiency while meeting strict user-facing SLOs (e.g., time-to-first-token) remains a fundamental challenge for cloud GPU clusters serving large language models (LLMs). Autoscaling is the key mechanism for cluster resource management, yet a basic system design question is open for serving LLMs: what should be the unit of scaling? Existing approaches primarily treat the entire model as a monolithic scaling unit--simple but unable to capture the fine-grained dynamics of inference workloads. As a result, such coarse-grained scaling often leads to either SLO violations under bursty demand or significant GPU under-utilization. Our characterization reveals substantial operator heterogeneity, exposing operator-level elasticity as a viable scaling primitive. We present OpScale, a practical operator-level orchestration framework of profiling, provisioning, placement, and runtime serving. OpScale is designed to tackle the high complexity and the space explosion problem, arising from operating at this finer granularity. Evaluated with production traces on up to 40 A100s and 24 GB200s, OpScale attains SLOs with up to 36.3% fewer GPUs and 28% less power, or achieves 44% higher throughput under fixed cost budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04463v2">The Evaluator Is Part of the Experiment: Measuring Open-Ended LLM Conformity</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Prior work on LLM conformity largely measures discrete answer flips under verifiable labels. Open-ended revisions require a different measurement strategy because answer quality is graded, latent, and judged imperfectly. We introduce an experimental protocol implemented across a pooled main peer-condition corpus and separately constructed decomposition corpora, allowing us to separate ordinary re-answering, candidate-content exposure, a bundled peer-presentation residual, and directional judge sensitivity to visible peer context. Across four open-weight generators and three benchmarks, all-wrong peer input produces the lowest-quality revisions in every generator-dataset cell. Blind and informed ratings of identical answers also differ by evaluator: one judge shifts toward the peer-endorsed position, two shift away, one is approximately neutral, and GPT-4o and GPT-5.4-mini audits are likewise non-neutral. Finally, an anchor audit shows that terse correct anchors can be misread often enough to destabilize the latent scale unless calibration is checked explicitly. These results support four conclusions: flip rates are insufficient as a complete measure of open-ended conformity, wrong peers harm open-ended revision, evaluators are not neutral, and anchor calibration is necessary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13484v1">Toward a Gricean Retreat: Probing LLMs for Knowledge Boundaries and Referent Specificity</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      When asked about entities outside their knowledge boundary, LLMs routinely fabricate plausible-sounding details rather than backing off to safer, more general claims. We frame this failure through a Gricean lens: a cooperative speaker who is uncertain about a referent retreats up the specificity hierarchy, trading informativeness for truthfulness. We ask whether LLMs have the ingredients to perform this retreat. Using a T-REx-based benchmark that varies entity familiarity and referent specificity, we probe models to answer two questions: (i) do their activations encode whether a referent falls inside the knowledge boundary, and (ii) do they anticipate the specificity of the referent they are about to generate? We find that the answer to both is yes, but the two signals are not reconciled in generation. Models overwhelmingly prefer specific referents even when the entity is unknown to them, and do so even when offered correct generic alternatives. The substrate for a Gricean retreat is present, but the policy that would act on it is not. We position our findings as a first step toward Gricean alignment, training or steering objectives that couple knowledge-boundary awareness to referent-specificity during generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07316v2">Certifiable Semantic Agreement Among LLM Agents: What the Admissibility Instrument Decides</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 43 pages, 5 figures, 20 tables, 1 algorithm. Substantial revision: new title, new framing, new downstream-audit experiment; supersedes v1
    </div>
    <details class="paper-abstract">
      Can a committee of LLM agents reach agreement that is certifiable at the level of meaning, not only at the level of a label? We build a protocol to find out. H-CSC emits one of three typed outcomes per round -- semantic commit, verdict commit, or typed abort -- under a common 2f+1 distinct-signer certificate, and we use it to measure what such agreement costs and buys. The answer is conditional, and the condition is not the protocol. We prove a containment lemma: whenever the semantic core is large enough to make the committed verdict deterministically valid, the verdict margin already exceeds f, so at matched deterministic guarantees no coverage separation from certificate-wrapped majority is possible. Measurement agrees: the two rules commit the identical task set. What matters instead is the admissibility instrument. Against adversaries that preserve the verdict and corrupt only the reasoning, a 442 MB fine-tuned encoder reaches AUROC 0.621-0.744 at 0-8% TPR (5% honest FPR), while a training-free lexical predicate reaches 0.865-0.982 at 38-80% (50 tasks, 200 attacks, 400 honest) and is exactly deterministic, discharging an assumption the digest proofs rely on. Honest agents disperse further than attacks displace (90th-percentile honest angular distance 0.992 rad against a 0.65 radius), so on free-text rationales no embedding filter is viable -- a property of the output schema, which when tightened collapses dispersion twenty-fold. The committed digest is not inert: under non-disclosure a similarity-preserving sketch separates faithful from substituted derived statements at AUROC 0.741 (n=350), where a cryptographic hash and a verdict-only digest score exactly 0.500. We also report a tie-break capture vulnerability found in our own protocol, and its four-line fix. We claim no safety or coverage advantage over verdict-only certification; the lemma explains why there is none to claim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13454v1">Before You Say It: Anticipating Verbal Behavior from Longitudinal Everyday Conversations with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Knowing someone deeply means not just understanding what they say or do but also how they will likely think, react, and engage across situations. Such predictions could eventually inform systems to anticipate when the individual is about to deviate from their goal, catch regrettable behaviors before they are made, and surface blind spots before they take hold. While many interactive systems model users to enable more personalized interactions, most cannot make such behavioral predictions, as this often requires longitudinal observation and inference of how the individual's behaviors unfold across various everyday situations. In this work, we introduce a novel LLM-based predictive behavioral modeling approach that anticipates a user's likely behavior across everyday conversational situations. We (1) collect a longitudinal dataset of over 1000 hours of naturalistic conversations from 14 participants using a wearable smartwatch; (2) evaluate LLM-based predictions against ground truth behaviors; and (3) use semi-structured interviews to explore participants perceptions of behavioral predictions and their views on possible forms of future behavioral support. Altogether, our findings provide evidence that person-specific verbal behavior can be predicted from longitudinal conversational data. This opens up new possibilities for potential future context-aware, anticipatory, proactive and personalized AI systems.
    </details>
</div>
