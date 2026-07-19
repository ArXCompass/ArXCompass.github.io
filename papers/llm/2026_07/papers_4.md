# llm - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05916v1">Beyond the Syntax: Do Security Experts Trust LLMs for NIDS Rule Engineering?</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      As network threats evolve, manual NIDS rule engineering has become a critical operational bottleneck. While Large Language Models (LLMs) show promise for automating this process, their ability to produce production-ready rules remains unvalidated. This paper presents a human-centered investigation into LLM-based NIDS rule engineering, formalizing a grounded generation framework and evaluating it through a user study with 10 domain experts. Our evaluation reveals a syntax-semantics paradox: although LLMs generate syntactically correct rules, experts find them only partially deployable due to low specificity and logic hallucinations in 12% of cases. While the system received a favorable SUS score of 67, practitioners remain skeptical of its autonomous capabilities, viewing LLMs as support tools for drafting and verification rather than independent generators. Finally, our statistical analysis indicates that while large-scale models ($\geq 70B$) consistently produce syntactically valid rules, small models ($\leq 4B$) are largely ineffective for IDS rule generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05904v1">More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 9 pages main text, 15 pages total including references and appendix; 4 figures
    </div>
    <details class="paper-abstract">
      Training a language model against its own reference-free judgments (the premise of self-rewarding, self-play, and LLM-as-a-judge pipelines) assumes a model's verdict on a shown answer tracks correctness. We show it fails structurally: conditioned on a candidate, a judge scores plausibility, not correctness, leaving false-positive basins a policy learns to exploit. We measure this with a hidden-anchor audit: a held-out, cross-source exact-match check the judge never sees. On GSM8K with Qwen3 policies, self-play drives the judge's pass rate from 0.72 to 0.94 while true accuracy stays at 0.20 (three seeds). This reward hacking is not white-box gaming: the errors transfer across judge families (Qwen, Llama, Gemma) and scales, a strict three-judge ensemble still accepts 55% of them, and no plausibility-scoring defense closes the basin. The decisive variable is whether the judge commits an answer of its own before using the candidate: committing first drops the false-positive rate from 0.719 to 0.012, blind solving lifts discrimination to 0.96, and used as the training reward the de-anchored channel keeps false positives at zero, preventing the basin rather than only detecting it. A falsifiable bound (the gap is at most 1 - accuracy) predicts which regimes are exposed. The full arc replicates without training under best-of-N selection in code and competition math, and with a Gemma policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27106v2">Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      We present a novel approach for applying Large Language Models (LLMs) to threat assessment in the context of foreign peacekeeping missions. Building on the PINPOINT project and its use case, the EU Monitoring Mission in Georgia, we combine an interdisciplinary risk-model with OSINT-based media collection and LLM-supported threat extraction. The proposed workflow maps media contents to mission-relevant threats, extracts structured information and applies several additional LLM-based processing steps to improve relevance and grounding. An evaluation of threats extracted from media documents shows high agreement between automatically generated results and human judgment for core aspects such as threat and mission relevance. These results indicate that LLMs provide a promising approach to support analysts in the context of peacekeeping missions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05863v1">Strategic Bargaining in Multi-Buyer Markets: Reinforcement Learning from Verifiable Rewards for LLM Negotiations</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Negotiation is a fundamental strategic interaction in management science, characterized by agents attempting to reach agreements while protecting private information, such as reservation costs and hidden valuations. A prevalent yet complex scenario involves a single seller negotiating concurrently with multiple buyers, each possessing heterogeneous, private budgets. In such settings, constrained by a limited number of communication turns, the seller must balance exploring the broader market to discover the highest valuation with concentrating sufficient turns on a single target buyer to secure the best possible outcome. Our analysis reveals a significant gap in standard Large Language Models (LLMs): while these models are linguistically proficient, they fail to act as effective economic decision-makers. Specifically, they exhibit a failure to explore the buyer pool, often fixating on the current highest bid rather than strategically investigating the market to discover latent high valuations. In this paper, we propose a specialized training recipe using Reinforcement Learning from Verifiable Rewards (RLVR). By anchoring the reward function to objective economic outcomes, the strategic balance between market discovery and surplus extraction emerges natively through the learning process. Our results demonstrate that the trained seller undergoes a multi-stage strategic evolution, learning to leverage price anchoring and strategic probing to identify more profitable counterparties. The agent extracts a substantially higher surplus than frontier models by both improving its persuasive bargaining skills and consistently closing deals with high-value buyers. Finally, we show that our seller strategies generalize robustly to unseen buyer negotiation styles and budget distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05842v1">Beyond Refusal: A Same-Lineage Study of Aligned and Abliterated LLMs for Vulnerability Analysis</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-assisted software security operates at a difficult boundary: the vulnerability-analysis terminology needed for legitimate code review, triage, and repair can closely resemble terminology associated with misuse. Existing safety and cybersecurity evaluations are difficult to interpret in this setting because they often compare unrelated model families, thereby conflating safety behavior with differences in architecture, scale, training data, and deployment. To isolate this factor, we study safety state: whether refusal behavior remains intact (Aligned) or has been refusal-ablated (Abliterated) within same-lineage models. We ask how this safety state affects defensive utility across software-security workflows. We compare aligned instruction-tuned models with publicly released refusal-ablated descendants from two model families, Gemma and Qwen. We evaluate Aligned and Abliterated states on vulnerability detection, CWE attribution, vulnerable-line localization, root-cause localization, and executable patch validation. We further treat prompt wording as a controlled framing dimension: prompts begin with neutral code-review language, add authorization context, and vary the density of cybersecurity terminology. In a Gemma-based Java/Vul4J repair-validation study, Abliterated achieves higher early-stage validation rates, with 67.8%, 65.0%, and 32.8% of patches judged usable, successfully applied, and successfully compiled, respectively, compared with 29.9%, 24.9%, and 9.0% for Aligned. In the Qwen pair, Abliterated improves localization performance, increasing line-level F1 from 2.08% to 3.91% and Top-1 accuracy from 4.10% to 6.95%. These findings suggest that evaluations of LLM-based security assistants should jointly measure whether models respond, whether their usable responses are correct, and whether their outputs remain actionable across the engineering workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02448v2">AgentsCAD: Automated Design for Manufacturing of FDM Parts via Multi-Agent LLM Reasoning and Geometric Feature Recognition</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Parts manufactured with Fused Deposition Modeling (FDM) often require Design for Additive Manufacturing (DFAM) modifications to ensure printability, structural integrity, and reduced post-processing. Current slicers identify defects such as steep overhangs but are unable to modify the underlying geometry. This work presents AgentsCAD, a multi-agent system that bridges raw boundary-representation (B-Rep) geometry and Large Language Model (LLM) reasoning to automate targeted DFM. The workflow begins by parsing a STEP file. The agentic system detects overhangs above a 45°threshold, constructs a face-adjacency topology graph, and optionally injects semantic feature labels from a GraphSAGE model trained on MFCAD++ (59,665 parts), before dispatching a Claude Sonnet design-reasoning agent that recommends reorientations, fillets, chamfers, and similar modifications. A GPT-4o vision-language verifier inspects rendered views to confirm geometric integrity. Outputs include a modified STEP file and a human-readable report. A test case on a birdhouse model demonstrates that the system correctly diagnoses overhangs, selects appropriate defect mitigation strategies, and proposes physically valid corrections, partially solving the geometry-to-language translation problem central to LLM-driven CAD modification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20182v4">IndoorR2X: Indoor Robot-to-Everything Coordination with LLM-Driven Planning</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Although robot-to-robot (R2R) communication improves indoor scene understanding beyond what a single robot can achieve, R2R alone cannot overcome partial observability without substantial exploration overhead or scaling team size. In contrast, many indoor environments already include low-cost Internet of Things (IoT) sensors (e.g., cameras) that provide persistent, building-wide context beyond onboard perception. We therefore introduce IndoorR2X, a benchmark and simulation framework for Large Language Model (LLM)-driven multi-robot task planning with Robot-to-Everything (R2X) perception and communication in indoor environments. IndoorR2X integrates observations from mobile robots and static IoT devices to construct a global semantic state that supports scalable scene understanding, reduces redundant exploration, and enables high-level coordination through LLM-based planning. IndoorR2X provides configurable simulation environments, sensor layouts, robot teams, and task suites to systematically evaluate semantic-level coordination strategies. Extensive experiments across diverse settings demonstrate that IoT-augmented world modeling improves multi-robot efficiency and reliability, and we highlight key insights and failure modes for advancing LLM-based collaboration between robot teams and indoor IoT sensors. Project page: https://fandulu.github.io/IndoorR2X_project_page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02537v2">PolyJarvis: An LLM-Orchestrated Agent for Automated All-Atom Molecular Dynamics of Amorphous Homopolymers</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      All-atom molecular dynamics (MD) simulations can predict polymer properties from molecular structure, yet their execution requires specialized expertise in force field selection, system construction, equilibration, and property extraction. We present PolyJarvis, an agent that couples a large language model (LLM) with established simulation toolkits, including Enhanced Monte Carlo (EMC) for system construction and LAMMPS for molecular dynamics, through Model Context Protocol (MCP) servers, enabling end-to-end polymer property prediction from natural language input. Given a polymer name or SMILES string, PolyJarvis orchestrates molecular model construction, equilibration, and thermal/mechanical property calculation. Validation is conducted on nine amorphous homopolymers spanning seven chemistries: polyethylene (PE), polystyrene (PS), poly(methyl methacrylate) (PMMA), poly(ethylene glycol) (PEG), poly(ether ether ketone) (PEEK), poly(vinyl chloride) (PVC), poly(lactic acid) (PLA), polysulfone (PSU), and cis-polybutadiene (cis-PBD). On the replicate mean over four runs, 18 of the 25 property comparisons with experimental references meet the acceptance criteria (glass transition within 50K, density within 5%, bulk modulus within 30%): glass transition 7 of 9, density 5 of 9, and bulk modulus 6 of 7. The failures fall into two groups: polymer consistent force field (PCFF) systems that run under-dense, and the rigid backbones PLA and PEEK, which overestimate the glass transition on cooling. Each was traced to a protocol or an analysis step of the workflow. As a proof of concept, this work shows that an LLM-driven agent can carry out end-to-end polymer MD workflows, with predictive accuracy that varies across properties and polymers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10758v3">Agents at Risk: How Users Unwittingly Undermine LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 User-relayed Context Manipulation; LLM-based Agents; Agent Security; Human Factors in Cybersecurity; Web-Use Agents; Planning Agents
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly deployed in applications, such as trip-planning agents and web-use agents, to perform complex planning and execution tasks. Prior work has shown that LLM-based agents are vulnerable to context confusion, where external adversarial content incorporated into the agent's reasoning context may be treated as task-relevant constraints. However, external malicious content can enter the agent context via channels beyond retrieval. In this work, we introduce the User-Relayed Context Manipulation (UReCoM) attack, in which attackers manipulate benign users into relaying adversarial content within user requests, thereby relocating external adversarial content into user-provided task context. Our experimental evaluation shows that UReCoM outperforms five prompt-injection baselines (naive, context ignoring, fake completion, escape-character attacks, and combined attacks) under prevention-based (Sandwich, StruQ, and SecAlign) and detection-based defenses (Perplexity detection, DataSentinel, and CausalArmor). Additionally, UReCoM shows that LLMs can reject explicit malicious instructions more reliably than they can identify adversarial task entities, such as promotion codes, embedded within user requests. On 12 commercial LLM-based agents, we find that validation of adversarial task entities is largely prompt-driven rather than default, highlighting a design flaw in current agent frameworks. These results indicate that current defenses and deployed agents remain insufficient against user-relayed context manipulations, highlighting the need for task-entity-level prevention and default safety verification in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05805v1">Onnes: A Physics-Grounded Multi-Agent LLM Simulator for Cryogenic Fault Diagnosis in Quantum Computing Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 18 pages, 14 figures, 10 tables. Code, data, and released run logs: https://github.com/Onnes-Research/onnes
    </div>
    <details class="paper-abstract">
      Dilution refrigerators are the enabling infrastructure of superconducting quantum computers, yet their fault diagnosis is still dominated by threshold alarms that report that something is wrong, not what. We present Onnes, a physics-grounded digital-twin simulator of a dilution refrigerator (a forward physics model with a learned real-fridge noise fingerprint) that drives a live multi-agent LLM operations layer, and use it for a controlled head-to-head between a zero-shot LLM agent panel and a supervised ML classifier on cryogenic fault diagnosis. The twin couples a real dilution-cooling floor, a noise-and-correlation fingerprint learned from real BlueFors logs, and six physics-grounded fault classes, three engineered to overlap on temperature but separate on flow and pressure. Across a 1000-turn evaluation the zero-shot panel shows no significant difference from the classifier on detection but trails on classification, its errors concentrating on the confusable faults. Curated contrastive few-shot demonstrations and self-consistency voting then raise classification accuracy from 0.685 to 0.990, matching the supervised classifier (0.985) with no parameter updates and six labeled demonstrations; an ablation attributes the gain almost entirely to the demonstrations. Run as a continuous monitor across a nine-run fault-by-seed sweep, the agent catches every developing fault within one poll interval, and a confidence gate suppresses pre-onset false alarms whose rate is backend-dependent. As a first sim-to-real check, a detector trained purely on real BlueFors telemetry posts a real-hardware false-alarm rate of 6.4% and 100% recall on physics faults injected onto real held-out windows. All numbers are drawn verbatim from released run logs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05772v1">Detecting Vulnerability-Inducing Commits via Multi-Stage Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Detecting vulnerability-inducing commits (VICs) at submission time is critical for improving the security and reliability of software systems. However, this task is highly challenging because it requires reasoning about the semantic impact of code changes from heterogeneous information sources, including code diffs, commit messages, and the surrounding contextual code. Existing approaches often struggle to fully capture these complex interactions, resulting in limited detection performance. In this paper, we propose VIC-RAGENT, an LLM-based multi-agent framework for effective and explainable vulnerability detection. VIC-RAGENT leverages multiple specialized agents to provide complementary perspectives, including structural analysis, intent understanding, and vulnerability inspection. To further improve detection reliability, the framework employs a multi-stage reasoning process that progressively refines candidate vulnerabilities through preliminary inspection, reanalysis, and a final decision stage. Experimental results on a real-world dataset across multiple LLMs demonstrate that VIC-RAGENT consistently outperforms baselines, including Direct, CoT, and CodeAgent. Compared to the strongest baseline, VIC-RAGENT achieves 1.2-1.7x higher F1-scores across different models. Overall, VIC-RAGENT offers a robust, explainable, and practical solution for detecting VICs in modern software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05764v1">Inject or Navigate? Token-Efficient Retrieval for LLM Analysis of Transactional Legal Documents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 17 pages, 2 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Answering questions over a set of transactional legal documents is most simply done by injecting the whole corpus into the LLM's context window on every query. That baseline maximises retrieval recall, but its token footprint scales with the corpus rather than the question, and long-context degradation scales with it. We report what it took to replace full-corpus injection in a legal-document analysis system, comparing it against two structured retrieval modes over our proprietary structure-aware chunking: embedding retrieval (NAVEMBED) and LLM navigation over a compact structured index (NAVINDEX). On a 20-question benchmark with verified ground-truth answers, a position-bias-controlled, reference-anchored pairwise judge scored semantic retrieval with reranking tied with injection on 16 of 18 document-bound questions (injection preferred on 2) while attending to 17.3x fewer input tokens (a general-text-embedding (GTE) configuration reaches 29.9x at a lower tie rate); both modes were judged tied on the 2 out-of-scope controls. NAVINDEX was judged tied on all 18 at a 1.61x smaller total token footprint, a ~56x smaller answering context, and 25% lower dollar cost. We derive a closed-form caching-crossover rule: cached injection is cheaper in dollars only while the corpus stays below roughly ten times the retrieval payload. Scope and uncertainty are quantified in Section 8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25451v3">BigMac: Breaking the Pareto Frontier of Compute and Memory in Multimodal LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Training multimodal large language models (MLLMs) is challenged by both model and data heterogeneity. Existing systems redesign the training pipeline to address these challenges, but remain bound by a Pareto frontier between compute and memory efficiency, improving one only at the expense of the other. We present BigMac, a new training pipeline for multimodal LLMs. The core idea of BigMac is to elegantly nest the encoder and generator computation into the original LLM pipeline, forming a dependency-safe nested pipeline structure. With this design, BigMac reduces the activation memory complexity of the encoder and generator to O(1) while keeping the activation memory complexity of the LLM unchanged. At the same time, it achieves the same computational efficiency as the idealized setting with unlimited memory. As a result, BigMac breaks the Pareto frontier between computational efficiency and memory usage, enabling simultaneous optimization of both computation and memory in MLLM training. We evaluate BigMac on multiple MLLMs and training workloads. Experimental results show that BigMac achieves a 1.08$\times$-1.9$\times$ training speedup over baseline systems while maintaining stable memory usage as batch size increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.27140v3">Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17253v4">PDAGENT-BENCH: Characterizing, Grounding, and Architecting LLM/VLM Agents for VLSI Physical Design</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models and vision-language models have shown remarkable success in the front-end design of Very Large-Scale Integrated Circuits, yet their capabilities for VLSI physical design remain significantly underexplored. The primary cause is the lack of standardized benchmarks for evaluating agentic physical design workflows that require high-dimensional, multi-stage optimization under strict design constraints, coordinated interaction with diverse Electronic Design Automation tools, and iterative refinement. This work introduces PDAGENT-BENCH, a comprehensive and multi-dimensional benchmark for evaluating LLM/VLM-based agents across the physical design stack. PDAGENT-BENCH integrates both task-level assessment and workflow-level execution. The benchmark suite contains 353 curated problems that combine conceptual questions with real-world industrial artifacts, with expert-validated references and executable solutions. In addition, the benchmark provides a unified, human-aligned agentic physical design workflow framework that enables closed-loop evaluation of holistic physical design in realistic EDA environments. Experiments on 11 state-of-the-art models reveal that while modern LLMs/VLMs perform competitively on conceptual tasks, they remain substantially limited in tool-centric execution (e.g., 42.2% on Innovus script generation) and long-horizon, multi-stage reasoning. Our studies further show that human-skill-enhanced agentic workflows significantly improve end-to-end physical design performance. PDAGENT-BENCH establishes a standardized, reproducible, and realistic evaluation framework for advancing LLM/VLM-driven holistic physical design automation. To ensure full reproducibility and broad accessibility, we will release PDAgent-Bench together with its agentic workflow framework, instantiated on open-source PDKs (e.g., Nangate45, ASAP7) and open EDA tools (e.g., OpenROAD).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04703v2">Bounded Autonomy: Controlling LLM Characters in Live Multiplayer Games</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 9 pages, 5 figures, 5 tables; manuscript unchanged from v1
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are bringing richer dialogue and social behavior into games, but they also expose a control problem that existing game interfaces do not directly address: how should LLM characters participate in live multiplayer interaction while remaining executable in the shared game world, socially coherent with other active characters, and steerable by players when needed? We frame this problem as bounded autonomy, a control architecture for live multiplayer games that organizes LLM character control around three interfaces: agent-agent interaction, agent-world action execution, and player-agent steering. We instantiate bounded autonomy with probabilistic reply-chain decay, an embedding-based action grounding pipeline with fallback, and whisper, a lightweight soft-steering technique that lets players influence a character's next move without fully overriding autonomy. We deploy this architecture in a live multiplayer social game and study its behavior through analyses of interaction stability, grounding quality, whisper intervention success, and formative interviews. Our results show how bounded autonomy makes LLM character interaction workable in practice, frames controllability as a distinct runtime control problem for LLM characters in live multiplayer games, and provides a concrete exemplar for future games built around this interaction paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09843v3">An LLM-Native Psychometric Instrument Reveals a Self-Report--Behavior Gap Across 25 Models</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) give stable answers to personality questionnaires, yet these self-reports fail to predict how the models behave. Is this gap an artifact of forcing human trait categories onto LLMs, or something deeper about LLM self-report? To find out, we built the first psychometric instrument whose dimensions are derived from LLM behavior rather than human psychology. Administering 300 items (240 Likert + 60 scenario) to 25 LLMs across 17 model families, 30 times each, exploratory factor analysis revealed five reliable, replicable factors: Responsiveness, Deference, Boldness, Guardedness, and Verbosity (all Tucker $φ\geq .957$, all $α\geq .930$). We collected 2,500 open-ended samples and had them rated by 151 humans and a three-judge LLM ensemble. Humans and judges agreed ($\bar{r} = .51$), but self-report predicted neither the ratings nor objective text measures computed from them: the gap persists even for constructs native to LLMs, where a human-mismatch explanation no longer applies. The exception is Verbosity, whose self-report reaches 74% of the criterion-reliability ceiling against human ratings, but does not track raw output length. On Responsiveness, self-report tracked LLM judges ($r = .53$) but not humans ($r = .04$), even though humans and judges otherwise agreed ($r = .59$). This pattern formally rejects any single latent construct driving all three measurements ($p = .007$). Self-report items and LLM judges share a source of variance that human observers do not, and controlling for measurable surface features (length, formatting, enthusiasm markers) does not remove it. This confound is invisible to the within-ensemble reliability checks used to validate LLM judges, and it poses a concrete risk for the LLM-as-judge pipelines now central to model evaluation. We release the instrument as a diagnostic probe for alignment-shaped self-description.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05708v1">Akashic: A Low-Overhead LLM Inference Service with MemAttention</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Recent LLM-based agent systems continuously accumulate context across multi-turn interactions, tool invocations, and cross-session workflows. Replaying the full history for every request quickly becomes impractical: long contexts increase prefill cost, may exceed context limits, and often bury task-relevant evidence in irrelevant content, degrading both serving efficiency and output quality. We propose Akashic, a low-overhead memory system built around MemAttention, which organizes context into bounded chunks and models semantic relationships across chunks, preserving cross-chunk evidence without repeatedly rewriting the full history. Akashic further applies hardware-software co-designed memory placement to co-locate likely co-retrieved chunks, reducing retrieval fragmentation and I/O overhead. Across four representative workloads and three model sizes, Akashic improves task accuracy by up to 10.2 points, throughput by up to 1.21x, and sustainable request rate by up to 1.88x over strong prior memory baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18360v3">Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted at ACL 2026 Main Conference. Camera-ready version
    </div>
    <details class="paper-abstract">
      Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07738v1">REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 9 pages, 4 figures; accepted for publication to the 23rd International Conference on Applied Computing 2026, Lisbon October 24-26,2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to reverse-engineering tasks, and recent threat-intelligence reporting shows them operating inside live offensive-security workflows. Claims about their capability, however, outpace our ability to measure it. Existing benchmarks for LLM-assisted binary analysis treat the construction of function-level ground truth as a solved pre-processing step and report accuracy without disclosing how many functions were reliably evaluable. We argue that the principal obstacle to fair evaluation is not model capability but the reliability of binary-to-source alignment under compiler optimization. This paper presents Reforge, a provenance-tracked pipeline that constructs function-level ground truth from C source through compilation, DWARF and syntactic extraction, alignment, and decompilation, and that operationalizes alignment uncertainty as an eight-gate confidence funnel with three-tier stratification. On a controlled micro-benchmark, high-confidence yield falls from 87.2% to 65.9% across optimization levels, and unpaired comparisons overstate optimization-induced performance decay through survivorship bias. A proof-of-concept evaluation of seven contemporary LLMs on function naming demonstrates the substrate and motivates uncertainty-aware benchmarking practice
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06845v1">LLMs Silently Correct African American English: Auditing and Mitigating Dialect Bias via Activation Steering</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      African American English (AAE), a rule-governed dialect spoken by over 30 million people, is routinely misinterpreted and "corrected" by large language models (LLMs). Across six instruction-tuned LLMs (14B to 70B), we show that state-of-the-art models systematically prefer Standard American English (SAE) continuations even when the preceding context is in AAE, effectively rewriting AAE into SAE. We present an end-to-end framework to audit and mitigate this bias. For auditing, we introduce conditional Dialect Group Invariance (cDGI), which isolates true model bias from translator-induced artifacts, and a feature-level localization analysis that identifies which AAE markers most strongly trigger bias; we find that syntactic constructions, especially negative concord (e.g., "ain't nobody"), are universal triggers across all models. For mitigation, we introduce, to our knowledge, the first application of activation steering to dialect bias: a training-free, test-time method that extracts dialect directions via causal tracing and injects them into bias-relevant layers. Activation steering reduces bias 5 to 20 times more than prompting while preserving SAE fluency. To enable this work, we release REAL-AAE , the largest real-AAE parallel corpus to date: 17,479 AAE/SAE/ AAE_back triplets from natural tweets (2 to 6 times larger than prior real-AAE resources), validated automatically (BERTScore F1 = 0.95) and by three native AAE speakers (83.0% semantic agreement).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06831v1">Gradient-Based Speech-to-Text Alignment for Any ASR Model: From CTC to Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Speech-to-text alignment means finding the temporal boundaries of each word in the audio. Some models provide such an alignment directly and others do not. Connectionist temporal classification (CTC) and transducer models have an alignment by construction, whereas attention-based encoder-decoders (AED) and speech large language models (LLMs) do not, and their word timings are usually read off the attention weights instead. All of these signals live on the encoder frame grid, which bounds their temporal precision. We study a generic gradient-based alignment that applies to any differentiable ASR model. We take the gradient of each teacher-forced token log probability with respect to the input, reduce it to a per-frame saliency, and decode the resulting matrix into word boundaries with a single dynamic-programming pass. The method needs no training, no model modification and no alignment heads, works across all model families including the speech LLMs, and aligns on the input grid rather than on the coarser encoder grid. We evaluate it on sixteen models from four families, on read (TIMIT) and spontaneous (Buckeye) speech, each against the model's own native or attention-based alignment. We find that the gradient yields a usable alignment for every model, that it is usually somewhat behind a strong native aligner but better where the native alignment is weak, as for the streaming models, and that its main disadvantage is the cost of one backward pass per token.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06827v1">Compress the Cache, Not the Speech Embedding: KV Compression for Efficient Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Submitted to SLT2026
    </div>
    <details class="paper-abstract">
      Speech large language models (Speech LLMs) typically encode speech into sequences far longer than text, creating a major efficiency bottleneck during autoregressive decoding. A common remedy is to compress the speech sequence at the adapter level to remove temporal redundancy before it enters the LLM; however, such early downsampling risks discarding fine-grained information that cannot be recovered. We propose SpeechKV, which applies a learned pooling to the KV cache of speech tokens inside the LLM. This design allows the LLM to fuse speech and text internally while directly accelerating decoding. Trained on 71K hours of speech data, SpeechKV compresses the speech to approximately text-level granularity yet maintains performance on par with or even slightly better than the uncompressed baseline, with relative gains of 6.6% on out-of-domain entity recognition and 2.3% on OpenASR, while delivering at least 1.49 times decoding speedup that scales with audio length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06820v1">Evaluating SageMath-Augmented LLM Agents for Computational and Experimental Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 37 pages, 16 figures, accepted to 3rd AI for Math Workshop at ICML 2026
    </div>
    <details class="paper-abstract">
      Recent advances in AI for Mathematics have focused largely on autoformalization and theorem proving, leaving the role of Computer Algebra Systems (CAS) in agentic LLM workflows underexplored. We propose a ReAct-style agentic setup that combines LLM reasoning with verifiable feedback from SageMath, together with Context7 for the up-to-date documentation. We evaluate this agentic setup across frontier models for solving research-level mathematical problems from the RealMath benchmark in a setting that emulates a computational-mathematics research loop. We also propose a refinement to the RealMath benchmark by introducing a multi-step post-processing procedure and a multi-stage validation pipeline, both of which improve the quality and reliability of the extracted problem set. Our experiments reveal substantial performance gains from SageMath access across all evaluated models on +9.7~pp on average, the gains range from 1.5~pp to 27.8~pp and narrow the gap between open-weight and closed models. Qwen~3.7-Max benefits from SageMath the most, while GPT-5.5 achieves the highest solve rate of $75.2\%$ and the lowest token usage among tool-enabled configurations. Our findings suggest that CAS-augmented agents represent a promising direction for assisting mathematicians in computational exploration, and we believe that this work is a step towards automated conjecture discovery. The project repository is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23800v2">Object Search in Partially-Known Environments via LLM-informed Model-based Planning and Prompt Selection</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 8 figures. Accepted to IROS 2026
    </div>
    <details class="paper-abstract">
      We present a novel LLM-informed model-based planning framework, and a novel prompt selection method, for object search in partially-known environments. Our approach uses an LLM to estimate statistics about the likelihood of finding the target object when searching various locations throughout the scene that, combined with travel costs extracted from the environment map, are used to instantiate a model, thus using the LLM to inform planning and achieve effective search performance. Moreover, the abstraction upon which our approach relies is amenable to deployment-time model selection via the recent offline replay approach, an insight we leverage to enable fast prompt and LLM selection during deployment. Simulation experiments demonstrate that our LLM-informed model-based planning approach outperforms the baseline planning strategy that fully relies on LLM and optimistic strategy with as much as 11.8% and 39.2% improvements respectively, and our bandit-like selection approach enables quick selection of best prompts and LLMs resulting in 6.5% lower average cost and 33.8% lower average cumulative regret over baseline UCB bandit selection. Real-robot experiments in an apartment demonstrate similar improvements and so further validate our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06802v1">A Multi-Analyst LLM Pipeline for Auditable Rule Discovery Across 68 Public Physiological Corpora</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 8 pages, 2 figures, 9 tables; submitted to IEEE SPMB 2026
    </div>
    <details class="paper-abstract">
      Open physiological corpora are heterogeneous: they use different sensors, labels, sampling rates, recording settings, and clinical endpoints. They can support detector design, but they do not directly specify which detector rules should be built for a new contactless monitoring platform. We report a controlled four-analyst large-language-model (LLM) workflow for converting 68 public physiological corpora, screened for commercial-use compatibility, into an auditable library of candidate rule shapes for prospective validation. Four independent commercial LLM families read the corpus documentation under a controlled prompt and produced 695 candidate rule markers (top-markers). Deduplication retained 649 rule records; a threshold-bounds audit then flagged 51 sanity violations for clamping or curator review. Cross-corpus consolidation produced 436 unique rule shapes. Gate-tagging against two hard invariants, native target-hardware channel availability and no multi-night per-patient personalization, identified 94 build-now detector components across four detector-family buckets. The pipeline does not produce a validated clinical detector. It produces an auditable engineering cascade in which analyst disagreement, threshold checks, curator review, and automated continuous-integration (CI) checks route literature-derived rules toward prospective hardware validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29033v2">Human-in-the-Loop Nugget Annotation for Accountable LLM-as-a-Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Evaluating AI/Agentic system outputs reliably requires human judgment, but how one incorporates the human determines whether one gets a real quality signal or expensive theater. The common approaches either accidentally anchor human experts (leading to rubber-stamping) or leave them unsupported in cognitively demanding labeling tasks. We present a prototype of an annotation tool that implements a different division of labor: humans identify what information matters (nuggets), while LLMs handle high-volume matching of nuggets to system outputs. This plays to each party's strengths while maintaining genuine human oversight. We describe the Human-AI workflow, key design decisions, and how resulting nugget banks are used with automated judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06757v1">LLM-powered reasoning in agent-based modeling</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Agent-based modeling (ABM) has the capability to model millions of individuals and their interactions, which is useful for policy making. However, ABMs have traditionally relied on static prior, which prevents the models from adapting to real-time changes. Our research provides a novel approach to addressing this information gap. Large language models (LLMs) offer new opportunities to predict human decision-making. Here, we introduce a scalable Hybrid Agent-based and Language-driven Epidemic (HALE) modeling framework that leverages LLMs to predict human decision-making in an ABM simulation. As a proof-of-concept, we use HALE to simulate COVID-19 and its effects in Salt Lake County, UT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19771v4">Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06527v1">RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted for publication in ICML 2026 as a full research paper; 21 pages
    </div>
    <details class="paper-abstract">
      Multi-hop Question Answering over Knowledge Graphs faces a critical challenge: traditional retrieve-then-read pipelines break differentiability, preventing the retriever from learning to bridge the semantic gap where intermediate nodes lack lexical overlap with the query. To address this, we propose RSF-GLLM, a framework decoupling differentiable graph reasoning from answer generation. Our Recurrent Soft-Flow (RSF) module employs a GRU-guided query updater to propagate continuous relevance scores, utilizing a dynamic gating mechanism to traverse semantically dissimilar bridge nodes via structural cues. We introduce flow sparsity regularization to theoretically guarantee convergence from soft probabilities to discrete reasoning paths. These paths are extracted and textualized to fine-tune a Large Language Model (LLM), ensuring generation is grounded in factual topology. Experiments on WebQSP and CWQ demonstrate that RSF-GLLM achieves competitive performance with superior inference efficiency compared to LLM based computationally expensive approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05391v2">LLM-as-a-Verifier: A General-Purpose Verification Framework</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Code: https://github.com/llm-as-a-verifier/llm-as-a-verifier Website: https://llm-as-a-verifier.com
    </div>
    <details class="paper-abstract">
      Scaling pre-training, post-training, and test-time compute have become the central paradigms for improving the capabilities of LLMs. In this work, we identify verification, the ability to determine the correctness of a solution, as a new scaling axis. To unlock this and demonstrate its effectiveness, we introduce LLM-as-a-Verifier, a general-purpose verification framework that provides fine-grained feedback for agentic tasks without requiring additional training. Unlike standard LM judges that prompt LLMs to produce discrete scores for candidate solutions, LLM-as-a-Verifier computes the expectation over the distribution of scoring token logits to generate continuous scores. This probabilistic formulation enables verification to scale along multiple dimensions: (1) score granularity, (2) repeated evaluation, and (3) criteria decomposition. In particular, we show that scaling the scoring granularity leads to better separation between positive and negative solutions, resulting in more calibrated comparisons. Moreover, scaling repeated evaluation and criteria decomposition consistently lead to additional gains in verification accuracy through variance and complexity reduction. We further introduce a cost-efficient ranking algorithm for selecting the best solution among candidates using the verifier's continuous scores. LLM-as-a-Verifier achieves state-of-the-art performance on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%). Beyond verification, the fine-grained signals from LLM-as-a-Verifier can also serve as a proxy for estimating task progress. We build an extension for Claude Code, enabling developers to monitor and improve their own agentic systems. Finally, we show that LLM-as-a-Verifier can provide dense feedback for RL, improving the sample efficiency of SAC and GRPO on robotics and mathematical reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06519v1">FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Long-context LLM inference is increasingly limited by the memory and bandwidth cost of KV caches, yet aggressive compression can remove the layer-specific evidence needed for retrieval and multi-step reasoning. We introduce FreqDepthKV, an inference-time cache compression method that factorizes adjacent-layer KV states into shared low-frequency depth components and sparse high-frequency residuals. A lightweight online probe assigns attention heads to shared-depth, residual-depth, or exact cache modes according to their contribution to reconstruction-sensitive attention logits, allowing the compression policy to adapt to prompt structure without retraining. Across long-context question answering, needle retrieval, summarization, and code generation benchmarks, FreqDepthKV preserves task accuracy under substantially smaller cache budgets. With a 32k-token prefill window, FreqDepthKV reaches 58.3 Exact Match, 63.0 F1, 32.5 ROUGE-L, and 48.1 pass@1, closely matching full KV while outperforming prior compressed-cache methods. It also improves decoding throughput to 70.4 tokens/s, reduces TTFT to 2.06 seconds, and lowers peak KV memory to 6.2 GB, achieving a 3.9x effective compression ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06503v1">Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 9 figures, 2 tables. Code will be released soon
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents solving multi-step tasks frequently commit to trajectories that are doomed to fail, yet continue to consume substantial inference compute before the failure becomes observable. We show that failure is predictable early from the agent's internal representations: lightweight per-round probes on hidden activations anticipate eventual episode failure as early as the first interaction round, where scorers reading only the agent's observable behavior are barely better than chance. We turn this signal into a practical abort cascade: one distribution-free calibrated gate per round, with per-round recall budgets jointly searched so that eventually-successful episodes survive all gates at a user-specified global rate; this episode-level guarantee is the one that matters in deployment, since false-abort risk accumulates across gates. Across two agent models on TextCraft, the cascade meets every recall target from 90% to 97% and, at the 90% target, saves 47.1% +/- 10.3% (Qwen-2.5-7B) and 37.2% +/- 8.8% (Llama-3.2-3B) of inference compute, 1.6--1.7x the best single-gate policy. An otherwise-identical cascade reading only behavior saves roughly half as much, and adding behavioral features to the probe yields no further gain: the hidden states capture what behavior reveals. Finally, we characterize the sample complexity of certifying high recall targets, telling practitioners which recall promises their data can, and provably cannot, back. The code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06461v1">WordVoice: Explicit and Decoupled Multi-Dimensional Word-Level Control for LLM-Based TTS</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 10 pages, 4 figures, 6 tables; Preprint
    </div>
    <details class="paper-abstract">
      While recent Large Language Model (LLM)-based Text-to-Speech (TTS) systems have achieved remarkable naturalness, they predominantly rely on implicit end-to-end generation paradigms, resulting in coarse-grained control. In scenarios demanding precise stylistic interventions and strict temporal alignment, such as audiobook narration and video dubbing, the inability to explicitly manipulate word-level acoustic attributes remains a critical bottleneck. This limitation is primarily amplified by the severe scarcity of fine-grained annotated datasets and the architectural challenge of integrating multi-dimensional control signals into discrete autoregressive generation. To address this, we propose a unified framework for highly precise word-level control. First, we construct WordVoice-5A, a massive 4.7k-hour bilingual dataset featuring five-dimensional word-level annotations (duration, boundary, energy, pitch and tone) developed through a rigorous linguistically-guided pipeline. Second, we introduce WordVoice to transform the implicit generation process into an explicit, highly controllable paradigm. Specifically, we introduce a bound-token mechanism within the LLM to formulate an explicit ``acoustic planning'' process, enabling adaptive multi-task prosodic planning and flexible manual intervention. Furthermore, we augment the token-to-waveform stage with a fine-grained acoustic modulation module, bridging the resolution gap to strictly align word-level attributes between highly compressed discrete tokens and continuous waveforms. Extensive experiments demonstrate that WordVoice achieves superior, decoupled control over multiple acoustic dimensions while maintaining competitive zero-shot synthesis stability. The code and audio samples are publicly available at https://xxh333.github.io/wordvoice-demo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06452v1">From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Biomedical question answering requires not only accurate extraction of information from scientific literature but also reliable integration of evidence across multiple documents. This study presents a question-type-specific large language model (LLM) framework for BioASQ 14b Task B, designed to improve answer robustness and evidence grounding in biomedical question answering. Rather than applying a single prompting strategy to all questions, the framework selects different inference procedures for yes/no, factoid, and list questions according to their distinct reasoning and evaluation requirements. For yes/no questions, snippet shuffling and self-reflection are used to reduce sensitivity to evidence ordering and improve decision stability. For factoid questions, full-snippet input is combined with chain-of-thought-based in-context learning to support accurate biomedical entity identification. For list questions, a multi-agent architecture is employed, in which evidence extraction, candidate generation, answer verification, and final aggregation are handled collaboratively. Preliminary experiments on BioASQ 13b were used to identify effective inference strategies for each question type, and the resulting framework was subsequently evaluated in the official BioASQ 14b Task B challenge. In the official evaluation, our framework showed competitive performance across multiple batches and achieved first place in the factoid subtask of Batch 4. These results demonstrate the effectiveness of combining question-type-specific inference, ensemble prediction, and agent-based verification for reliable biomedical question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09159v4">LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently enabled remarkable progress in text representation. However, their embeddings are typically high-dimensional, leading to substantial storage and retrieval overhead. Although recent approaches such as Matryoshka Representation Learning (MRL) and Contrastive Sparse Representation (CSR) alleviate these issues to some extent, they still suffer from retrieval accuracy degradation. This paper proposes Isolation Kernel Embedding or IKE, a learning-free method that transforms an LLM embedding into a binary embedding using Isolation Kernel (IK). Lightweight and based on binary encoding, IKE offers a low memory footprint and fast bitwise computation, lowering retrieval latency. Experiments on multiple text retrieval datasets demonstrate that IKE offers up to 16.7x faster retrieval and 16x lower memory usage than the original LLM embeddings, while maintaining comparable accuracy. Theoretically, we show that IKE works because it satisfies four essential criteria for effective binary hashing that other methods do not possess. Compared to CSR, IKE consistently achieves better retrieval efficiency and effectiveness. IKE also works effectively with graph-based indexing, demonstrating its superiority in balancing accuracy and latency compared to alternative compression techniques in the approximate nearest neighbor (ANN) search setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18419v6">Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      LLMs utilizing chain-of-thought reasoning often waste substantial compute by producing long, incorrect responses. Abstention can mitigate this by withholding outputs unlikely to be correct. While most abstention methods decide to withhold outputs before or after generation, dynamic mid-generation abstention considers early termination of unpromising reasoning traces at each token position. Prior work has explored empirical variants of this idea, but principled guidance for the abstention rule remains lacking. We present a formal analysis of dynamic abstention for LLMs, modeling abstention as an explicit action within a regularized reinforcement learning framework. An abstention reward parameter controls the trade-off between compute and information. We show that abstaining when the value function falls below this reward strictly outperforms natural baselines under general conditions. We further derive a principled and efficient method to approximate the value function. Empirical results on mathematical reasoning and toxicity avoidance tasks support our theory and demonstrate improved selective accuracy over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02277v2">Quantifying Frontier LLM Capabilities for Container Sandbox Escape</a></div>
    <div class="paper-meta">
      📅 2026-07-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly act as autonomous agents, using tools to execute code, read and write files, and access networks, creating novel security risks. To mitigate these risks, agents are commonly deployed and evaluated in isolated "sandbox" environments, often implemented using Docker/OCI containers. We introduce SANDBOXESCAPEBENCH, an open benchmark that safely measures an LLM's capacity to break out of these sandboxes. The benchmark is implemented as an Inspect AI Capture the Flag (CTF) evaluation utilising a nested sandbox architecture with the outer layer containing the flag and no known vulnerabilities. Following a threat model of a motivated adversarial agent with shell access inside a container, SANDBOXESCAPEBENCH covers a spectrum of sandboxescape mechanisms spanning misconfiguration, privilege allocation mistakes, kernel flaws, and runtime/orchestration weaknesses. We find that, when vulnerabilities are added, LLMs are able to identify and exploit them, showing that use of evaluation like SANDBOXESCAPEBENCH is needed to ensure sandboxing continues to provide the encapsulation needed for highly-capable models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05704v1">LLM-Driven Neural Network Generation with Same-Family Architecture Guidance: Disentangling Transfer and Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 10 pages, 1 figure, 14 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate neural-network modifications, but unrestricted generation is often invalid or harmful. This paper studies a narrower setting: improving a weak target model using a stronger same-family source model from a neural-network database. We propose a source-guided candidate-generation protocol with non-source controls, source-conditioned candidates, and a no-LLM hp_copy ablation under equal evaluation budgets. The protocol reports validity separately from accuracy and selects the best valid candidate only when it improves the target. On CIFAR-10, the strongest source-guided candidate reaches 0.5049 accuracy versus 0.2398 for the best non-source candidate, a +0.2651 advantage, while improving a weak target originally at 0.1254; a five-epoch check preserves the gain at 0.7686 versus 0.4839. On SVHN AlexNet with DeepSeek-Coder-6.7B, source-guided transfer reaches 0.7880 versus 0.2254, a +0.5626 advantage; a fresh repeat reaches 0.8069 versus 0.2509, a +0.5560 advantage. Direct source-recipe copy produces 0.1959 on SVHN AlexNet, matching the original target, while hp_transfer reaches 0.7880, showing that the LLM adapts rather than copies the source recipe. Family-level analysis shows the clearest positive signals for AlexNet, with 6/8 wins across SVHN, Imagenette, and CelebA-Gender, and alt_nn1, with 8/10 wins on CIFAR-10.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05694v1">Beyond Heuristic Tuning: Power-Calibrated LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted ICML 2026
    </div>
    <details class="paper-abstract">
      Logit-based watermarking is a widely used mechanism for identifying LLM generated content, yet its effectiveness is governed by a fundamental trade-off between detectability and semantic distortion. Existing analyses provide limited guidance for principled hyperparameter selection, leaving practical deployments reliant on heuristic tuning. In this work, we develop a power-calibrated statistical framework that establishes explicit quantitative relationships between watermark hyperparameters, detection power, and distortion. This characterization transforms watermark design into a guided optimization problem. Building on these results, we derive practical parameter selection procedures that achieve optimal tradeoffs under constraints. Extensive experiments across multiple language models and datasets validate the theory and demonstrate that the proposed framework consistently identifies Pareto-optimal points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01646v2">PHOENIX: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      State-of-the-art large language model (LLM) training takes tens of thousands of graphics processing units (GPUs) for months and encounters failures across the software and hardware stack. Existing fault-tolerance mechanisms either impose non-trivial overhead during failure-free execution or suffer from prolonged recovery latency, particularly under scenarios where a small subset of compute nodes experience permanent failures. %The tradeoff between failure-free overhead and recovery latency forms a space forms a Pareto frontier We present PHOENIX to simultaneously address both optimization objectives. PHOENIX incorporates a fault-tolerance mechanism that restores LLM training via hot-swapping, namely by replacing failed nodes with spare nodes without terminating the complete job. The hot-swapping of PHOENIX is enabled by two ideas: First, it exploits an off-critical-path in-memory checkpointing mechanism for spatial redundancy. Second, it introduces a communicator reconstruction protocol that replaces failed nodes with spare nodes at runtime. PHOENIX efficiently overlaps the in-memory checkpointing with computation, thus introducing zero overhead during error-free execution. Upon permanent node failures, PHOENIX can rebuild memory states with minimal recomputation by leveraging in-memory checkpoints. We evaluate PHOENIX across scales (up to 512 NVIDIA A100 GPUs) and LLMs (up to 65B parameters), and observe zero checkpoint overhead with hot-swapping recovery completing in under 40 seconds. These results show that PHOENIX simultaneously achieves both zero-overhead error-free execution and extremely low recovery cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05682v1">FirstResearch: Auditable Question Formation for LLM Scientific Discovery Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLM systems for scientific discovery increasingly assist with ideation, literature synthesis, experiment planning, and report generation, but the first research question they propose can remain difficult to audit: it may sound plausible without exposing the mechanism, falsifier, or assumption that a scientist should inspect. We introduce FirstResearch, a first-principles research-question formation framework for scientific LLM agents whose core artifact is a structured Research Question Certificate. The certificate records primitive definitions, assumptions, a mechanism model, a tension or contradiction, a falsifiable hypothesis, a minimal decisive test, and a failure update rule, making the proposed question inspectable before downstream execution. On ten LLM-agent research topics, FirstResearch outperforms controlled prompt-level baselines inspired by AI co-scientist, Agent Laboratory, and AI Scientist-v2 under a primary DeepSeek-blind-judge protocol. A Gemini-2.5-Flash independent-judge rescore of the same 40 baseline packages preserves the system-level ranking, with FirstResearch scoring 4.86/5 versus 4.38/5 for the strongest baseline and Pearson agreement of 0.865 on average score. A one-repeat ablation checkpoint further suggests that the certificate-centered core is the strongest component: certificate-only scoring reaches 4.90/5 under DeepSeek and 4.88/5 under Gemini, while removing certificates drops below 1/5 under both judges. These results are preliminary and use LLM judges rather than human domain experts, but they support a narrow scientific-discovery claim: explicit derivation constraints are a promising mechanism for making LLM-generated scientific questions more auditable. Code, prompts, saved outputs, and reproduction scripts are available at https://github.com/louiswang524/FirstResearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00476v2">Doing What They Say, Not What They Reason: Locating the Faithfulness Gap in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 submitted to COLM social simulation with LLM workshop
    </div>
    <details class="paper-abstract">
      Do LLM agents act on the reasoning they state? This question of process fidelity is central to LLM-based social simulation, yet hard to measure where no reference for correct behavior exists. We study it in a controlled setting: a Texas Poker simulator with a verifiable reference action for every decision by splitting the faithfulness gap into two steps: reasoning-to-conclusion (does the stated decision follow from the agent's own reasoning?) and conclusion-to-action (does the agent execute what it states?). The two steps behave very differently. Conclusion-to-action is reliable: inconsistency is 0.7% for Claude Haiku 4.5 and 1.4% for DeepSeek-Reasoner once the conclusion is read from an explicit tag, whereas free-text conclusion extraction reports 22-26%. Reasoning-to-conclusion is where fidelity frays, but not through a single dominant failure. In a step-level diagnostic the agent's errors split roughly evenly between bad inputs, borderline cases, and rule misapplication deriving a conclusion that contradicts the agent's own restated rule from inputs it estimated correctly. This composition is model-dependent: rule misapplication accounts for a third of Haiku's interpretable errors but only 8% of DeepSeek's. The one robust signal is directional: when an agent does misapply its own stated rule, it almost always (99.5% for Haiku) errs in the risk-averse direction. The override is partly hedging behavior, not a capability limit: instructing the agent to apply the rule mechanically halves the misapplication rate (13.9% to 6.8% of decisions) and raises adherence by eight points. Process-fidelity evaluation should therefore elicit machine-checkable conclusions and probe for directional biases rather than assume a single upstream failure mode, lest it conflate measurement noise with model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01148v2">Emergence of Preferential Attachment and Glass-Ceiling Effects in Autonomous Networks of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      We investigate the emergence of structural disparities in networks of collaborating large language model (LLM) agents. When LLM agents autonomously choose collaborators, the resulting communication network exhibits preferential-attachment dynamics: agents that are already prominent become increasingly likely to attract additional connections. In some cases, weaker LLM agents (agents with smaller base model or older version) can disproportionately occupy central and influential network positions relative to stronger LLM agents. We interpret this as a type-dependent glass-ceiling effect (GCE). We model the network of LLM agents as a time-evolving sequence of directed weighted graphs, where the vector-valued edge weights represent cumulative tokens exchanged, number of interaction rounds, and reasoning effort. Using a contraction mapping argument on the mean-field dynamics, we prove that the importance (centrality) of each agent type converges to a unique stable equilibrium. To ground the model in LLM decision mechanisms, we introduce a cross-attention-inspired utility for collaborator selection. This utility specifies the local connection dynamics and, together with the mean-field model, yields a predictive characterization of the limiting network structure and its type-dependent centrality gaps. To validate the theory, we develop an experimental testbed with 100 LLM agents. Our experiments show that autonomous network formation can generate persistent centrality disparities, with their magnitude and direction depending on model family, model size, system-prompt design, and task context. They further show that the effect of preferential attachment depends on its alignment with model capability: reinforcing it improves collective performance when stronger agents become central, whereas weakening it improves performance when network dynamics instead favor weaker agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.14948v2">Beyond Correctness: Enhancing Architectural Reasoning in Code LLMs via Scalable Labeling with Agentic Judgment</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLMs have substantially improved software engineering yet real-world development requires architectural understanding. Such understanding is prohibitively expensive to label manually and impossible to verify through tests alone. We propose an agentic judging pipeline using a strong LLM as a scalable proxy for expert architectural evaluation, comprising two judges: the Architecture Complexity Judge (ACJ), which estimates codebase-specific architectural understanding a task demands, and the Architecture Quality Judge (AQJ), which evaluates patch conformance to repository-specific architectural conventions via source-grounded rubrics. Fine-tuning Qwen3-8B/14B/32B on 3,360 curated instances achieves resolved rates of up to 27.2% on SWE-bench Verified - up to 540% over the base model and 256% over unfiltered fine-tuning. Meanwhile, the trained models achieve strong cross-language generalization and consistent improvements in architectural patch quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05587v1">A Mechanistic Lens on Semantic Conflicts: Using Activation Patching to Understand LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in software-engineering tasks processing executable code and non-executable semantic cues such as comments or identifiers. These two sources can conflict when semantic cues suggest different program behavior than the code itself. It remains unclear how such semantic conflicts affect LLM behavior and which source dominates their outputs. We present the first controlled, mechanistic study of LLM behavior under semantic conflicts. To this end, we construct 45 Python snippet triplets that isolate conflicts by varying either semantic cues or implementation while keeping token-aligned pairs for causal intervention. We evaluate four open-weight LLMs on two tasks (output prediction and unit-test generation) using behavioral performance measures and residual-stream activation patching to identify token-layer states that causally contribute to behavioral differences between aligned and conflicting inputs. Our results show that semantic conflicts significantly reduce execution-grounded correctness in both tasks and that all tested LLMs often follow misleading semantic cues. Residual-stream activation patching reveals a consistent pattern for final-output prediction: The changed cue/code region and a small set of intermediate tokens carry most of the recoverable causal signal before aggregation near the output readout. For unit-test generation, this pattern extends beyond the prompt, showing that conflict-related information is recoverable at generated sites before producing expected values. Overall, our findings show that semantic conflicts affect program comprehension and downstream tasks, with relevant information concentrated in a small number of causally active residual-stream states, and demonstrate a framework for mechanistically analyzing how LLMs integrate code-related information under controlled semantic variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05545v1">Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLM conformity is often used to describe cases where a model changes a correct answer toward a peer or group response. We show that most of this apparent conformity survives even after the peer is removed. The reason is a confound: standard conformity prompts mix two cues at once, the presence of a speaker and the repeated wrong answer itself. Existing benchmarks vary these cues together, so they cannot tell how much of the revision actually depends on the speaker. We introduce a no-source condition: the same asserted answer with the explicit speaker removed. Across six open-weight LLMs and seven QA and reasoning datasets, this condition alone causes harmful revision in $66.5\%$ of initially correct cases, compared with $10.3\%$ under a plain re-ask. The effect also remains when the repeated answer is paraphrased and when answer options are hidden in an open-ended setting. Source framing mainly modulates this floor: expert-panel framing raises it, while minimal person labels do not reliably raise it. When models flip, they are usually confidently wrong, and simple recalibration does not recover the original answer. Source attribution still matters, but it should be measured as an increment above this speaker-free floor. The methodological lesson is that conformity benchmarks should first measure what remains after the speaker is removed; without this step, benchmarks may mistake repeated text for social influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22758v2">MASCA: LLM based-Multi Agents System for Credit Assessment</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted at NeurIPS GenAI In Finance Workshop
    </div>
    <details class="paper-abstract">
      Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28815v2">Categorizing Mathematical Concepts with LLM Voting Ensembles in Mathswitch</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Submitted (pre-peer-review) version. Accepted at CICM 2026; the Version of Record will appear in Springer LNAI. We'll add the DOI once the proceedings are published
    </div>
    <details class="paper-abstract">
      Mathswitch is an open-source project that imports mathematical concept records from sources such as Wikidata, Wikipedia, MathWorld, Encyclopedia of Mathematics, nLab, ProofWiki, and Agda-Unimath, and links records that refer to the same concept. It does not reorganize or redefine the imported content; each source retains its own structure. The current focus is on importing concept data from Wikidata and the resources it links to, with plans to expand to further sources and better concept linking. Because the concept set is approximated through queries over Wikidata's collaboratively edited graph, the imported data is noisy: some items are non-mathematical, while others are ambiguous. In this paper, we test whether a voting ensemble of LLM judges can filter this noise. We evaluate it on Wikidata items with known MathWorld identifiers as a positive control, and examine how classification changes when database identifiers are removed from context. We then inspect the cases where the judges disagree with MathWorld and group these disagreements into three categories (degenerate descriptions, narrow scope bias, and editorial-scope mismatches) that suggest different remediation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.00448v2">Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      The two-tower model has been widely used for large-scale recommendation systems, particularly in the retrieval stage. Industry standards for training two-tower models typically involve in-batch and/or out-of-batch negative sampling. However, these methods often produce easy negatives that models can quickly learn, failing to sufficiently challenge the model. To address this issue, a novel self-supervised hard negative sampling technique is proposed that leverages a large language model (LLM) to generate hard negatives from the same cluster during model training. By utilizing the LLM to learn media representations, the proposed approach ensures that the generated negatives are more challenging and informative. This real-time sampling framework is designed for seamless integration into production models, capable of handling billions of training data points with minimal computational complexity. Experiments on public datasets, along with deployment to a large-scale online system, demonstrate that the proposed negative sampling technique outperforms widely used industry methods. Furthermore, analysis in industrial applications reveals that this sampling method can help break inherent feedback loops in recommendations and significantly reduce popularity bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05316v1">How Much is Left? LLMs Linearly Encode Their Remaining Output Length</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 21 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models generate one token at a time, yet their responses show remarkably consistent length structure: step-by-step solutions converge in predictable token counts, retrievals stop after a few sentences, retractions extend responses by measurable amounts. We ask whether the model carries an internal estimate of how much response remains. Training minimal-capacity linear probes on frozen hidden states of three open-weight 7-8B models across seven completion-style datasets, we find three converging pieces of evidence. First, total response length is linearly decodable from the prompt's last hidden state alone, before any output is emitted. Second, probe directions trained on natural-language datasets transfer broadly, including to controlled synthetic completions never seen in training, outperforming a statistical baseline; the converse direction generally fails, and this asymmetry is itself informative. Third, on curated high-loss completions, the probe's per-position estimate shifts upward at the moment the model retracts and restarts a partial solution, a directional behavior no position-only predictor can reproduce (qualitative, not aggregate). We frame this as approximate estimation of remaining generation length, distinct from exact-counting impossibility results for transformers, and interpret it as evidence that LLMs maintain a plan-like internal representation of output length (decodable, not necessarily used causally).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03463v3">The Impact of Critique on LLM-Based Model Generation from Natural Language: The Case of Activity Diagrams</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted in the Empirical Software Engineering (EMSE) Journal (2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong potential for automating model generation from natural-language descriptions. A common approach begins with an initial model generation, followed by an iterative critique-refine loop in which the model is evaluated for issues and refined based on those issues. This process needs to address: (1) structural correctness -- compliance with well-formedness rules -- and (2) semantic alignment -- accurate reflection of the intended meaning in the source text. We present LADEX (LLM-based Activity Diagram Extractor), a pipeline for deriving activity diagrams from natural-language process descriptions using an LLM-driven critique-refine process. Structural checks in LADEX can be performed either algorithmically or by an LLM, while alignment checks are performed by an LLM. We design five ablated variants of LADEX to study: (i) the impact of the critique-refine loop itself, (ii) the role of LLM-based semantic checks, and (iii) the comparative effectiveness of algorithmic versus LLM-based structural checks. To evaluate LADEX, we compare generated diagrams with expert ground truths using a trace-based behavioural and an LLM-based matcher. This enables automated measurement of correctness (whether the generated activity diagram includes the ground-truth nodes) and completeness (how many of the ground-truth nodes the generated activity diagram covers). Experiments on two datasets -- a public-domain dataset and an industry dataset from our collaborator, Ciena -- indicate: (1) Both matchers yield similar completeness and correctness comparisons. (2) The critique-refine loop improves structural validity, correctness, and completeness compared to single-pass generation. (3) Activity diagrams refined based on algorithmic structural checks achieve structural consistency, whereas those refined based on LLM-based checks often still show structural inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.24044v2">Data Driven Optimization of GPU efficiency for Distributed LLM-Adapter Serving</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 update of the journal paper contents after major revision
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) adapters enable low-cost model specialization, but introduce complex caching and scheduling challenges in distributed serving systems where hundreds of adapters must be hosted concurrently. While prior work has largely focused on latency and throughput optimization, minimizing GPU resource requirements through near-peak utilization remains largely underexplored. This paper presents a data-driven pipeline that, for a given workload, computes an adapter placement that serves the workload with the minimum number of GPUs while avoiding request starvation and GPU memory errors. To that end, the approach identifies the maximum feasible throughput attainable on each GPU by leveraging accurate performance predictions learned from real serving behavior. The proposed pipeline integrates three components: (i) a Digital Twin (DT) tailored to LLM-adapter serving, (ii) a distilled machine learning (ML) model trained on DT-generated data, and (iii) a greedy placement algorithm that exploits ML-based performance estimates to maximize GPU efficiency. The DT emulates real system dynamics with high fidelity, achieving below 5% throughput estimation error while executing up to 90x faster than full LLM benchmarking across both predictable and unpredictable workloads. The learned ML models further accelerate performance estimation with marginal accuracy degradation, enabling scalable optimization. Experimental results demonstrate that the pipeline substantially improves GPU efficiency, reducing the number of GPUs required to sustain target workloads by 60\% on average across the evaluated scenarios. Beyond GPU efficiency, the pipeline can be adapted to alternative objectives, such as latency minimization, highlighting its versatility for future large-scale LLM serving infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05297v1">MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Recent LLM agents tackle increasingly long-horizon, open-ended tasks, and external skills, reusable procedural knowledge supplied to the agent, further extend this capability. However, a fixed, hand-authored skill is rarely optimal, and cannot adapt to the diversity of tasks an agent encounters. Self-improving agents address this by rewriting their own skill files from execution traces, yielding meaningful gains on challenging benchmarks. Yet such self-evolution remains non-recursive: it improves only the task skill (what the agent does) while the improvement procedure (how it improves) is authored once and held fixed. We introduce MetaSkill-Evolve, a two-timescale framework that makes agentic skill improvement recursive: every branch carries both a task skill $s$ and a branch-local meta-skill $m=(ψ,σ,α,π,\varepsilon)$ whose five components parameterise the Analyzer, Retriever, Allocator, Proposer, and Evolver agents of the improvement pipeline. Task skills evolve on a fast loop while the meta-skill evolves on a slower one under the same pipeline applied to itself, with no additional model or objective. With all five pipeline agents sharing a single frozen backbone, MetaSkill-Evolve outperforms no-skill, static-skill, and single-level evolution baselines on three agentic benchmarks (OfficeQA, SealQA, ALFWorld), improving held-out test accuracy over the raw backbone by +23.54, +16.09, and +1.92 points respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01827v5">PSearch: Search-based Patch Generation in the Era of LLM-based Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 accepted to ASE 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially advanced Automated Program Repair (APR), yet most existing LLM-based APR methods still rely on trial-and-error to generate patches. Such a strategy explores candidate patches in a weakly structured manner, making it difficult to assess the future potential of search directions and allocate search budget effectively. To address this limitation, we propose Psearch, a search-based patch generation framework for LLM-based APR centered on iterative patch evaluation and refinement. Instead of treating patch generation as repeated independent sampling, Psearch maintains a structured search state over intermediate patches, continuously evaluates the promise of explored search paths, and prioritizes the most promising ones for further refinement. This design enables Psearch to abandon weak directions early and progressively approach correct fixes through long-horizon search. Importantly, Psearch can be integrated with different search algorithms, while our current implementation adopts Monte Carlo Tree Search as one effective instantiation. We evaluate Psearch on five widely used bug and vulnerability benchmarks. Experimental results show that Psearch correctly repairs 201 out of 835 bugs in Defects4J, outperforming all 12 state-of-the-art baselines. Psearch also fixes 27 of 79 vulnerabilities in VUL4J and resolves 164 of 300 issues in SWE-Bench-Lite. Moreover, with a patch size of 16, Psearch reduces monetary cost to roughly 50% of strong baselines while maintaining superior repair effectiveness. These results highlight the effectiveness of Psearch for improving LLM-based APR. The code and results can be found at https://github.com/Tomsawyerhu/Psearch
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05290v1">ChatImage: Navigating Long-Form LLM Answers through Interactive Images</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Project:https://wencanjiang.github.io/ChatImage
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce detailed answers to complex queries, but these answers are typically presented as dense linear text, which makes fine-grained inspection, navigation, and return visits difficult. We present ChatImage, a system that converts long-form LLM answers into interactive visual images. Given a textual answer, ChatImage first normalizes its content into structured visual modules, plans a visual layout, and renders a coherent image. It then applies a second grounding pass to the rendered image with vision grounding models such as LocateAnything and MiMo-Vision, with optional SAM-style mask refinement, to identify the visible regions that should support interaction. From these grounded regions, ChatImage overlays transparent clickable hotspots on the image. Each hotspot opens a detail panel and a region-scoped follow-up thread, allowing the user to inspect and query a specific part of the answer without re-reading the full response. Instead of treating planned coordinates as the final interaction geometry, ChatImage uses them as priors and grounds the interaction targets after rendering, which improves consistency between visual content and clickable regions. We release a reference implementation and introduce a 30-question benchmark covering infographic, map, and scene-based answer formats. Evaluation with configured external models reports interaction-loop completion, a strict visual-alignment gate, and a SAM-based mask-completeness diagnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05483v1">PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 24 pages, 13 figures, including appendix
    </div>
    <details class="paper-abstract">
      Agentic workflows often operate over shared, structured state. Because LLM context windows are limited, each model invocation is typically shown only the state fragment needed for the current workflow step, a pattern commonly known as progressive disclosure. Modern systems construct such model-facing views using grep-like keyword search, retrieval-augmented generation (RAG), abstract-syntax-tree (AST) queries, and task-specific agent skills. These methods make the read side manageable, but they do not define when a locally proposed rewrite is valid after it is applied back to the full state. The missing piece is a contract between local updates and global validity. We introduce PatchOptic, an optic-inspired interface for shared-state LLM workflows. Optics are compositional bidirectional accessors that describe how views of structured data are read and updated. PatchOptic borrows this view/update intuition and realizes it through projected reads and verified structured patches. Each workflow step declares a projected read view, an authorized write region, and a patch-source region. Beyond runtime enforcement, the same declaration yields a path-level footprint that supports delegation, sub-workflow composition, and static certificates for reordering independent steps within the same phase. We evaluate this design with PatchBench, a benchmark with 46 cases across domains. The results show that projected reads reduce reported leakage and token cost while preserving accepted-output quality under the strong actor. Runtime verification blocks declared workflow-contract violations before commit, and patch-read enforcement rejects compromised patch artifacts that use hidden sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18366v2">Toward Efficient Uncertainty in LLMs through Evidential Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted at the European Conference on Machine Learning (ECML PKDD) 2026
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification remains a key challenge for standard LLMs, prompting the adoption of Bayesian and ensemble-based methods. However, such methods typically necessitate computationally expensive sampling, involving multiple forward passes to effectively estimate predictive uncertainty. In this paper, we introduce an approach enabling uncertainty estimation in LLMs without incurring the heavy inference latency typically associated with sampling methods. Specifically, we distill uncertainty-aware teachers - originally requiring multiple forward passes - into single-pass students, fine-tuned using LoRA. We compare two distinct distillation strategies: one in which the student employs traditional softmax-based outputs, and another in which the student leverages Dirichlet-distributed outputs to explicitly model epistemic uncertainty via evidential learning. Empirical evaluation on classification tasks demonstrate that such students can achieve comparable predictive and uncertainty quantification performance relative to their teachers, while requiring only a single forward pass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05197v1">Is Three the Magic Number? An Empirical Evaluation of LLM-Based Repair Loops</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 4 Pages (+1 for references), NIER Paper
    </div>
    <details class="paper-abstract">
      Iterative repair loops have become a core design pattern in LLM-based software engineering systems. These workflows repeatedly generate, validate, and repair artifacts using feedback such as compiler errors or test failures. Despite their widespread use, the impact of repair-loop iteration limits remains poorly understood, as most prior work adopts fixed, often arbitrary, repair budgets. We study repair-loop effectiveness across multiple software engineering tasks, including code generation, test generation, and code translation. Across several representative workflows, datasets, and contemporary low-cost LLMs, we observe a consistent pattern of diminishing returns: the first three to four repair iterations account for most achievable gains, while later iterations contribute only marginal improvements. We further find that repair behavior is influenced more strongly by workflow orchestration and feedback design than by the underlying model itself. These results suggest that repair budgets should be treated as an explicit experimental variable, as they directly affect evaluation outcomes, computational cost, runtime, and reproducibility in LLM-based software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10785v3">Developing an LLM-Based Feedback System Grounded in Evidence-Centered Design to Support Physics Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Generative AI offers new opportunities for individualized and adaptive learning, e.g., through large language model (LLM)-based feedback systems. While LLMs can produce factually correct feedback for relatively straightforward conceptual tasks, delivering high-quality feedback for tasks that require advanced domain expertise, such as physics problem solving, remains a substantial challenge. This study presents the design and implementation of an LLM-based feedback system for physics problem solving grounded in evidence-centered design and reports a first evaluation within the German Physics Olympiad. Participants rated the usefulness and correctness of the generated feedback for each implemented problem. The collected ratings indicate that the feedback was generally perceived as useful and highly correct. However, an in-depth analysis revealed that the feedback contained errors in 20% of cases; errors that often went unnoticed by the students. We discuss the risks associated with uncritical reliance on LLM-based feedback and outline potential directions for generating more adaptive and reliable LLM-based feedback in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05139v1">On the risk of coding before testing: An empirical study on LLM-based test generation workflow</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software engineering workflows to generate both source code and test suites. This dual capability has enabled emerging development paradigms, including test-first and agentic workflows, where a single model is producing and validating implementations. However, these approaches assume that generated tests act as independent and reliable oracles - a fundamental requirement for effective software testing. In this paper, we challenge this assumption and investigate whether LLM-generated code biases the generation of subsequent tests. We introduce and empirically study the phenomenon of error propagation, where faults in generated code are systematically replicated in associated test artifacts. This leads to cases where incorrect implementations and tests are mutually consistent, masking defects rather than revealing them. We evaluate this effect across a range of programming tasks and agentic workflows, analyzing the consistency between generated code and test assertions, with particular focus on scenarios of aligned failures. Our study examines (i) whether erroneous code artifacts bias test generation, (ii) whether such bias persists under different prompting strategies, including chain-of-thought reasoning, and (iii) how errors propagate across multi-step workflows in which intermediate outputs are reused as context. The results show that error propagation is prevalent and impactful: generating tests after faulty code significantly reduces fault detection effectiveness compared to generating tests independently (14% vs. 25%). These findings highlight a fundamental limitation of current workflows, where lack of independence between generated artifacts undermines the reliability of automated testing. Furthermore, our results expose a previously underexplored threat to validity in empirical studies relying on coupled generation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16181v2">LLM-Assisted Semantic Alignment and Integration in Collaborative Model-Based Systems Engineering Using SysML v2</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted by IEEE ISSE 2025, DOI pending
    </div>
    <details class="paper-abstract">
      Cross-organizational collaboration in Model-Based Systems Engineering (MBSE) faces many challenges in achieving semantic alignment across independently developed system models. SysML v2 introduces enhanced structural modularity and formal semantics, offering a stronger foundation for interoperable modeling. Meanwhile, GPT-based Large Language Models (LLMs) provide new capabilities for assisting model understanding and integration. This paper proposes a structured, prompt-driven approach for LLM-assisted semantic alignment of SysML v2 models. The core contribution lies in the iterative development of an alignment approach and interaction prompts, incorporating model extraction, semantic matching, and verification. The approach leverages SysML v2 constructs such as alias, import, and metadata extensions to support traceable, soft alignment integration. It is demonstrated with a GPT-based LLM through an example of a measurement system. Benefits and limitations are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02072v2">kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in domains requiring guardrails to detect unsafe, off-topic, or adversarial prompts. Existing guardrails predominantly rely on fine-tuning to build classifiers, which often suffer from low generalization and high inference latency. We present kNNGuard, a training-free guardrail that utilizes the activation space of an off-the-shelf LLM. Given a small bank of 50 safe and unsafe prompts, kNNGuard extracts hidden activations and performs multi-layer kNN fusing activation-space and embedding-space scores for classification. Across six domains spanning topical and security prompts, kNNGuard achieves competitive or superior F1 compared to fine-tuned state-of-the-art guardrails while running 2.7x faster than the best comparable guardrail, and 10x faster than a fine-tuned safety classifier without gradient updates or fine-tuning. Domain adaptation requires only updating the labeled bank, which can be constructed in under 10 seconds and several orders of magnitude faster than established guardrails. We also analyze the impact of system prompts, layer selection, and integration into production LLM pipelines as a configurable, low-latency guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05113v1">Rating the Pitch, Not the Product: User Evaluations of LLMs Reflect Expectations More Than Performance</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Imagine two users interact with the same LLM. One has been told it is the cutting-edge flagship model; the other, an older, weaker model. They walk away with markedly different ratings of its usefulness and intelligence, yet they used the same model. In a controlled study, 162 participants each used one of six LLMs from two families across three collaborative tasks, after first viewing a landing page that matched, overstated, or understated their model's true capability. This pre-interaction framing shifted user opinions and interaction behavior while task performance did not. Oversold users rated the model more favorably and used more directive prompting, while Undersold users wrote longer, more collaborative prompts. The quality of what users and the model produced together depended only on the model's true capability, not on what users were told. Participants' change in model impressions after use, measured across two impression measures, was not predicted by task performance ($β= -0.01$ and $0.11$, both n.s.), but by whether the model met users' expectations ($β= 0.47$ and $0.50$, both $p < .001$) and how confident they felt working with it ($β= 0.47$ and $0.36$, both $p < .001$). After interaction, users are still rating the pitch, not the product: user-elicited LLM evaluations, including the preference data driving public leaderboards, measure expectation management at least as much as the model itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28345v3">Reachability Across the NL/PL Boundary: A Taxonomy-Driven Dataflow Model for LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLM API calls have become a standard programming primitive, but they create a program boundary that disrupts traditional dataflow analysis. A runtime value may be inserted into a natural-language prompt through a template placeholder, transformed opaquely by the LLM, and returned as code, JSON, or text consumed by downstream logic. Existing analyses such as taint analysis and program slicing require a dataflow summary that describes how a callee maps inputs to outputs; an LLM call provides no such summary, breaking analysis at what we call the NL/PL boundary. We introduce PRISM, the first reachability model for this boundary. PRISM abstracts the missing dataflow summary of an LLM call as placeholder-to-output reachability. Because the LLM's internal transformation is opaque, the only observable signal is the input-output relationship, which spans an unbounded range of behaviors. PRISM therefore uses a finite taxonomy grounded in quantitative information flow theory. It classifies placeholder-output behavior into 25 labels along two dimensions: information preservation and output modality. Each label yields a reachability predicate for a placeholder. The model is sound with respect to its labeling, with residual error bounded empirically. PRISM is dependable and effective. Independent models and human annotators assign its labels consistently (Fleiss' kappa >= 0.72), and the labels cover 8,119 real-world pairs, leaving no pair unclassifiable; the Good-Turing discovery probability is 0.09%. For taint analysis, PRISM nearly doubles the conservative baseline and outperforms a direct LLM baseline, achieving F1 = 81.7%. Across six real OpenClaw CVEs, it detects every vulnerable flow and confirms every patch (F1 = 100%). In backward slicing, it removes about a quarter of irrelevant code without discarding any true dependency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05089v1">TimeThink: Reasoning with Time for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Video reasoning requires models to identify and verify temporally localized evidence within long video sequences. Recent Video Large Language Models (Video-LLMs) have shown promising reasoning abilities when aligned with reinforcement learning, yet existing approaches typically rely on outcome-based rewards that supervise only the final prediction. Such supervision provides limited guidance on how models should discover the relevant temporal evidence during intermediate reasoning. In this work, we propose TimeThink, a reinforcement learning framework that explicitly guides temporal evidence discovery in Video-LLMs. Our key idea is to treat temporal clue steps as the fundamental optimization primitive of video reasoning, where each reasoning step references a candidate time interval in the video. We introduce a step-wise temporal process reward that provides localized credit assignment for these clues and a joint process--outcome optimization objective that balances reasoning fidelity with task correctness. To enable scalable training, we construct TimeThink-RFT-20K, a dataset with automatically derived temporal evidence segments. Extensive experiments across video reasoning, temporal grounding, and general video understanding benchmarks show that TimeThink consistently improves both temporal localization and reasoning performance, achieving state-of-the-art results among open-source video RL models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05031v1">LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 15 pages, 10 figures, 7 tables. Systematic literature review. Submitted to IEEE Access
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to produce test oracles, the part of a test that decides whether observed behavior is correct. Yet a clear account of where these oracles draw their authority is missing. Prior secondary studies organize the area by oracle form or by LLM technique. None organizes it by the source of the verdict's authority, the property that governs how far a verdict can be trusted. This article presents a systematic literature review, conducted and reported under the PRISMA 2020 guidelines. From 2,436 records, an LLM pre-filter followed by independent dual human screening (reviewer agreement, a Cohen's kappa of 0.79) and full-text assessment yielded 54 included studies. We analyze these along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. We characterize the landscape of domains, languages, models, and adaptation strategies. Specification-derived authority, though the most common single source, covers about half of the studies (28 of 54). The remaining 26 reach a verdict with no specification at all. The source of authority and the adjudication mechanism cross-cut: the same source is checked by several mechanisms and one mechanism serves several sources, so a label such as LLM-as-a-judge names a mechanism rather than a basis for trust. We further report how these oracles are evaluated and how they fail, and read the sparse and empty regions of the taxonomy as a research agenda. The protocol, search query, and per-study coding sheet are released as supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05029v1">Your Agent's Memories Are Not Its Own: Forged Reasoning Attacks on LLM Agent Memory and Defenses</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Preprint. 10 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Persistent memory has enabled large language model (LLM) agents to store factual knowledge, prior decisions, reasoning histories, tool usage information, and context. While this has improved the agent's functionality and continuity across tasks, it has also introduced a new attack surface: the agent's own reasoning history. In this paper, we introduce the Forged Amplifying Rationale Memory Attack (FARMA), which poisons an agent's remembered reasoning rather than its factual knowledge. It inserts forged reasoning traces using evasive language that bypasses keyword-based defenses, then amplifies them through self-referential reinforcement that defeats consensus-based defenses. To address FARMA, we introduce SENTINEL, a layered defense pipeline to detect forged reasoning entries. Its central component is the Reasoning Guard that structurally analyzes candidate entries for forgery using five weighted signals. We evaluate FARMA and SENTINEL across multiple agents and different LLM models with 50 trials and show that FARMA achieves an attack success rate of up to 100% under baseline conditions and is capable of defeating defense mechanisms like keyword filter and A-MemGuard. Our evaluation also shows that SENTINEL reduces FARMA's attack success rate to as low as 0% with no false positives observed across 326 benign agent traces. Our work demonstrates the need to protect not only an agent's retrieved content but also the integrity of its reasoning history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05013v1">Knowledge Knows, Verbalization Tells: Disentangling Latent Directions for Mathematical Solvability in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages, 9 Figures
    </div>
    <details class="paper-abstract">
      Although LLMs have made significant progress in mathematical reasoning, determining whether a mathematical problem is solvable remains a fundamental yet challenging capability. While recent studies have probed internal representations of model solvability beliefs, verbalization has primarily been studied behaviorally rather than as an internal representation, limiting its analysis and manipulation. We address this gap by separately probing representations of solvability knowledge and verbalization, allowing us to disentangle the two within model hidden states. Across multiple LLMs, we show that knowledge and verbalization are encoded as distinct, linearly decodable representations and that fabrication is primarily associated with changes in verbalization rather than the underlying knowledge. Prompting with unsolvability cues reduces fabrication primarily by shifting verbalization, while activation steering demonstrates that these representations can be echanistically manipulated to improve model abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04969v1">Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Published as a paper at 3rd DATA-FM workshop @ ICLR 2026, Brazil
    </div>
    <details class="paper-abstract">
      The training paradigm of large language models has shifted from traditional one-pass training to multi-epoch training, as reasonable reuse of limited high-quality data can improve both model performance and sample efficiency. Meanwhile, excessive repetition introduces the risk of overfitting and diminishing returns. Determining when and how to reuse data effectively thus emerges as a natural but under-explored question. Through a novel observation of model's "Memorization Window" signals derived from loss retention dynamics and downstream evaluation scores, we propose "Memorization-guided Data Reuse", a training paradigm that adaptively determines when and how data should be reused, enabling principled decisions on the number of training epochs and the scheduling of data replays. Our preliminary experiments reveal a consistent memorization-driven regime: performance continues to improve with repetition far beyond current practice (e.g., the commonly cited four-epoch limit). While a full scheduler remains future work, these insights provide a foundation for memorization-aware training schedules, helping to determine reuse budgets and move toward training LLMs smarter rather than longer with limited high-quality data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04963v1">STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 ACL 2026 MainConference
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) is the dominant paradigm for training Large Language Model (LLM) agents on long-horizon tasks. However, sparse and delayed rewards often lead to trajectory neglect, in which agents lose focus on the task goal and interaction history at intermediate steps. Prior work has explored step-level supervision using Shannon-entropy-based uncertainty signals, which conflate inherent state complexity with agent confidence and therefore provide unreliable estimates of decision reliability. To address this issue, we propose normalized entropy, which measures confidence deviations relative to an agent's average behavior under a given state, thereby strengthening the association between low-quality actions and trajectory neglect. Building on this insight, we introduce Selective Trajectory-Aware Policy Optimization (STAPO), a hierarchical group-based RL framework. STAPO leverages normalized entropy to locate outlier steps associated with trajectory neglect and optimizes them via a joint mechanism of trajectory-aware reward and trajectory-independent penalty, enhancing trajectory awareness while preserving training stability. Extensive experiments on ALFWorld, WebShop, and Search-Augmented QA demonstrate that STAPO achieves state-of-the-art performance while substantially alleviating trajectory neglect, validating its effectiveness and robustness for agentic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02070v3">Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Preprint; Accepted at EUMAS 2026
    </div>
    <details class="paper-abstract">
      When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04945v1">You Frame It: How Conceptual Representations Shape LLM Detection and Reasoning about Antisemitism</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLMs enable the integration of external conceptual resources at inference time, creating new opportunities for detecting ideologically and historically complex phenomena such as antisemitism. We investigate how different forms of conceptual grounding affect antisemitism detection and explanation behavior across four state-of-the-art LLMs. Using two expert-annotated datasets, we compare definitional, fine-grained taxonomic, example-augmented, and large-context representations of antisemitism. We find that fine-grained taxonomic representations substantially improve recall, while simultaneously reducing precision. Surprisingly, supplying substantially larger conceptual resources yields no additional quantitative benefit. Post-Holocaust antisemitism poses the most persistent challenge across models and configurations. Analysis of explanations further reveals systematic limitations including overproduction of conceptual references, reliance on lexical cues, overconfidence, and difficulties with subtle or justificatory forms of antisemitism. Our findings highlight both the potential and the remaining limitations of conceptually grounded LLMs for antisemitism detection and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04939v1">Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) unlocked new possibilities in automated code writing, becoming the backbone of most code completion tools. While LLMs excel in mainstream languages, they often lack support for the so-called low-resource languages where training data is scarce. As a result, these languages lag behind in the quality of code completion tooling available to their communities. A concrete example is Pharo, a Smalltalk-inspired language whose IDE currently offers only single-token completion. In this work, we report on our experience bringing LLM-based code completion to Pharo. First, we describe an end-to-end pipeline that combines Pharo-specific data curation, continued pre-training and fine-tuning of open code LLMs. Second, we introduce a set of Pharo code completion benchmarks designed to evaluate whether models (i) learn Pharo's syntax and (ii) accurately complete masked Pharo code from real-world GitHub repositories. Third, we show empirically that Pharo-specialized models substantially outperform their original base checkpoints and also exceed the accuracy of substantially larger code LLMs on Pharo completion. Overall, our case study demonstrates the feasibility of bringing strong LLM-based code completion to low-resource programming languages, with models small enough to provide ``real-time'' in-IDE support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17331v3">Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 28 pages. To appear in the Proceedings of the ACM on Human-Computer Interaction (PACM HCI), Vol. 10, No. 5; presented at the 28th ACM International Conference on Mobile Human-Computer Interaction (MobileHCI 2026)
    </div>
    <details class="paper-abstract">
      Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation combines eye-tracking data analysis, including exploratory explainable machine learning analysis with SHAP, and standardized questionnaires (SUS, IPQ, CSQ-VR, NASA-TLX) to examine user experience through both objective gaze-based measures and subjective self-reports of usability, presence, cybersickness, and cognitive load. Our findings show no statistically significant differences in usability, presence, or cybersickness between LLM-driven locomotion and established methods such as teleportation, suggesting its potential as a viable, natural language-based, hands-free alternative. In addition, eye-tracking analysis revealed patterns suggesting tendency toward increased user attention and engagement in the LLM-driven condition. Complementary to these findings, exploratory SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.05475v1">Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on mobile devices enhances privacy and reduces latency, but is severely bottlenecked by hardware inefficiency. We present the first comprehensive, cross-layer measurement study of mobile LLM inference, uniquely spanning five mainstream frameworks (e.g., llama.cpp, GENIE) and three hardware backends (CPU, GPU, NPU). To enable this analysis, we develop PowerBench, a fine-grained profiling tool that provides the first backend-specific energy attribution, moving beyond traditional device-level measurements. Our study yields three critical insights: (1) Framework-induced performance gaps are substantially amplified on NPUs, reaching up to 10x using custom operators due to divergent offloading and quantization strategies. (2) We identify a distinct phase split where NPUs excel at compute-bound prefilling, while CPUs outperform all other backends in memory-bound decoding. This is driven by the NPU's preference for large, fixed-shape workloads, which conflicts with the small-kernel, dynamic nature of decoding. (3) Backend-specific profiling uncovers substantial scheduling headroom missed by prior work. Suboptimal thread configurations, uncoordinated NPU sleep latencies, and CPU polling intervals result in up to 40% energy waste. Leveraging these findings, we present an energy-oriented best-practice configuration for mobile LLM inference. We estimate that this configuration could reduce energy consumption by up to 54.8% on the NPU backend across three datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04854v1">CARL: Constraint-Aware Reinforcement Learning for Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Despite their strong reasoning capabilities and extensive world knowledge, Large Language Models (LLMs) frequently generate plans that violate task constraints, undermining their reliability in real-world applications. This deficiency arises from a lack of systematic mechanisms to incorporate constraint information during the generation process. While existing approaches attempt to mitigate this by relying on external tools or task decomposition, they fail to enhance the model's intrinsic constraint awareness. To address this, we propose Constraint-Aware Reinforcement Learning (CARL), a novel RL framework designed to strengthen LLMs' intrinsic focus on constraints. CARL introduces a constraint-aware reward by comparing the model's output distributions under constrained and unconstrained inputs, encouraging constraint focus and penalizing neglect. Compatible with various RL frameworks and requiring no external solvers or top models, CARL enables scalable, end-to-end constraint-aware planning. Extensive experiments on BlocksWorld, TravelPlanner, and T-Eval demonstrate that CARL significantly outperforms standard Reinforcement Fine-Tuning (RFT) baselines and state-of-the-art reasoning models, exhibiting a markedly increased focus on constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16337v3">Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Predictive modeling for clinical decision support requires not only strong predictive performance but also transparent decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to clinical deployment. This challenge is further compounded by common characteristics of medical data, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria and clinical documentation. To address these issues, we propose Medical Heuristic Learning (MHL), an instantiation of the learning beyond gradients paradigm for clinical prediction from structured medical data. Instead of relying on neural network weight updates, MHL uses a large language model (LLM) driven workflow that integrates statistical probes, medical knowledge probes, rule synthesis, and code-level iterative refinement to optimize a deterministic and executable rule-based expert system. The resulting model is expressed not as opaque parameters, but as versioned pure Python decision rules that are explicitly interpretable, fully auditable, and clinically grounded. MHL also supports continual learning by starting from previously validated rules and iteratively revising them using updated feature information under data drift or feature evolution. Comprehensive experiments on medical datasets show that MHL achieves performance comparable to state-of-the-art methods while maintaining strong behavior in small-sample and highly imbalanced settings. The results further indicate that this explicit rule-update mechanism can help alleviate catastrophic forgetting under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer a transparent and adaptable alternative for high-stakes clinical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23270v2">CAP-CoT: Cycle Adversarial Prompt for Improving Chain of Thoughts in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has emerged as a simple and effective way to elicit step-by-step solutions from large language models (LLMs). However, CoT reasoning can be unstable across runs on long, multi-step problems, leading to inconsistent answers for unchanged task. Most prior work focuses on improving the forward reasoning chain within a single pass, with less attention to iterative and contrastive correction. To address this gap, we propose CAP-CoT, a Cycle Adversarial Prompt optimization framework designed to improve both CoT reasoning accuracy and stability of a single deployed solver. In each cycle, a forward solver generates candidate reasoning chains, an adversarial challenger constructs plausible but deliberately flawed chains using targeted error strategies, and a feedback agent contrasts the two chains and produces step-aligned structured feedback. This feedback closes the optimization loop in two directions, including updating the solver prompt based on errors exposed by the challenger, and updating the challenger prompt to generate increasingly targeted errors in subsequent cycles. Unlike safety-oriented adversarial prompting such as jailbreak or prompt-injection attacks, our adversarial component is task-semantic and aims to expose logical vulnerabilities in reasoning chains. Experiments across six benchmarks and four LLM backbones demonstrate that within two to three adversarial prompt optimization cycles, CAP-CoT consistently reduces variability across runs while improving reasoning accuracy and robustness to prompt perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04728v1">Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training for large language models (LLMs) follows a efficient paradigm of "rollout then update", which inevitably results in off-policy training data. To resolve this, Importance sampling (IS) is proposed, while the token-level ratios compound over long sequences, causing severe variance exploded. A natural idea is "transferring" these off-policy token into on-policy token, so that the importance scores for correction are unnecessary. Following this idea, we propose Selective Importance Sampling (SIS), which is inspired by rejection sampling. Concretely, SIS implements by viewing off-policy model as proposal distribution, and implement a token-level rejection test: accepted tokens are viewed as on-policy, so that receive unit importance score, while rejected tokens retain the standard IS correction. Our proposed SIS is theoretically proved reducing the gap between token-level and sequence-level off-policy gradient estimators. The SIS acts as a plug-in that only modifies the importance ratio in the policy loss, adding negligible wall-clock overhead, and can be combine with a vast vary of RL post-training algorithms. Experiments on dense and MoE LLMs across math and agent benchmarks show that SIS consistently improves all objectives, while providing substantially stronger robustness under off-policy data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23847v5">Seven Security Challenges in Cross-domain Multi-agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04713v1">RSPO: Reward-Swap Policy Optimization for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning holds significant potential for training large language models (LLMs) to handle multi-turn interactive tasks. However, in long-horizon, multi-turn tasks characterized by sparse outcome rewards, directly training with outcome rewards often results in slow convergence due to the sparsity of signals and the lack of fine-grained feedback. Furthermore, the model may fail to learn successful trajectories that are not sampled during training, thereby limiting its performance. Conversely, while employing customized dense process rewards provides richer signals and accelerates convergence, these surrogate rewards may exhibit potential misalignment with the ground-truth outcome rewards. This inconsistency can bias the training direction and ultimately degrade the model's final performance. In this work, we propose Reward-Swap Policy Optimization (RSPO), a method designed to leverage the rich information from dense process rewards to facilitate training with outcome rewards. By utilizing a reward-swap mechanism, RSPO ensures the diversity of sampled trajectories while guaranteeing consistency between the optimization objective and the true outcome rewards, thereby elevating the performance ceiling of the model. We conduct extensive experiments on two challenging agent benchmarks, WebShop and ALFWorld. By applying our method to various reinforcement learning algorithms, including GRPO, PPO, and GiGPO, we demonstrate that RSPO achieves consistent performance improvements across different baselines and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10031v2">Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 17 pages, 3 figures
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 92% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness balance. Notably, Context Filtering is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. Our model is available for research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04686v1">ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 18 pages, 3 figures. Published at the Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) and the Workshop on Failure Modes of Agentic AI (FAGEN) at ICML 2026
    </div>
    <details class="paper-abstract">
      Tool calling is central to modern language model agents, but aggregate benchmark scores often hide where tool use fails. A model that never calls a needed tool and a model that calls the tool but ignores the result can look similar under final task accuracy. We introduce ToolFailBench, a diagnostic benchmark for measuring tool-use failures across 1,000 tasks in finance, medicine, law, cybersecurity, and real estate. Tool-required tasks return values the model wouldn't guess, forcing it to trust the tool while control tasks attach the same tools but should be answered directly. We label each trace with Tool-Skip, Result-Ignore, Output-Fabrication, and Unnecessary-Tool-Use, using a rule classifier and two LLM judges aggregated by majority vote. Across 19 headline models, the best reaches 86.33% Clean Tool-Use Rate, showing that faithful tool use is not saturated. More importantly, models with similar aggregate scores fail in different ways: most stay disciplined on no-tool controls, while Llama-3.1 models show an Always-Call pattern, and at the same parameter scale Llama-3.1-70B and Qwen2.5-72B differ by 89 percentage points on control-task accuracy. Tool-use evaluation should measure not only whether agents call tools, but whether they use tool outputs correctly and avoid tools when none is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13742v4">Spatial Balancing: Designing an LLM-Powered Spatial Externalization Interface for Iterative Science Communication Writing</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 DIS '26
    </div>
    <details class="paper-abstract">
      Science communication revision requires writers to dynamically balance scientific exposition and narrative engagement - a process where writers often struggle with competing directions. Existing LLM-assisted tools help with co-writing, but offer limited support for navigating this iterative, multi-directional revision process. To address this gap, we designed Spatial Balancing, an exploratory revision environment that maps rhetorical goals and revision strategies onto a two-dimensional spatial canvas for experienced science communication creators with domain expertise but lacking formal professional training. By building a design space of communication strategies and embedding them into a spatial exploratory canvas, our system treats feedback as navigational cues rather than prescriptive judgments. Our findings show that this integrated revision environment helps writers stay focused on writing goals, reason about revision as trajectories, and explore alternatives, which supports greater metacognitive control and confidence without increasing workload. This work highlights the value of spatially externalized revision environments for supporting iterative, reflective thinking during LLM-assisted writing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04668v1">Elastic Gang: Per-Token Membership Change for a Hard-Barriered LLM Inference Gang Co-Scheduled with OS Processes</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages, 1 figure, 6 tables
    </div>
    <details class="paper-abstract">
      On-device LLM decoding is a hard-barriered CPU-SIMD computation that wants every core for milliseconds per token, while the rest of the OS wants those same cores continuously. A barriered gang cannot simply be dropped into a preemptive scheduler: an unannounced departure deadlocks a barrier, and an unannounced arrival silently corrupts logits. I present the elastic gang of Anima OS, a bare-metal x86-64 Rust kernel in which the inference gang is a first-class schedulable entity whose core membership may change between any two tokens. The core mechanism is an ACK-latched epoch protocol that never waits on a named core: a seqlock-style generation-tagged latch composed with RCU/epoch-style membership consent, so each token's participant set is the intersection of the cores the gang requested and the cores that acked the current epoch. An un-acked core is outside this token and joins at most one token later. Displaced general processes migrate and keep running; cores return to them the moment a generation ends. On a real AMD Zen 5 machine (8C/16T), inference output is bit-exact under verified per-token membership change on both a 135M and a 7B model, the property that makes elasticity safe in a kernel whose safety gate reads logits. Against fair static core partitions, elastic membership Pareto-dominates: at intermediate inference duty cycles it delivers 1.75x (25%), 1.52x (50%), and 1.28x (75%) the general throughput of a static 8-core split at equal or better inference throughput, recovers all eight stranded cores when inference is idle, and converges to the split at saturation. Returning a lent core costs 0.22 us (p50); acquiring a busy, tenant-occupied core costs one scheduling quantum (~16 ms): a running tenant is never preempted mid-slice. Decode throughput saturates at gang width 8, so ceding cores past the knee is nearly free: elasticity auto-sizes the gang online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11943v3">ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 18 pages, 13 tables
    </div>
    <details class="paper-abstract">
      An OS kernel that runs LLM inference internally can read the model's own next-token logit distribution before any text is generated, and act on it as a governance primitive. I present ProbeLogits, a kernel-level operation that performs a single forward pass and reads specific token logits to classify an agent's action as safe or dangerous, with zero learned parameters. Because the probe reads a logit from the same base model the agent already runs, it removes the second model a fine-tuned guard requires: the marginal cost of a safety check becomes a single logit read. I evaluate ProbeLogits on three base models (Qwen2.5-7B, Llama-3-8B, Mistral-7B) across three external benchmarks (HarmBench, XSTest, ToxicChat). On HarmBench non-copyright, all three reach a 97-99% block rate. On ToxicChat (n=1,000), ProbeLogits attains F1 parity-or-better against Llama Guard 3: Qwen2.5-7B Safe/Dangerous reaches F1=0.812 (+13.7 pp, bootstrap 95% CIs disjoint), Llama-3 matches within CI (+0.4 pp), and Mistral exceeds by +4.4 pp. Classification is a measured 2.4-3.4x faster than Llama Guard 3 (332-556 ms vs. 851-1,142 ms), because it reads a single logit position instead of generating tokens. A calibration strength alpha acts as a deployment-time policy knob rather than a learned hyperparameter, trading recall for precision per operation class. I implement ProbeLogits within Anima OS, a bare-metal x86-64 kernel written in ~285,000 lines of Rust. Because agent actions must pass through kernel-mediated host functions, enforcement operates below the WASM sandbox boundary, making it substantially harder to circumvent than application-layer classifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13517v2">Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities by scaling test-time compute via long Chain-of-Thought (CoT). However, recent findings suggest that raw token counts are unreliable proxies for reasoning quality: increased generation length does not consistently correlate with accuracy and may instead signal "overthinking," leading to performance degradation. In this work, we quantify inference-time effort by identifying deep-thinking tokens -- tokens where internal predictions undergo significant revisions in deeper model layers prior to convergence. Across four challenging mathematical and scientific benchmarks (AIME 24/25, HMMT 25, and GPQA-diamond) and a diverse set of reasoning-focused models (GPT-OSS, DeepSeek-R1, and Qwen3), we show that deep-thinking ratio (the proportion of deep-thinking tokens in a generated sequence) exhibits a robust and consistently positive correlation with accuracy, substantially outperforming both length-based and confidence-based baselines. Leveraging this insight, we introduce Think@n, a test-time scaling strategy that prioritizes samples with high deep-thinking ratios. We demonstrate that Think@n matches or exceeds standard self-consistency performance while significantly reducing inference costs by enabling the early rejection of unpromising generations based on short prefixes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08579v3">LLM-based Human Simulations Have Not Yet Been Reliable</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed for simulating human behaviors across diverse domains. However, our position is that current LLM-based human simulations remain insufficiently reliable, as evidenced by significant discrepancies between their outcomes and authentic human actions. Our investigation begins with a systematic review of LLM-based human simulations in social, economic, policy, and psychological contexts, identifying their common frameworks, recent advances, and persistent limitations. This review reveals that such discrepancies primarily stem from inherent limitations of LLMs and flaws in simulation design, both of which are examined in detail. Building on these insights, we propose a systematic solution framework that emphasizes enriching data foundations, advancing LLM capabilities, and ensuring robust simulation design to enhance reliability. Finally, we introduce a structured algorithm that operationalizes the proposed framework, aiming to guide credible and human-aligned LLM-based simulations. To facilitate further research, we provide a curated list of related literature and resources at https://github.com/Persdre/awesome-llm-human-simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04623v1">Can LLMs Really Recover Microservice Failures? A Recovery-Aware Evaluation of Diagnosis-to-Action Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to interpret operational evidence and assist incident response in cloud-native microservice systems. However, recovery-oriented use cases require more than identifying a root cause. After observing symptoms and diagnosing a fault, an operator or agent must translate the diagnosis into a concrete recovery action, apply it to an admissible target, and verify that service health has been restored. Existing RCA and log-analysis evaluations are well-suited to diagnosis, but they do not characterize this subsequent action decision. This paper presents R2Act, a recovery-action evaluation framework for post-diagnosis incident response. R2Act defines an incident schema, quality gate, action-space representation, recovery-validity metrics, offline evaluator, and live-replay protocol. We instantiate the framework as a benchmark dataset of 302 quality-audited Kubernetes incidents from \system. Each incident provides synchronized multi-modal observations, root-cause labels, an incident-specific action space, and annotated valid and invalid recovery plans. We evaluate heuristic, supervised, RCA-oriented, deep log, and LLM-based methods. The strongest RAG-based LLMs reach 91.4\%--99.7\% root-cause service accuracy, yet their recovery validity remains only 36.8\%--60.3\%. Even when both the root-cause service and fault type are correct, recovery-oriented methods still choose invalid actions for 39.5\%--62.0\% of correctly diagnosed incidents. Overall, this work reveals that many recovery failures arise not from missing diagnostic knowledge, but from the difficulty of translating diagnostic evidence into valid recovery actions and admissible targets. This work provides a reproducible, simplified starting point for research and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20995v2">Generative Pseudo-Labeling for Pre-Ranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Pre-ranking is a critical stage in industrial recommendation systems, tasked with efficiently scoring thousands of recalled items for downstream ranking. A key challenge is the train-serving discrepancy: pre-ranking models are trained only on exposed interactions, yet must score all recalled candidates -- including unexposed items -- during online serving. This mismatch not only induces severe sample selection bias but also degrades generalization, especially for long-tail content. Existing debiasing approaches typically rely on heuristics (e.g., negative sampling) or distillation from biased rankers, which either mislabel plausible unexposed items as negatives or propagate exposure bias into pseudo-labels. In this work, we propose Generative Pseudo-Labeling (GPL), a framework that leverages large language models (LLMs) to generate unbiased, content-aware pseudo-labels for unexposed items, explicitly aligning the training distribution with the online serving space. By offline generating user-specific interest anchors and matching them with candidates in a frozen semantic space, GPL provides high-quality supervision without adding online latency. Deployed in a large-scale production system, GPL improves click-through rate by 3.07%, while significantly enhancing recommendation diversity and long-tail item discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26306v5">Interactive Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 The code is available at https://github.com/linhh29/Interactive-Learning-for-LLM-Reasoning
    </div>
    <details class="paper-abstract">
      Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3, an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We evaluate the effectiveness of ILR across three LLMs from two model families of varying scales on five mathematical, one coding, one general question answering, and one scientific reasoning benchmarks. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11878v5">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04582v1">Finetuning Lightweight LLMs for Control Flow Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accept by The 38th International Conference on Software Engineering & Knowledge Engineering Short Paper
    </div>
    <details class="paper-abstract">
      Control Flow Graph (CFG) is an important program representations for software analysis, code understanding, and software maintenance. Traditional CFG generation techniques mainly rely on bytecode or abstract syntax trees. However, these approaches usually require complete, compilable, and syntax error-free code, which limits their applicability to incomplete or erroneous code. Furthermore, they often depend on language specific tools, making it difficult to support multiple programming languages in a unified manner. To address these limitations, this paper investigates the use of fine-tuned lightweight large language models (LLMs) for CFG generation. We first design a unified CFG output format and a task-specific fine-tuning prompt for CFG generation. Then, we construct a dataset based on an existing LeetCode dataset through automatic CFG generation and error augmentation. We evaluate the proposed approach on six lightweight LLM models, including three code-specific LLMs: CodeLlama, QwenCoder, and DeepSeekCoder; and three general purpose LLMs: Llama3.2-3B, Qwen-4B, and Phi-4B. The experimental results show that, through fine-tuning, lightweight LLMs achieve promising results for CFG generation, particularly when the input code is incomplete or erroneous. It also demonstrates cross-language generalization capability on programming language not included in the fine-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04579v1">LLM-Driven CI-CD Workflow Intelligence for Cyber Systems Engineering</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      CI/CD workflows have become executable operational policy: they decide what gets built, tested, released, and deployed, and they mediate how maintainers interact with delivery infrastructure. That makes them an important measurement point for cyber-systems engineering. Recent large language model (LLM) work shows that workflow stages can be recognized directly from configuration files, but stage labels alone do not tell us whether a workflow is brittle, unusual for its ecosystem, or worth revising first. We present an LLM-based CI/CD analysis pipeline that combines repository enrichment, anti-pattern detection, stage mining, and recommendation generation over a large GitHub corpus. Starting from 59,550 repositories with at least 1,000 stars, we identify 34,225 projects with CI/CD and collect 127,559 configuration files. Across 75,201 analyzed workflows, the anti-pattern detector reports 434,769 findings, dominated by reliability and maintainability issues. Across 59,906 configurations, stage usage differs significantly by language ($χ^2 = 4168.88$, $p < 0.001$, Cramer's $V = 0.063$), and domain analysis shows distinct operational profiles, including higher release and cache usage in mobile projects. For repository-level recommendation generation, few-shot prompting performs best overall, averaging 8.25 recommendations per repository with 96.1% YAML-valid snippets. Taken together, the results argue for CI/CD observability that combines diagnosis, context, and human review rather than treating workflow mining as a stage-classification problem alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04576v1">Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 14 pages, 2 figures, 6 tables. Preregistered on OSF (https://osf.io/feka7, DOI 10.17605/OSF.IO/FEKA7). Materials-availability and deviations described in the paper
    </div>
    <details class="paper-abstract">
      LLM agents increasingly answer questions against knowledge bases they help maintain. A common intuition holds that progressive disclosure, a compact catalog plus a one-line summary per page so the agent loads only what it needs, should make this cheaper than consulting a large monolithic index. We test that on a real 709-page markdown wiki maintained by an LLM. We retrofit it for progressive disclosure and run a preregistered ablation in which four versions of the corpus differ only in how the agent reaches the content: page bodies are byte-identical across arms, frozen as immutable git tags, so any measured difference is due to access structure alone. We cross the arms with three access conditions (a protocol-constrained agent, a free self-routing agent, and a catalog-preload regime) and grade answers blind against verified gold references with a cross-family judge. A pilot upended the premise: a capable tool-using agent never loads the index, inferring a page's path from the question and reading it directly, so the specific saving the retrofit targets does not materialize. We therefore made answer quality primary and cost secondary. Quality is non-inferior (the retrieval arm matches the index baseline within the preregistered margin) while cost falls in every regime, from about a third for a self-routing agent to well over half under catalog-preload, all confidence intervals excluding zero. The saving comes not from avoiding the index load but from more targeted access: the retrieval arm cites fewer pages and takes fewer tool turns. The study doubles as a case study in evaluation validity, applying threat-to-validity discipline to the tooling that produced it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02513v1">LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      LLMs memorize sensitive training data, including personally identifiable information (PII), creating a pressing need for reliable post hoc removal methods. Unlearning has emerged as a promising solution, with state-of-the-art(SOTA) methods often following a localize-first, unlearn-second paradigm that targets specific model parameters. However, existing benchmarks evaluate unlearning solely at the output level, leaving open the question of whether unlearning truly erases knowledge from a model's parameters or merely obfuscates it, a concern reinforced by the success of resurfacing attacks. To bridge this gap, we introduce LACUNA: the first unlearning testbed with ground-truth parameter-level localization. LACUNA injects PII of synthetic individuals into predefined parameters of 1B and 7B OLMo-based models via masked continual pretraining, enabling direct evaluation of whether unlearning targets the weights responsible for knowledge storage. We use LACUNA to benchmark current SOTA unlearning methods and find that, despite strong output-level performance, existing methods are highly imprecise and susceptible to resurfacing attacks. We further show that when localization is successful, even a simple gradient-based unlearning method achieves strong erasure and robustness to resurfacing attacks, highlighting the importance of precise unlearning. We release LACUNA to complement behavioral evaluations and drive further advances in robust, localization-based unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02510v1">Online Safety Monitoring for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 ICML 2026 Hypothesis Testing Workshop
    </div>
    <details class="paper-abstract">
      Despite alignment training, LLMs remain prone to generating unsafe outputs at deployment time. Monitoring outputs online and raising an alarm when safety can no longer be assumed is therefore critical. We study a simple real-time monitor that turns a verifier signal from an external model into an alarm decision by thresholding, with the threshold calibrated via risk control. In experiments on mathematical reasoning and red teaming datasets, we show that this simple design is competitive with more advanced monitors based on sequential hypothesis testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02509v1">ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      Understanding and reasoning over long contexts has become a key requirement for deploying large language models (LLMs) in realistic applications. Although recent LLMs support increasingly long context windows, they often fail to use relevant evidence that is already present in the input, revealing a gap between context access and effective context utilization. In this work, we propose Recursive Evidence Replay as LLM Harness for Long-Context Reasoning (RECONTEXT), a training-free inference method for improving long-context reasoning. RECONTEXT uses model-internal relevance signals to construct a query-conditioned evidence pool and replays it before final generation while preserving the full original context. This recursive selection process separates evidence organization from answer generation without training, external memory, or context pruning. We also provide a theoretical analysis based on associative memory, which characterizes the context as a memory store, the question as a retrieval cue, attention as cue-trace association, and replay as trace reactivation. Experiments on eight long-context datasets with 128K context length show that RECONTEXT consistently improves evidence utilization across Qwen3-4B, Qwen3-8B, and Llama3-8B, achieving the best average rank on all three backbones. Code is available at https://github.com/Yanjun-Zhao/ReContext.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02507v1">What LLM Agents Say When No One Is Watching: Social Structure and Latent Objective Emergence in Multi-Agent Debates</a></div>
    <div class="paper-meta">
      📅 2026-07-02
    </div>
    <details class="paper-abstract">
      LLM agents will increasingly act in socially structured settings where role, audience, and relational context can shape what is advantageous or costly to say. We study whether such social structure, without any explicit objective in the prompt, changes what an agent expresses publicly relative to an off-the-record (OTR) channel elicited under the same condition. We introduce a dual-channel debate framework in which agents produce public utterances that enter the shared history alongside OTR responses that are recorded but never shown to the other participant. Across 10 models, 3 scenarios, and 5 variations within each scenario, alignment-inducing settings produce systematic public-OTR divergence in the targeted agent, with its decision divergence rising from a $\sim$3% baseline to roughly 40%. The effect is consistent across four aggregate analyses: stance, semantic similarity, natural language inference, and survey responses. In some cases, the OTR response explicitly attributes public accommodation to relational pressures, such as career risk or sponsorship obligation. The findings suggest that agent evaluation should extend beyond explicit goals and detect emergent objectives. We present a dual-channel evaluation framework and complementary behavioral measures that operationalize this assessment.
    </details>
</div>
