# llm - 2026_08

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06135v1">LLM Inference Under Bursty Workload Distribution: Modifying the WAIT Algorithm</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT and Claude are widely used for information retrieval and problem-solving. Recent work has focused on improving scheduling algorithms to boost throughput while maintaining low latency. However, these approaches often assume Poisson request arrivals with constant rates - an assumption that fails to reflect the inherently bursty and dynamic nature of real-world traffic. We propose a lightweight extension to the state-of-the-art WAIT algorithm [1], which adapts to time-varying arrival rates without prior traffic knowledge. The proposed algorithm performs online estimation of request intensity based on observed interarrival times. Using Markov Modulated Poisson Process (MMPP)-based synthetic workloads with diverse request types, we conduct a simulation-based evaluation demonstrating that the proposed method achieves higher throughput than Sarathi-Serve [2], ORCA [3], and vLLM [4] in the evaluated low arrival-rate shift scenarios while maintaining comparable latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03997v2">Mapping Patient-Perceived Physician Traits from Nationwide Online Reviews with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted in npj Digital Medicine
    </div>
    <details class="paper-abstract">
      Understanding how patients perceive their physicians is essential to improving trust, communication, and satisfaction. Patients increasingly consult large language models (LLMs) to summarize physician reviews and shape provider choices, yet the national landscape of patient-perceived physician traits remains poorly characterized. We present an LLM-based pipeline that extracts ten patient-perceived physician trait scores from review text: five Big-Five-style and five patient-oriented dimensions. From one million U.S. physicians, we analyze 4.1 million reviews of 226,999 physicians. We validate the pipeline through multi-model comparison and human expert benchmarking. LLM and human-rater trait scores from reviews are consistent. Trait scores correlate strongly with review rating scores yet retain substantial independent variance. Two national-scale patterns emerge: male physicians receive higher trait scores across all traits, with the largest gap in clinical competence; specialty differences are driven by encounter context, with surgical specialties leading interpersonal qualities and psychiatry lowest. Cluster analysis identifies four physician archetypes, from "Uniform High" (33.8%, high across traits) to "Uniform Low" (22.6%, low across traits). This map of LLM-derived physician traits exposes how LLMs read the U.S. clinical workforce. Pending clinical validation, it opens future research on fairness, bias, and how LLM-mediated provider search shapes patient choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06085v1">Signal or Spurious Cue? A Randomized Audit of Survey-Country Metadata in LLM Social Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Survey-country metadata can improve an LLM's forecast of an individual response when informative, yet the same cue may redirect the forecast when assigned at random. A within-record audit tests whether disclosing a random label's uniform, record-independent origin reduces its country-directed uptake, and whether verified survey country lowers held-out Brier loss. Independent population anchors and recorded human answers measure direction and consequence across five fixed API models, six countries, and seven development-selected targets. In the primary post-review 72-record panel, opaque and disclosed-random labels each produced country-direction shifts of 0.214. Paired attenuation was 0.0003 (95% CI [-0.0157, 0.0166]). Verified country reduced Brier loss by 0.040 (95% CI [0.024, 0.056]), while random-label regret included zero. A non-overlapping mixed-coverage consistency panel retained positive disclosed-random movement and verified utility, while attenuation remained uncertain. On the selected targets, verified metadata was useful in both panels, but disclosure did not reliably attenuate random-label uptake. PROV-FORECAST contains 14,400 paired item-level probability distributions from the corrected panel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14751v3">One Leak Away: How Pretrained Model Exposure Amplifies Jailbreak Risks in Finetuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 This paper has been accepted to the ACM SIGSAC Conference on Computer and Communications Security (ACM CCS)
    </div>
    <details class="paper-abstract">
      Finetuning pretrained large language models (LLMs) has become the standard paradigm for developing downstream applications. However, its security implications remain unclear, particularly regarding whether finetuned LLMs inherit jailbreak vulnerabilities from their pretrained sources. We investigate this question in a realistic pretrain-to-finetune threat model, where an attacker has full access to a released pretrained LLM but no access to its proprietary finetuned derivatives. Empirical analysis shows that adversarial prompts optimized on the pretrained model transfer most effectively to its finetuned variants, revealing inherited vulnerabilities from pretrained to finetuned LLMs. To further examine this inheritance, we conduct representation-level probing, which shows that transferable prompts are linearly separable within the pretrained hidden states, suggesting that transferability-relevant structure is already encoded in pretrained representations. Building on this insight, we propose the Probe-Guided Projection (PGP) attack, which steers optimization toward transferability-relevant directions. Experiments across multiple LLM families and diverse finetuned tasks confirm PGP's strong transfer success, underscoring the security risks inherent in the pretrain-to-finetune paradigm. Finally, we demonstrate that the same representation-level insights also enable a lightweight defense that mitigates pretrain-to-finetune jailbreak transfer while preserving downstream utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06069v1">Training-Free Token-Level Steering for LLM Personalized Co-Writing</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) show great promise for personalization, they often lack specialized domain knowledge. Conventional solutions like fine-tuning struggle with high computational costs and rapid data updates, while Retrieval-Augmented Generation fails to provide fine-grained, token-level steering. Furthermore, chat-based interfaces remain dominant, whereas productive co-writing paradigms have not yet been well exploited beyond the coding domain. To this end, we introduce SteerWrite, a training-free framework designed for personalized co-writing. Our method effectively adapts the base model to specialized domains without gradient updates, with specific designs tailored to small datasets. Experiments demonstrate that SteerWrite achieves state-of-the-art performance across diverse datasets, metrics, and models, significantly reducing human editing effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00737v3">To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Agentic AI architectures augment LLMs with external tools, unlocking strong capabilities but potentially incurring substantial costs. Moreover, tool use is not always beneficial: redundant or low-utility calls can even harm task performance. Effective tool use, therefore, hinges on a core LLM decision: whether to call or not call a tool when performing a task. We introduce a principled framework inspired by decision-making theory to understand tool-use decisions along three key factors: necessity, utility, and affordability. Our analysis combines two complementary lenses: a normative perspective that infers true need and utility for optimal tool calls, and a descriptive perspective that infers the model's self-perceived need and utility from their observed behaviors. We evaluate six open models and a proprietary OpenAI model across native and customized harnesses, two tools, and six tasks. Models' perceived need and utility remain misaligned with their true values, particularly under budget constraints. This misalignment produces both costly overuse and performance-degrading calls. To improve the tool decisions, we train lightweight latent estimators of need (LNEs) from model hidden states. LNEs generally predict true need more accurately than model self-reports and improve budgeted tool allocation across model scales and tool types. Code and dataset available at https://github.com/QinyuanWu0710/ToCall_or_NotToCall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06041v1">LangChoiceBench: Measuring and Explaining Programming-Language Choice in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 19 pages, 9 tables, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to exhibit strong Python preferences when generating project-level code, but there is currently no systematic way to measure this behaviour across new models. To bridge this gap, we introduce LangChoiceBench, a project-level code-generation benchmark for measuring Python preference, recommendation-implementation consistency, and language diversity. LangChoiceBench covers 28 projects across seven software areas where Python is often a poor default. We evaluate 25 diverse LLMs and find that Python remains heavily over-selected, recommendation-implementation consistency is low, and smaller open-weight models generally show stronger Python preference and lower language diversity. We further analyse 9,826 reasoning traces and find that most Python choices are automatic or driven primarily by ease, rather than explicit consideration of project requirements. In a smaller but important set of cases, models fabricate contextual support for choosing Python - a failure mode we call phantom evidence - or produce code that contradicts the language selected in their own reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06022v1">EpiBench: Can LLMs Understand Epitopes for Antibody Drug Discovery?</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Epitopes determine where antibodies bind antigens and shape downstream therapeutic properties such as functional blockade and escape resistance, making epitope understanding central to antibody drug discovery. Although large language models (LLMs) have shown strong biomedical reasoning ability, it remains unclear whether they can infer epitope information directly from antigen and antibody sequences. Existing epitope resources typically focus on isolated prediction tasks or rely on specialized structural settings, while general protein benchmarks do not evaluate epitope-centered decisions across the antibody development workflow. To address this gap, we introduce EpiBench, a closed-book, sequence-based, and automatically scorable benchmark for evaluating epitope reasoning in LLMs. EpiBench contains 1,609 curated samples grounded in structural antibody--antigen contacts, curated functional B-cell assays, and deep mutational scanning escape measurements. It covers five connected tasks: targetable region discovery, antibody-conditioned epitope identification, epitope binning, functional epitope assessment, and antibody escape assessment, with controlled sampling to reduce shortcut-based evaluation artifacts. We evaluate nine general-purpose LLMs and analyze their behavior through task-specific baselines, antigen length stratification, explicit-reasoning comparison, and failure-mode inspection. The results show that current LLMs capture partial epitope-related signals but remain limited in antibody-specific sequence grounding, long-context residue localization, and biologically grounded reasoning. Therefore, EpiBench provides a diagnostic testbed for measuring and improving sequence-aware biomedical LLMs toward reliable LLM-assisted antibody discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00152v2">PrivacyPeek: Auditing What LLM-Based Agents Acquire, Not Just What They Say</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 21 pages, 17 figures
    </div>
    <details class="paper-abstract">
      LLM-based agents are rapidly advancing, autonomously invoking external tools to complete multi-step tasks for users. However, agents often acquire more sensitive information than the task requires. Existing privacy benchmarks audit what the agent's response or outgoing actions disclose, but overlook the acquisition stage where data first enters the agent's context. The over-acquired information is then one careless action or one attack away from an outright leak. To assess its prevalence, we introduce \emph{PrivacyPeek}, a benchmark for evaluating acquisition-stage privacy leakage of LLM-based agents, with $1{,}182$ cases across $7$ acquisition behaviours and $16$ application domains. Specifically, \emph{Acquisition Inspection} examines the agent's tool-call trajectory, both the tools it invokes and the data it receives, to detect when it acquires sensitive information beyond the task scope. \emph{Probe Elicitation} then issues a follow-up probe and measures how readily an attacker could elicit sensitive information the agent acquired but did not disclose. Our experiments on $10$ LLM-based agents across $4$ model families show that the unnecessary acquisition of sensitive information is widespread. In addition, we observe a correlation between the task-completion capability and acquisition-stage leakage. Prompt-level defences reduce only a small fraction of acquisition-stage leakage, leaving the majority unmitigated. These results make auditing acquisition-stage privacy both urgent and necessary. Our dataset and code are available at https://github.com/Xuan269/PrivacyPeek-Resource.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02940v2">When Compression Scores Cannot Decide: Information Boundaries for Group-Robust LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 19 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      A stable compression score can still select the worse model. In our dense study, a split-half reliable path-quadratic score predicted a 16.1\% gain, while the selected endpoints were 6.0--7.7% worse than two controls. We ask what a compression statistic can justify when deployment cares about the worst supplied group. We treat each statistic as an information interface. Its observation leaves a fiber of compatible endpoint-risk tables, and only orders fixed across that fiber are identified. Cone and fiber identities quantify the remaining uncertainty, while matched observations reverse endpoint order for pooled moments, group-local moments, and reference-path curvature. Sequential composition adds one state variable: the slack from each group risk to the current maximum. This vector determines every unrestricted one-step response, and a margin condition keeps the active group fixed along paths with bounded relative drift. The experiments follow the same ladder. Across three dense LLMs, an early-preserving allocation reduces worst-group perplexity inflation by 12.6--20.9%; target-matched complete-menu selection improves over its references by 2.7--8.0%. Across all 16 routed layers of OLMoE, pooled endpoint refresh lowers held-out worst-group teacher KL by 15.8% over the best static score. A compute-matched hard-max trajectory ends 32.7% worse than pooled, and neither adaptive trajectory improves excess NLL. Local evidence can narrow a menu. Complete endpoints rank that menu, while multistep claims also require control of the evolving active face and future candidates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05926v1">BALANCE: Hybrid Autoregressive-Speculative LLM Inference in Wireless Edge Networks</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Edge inference is a promising paradigm to provide large language model (LLM) inference services in next-generation mobile networks. LLM inference mainly relies on two approaches: Autoregressive decoding (AD) generates output tokens sequentially, resulting in long latency; Speculative decoding (SD) accelerates inference by using a small language model (SLM) to generate multiple draft tokens for LLM verification, but incurs extra memory costs. Due to this latency-memory tradeoff, neither approach alone can efficiently serve users with heterogeneous demands under limited edge computing resources. To address this challenge, we propose a hybrid autoregressive-speculative inference (BALANCE) framework for edge LLM inference. In BALANCE, an edge server hosts both an SLM and an LLM, assigns each user to AD or SD, and performs the two modes simultaneously. To maximize the number of served users, we formulate a task throughput maximization problem to jointly determine user scheduling and computing resource allocation between AD and SD under user latency requirements and server memory constraints. Since the problem is NP-hard, we develop a polynomial-time algorithm that transforms the original problem into two sub-problems and obtains a sub-optimal solution with a constant approximation guarantee. Experiments demonstrate that BALANCE consistently outperforms conventional AD and SD and significantly improves task throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.21642v2">CARE: Pre-Execution Command Verification for Shell-Executing LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted by ISSRE 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly used for coding and terminal automation, making shell-command dispatch a high-stakes runtime control point. We study command-level pre-execution mediation for individual shell commands produced by LLM agents under bounded path context. Existing safeguards remain limited: generic guardrails do not model shell structure in sufficient detail, always-on LLM judges are relatively costly and variable, and shell parsers do not directly prevent harmful execution. We present CARE (Canonicalization, Attribution, and Resolution Engine), a shell-specific, static-first verifier for individual shell commands before execution. CARE canonicalizes generated commands into stable verification targets, derives deterministic evidence over syntax, command semantics, path context, and provenance-backed risk patterns, and escalates only underdetermined cases to an LLM judge. This design keeps the common case fast, reproducible, and auditable while reserving neural adjudication for borderline commands. On the balanced main split, CARE reaches 85.64% F1 with a 0.91% false-positive rate at 2.32 ms mean latency. When deployed in its static enforcement profile, CARE retains 84.99% F1 at 0.34 ms and reduces realised harm on RedCode-gen to 37.33%. Across external-generalization tests and controlled Docker-sandbox execution, these profiles expose a practical trade-off between benign recovery, false-positive burden, latency, and harm reduction. Overall, command-level shell mediation can reduce dispatch-boundary risk for LLM agents while preserving most benign workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05886v1">CodeGrep: An RL-Trained Retrieval Agent for LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Modern LLM coding agents such as Claude Code and OpenHands share a common inefficiency: they spend much of their token budget finding the file to patch, rather than patching it. On SWE-Bench Verified, a 30B OpenHands agent averages 23 rounds and 631K tokens per resolved issue, with many calls spent on grep, glob, and view_file during repository exploration. We introduce CodeGrep, a 14B retrieval agent trained end-to-end with GRPO to issue multi-turn parallel grep, glob, and read tool calls and return candidate files to a frozen downstream coding agent. On all 500 SWE-Bench Verified instances, CodeGrep preserves resolve rate while substantially improving efficiency: 27.0% versus 25.8% for the no-retrieval baseline, with 15% fewer rounds and 19% fewer tokens on resolved instances. Across retrievers, downstream utility follows a precision threshold: BM25 with precision 0.375 degrades the agent, Jina with precision 0.445 is neutral, and CodeGrep with precision 0.677 crosses the threshold at which retrieval begins to reduce rollout cost. To enable this study, we mine supervision from 67K open-source agent trajectories using CATM and build a Git-worktree environment for multi-turn agent RL. In our setting, applying the efficiency signal at the advantage layer rather than the reward layer reduces KL drift and translates cleanly into downstream efficiency. We will release the model, training pipeline, RL environment, and evaluation harnesses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09443v3">LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and modularly extensible framework designed to systematically evaluate the robustness of LLM-as-a-Judge across task datasets, judge prompt templates, judge models, attacks, and defenses. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 8 defense strategies across 13 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three main findings: (1) LLM-based judges remain susceptible under both pointwise and pairwise protocols. The combined Attack exhibits the strongest and most transferable performance among those evaluated, while the evaluated defenses exhibit no universal dominance across robustness, benign utility, and computational cost. (2) Robustness is highly sensitive to both prompt-template and judge model choices. (3) Optimization-based attack combined with long suffixes can substantially inflate scores returned by both PAI-Judge variants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05589v3">Invariant Representation Learning for Source-Free Time Series Forecasting with LLM-Centric Proxy Denoising</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted by ICML2026
    </div>
    <details class="paper-abstract">
      Effective time series forecasting enables various real-world applications, benefiting from the proliferation of mobile devices. However, the volume of time series data may vary significantly across domains due to high data acquisition costs and data regulations. To maximally create value from sparse data, this study focuses on a new problem of source-free time series forecasting, aiming to adapt a pretrained model from sufficient source time series to the sparse target time series without access to the source data, enabling data protection. To achieve this, we propose TimeID, a novel source-free time series forecasting framework with a large language model (LLM) centric proxy denoising inspired by the powerful generalization capabilities of LLMs. Specifically, TimeID consists of three key components: (1) dual-branch invariant disentangled feature learning that enforces representation- and gradient-wise invariance by means of season-trend decomposition; (2) lightweight, parameter-free proxy denoising that dynamically calibrates systematic biases of LLMs; and (3) knowledge distillation that bidirectionally aligns the denoised prediction and the original target prediction. Extensive experiments on real-world datasets demonstrate that TimeID outperforms state-of-the-art baselines, improving MSE and MAE by 10.7% and 9.3% on average. The code is available at https://github.com/decisionintelligence/TimeID.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05832v1">Enhancing Social Intelligence in LLMs with Hierarchical Reasoning and Utterance-Level Goal Rewarding</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in structured tasks but struggle with dynamic social interactions, where success requires long-term goal coordination and rapid adaptation. Current methods often apply uniform goal-based rewards to every utterance, overlooking the specificity of objectives at each dialogue turn and failing to account for the rationale of potential strategies. Inspired by the Theory of Planned Behavior, we propose the Think-Strategy-Response (TSR) framework, which decomposes social dialogue into two hierarchical stages: high-level strategic planning and low-level linguistic execution. To optimize TSR, we introduce Linearized Hierarchical Reinforcement Learning with Variance-Gated Rewards (LHRL-VGR), a novel algorithm that dynamically routes rewards - balancing goal completion and strategy adherence - based on the variance of goal achievement scores. Experiments on the SOTOPIA benchmark show that our approach fine-tunes a Qwen2.5-7B agent to surpass the GPT-4o baseline by 7.32% in goal completion success, demonstrating state-of-the-art performance in multi-agent social negotiation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15536v2">'OpenBloom': A Stigma-Sensitive LLM Design Probe for Navigating Reproductive Well-being Conversations with Young Adults</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      The growing use of large language models (LLMs) by young adults seeking sensitive health information has raised important questions in Human-AI Interaction about how these systems can support understanding and navigation of reproductive well-being. In response to Feminist HCI principles, we introduce OpenBloom, a web application and an exploratory design probe that uses LLMs to generate question-based prompts from reproductive health articles. Through a user study with 34 young adults across 136 interactions with OpenBloom, we provide an initial assessment of the system while exploring how participants' reflections engage with culture and value sensitivities. We found that while OpenBloom outputs meet expectations of "safe" and non-offensive, they tend to paraphrase or rely on factual recall, which may lead to value dilution. We discuss implications under contestability and value-sensitive frameworks for future LLM-mediated reproductive health technologies and towards responsible AI discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17124v2">Auditing Sex/Gender Disparities in Emergency Triage with LLM-based Paired Comparisons</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted for publication in npj Digital Medicine. This version incorporates revisions made during peer review. In Press
    </div>
    <details class="paper-abstract">
      We present a domain-agnostic paired-comparison approach that uses Large Language Models (LLMs) to quantify sex/gender-related asymmetries in documented clinical decision-making. The method trains an LLM to emulate observed decisions, then evaluates sex-swapped pairs in which only sex is flipped, holding documented clinical content constant. We apply it to emergency triage, analyzing more than 140,000 Bordeaux University Hospital (France) admissions and testing methodological portability on MIMIC-IV, spanning a different language, population, and healthcare system. Fine-tuning Mistral NeMo 12B for triage prediction and using Mistral Small 24B for pair generation, we find otherwise identical presentations were more likely to receive a lower-severity predicted score as female than male: 1.1% (95% CI 0.9-1.3) in the French cohort, 2.2% (1.7-2.7) in MIMIC-IV. Predictions are sensitive to both tabular and textual sex markers, with the asymmetry emerging primarily in the combined bimodal setting. A model retrained on sex-neutralized inputs eliminated the between-sex prediction gap, indicating the asymmetry is mediated by explicit sex markers. Patterns vary with nurse-patient sex concordance, suggesting the model captures stable features of the recorded data rather than random artifacts. These effects are small and documentation-level. We therefore present this as a methodological feasibility study: LLMs can serve as scalable probes of documented decisions, generating hypotheses rather than establishing bedside clinician behavior or clinically meaningful undertriage, which would require clinician-anchored validation. Beyond emergency care, the approach supports bias audits in other domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05192v2">LELA: an LLM-based Entity Linking Approach with Zero-Shot Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted at ISWC 2026. Extended version with appendices
    </div>
    <details class="paper-abstract">
      Entity linking (mapping ambiguous mentions in text to entities in a knowledge base) is a foundational step in tasks such as knowledge graph construction, question-answering, and information extraction. Our method, LELA, is a modular coarse-to-fine approach that leverages the capabilities of large language models (LLMs), and works with different target domains, knowledge bases and LLMs, without any fine-tuning phase. Our experiments across various entity linking settings show that LELA is highly competitive with fine-tuned approaches, and substantially outperforms the non-fine-tuned ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05791v1">A Two-Tier Perspective on Inference-Time Parallelism in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-driven multi-agent systems typically require multiple model invocations and complex coordination during inference, and their execution strategies directly affect system accuracy, latency, and computational cost. Parallel execution provides a means to improve inference-time efficiency. From the perspective of inference-time execution, this paper models parallelism in multi-agent systems as two distinct levels of decision processes: Replica Parallelism, which explores multiple complete solution paths at the task level, and Structural Parallelism, which enables concurrent execution within a single solution path through task decomposition. However, the roles of different forms of parallelism and their interrelationships still lack systematic study in terms of unified organization and coordination. We therefore propose TIPEX, a controllable execution framework that unifies these two levels of parallelism and coordinates their roles within the inference process under a unified execution semantics while supporting systematic combinations and analyses of different parallel strategies and parameter configurations. Systematic experiments on the GAIA benchmark demonstrate that inference-time parallelism can significantly improve accuracy and reduce end-to-end latency at the cost of increased token consumption. Further analysis shows that Replica and Structural Parallelism exhibit complementary effects across task complexities, with tasks of intermediate difficulty benefiting most from their coordination, while overly aggressive parallel strategies do not necessarily yield better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05759v1">How to Recognize New Words: A Comparison Between Context Biasing Methods and Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Recognizing new and rare words - named entities, acronyms, domain specific special words, and other items scarce in training data - remains a key challenge for automatic speech recognition (ASR). We compare two strategies for this: context biasing methods, where an ASR model is extended such that during inference a word list can be supplied, and speech large language models (LLMs) prompted with context directly. We evaluate two context biasing methods based on Whisper against three speech LLMs across read and non-read speech, reporting biased, unbiased, and overall word error rate (WER). The context biasing methods cut biased WER by up to 88% relative while leaving other words largely unaffected. Speech LLMs excel on read speech but generalize less well to non-read speech, and prove sensitive to distractor count and prompt word order. We characterize the resulting trade-offs to guide method selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05741v1">Once a Response, Always a Response: Detecting LLM-generated Text via Latent Prompt Restoration</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate fluent and convincing text at scale, creating growing risks for misinformation dissemination, educational misuse, and platform governance. These concerns make robust detection of machine-generated text increasingly necessary. Recent zero-shot detectors mainly exploit probability-based statistical discrepancies, but they do not explicitly account for the training process of LLMs, which leaves a distinct generation mechanism insufficiently modeled and limits detection robustness. To address this issue, we propose EchoPrompt, a training-free detector based on latent prompt restoration. Our key intuition is that machine-generated text is typically produced conditioned on an upstream prompt, and this hidden dependency can be partially reactivated by prepending a unified generic prefix. Specifically, EchoPrompt restores a generic assistant-response context, measures the induced likelihood gain with an instruction-tuned model, calibrates it against the corresponding base model, and aggregates the resulting differences into a score that quantifies latent prompt dependency. Extensive experiments show that EchoPrompt achieves state-of-the-art performance among zero-shot detectors while maintaining strong robustness across challenging evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.15557v5">SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Agent skills, SKILL files that package reusable procedural knowledge for an LLM agent, are a popular mechanism for extending agent capabilities. Public repositories now host them in large and growing numbers, yet these artifacts are fragmented, redundant, and uneven in quality, and their value in practice is unclear. A core question remains open, namely how to consolidate this open-source SKILL ecosystem into a single usable corpus, and what bounds its benefit on real-world agent tasks. We present SkillCorpus, a framework that aggregates, curates, matches, and evaluates the open skill ecosystem at scale. It filters ~821,000 crawled skills through a multi-stage pipeline into 96,401 skills organised by a 16-class taxonomy and three quality facets (utility, robustness, safety), and pairs them with a fine-tuned retrieval-and-selection stack that matches task-relevant skills. We evaluate end-to-end across three benchmarks (SkillsBench, GDPVal, QwenClawBench), two harnesses, and two open backbones with a frontier robustness check. Integrating SkillCorpus yields consistent gains across all three benchmarks, largest on SkillsBench (+7.5 pp). An operational analysis traces the gains to a coverage boundary and a harness boundary. SkillCorpus is, to our knowledge, the first end-to-end account of when a curated, retrieval-served community corpus improves real agent tasks, and where it does not. The dataset, models, and code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05726v1">Mitigating Scoring Bias in LLM-as-a-Judge via Random Number Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often used as evaluators of text quality, known as LLM-as-a-Judge, which can outperform conventional automatic evaluation metrics that rely on reference texts. However, LLM evaluators tend to generate particular scores regardless of the context of the evaluated text, which is known as scoring bias. This study proposes a novel method to mitigate this scoring bias. An LLM is instructed to randomly generate number tokens, and the latent numerical bias of the LLM is identified by measuring the deviation of the observed distribution of numbers from the uniform distribution. A definition of a downstream task, for which an LLM evaluator is used, is added to the prompts for random number generation to measure task-specific latent number bias. In the evaluation by an LLM, the token generation probabilities for a given input are rectified considering the LLM's latent number bias. Results of the experiment on four different tasks, evaluation of LLM alignment, evaluation of summarization, Semantic Textual Similarity, and Semantic Textual Relatedness, demonstrate that our proposed method outperforms the baselines, including an LLM without debiasing and previous calibration methods. In addition, it is confirmed that scoring bias varies across LLMs, tasks, and score ranges, indicating the importance of measuring latent number bias as the case may be.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05695v1">DreamGuard: Efficient Runtime Guardrail for LLM Agents via Risk-Aware World Model</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      As large language model (LLM) agents increasingly invoke external tools and interact with real-world systems, unsafe actions may cause irreversible consequences on external states, user data, and downstream services. Recent runtime guardrails mitigate such risks by checking proposed actions before execution, but many remain reactive: they primarily assess the apparent safety of the current action, lacking an explicit model of how risk evolves across the trajectory. This limitation creates a critical blind spot for long-horizon risks, where individually benign-looking actions can gradually drift the agent toward hazardous states. In response, we propose DreamGuard, a proactive guardrail for LLM agents built around a risk-aware world model. The world model maintains a compact recurrent latent state over the trajectory and predicts future latent states from which DreamGuard derives immediate-hazard and prefix-risk evidence. It then fuses these multi-horizon signals into intervention decisions before execution. Experiments across four benchmarks and an online guardrail evaluation show that DreamGuard outperforms generic, reactive, and proactive guardrail baselines, achieves the best safety-utility trade-off among evaluated guardrails, and maintains an average end-to-end latency of 25 ms per call.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13105v2">Knowledge-Driven Hybrid SSD Management Enhanced by Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 DAC 2026
    </div>
    <details class="paper-abstract">
      A hybrid Solid-State Drives (SSDs) integrates different modes of flash cells (e.g., single-level cell (SLC) and Quad-Level Cell (QLC)) and enables them to convert between each other, achieving both high performance and storage capacity. However, this hybrid design introduces a significantly larger design space than traditional SSDs with additional design factors such as flash conversion and data migration across different flash modes, leading to higher optimization complexity. Efficient management of such complexity requires deep hybrid SSD knowledge and dynamic adjustment mechanisms. Large language models (LLMs) offer a promising solution through their contextual reasoning and adaptive coordination capabilities. In this work, we explore the potential of using LLMs in understanding and efficiently managing hybrid SSD design space. We find that leveraging LLMs for knowledge-guided optimization of management parameters enables substantial performance gains. Building on these insights, we propose LLM-hybridSSD, an integrated optimization framework that formulates hybrid SSD management as a parameter-tuning problem, employs an LLM-based tuner for adaptive configuration, and applies reinforcement learning-based fine-tuning to align local lightweight models with domain-specific knowledge. Experimental results show an average 58.92% increase in throughput and a 28.56\% reduction in write amplification (WA) compared with state-of-the-art schemes under different real-world workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05659v1">Breaking Customized LLMs for Coding: Automated Red Teaming for Instruction Backdoor Attacks</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted to the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026
    </div>
    <details class="paper-abstract">
      LLM customization platforms allow users to build task-specific models for code intelligence tasks by embedding instructions into system prompts, without modifying the underlying model parameters. While these platforms lower the barrier to developing customized LLMs, they also introduce a new attack surface: instruction backdoor attacks, in which adversaries implant hidden malicious behaviors into customized instructions. However, existing attacks suffer from two key limitations. First, they often rely on explicit trigger patterns readily detected by platform-side or user-side inspection. Second, they require substantial manual effort to craft task-specific backdoored instructions, limiting their scalability. In this paper, we propose ARIA, an automated red-teaming framework for crafting covert and effective backdoored instructions against customized LLMs. ARIA leverages an attacker LLM to iteratively generate and refine backdoored instructions, guided by structured feedback from the target LLM along three dimensions: stealthiness, clean-task utility, and backdoor effectiveness. We evaluate ARIA on three code intelligence tasks, using four representative LLMs, and compare it with three baseline attacks. Experimental results show that ARIA achieves the highest attack success rate of 0.945, while maintaining the best clean-task utility across all tasks. ARIA also generalizes well across programming languages and remains robust to generation temperature. Furthermore, ARIA significantly outperforms existing attacks in evading platform-side and user-side detection, achieving a false negative rate of up to 1.000, and stays effective against existing defense methods, demonstrating its strong generalizability and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05651v1">Relay, Don't Route: Adaptive Population Handoff for Cost-Efficient LLM-Driven Evolution</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-driven evolution has shown promise for program search and algorithm discovery, but relying on strong models throughout long evolutionary runs is costly. A natural alternative is to combine cheap and strong models under a fixed inference budget. However, existing approaches typically allocate models at the level of individual queries or mutation steps, overlooking that evolutionary search is \textit{stateful}: each generated candidate changes the population from which subsequent mutations are produced. We empirically analyze LLM-driven evolutionary trajectories and find that search progress is strongly front-loaded, early trajectory performance is informative but noisy, and cheap models recover much of the early progress achieved by strong models at lower cost. Motivated by these findings, we propose \textbf{\model}, a training-free framework that shifts budget allocation from individual calls to evolving populations through adaptive \textit{population handoff}. A cheap model explores multiple trajectories in short blocks allocated by a bandit scheduler. Relay Gain, defined as the marginal improvement of a compact, quality-diverse candidate bank constructed for handoff, serves as the scheduler reward and determines when to hand off. The curated candidates initialize a shared strong model population for refinement. Across four benchmarks and three budgets, \model achieves the highest mean score in 11 of 12 settings, outperforming competitive baselines. Our results suggest that in stateful search, budget allocation should be organized around the population, not the individual call.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18056v2">An Early Warning of Emerging Biosecurity Risks in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 22 pages, 7 figures, authors are listed alphabetically by surname; update Figure 7 on page 15 due to arXiv format requirements
    </div>
    <details class="paper-abstract">
      Frontier large language models (LLMs) are increasingly integrated into scientific workflows, yet their growing biological capabilities may outpace current safeguards. To assess the biological risks of frontier models, we develop Intern-BioBreaker, a specialized bio-red-teaming model, together with an integrated computational-to-physical framework that couples model-level stress testing with wet-lab validation. Within this framework, Intern-BioBreaker generates targeted jailbreak prompts to test whether aligned models can be induced to provide operational guidance for safety-sensitive biological tasks or produce sequence-level outputs with potentially harmful properties. Selected sequence outputs are then carried forward for DNA synthesis, host expression, and orthogonal protein verification to assess whether model-generated designs can yield the intended biological products. Our evaluation reveals a concerning gap between text-level safeguards and the risks posed by capable scientific models: (i) Intern-BioBreaker outperforms baseline attack models and reveals widespread bio-risk jailbreak vulnerabilities across both open-weight and proprietary frontier LLMs, with several targets reaching near-saturated or 100% task-level attack success rate (ASR); (ii) in sequence-level case studies, GPT-5.5 can be induced to generate modified viral candidate sequences with pathogenic potential; the corresponding translated proteins may exhibit even stronger receptor-binding affinity and thus enhanced infection potential; and (iii) end-to-end verification shows that selected model-generated biological designs are not merely textual artifacts, but can be physically realized under controlled experimental settings. These findings underscore the need for stronger biological red-teaming, nucleic acid synthesis screening, and safety mechanisms that keep pace with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05643v1">Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Submitted to EMNLP 2026
    </div>
    <details class="paper-abstract">
      Test-time scaling improves LLM reasoning by using additional inference compute, but wider sampling alone can suffer from diminishing returns: new rollouts often repeat existing answer patterns instead of adding useful reasoning diversity. Verifier-based selection offers an alternative, but its performance depends on the calibration of an external reward model. We propose a verifier-free breadth--depth refinement framework that uses test-time compute to both explore and improve candidate solutions. The method samples multiple independent reasoning rollouts, refines each rollout through iterative self-critique and self-correction, and aggregates the refined answers by majority voting. Breadth preserves diverse initial attempts, while depth repairs local reasoning errors before aggregation. Across AIME24, AIME25, AMC, OlympiadBench, and MATH500, our method consistently improves over greedy decoding, majority voting, verifier-based best-of-$N$, beam search, and lookahead decoding across multiple open-weight models. For instance, with Qwen2.5-1.5B, accuracy increases from the strongest verifier-based baseline to $58.0\%$ on MATH500, and from $25.0\%$ to $32.5\%$ on AMC. These results show that test-time compute can be more effective when used to refine sampled trajectories rather than only to sample more candidates or rely on verifier-guided selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05611v1">FOCUS: Decoupling Expert Personas in LLMs to Enhance Domain Expert Capabilities</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can exhibit diverse personas, and activating expert personas has been shown to improve domain expertise and task accuracy. However, existing persona control methods often suffer from cross-domain coupling, which may lead to overly aggressive behavior in high-caution domains such as healthcare, or excessive conservatism in risk-sensitive domains such as financial trading. To address this issue, we propose FOCUS (\textbf{\underline{F}}ine-tuning with \textbf{\underline{O}}rthogonal \textbf{\underline{C}}ontrol for \textbf{\underline{U}}ncoupled persona\textbf{\underline{S}}). FOCUS first automatically extracts expert persona vectors from LLMs, then applies orthogonal decomposition to decouple domain-specific expert personas, and finally introduces an expert gating module to adaptively control persona activation according to task contexts. With a two-stage training strategy and a gated selection regularizer, the model learns to activate appropriate personas for both single-domain and cross-domain tasks. Experiments on financial, legal, medical, and cross-domain benchmarks show that FOCUS improves task accuracy and outperforms existing persona control methods. Our code is available at \href{https://anonymous.4open.science/r/openpersona-48F4}{this url}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03421v2">When Truth Is Distributed: Misinformation Derails Collective Fact Recovery in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems promise effective collaborative reasoning, but communication may amplify local errors into collective risks. Existing evaluations emphasize final outcomes, leaving the reliability and propagation dynamics of distributed information aggregation unclear. We introduce Hi-Agreement, a controlled evaluation framework that strictly pairs all-honest collaboration with controlled deception by a key evidence holder and analyzes the aggregation process through multi-stage voting, testimony adoption, and evidence-root lineage propagation. Using 120 five-agent object-movement environments where partial observations jointly determine a unique endpoint, we evaluate 3 homogeneous LLM-based multi-agent systems. Across these paired conditions, aggregate truth recovery falls from 72.50% to 14.17%, with significant declines for every system. Process tracing and exit ablations show that a single false testimony is adopted more readily than truthful testimony, propagates to higher orders, and persists through honest agents after the deceiver exits. Observers without first-hand evidence suppress incorrect consensus but do not improve truth recovery. Together, these findings reveal both the fragility of distributed fact recovery and its underlying mechanism: false evidence gains collective influence through its adoption and continued propagation by other agents after entering communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05583v1">The Judgment-Consequence Gap: LLM Moral Reasoning in Healthcare Decisions</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted at AIES 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) enter high-stakes domains such as healthcare, understanding their moral reasoning becomes essential. Decisions about scarce medical resources often hinge on judgments of responsibility, particularly when patients' own actions contribute to illness. We investigate how LLMs reason about responsibility and its consequences, tracing their judgments across successive levels, from the behavior, to the resulting illness, to the denial of care. We evaluate a wide range of LLMs, spanning different model families and capability levels, on various clinical vignettes adapted from prior studies. Our results identify a judgment-consequence gap: LLMs largely agree with humans that patients bear responsibility for health-harming behaviors, yet overwhelmingly refuse to let that judgment influence how they allocate scarce resources. Specifically, LLMs default to random allocation, whereas humans consistently favor the less-culpable patient. Compared to humans, LLMs also place greater emphasis on access to information, reducing responsibility judgments when health-risk knowledge is unavailable. These findings reveal that LLMs apply a systematically different moral framework than humans when responsibility and resource scarcity intersect, surprisingly often amplifying normative disagreement with humans as reasoning capability increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.02647v2">From Guessing to Seeing: Enhancing LLM-Based Program Repair via Trace-Guided Multi-strategy Debate</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 13 pages, 4 figures, 10 tables. Accepted at the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) aims to resolve software bugs without human intervention, but complex logic errors and silent failures remain challenging. Existing LLM-based APR methods mainly rely on source code and coarse test feedback, making it difficult to capture runtime behaviors and dynamic data dependencies. Execution traces expose concrete state transitions, yet a single LLM interpreting them in isolation may commit to an incorrect repair hypothesis and produce test-overfitting patches. We therefore treat runtime evidence as shared constraints for validating repair hypotheses rather than merely as additional prompt context. We propose TraceRepair, a multi-agent framework in which a Probe Agent captures execution snapshots of critical variables, while specialized repair agents generate, compare, and iteratively refine candidate patches against the observed runtime evidence. A Judge Agent then arbitrates the remaining hypotheses and synthesizes the final patch. Evaluated on Defects4J, TraceRepair correctly fixes 392 defects and outperforms existing LLM-based approaches. Further experiments demonstrate improved efficiency and strong generalization on a newly constructed dataset of recent bugs, suggesting that the gains arise from dynamic reasoning rather than memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01707v3">Memory in the LLM Era: Modular Architectures and Strategies in a Unified Framework</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Memory emerges as the core module in the large language model (LLM)-based agents for long-horizon complex tasks (e.g., multi-turn dialogue, game playing, scientific discovery), where memory can enable knowledge accumulation, iterative reasoning and self-evolution. A number of memory methods have been proposed in the literature. However, these methods have not been systematically and comprehensively compared under the same experimental settings. In this paper, we first summarize a unified framework that covers existing representative agent memory methods from a high-level perspective. We then extensively compare representative agent memory methods on two long-term conversational benchmarks and an agentic memory benchmark, and examine the effectiveness of representative methods, providing a thorough analysis of those methods. As a byproduct of our experimental analysis, we also design a new memory method by exploiting modules in the existing methods, which outperforms the state-of-the-art methods. Finally, based on these findings, we offer promising future research opportunities. We believe that a deeper understanding of the behavior of existing methods can provide valuable new insights for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05997v2">MM-ISTS: Cooperating Irregularly Sampled Time Series Forecasting with Multimodal Vision-Text LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Irregularly sampled time series (ISTS) are widespread in real-world scenarios, exhibiting asynchronous observations on uneven time intervals across diverse variables. Existing ISTS forecasting methods often solely utilize historical observations to predict future ones while falling short in learning contextual semantics and fine-grained temporal patterns. To address these problems, we propose MM-ISTS, a multimodal ISTS forecasting framework augmented by vision-text large language models, which bridges temporal, visual, and textual modalities. MM-ISTS encompasses a two-stage encoding mechanism. In particular, a Cross-Modal Vision-Text Encoding module is proposed to automatically generate informative visual images and textual data, enabling the capture of intricate temporal patterns and comprehensive contextual understanding, in collaboration with multimodal LLMs (MLLMs). In parallel, ISTS encoding extracts complementary yet enriched temporal features from historical ISTS observations, including multi-view embedding fusion and a Temporal-Variable Encoder. Further, we propose an Adaptive Query-Based Feature Extractor to compress MLLM token embeddings while preserving useful knowledge, which in turn reduces computational costs. In addition, a Multimodal Alignment module with Modality-Aware Gating is designed to alleviate the modality gaps. Extensive experiments on real data offer insight into the effectiveness of the proposed solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03573v2">SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Code: https://github.com/GaryStack/Parallel-RL
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) exhibit fundamentally different behaviors in enhancing multi-task reasoning for large language models (LLMs). Our preliminary experiments revealed a phenomenon: SFT suffers from severe task conflicts under multi-stage training, whereas RL enables stable coexistence across diverse tasks. Empirically, we trace this to the parameter level, observing that RL induces sparse and approximately orthogonal updates across tasks. We provide a theoretical explanation for this mechanism by analyzing multi-task gradient interference. Our results reveal a distinction: interference in SFT is norm-limited, scaling with the absolute gradient magnitude, whereas interference in RL is variance-limited, bounded by the gradient variance induced by advantage normalization and on-policy optimization. This small variance bound yields near-orthogonal optimization directions across tasks. Leveraging this insight, we propose Parallel-RL, a paradigm that decouples multi-task training, significantly improving efficiency and flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05541v1">Hyper-ES: Effective Evolution Strategies for LLM Reasoning via Descent Direction Merging</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 19 pages, 4 figures, 14 tables. Code: https://github.com/kuangrepi/Hyper-ES
    </div>
    <details class="paper-abstract">
      Evolution Strategy (ES) is a promising alternative to gradient-based fine-tuning for resource-constrained Large Language Model (LLM) reasoning. However, directly applying ES to billion-parameter LLMs is highly ineffective. In such high-dimensional parameter spaces, most random perturbations are nearly orthogonal to useful update directions, leading to unstable optimization. We propose Hyper-ES, a subspace-based ES framework that avoids the weakness of ES in full-parameter search while exploiting its strength in low-dimensional optimization. Instead of asking ES to discover useful directions from random perturbations in the LLM parameter space, Hyper-ES first performs a small number of inexpensive gradient-based fine-tuning runs to obtain descent directions. Although each direction may provide only a limited improvement on its own, their span forms a compact adaptation subspace that captures useful reasoning updates. Hyper-ES then applies CMA-ES to optimize layer-wise DARE-TIES merging coefficients within this subspace, allowing ES to search over combinations of meaningful descent directions rather than over arbitrary full-model perturbations. We evaluate Hyper-ES on three Qwen2.5-Instruct and DeepSeek-R1-Distill backbones across six mathematical reasoning datasets. Results show that Hyper-ES consistently outperforms GRPO-LoRA by 1% while requiring 10% fewer space-consuming gradient updates. Code at https://github.com/kuangrepi/Hyper-ES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05519v1">EcoAgent-Bench: Evaluating Economic Decision-Making in Budget-Constrained LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 8 pages, 3 figures, 4 tables. Benchmark, dataset (304 budget-conditioned agent tasks), and evaluation harness; artifacts to be released
    </div>
    <details class="paper-abstract">
      Agent benchmarks usually measure task completion and treat resource use as an auxiliary statistic. In deployment, however, the choice among a local lookup, broad search, composite research tool, stronger model, or human escalation is part of the task itself. We introduce EcoAgent-Bench, in which every task specifies priced actions and an explicit budget. Its 304 real-derived tasks span five families adapted from GAIA, HotpotQA, and MuSiQue, and test four decisions: avoiding unnecessary escalation, escalating when local evidence is insufficient, selecting a model tier, and stopping on unsupported premises. We evaluate seven LLM agents in tool-API and workspace-CLI settings, together with four oracle scripted controls. Micro-averaged accuracy rewards one-sided policies: always-escalate controls achieve high micro success while failing save-oriented tasks. We therefore also report an economic-consistency score (the worse of accuracy on upgrade-oriented and save-oriented family groups) which exposes this failure. Tool-API agents attain only 3.9-24.0% micro strict success (at most 7.3% economic consistency), often either stopping before warranted escalation or overspending on cheap tasks. A threshold-crossing budget sweep changes GPT-5.4's escalation rate from 0% to only 3%. These results show that completion under a budget and economical action selection are distinct properties. We release the task bundle, transformation pipeline, frozen evaluation environments, and integrity-bound result artifacts needed to study both.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01791v3">PICopilot: An LLM-based Agentic Framework for Assisting Photonic Integrated Circuit Design via Script Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The rapid development of photonic integrated circuits (PICs) is shifting the design flow from traditional graphical user interface (GUI)-based methods to script-based methods for higher flexibility, portability, and maintainability. However, script-based design introduces new challenges, requiring designers to possess additional proficiency in tool application programming interfaces (APIs) and programming. It also demands greater effort and time because it is inherently less intuitive and more complex than GUI-based methods. As PICs grow in scale and complexity, the productivity gap between design needs and manual scripting capabilities continues to widen. To address this gap, we introduce PICopilot, the first large language model (LLM)-based agentic framework that assists in PIC design via automated design script generation from natural language instructions. PICopilot leverages a multi-agent architecture with a feedback mechanism and a specifically designed retrieval-augmented generation (RAG) pipeline, achieving a high success rate and reliability. Experimental results on a benchmark of diverse PIC scripting tasks demonstrate that PICopilot successfully completes all 48 tasks and outperforms other LLM-based approaches without incurring substantial extra latency or cost, even solving 21 more tasks than the advanced GPT-5 model with a general RAG pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02687v3">PolicyGuard: Prompt-Configurable Semantic DLP for LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 This paper is being withdrawn because it was submitted prior to completion of a required institutional review process. The authors intend to resubmit after the review is complete
    </div>
    <details class="paper-abstract">
      AI coding agents accept free-form natural language prompts that may inadvertently contain credentials, personally identifiable information (PII), or proprietary business data. Existing data loss prevention (DLP) solutions rely on rigid regex patterns, model fine-tuning, or vendor-managed classifiers with limited customizability. We present PolicyGuard, a pre-model interception framework that classifies user prompts using an LLM guided by a natural language policy file. Our key contributions are: (1) the policy-as-prompt paradigm, where DLP classification criteria are defined entirely in a plaintext policy document editable by non-engineers without code changes or model retraining; (2) a sealed evaluation protocol with template-family-level data splits, hidden holdouts, and frozen test sets to rigorously assess generalization; and (3) a comprehensive empirical evaluation across 2,000 multilingual prompts demonstrating 96.5% effective block rate (EBR) with only 3.0% false positive rate (FPR) on a frozen test set of 927 prompts, and perfect 100% accuracy on a 217-prompt hidden holdout. Information-matched baseline experiments show that PolicyGuard's natural language format significantly outperforms equivalent content in JSON format (McNemar chi-squared = 31.58, p < 0.001) and dramatically outperforms zero-shot classification (Cohen's h = 0.915). Cross-model portability experiments demonstrate that the same policy achieves 86.4-96.5% EBR across four different LLMs without modification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04872v2">A-SR: Self-Evolving Agentic LLMs for Symbolic Regression via Hierarchical Coordination</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 18 pages, 8 figures, including appendix
    </div>
    <details class="paper-abstract">
      Symbolic regression aims to discover closed-form equations from data, but existing LLM-guided methods often rely on a unified proposal loop that compresses heterogeneous search failures into a scalar score and a single prompt. We propose A-SR, a self-evolving agentic framework that shifts the control unit from expression edits to role-conditioned evidence views. A-SR coordinates formula discovery through routing among coordination protocols, an online evaluator-reward role policy, and state-routed process memory. During search, evaluator feedback characterizes reliability and productivity, updates role-level utilities, and routes elite motifs, failure traces, and validity diagnostics to different agents. The framework self-evolves at two timescales: within a run, it adapts the search process without updating LLM parameters; across runs, recorded trajectories can be distilled into open-source LLMs as role-conditioned proposal priors. Averaged over the four LSR-Synth scientific domains in LLM-SRBench, A-SR improves Acc@0.01 over baselines from 25.79% to 48.30% with Llama3.1-8B, while A-SR-LoRA improves the corresponding Qwen3-4B result from 24.58% to 38.29%. On four real-world scientific discovery tasks, A-SR obtains the best in-distribution or out-of-distribution normalized mean squared error on 7 of 8 reported metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04333v1">Cost-Aware Multi-Objective Bandits: Theory and Application to Budgeted LLM Configuration Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM) configuration evaluation is challenging due to limited evaluation budgets, varying costs, and multiple competing objectives. In this paper, we formulate LLM configuration evaluation as a cost-aware multi-objective bandit problem, where each configuration evaluation incurs a configuration-dependent cost and yields a noisy vector-valued outcome. Under this framework, we study two fundamental problems: online configuration selection and Pareto configuration identification. For online configuration selection, we propose a hypervolume-based UCB algorithm that optimizes an optimistic hypervolume-per-cost index. We establish a budgeted regret bound of order $O\bigl(\sum_{i\ne i^\star}\frac{\log B}{Δ_i}\bigr)$, where $B$ is the evaluation budget, $i^\star$ is the optimal configuration in terms of hypervolume efficiency, and $Δ_i$ is the corresponding efficiency gap of configuration $i$. This bound retains the logarithmic budget dependence of classical single-objective budgeted bandits. For fixed-budget Pareto identification, we develop a cost-aware empirical gap elimination algorithm and prove that its error probability is of order $O\bigl(\exp(-\frac{B}{H_{μ,c}})\bigr)$, where $H_{μ,c}$ is a cost-aware Pareto identification complexity depending on configuration costs and Pareto classification gaps. This error probability decays exponentially with the evaluation budget and recovers the standard Pareto set identification guarantee when all configuration costs are identical. Experiments on LLM configuration evaluation tasks demonstrate that the proposed framework enables efficient online decision-making and accurate cost-aware Pareto identification under limited budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.23933v2">SpecBox: Speculative Sandbox Scheduling for Efficient LLM Agent Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      As LLM agents increasingly rely on the Model Context Protocol (MCP) to invoke isolated external sandboxes, disaggregated sandbox deployment introduces a fundamental tension between resource utilization and interactive tail latency. Persistent long-lived sandbox reservations incur excessive memory overhead at scale, while lazy on-demand instantiation generates severe cold-start penalties that degrade response performance under multi-tenant, multi-turn agent workloads. To resolve this dilemma, we present SpecBox, a runtime built around speculative sandbox preallocation tailored for dynamic LLM agent execution pipelines. At its core, SpecBox implements keyword matching and streaming semantic embedding to enable intent-driven sandbox prewarming, which identifies pending tool execution demands mid-LLM token generation and fully overlaps sandbox bootstrapping with model inference. To extend prewarming windows across sequential agent steps, the framework leverages context-aware stochastic prefetching atop a sandbox dependency graph to probabilistically forecast future sandbox switches ahead of execution. We complement these speculative mechanisms with two orthogonal optimizations: a semantic result cache that prunes redundant repeated sandbox invocations, and a dedicated out-of-band shared-memory transport plane that bypasses conventional network serialization to deliver zero-copy artifact transfers. Evaluated on high-concurrency multi-turn agent traces, our prototype demonstrates that SpecBox cuts P99 end-to-end latency by up to $2.9\times$ relative to the on-demand sandbox baseline, while slashing peak memory consumption by $45.9\%$ compared to permanently reserved sandbox deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04309v1">Structured LLM Reasoning for Zero-Shot Human--Robot Coordination Under Hidden Goals</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      We present a structured large-language-model (LLM) architecture for zero-shot human--robot coordination in a cooperative construction task with private goal views. Guided by a Dec-POMDP formulation, the architecture decomposes decision-making into (i) action-conditioned Theory-of-Mind (ToM) inference, (ii) hierarchical planning, (iii) conversation interpretation, (iv) action verification, and (v) feedback-based replanning. We compare the proposed method with an ablation without ToM inference and a multi-agent reinforcement-learning policy trained offline over many goal pairs. In human-participant experiments, the proposed method required fewer interaction steps and yielded higher post-interaction trust ratings than both baselines. These results suggest that systematically decomposing the team decision problem, using LLMs as tractable surrogates for otherwise intractable inference and planning computations, and retaining conventional verification for physical feasibility can improve both task coordination and the human experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04307v1">MIDAS: Multi-LLM Iterative Data-Adaptive Summarization</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted at the 20th International Conference on Document Analysis and Recognition (ICDAR 2026). 17 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Text summarization is deceptively difficult. While condensing information seems straightforward, real-world enterprise summarization of support tickets, legal documents, incident reports, and more, demands strict adherence to domain-specific guidelines, output formats, and organizational conventions. Crafting prompts that reliably satisfy these constraints is labor-intensive, requiring significant human expertise and continuous maintenance as requirements evolve. Existing automated prompt optimization methods reduce this burden through Large Language Model (LLM) critique-driven refinement, yet remain limited by static prompts that cannot adapt to the diversity of summary applications. We propose Multi-LLM Iterative Data-Adaptive Summarization (MIDAS), a multi-LLM framework that extends this paradigm with data-driven pattern learning and use-case-specific personalization, enabling automatic adaptation to different summarization requirements without manual prompt engineering. Applied to enterprise customer ticket summarization across five output formats, MIDAS achieves the strongest overall performance against state-of-the-art critique-driven optimization frameworks such as CriSPO and ZERA, improving ROUGE-1 by up to 11.0%, ROUGE-2 by up to 18.2%, and ROUGE-L by up to 8.0%, while consistently improving BERTScore F1 across all formats and output types. We additionally demonstrate cross-model and cross-domain generalization through multi-LLM configurations and finance-domain summarization benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.18141v3">A CXL Memory Rack for Multi-Turn LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Long-context, multi-turn, and agentic LLM workloads increasingly reuse previously processed context, making KV-cache reuse essential for reducing redundant computation. However, this reuse shifts the bottleneck to the memory tier that stores and serves reusable KV states at cluster scale. GPU HBM and host DRAM are too costly to scale to TB-scale shared context capacity, motivating remote tiers built from lower-cost, higher-capacity media. This paper presents HyMCache, a CXL memory rack for multi-turn LLM serving. We build the memory rack using cost-efficient CXL-hybrid memory (CXL-HM), which combines a small amount of in-device DRAM with large SSD-backed capacity behind a CXL interface. By exploiting the read-dominant, predictable, and append-only nature of multi-turn KV-cache access, HyMCache rethinks DRAM management within CXL-HM to efficiently support TB-scale SSD-backed KV reuse. It uses request-level prefix prefetching and opportunistic write buffering to stage latency-critical reads in device DRAM, enabling DRAM-scale KV-cache efficiency at SSD-level cost. We evaluate HyMCache on a real CXL-HM prototype under both single-aggregator and PD-disaggregated serving configurations. Under the same DRAM budget, HyMCache outperforms local LMCache by 3.0x in single-node serving and 1.45x in PD-disaggregated serving. Compared with 1 TB distributed-DRAM Mooncake, HyMCache incurs about 30% lower performance but uses 16x less DRAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05446v1">EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted to LLA@COLM 2026
    </div>
    <details class="paper-abstract">
      Long-horizon LLM agents increasingly rely on external execution support to maintain state, track progress, invoke tools, verify outcomes, and reuse experience across interactions. However, effective harness use raises two coupled challenges: state formation from noisy interaction traces and runtime control over external-state access. Existing agents usually handle both through prompts, heuristics, or domain-specific conventions, leaving the external workspace and its usage policy manually engineered. To address this, we study the problem of harness policy learning, where agents learn harness policies offline and deploy them to construct and update external harness state online during runtime task execution. We introduce EvoHarness-RL, which exposes Belief, Progress, and Experience (BPE) as policy-facing harness state. Supervised harness fine-tuning teaches the base agent the harness action space and how to construct useful external state, while cost-aware GRPO explores coordination policies to selectively read, update, and consolidate that state during long-horizon interaction. Instantiated on ALFWorld with a Qwen3-8B LLM, EvoHarness-RL reaches 96.9% success and reveals two key dynamics: harness annealing, where training internalizes recurring harness-use patterns into the model policy and shifts the agent from frequent harness calls toward selective external-state access, and harness evolution, where progress updates and experience consolidation refine the harness into a compact, task-adaptive state substrate. These results suggest that long-horizon agents benefit from trainable policies for constructing and coordinating with external harness workspaces, beyond simply adding stronger tools or larger memories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00059v3">Stochastic Parrots or Singing in Harmony? Testing Five Leading LLMs for their Ability to Replicate a Human Survey with Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 V2
    </div>
    <details class="paper-abstract">
      How well can AI-derived synthetic research data replicate the responses of human participants? An emerging literature has begun to engage with this question, which carries deep implications for organizational research practice. This article presents a comparison between a human-respondent survey of 420 Silicon Valley coders and developers and synthetic survey data designed to simulate real survey takers generated by five leading Generative AI Large Language Models: ChatGPT Thinking 5 Pro, Claude Sonnet 4.5 Pro plus Claude CoWork 1.123, Gemini Advanced 2.5 Pro, Incredible 1.0, and DeepSeek 3.2. Our findings reveal that while AI agents produced technically plausible results that lean more towards replicability and harmonization than assumed, none were able to capture the counterintuitive insights that made the human survey valuable. Moreover, deviations grouped together for all models, leaving the real data as the outlier. Our key finding is that while leading LLMs are increasingly being used to scale, replicate and replace human survey responses in research, these advances only show an increased capacity to parrot conventional wisdom in harmony with each other rather than revealing novel findings. If synthetic respondents are used in future research, we need more replicable validation protocols and reporting standards for when and where synthetic survey data can be used responsibly, a gap that this paper fills. Our results suggest that synthetic survey responses cannot meaningfully model real human social beliefs within organizations, particularly in contexts lacking previously documented evidence. We conclude that synthetic survey-based research should be cast not as a substitute for rigorous survey methods, but as an increasingly reliable pre- or post-fieldwork instrument for identifying societal assumptions, conventional wisdoms, and other expectations about research populations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18519v2">As You Wish: Mission Planning with Formal Verification using LLMs in Precision Agriculture</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Though robotic systems are now being commercialized and deployed in various industries, many of these systems are highly specialized and often require an advanced skill set to operate and ensure they perform as instructed. To mitigate this problem, we recently introduced a mission planner leveraging LLMs to synthesize mission plans in precision agriculture based on mission descriptions provided in natural language. While the system demonstrates impressive performance, it also suffers from the inherent ambiguities of natural language. In this paper, we extend our system to address this issue by introducing multiple feedback loops in the planning architecture that leverage linear temporal logic (LTL) to ensure the mission planning system meets the specifications formulated by the user while still using natural language. To mitigate potential bias, this is achieved by using two different commercial LLMs in charge of the specification and verification subtasks. Through extensive experiments, we highlight the strengths and limitations of integrating mission verification into a fully autonomous pipeline, particularly regarding an LLM's ability to generate valuable LTL formulas, and show how our proposed implementation addresses and solves these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21124v2">PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Current multimodal LLMs process audio as a mono stream, ignoring the rich spatial information essential for embodied AI. Existing spatial audio models, conversely, are constrained to fixed microphone geometries, preventing deployment across diverse devices. We present PhaseCoder, a transformer-only spatial audio encoder that is agnostic to microphone geometry. PhaseCoder takes raw multichannel audio and microphone coordinates as inputs to perform localization and produces robust spatial embeddings. We demonstrate that Gemma 3n LLM can be fine-tuned to reason over "Spatial Audio Tokens" produced by PhaseCoder. We show our encoder achieves state-of-the-art results on microphone-invariant localization benchmarks and, for the first time, enables an LLM to perform complex spatial reasoning and targeted transcription tasks from an arbitrary microphone array.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05420v1">Can Open-Weight LLMs Produce Kernel-Verified Coq Proofs? A Pilot Study</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate text that resembles a mathematical proof, but resemblance does not establish correctness. A formal proof checker verifies whether each proof step follows established logical rules. Coq bases its rules on the Calculus of Inductive Constructions, a logical framework that defines which proof steps the system may accept. This pilot study evaluated six open-weight LLMs on the same 100 theorems from CoqStoq, a benchmark derived from real Coq projects. Each LLM received one attempt per theorem with the temperature set to 0, and Coq checked every proposed proof in the theorem's original project environment. We counted a proof as successful only if the Coq kernel accepted it. Gemma 4 verified 12 of 100 theorems, Llama 3.3 verified 8, and DeepSeek Coder V2 Lite verified 1. Qwen 3.5, Mistral Small 3.1, and GPT-OSS verified none. The 21 successful model-theorem results covered 15 distinct theorems, 11 of which were not solved by a baseline of standard Coq tactics. All verified theorems had short or medium human-written reference proofs; no model verified a theorem with a long reference proof. Because the proof-length analysis was exploratory, this pattern does not establish that proof length caused the difference. For the three models with at least one success, the total generation cost per verified proof ranged from 741 to 36,193 output tokens, 14.9 to 178.0 seconds, and 0.0167 to 0.2000 aggregate GPU hours. We could not calculate these ratios for models with no verified proofs. Across 600 attempts, the models produced 21 kernel-verified proofs, giving an overall success rate of 3.5%. The study reports descriptive differences among the models but does not statistically test whether one model outperforms another. Therefore, the results do not establish a universal ranking of the six models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05411v1">Evaluating and Improving Pedagogical Fit in LLM-Based AI Tutors with the Pedagogical Suitability Index</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 This paper has already been accepted and presented at the IEEE 27th International Conference on Information Reuse and Integration for Data Science (IRI 2026) from July 31 to August 2, 2026, in Seattle, WA, USA
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as AI tutors, but a correct answer is not always a pedagogically appropriate one. In classroom learning, effective help depends not only on correctness, but also on whether a response matches the learner's current foundation, the course sequence, and the timing of concept introduction. Existing evaluations focus mainly on answer quality, leaving this instructional fit under-measured. We present the Pedagogical Suitability Index (PSI), a composite metric of six theory-informed sub-scores that evaluates how well LLM-generated tutoring responses align with learner readiness and curricular progression, and we further use PSI as a structured feedback signal for response improvement. We evaluate four LLM tutors (ChatGPT, Gemini, Gemma4, and Qwen3) across 240 scenario-based evaluations using paired standard and defective prompts, then apply a PSI-guided regeneration protocol to 62 weak-performing cases. Baseline differences across the four tested models were modest overall (PSI range: 0.557 to 0.638), and open-weight and closed models did not exhibit a clear separation in pedagogical fit. Under the tested prompt perturbations, overall PSI remained largely stable (Delta = -0.002), though sub-score trade-offs emerged. More importantly, PSI-guided feedback substantially improved weak-performing cases: 51 of 62 cases improved (82.3%). Focused manual evaluation of the 62 PSI-selected weak cases provides initial evidence that the identified weaknesses are instructionally meaningful and that many PSI-guided regenerations correspond to human-judged improvement. These results suggest that learner- and curriculum-aware alignment may matter more for effective tutoring than model category alone, and that such alignment is both measurable and improvable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05353v1">Evidence Lock Before Commitment: A Frozen Interface Degrades LLM-as-Judge Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      LLM judges are often asked to extract criteria and evidence before choosing between candidate answers. This workflow assumes that the intermediate record preserves the information needed for a later verdict. For reasoning-capable models, visible field order does not reveal internal decision order, so we test an observable alternative: persist the evidence in one call and make it the exclusive input to the next. Across 24,000 judgments over HelpSteer3, FeedbackQA, and CoVal, we compare standard pairwise judging, structured one-call judging, two-call evidence locking, and three-call pointwise locking with Claude Sonnet 4.5 and GPT-5. Evidence locking reduces agreement with released human preferences by 4 to 6 percentage points and increases answer-order inconsistency by 8 to 10 points relative to structured one-call judging. Pointwise locking is also harmful, while structured evidence elicitation remains close to standard judging. The result holds for both judges and all three datasets. Persisted evidence can support auditability, but it should not replace the source answers at decision time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05303v1">EdgeXpert: An Edge Device for Memory-Efficient LLM Inference with Mixture-of-Experts and Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted at the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO 2026)
    </div>
    <details class="paper-abstract">
      On-device deployment of Large Language Models (LLMs) has become essential for personalized edge applications. A primary bottleneck is external memory access (EMA) in feed-forward network (FFN) layers. Speculative decoding and mixture-of-experts (MoE) are promising solutions. Speculative decoding reduces the number of decoding stages by generating multiple tokens per stage, and MoE minimizes per-stage cost through sparse expert activation. However, there is an incompatibility when combining these two techniques. We propose EdgeXpert, a software-hardware co-designed LLM accelerator that resolves this incompatibility. In the prefill stage, the prompt-wise expert reuse reformulates routing as prompt-level expert reuse rather than independent per-token expert selection. It identifies important tokens using a lightweight encoder, constructs a shared expert set from them, and routes less important tokens with a reduced expert budget to lower expert EMA. In the decode stage, depth-aware expert coalescing exploits the contextual similarity and mutual exclusivity of same-depth candidate tokens. Rather than loading the union of all required channels, EdgeXpert loads only salient channels and applies computational calibration to recover accuracy without additional memory access. Synthesized in Samsung 28nm technology at 800 MHz, EdgeXpert achieves up to 56.3% latency reduction and 44.1% energy reduction compared to prior works, while maintaining near-baseline accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05139v1">Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 https://github.com/Gen-Verse/Skill-Entropy-RL
    </div>
    <details class="paper-abstract">
      Long-horizon reasoning in recent LLMs demands that the model switch between distinct skills inside a reasoning chain, such as first doing a math derivation, then using the result to plan a schedule. We call such problems cross-skill long-horizon tasks: multi-step tasks whose steps require different reasoning skills and depend on earlier outputs. Existing benchmarks often evaluate individual skills, lacking a principled way to measure how well a model switches between skills. We address this gap from both the evaluation and training sides. We introduce Skill Entropy, a measure of the difficulty of switching from one skill to another. We then propose Skill^2-Bench, a benchmark of cross-skill long-horizon tasks built over 558 skills across 9 verifiable and open-ended domains. Each task is assigned a task-level skill-entropy score and grouped into three difficulty levels. Evaluating 8 frontier and 4 open-source models on Skill^2-Bench reveals a skill-switching gap: accuracy decreases on higher-entropy tasks. We then turn skill entropy from a benchmark scale into a training signal. We propose Skill-Entropy RL, an RL framework where the model predicts not only the answer at each step but also the skill used to produce it. The reward combines step-level correctness with a skill-entropy reward that measures the alignment between the model-predicted skill sequence and the gold skill sequence. On Qwen3-4B-Instruct and Qwen3-1.7B, Skill-Entropy RL improves the Skill^2-Bench score from 34.4% to 68.4% and from 14.6% to 40.1%, respectively, outperforming competitive baselines. The same pipeline can be applied to off-the-shelf training data such as OpenR1-Math, indicating that skill entropy is a reusable training signal. Code available at: https://github.com/Gen-Verse/Skill-Entropy-RL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05120v1">DASyR-LLM: Domain-Aware Symbolic Regression with LLMs for Kinetic Model Discovery</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Kinetic model discovery is a central challenge in chemical engineering, as accurate rate expressions are essential for understanding and controlling chemical and biological processes. Symbolic regression (SR) has emerged as a powerful data-driven approach for identifying interpretable kinetic models, but usually operates without domain knowledge, often exploring physicochemically implausible models. Large language models (LLMs) offer a promising avenue for injecting domain expertise into this search. Here, we introduce an LLM-guided SR framework, embedding an LLM module within an iterative SR algorithm for automated kinetic model discovery. The LLM performs two roles at each iteration: (1) a qualitative physicochemical critique of the best SR candidates, and (2) the proposal of new candidate rate expressions guided by the SR-generated models and embedded chemical knowledge. Our framework is evaluated on four in silico case studies of increasing complexity, spanning heterogeneous catalysis and bioprocess systems. Results show the LLM-guided framework reduces iterations to identify the ground-truth model by $41.7-79.3\%$ versus a state-of-the-art SR framework, with the LLM directly proposing the correct model structure in over half of the guided runs. In practical settings, where each iteration typically requires a new wet-lab experiment, this translates into a substantial reduction in experimental effort. Predictive performance on an independent validation set is equivalent between both approaches, with $R^2>0.98$ in all case studies. Ablation studies indicate that both the SR component and the LLM scale contribute to this performance, with a reduced-size LLM largely retaining discovery efficiency. These findings demonstrate that LLMs can effectively inject domain knowledge into scientific model discovery, paving the way toward fully automated, domain-aware kinetic modelling pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05063v1">Hardware Design and Security in the Era of Chiplets and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      The semiconductor industry is undergoing a dual revolution: the shift toward heterogeneous 2.5D chiplet systems and the integration of Large Language Models (LLMs) into Electronic Design Automation (EDA) flows. While these paradigms offer unprecedented benefits in yield, modularity, design productivity, etc., they radically expand the hardware attack surface. This paper provides a unified analysis of these frontiers, ranging from attacks on chiplet systems (including hardware stacks for LLM acceleration) across architectural, logical, and physical levels, to various exploits against LLM-driven EDA pipelines. To secure chiplet systems, we review a powerful defense approach that leverages 2.5D split manufacturing and active interposers for physically isolated Root of Trust (RoT) architectures. To secure LLM-driven EDA pipelines, we first identify native threats and then review state-of-the-art defense techniques. Finally, we discuss how LLM systems can advance hardware security efforts for modern systems, including chiplets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01742v2">MemSIF: From Structured Interactions to Dual-Track Fact Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Submitted to AAAI 2027. 19 pages, 10 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Long-term memory is critical for LLM agents operating over long-horizon interactions. However, several persistent limitations of existing memory systems can be traced to two recurring misalignment patterns in long-term interaction settings: Temporal-Structural Misalignment (TSM) and Delayed Utility Manifestation (DUM). TSM arises when temporal proximity does not reliably align with topical or event-level relatedness, whereas DUM arises when write-time salience does not reliably predict future query utility. To mitigate these misalignment patterns, we propose MemSIF (Memory with Structured Interactions and Facts), a structured interaction-to-fact memory framework. Structured Interaction Memory organizes raw interactions into Topical Segments that preserve local topical coherence and Event Trajectories that maintain cross-time event continuity. Dual-Track Fact Memory uses two complementary tracks: CoreFact memory consolidates stable, schema-guided information at write time, whereas ActiveFact memory forms facts on demand and promotes those supported by multiple historical sources and recurring query demand for reuse. Experiments on LoCoMo and LongMemEval-S across five backbone LLMs show that MemSIF achieves the highest Total ACC in all settings, outperforming the strongest baseline by 2.29%-8.79% on LoCoMo and 2.87%-6.15% on LongMemEval-S. These results support the effectiveness of combining Structured Interaction Memory with Dual-Track Fact Memory to mitigate TSM and DUM. Code is available at https://github.com/luoyufeihaha/MemSIF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05040v1">Private Direct Preference Optimization for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 accepted for publication at CCS 2026, extended version
    </div>
    <details class="paper-abstract">
      Direct preference optimization (DPO) is now a standard method for aligning large language models (LLMs) using human preference data. Each DPO example contains a prompt and a pair of candidate model responses. While prompts and responses are often public or model-generated, the relative preference between responses reflects subjective judgments and can reveal sensitive attributes of annotators or end users. Off-the-shelf privacy-preserving approaches are not well matched to this structure, leading to unnecessary noise injection and biased updates in training. In this paper, we formalize preference privacy, a label-DP-style privacy notion for DPO that protects only the relative preference between candidate responses, assuming an adversary who already knows the prompt and responses. We then design PrivDPO, a DPO variant that enforces preference privacy while remaining compatible with large-scale LLM training. Our main observation is that, for neighboring examples differing only in their preference signal, the gradient difference lies on a one-dimensional preference axis determined solely by the text; all preference information flows through this axis. PrivDPO adds calibrated randomness only along this axis via an unbiased randomized rescaling of the DPO objective, avoiding per-example gradient operations. Our experiments on three alignment benchmarks and three LLM families show that PrivDPO consistently achieves strong privacy-utility trade-offs compared with privacy-preserving baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05033v1">SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Sparse matrix kernels are fundamental to scientific computing, graph analytics, and machine learning. Their GPU performance depends strongly on the input sparsity pattern and execution strategy. For the same SpMM on the same matrix, cuSPARSE exhibits a 350x performance gap between CSR and Blocked-ELL. Our study of multiple data formats, specialized systems, and sparse compilers shows that no single implementation consistently dominates across sparsity patterns and operators. This motivates a system that can adapt its representation, execution strategy, and hardware mapping to each workload and target GPU. We present SparseDitto, an LLM-based system that constructs a GPU kernel for each matrix, operator, and target GPU. SparseDitto supports SpMV, SpMM, and SpGEMM within a unified design framework. A lightweight additive model ranks established strategies using structural features of the input matrix. An architecture-aware planner then proposes several candidate designs. Coding and verification agents implement and refine them using measurements from the target GPU. Across three sparse operators and a diverse set of matrices, SparseDitto achieves a geometric-mean speedup of 2.68x over cuSPARSE on an NVIDIA RTX PRO 6000 GPU, with a maximum of 146.61x. On an NVIDIA H200 GPU, it achieves 2.79x, with a maximum of 78.5x. Its generated SpMM kernels also accelerate full-batch GCN training by up to 3.39x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06337v2">Can Post-Training Transform LLMs into Causal Reasoners?</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Causal inference is essential for decision-making but remains challenging for non-experts. While large language models (LLMs) show promise in this domain, their precise causal estimation capabilities are still limited, and the impact of post-training on these abilities is insufficiently explored. This paper examines the extent to which post-training can enhance LLMs' capacity for causal inference. We introduce CauGym, a comprehensive dataset comprising seven core causal tasks for training and five diverse test sets. Using this dataset, we systematically evaluate five post-training approaches: SFT, DPO, KTO, PPO, and GRPO. Across five in-domain and four existing benchmarks, our experiments demonstrate that appropriate post-training enables smaller LLMs to perform causal inference competitively, often surpassing much larger models. Our 14B parameter model achieves 93.5% accuracy on the CaLM benchmark, compared to 55.4% by OpenAI o3. Furthermore, the post-trained LLMs exhibit strong generalization and robustness under real-world conditions such as distribution shifts and noisy data. Collectively, these findings provide the first systematic evidence that targeted post-training can produce reliable and robust LLM-based causal reasoners. Our data and GRPO-model are available at https://github.com/OpenCausaLab/CauGym.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05030v1">From Score Matrices to Football-Aware Match-State Simulation: An Auditable LLM Harness for Exact-Score Reranking</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 9 pages, 1 figure, 5 tables. Interim chronological benchmark on the first 150 matches of the 2025-26 English Premier League
    </div>
    <details class="paper-abstract">
      Football score forecasting combines a strong statistical core with a difficult contextual edge. Dynamic Poisson-family models estimate team strength, expected goals, and coherent score probabilities, but do not directly understand roles, tactical matchups, motivation, or how a first goal changes behaviour. Large language models (LLMs) can reason about such concepts, yet are not calibrated probability engines. We combine both components through an auditable information harness. This paper documents four iterations: V1, a dynamic score-driven Dixon-Coles baseline; V2, which maps LLM contextual ratings back into expected-goal parameters; V3, which replaces scalar correction with goal-by-goal simulations over a frozen score-candidate set; and V4, which adds shared first-breakthrough and post-goal cascade judgments, time-aware stopping, and deterministic tail candidates. The harness defines input semantics, supplies pre-match evidence, and constrains the LLM to an inspectable reasoning route. On a chronological replay of the first 150 matches of the 2025-26 English Premier League, V1 achieved 10.0% Top-1 and 26.7% Top-3 exact-score accuracy. V3 reached 12.0% and 30.0%, while V4 reached 14.7% and 30.7%. V4 increased candidate coverage from 77.3% to 84.7%, although no added tail candidate became a Top-3 exact hit. V1's native 1X2 distribution achieved 53.3% argmax accuracy, 0.9878 log loss, 0.5870 Brier score, and 0.2095 ranked probability score. These results are exploratory: the development slice is not an untouched benchmark, and temporal input isolation cannot exclude outcome memory in a closed LLM. The contribution is an auditable hybrid architecture, a clear design evolution, and negative findings showing where football-aware simulation does and does not improve score selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04991v1">RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 10 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents repeatedly process long, privacy-sensitive contexts, while cloud-only deployment exposes user data beyond the trusted endpoint and fully local deployment often requires costly hardware. Split inference offers a middle ground by executing the model head, tail, and tools locally and the middle layers in the cloud, but its local-cloud-local path transfers boundary hidden states at every invocation and creates a critical communication bottleneck. We present \system, a reference-aware codec that retrieves exact-token historical spans for prefill uplinks, reuses the reconstructed uplink state for same-round prefill downlinks, and generates boundary-specific decode references with lightweight causal predictors. RAC applies grouped affine alignment and calibrated residual quantization with optional prefill outliers, while sender-side wire-format reconstruction synchronizes subsequent references and offline calibration accounts for quality and packed representation costs. Across three models and nine evaluated model-link pairs, Raw-to-RAC mean time to first token (TTFT) and time per output token (TPOT) ratios are 1.24-2.72$\times$ and 1.01-2.79$\times$, while the 12 non-perplexity task-score changes range from $-0.40$ to $+2.50$ points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04426v4">XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 9 pages, ACM CAIS 26
    </div>
    <details class="paper-abstract">
      Modern LLM agents increasingly rely on dynamic structured generation, such as tool calling and response protocols. Unlike traditional structured generation with static structures, these workloads vary both across requests and within a request, posing new challenges to existing engines. We present XGrammar-2, a structured generation engine for dynamic agentic workloads. Our design is based on two key ideas: first-class support for tag-triggered structure switching, and fine-grained reuse across requests with different output structures. Concretely, XGrammar-2 introduces TagDispatch for dynamic structural dispatching and Cross-Grammar Cache for substructure-level cache reuse across grammars. It further improves efficiency with an Earley-based adaptive token mask cache, just-in-time compilation, and repetition state compression. Experiments show that XGrammar-2 achieves over 6x faster compilation than prior structured generation engines, and incurs near-zero end-to-end overhead in modern LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07322v3">FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04934v1">State2State: Environment-Derived Mid-Training for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Training LLM agents commonly relies on supervised fine-tuning from expert trajectories or online reinforcement learning over human-specified tasks with handcrafted verifiers. Though effective, both remain bottlenecked by externally specified tasks and supervision signals, limiting the scalability and diversity of agent training. We study an environment learning paradigm in which agents acquire interaction and manipulation capabilities solely through environment interaction, without externally specified tasks. We propose State2State, an environment-derived mid-training method that converts explored environment states into training objectives, challenging agents to reach a specified target state. By deriving tasks from environment exploration and verifying success through rule-based state matching, State2State provides scalable and verifiable training objectives without expert supervision or manual task design. Experiments on ALFWorld and ScienceWorld show that State2State improves agent performance as a standalone environment-learning stage in most settings. As initialization for downstream RL, it further improves final performance and learning efficiency, with promising evidence of cross-environment generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14137v3">From Feelings to Metrics: Understanding and Formalizing How Users Vibe-Test LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Published at COLM 2026. 50 pages, 20 figures. Code and data at https://technion-cs-nlp.github.io/vibe-testing-llms
    </div>
    <details class="paper-abstract">
      Evaluating LLMs is challenging, as benchmark scores often fail to capture models' real-world usefulness. Instead, users often rely on ``vibe-testing'': informal experience-based evaluation, such as comparing models on coding tasks related to their own workflow. While prevalent, vibe-testing is often too ad hoc and unstructured to analyze or reproduce at scale. In this work, we study how vibe-testing works in practice and then formalize it to support systematic analysis. We first analyze two empirical resources: (1) a survey of user evaluation practices, and (2) a collection of in-the-wild model comparison reports from blogs and social media. Based on these resources, we formalize vibe-testing as a two-part process: users personalize both what they test and how they judge responses. We then introduce a proof-of-concept evaluation pipeline that follows this formulation by generating personalized prompts and comparing model outputs using user-aware subjective criteria. In experiments on coding benchmarks, we find that combining personalized prompts and user-aware evaluation can change which model is preferred, reflecting the role of vibe-testing in practice. These findings suggest that formalized vibe-testing can serve as a useful approach for bridging benchmark scores and real-world experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04910v1">AutoCue: Multimodal LLM-Assisted Externalization of Implicit Inputs as Instructional Visual Cues in Screencast Tutorials</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted to Graphics Interface 2026 (GI 2026)
    </div>
    <details class="paper-abstract">
      Tutorial videos are widely used for learning feature-rich software, yet following screencast tutorials often breaks down in practice. Through a survey and contextual inquiry, we found that learners frequently rewind or get stuck because critical input information, especially mouse actions and keyboard-modified operations, is often implicit or missing in tutorials without input metadata. To address this problem, we present AutoCue, a multimodal LLM-assisted, human-in-the-loop tutorial augmentation pipeline for externalizing implicit inputs as instructional visual cues. AutoCue integrates frame-to-frame visual changes, narration signals, and operation guidance from official software manuals to infer likely mouse and key-modifier actions, then produces aligned cue layers and editable artifacts for human refinement. Grounded in multimedia learning and cognitive load theory, we further develop a visual cue grammar for representing mouse, keyboard, and combined inputs in software-learning tutorials. We instantiate and evaluate AutoCue in Autodesk Maya, focusing automatic inference on selected UI-mediated interactions with observable visual or textual feedback while supporting more ambiguous state changes through editable authoring artifacts. In a between-subjects study with 24 participants, the AutoCue-augmented tutorial reduced task completion time and interaction breakdowns and showed improved learner-reported experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04907v1">LLM-Assisted Detection and Repair of Hardware Security Vulnerabilities in Verilog Designs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 6 Pages, 3 figures, technical report
    </div>
    <details class="paper-abstract">
      Hardware designs, like software, are susceptible to bugs that can introduce security vulnerabilities and create opportunities for malicious exploitation. Unlike software vulnerabilities, however, hardware flaws become permanently embedded in silicon after fabrication, making them difficult or impossible to patch. Many of these weaknesses are categorized under the Common Weakness Enumeration (CWE) framework and include improper access control, exposure of sensitive information, and unintended privilege escalation. To improve the detection of such vulnerabilities, we propose a methodology that leverages a Large Language Model (LLM) to identify potential hardware CWEs directly from hardware designs in Verilog. The proposed approach is evaluated iteratively on a dataset of single-module Verilog designs to assess its effectiveness in detecting hardware security weaknesses. Our results demonstrate the potential of LLMs to augment traditional hardware security analysis by providing automated, scalable assistance for identifying security vulnerabilities during the hardware design process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04899v1">Evaluation Pitfalls and Sparsity Limitations in LLM-based Confidence Estimates for Classification</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Published at Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Confidence estimation is essential when LLMs are used for classification, indicating when predictions can be trusted. However, common approaches such as verbalization produce extremely sparse outputs. For instance, Qwen3-32B verbalizes only eight unique confidence values on SST-2, with over half being exactly 95%, a pattern we observe consistently across four datasets and two LLMs. Besides limiting practical utility, we show that this sparsity critically affects evaluation: the choice of interpolation in area under the accuracy-rejection curve (AUARC) dramatically alters rankings, with consistency sampling dropping from best to worst under stepwise versus linear interpolation. We advocate for standardizing stepwise interpolation for a fairer comparison. Under such a fair evaluation, we find that weighting verbalized digits by token probabilities, a method we term verbalization logprobs, addresses sparsity and achieves the best AUARC (+2.3 points over vanilla verbalization) without incurring additional inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04893v1">When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems relay key--value caches instead of text and credit their gains to exchanged ``latent thoughts''. That credit is a claim about \emph{which} example's cache is relayed, not merely that one is. We audit it causally in released systems. The cache is replaced with deranged (mismatched-example), zeroed, and moment-matched random counterparts, under two regimes defined by whether the receiver needs the sender's private information. Where it does, the battery reads ceiling: 100% against 23--25% for answer-irrelevant relays on the primary backbone, a contrast replicated across three families, five checkpoints, and a prose document-QA surface. Where it does not, a pre-registered five-seed protocol establishes equivalence within 2.8 points, a margin anchored to the audited system's reported gain, under Holm-corrected TOST on GSM8K and ARC-Challenge across three Qwen3 scales and on MedQA at 8B (one cell shows a small detected advantage inside the margin); a second family shows no detected advantage. A large cache effect need not be a pairing effect. In one natural cell, zeroing the relay costs 14.7 points; a mismatched cache, 0.4. Nor is need sufficient: under the same test, delivered channels span ceiling (LatentMAS's native relay), partial (KVComm's layer subset), and no detected example-specific transfer (C2C's released projector). Benchmark deltas do not by themselves establish latent-thought transmission; establishing it takes a mismatched-cache audit, which we release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05234v1">PPDL: LLM-Based Flows as Probabilistic Programs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Published at ICML 2026
    </div>
    <details class="paper-abstract">
      Building reliable applications that leverage large language models (LLMs) remains a significant challenge. While LLMs offer impressive capabilities across diverse tasks, their outputs often lack accuracy and provide no clear measure of confidence. This uncertainty compounds in flows of multiple calls to LLMs and other tools, making it difficult for developers and end-users to trust the results. This paper introduces a probabilistic language for programming LLM-based flows. It enables developers to quantify and propagate uncertainty throughout the application's flow, and experiment with different inference scaling techniques without adding a single line of code beyond the flow's logic. We present an experimental study to demonstrate this capability, and a case study building a theorem proving agent for the Rocq theorem prover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04828v1">Skill-Use: Can LLMs Actually Use Skills in Agentic Harnesses?</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on skills, structured documents that specify when to act, which procedure to follow, and which tools are allowed. Existing evaluations mostly judge the quality of a skill or its contribution to task success, leaving unexamined whether an agent can recognize a relevant skill and apply it on its own. We introduce Skill-Use, a benchmark that evaluates skill use under progressive disclosure, where an agent sees only a skill's name and short description and must retrieve the full procedure before following it. Skill-Use separates three facets of skill use. Trigger measures whether the agent invokes the relevant skill, Compliance measures how faithfully it follows the prescribed procedure, and Boundary measures whether it avoids forbidden operations. A Skill-Use (SU) score combines the three and credits execution only after the skill is triggered. Skill-Use pairs 79 real skills with 177 executable tasks across nine domains, each grounded in real files, run in an isolated Docker sandbox, and scored by a trajectory-based rubric. Evaluating eight LLMs under two agent harnesses, we find that reliable skill use remains out of reach, as the strongest configuration reaches an SU of only 0.613. Triggering and procedural compliance fail as independent bottlenecks, and both scores and model rankings shift with the harness, so skill use behaves as a capability conditioned on the harness rather than a fixed property of the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05232v1">Analysis of Numerical Localisation in LLM Translations</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 13 pages, 7 tables, 2 figures
    </div>
    <details class="paper-abstract">
      The work of Tang et. al. (2025) on numerical translation is extended by analysing the capability of five large language models (LLMs) for the localisation of times, numbers, and dates instead of translation. Models were selected that could be loaded onto and run on commodity hardware and a baseline quality for each mode is computed, then three different strategies to improve on that accuracy were tested. In contrast to Tang et. al., it was discovered that on the tested LLMs, embedding the localisation principles into the prompt context provided a statistically significant improvement in accuracy compared to direct translation or the alternative strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02567v2">Feedback Loops and Code Perturbations in LLM-based Software Engineering: A Case Study on a C-to-Rust Translation System</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The advent of strong generative AI has a considerable impact on various software engineering tasks such as code repair, test generation, or language translation. While tools like GitHub Copilot are already in widespread use in interactive settings, automated approaches require a higher level of reliability before being usable in industrial practice. In this paper, we focus on three aspects that directly influence the quality of the results: a) the effect of automated feedback loops, b) the choice of Large Language Model (LLM), and c) the influence of behavior-preserving code changes. We study the effect of these three variables on an automated C-to-Rust translation system. Code translation from C to Rust is an attractive use case in industry due to Rust's safety guarantees. The translation system is based on a generate-and-check pattern, in which Rust code generated by the LLM is automatically checked for compilability and behavioral equivalence with the original C code. For negative checking results, the LLM is re-prompted in a feedback loop to repair its output. These checks also allow us to evaluate and compare the respective success rates of the translation system when varying the three variables. Our results show that without feedback loops LLM selection has a large effect on translation success. However, when the translation system uses feedback loops the differences across models diminish. We observe this for the average performance of the system as well as its robustness under code perturbations. Finally, we also identify that diversity provided by code perturbations can even result in improved system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11528v4">HermesHFL: Incentive-Compatible Hierarchical Federated Unlearning for Dynamic LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 15pages,8 figures
    </div>
    <details class="paper-abstract">
      Hierarchical federated unlearning (HFUL) for large language model (LLM) fine-tuning faces significant challenges due to hierarchical aggregation, dynamic client participation, and strong parameter coupling in LLM adaptation. Selectively removing client contributions is particularly difficult because model updates propagate across multiple aggregation stages while unlearning requests may coincide with client departures and rejoining. To address these issues, we propose HermesHFL, a hierarchical federated learning framework that supports selective unlearning, dynamic client participation, and client reintegration for scalable LLM fine-tuning via parameter-efficient fine-tuning (PEFT) with LoRA. We formulate a unified optimization problem that jointly models client participation, edge association, incentive allocation, and unlearning under heterogeneous client behaviors. To solve this problem efficiently, we develop Neogen, a neural-guided bilevel evolutionary optimization framework that combines CMA-ES for continuous incentive optimization with a CHC-based evolutionary mechanism for discrete participation and association decisions. A neural surrogate further accelerates optimization and improves search efficiency. Extensive experiments on LLM fine-tuning tasks demonstrate that HermesHFL consistently outperforms state-of-the-art baselines in model utility, unlearning effectiveness, convergence stability, and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04741v1">LoginTrap: Uncovering Task-Agnostic Phishing-Style Indirect Prompt Injection Attacks against LLM-based Web Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      LLM-based web agents automate user tasks by observing webpages and executing browser actions on behalf of users. As these agents operate on real web services, login becomes a sensitive authentication boundary because it involves credentials and sensitive information. Existing work shows that malicious webpage content can manipulate web agent actions, but it has not fully examined whether such content can induce login and cause end-to-end private data leakage. We study this attack surface and present LoginTrap, a task-agnostic login-inducing attack against LLM-based web agents. LoginTrap assumes a black box attacker that controls the webpage context and the induced login flow without knowing the user task or web agent internals. Under this threat model, LoginTrap uses webpage context to generate page-specific indirect injections through a fuzzing-inspired process, making login appear as a plausible prerequisite for continuing the task and guiding the agent to a controlled login page. We conduct a comprehensive analysis of LoginTrap across realistic web agent executions. The results show that LoginTrap reaches 86\% average end-to-end attack success across LLM backbones and remains effective across agent architectures and defenses. These findings identify login inducement as a systematic authentication boundary risk and motivate further research on authentication-aware defenses for web agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26987v2">How Developers Experience Debugging Unfamiliar Codebases with Code Tours Generated and Evaluated by Local LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Code tours are interactive, onboarding documentation to guide developers through a codebase. Large Language Models (LLMs) can automatically synthesize code tours. Prior work on code tour generation has not studied developer experience or trust calibration when debugging unfamiliar codebases with code tours generated and evaluated by open-weight LLMs. This study surveys how the properties of components in open-weight LLM-authored code tours influence developers' experiences when debugging unfamiliar codebases. We built a pipeline that generated and evaluated code tours from real reproducible bugs. 26 developers with varying backgrounds participated in a user study. In total, 26 code tours were authored from real Java bugs mined from 2025 GitHub commits, with each tour independently judged by two different LLMs, resulting in 52 evaluated configurations. Participants thought aloud as they explored each tour. Three authors qualitatively coded the interviews to identify recurring themes. Developers generally preferred tours that scaled detail with the code length, avoided merely restating code, were easily scannable, and adopted a guiding tone. However, some preferences were mutually exclusive, such as the use of imperative mood. Stack traces were often insufficient to identify all steps developers found relevant. Developers also trusted descriptions they perceived as human-written more than those they believed were AI-generated. Finally, LLM-generated annotations of tour quality were unreliable: sycophancy, confabulation, and incoherence were pervasive. This work lays a basis for future research on fine-tuning open-weight models for code tour generation, personalizing generation to accommodate diverging preferences, selecting relevant steps beyond stack traces, calibrating users' trust to avoid both disuse and misuse, and improving open-weight LLMs' ability to be more trustworthy evaluators
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06307v3">LLM-Based Educational Simulation: Evaluating Temporal Student Persona Stability Across ADHD Profiles</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Student simulation with Large language models (LLMs) offers a scalable alternative for educational research and teacher training. Yet, its validity depends on whether models maintain stable personas across extended interactions. We test this prerequisite using a dual-assessment framework measuring self-reported characteristics and observer-rated behavioral expressions. Across two experiments testing four clinically-grounded ADHD persona conditions, five LLMs, and three prompt designs, we quantify between-conversation stability (N=4,968) and within-conversation stability (N=3,952 across 9 turns). Self-reported characteristics remain stable for high intensities, constituting a necessary prerequisite for valid behavioral simulation. Observer-rated behavioral expression reveals selective instability: within-conversation drift occurs in unscripted dialog for high and moderate ADHD personas. Scripted interactions with explicit task prompts eliminate this drift entirely. Stable, persona-aligned simulated learners benefit from a structured interaction design to maintain behavioral coherence, which holds significant implications for teacher training, adaptive tutoring, and any application requiring sustained, path-dependent learner interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05227v1">BioMedJImpact: A Comprehensive Dataset and LLM Pipeline for AI Engagement and Scientific Impact Analysis of Biomedical Journals</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Assessing journal impact is central to scholarly communication, yet existing resources rarely capture how collaboration and artificial intelligence (AI) research jointly shape venue prestige in biomedicine. We present BioMedJImpact, a large-scale, biomedical-oriented dataset built from 1.74 million PubMed Central articles across 2,744 journals. BioMedJImpact integrates bibliometric indicators, collaboration features, and an LLM-derived AI engagement rate, defined as the proportion of AI-related articles within each journal-year. Specifically, AI engagement rate is extracted through a reproducible three-stage LLM pipeline. We analyze how collaboration intensity and AI engagement rate jointly influence scientific impact across two temporal subsets (2016-2019, 2020-2023). Two main patterns emerge: journals with larger author teams tend to have higher citation impact, while AI engagement rate is positively associated with Impact Factor only in the 2019 subset. To validate the LLM pipeline for deriving the AI engagement rate, we conduct human evaluation, confirming substantial agreement in AI relevance detection and consistent subfield classification. Together, BioMedJImpact provides both a comprehensive dataset at the interface of biomedicine and AI and a validated framework for scalable, content-aware scientometric analysis. Code and dataset are available at https://github.com/JonathanWry/BioMedJImpact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04670v1">Easy to Complete, Hard to Choose: Investigating LLM Performance on the ProverbIT Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed computational linguistics and achieved remarkable performance across numerous natural language processing tasks, yet significant gaps persist in understanding how these systems process culturally embedded linguistic expressions. This paper introduces ProverbIT, a novel Italian benchmark comprising 100 multiple-choice questions designed to evaluate LLMs' ability to complete Italian proverbs. We assess 13 frontier models, including Large Reasoning Models (LRMs) and traditional LLMs, across three tasks: proverb completion, multiple-choice selection with correct answers, and multiple-choice selection without correct answers. Our evaluation reveals surprising results: while nearly all models demonstrate knowledge of the proverbs through successful completion tasks, performance drops dramatically when transitioning to multiple-choice formats without correct answers, with even state-of-the-art reasoning models showing substantial degradation. Through detailed Chain-of-Thought analysis of two LRMs, we uncover that models exhibit a strong bias toward selecting literal synonyms and frequently mention correct proverb endings during reasoning without successfully identifying their absence from the given options. These findings suggest that current LLMs rely heavily on memorized patterns rather than deeper semantic understanding of culturally grounded expressions, highlighting important limitations in their reasoning capabilities for figurative language comprehension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04588v1">EASy: Towards Efficient LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Agentic systems have emerged as a promising paradigm for solving complex tasks by coordinating specialized LLM-based agents. However, most existing systems primarily optimize task success while giving limited consideration to execution efficiency under practical constraints such as executor capability and computational cost. Existing router-based methods have limited ability to reason over rich, evolving task contexts, multi-step dependencies, and intermediate execution feedback, and often generalize poorly to unseen executors. We propose EASy, a trainable agentic framework that jointly optimizes task performance and computational efficiency through reinforcement learning. EASy equips an LLM-based orchestrator with explicit knowledge of the capability and cost profiles of heterogeneous executors, enabling context-sensitive coordination beyond performance-only routing. It further introduces a milestone-plan-act workflow that decomposes complex tasks into manageable milestones, constructs dependency-aware execution graphs, assigns suitable executors, and parallelizes independent steps while adapting subsequent decisions to intermediate outcomes. To train the orchestrator, we develop a tree-structured rollout procedure that explores alternative milestone decompositions and execution plans, together with multi-component rewards that capture task correctness, execution efficiency, and trajectory completeness. Extensive experiments on mathematical reasoning, embodied decision-making, and deep research benchmarks show that EASy consistently achieves stronger performance-efficiency trade-offs than strong agentic baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11506v4">RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      The transition toward localized intelligence through Small Language Models (SLMs) has intensified the need for rigorous performance characterization on resource-constrained edge hardware. However, objectively measuring the theoretical performance ceilings of diverse architectures across heterogeneous platforms remains a formidable challenge. In this work, we propose a systematic framework based on the Roofline model that unifies architectural primitives and hardware constraints through the lens of operational intensity (OI). By defining an inference-potential region, we introduce the Relative Inference Potential as a novel metric to compare efficiency differences between Large Language Models (LLMs) on the same hardware substrate. Extensive empirical analysis across diverse compute tiers reveals that variations in performance and OI are significantly influenced by sequence length. We further identify a critical regression in OI as model depth increases. Additionally, our findings highlight an efficiency trap induced by hardware heterogeneity and demonstrate how structural refinements, such as Multi-head Latent Attention (MLA), can effectively unlock latent inference potential across various hardware substrates. These insights provide actionable directions for hardware-software co-design to align neural structures with physical constraints in on-device intelligence. The released code is available in the Appendix C.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04570v1">The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Personalized LLMs with persistent memory are increasingly deployed, yet the faithfulness of their user models remains unexamined. We study over-inference (OI): the phenomenon where LLMs fabricate user attributes beyond what evidence supports. We introduce MirageBench, comprising 150 personas balanced across stereotypical, counter-stereotypical, and neutral profiles, 6 personalization tasks spanning an ``imagination gradient'', a four-way faithfulness taxonomy operationalized by an independent judge (validated against a blind human annotator on 400 claims: Cohen's kappa = 0.863 four-class, kappa = 0.900 binary), and a leaderboard of 12 models across 7 families on 143616 judged claims. We find that over-inference is pervasive: every one of the 12 models over-infers 35%--49% of its claims (cross-model mean 41.6%; claim-weighted 41.8%), with no model in this evaluation escaping it. Most strikingly, we surface a Self-Monitoring Inversion: at the model-selection level, models' self-assessed OI is negatively rank-correlated with their judge-measured OI (rho = -0.60, p = 0.044; exploratory, wide bootstrap CI [-0.90, +0.06], n = 12). The models that report the least over-inference tend to be flagged as fabricating the most, so self-reported confidence is a misleading signal for comparing models, even though within a single model self-audit still ranks that model's own claims moderately well (AUROC 0.58--0.83). We further show that OI is task-dependent (27%--59%) and that, in a multi-turn pilot, inferred attributes accumulate approximately linearly with little revision. MirageBench positions external verification, rather than model self-report, as a more reliable foundation for trustworthy personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07396v2">SHIELD: A Segmented Hierarchical Memory Architecture for Energy-Efficient LLM Inference on Edge NPUs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted to 2026 IEEE 8th International Conference on Artificial Intelligence Circuits and Systems (AICAS'26)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference on edge Neural Processing Units (NPUs) is fundamentally constrained by limited on-chip memory capacity. Although high-density embedded DRAM (eDRAM) is attractive for storing activation workspaces, its periodic refresh consumes substantial energy. Prior work has primarily focused on reducing off-chip traffic or optimizing refresh for persistent Key-Value (KV) caches, while transient and error-resilient Query and Attention Output (QO) activations are largely overlooked. We propose SHIELD, a lifecycle-aware segmented eDRAM architecture that jointly exploits temporal residency and bit-level sensitivity in bfloat16 (BF16) activations. SHIELD isolates the sign and exponent fields from the mantissa, disables refresh for transient QO mantissas, and applies relaxed refresh to persistent KV mantissas. Across multiple LLMs and inference scenarios, SHIELD reduces eDRAM refresh energy by 35% relative to a standard-refresh baseline while preserving accuracy on WikiText-2, PIQA, and ARC-Easy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.18709v2">LLMs Struggle to Measure What Distinguishes Students of Different Proficiency Levels: A Study of Item Discrimination in Reading Comprehension Assessment</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Existing work on LLM-based educational assessment has focused largely on item difficulty, but difficulty alone does not indicate whether an item meaningfully distinguishes higher- from lower-proficiency students. Item discrimination captures this complementary and fundamental psychometric property. We investigate whether LLMs can predict human item discrimination from assessment content. We evaluate 42 proprietary and open-weight LLMs using two complementary approaches. Direct discrimination prediction asks models to explicitly predict an item's discrimination value, while response-based proxy estimation treats LLM answers as synthetic responses and applies a Classical Test Theory (CTT)-inspired item-rest calculation. Direct predictions show weak alignment with human item discrimination. The response-based proxy provides a stronger but still limited ranking signal, reaching a CEFR-stratified rank correlation of 0.231. Further analysis shows that this correlation comes mainly from differences across models rather than proficiency prompts that reliably simulate students at different ability levels. Current LLMs therefore contain some discrimination-relevant information, but they do not yet reliably model the ability-conditioned human response behavior that gives item discrimination its psychometric meaning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05204v1">SkillTrace: Multi-Trace Provenance Auditing for LLM-Agent Skill Reuse</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      LLM-agent ecosystems are rapidly growing around reusable skills: mixed-modality packages of metadata, natural-language instructions, code, tools, references, and operational workflows. As skills become marketplace artifacts, auditing their reuse is no longer the same problem as ordinary code clone detection. Existing detectors target single-modality source code or whole-package similarity, yet skill reuse evidence is distributed across authored text, implementation fragments, and operational structure. As a result, they can miss reuse that preserves only one part of a skill. We present SKILLTRACE, a multi-trace provenance auditing framework for LLM-agent skill reuse. SKILLTRACE extracts three provenance traces: Expression, Implementation, and Operational. It represents the Operational Trace as a Skill Operational Graph (SOG) that captures activation, procedure, and resource-flow structure. An LLM assists only the Operational-trace extraction, once at ingestion; at audit time SKILLTRACE compares cached traces deterministically, calibrates each trace against same-function strict negatives, and reports which trace supports a reuse decision. On SKILLTRACE-BENCH, with 820 transformed reuse positives over 100 marketplace anchors and 751 negative controls, SKILLTRACE achieves AUROC 0.938 and F1 0.898. A 36,446-skill wild audit further shows that trace-attributed evidence surfaces actionable reuse review queues beyond repository-level baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06781v2">When Better Codebooks Are Not Enough: Predictive Performance and Behavioral Reliability in LLM Political Event Coding</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 13 pages, 3 figures, 13 tables. Revised version with updated experiments, behavioral reliability analyses, and additional API-model results
    </div>
    <details class="paper-abstract">
      High accuracy does not necessarily make an LLM a faithful coder. This issue matters because many social-science studies rely on expert-written codebooks to turn text into structured data. We study political event coding, where a model must identify the action that one actor directs toward another under detailed coding rules. We compare label names alone with concise definitions and enriched guidance that adds examples, event-mode instructions, and boundary rules. We also evaluate alternative prompting and retrieval methods. We then test behavioral reliability under changes to codebook order, label names, and label-definition mappings. Enriched guidance raises mean root-level macro-F1 from 0.457 to 0.633. Methods with access to definitions remain effective when meaningful label names are removed, but no evaluated method exceeds 0.20 weighted F1 after the label-definition mapping is reassigned. These results motivate separate evaluation of predictive performance and adherence to the supplied coding rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02150v3">PhyCheck: Fine-Grained Evidence-Grounded Dataset for Physical Law Understanding in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 15pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Embodied intelligence and world models require video understanding systems to go beyond recognizing objects and actions and develop an understanding of physical regularities. However, despite their strong performance on general video understanding tasks, current video-language models still struggle to reliably determine whether an observed event conforms to specific physical laws. Existing benchmarks primarily assess the physical quality of generated videos, providing limited support for systematically evaluating and improving the physical-law understanding of Video Large Language Models (VideoLLMs). To address this gap, we introduce PhyCheck, a video question answering dataset organized at two complementary levels of granularity. The coarse-grained subset asks models to determine whether the phenomenon shown in a video conforms to or violates physical laws, while the fine-grained subset further examines whether models can capture physical details responsible for the violation or compliance. We use these subsets as structured supervision to improve physical understanding. In addition, the dataset contains a diagnostic subset with external causal context that reveal hidden factors affecting physical plausibility, assessing whether models can recalibrate their judgments accordingly. Experiments with Fine-tune Qwen2.5-VL show that training with the proposed data substantially improves the understanding of physical-consistency, while evaluations in the diagnostic subset reveal that current models still have difficulty incorporating additional causal conditions into their decisions. These findings highlight the gap between recognizing surface-level inconsistencies and understanding underlying physical mechanisms, and provide a foundation for evaluating and improving physical understanding in Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04507v1">Emergence of Reputation-Based Cooperation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Can cooperation among large language model (LLM) agents be evolutionarily stable against free-rider invasion? We study an indirect reciprocity donation game where LLM agents observe behavioral traces and donate on a continuous scale. Strategies, represented as natural language prompts, evolve through cultural transmission across generations. Across four LLM backends, robustness to free-rider invasion varies by more than an order of magnitude. The strongest predictor of this robustness is opponent endowment sensitivity, the degree to which agents discriminate between cooperative and uncooperative opponents, operationalizing the classical Image Scoring mechanism. By contrast, adherence to the Leading-Eight L1 norm does not predict robustness. Robustness depends on defector exclusion: while both cooperator reward and defector punishment vary across models, only the stringency of defector exclusion predicts resistance to free-rider invasion. These findings reveal that LLM agents are confined to Image Scoring-like discrimination and fail to develop the more robust Leading-Eight norms, highlighting a fundamental vulnerability in culturally evolved LLM cooperation and motivating bottom-up approaches to norm construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23299v2">GRIMIP: A General Framework for Instance-Specific Configuration of MIP Solvers Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Configuring the hyperparameters of Mixed-integer programming (MIP) solvers is a high-dimensional, instance-dependent optimization problem where suboptimal settings can degrade solving time by orders of magnitude. Default configurations are often suboptimal, while traditional tuning methods either suffer from the ``cold-start'' problem and inefficient search or heavily rely on expert experience. This paper introduces \textbf{GRIMIP} (\textbf{\underline{G}}eneral \textbf{\underline{R}}easoning for \textbf{\underline{I}}nstance-specific \textbf{\underline{MIP}} configuration), a novel hybrid intelligence framework that synergistically integrates the semantic reasoning capabilities of Large Language Models (LLMs) with the sample-efficient search of Bayesian Optimization (BO). GRIMIP enables the LLM to function as a complete probabilistic surrogate within the BO loop, significantly improving performance and reducing sampling and evaluation costs. On seven benchmarks including MIPLIB, GRIMIP achieves over 40\% reduction in Primal-Dual Integral on hard instances, outperforming SMAC and other LLM-assisted BO methods. By granting LLMs sufficient autonomy, GRIMIP combines the expert-level reasoning of LLMs with the efficient search of BO, achieving state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04463v1">The Evaluator Is Part of the Experiment: Measuring Open-Ended LLM Conformity</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Prior work on LLM conformity largely measures discrete answer flips under verifiable labels. Open-ended revisions require a different measurement strategy because answer quality is graded, latent, and judged imperfectly. We introduce an experimental protocol implemented across a pooled main peer-condition corpus and separately constructed decomposition corpora, allowing us to separate ordinary re-answering, candidate-content exposure, a bundled peer-presentation residual, and directional judge sensitivity to visible peer context. Across four open-weight generators and three benchmarks, all-wrong peer input produces the lowest-quality revisions in every generator-dataset cell. Blind and informed ratings of identical answers also differ by evaluator: one judge shifts toward the peer-endorsed position, two shift away, one is approximately neutral, and GPT-4o and GPT-5.4-mini audits are likewise non-neutral. Finally, an anchor audit shows that terse correct anchors can be misread often enough to destabilize the latent scale unless calibration is checked explicitly. These results support four conclusions: flip rates are insufficient as a complete measure of open-ended conformity, wrong peers harm open-ended revision, evaluators are not neutral, and anchor calibration is necessary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.02012v2">Automated Database Testing via LLM-Synthesized SQL Features</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Database Management Systems (DBMSs) have been tested by various automated testing approaches. Many of them generate pairs of equivalent queries to identify bugs that cause DBMSs to compute incorrect results, and have found hundreds of bugs in mature, widely used DBMSs. Most such approaches are based on manually written SQL generators; however, their bug-finding capabilities remain constrained by the limited set of SQL features supported by the generators. In this work, we propose ShQveL, an approach that augments existing SQL test-case generators by leveraging Large Language Models (LLMs) to synthesize SQL fragments. Our key idea is to systematically incorporate SQL features gained through automated interactions with LLMs into the SQL generators, increasing the features covered while efficiently generating test cases. Specifically, ShQveL uses SQL sketches---SQL statements with incomplete code segments that LLMs fill---to integrate LLM-generated content into the generator. We evaluated ShQveL on 5 DBMSs and discovered 55 unique and previously unknown bugs, 50 of which were promptly fixed after our reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04450v1">CommBench: Can LLMs Write Correct and Efficient GPU Communication Code?</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Training and serving large language models (LLMs) rely heavily on high-performance GPU communication, yet implementing efficient GPU communication primitives requires deep expertise in GPU architectures, networking hardware, and distributed communication patterns, making them particularly challenging for code generation models. We present CommBench, a comprehensive benchmark for GPU communication programming, consisting of over 100 expert-curated tasks spanning point-to-point communication, collective operations, expert-parallel communication, compute--communication fusion, and communication utility functions, with reference implementations either written by GPU communication experts or distilled from production codebases. We further introduce a cheat-resistant evaluation framework that automatically compiles, executes, and validates generated code on multi-GPU systems, and a unified metric that jointly measures functional correctness and communication performance. Evaluating leading frontier and open-source code generation models on both intra-node NVLink and inter-node RDMA platforms reveals that even the strongest model, GPT-5.5, correctly implements and achieves competitive performance on only 30.7\% of the benchmark tasks. Our results expose a substantial gap between current LLMs and expert-written GPU communication code, establishing CommBench as a challenging benchmark for advancing AI-assisted systems programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04415v1">Social Pressure Breaks Majority Voting in LLM Safety Panels</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to detect unsafe content. A common approach is to combine judgments from a panel of models to correct individual mistakes, but this benefit may disappear when every model sees the same misleading context before voting. We study this risk in a controlled two-round experiment. Each model first judges an item alone, then judges it again after six simulated peers either assert the wrong label or abstain. We combine the final judgments by majority vote. Across six open-weight LLMs and six datasets, we find that the wrong-label peer message raises the average reviewer false-alarm rate from 56.5% under silent peers to 87.5%, and majority voting raises the panel false-alarm rate to 100%. Without an asserted label, the same panel outperforms its average member. The effect is strongly asymmetric: reviewers follow pushes toward "unsafe" far more than pushes toward "safe" (about 75% versus 17%), so the panel's false-alarm rate rises sharply while its harmful-miss rate changes little. The proprietary-model probe shows substantial variation across models. These results identify susceptibility to shared social cues as a failure mode of safety panels and provide a simple pre-deployment diagnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03505v2">ConlangBench: Exploring Language Knowledge and Learning in LLMs through Diverse Constructed Languages</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 29 pages, 12 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Constructed languages (conlangs) are intentionally created human languages with a rich tradition of linguistic creativity. Despite their potential for studying language learning in large language models (LLMs), existing conlangs remain largely underexplored in LLM research. We present ConlangBench, the first large-scale benchmark for evaluating and training LLMs on 21 existing conlangs. We collect over 21M conlang-English parallel sentence pairs (including 430K pairs across the 20 non-Esperanto conlangs) and 321K vocabulary entries. In bidirectional translation experiments, we find that models perform better on a posteriori conlangs, whose vocabularies are derived from natural languages, reflecting the design characteristics of conlangs. Training on ConlangBench also shows that models can learn all eight conlangs for which sufficient parallel corpora are available, while their learning curves vary depending on how the conlangs were created. Our findings suggest that conlangs provide a unique testbed for investigating how LLMs acquire low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00209v2">Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in high-stakes settings, yet they frequently violate contextual privacy by disclosing private information in situations where humans would exercise discretion. This raises a fundamental question: do LLMs internally encode contextual privacy norms, and if so, why do violations persist? We present the first systematic study of contextual privacy as a structured latent representation in LLMs, grounded in contextual integrity (CI) theory. Probing multiple models, we find that the three norm-determining CI parameters (information type, recipient, and transmission principle) are encoded as linearly separable and functionally independent directions in activation space. Despite this internal structure, models still leak private information in practice, revealing a clear gap between concept representation and model behavior. To bridge this gap, we introduce CI-parametric steering, which independently intervenes along each CI dimension. This structured control reduces privacy violations more effectively and predictably than monolithic steering. Our results demonstrate that contextual privacy failures arise from misalignment between representation and behavior rather than missing awareness, and that leveraging the compositional structure of CI enables more reliable contextual privacy control, shedding light on potential improvement of contextual privacy understanding in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04341v1">Continuous Improvement and Parallel Autonomous Exploration: An LLM-Agent Framework for Searching Large Solution Spaces</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 5 pages, ACM KDD'26 Workshop on SciSoc Agents & LLMs
    </div>
    <details class="paper-abstract">
      We present a framework that gives LLM agents two mechanisms for searching large solution spaces autonomously. First, a leaderboard scored on held-out data acts as a reward signal that drives each agent to refine its solutions over repeated submissions, a loop that operates even with a single agent. Second, the framework enables running many agents in parallel, fully autonomously, with no human in the loop: agents independently analyze, survey methods, implement, self-evaluate, submit, and revise, while a moderator agent handles only logistics. Running agents in parallel under the shared reward broadens the explored region of the solution space rather than refining the single seeded paradigm. We instantiate the framework on product-to-catalog matching (a core e-commerce retrieval task with a large, category-structured solution space), posed as selective prediction with a precision-coverage operating point. A single agent refines within its seeded paradigm, whereas parallel autonomous agents surface qualitatively different solutions. On this testbed, best qualified coverage (>=95% P@1 per category) reaches 47.8-57.4% with a single agent and 62.8-69.4% with five, against a 33.3% baseline. Our contribution is the framework itself: a continuous-improvement reward loop and a substrate for fully autonomous parallel exploration, backed by case-study evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05199v1">Post-Hoc Trajectory-Risk Certification for Modular LLM-Based Security Agents</a></div>
    <div class="paper-meta">
      📅 2026-08-04
      | 💬 18 pages, 11 tables; preprint
    </div>
    <details class="paper-abstract">
      Autonomous security agents operate as staged pipelines, such as classifying network traffic and then attributing attacks to a specific technique. Split conformal prediction gives each stage finite-sample coverage, but deployment requires a trajectory-level guarantee across the full chain. These guarantees do not compose automatically when stages are independently trained and calibrated. Bonferroni allocation is distribution-free but conservative under correlated errors. We show that a natural pairwise-correlation extension to three or more stages is invalid because it gives a lower rather than an upper bound, and derive a valid spanning-tree alternative. We distinguish whether stages are dependent from whether an audit sample is large enough to certify that dependence, and give matching upper and information-theoretic lower sample-complexity bounds. We also show that coarse-to-fine label selection can create near-perfect measured correlation without learned dependence. On a two-stage intrusion-detection pipeline across 6 open LLMs and 2 datasets, removing this artifact reduces measured correlation from near 1 to 0-0.78. A direct audit of trajectory failure becomes 13.7% tighter than Bonferroni once the audit reaches the required sample size, but is worse when undersized. A modular certificate using per-stage certificates and a pairwise overlap bound yields a positive average gain of 0.6%, quantifying the cost of lacking joint access. Same-model, cross-model, and permuted-pairing tests show that residual dependence reflects shared sample difficulty, not shared model representations. Average trajectory coverage across 12 configurations is 92.7% +/- 2.4% at alpha = 0.10. Under cross-dataset deployment, single-step miscoverage reaches 100% even when accuracy remains 78%, showing that distribution shift destroys calibrated confidence before raw accuracy.
    </details>
</div>
