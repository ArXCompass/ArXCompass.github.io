# llm - 2026_06

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12142v1">AerialClaw: An Open-Source Framework for LLM-Driven Autonomous Aerial Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicles (UAVs) are increasingly used in inspection, search and rescue, environmental monitoring, and emergency response. However, most UAV applications still rely on pre-defined command sequences or task-specific pipelines, where developers manually connect perception, planning, flight control, simulation, logging, and safety modules. This limits the flexibility, reproducibility, and extensibility of autonomous aerial systems. This paper presents AerialClaw, an open-source software framework that enables UAVs to operate as decision-making aerial agents rather than merely command-following platforms. Given a natural-language mission, AerialClaw allows an LLM-based agent to understand the task, maintain context, invoke executable aerial skills, observe perception and runtime feedback, and iteratively update its decisions in a closed loop. The framework adopts a modular brain-skill-runtime architecture, combining hard skills for atomic UAV operations, Markdown-based soft skills for reusable task strategies, document-driven agent state and capability boundaries, memory-driven reflection, safety-oriented runtime validation, and platform-agnostic execution adapters. AerialClaw supports lightweight mock execution, PX4 SITL with Gazebo, and AirSim-based simulation, together with a web console, pluggable model backends, example missions, simulation assets, and staged deployment scripts. By combining standardized aerial skills, document-driven agent state, memory, and closed-loop LLM decision-making, AerialClaw provides a reproducible and extensible open-source framework for building UAV systems that can interpret missions, make decisions, execute skills, and adapt their behavior from feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13628v3">Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper investigates compact large language model (LLM) deployment and world-model-assisted inference offloading in mobile edge computing (MEC) networks. We first propose an edge compact LLM deployment (ECLD) framework that jointly applies structured pruning, low-bit quantization, and knowledge distillation to construct edge-deployable LLM variants, and we evaluate these models using four complementary metrics: accessibility, energy consumption, hallucination rate, and generalization accuracy. Building on the resulting compact models, we formulate an MEC offloading optimization problem that minimizes the long-term average inference latency subject to per-device energy budgets and LLM-specific quality-of-service constraints on effective accuracy and hallucination. To solve this problem under unknown and time-varying network dynamics, we develop a world model-proximal policy optimization (PPO) algorithm, which augments an on-policy PPO algorithm with a learned recurrent world model that provides improved value targets and short imagination rollouts. Extensive experiments on Llama-3.1-8B, Qwen3-8B, and Mistral-12B show that ECLD compresses base models by about 70-80% in storage (i.e., from 15.3 GB to 3.3 GB for Llama-3.1-8B) and reduces per-query energy consumption by up to 50%, while largely preserving accuracy and often lowering hallucination compared with quantization-only or pruning-only baselines. Moreover, they also show that world model-PPO speeds up convergence by about 50%, improves the final reward by 15.8% over vanilla PPO, and reduces average inference latency by 12-30% across different user populations, while satisfying the accuracy and hallucination constraints and approaching the generation quality of always-offloading with much of the efficiency of local execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12117v1">Soft-Prompt Tuning for Fair and Efficient LLM Benchmark Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Benchmark scores often misrepresent a large language model's (LLM's) knowledge, because they rely, e.g., on the model's ability to follow specific formatting requirements. This especially penalizes base models that may know the correct answers but lack the ability -- typically introduced in post-training -- to structure them as instructed. To overcome this, we propose soft-prompt tuning, an efficient, fair, and architecture-agnostic model evaluation. By optimizing only 10 soft-prompt vectors (roughly 0.0006% parameters for a 7B model) over a short tuning period, we adapt models to specific benchmark formats, closing gaps in format-following and ensuring that underlying knowledge is accurately reflected in benchmark scores. This allows one to fairly compare different base models -- trained with various pre-training recipes -- on benchmarks without the need for full post-training. We evaluated soft-prompt tuning across 7 models and 7 datasets. The results show that (a) soft-prompt tuning saturates format-following within 80 steps (~640 samples) making it highly efficient, (b) soft-prompt tuning significantly outperforms zero- and few-shot prompting, surfacing base model knowledge that standard prompting misses, that (c) even post-trained models can benefit from soft-prompts to maximize format compliance, and that (d) soft-prompted base model performance predicts post-trained model rankings more reliably than zero- and few-shot baselines, offering a low-cost proxy for downstream model quality. Our contributions include (1) metrics which disentangle format-following and knowledge accuracy, (2) a fairer benchmarking protocol of LLM knowledge, and (3) a cost- and memory-effective recipe to identify optimal pre-training strategies early in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12073v1">"That's AI Slop, You Bot!" Studying Accusations, Evidence, and Credibility in Online Discourse Towards LLM-Generated Comments</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Generative AI has made fluent prose cheap to produce, breaking the old promise to readers that good writing meant real thinking. How have readers responded, and what can this tell us about changing anti-AI attitudes? We analyzed 25 million comments from Hacker News and Reddit (2023-2026), combining LLM judgment on 7,500 sampled accusations of AI use, sentiment trajectories, speech-act coding of 300 confirmed accusations of AI use, and a matched-control test of accused versus non-accused parent comments. We found that the pejorative-label share of accusations rose more than tenfold on both platforms while a placebo vocabulary of pre-2022 inauthenticity terms (shill, astroturf) did not. This shift reflected a fast-growing trend of branding any suspicious or seemingly inauthentic prose as "AI slop". The slop frame now constitutes 94 percent of pejorative mentions, with the dominant comments shifting in tone from mockery toward gatekeeping and structural protest. The key surprise comes from a matched-control test which found that prose features that statistically distinguish AI from human text do not predict which human text gets accused as AI. The new accusations work as social gatekeeping of perceived authenticity without actually screening for AI. This research extends signaling theory by showing that substitute signals used socially can grow even when inaccurate if the underlying detection problem cannot be solved at the non-expert level. It shows that AI's effects on writing from the reader side are distinct from those on the production (writer) side. Detection technology cannot resolve this dynamic because the social function of accusations is increasingly to perform social gatekeeping and in-group signaling as opposed to identifying AI-generated writing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12071v1">On the Limits of LLM-as-Judge for Scientific Novelty Assessment</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to generate and judge scientific ideas. This makes novelty evaluation a central problem. Full idea evaluation is difficult because it often requires judging a method, its feasibility, and its empirical promise. We therefore study a cleaner upstream object: the research question (RQ). RQ generation is a prerequisite for scientific ideation, and RQs can be compared against questions pursued in real papers. We introduce RQ-Bench, a benchmark built from recent arXiv papers. For each paper, we reconstruct author-anchored RQs from its cited background, gaps, and contributions. These RQs are not the only valid questions for the same background. They are author-anchored reference points for testing novelty judgments. We evaluate model-generated RQs with standalone LLM judging, comparative LLM judging, and human expert evaluation. LLM judges consistently rate model-generated RQs as highly novel, producing a novelty mirage; in comparative evaluations, this preference becomes even stronger. Domain experts, however, reach the opposite conclusion and prefer the author-anchored reference questions. We further find that many generated RQs are narrow or source-bound, a dimension that LLM judges often miss unless explicitly tested. Overall, the contradictory novelty evaluations between LLM judges and human experts raise a serious concern about the reliability of using LLMs to assess the scientific novelty of research questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03308v3">The Security Budget of Code-LLM Prompt Hardening: Provable Limits Under Pass-Only Acceptance</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      We give a quantitative impossibility result for pass-only prompt hardening of code LLMs. For any deterministic prompt filter $h$ and a registered family of finite executable-equivalence task variables $\mathcal Y_{\mathrm{exec}}$, the shared filtered-prompt channel $\rmI(h(p);h(\tilde p))$ is lower-bounded by a worst-$Y$ Fano floor; on HumanEval and MBPP the universal pass-only floor evaluates to $\mathcal F^{\mathrm{op}}\ge 0.84$ and $1.20$ nats at $η=0.05$ task-collapse tolerance, and the identity row realizes $\mathcal F^{\mathrm{id}}\ge 1.67$ and $1.80$ nats. An estimator-invariance corollary lifts the floor to any deterministic embedding pipeline; a dataset-agnostic corollary states the floor in visible-spec entropy and is empirically witnessed by $164/164$ HumanEval+ and $224/224$ MBPP+ $V(p)$-invariance. We operationalize the floor as the \emph{Tri-Audit Protocol}, a two-axis reporting protocol that separates a prompt-side deductive registry attribute (Shannon nats on the visible-spec representation) from a model-side empirical proxy (KSG-1 primary, MINE secondary, on hidden states). A constrained best-of-family search over deterministic and guarded learned filters on CodeLlama-7B, Qwen2.5-Coder-7B/1.5B and DeepSeek-Coder-6.7B at $n=164$ yields the \emph{Cross-Model Tri-Audit Invariance}: of twenty-eight pass-preserving rows, twelve antecedent-preserving deterministic rows fail proxy-axis leakage reduction on every backbone with sign-invariant positive deviations, twelve antecedent-changed-of-record learned-canonicalizer rows fail proxy-axis leakage on every backbone, and four antecedent-violating rows are reported as registered-family collapse; no filter produces a shared Tri-pass on a nine-cell gate-sensitivity sweep. Pass@1 alone cannot certify code-LLM prompt hardening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11976v1">Exploration Structure in LLM Agents for Multi-File Change Localization</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Software engineering tools increasingly rely on LLM based agents to localize files to change to resolve a software issue. Most AI agents explore repositories linearly, that is, visiting one directory or file per step. We postulate that this is a structural mismatch for changes that span several subsystems. We compare linear sequential exploration against non-linear, domain-scoped parallel agentic exploration. Using SWE Bench Pro as initial benchmark, we focus on ansible as an exemplar. We construct an approach for persistent-session evaluation of GitHub issues anchored at a single base commit. We compare our non-linear domain-agent file traversal system against a base LLM without direct repository access, a single agent Recursive Language Model (RLM) baseline with a persistent Python REPL and an external CLI baseline using Codex 5.5 High. Domain scoped parallel agent spawning with a small Haiku-class model achieves the highest micro F1 among Haiku class models by a large margin. Domain-agents is the second highest behind only the much larger Codex 5.5 High on our own expanded benchmark including over more recent PRs from 2025 and 2026. On the original, curated, 2020 SWE-bench Pro benchmark, a larger Sonnet plain LLM baseline attains higher micro F1 by predicting few files, leading to higher precision, but at significantly lower all gold recall. We also present three additional findings. First, documentation evolution is a latent dependency unresolved by any approach. Second, naive file system access can degrade localization driven by test-file over prediction. Lastly, forced multi-agent consultation does not measurably help and raises token cost substantially.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29128v2">Apertus LLM Family Expansion via Distillation and Quantization</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      The wide adoption of LLMs has led to their use in great variety of applications and scenarios, such as chatbot assistants and data annotation, creating the need for the models to satisfy certain budget and hardware constraints. This has led to the trend of LLMs being released in batches consisting of similar models of various sizes for the family of models to adhere to as wide of a range of constraints as possible. In this paper, we validate distillation and quantization as a cost-effective way to expand model families to new sizes and hardware formats. Based on the open-recipe Apertus 8B LLM, we produce Apertus-v1.1 - a distilled family of models with up to 4B parameters trained on 1.7T permissive license tokens. We demonstrate cost-efficiency and strong accuracy performance of our approach for covering large ranges of hardware and systems requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11925v1">Corpus Augmentation for Sign Language Translation via LLM-Guided Video Stitching</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Sign language translation (SLT) converts sign language video into spoken language text and holds significant promise for improving accessibility and enabling communication between signing and non-signing communities. While large weakly-aligned datasets have enabled pre-training at scale and gloss-free methods have reduced reliance on expert annotation, high-quality parallel sign video-text pairs for fine-tuning remain scarce, limiting generalisation on long-tail vocabulary and unseen constructions. We propose a corpus augmentation approach that requires no additional human annotation, external sign-language video corpora, or generative video models, relying only on the existing gloss-annotated training corpus and an LLM for sentence generation: per-gloss clips are extracted from training videos via CTC forced-alignment, novel gloss-sentence pairs are generated by a corpus-anchored LLM, and synthetic sequences are assembled through random sentence sampling and clip assignment. The resulting synthetic RGB video-text pairs are architecture-agnostic at the downstream training stage and can be consumed directly by RGB-based SLT models, or converted into pose or feature representations by pipelines that derive such inputs from video. Sincan et al. re-evaluated five recent gloss-free methods under strictly identical conditions; the largest verified gain over the GFSLT-VLP baseline was only 0.98 BLEU-4. Our augmentation, applied within the same framework, achieves +2.92 BLEU-4 without any change to architecture or training protocol. We further identify that synthetic data harms vision-language pretraining despite improving its objectives, and that optimising clip transitions for visual smoothness is counter-productive under L2-based criteria; we propose that abrupt boundaries may act as a form of implicit regularisation. Code is available at https://github.com/robizso/slt-datagen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11916v1">Characterizing Software Aging in GPU-Based LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      This paper proposes an empirical methodology to study software aging in GPU-based LLM serving systems. Traditional aging studies focus on CPU-centric software with relatively regular workloads; LLM serving is different, spanning a Python host and a CUDA device, handling requests whose cost varies by orders of magnitude, and relying on rapidly evolving software stacks. We run a 216-hour campaign across six co-located deployments under identical stress conditions, monitor host, device, and client metrics in parallel, and apply a statistical pipeline that accounts for autocorrelation and multiple testing. Our results reveal statistically significant memory aging in all deployments, with leak rates strongly dependent on the serving runtime and deployment configuration. Beyond these findings, we provide a reproducible framework that opens a research direction at the intersection of the software aging and rejuvenation and LLM serving communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11877v1">LLM-Enabled NWDAF: A Step Toward AI-Native 6G Network Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The Network Data Analytics Function (NWDAF) is central to enabling zero-touch network management in fifth-generation (5G) networks by supporting real-time analytics and closed-loop automation. Despite its critical role, open-source NWDAF implementations remain limited in scope and accessibility. In this paper, we develop an open-source NWDAF, compatible with the open-source core network Free5GC, that collects network data via subscriptions to Network Functions (NFs), and also includes an integrated Large Language Model (LLM) interface that enables natural language interaction with human operators. The interface processes user intents, encodes them using a semantic embedding model, and maps them to one of seven predefined intent categories to trigger analytics queries or event subscription commands. This architecture abstracts the complexity of traditional interfaces, allowing non-expert users to manage network analytics and subscriptions with ease. The system supports Access and Management Function (AMF) and Session Management Function (SMF) event subscriptions, real-time monitoring, and analytics retrieval via Prometheus, all accessible through a conversational interface. By bridging AI-driven intent recognition with standardized network analytics, our implementation enhances operator usability and provides a foundation towards AI-native 6G networks. The source code and datasets generated during the current study are available in the github repository, https://github.com/HenokDanielbfg/testbed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11863v1">Enhancing LLM-Based Code Translation with Verified Multi-Semantic Representations</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise for automated code translation, yet existing approaches often rely on token-level statistical patterns rather than sufficient understanding of program semantics. As a result, translated programs may still contain logical and semantic errors. Although high-quality semantic guidance, such as functional descriptions and test cases, can help mitigate these errors, such resources are often unavailable in real-world scenarios. This raises two key challenges: how to construct rich semantic information directly from source code, and how to ensure that such semantics are accurate and reliable enough to guide translation.To address these challenges, we propose Multisage, a multi-semantic augmentation and self-calibration framework for LLM-based code translation. Multisage consists of three modules. First, a semantic representation parsing module extracts structured base semantics from source code, including data-flow graphs, type constraints, and external API information. Second, a multi-semantic augmentation module builds on these representations to generate diverse augmented semantics, including code summaries, function-level test cases, and API-oriented descriptions and tests. Third, a semantic consistency calibration module uses semantics-preserving mutations and cross-semantic consistency verification to filter, calibrate, and refine the generated semantics.Experiments on the HumanEval-X code translation benchmark show that Multisage improves translation success rates by up to 2.22 times across diverse backbone models. It consistently outperforms vanilla prompting, instruction-tuned LLMs, and Chain-of-Thought reasoning, with the largest gains observed on smaller models. These results demonstrate that explicit semantic augmentation can substantially improve the reliability of LLM-based code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11854v1">Fine-tuning Multi-modal LLMs with ART: Art-based Reinforcement Training</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      There are two main Parameter-Efficient Fine-Tuning (PEFT) techniques for Large Language Models (LLMs). While Low-Rank Adaptation (LoRA) introduces additional weights between the LLM layers, Soft Prompting introduces additional fine-tuning-specific raw tokens to an LLM input. However, both require modification to the computational graphs of precompiled, preoptimized LLMs. As a result, neither is fully supported in high-throughput engines like vLLM. We propose fine-tuning with ART (Art-based Reinforcement Training). The method injects information into a frozen Multimodal Large Language Model (MLLM) by optimizing only its raw visual input, thus enabling the soft-token approach on pre-compiled computational graphs. It relies on backpropagation of gradients back into a plain pixel array and thus supports any fine-tuning objective. Moreover, the optimized visual input can be stylized as task-relevant computational artworks. The approach's effectiveness is confirmed for different sizes of a popular open Qwen architecture and for several textual benchmarks. Specifically, ART reaches accuracy competitive with LoRA across mathematics and structured-tool-use benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11817v1">Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for code generation, raising concerns that they may be misused to produce malicious code. Meanwhile, Grammar-Constrained Decoding (GCD) has been widely adopted to improve the reliability of LLM-generated code by enforcing syntactic validity. In this paper, we reveal a counterintuitive risk: this reliability-oriented technique can itself become an attack surface. We uncover a new jailbreak attack, termed CodeSpear, that exploits GCD to induce LLMs into generating malicious code. Our experiments show that simply applying a benign code grammar constraint can effectively jailbreak LLMs. To address this vulnerability, we propose CodeShield, a safety alignment approach that robustly preserves safe behavior even under attacker-controlled grammar constraints. CodeShield aligns the model in the code modality by teaching it to generate honeypot code under GCD. Such code is semantically harmless, so it does not implement the malicious request, and structurally diverse, so it is difficult to suppress through grammar tightening. At the same time, CodeShield still preserves natural-language refusals when natural language is available. Experiments on 10 popular LLMs across 4 benchmarks show that CodeSpear outperforms representative jailbreak baselines and increases the attack success rate by more than 30 percentage points on average. CodeShield also restores safety under CodeSpear while preserving benign utility. Our findings reveal a fundamental risk of GCD and call for greater attention to its potential security implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25359v2">Geometric Metrics and LLMs: What They Measure and When They Work</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      We present a systematic stress-test of geometric metrics for LLM evaluation. Rank-based geometric properties of internal representations have shown promise as reference-free quality signals, but the conditions under which they are reliable remain unclear. We evaluate eight commonly-used metrics: intrinsic-dimensionality estimators, spectral norms, and related quantities across six tester models (0.5-8B) and eight generators on contrasting tasks, separating genuine geometric signal from text-length effects and from what standard text statistics already capture. Three findings emerge. First, some metrics (notably Schatten Norm and MOM) mainly reflect output length, and their apparent discriminative power collapses once length is controlled. Second, geometric metrics add modest but real information beyond text statistics: combined with them, a classifier reaches 78% accuracy on 6-way generator identification versus 69% for text statistics alone. Third, rather than tracking a general notion of text quality, the metrics demonstrate only moderate association between the intrinsic-dimensionality and lexical diversity (RTTR). We give use-case-specific recommendations and identify failure detection as the most promising near-term application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11806v1">External Experience Serving in Production LLM Systems: A Deployment-Oriented Study of Quality-Cost Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Production LLM systems accumulate reusable operational experience, but the practical deployment issue is not merely whether such experience can help. It is how different serving strategies trade off quality against online cost under realistic constraints. Injecting external experience can improve task quality, yet it also increases prompt burden, latency, and serving pressure. We study \textit{external experience serving} as a deployment-oriented quality-cost trade-off problem. We evaluate this question in a real production moderation setting, with tool-use and GPQA as supporting contrast tasks that expose different output-cost regimes. We compare no-experience baselines, random experience controls, global prompt injection, and retrieval-based selective injection, and analyze both task quality and serving cost. The results show that, once experience becomes case-dependent, selective retrieval provides a stronger operating point than unconditional global injection. They further show that retrieval quality matters more than simply increasing Top-$K$, and that the same serving policy can exhibit substantially different cost-benefit profiles across short-output and decode-heavy regimes. These findings suggest that external experience is best treated as a selective, cost-aware serving decision rather than as a universal add-on. Overall, in the settings studied here, external experience pays off only when both the serving interface and the task-specific cost structure make its quality gains worth the online cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05727v2">LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Collaborative edge computing uses edge nodes in different locations to execute tasks, necessitating dynamic task offloading decisions to maintain low latency and high reliability, especially under unpredictable node failures. Although deep reinforcement learning (DRL) and large language models (LLMs) have shown promise for task offloading, DRL often suffers from poor sample efficiency and local optima, while LLMs are difficult to use directly due to inference overhead and output uncertainty. To address these limitations, we propose \textbf{LeDRL}, a hybrid decision framework that couples a \emph{lightweight LLM} with self-attention-enhanced DRL for real-time task offloading. LeDRL constructs structured, context-aware prompts capturing node status, task semantics, and link dynamics to derive high-level strategy priors. These are selectively processed by a self-attention-based alignment module for context-aware policy optimization. A reflective evaluator further distills semantic feedback from past trajectories to refine subsequent prompts and provide consistent guidance. Extensive experiments show that LeDRL outperforms representative baselines in task success rate, convergence speed, and real-time responsiveness across diverse network scales, achieving over 17\% improvement in success rate. Furthermore, we deploy LeDRL on Jetson-based edge devices using our prototype system \textit{CoEdgeSys}, demonstrating its robustness and feasibility under resource constraints. Our code is available at:https://github.com/GalleyG5/LeDRL.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04567v3">GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Accepted as an oral presentation at the GFM @ ICML 2026 Workshop
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) are powerful tools for processing relational data but often struggle to generalize to unseen graphs, giving rise to the development of Graph Foundational Models (GFMs). However, current GFMs are challenged by the extreme heterogeneity of graph data, where each graph can possess a unique feature space, label set, and topology. To address this, two main paradigms have emerged. The first leverages Large Language Models (LLMs), but is fundamentally text-dependent, thus struggles to handle the numerical features in vast graphs. The second pre-trains a structure-based model, but the adaptation to new tasks typically requires a costly, per-graph tuning stage, creating a critical efficiency bottleneck. In this work, we move beyond these limitations and introduce \textbf{G}raph \textbf{I}n-context \textbf{L}earning \textbf{T}ransformer (GILT), a framework built on an LLM-free and tuning-free architecture. GILT introduces a novel token-based framework for in-context learning (ICL) on graphs, reframing classification tasks spanning node, edge and graph levels in a unified framework. This mechanism is the key to handling heterogeneity, as it is designed to operate on generic numerical features. Further, its ability to understand class semantics dynamically from the context enables tuning-free adaptation. Comprehensive experiments show that GILT achieves stronger few-shot performance with significantly less time than LLM-based or tuning-based baselines, validating the effectiveness of our approach. Our code is available at: https://github.com/yiming421/inductnode/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08211v2">MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 26 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are moving from cloud-centric services toward on-device embedded AI, where models interact with private, longitudinal signals sensed from users and their physical environments. Mobile phones are a natural platform for such applications because they are continuously carried by users, connected to wearable sensors, and deeply integrated with daily mobile applications. However, practical LLM fine-tuning on commodity phones remains difficult. Existing fine-tuning frameworks are largely Python-based and server-oriented, making them hard to deploy inside mobile applications. We present MobileFineTuner, a mobile-native open-source framework for end-to-end LLM fine-tuning on commodity mobile phones. MobileFineTuner is implemented in C++ and provides a reusable training stack. To make fine-tuning feasible under mobile resource constraints, MobileFineTuner integrates a resource-aware training runtime with memory-efficient attention, activation checkpointing, gradient accumulation, parameter sharding, and energy-aware scheduling. We evaluate MobileFineTuner on real mobile phones using GPT-2, Gemma 3, and Qwen2.5 models across multiple fine-tuning tasks. The results show that MobileFineTuner reproduces standard Full-FT and LoRA fine-tuning behavior, substantially reduces memory pressure and improves executability on memory-constrained phones. We further demonstrate MobileFineTuner through a private campus health-agent application, where a local LLM is fine-tuned on user-specific wearable-sensing records to provide more personalized responses while keeping raw records on the phone. These results establish MobileFineTuner as a practical toolkit for studying and building on-device LLM fine-tuning applications in embedded AI and sensing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10546v2">SkillAxe: Sharpening LLM-Authored Agent Skills Through Evaluation-Guided Self-Refinement</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 9 pages, under review
    </div>
    <details class="paper-abstract">
      Skill documents, structured natural-language instructions that guide Large Language Model (LLM) agents, are critical to modern agent frameworks, yet LLMs struggle to write skills that actually work. On SkillsBench, human-authored skills improve pass rates by 16.2 percentage points, while LLM-authored skills provide no measurable gain. We introduce SkillAxe, a fully unsupervised framework that enables LLMs to iteratively diagnose and refine their own skills. SkillAxe decomposes skill quality into four interpretable dimensions (quality impact, trigger precision, instruction compliance with fault attribution, and solution-path coverage), producing structured improvement briefs that require no ground-truth labels, test suites, or environment rewards. On SkillsBench, SkillAxe improves pass rates by 28\% relative over unimproved LLM skills and closes 47--67\% of the gap to human-authored skills. We validate the approach as a continuous improvement engine in the wild on SpreadsheetBench, where a SkillAxe-built skill library learns from past agent trajectories and raises pass rate from 16.0\% to 52.0\% using only 22 skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23248v3">Resource-Aware LLM Reasoning for Mobile Edge General Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled an emergence of agentic artificial intelligence (AI) with powerful reasoning and autonomous decision-making capabilities. This integration with edge computing has led to the development of Mobile Edge General Intelligence (MEGI), which brings real-time, privacy-preserving reasoning to the network edge. However, deploying LLM-based agentic AI reasoning in MEGI environments poses significant challenges due to the high computational demands of reasoning and the limited resources of edge devices. To address these challenges, we propose a joint optimization framework for efficient LLM reasoning deployment in MEGI. First, we systematically review enhancement methods to identify mechanisms suitable for edge adaptation. Subsequently, we present a distributed framework that synergizes reasoning enhancement via adaptive CoT prompting with scalable deployment through a distributed MoE architecture. An important innovation of this approach involves modeling reasoning depth as a dynamic network resource variable, which is optimized jointly with expert activation and transmission power. This mechanism allows the system to dynamically regulate expert networks and reasoning complexity according to task requirements and device capabilities. Experimental evaluations in mobile edge environments demonstrate that the proposed framework effectively balances reasoning quality and resource efficiency. The results show that with less than one second of additional inference time, both accuracy and latency satisfaction rate can reach 90\%, validating the practical viability of deploying sophisticated LLM reasoning in resource-constrained MEGI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12481v1">Representing Time Series as Structured Programs for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong reasoning and instruction-following capabilities, making them potentially powerful tools for time-series analysis. However, time series lie outside their native textual modality, raising a fundamental question: how should time series be represented so that LLMs can reason about them effectively? Existing work typically serializes raw numerical sequences or fine-tunes pre-trained LLMs on time-series data. These approaches place the burden of extracting temporal structure directly on the LLM, creating a modality mismatch that often degrades performance on long sequences and introduces substantial computational overhead. In this work, we introduce Time-Series-to-Structured-Program representation (T2SP), a deterministic, training-free method that represents a time series as a structured symbolic program. T2SP decomposes time series into trends, periods, and salient events, expressing them in a program-friendly format aligned with the textual and code-like modalities on which LLMs are natively trained. By shifting temporal-structure extraction from the model to the representation itself, T2SP enables off-the-shelf LLMs to leverage their existing reasoning capabilities for time-series understanding. We evaluate T2SP on three reasoning tasks -- editing, captioning, and question answering -- where it consistently improves performance, reduces reasoning time, and lowers failure rates compared with raw-string representations. Our results demonstrate that T2SP provides an effective interface between time series and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22025v2">When Generic Prompt Improvements Hurt: Evaluation-Driven Iteration for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Technical report. 42 pages, 3 figures. Code, test suites, and result logs: https://github.com/dcommey/llm-eval-benchmarking
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Model (LLM) applications differs from conventional software testing because outputs are probabilistic, semantically variable, and sensitive to prompt and model changes. This technical report proposes the Minimum Viable Evaluation Suite (MVES), an audit-oriented structure for application-level LLM evaluation. MVES links application categories to failure modes, metrics, required artifacts, and validation evidence across general LLM applications, retrieval-augmented systems, and agentic workflows. We pair the framework with a reproducible local evaluation harness covering structured extraction, RAG citation/content-compliance, and instruction-following checks. Using Ollama with Llama 3 8B Instruct and Qwen 2.5 7B Instruct, we evaluate five prompt conditions over expanded 30-case-per-suite ablations. The results show that, in the tested local conditions, generic prompt additions do not produce monotonic improvements: stronger output-contract prompts improve strict extraction for both models, while RAG citation/content-compliance declines under some generic-rule conditions. The largest observed decline occurs for Qwen 2.5 on RAG when generic rules are appended to the user prompt, from 26/30 to 9/30. These findings support evaluation-driven prompt iteration: prompt changes should be treated as potential regression risks and tested against task-specific suites before deployment. The accompanying repository contains the test suites, prompt variants, evaluation harness, raw result logs, and scripts needed to reproduce the reported local ablations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23243v2">Are Frontier LLMs Ready for Cybersecurity? Evidence for Vertical Foundation Models from Dual-Mode Vulnerability Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      We evaluate whether frontier LLMs are ready for cybersecurity through a dual-mode benchmark: white-box function-level vulnerability detection (VulnLLM-R, across C/Java/Python) and black-box web application security testing (five production-style applications with 118 ground-truth vulnerabilities across 20+ CWE families, which we will open-source). We test six frontier models (GPT-5.4, Codex~5.3, Claude Opus~4.6, Sonnet~4.6, Gemini~3.1~Pro and Gemini~3~Flash) and two domain-specialized models across four testing paradigms. Our findings are sobering: (1)~every frontier model produces 10-50% false positive rates in white-box detection, systematically over-predicting vulnerabilities; (2)~in black-box testing, frontier models achieve only 4-8% ground-truth coverage, improving to just 10-19% even with external security tools (Playwright MCP, Burp Suite MCP); (3)~structured penetration-testing methodology encoded in domain-specialized agents raises per-family detection above 50%, demonstrating that methodology, not scale, is the primary lever; and (4)~a domain-specialized defense model achieves the highest precision (0.904) and lowest false positive rate (9.7%) among all models, on a single GPU. We identify the absence of structured security testing traces end-to-end request/response sequences, failure-heavy data, and multi-step attack chains as the fundamental training data bottleneck, and propose self-play security testing as a data generation strategy. Our results make the case for vertical foundation models purpose-built for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11470v1">The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved strong performance across natural language processing tasks, yet reliable reasoning remains an open challenge. Although modern LLMs show progress in structured inference, multi-step problem solving, and contextual understanding, their reasoning behavior is often inconsistent and sensitive to prompting strategies, task design, and model scale. This survey provides a systematic analysis of more than 300 recent papers from arXiv, Semantic Scholar, Google Scholar, Papers with Code, and the ACL Anthology to examine how reasoning capabilities emerge in LLMs and where they fail. We make three main contributions. First, we introduce a structured taxonomy of LLM reasoning research, covering Chain-of-Thought reasoning, multi-hop reasoning, mathematical reasoning, common sense reasoning, visual and temporal reasoning, code and algorithmic reasoning, retrieval-augmented reasoning, tool-augmented and agentic reasoning, and reinforcement learning-based reasoning. Second, we analyze methodological trends across these paradigms, including prompting methods, model architectures, training objectives, reward modeling, and evaluation benchmarks. Third, we synthesize recurring limitations and failure modes, such as reasoning hallucinations, brittle multi-step inference, weak causal abstraction, and poor cross-domain generalization. By organizing a rapidly expanding literature, this survey offers a unified view of the current capabilities and limitations of reasoning in LLMs. We also identify emerging research directions, including meta-reasoning, self-evolving reasoning frameworks, multimodal reasoning, and socially grounded reasoning. Overall, this work aims to serve as a reference for developing more robust, interpretable, and generalizable reasoning systems in future language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12164v3">The Language You Ask In: Language-Conditioned Ideological Divergence in LLM Analysis of Contested Political Documents</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as analytical tools across multilingual contexts, yet their outputs may carry systematic biases conditioned by the language of the prompt. This study presents an experimental comparison of LLM-generated political analyses of a Ukrainian civil society document, using semantically equivalent prompts in Russian and Ukrainian administered to two frontier models from different developers, ChatGPT 5.2 and Claude Opus 4.5. Despite identical source material and parallel query structures, both models diverged along the same axis: Russian-language outputs leaned toward delegitimizing framings, characterizing civil society actors as externally funded elites constraining a democratic mandate, while Ukrainian-language outputs treated the same actors as legitimate stakeholders in democratic contestation. The magnitude of this divergence, however, was model-dependent. ChatGPT's Russian output reproduced vocabulary characteristic of Russian state discourse; Claude Opus's stayed in a mainstream critical idiom and hedged its judgments in both languages. These findings demonstrate that prompt language alone can systematically shift the ideological orientation of an unchanged model analyzing identical content. The shift is a general property of multilingual LLMs whose severity, and whose alignment with propaganda narratives, varies across systems. The implications reach AI deployment in polarized information environments, cross-lingual research, and AI governance in multilingual societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17717v3">A Survey on Evaluating Quality and Trustworthiness in LLM-Generated Data</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Published at TMLR. Title changed in the final version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11416v1">MPC-Patch-Bench: Security-Aware LLM Code Patch for Multi-Party Computation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Repository-level benchmarks for evaluating Large Language Model (LLM) code repair on Secure Multi-Party Computation (MPC) software do not yet exist, and directly transplanting general-purpose benchmarks such as SWE-bench fails on three structural fronts: (i) MPC repositories are dominated by generic Python infrastructure rather than cryptographic logic; (ii) high-value MPC fixes lack the standardized tests rigid extraction pipelines require; and (iii) standard fail-to-pass evaluation is insufficient for code that must also be cryptographically safe. MPC is increasingly deployed for privacy-preserving machine learning, biomedical collaboration, and secure analytics. Existing MPC-specific code-synthesis efforts cover only operator-level or single-framework tasks; evaluating LLM agents on real repository-level MPC repair instead demands MPC-aware data curation and a verifier matched to the security and numerical-fidelity guarantees MPC programs must obey neither of which existing benchmarks provide. We introduce MPC-Patch-Bench, a repository-level benchmark organised around two frameworks. (1)The Data Curation Framework combines a domain-specific curation agent that filters raw pull requests through three cryptographic layers with a human-AI completion engine that synthesizes missing problem statements and Fail-to-Pass/Pass-to-Pass tests, yielding 205 fully verified instances. (2)The MPC Verifier provides dedicated security and numerical-fidelity checks via dynamic differential testing against plaintext oracles and MPC-specific static analysis rules that flag unsafe reveals, insecure arithmetic, and illegal public/private casts. The strongest evaluated LLM functionally resolves only 22.9% of MPC-Patch-Bench tasks; the MPC Verifier further reduces verified resolution to 17.1%, with up to 40% of functionally-passing patches rejected for cryptographic or numerical-fidelity violations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09885v6">Diffusion-Inspired Masked Fine-Tuning for Knowledge Injection in Autoregressive LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used in environments where facts evolve, yet factual knowledge updates via fine-tuning on unstructured text often suffer from 1) reliance on compute-heavy paraphrasing augmentation and 2) the reversal curse. Recent studies show diffusion large language models (dLLMs) require fewer training samples to achieve lower loss in pre-training and are more resistant to the reversal curse, suggesting dLLMs may learn new knowledge more easily than autoregressive LLMs (arLLMs). We test this hypothesis in controlled knowledge fine-tuning experiments and find that while arLLMs rely on paraphrase augmentation to generalize knowledge text into question-answering (QA) capability, dLLMs do not require paraphrases to achieve high QA accuracy. To further investigate whether the demasking objective alone can induce such a knowledge injection advantage in dLLMs regardless of their diffusion denoising paradigm, we propose masked fine-tuning for arLLMs, which prompts an arLLM to reconstruct the original text given a masked version in context. The masked fine-tuning for arLLMs substantially improves the efficacy of knowledge injection, i.e. no paraphrase needed and resistant to the reversal curse, closing the gap between arLLMs and dLLMs. We also demonstrate broader applicability: on a large-scale knowledge-intensive dataset (1.2M samples), masked SFT achieves the best downstream accuracy on GPQA-diamond among all fine-tuning variants. The demasking objective also improves SFT on math tasks, suggesting broad utility beyond factual knowledge injection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11379v1">Automated Mediator for Human Negotiation: Pre-Mediation via a Structured LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Pre-mediation, the preparatory phase preceding direct human negotiation, plays a critical role in achieving mutually beneficial agreements, yet is often omitted due to cost, time, and limited access to trained mediators. We introduce an automated mediator for human negotiation, implemented as a structured pipeline of LLM modules, that supports pre-mediation in integrative negotiation settings. The pipeline decomposes preparation into specialized modules for dialogue, preference prediction, response-level critique, and structured summarization, separating inference, generation, and evaluation to address limitations of monolithic single-prompt approaches. We use the term "agent" for each module following common LLM-systems terminology, but the components are not autonomous and do not interact peer-to-peer; outputs are passed forward in a fixed sequence. We evaluate the system in two controlled human-subject experiments comparing AI-based pre-mediation with professional human mediators in a multi-issue negotiation scenario. On short-term self-reported measures, the automated mediator achieves preparation outcomes broadly comparable to human mediators, including trust in the mediator and confidence in reaching mutually beneficial agreements, while achieving substantially lower error on the preference-inference task under our scenario and prompts (36% lower RMSE). A second study shows that targeted prompt refinements reduce excessive affirmation patterns from 36.6% to 16.8%, matching human mediator baselines. Our findings suggest that structured LLM pipelines can provide scalable, low-effort pre-mediation support broadly comparable to human mediators on short-term self-reported preparation outcomes. The pipeline's single-party design mirrors how human mediators run pre-mediation today and enables parallel deployment across all parties to a dispute, supporting scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11357v1">TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 13 pages excluding reference, 11 figures
    </div>
    <details class="paper-abstract">
      With the growing demand for on-device LLM inference, edge SoCs increasingly integrate NPUs to improve performance and energy efficiency under tight power and thermal budgets. However, practical LLM deployment on current client NPUs remains difficult: widely used quantization formats such as AWQ do not map cleanly onto many existing NPU software stacks, which are often proprietary and expose limited low-level control. In this work, we present \textit{TileFuse}, a close-to-metal mixed-precision kernel library for AMD XDNA2 NPUs that targets transformer linear layers in quantized LLM inference. TileFuse brings practical low-bit formats such as AWQ-style W4A16 and W8A16 directly onto XDNA2, rather than forcing the model to be reshaped around an NPU-specific quantization scheme. TileFuse co-designs weight layout, metadata placement, mixed-precision microkernels, and array-level dataflow. Specifically, it fuses unpacking, dequantization, and GEMM/GEMV execution into a single kernel flow, introduces an interleaved pre-tiling layout that supports GEMM dimensions up to 32K, and redesigns GEMV dataflow to utilize the full 4x8 AIE array. Across kernel-level evaluations, TileFuse improves performance by up to 121.6% for GEMM and 281% for GEMV over full-precision baselines, while delivering more than 2x performance and energy-efficiency gains over strong iGPU baselines on GEMM. In end-to-end LLM experiments on Ryzen AI laptops, TileFuse achieves up to 2.0x lower prefilling latency with more than 64.6% lower energy consumption. Together, these results show that XDNA2 is a practical target for AWQ-style edge LLM inference and that native NPU support for off-the-shelf quantization can make NPUs substantially more usable in real client deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11316v1">Schützen: Evaluating LLM Safety in Bulgarian and German Contexts</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 19 pages, 13 tables, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed across professional domains, bringing hard-to-predict risks, including the generation of harmful or disrespectful content. Although substantial progress has been made in developing safety evaluation datasets, existing resources remain overwhelmingly English- and Chinese-centric. This limitation is particularly pronounced when evaluating languages that operate within shared sociocultural, legal, and ethical contexts. To address this gap, we introduce Schützen: a German--Bulgarian safety dataset designed to assess model answerability under risk, covering both a low-resource language (Bulgarian) and a high-resource language (German). Experiments with multilingual and language-specific LLMs reveal pronounced cross-language differences in safety behavior, highlighting the necessity of tailored, region-specific evaluation resources to support the responsible deployment of LLMs in Germany and Bulgaria. Datasets and code are available at https://github.com/xnlp-lab/Schutzen. Warning: this paper contains examples that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11166v1">Flaws in the LLM Automation Narrative</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly described as performing at the level of human experts on knowledge economy tasks. These claims are primarily based on how LLMs perform on benchmarking tasks that measure average performance across standardized datasets. Primary limitations of many benchmarking tasks are that they often measure performance based on content directly included in LLM training data, and they frequently do not assess the reliability of LLM performance or the magnitude of LLM errors. However, in high stakes contexts, these qualities are critically important. Through a novel LLM benchmarking task that requires writing computer code to complete a data analysis task, we compare the performance of a frontier LLM against submissions from human experts and explicitly measure the variance of responses and the magnitude of errors. Our study reveals that the human experts perform better on average on a range of metrics and demonstrate less variability in performance. Our results provide evidence that LLMs do not consistently perform at the level of human experts and demonstrate the importance of measuring variance and assessing error magnitude in LLM benchmark evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22714v3">AMEL: Accumulated Message Effects on LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 24 pages, 14 figures, 8 tables. Single author. Code, data (84,088 deduplicated API responses), and analysis pipeline at https://github.com/chutapp/amel
    </div>
    <details class="paper-abstract">
      Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 84,088 API calls to 12 models from 5 providers (OpenAI, Anthropic, Google, DeepSeek, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations. Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-53). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.36 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.52x more bias than positive (t = 13.03, p < 10^-36, n = 2,733). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17). Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11145v1">OpenPCC: Open and Confidential LLM Serving on Commodity TEEs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Generative AI applications such as personal AI agents, image generators, and chat assistants offer advanced capabilities to improve user experience. Behind the scenes, Large Language Models (LLMs) that power these services require a massive amount of computation and are usually deployed in the cloud, available as APIs, meaning that a user's request has to be sent to a Cloud Inference Service (CIS) for processing. However, the strong capabilities of LLM also mean that user's requests now contain much more personal sensitive or enterprise confidential information, demanding equally strong protection in CIS. While early industry efforts such as Apple Private Cloud Compute (PCC) and Google Private AI Compute have emerged to show the potential of secure CIS, they are not adoptable for deployment by others due to their reliance on proprietary hardware and closed ecosystem. In addition, they all suffer from their own design glitches that can undermine the ambitious goal of bringing in true privacy protection to end users. In this paper, we present our analysis of the fundamental requirements of building a secure yet open CIS. We then present OpenPCC, a Confidential CIS framework that does not rely on proprietary hardware but instead uses commercially available TEEs. We implement an open-source prototype and characterize it end-to-end on a Llama-3 8B vLLM workload, separating OpenPCC's own cost from the underlying TEE hardware. Our analysis and evaluation demonstrated the feasibility and security of the system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05463v2">PSEBench: A Controllable and Verifiable Benchmark for Evaluating LLMs in Patient Safety Event Triage</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Patient safety event triage, determining whether a clinical event is reportable under jurisdiction-specific policy, is a high-stakes task typically performed manually by patient safety experts. Although LLMs may support this workflow, reliable evaluation is limited by the lack of benchmarks to capture evidence-grounded policy reasoning, proactive information seeking for incomplete reports, and principled abstention in irreducibly ambiguous cases. We address this gap with a policy-grounded construction methodology centered on the clause card, a structured representation that factorizes regulatory text into auditable decision specifications. Combining clause cards with anchor-driven instantiation and closed-loop verification, our scalable pipeline produces narratives with by-construction ground truth and naturally supports generating missing information and uncertain variants. We instantiate this method on Minnesota's 29 Reportable Adverse Health Events, producing PSEBench, a 5,074-case benchmark with an agentic evaluation environment. Evaluation on 15 representative LLMs reveals consistent capability trends, demonstrates the benchmark's utility, and identifies actionable gaps toward reliable LLM-based patient safety event triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01993v2">SAFE: An LLM-as-Verifier Framework for Evidence-Grounded Multi-Hop Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Multi-hop QA benchmarks often reward Large Language Models (LLMs) for spurious correctness, where models reach correct answers through invalid intermediate reasoning. We propose SAFE, an LLM-as-verifier framework for evidence-grounded multi-hop QA. Rather than judging only the final answer after generation, SAFE verifies reasoning during generation by checking intermediate steps against the provided passages and previous reasoning trajectory. To make this process checkable, SAFE decomposes reasoning into atomic, evidence-grounded units represented with Knowledge Graph (KG) triples. At train-time, SAFE verifies benchmark supervision under KG-grounded constraints and constructs reliable verifier training data. At inference-time, an external verifier checks each generated step, identifies invalid reasoning, and provides correction feedback before errors propagate. Across three multi-hop QA benchmarks, SAFE improves accuracy by 8.8 pp on average. These results show that evidence-grounded multi-hop QA benefits from shifting LLM-based evaluation from post-hoc answer judgment to stepwise reasoning verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11081v1">Unifying Local Communications and Local Updates for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 38 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Communication-efficient pre-training of LLMs is increasingly important as training draws on compute distributed across clusters, data centers, and lower-bandwidth links. Many practical methods reduce communication frequency but still rely on synchronous All-Reduce operations that maintain identical model states and tie progress to global collectives. This can become a bottleneck when bandwidth or worker speed is heterogeneous. We introduce GASLoC, a novel decentralized pre-training algorithm that generalizes the notion of communication acceleration to the recently popular "outer optimizer" to allow a practical gossip-based training framework that is compatible with adaptive optimizers, allows for local optimizer steps, and can utilize sparse randomized peer communication. Empirically, on a number of standard LLM training tasks, we demonstrate that GASLoC outperforms state-of-the-art decentralized algorithms in single step per communication setting for a number of topologies and, unlike existing decentralized methods in the LLM setting, it allows to obtain performance competitive with DiLoCo when utilizing multiple local steps. In the heterogeneous bandwidth setting we demonstrate the advantage of GASLoC showing that it can significantly outperform DiLoCo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11050v1">LLM-Mediated Demand Response Coordination in Smart Microgrids</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted for publication in 18th International Conference on Sustainability in Energy and Buildings (SEB-26), to appear in Springer Nature proceedings (KES Smart Innovation Systems and Technologies). The final authenticated version will be available online at Springer
    </div>
    <details class="paper-abstract">
      Effective demand response in smart microgrids requires prosumers to cooperate voluntarily under strategic self-interest, a coordination problem structurally equivalent to a repeated Prisoner's Dilemma on a social network. This paper presents a multi-agent simulation in which a Large Language Model (LLM) Influence Compiler issues structured demand-response directives to a population of heterogeneous prosumer agents, each governed by a hybrid decision architecture combining game-theoretic base probability (derived from payoff history, neighbour imitation, and exploitation memory) with LLM narrative evaluation of incoming coordination signals. The hybrid architecture resolves a key methodological challenge: LLMs aligned via Reinforcement Learning from Human Feedback (RLHF) exhibit strong cooperation bias when used as direct decision-makers, producing flat dynamics regardless of grid conditions. By separating strategic reasoning from grounded narrative evaluation, the model generates realistic prosumer behaviour across six personality archetypes, with baseline cooperation near 50% and clear differentiation under influence. Compiled structured directives achieve 33.3% demand-curtailment cooperation versus 27.0% for unstructured messaging and 28.0% for a no-intervention baseline ($Δ_\mathrm{comp} = +0.063$), with the advantage preserved across both grounded and idealized agent substrates ($Δ= +0.083$) and across all resistance levels ($R = 0.1$ to $0.7$). Hub-targeted dissemination via high-centrality network nodes outperforms peripheral or random targeting, confirming that grid topology provides mechanistic amplification independent of message content. These results suggest that structured LLM compilation, grounded agent reasoning, and network-aware targeting are complementary design principles for scalable, interpretable demand-response coordination in smart-city energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11033v1">AuRA: Internalizing Audio Understanding into LLMs as LoRA</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Recent efforts to extend large language models (LLMs) to speech inputs typically rely on cascaded ASR-LLM pipelines, end-to-end speech-language models, or bridge/distillation-based adaptation. While these routes respectively reuse strong pretrained components, enable native speech-language interaction, or offer lightweight adaptation, they often suffer from transcript-interface latency, costly multimodal training, or sequential speech-language coupling. To address these limitations, we present AuRA, a method that distills audio encoding capability into the LLM. Specifically, AuRA feeds the same speech input to an ASR encoder (as a teacher) and a LoRA-adapted LLM (as a student) through a lightweight audio embedding layer, and uses layer-wise distillation to align the student's hidden states with corresponding teacher representations, thereby internalizing speech representations into lightweight LLM-side adaptations. Compared with cascaded and serial bridge methods, AuRA enables tighter speech-language joint modeling and efficient parallel end-to-end inference, while also reusing pretrained speech and language models rather than requiring large-scale multimodal training. On multiple speech-language benchmarks, AuRA consistently outperforms cascaded systems, speech-to-LLM adaptation baselines, and large-scale speech-language and multimodal models in both effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21050v2">ERM-MinMaxGAP: Benchmarking and Mitigating Gender Bias in Multilingual Multimodal Speech-LLM Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 This paper has been accepted for presentation at INTERSPEECH 2026
    </div>
    <details class="paper-abstract">
      Speech emotion recognition (SER) systems can exhibit gender-related performance disparities, but how such bias manifests in multilingual speech LLMs across languages and modalities is unclear. We introduce a novel multilingual, multimodal benchmark built on MELD-ST, spanning English, Japanese, and German, to quantify language-specific SER performance and gender gaps. We find bias is strongly language-dependent, and multimodal fusion does not reliably improve fairness. To address these, we propose ERM-MinMaxGAP, a fairness-informed training objective, which augments empirical risk minimization (ERM) with a proposed adaptive fairness weight mechanism and a novel MinMaxGAP regularizer on the maximum male-female loss gap within each language and modality. Building upon the Qwen2-Audio backbone, our ERM-MinMaxGAP approach improves multilingual SER performance by 5.5% and 5.0% while reducing the overall gender bias gap by 0.1% and 1.4% in the unimodal and multimodal settings, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11018v1">Measuring Human Value Expression in Social Media Texts: Calibrated LLM Annotation and Encoder Transfer</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Measuring subjective constructs in naturally occurring social media text requires annotation procedures that are theoretically grounded, empirically validated, and transferable to an encoder model for scalable prediction. Using non-English social media posts annotated according to Schwartz's theory of basic human values, we investigate how different LLMs, prompts, and instruction languages operationalize the expression of values in text. We argue that although texts may permit multiple plausible interpretations, theory-based value definitions can constrain interpretations and reduce spurious value attributions. Beyond precision, recall, and F1, we evaluate structural alignment between values, error structure, confidence-ambiguity relations, and annotation stability. We show that different LLMs produce different value interpretations. Iterative prompt calibration through error analysis reduces misattributions and improves alignment with expert annotations. We also derive targeted expert verification rules from recurrent error structures and use them during corpus annotation. Finally, we show that LLM annotations can be transferred to an encoder model through soft-label training, retaining theory-based value interpretations and information about uncertainty in value expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11016v1">Superficial Beliefs in LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      We ask whether large language models (LLMs) merely imitate rationales when choosing between two options, or whether their choices reflect a systematic underlying decision structure. Using synthetic binary decision settings in which models choose between profiles defined by graded attributes, we compare the attribute a model says mattered most with the attribute that best explains its choice under a behavioural model fit to prior decisions. The behavioural model predicts held-out choices well, showing that model behaviour is systematically related to the visible attributes rather than being random. However, direct self-reports and a separate score-based judge recover the behaviourally inferred driver only partially. The resulting picture is neither one of arbitrary behaviour nor one of fully articulated belief - outputs are structured enough to support prediction, but explicit reasons track the recovered driver only imperfectly. This qualitative pattern persists across prompt-order and sampling perturbations, alternative behavioural models, targeted occlusion analyses, and structurally varied decision settings. We interpret this as evidence for ``superficial belief'' in LLM decision-making: models behave as if guided by probabilistic local priorities over attributes, while having only limited verbal access to the attributes that drive their decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11015v1">Structure from Reasoning, Numbers from Search: On-Premise Open LLMs as Structural Priors for Coupled MIMO Controller Tuning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 10 pages, 7 figures, 6 tables. Submitted to IEEE Access
    </div>
    <details class="paper-abstract">
      Tuning controllers for strongly coupled multi-input multi-output (MIMO) industrial processes is hard: decentralized classical auto-tuning ignores loop interaction, and local numerical optimization from natural initializations stalls in the resulting non-convex cost landscape. We ask whether on-premise open-source large language models (LLMs), which keep data on-site and need no plant model, can help. On a single-loop CSTR, classical relay-feedback tuning (IAE 0.106, near the 0.102 optimum) beats an LLM tuner (0.162): for simple loops the LLM adds nothing. The picture inverts on a strongly coupled quadruple-tank with conflicting set-points, scored by a penalized cost J = IAE + lambda*TV(u) that rewards tracking without chattering actuators. There, naive relay tuning (J ~ 28.6) and naive LLM tuning (29.7) are no better than open loop (22.7), and a local optimizer from balanced starts fails in 10/10 runs. A scaffolded open LLM instead reasons about the coupling, proposes the counter-intuitive asymmetric structure, and reaches J ~ 16.9 +/- 0.2 from any start; refining it with a classical optimizer attains the smooth global optimum (J ~ 12.0, 10/10 vs. 0/10), which even applies a non-obvious negative integral correction decentralized tuning cannot. A global optimizer (differential evolution) also reaches this optimum, so the LLM is not the only route; its advantage is sample efficiency and interpretability: a usable controller in 18 evaluations (where the global optimizer is worse than open loop) plus a stated rationale. This edge grows with dimension, reaching ~6x fewer evaluations on a 3x3 plant. The behaviour generalizes across four open models, and on a benign plant the LLM offers no advantage, sharpening the boundary. We contribute a reproducible benchmark delimiting when open LLMs help in control tuning: not as optimizers, but as a sample-efficient, interpretable structural prior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02221v2">MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs for Clinical Tabular Prediction</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      In clinical tabular prediction, classical machine learning models with feature engineering often outperform neural methods. LLMs are increasingly used to automate this process, acting as domain experts that propose diverse feature transformations to boost downstream performance. However, existing LLM-based methods decouple feature generation from the downstream model: the LLM receives no signal about which features currently drive predictions or where the model's representational capacity falls short, so proposals are neither targeted to promising regions of the feature space nor tailored to the learner's inductive bias. This shortcoming is amplified in healthcare data, which simultaneously exhibits class imbalance, heterogeneous feature spaces, and strict interpretability requirements. In this paper, we propose MedFeat, the first feature engineering framework inspired by the workflow of machine learning practitioners, leveraging model-awareness and feature importance signals to iteratively guide feature discovery for clinical tabular learning. We evaluate MedFeat on a broad range of challenging real-world clinical tasks and show that it statistically significantly outperforms state-of-the-art baselines, with an average improvement of more than 10% over the baseline across models with distinct inductive biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v3">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 ICML 2026 Workshop on Graph Foundation Models
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24668v3">The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICLR 2026 FinAI Workshop
    </div>
    <details class="paper-abstract">
      Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10956v1">Mind the Gap: Can Frontier LLMs Pass a Standardized Office Proficiency Exam?</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Model (LLM) agents for computer automation is accelerating, yet their ability to navigate complex, professional-grade productivity software is largely untested. We argue that Office automation is an ideal environment for benchmarking document-automation capability, as it requires long-horizon planning and reasoning, precise parameter configuration, and multi-application integration. To quantify this capability, we introduce an evaluation based on China's National Computer Rank Examination (NCRE), featuring 200 comprehensive practical-operation tasks across Word, Excel, and PowerPoint. Each task is scored on a 100-point rubric scale using 7,118 machine-gradable criteria, and Score Rate (SR) denotes the mean percentage of rubric points earned across these tasks. We benchmark 7 frontier LLMs and observe stark limitations: single-turn models score a maximum of 36.6%. A stronger agentic system with execution feedback, iterative repair, and broader Office automation access reaches 68.8%, but remains below the 95.5% community-reference score used as a scoring sanity check. Ultimately, our experiments demonstrate that despite recent advancements in code generation, achieving reliable fine-grained Office document automation remains a significant challenge for current code-generating LLM and agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10942v1">Generative Explainability for Next-Generation Networks: LLM-Augmented XAI with Mutual Feature Interactions</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 7 pages, with one page for appendix. Accepted for publication at the 2025 21th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob)
    </div>
    <details class="paper-abstract">
      As artificial intelligence and machine learning (AI/ML) models become integral to network operations, their lack of transparency poses a significant barrier to operator trust. Existing explainable artificial intelligence (XAI) techniques often fail to bridge this gap for non-specialists, producing technical outputs that are difficult to translate into actionable insights. This paper presents a framework specifically designed to address this shortcoming. It leverages a moderately sized large language model (LLM) and extends beyond the standard use of SHapley Additive exPlanations (SHAP) feature influence values. The framework employs a structured prompt enriched with mutual feature interaction data to generate human-understandable natural language explanations. To validate our framework, we performed an empirical evaluation on an optical quality of transmission (QoT) estimation use case with human evaluators. We collected independent performance evaluations from specialists, which showed a high inter-evaluator agreement. Compared to a state-of-the-art baseline that uses only SHAP feature influence values in a straightforward prompt, our approach improves the explanation usefulness and scope by 12.2% and 6.2%, while achieving 97.5% correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14717v2">What Really Matters for Table LLMs? A Meta-Evaluation of Model and Data Effects</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Table modeling has progressed for decades. In this work, we revisit this trajectory and highlight emerging challenges in the LLM era, particularly the paradox of choice: the difficulty of attributing performance gains amid diverse base models and training sets in the context of table instruction tuning. We replicate four table LLMs by instruction-tuning three foundation models on four existing datasets, yielding 12 models. We then evaluate these models across 16 table benchmarks. Our study is the first to quantitatively disentangle the effects of training data and base model selection, revealing that base model choice plays a more dominant role than the training data itself. Generalization and reasoning remain challenging, inviting future effort on table modeling. Based on our findings, we share our thoughts on the future directions for table modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10917v1">Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 20 pages, including 12 pages of main text and 8 pages of appendix; work in progress
    </div>
    <details class="paper-abstract">
      Although Large Language Model (LLM) agents have demonstrated strong performance on complex tasks, their learning is often limited by inefficient interaction feedback and static training environments, which hinder broader generalization. To address these limitations, this paper introduces Role-Agent, \textcolor{black}{a framework} that harnesses a single LLM to function concurrently as both the agent and the environment, enabling a bootstrapped co-evolution. Role-Agent comprises two synergistic components: World-In-Agent (WIA) and Agent-In-World (AIW). In WIA, the LLM acts as the agent and predicts future states after each action; the alignment between predicted and actual states is then used as a process reward, encouraging environment-aware reasoning. In AIW, the LLM analyzes failure modes from failed trajectories and retrieves tasks with similar failure patterns, thereby reshaping the training data distribution for targeted practice. Experiments on multiple benchmarks show that Role-Agent consistently improves performance, yielding an average gain of over 4\% over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10875v1">Pushing the Limits of LLM Tool Calling via Experiential Knowledge Integration and Activation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) rely on tool use to act as autonomous agents, yet often fail in multi-step execution due to insufficient tool-related knowledge and ineffective knowledge activation. Therefore, we present a systematic study on how knowledge influences tool-use performance, covering the stages of knowledge acquisition, activation, and internalization. In the knowledge acquisition stage, we acquire and evaluate various forms of experiential knowledge, and our analysis shows that simple instance-level knowledge can already provide strong and reliable gains, while abstract intent-level knowledge offers limited benefits. At inference time, to activate knowledge, we find that prompting LLM to expand the depth of reasoning yields diminishing returns, whereas expanding the width of reasoning by parallel sampling with aggregation more effectively activates latent experiential knowledge. At training time, for knowledge internalization, post-training with knowledge-augmented data further improves performance, with reinforcement learning outperforming supervised fine-tuning. Based on these insights, we propose the Knowledge-Augmented Tool Execution (KATE), a knowledge-augmented tool execution framework that integrates experiential knowledge with reasoning-width-expanded inference and knowledge-aware training. Experiments on BFCL-V3 and AppWorld demonstrate consistent and substantial improvements over strong baselines across model scales. Our Code is available at https://github.com/hypasd-art/KATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07586v2">From Human Guidance to Autonomy: Agent Skill System for End-to-End LLM Deployment on Spatial NPUs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to the Machine Learning for Architecture and Systems Workshop (MLArchSys), co-located with ISCA 2026
    </div>
    <details class="paper-abstract">
      Spatial neural processing units (NPUs) provide an energy-efficient platform for edge LLM inference, but efficiently deploying an LLM end-to-end on such hardware remains labor-intensive. Although AI coding agents have begun to lower this cost, existing studies have largely focused on single-kernel optimization rather than end-to-end LLM deployment on resource-constrained spatial NPUs. We present a two-stage methodology, instantiated on the AMD XDNA 2 NPU, that progresses from human-guided development to agent autonomy. In the first stage, we develop a reference deployment of Llama-3.2-1B through human-guided agent assistance. The resulting implementation achieves a speedup of 2.2x on prefill and 4.0x on decode over the hand-optimized baseline, with the optimization trajectory and its lessons recorded as structured documentation throughout. In the second stage, we distill the documentation into an agent skill system consisting of eight phases, orchestrating the optimization and debugging skill sets, with numerical correctness strictly enforced at each phase. Using our agent skill system, we autonomously deploy eight additional decoder-only LLMs (Llama-3.2-3B, SmolLM2-1.7B, Qwen2.5-{0.5B, 1.5B, 3B}, Qwen3-{0.6B, 1.7B, 4B}) end-to-end on the AMD XDNA 2 NPU using the open-source compiler stack. To our knowledge, these models have not previously been deployed on AMD NPUs via any open-source software stack. Each deployment completes in 0.5-4 hours of agent wall time with almost no human guidance, and passes the numerical-correctness gates, demonstrating functional generalization to previously unencountered LLMs. Three of the eight match or exceed the sustained performance of our Llama-3.2-1B reference deployment, suggesting that the resulting implementations can be competitive without additional model-specific human engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13717v3">On Cost-Effective LLM-as-a-Judge Improvement Techniques</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted at the ICML 2026 workshops "Statistical Frameworks for Uncertainty in Agentic Systems" and "Combining Theory and Benchmarks: Towards a Virtuous Cycle to Understand and Guarantee Foundation Model Performance". 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Using a language model to score or rank candidate responses has become a scalable alternative to human evaluation in reinforcement learning from human feedback (RLHF) pipelines, benchmarking, and application layer evaluations. However, output reliability depends heavily on prompting and aggregation strategy. We present an empirical investigation of four drop-in techniques -- ensemble scoring, task-specific criteria injection, calibration context, and adaptive model escalation -- for improving LLM judge accuracy on RewardBench 2, with a unifying lens of noise control on the stochastic judge: ensembling as Monte Carlo averaging over per-call noise, criteria injection as between-response discrimination sharpening, and per-response score variance as an uncertainty signal. Ensemble scoring and task-specific criteria injection (the latter virtually cost free) together reach up to 85.8% accuracy, +13.5pp over baseline. Calibration context and adaptive model escalation also improve over baseline but are dominated by criteria + ensembling on the cost-accuracy Pareto frontier. Small models benefit disproportionately from ensembling, making high-accuracy LLM judges accessible at low cost. We show that these techniques generalise across model providers, evaluating on both OpenAI GPT and Anthropic Claude families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10861v1">From Perception to Action: Can UI Interventions Foster Sustainable LLM Chatbot</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      LLM-powered chatbots are increasingly embedded in everyday workflows, raising sustainability concerns due to their energy use. Most mitigation strategies emphasize model or infrastructure efficiency, while the user-interface (UI) layer remains underexplored despite its potential to shape interaction behavior. We investigate whether sustainability-oriented UI interventions can increase users' energy awareness and encourage more energy-responsible chatbot use without reducing usability. We first conducted a baseline survey with 77 participants to assess awareness and receptiveness to intervention concepts. Guided by prior work on persuasive technology and choice architecture, we implemented a web-based chatbot prototype with a three-mode switch (Energy-efficient, Balanced, Performance), per-response energy feedback, pre-send energy estimates, a usage metrics dashboard, and energy analogies. We then evaluated the prototype in a five-day field study with 11 participants. In the baseline survey, 94.8% of respondents reported at least some awareness of AI energy use, yet 88.3% misestimated actual consumption. Although concern about environmental impact was high, only 39.0% indicated willingness to accept a performance trade-off for lower energy use. In the field study, Energy-efficient mode accounted for 55.8% of logged prompts, while 90.9% self-reported actively choosing Eco-mode when high accuracy was not required. Participants did not reduce prompt length, suggesting mode switching as the primary behavioral mechanism. Sustainability-oriented UI interventions can improve awareness and support more energy-responsible interaction patterns in LLM chatbots. These effects are best interpreted as behavioral and model-based estimates that complement backend efficiency work, and the provided prototype and replication package support further research on energy-aware conversational AI design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10860v1">Training LLMs to Enforce Multi-Level Instruction Hierarchies via Gravity-Weighted Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Production LLMs receive instructions from sources with very different levels of trust, yet attend to every token with uniform architectural privilege. This is the structural vulnerability that enables malicious prompt injections and, more broadly, leaves models without a principled way to resolve conflicts between legitimate but competing instructions. A common training-based response is to teach models an explicit instruction hierarchy; existing approaches, however, formalize hierarchies of only three or four levels, treat all violations as equally severe, and rarely evaluate the full set of pairwise level interactions. We formalize a k-level instruction hierarchy problem and instantiate it for k=5, yielding ten pairwise priority relations that a compliant model must enforce. We then introduce Gravity-Weighted DPO (GW-DPO), a preference-optimization objective whose per-sample offset scales with the structural distance between conflicting levels under a linear or bilateral schedule, the latter weighting severity by both the privilege gap and the privilege of the victim level. Combined with hierarchy-specific delimiter tokens (Chen et al., 2025) and Instructional Segment Embeddings (ISE; Wu et al., 2025), GW-DPO with the bilateral schedule Pareto-improves over standard DPO and the linear variant on Llama-3.1-8B-Instruct, raising macro pairwise priority adherence while keeping over-refusal at half the standard DPO rate. Ablations isolate ISE as a refusal-threshold calibrator and recast five- versus three-level training as a generality-specialization tradeoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10852v1">Janus: A Benchmark for Goal-Conditioned Information Distortion in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      LLM deception is often evaluated through direct markers such as fabricated claims, explicit lies, or strategic concealment. However, many real-world misleading communications do not depend on false statements, rather, they arise from selective treatment of true material facts: omitting adverse evidence, softening unfavorable details, emphasizing favorable details, or replacing precise qualifications with vague language. Existing benchmarks largely miss this subtler and arguably more dangerous failure mode. We introduce JANUS, a benchmark for measuring goal-conditioned pragmatic distortion in fact-grounded LLM outputs. Each scenario in our benchmark provides a fixed pool of favorable and adverse facts and compares a neutral condition against a goal-directed condition, such as increasing adoption, enrollment, approval, or support, despite potential harm to directly affected individuals or groups. Because all outputs are constrained to use the same fact pool, JANUS isolates misleading net impressions from hallucination and fabrication. JANUS contains 160 scenarios across 8 domains, with each scenario paired with neutral and goal-conditioned prompts and annotated material facts. Extensive experiments across 12 LLMs reveal consistent goal-conditioned distortions, demonstrating that current models remain sensitive to incentive and framing objectives and lack robust safeguards against selectively misleading communication. We publicly release our corpus and code for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10796v1">Dep-LLM: Training-Free Depression Diagnosis via Evidence-Guided Structured Multi-factor with Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Automatic Depression Detection (ADD) from clinical interviews is a pivotal task in computational mental health, yet it remains challenging due to two critical obstacles: 1) difficulty in modeling complex but sparsely distributed depression clues within lengthy, multi-topic clinical interviews, leading to superficial and unreliable reasoning; 2) scarcity of labeled data due to clinical privacy, together with high cost of training and fine-tuning, limiting the deployment of supervised ADD systems. To jointly address these challenges, we propose Dep-LLM, a training-free framework that mirrors the step-by-step reasoning of clinical psychiatrists and operates entirely on frozen off-the-shelf foundation LLMs. Dep-LLM comprises three stages. First, a Chain-of-Thought (CoT) Depression Multi-factor Analysis module structurally decomposes the long dialogue into five clinically aligned themes and produces evidence-grounded rationales, effectively handling long-context dependencies. Second, we introduce Confidence Analysis and Modulation module that quantifies the epistemic reliability from token-level entropy of each rationale and applies an intra-label and inter-theme modulation that amplifies trustworthy signals while suppressing uncertain ones without extra training. Third, a Collaborative Multi-factor Prediction module dynamically integrates multi-factor signals weighted by confidence into the final diagnosis. Extensive experiments on the DAIC-WOZ and E-DAIC datasets demonstrate the effectiveness and generalizability of Dep-LLM: it surpasses zero-shot baseline on nearly all 21 foundation LLMs across 9 metrics such as accuracy, macro F1 and weighted-average F1, and further outperforms state-of-the-art supervised domain-specific LLMs as well as the latest closed-source commercial LLMs, while requiring no extra training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10749v1">Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are rapidly moving from conversational interfaces to software components that plan, invoke tools, maintain memory, and act on external environments. This transition changes the nature of security risk. In agentic settings, failures are no longer limited to unsafe text generation. Untrusted content may redirect control flow, misuse tool privileges, corrupt persistent state, leak sensitive information, or trigger harmful external actions. At the same time, research on LLM agent security is expanding quickly but remains fragmented across attack families, defense layers, application domains, and evaluation settings. This paper synthesizes 247 papers through a lifecycle-based, systems-oriented framework that models agent security around the interaction of information flow, delegated authority, and persistent state. We organize the literature around four questions: how LLM agent security should be modeled, which threat surfaces and attack families dominate, what defenses have been proposed and with what tradeoffs, and how security claims are evaluated. We find that prompt injection and tool-mediated control-flow hijacking still dominate the field, while persistent state corruption and multi-agent propagation are becoming central emerging concerns. We further find that current defenses provide useful building blocks but remain weakly compositional, and that existing benchmarks still underrepresent long-horizon, stateful, and deployment-sensitive risks. We argue that secure LLM agents require explicit trust boundaries, principled privilege control, provenance-aware state management, and evaluation practices aligned with realistic operational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10738v1">Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Recent multimodal large language models mainly process audio as monaural signals, thereby discarding the spatial cues contained in spatial audio for sound localization, spatial relation reasoning, and spatial scene understanding. We propose Spatial-Omni, a lightweight method that implements SO-Encoder to inject First-Order Ambisonics (FOA) spatial audio into existing Omni LLMs as an independent modality, without modifying their original audio encoders. SO-Encoder provides spatial tokens with limited additional context cost and improves spatial audio understanding through efficient staged training. To support training and evaluation, we construct SO-Dataset, SO-QA, and SO-Bench from open-source data, real recordings, and simulations, containing 400K FOA spatial audio clips and 2.1M spatial question answering pairs. SO-Bench covers 16 spatial audio understanding subtasks, including basic detection and location estimation, spatial relation understanding, and complex spatial reasoning. Experiments show that Spatial-Omni outperforms existing open-source Large Audio-Language Models (LALMs) and Omni LLM models on spatial audio understanding tasks while retaining a reasonable level of general audio understanding. Code and data are available at https://github.com/dieKarotte/Spatial-Omni.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10706v1">Unifying Data, Memory, and Compute Efficiency in LLM training: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accpeted for publication in IEEE Transactions on Artificial Intelligence (TAI)
    </div>
    <details class="paper-abstract">
      Resource constraints increasingly determine what can be trained, fine-tuned, and deployed in large language models (LLMs), yet efficiency is often studied through isolated techniques rather than as an interacting system of limits. This survey adopts a constraint-centric perspective and organizes recent progress around three coupled bottlenecks: data efficiency (what to train on), memory efficiency (how to fit training), and compute budget awareness (when and where to spend FLOPs). On the data axis, we review selection and pruning methods that maximize learning per token, ranging from scalable proxy signals based on learning dynamics to gradient- and influence-based scoring, as well as difficulty-aware and curriculum-style strategies. We highlight emerging evidence that different notions of good data dominate in different regimes, implying that optimal subsets depend on the task objective and resource budget rather than being universal. On the systems side, we show that GPU memory, not raw compute, is often the dominant bottleneck in fine-tuning, and that effective scaling requires jointly reducing weight storage, optimizer states, and activation memory rather than optimizing any single component in isolation. Beyond memory, we frame training and inference as compute-governed processes in which optimization, data selection, and decoding must explicitly account for finite FLOP budgets. We review evidence for compute-optimal allocation and stopping rules, where computation should be halted or reallocated once marginal performance gains fall below a budget-dependent threshold. Together, these results unify compute-aware data selection, scaling laws, and adaptive inference under a common principle of resource-conditioned decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10694v1">REAL: A Reasoning-Enhanced Graph Framework for Long-Term Memory Management of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to interact with users over long time horizons. However, due to their finite context window, LLMs cannot retain all past interactions, making long-term memory management essential for storing, updating, and retrieving historical information beyond the context limit. Although recent memory systems attempt to address this issue by storing historical information externally, existing approaches suffer from three key limitations: flat text-based memory organizations fail to capture explicit relations among memories, structured memory systems often destructively overwrite evolving facts, and current retrieval mechanisms remain query-agnostic and passive when evidence is incomplete. REAL constructs long-term conversational memory as a temporal and confidence-aware directed property graph, where each atomic fact is represented with entities, relations, valid-time intervals, confidence scores, and exploration intent labels. During memory construction, REAL adopts a non-destructive temporal update strategy that preserves parallel fact versions and their validity intervals, enabling faithful tracking of fact evolution. During retrieval, REAL anchors query-relevant root entities, decouples their exploration intents, and performs semantic evaluator-guided hybrid beam search to extract compact memory subgraphs. It further incorporates counterfactual inference to repair unreliable retrieval states and recover missing memory evidence through implicit logical relations. Comprehensive experiments demonstrate that REAL substantially improves long-term memory performance over flat-text, graph-based, and existing memory baselines, achieving an average improvement of 22.72\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10677v1">Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Long-term LLM agents need persistent memory that can track changing facts and provide relevant evidence across sessions. Existing memory systems often store observations as isolated records, summaries, or indexed fragments, which makes evidence aggregation, fact revision, and memory maintenance difficult. We propose Infini Memory, a maintainable text-based persistent memory architecture that treats agent memory as topic-structured documents. Each topic document serves as a semantic unit for collecting related evidence, preserving metadata, and revising facts over time. New observations are first staged in a buffer and periodically consolidated into coherent textual contexts. At inference time, an agentic retrieval procedure lets the LLM read memory through iterative tool calls rather than a single retrieval step. On MemoryAgentBench, Infini Memory achieves 64.7% overall score. Ablations show that topic-structured maintenance and iterative evidence inspection improve complementary aspects of long-term memory use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10646v1">How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 25 pages, 7 figures, 11 tables. Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Token-level credit assignment remains a key obstacle for reinforcement learning (RL) in large language models (LLMs), where RL recipes typically treat all tokens equally, failing to distinguish decisive reasoning steps from routine formatting or fluent filler. Recent attempts leverage model-internal signals to assign finer-grained credit, but these are often point-wise heuristics that ignore the global structure of information propagation. We propose FlowTracer, an RL framework that traces answer-targeted reasoning flow on an attention-induced directed acyclic graph in which nodes correspond to tokens and edge capacities come from aggregated attention weights and derives token credit from this global structure. The edge capacities are reweighted to retain only the influence that can reach the answer region, while enforcing local flow conservation so intermediate tokens neither lose nor gain effective mass due to path length or irrelevant branches. On this graph, FlowTracer extracts an information-flow backbone connecting the question to the answer and scores tokens by flow throughput, revealing high-impact hubs and aggregation checkpoints that mediate long-range dependencies. These derived importances are used to shape token-level rewards, enabling learning signals to focus precisely on the tokens that route information toward (or away from) correct answers and delivering consistent performance gains across a range of reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17188v2">LLM-Aided Joint Secrecy Precoding and Trajectory for RSMA-Based Heterogeneous UAV Networks</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      This paper investigates secure communications in rate-splitting multiple access (RSMA) enabled heterogeneous UAV networks, where multiple UAVs collaboratively serve ground terminals in the presence of eavesdroppers. By jointly considering secrecy rate maximization and propulsion energy consumption minimization, we formulate a multi-objective optimization problem involving UAV trajectory design, service association, power allocation, and secrecy precoding under mobility, collision-avoidance, service-capacity, and communication constraints. The formulated problem is highly non-convex due to the coupling among UAV trajectories, RSMA transmission variables, and secrecy constraints.To address the resulting non-convex and highly coupled optimization problem, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (D.C.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates LLM-generated expert heuristic policy, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10612v1">GaussTrace: Provenance Analysis of 3D Gaussian Splatting Models with Evidence-based LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted by ICML2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful technique for creating high-fidelity 3D assets. However, the widespread sharing and iterative modification of 3DGS models across digital platforms create pressing challenges for intellectual property protection and forensic traceability. To address this, we propose GaussTrace, a novel framework for constructing directed provenance graphs for 3DGS models. GaussTrace formulates provenance analysis as an evidence-based reasoning problem. It builds upon attribute-wise statistical profiling of 3DGS parameters to capture intrinsic properties. Moreover, we introduce hypothesis-driven editing simulations of common operations to provide auxiliary evidence for plausible transformation pathways. These statistical and simulated cues jointly enable a Large Language Model (LLM) to perform structured Chain-of-Thought (CoT) reasoning, yielding directional provenance inferences and explainable edge reasons. Experimental results demonstrate that GaussTrace effectively constructs evolutionary relationships among diverse 3DGS models, delivering accurate, interpretable, and robust provenance graphs without requiring model training or access to editing histories. Project page: https://haolianghan.github.io/GaussTrace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10607v1">Causal Ensemble Agent: Hierarchical Causal Discovery with LLM-guided Expert Reweighting</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Causal discovery aims to uncover causal structures from observational data, which is crucial for real-world decision-making. However, different causal discovery algorithms can produce divergent results that conflict with each other, complicating the identification of accurate causal graphs. Traditional approaches rely on numerical values and statistical assumptions, often ignoring rich domain-specific information, such as feature descriptions, which could also help structure learning. While recent works explore using Large Language Models (LLMs) to infer causal relations via direct queries, such methods can be unreliable due to a lack of alignment with the actual data. To address these limitations, we propose Causal Ensemble Agent (CEA), a novel framework that aggregates structural insights from statistical discovery experts across different graph levels via linear opinion pooling, and uses an LLM as a meta-referee to dynamically reweight experts when the aggregated confidence is close to the decision boundary, thereby composing an improved and more complete causal graph. Extensive experiments on both synthetic and real-world datasets demonstrate that CEA achieves the strongest overall performance across a wide range of causal discovery methods, highlighting the effectiveness of using LLMs for meta-analysis in causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10532v1">ActiveMem: Distributed Active Memory for Long-Horizon LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Memory is essential for enabling large language model (LLM) agents to handle long-horizon reasoning tasks. Existing memory mechanisms are largely centralized, typically organizing retrieved information and interaction history within a single model context. This design imposes a fundamental trade-off: scaling reasoning trajectories risks context overload, whereas aggressive content pruning may result in irreversible information loss. Seeking a better trade-off, we draw inspiration from human cognitive systems, especially the functional complementarity between the prefrontal cortex (executive control) and the hippocampus (memory management), suggesting that such a trade-off need not be inherent, but may instead stem from centralized memory organization. To this end, we propose ActiveMem, a heterogeneous framework that decouples agent memory from the core reasoning process. Specifically, a high-level Planner utilizes distilled semantic gists to execute reasoning, while a lightweight, distributed memory system operates in parallel to actively accumulate and consolidate these gists throughout the task. Experiments on BrowseComp-Plus and GAIA show that ActiveMem achieves state-of-the-art accuracy with significantly reduced overhead, demonstrating the effectiveness of distributed active memory for long-horizon reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10531v1">LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Quantization-aware training (QAT) is essential for extremely low-bit large language models (LLMs). Current QAT methods are mainly based on scalar quantization (SQ), which enables efficient optimization but suffers from severe performance degradation at 2-bit precision. On the other hand, vector quantization (VQ) provides substantially higher representational capacity, but its discrete codebook lookup prevents end-to-end training. We propose LC-QAT, a 2-bit weight-only VQ-QAT framework that represents quantized weights via a learned affine mapping over discrete vectors, which yields a high-quality PTQ initialization and enables fully differentiable end-to-end optimization without explicit codebook lookup in the training forward pass. This strong post-training initialization makes LC-QAT highly data-efficient. Experiments across diverse LLMs demonstrate that LC-QAT consistently outperforms state-of-the-art QAT methods while using only 0.1%--10% of the training data. Our results establish LC-QAT as a practical and scalable solution for extreme low-bit model deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11458v3">Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 11 pages, 4 figures; code not released yet
    </div>
    <details class="paper-abstract">
      On-policy self-distillation has become a strong recipe for LLM reasoning, where a privileged teacher supervises the student's own rollouts while conditioning on the reference solution. A design choice shared by nearly all such methods, however, has gone unquestioned: the teacher always sees the full reference reasoning. We argue that this default itself is part of the problem and identify a teacher-side exposure mismatch: when the teacher conditions on reasoning far beyond the student's current competence, the resulting token targets become too strong to absorb. A controlled fixed-exposure sweep makes this concrete on two fronts: 1) full exposure is not reliably the best choice, and 2) student-teacher mismatch grows monotonically as the teacher sees more privileged reasoning. This motivates treating teacher exposure not as a fixed hyperparameter but as a learnable training-time control variable. We therefore propose Adaptive Teacher Exposure for Self-Distillation (ATESD). ATESD models the reveal ratio with a lightweight Beta-policy controller conditioned on compact training-state statistics, and uses one sampled exposure for a short hold window of student updates. To make this exposure controller learnable, we optimize it with a discounted learning-progress reward that scores each held decision by its effect on the student's future improvement rather than its immediate loss change, addressing the delayed credit assignment induced by on-policy distillation. Experiments on AIME 24, AIME 25, and HMMT 25 across Qwen3-{1.7B, 4B, 8B} show that ATESD consistently outperforms competitive self-distillation and RL baselines, improving over OPSD by +0.95, +2.05, and +2.33 Average@12 points respectively, and establishing adaptive teacher exposure as an effective new axis for reasoning self-distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08779v2">Reformulate LLM Reinforcement Learning for Efficient Training under Black-box Discrepancy</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has emerged as a pivotal post-training paradigm, yet it frequently suffers from unpredictable sub-optimum performance or even training collapses. Recent findings attribute these failures to a hidden train-inference discrepancy (or mismatch), stemming from the disparate underlying engines and architecture. We find that the training policy can actively self-correct such a discrepancy when provided with an appropriate learning signal. Then, we further empirically identify a discrepancy tolerance region: within this region, aggressively narrowing the discrepancy can suppress policy exploration and reduce learning efficiency, whereas outside this region, reducing excessive discrepancy improves optimization consistency and raises the achievable local performance ceiling. According to such findings, we formulate this problem as a Discrepancy-Constrained Markov Decision Process (DCMDP), where reward maximization is coupled with a constraint that aligns training-Inference behavior, achieving stable dual-objective optimization. To adaptively balance performance improvement and discrepancy control, we introduce a Lagrangian relaxation mechanism that dynamically adjusts the relative weight of the two objectives according to the current degree of discrepancy violation. This enables stable dual-objective optimization: the policy is allowed to explore freely within the tolerance region, while being guided back when the discrepancy exceeds the safe boundary. Empirically, DCMDP significantly improves the performance of 8B dense model (Qwen-3-8b) and 30B Mixture-of-Expert model (Qwen-3-30bA3b), and enables a heterogeneous training paradigm, where LLMs can be optimized in high-fidelity training setup while being explicitly aligned for low-cost, resource-constrained inference deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10507v1">HIPIF: Hierarchical Planning and Information Folding for Long-Horizon LLM Agent Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents across a wide range of tasks, their performance often degrades in multi-turn long-horizon agentic tasks. Existing methods have made progress through fine-grained credit assignment to alleviate long-horizon sparse rewards and hierarchical reinforcement learning to decompose tasks and reduce long-term dependency. However, these methods still do not directly address long-context interference, in which continuously growing histories weaken the agent's ability to track the global task state and impair subsequent reasoning and decision-making. Inspired by the way humans handle complex tasks through subgoal decomposition and completed progress summarization, we propose Hierarchical Planning and Information Folding (HIPIF) for long-horizon LLM agent learning. HIPIF trains the agent end-to-end to organize long-horizon execution around explicit subgoals while folding completed subgoal histories to reduce long-context interference. Furthermore, to stabilize subgoal-based planning and execution, HIPIF combines hierarchical reflection and subgoal-oriented process rewards to guide subgoal generation, transition, and execution, without relying on costly auxiliary models or task-specific expert trajectories. Extensive experiments on three publicly available agentic benchmarks demonstrate the validity of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28054v2">Who Wrote the Book? Detecting and Attributing LLM Ghostwriters</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 WIP
    </div>
    <details class="paper-abstract">
      In this paper, we introduce GhostWriteBench, a dataset for LLM authorship attribution. It comprises long-form texts (50K+ words per book) generated by frontier LLMs, and is designed to test generalisation across multiple out-of-distribution (OOD) dimensions, including domain and unseen LLM author. We also propose TRACE -- a novel fingerprinting method that is interpretable and lightweight -- that works for both open- and closed-source models. TRACE creates the fingerprint by capturing token-level transition patterns (e.g., word rank) estimated by another lightweight language model. Experiments on GhostWriteBench demonstrate that TRACE achieves state-of-the-art performance, remains robust in OOD settings, and works well in limited training data scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28066v2">PromptEmbedder: Efficient and Transferable Text Embedding via Dual-LLM Soft Prompting</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable efficacy in text embedding, yet current adaptation methods like LoRA face significant bottlenecks in computational efficiency and cross-architecture transferability. Whenever a new backbone emerges, existing approaches require costly retraining from scratch. To address this, we propose PromptEmbedder, a novel dual-LLM framework that decouples embedding knowledge from specific backbone weights. PromptEmbedder utilizes a Prompting LLM to generate instruction-aware soft prompts for a frozen Embedding LLM via a differentiable generation process with continuous relaxation, ensuring full gradient flow during contrastive training. By localizing task-specific knowledge within the Prompting LLM, adapting to new architectures requires only retraining a lightweight linear alignment matrix. Evaluations on the MTEB benchmark show that PromptEmbedder achieves comparable performance with LoRA finetuning while reducing GPU memory by 40% and accelerating training by 3.7x. Our approach establishes a scalable, architecture-agnostic paradigm for efficient LLM-based representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.02817v2">Conditional Vendi Score: Prompt-Aware Diversity Evaluation for Generative AI Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Generative models guided by text prompts are widely evaluated for fidelity and prompt alignment, yet their ability to produce outputs remains underexplored. Existing diversity metrics such as Vendi and RKE, which are based on the von Neumann and Rényi entropies of kernel matrices, were developed for unconditional models and cannot distinguish prompt-induced from model-induced variability. We address this gap by introducing \textit{Conditional-Vendi} and \textit{Conditional-RKE}, diversity measures derived from the conditional entropy of positive semidefinite matrices. These scores isolate model-induced diversity in prompt-guided generation, with Conditional-RKE enjoying an $O(1/\sqrt{n})$ convergence rate. For Conditional-Vendi, we introduce a truncated-spectrum approximation that yields scalable and consistent estimates. Experiments on text-to-image, image-captioning, and LLM tasks show that the conditional scores recover ground-truth diversity orderings and can also guide diffusion models toward more diverse samples. The codebase is available at https://github.com/mjalali/conditional-vendi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10461v1">ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Text-attributed Graphs (TAGs) incorporate textual node attributes with graph structures to describe rich relational semantics. Recent efforts to integrate Graph Neural Networks (GNNs) and Large Language Models (LLMs) have shown promise for learning on TAGs, yet achieving well-aligned representations remains challenging. Prior studies largely rely on heuristics that perform coarse-grained matching. They lack sufficient constraints and ignore distributional alignment, leading to representation drift and limited generalization. Building on Energy-based Models (EBMs), we propose an Energy-based Representation Alignment (ERAlign) framework that projects GNN-encoded graph structure and LLM-derived text embeddings in a shared latent space to achieve distribution consistency. Concretely, layer-wise alignment is quantified by a distance metric and optimized via an EBM objective. By decreasing energy values, our framework yields well-aligned representations for downstream tasks. During training, we introduce Energy Discrepancy (ED) to avoid high sampling costs associated with intractable normalization. ED also carries theoretical guarantees of higher training efficiency and reduced energy landscape distortion. Empirical evaluations on eight TAG datasets demonstrate that ERAlign obtains state-of-the-art performance across varying levels of supervision and cross-task transfer scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10454v1">Entropy-Aware Domain-Routed Mixture-of-Experts Speech-LLM Framework: A Case Study of Multi-Domain Child-Adult ASR</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      While Speech Large Language Models (Speech-LLMs) have achieved strong performance on adult Automatic Speech Recognition (ASR), their effectiveness on child speech remains under-explored, and single models often struggle to handle diverse adult and child age groups simultaneously. This paper proposes a Mixture-of-Experts (MoE) Speech-LLM for unified ASR across adult and child speech spanning diverse environments and age groups. The framework employs a Classifier-based Domain Router (C-DR) with a coarse-to-fine strategy and integrates both a Mixture-of-Projectors (MoP) and a Mixture-of-LoRAs (MoL) to model domain-specific variations. To address routing uncertainty near domain boundaries, an Entropy-Aware Routing (EAR) mechanism is introduced to dynamically incorporate a shared expert. Experiments on public child corpora demonstrate consistent improvements over baselines while preserving adult ASR performance. To our knowledge, this is the first work leveraging Speech-LLMs for unified, multi-domain ASR encompassing both children and adults.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16189v4">Mitigating hallucinations in healthcare LLMs with granular fact-checking and domain-specific adaptation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Published in Expert Systems with Applications
    </div>
    <details class="paper-abstract">
      In healthcare, it is essential for any Large Language Model (LLM)-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRA) on the MIMIC-III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10445v1">SpenseGPT: Practical One-shot Pruning Enabling Sparse and Dense GEMMs for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Semi-structured 2:4 sparsity is widely supported by modern accelerators, providing up to a 2x theoretical speedup. However, its strict 50% sparsity constraint often causes non-negligible accuracy degradation under post-training pruning. Meanwhile, existing relaxed sparsity formats either require specialized compiler support or introduce runtime overheads that limit end-to-end speedup. We propose Spense, a practical hybrid sparse-dense format that splits each weight matrix into a 2:4 sparse region and a dense region. This design relaxes the effective sparsity constraint while remaining compatible with existing high-performance sparse and dense GEMM libraries, avoiding both custom compiler support and input activation expansion. Building on this format, we introduce SpenseGPT, a one-shot post-training pruning method that produces sparse and dense regions. Notably, we show that selecting the right dense regions is important, and we devise two different strategies to choose them. Experiments on Qwen3-32B and Seed-OSS-36B demonstrate that our method achieves up to 1.2x end-to-end decoding speedup on B200 GPUs with FP8 precision, while preserving accuracy. To the best of our knowledge, this is the first one-shot pruning demonstration of real-world end-to-end LLM decoding speedup from semi-structured sparse tensor cores on recent GPUs such as B200s, while maintaining model quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10439v1">Enhancing Multilingual LLM-based ASR with Mixture of Experts and Dynamic Downsampling</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) has opened up a new frontier for automatic speech recognition (ASR), making their effective integration a critical and challenging research direction. To this end, this work proposes a projector-based LLM-ASR framework targeting the key challenges of multilingual generalization and modality alignment. Our approach incorporates a Mixture of Experts (MoE) architecture to improve cross-lingual adaptability, and a Continuous Integrate-and-Fire (CIF) mechanism for dynamic downsampling and modality alignment. Experimental results show that the combination of these components yields substantial performance improvements, surpassing strong baseline models. The proposed method represents a step toward building more accurate, robust, and generalizable LLM-based ASR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10434v1">Profiling cognitive offloading in LLM-mediated synthesis writing: Volume vs. content</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to the Proceedings of the European Conference for Tecnology-Enhanced Learning' 2026
    </div>
    <details class="paper-abstract">
      This study compares two approaches to profiling how learners offload cognitive activity to LLMs during a synthesis writing task. Drawing on Salomon's distributed cognition and the Kintsch and van Dijk model of text comprehension, the study operationalises offloading to an LLM in two ways: as a volume of LLM use and as content of what is offloaded, both along with prior knowledge. Data from 97 university students interacting with a general-purpose LLM via a custom interface were analysed using k-means clustering. To capture the content of offloading, their prompts were interpreted as to who performs the activity (active or passive) and at what level of comprehension (local or global). Volume-based profiling (k=4) differentiated learners primarily by prior knowledge, with volume negatively associated with essay authorship. Content-based profiling (k=5) revealed qualitatively distinct patterns of offloading, from vocabulary clarification to active direction of structuring and generation to passive delegation of comprehension at both levels. These patterns reflect different fragmentation of the cognitive process, with differences in learning strategies, behavioural markers, and essay authorship. Combining volume and content of offloading could improve future analyses on how LLM use redistributes cognitive activity and its effects on learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20242v5">BadRobot: Jailbreaking Embodied LLM Agents in the Physical World</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICLR 2025. Please cite the conference version. Project page: https://Embodied-LLMs-Safety.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Our code is available at https://github.com/Rookie143/BadRobot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03093v3">ATLAS: Verifier-Guided Adaptive Latent Activation Steering for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent work on activation and latent steering has demonstrated that modifying internal representations can effectively guide large language models (LLMs) toward improved reasoning and efficiency without updating model parameters. However, most existing approaches rely on fixed steering policies and static intervention strengths, which limit their robustness across problem instances and often result in over- or under-steering. We propose Adaptive Test-time Latent Steering (ATLAS), a lightweight framework that dynamically controls steering decisions at inference time using a trained, lightweight verifier over the latent states. Given intermediate hidden states, the verifier predicts the quality of ongoing reasoning and adaptively selects which steering action to apply, enabling per-example and per-step adjustment with minimal overhead. ATLAS provides a unified framework for combining learned latent verification with test-time activation steering, enabling adaptive reasoning control without additional LLM decoding or inference-time process reward model calls. Experiments on multiple mathematical and coding reasoning benchmarks show that ATLAS consistently outperforms both vanilla decoding and fixed steering baselines, achieving higher accuracy while substantially reducing test-time token usage. These results demonstrate that verifier-guided latent adaptation provides an effective and scalable mechanism for controlling reasoning efficiency without sacrificing solution quality. All source code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19274v2">HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 ACL 2026 Main Camera-Ready
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at https://github.com/untae0122/HarDBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14463v2">An Industrial-Scale Insurance LLM Achieving Verifiable Domain Mastery and Hallucination Control without Competence Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 21 pages, 12 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) to high-stakes vertical domains like insurance presents a significant challenge: scenarios demand strict adherence to complex regulations and business logic with zero tolerance for hallucinations. Existing approaches often suffer from a Competency Trade-off - sacrificing general intelligence for domain expertise - or rely heavily on RAG without intrinsic reasoning. To bridge this gap, we present INS-S1, an insurance-specific LLM family trained via a novel end-to-end alignment paradigm. Our approach features two methodological innovations: (1) A Verifiable Data Synthesis System that constructs hierarchical datasets for actuarial reasoning and compliance; and (2) A Progressive SFT-RL Curriculum Framework that integrates dynamic data annealing with a synergistic mix of Verified Reasoning (RLVR) and AI Feedback (RLAIF). By optimizing data ratios and reward signals, this framework enforces domain constraints while preventing catastrophic forgetting. Additionally, we release INSEva, the most comprehensive insurance benchmark to date (39k+ samples). Extensive experiments show that INS-S1 achieves SOTA performance on domain tasks, significantly outperforming DeepSeek-R1 and Gemini-2.5-Pro. Crucially, it maintains top-tier general capabilities and achieves a record-low 0.6% hallucination rate (HHEM). Our results demonstrate that rigorous domain specialization can be achieved without compromising general intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04397v3">Context-as-AI-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 8 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM agents increasingly write and maintain developer documentation, but usefulness and accuracy often rely on dependency chains that are not obvious to follow. Even with more files in context, the agent must still decide which cross-file dependencies to trace. We present Context-as-AI-Service (CAIS), a retrieval layer that LLM agents query to find evidence across the codebase as they review or generate documentation. CAIS indexes source code, API references, and upstream documentation, then enables agents to query the index through tool calls that combine keyword and semantic search. We evaluate CAIS in two case studies using Claude Sonnet 4.6 on a production SDK: improving API reference comments in a core source file and validating an LLM-generated tutorial. In both studies, the baseline already had ordinary repository tools such as file reads, keyword search, and symbol navigation. CAIS adds a retrieval layer on top, so the comparison isolates added retrieval rather than basic repository access. In the API-reference review, the CAIS-augmented agent produced the same 5 missing-documentation fixes as the baseline and surfaced 4 findings the baseline missed: 2 cross-file factual errors and 2 underspecified API comments. In the tutorial validation, it surfaced 1 executable bug, 1 API-usage improvement, and 2 missing prerequisites that the baseline pipeline did not catch. These findings required tracing non-obvious dependency chains across utility files, framework internals, usage examples, tests, and component-creation logic. Over five runs per condition, adding CAIS reduced wall-clock time by 22% to 34% across the two tasks and lowered input-token usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12966v2">ProbeLLM: Automating Principled Diagnosis of LLM Failures</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Understanding how and why large language models (LLMs) fail is becoming a central challenge as models rapidly evolve and static evaluations fall behind. While automated probing has been enabled by dynamic test generation, existing approaches often discover isolated failure cases, lack principled control over exploration, and provide limited insight into the underlying structure of model weaknesses. We propose ProbeLLM, a benchmark-agnostic automated probing framework that elevates weakness discovery from individual failures to structured failure modes. ProbeLLM formulates probing as a hierarchical Monte Carlo Tree Search, explicitly allocating limited probing budgets between global exploration of new failure regions and local refinement of recurring error patterns. By restricting probing to verifiable test cases and leveraging tool-augmented generation and verification, ProbeLLM grounds failure discovery in reliable evidence. Discovered failures are further consolidated into interpretable failure modes via failure-aware embeddings and boundary-aware induction. Across diverse benchmarks and LLMs, ProbeLLM reveals substantially broader, cleaner, and more fine-grained failure landscapes than static benchmarks and prior automated methods, supporting a shift from case-centric evaluation toward principled weakness discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10389v1">Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Recent advances in LLM-driven code evolution have enabled automated discovery by iteratively generating and improving programs. However, applying these methods to adversarial multi-agent games introduces a fundamental challenge: the evaluation landscape shifts as strategies improve, causing fixed evaluators to become unreliable and evolution to stagnate. We propose three mechanisms to address this challenge: evaluator co-evolution, which incorporates discovered champions into the opponent pool; hierarchical deep evaluation, which replaces noisy few-game scores with statistically reliable assessments; and weakness pressure, which dynamically up-weights the most difficult opponents to break through plateaus. We implement these mechanisms within FAMOU, a framework built upon the same foundation-model code-evolution paradigm as OpenEvolve and ShinkaEvolve. On the MCTF 2026 3v3 maritime capture-the-flag task, FAMOU consistently outperforms both baselines under two backbone LLMs, achieving the highest combined score (0.526) and the best generalization to unseen opponents (61.7% win rate), while ablations confirm that each mechanism contributes to performance. Notably, the LLM mutation process generates tactical structures entirely absent from the seed strategies -- including lookahead search and adaptive interception -- demonstrating that code-level evolution can produce nontrivial algorithmic innovations in adversarial settings. The FAMOU-evolved strategy further achieved 1st place in the hardware round-robin and 3rd in simulation at the AAMAS 2026 MCTF Competition, validating its real-world transferability. The optimized implementation and corresponding evaluation codes developed through our evolutionary process are available at: https://github.com/1xiangliu1/FAMOU-CoEvo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16384v2">SemOpt: LLM-Driven Code Optimization via Rule-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Automated code optimization improves program performance through refactoring, and recent studies leverage LLMs for this purpose. Existing approaches mine optimization commits from open-source codebases to build large-scale knowledge bases, then employ retrieval techniques such as BM25 to obtain relevant examples for hotspot code, guiding LLMs in optimization. However, semantically equivalent optimizations often appear in syntactically dissimilar code, so current retrieval methods fail to identify pertinent examples, leading to suboptimal results. To address these limitations, we propose SemOpt, a framework that leverages static program analysis to identify code segments, retrieve optimization strategies, and generate optimized results. SemOpt has three LLM-powered components: (1) a strategy library builder that extracts and clusters strategies from code modifications, (2) a rule generator that produces Semgrep static analysis rules to capture each strategy's applicability, and (3) an optimizer that generates optimized code using the strategy library. On a benchmark of 151 C/C++ and 150 Python optimization tasks, SemOpt shows consistent improvements across different LLMs, increasing successful optimizations by 1.38 to 28 times on C/C++ and 4.60 to 6.33 times on Python versus the baseline. On large-scale projects, SemOpt improves performance metrics by 5.04% to 218.07% on C/C++ and 61.77% to 479.90% on Python, showing cross-language generalization and practical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10359v1">ReflectiChain: Epistemic Grounding in LLM-Driven World Models for Supply Chain Resilience</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      AI agents in supply chains face a fundamental epistemic gap: large language models (LLMs) interpret policies but lack physical grounding, while reinforcement learning (RL) optimizes flows but is semantically blind to unstructured constraints. We introduce REFLECTICHAIN, bridging this gap through a Generative Supply Chain World Model (SC-WM) - encoding heterogeneous supply networks into a 6-dim graph-latent space with physical conservation - and Double-Loop Learning that separates epistemic uncertainty (KL-trust-region-bounded policy adaptation) from aleatoric uncertainty (stochastic latent rollouts). On Semi-Sim, a 10-node semiconductor benchmark with SIR risk propagation, 6 perturbation types, and 10 policy constraint templates, REFLECTICHAIN improves Rationale Consistency Score by 33.0% (p < 0.0001, d = 2.78), maintains 82.3% operability under adversarial shocks, and exhibits anti-fragile behavior (+40.2% gain under moderate pressure). We identify three operational epistemic mechanisms - uncertainty separation, knowledge-boundary detection, and empirical Bayesian policy updating - and discuss five limitation categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10357v1">Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Cross-domain recommendation is a core problem in content-to-e-commerce platforms. Its objective is to leverage user interactions with content to infer potential purchasing intent on the e-commerce side, thereby enhancing conversion rates and commercial value. However, in real industrial scenarios, cross-domain recommendation faces multiple challenges: significant semantic gaps exist between different domains, and user cross-domain behavior sequences are often massive in scale and rich in noise. Although large language models (LLMs) possess powerful semantic understanding and reasoning capabilities, their millisecond-level inference latency makes direct application in online recommendation systems difficult. To address these issues, this paper introduces AIR (Atomic Intent Reasoning), an LLM-driven cross-domain recommendation framework designed for industrial-grade deployment. By migrating LLM inference to the offline phase and dynamically constructing user intent representations through efficient retrieval and composition during online operations, it achieves approximately 400* inference acceleration while maintaining semantic consistency. Experimental results across multiple public datasets demonstrate that our method achieves state-of-the-art performance in cross-domain recommendation tasks. Furthermore, large-scale online A/B testing conducted in Kuaishou E-commerce's real-world business scenarios shows that our approach delivers stable and significant improvements across multiple core business metrics, including a +3.446% increase in GMV, fully validating its effectiveness and practical value in industrial-scale recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09788v3">TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, including preliminary experiments with real human behavior as control. Results highlight possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at https://github.com/microsoft/tinytroupe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10322v1">Game-Theoretic Multi-Agent Control for Robust Contextual Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-turn interactions maintain evolving context rather than generating isolated responses, making them vulnerable to prompt-injection and context-poisoning attacks in which locally plausible adversarial fragments gradually distort reasoning trajectories. Existing defenses mainly filter individual outputs and often ignore context evolution across turns, leaving long-horizon reasoning exposed. Although the Model Context Protocol (MCP) standardizes context exchange and tool invocation, it functions as a passive routing layer and does not enforce contextual stability. To address these limitations, we introduce the Game-Theoretic Secure Model Context Protocol (GT-MCP), a controller-driven multi-agent method that treats context management as a closed-loop dynamical process. GT-MCP coordinates three heterogeneous LLM agents and selects outputs through a trust function that jointly evaluates causal consistency against a validated context graph, semantic agreement among agents, and distributional drift over time. When instability is detected, a rollback-based self-healing mechanism restores the validated context and prevents unsupported fragments from propagating. Empirical evaluation over 500 interaction turns under an adaptive adversarial threat model shows that contextual drift remains bounded in 99.6% of turns, with recovery required in only 0.4%. Per-turn utility remains tightly concentrated, with median = -0.19, P05 = -0.72, and P95 = 0.30; severe degradation below -1 occurs in only 0.4% of cases, and no injection attempt succeeds at the controller level. Selected outputs maintain stable win rates above 98%, and computational overhead remains predictable, with latency per token = 1.63e-3 s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06470v1">PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      We propose a preconditioning (PC) layer, a weight parameterization via polynomial preconditioner that ensures stable weight conditioning throughout LLM training. The PC module reshapes the singular-value spectrum of weight matrices via low-degree polynomial preconditioning. After training, the preconditioned weights can be merged back into the original architecture, incurring no inference overhead. We demonstrate the advantage of the proposed PC layer over standard transformers in Llama-1B pre-training, for both the AdamW and Muon optimizers. Theoretically, we justify this spectrum-control principle by proving that uniformly bounding each layer's singular values ensures geometric convergence of gradient descent to global minima, for certain deep linear networks. Our code is available at https://github.com/Empath-aln/PC-layer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06460v1">Will the Agent Recuse Itself? Measuring LLM-Agent Compliance with In-Band Access-Deny Signals</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 8 pages, 1 figure. Code, specification, and experiment harness: https://github.com/mthamil107/Recuse
    </div>
    <details class="paper-abstract">
      As autonomous LLM agents increasingly hold real credentials and operate infrastructure without a human in the loop, operators have no standard way to tell an agent that a resource is off-limits. Access controls either let the agent in (it has valid credentials) or hard-fail it (indistinguishable from any other client). We propose a third mode: a lightweight, published in-band deny signal -- the Recuse Signal -- that a server emits over a protocol's existing channels (an SSH banner, a PostgreSQL NOTICE) asking a connecting automated agent to voluntarily withdraw. This is a cooperative governance control, the robots.txt analogue for live access; it is explicitly not a security boundary. Its value is entirely empirical and, to our knowledge, unmeasured: do compliant LLM agents actually honor such a signal? We define the signal as an open mini-standard, implement two zero- or low-footprint adapters (an SSH banner/PAM hook and a PostgreSQL wire-protocol proxy), deploy them on a live production host, and run a controlled experiment in which fresh agents are given a benign operations task and observed for recusal. In a pilot (SSH; OpenAI GPT-4o and GPT-4o-mini; and Claude Code as a deployed agent), the signal cleanly induces recusal -- 100% recusal when present versus 100% task completion in a no-signal control -- and, revealingly, behaves as a cooperative rather than absolute signal: an explicit operator-authorization framing flips the most capable model to proceed, while other agents continue to defer to the on-host policy. We release the standard, adapters, and experiment harness for reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06443v1">Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to simulate social media users and infer how individuals may respond to online discussions. However, it remains unclear whether these simulations reflect precise user-specific beliefs or whether they are highly sensitive to semantically independent changes in conversational contexts. In this work, we study counterfactual context revision as a framework for auditing LLM-based stance simulation. Given an original online conversation, we first infer a target user's stance toward a specific topic. We then apply controlled revision strategies to the conversational context and simulate the user's stance again under the revised context. We compare text-only revision strategies with a multimodal one that incorporates meme-based context and evaluate two main effectiveness metrics, i.e., average directional stance shift and stance transition rate. The results reveal effective and robust stance transitions in both text-only and multimodal strategies across different polarization-preference mechanisms. Our study contributes an evaluation framework for understanding the context sensitivity of LLM-based stance simulation. More broadly, it highlights both the promise and risk of using LLMs to simulate online opinion dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06420v1">A Komi-Yazva--Russian Parallel Corpus and Evaluation Protocol for Zero- and Few-Shot LLM Translation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 18 pages, 6 tables, 3 figures
    </div>
    <details class="paper-abstract">
      We present the first Komi-Yazva--Russian parallel corpus together with an explicit evaluation protocol for studying LLM translation in an endangered, extremely low-resource setting. The dataset contains 457 aligned sentence pairs from 74 narrative texts and is accompanied by documented provenance, sentence-level alignment, and story identifiers that enable leakage-aware evaluation. We use this setup to compare modern large language models on Komi-Yazva-to-Russian translation under severe parallel-data scarcity in zero-shot and retrieval-based few-shot regimes. The protocol includes story-level cross-validation, deterministic retrieval for few-shot prompting, strict validation of generated outputs, complementary reference-based and judge-based metrics, and story-level uncertainty estimates. Across models, LLMs produce non-trivial translations, but performance varies strongly by model family and prompting regime. Retrieval-based few-shot prompting consistently improves over zero-shot prompting, while gains beyond a small retrieved context remain limited. The results show that evaluative conclusions in this setting depend materially on metric choice and failure handling, so the paper frames the corpus as both a dataset contribution and a reproducible evaluation testbed for endangered-language machine translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06399v1">CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) built on large language models have shown growing promise, with their effectiveness resting on agents' ability to coordinate through text-based channels much as human teams do. Yet recent study suggests that MAS often falter not because agents lack individual task-solving ability, but because they lack collaborative competence: the capacity to establish common ground, maintain shared task understanding, balance individual and collective incentives, and repair misalignment as interaction unfolds. Decades of research in Computer-Supported Cooperative Work have characterized these requirements for human teams coordinating under constrained communication, yet existing MAS evaluations focus mainly on task outcomes or single-agent proficiency in reasoning, planning, and tool use. To enable a systematic analysis of agents' collaborative competence in MAS, we introduce CollabSim, a configurable simulation framework that combines a theory-grounded definition of collaborative capabilities, controlled manipulation of interaction conditions, and action-level probing of agents' internal states. Experiments across four LLMs show that CollabSim can capture condition effects, separate model performance patterns, and reveal task-dependent effects of agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02776v3">Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what the main driver of these disparities is, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions, and readability. We find that conversation topics are most predictive of LLM-generated advice within a conversational context, which, to some extent, function as proxies for sociodemographic groups and often affect advice in unpredictable ways. This is cause for concern and highlights the need for future research to better understand and, if needed, mitigate the effect of conversational context on LLM outputs in high-stakes scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06350v1">EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Reliable rubric grading requires more than accurate score prediction. Each judgement must be grounded in the mark scheme and evidence from the student answer. Existing credit-assignment and intervention methods, primarily designed for self-contained reasoning tasks such as mathematics reasoning, struggle in this setting because they do not identify where grading reasoning goes wrong or how the model's belief about the final mark changes during reasoning. We propose Evidence-Diagnosed Intervention Training (EDIT), a two-phase framework for training more rubric-faithful LLM graders. First, EDIT-SFT locates problematic reasoning steps using internal model signals: posterior belief over the final mark and input-grounding scores. It then revises only these local steps with help from a rubric checklist. Second, EDIT-RL calibrates the grader with belief-guided reward shaping, penalising large harmful belief drifts while still allowing helpful exploration. Experiments on two real-world, multi-subject grading benchmarks demonstrate that EDIT consistently outperforms strong supervised fine-tuning and reinforcement learning baselines on both in-domain and out-of-domain splits, with ablation studies confirming that internal-state diagnostics drive these gains.
    </details>
</div>
